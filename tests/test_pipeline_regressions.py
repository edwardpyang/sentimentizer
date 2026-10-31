"""Regression tests for training, tuning, resume, and saved-weight failures.

These run real code paths on tiny models rather than mocks: earlier tests replaced
``prepare_model`` with a pass-through and inspected callback source text, which let
multi-worker training and every Ray Tune run crash while the suite stayed green.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import ray
import ray.cloudpickle
import torch
from ray.train.torch import TorchTrainer
from torch import nn

import sentimentizer.metrics_publisher as metrics_publisher
import sentimentizer.models.base as base
import sentimentizer.models.decoder as decoder_module
import sentimentizer.models.encoder as encoder_module
import sentimentizer.models.rnn as rnn_module
import sentimentizer.trainer as trainer_module
from sentimentizer.agent.diagnose_model import _publish_tuned_weights
from sentimentizer.config import DecoderConfig, EncoderConfig, ModernBERTConfig, TrainerConfig
from sentimentizer.loader import CorpusDataset, load_train_val_corpus_datasets
from sentimentizer.metrics import ClassificationMetrics
from sentimentizer.models.decoder import Decoder
from sentimentizer.models.encoder import Encoder
from sentimentizer.models.modernbert import ModernBERT
from sentimentizer.models.rnn import RNN
from sentimentizer.trainer import (
    _LinearWarmupCosineScheduler,
    _report_metrics,
    latest_checkpoint,
    load_checkpoint,
    new_ray_trainer,
    new_trainer,
    restore_best_checkpoint,
    save_checkpoint,
)
from sentimentizer.tuner import TunePrometheusCallback, _get_scheduler, _trainable_wrapper
from workflows.lifecycle import State
from workflows.stages.train import _persist_metrics_to_file, validate_train_options

EMBEDDINGS = (50, 8)


def _tiny_rnn(num_layers: int = 2) -> RNN:
    return RNN(emb_weights=torch.randn(*EMBEDDINGS), hidden_size=8, num_layers=num_layers)


def _reviews(rows: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "data": [rng.integers(1, EMBEDDINGS[0] - 1, size=10) for _ in range(rows)],
            "target": [row % 3 for row in range(rows)],
        }
    )


def _config(**overrides) -> TrainerConfig:
    defaults = {
        "device": "cpu",
        "dataloader_workers": 0,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "use_amp": False,
        "use_8bit_optimizer": False,
        "epochs": 4,
        "early_stopping_patience": 0,
    }
    return TrainerConfig(**{**defaults, **overrides})


@pytest.fixture(autouse=True)
def isolated_metrics(tmp_path, monkeypatch):
    """Keep metric writes out of the real metrics directory and skip Ray gauges."""
    monkeypatch.setattr(metrics_publisher, "_METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(trainer_module, "_get_ray_gauges", lambda *_args, **_kwargs: None)


def _record_lr_by_epoch(trainer) -> list[float]:
    """Record the optimizer LR at each evaluation, in epoch order."""
    lrs: list[float] = []
    evaluate = trainer.evaluate

    def recording_evaluate(model, loader, epoch):
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
        return evaluate(model, loader, epoch)

    trainer.evaluate = recording_evaluate
    return lrs


class TestResume:
    def test_resumed_lr_matches_uninterrupted_schedule(self, tmp_path) -> None:
        dataset = CorpusDataset(_reviews())
        cfg = _config(checkpoint_dir=str(tmp_path))

        model = _tiny_rnn()
        trainer = new_trainer(model, cfg, "rnn")
        uninterrupted = _record_lr_by_epoch(trainer)
        trainer.fit(model, dataset, dataset)

        resumed_model = _tiny_rnn()
        resumed = new_trainer(resumed_model, cfg, "rnn")
        checkpoint = load_checkpoint(
            tmp_path / "checkpoint_epoch_2.pth",
            resumed_model,
            resumed.optimizer,
            resumed.scheduler,
        )
        resumed_lrs = _record_lr_by_epoch(resumed)
        resumed.fit(resumed_model, dataset, dataset, start_epoch=checkpoint["epoch"] + 1)

        # fit() re-evaluates the checkpoint's epoch first, so entries are epochs 2, 3, 4.
        assert resumed_lrs == pytest.approx(uninterrupted[1:])

    def test_scheduler_state_restores_real_step_counts(self) -> None:
        def scheduler(total_steps: int) -> _LinearWarmupCosineScheduler:
            optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1e-3)
            return _LinearWarmupCosineScheduler(optimizer, 1, total_steps, eta_min=1e-6)

        real = scheduler(1000)
        for _ in range(10):
            real.optimizer.step()
            real.step()

        placeholder = scheduler(4)
        placeholder.load_state_dict(real.state_dict())
        placeholder.optimizer.step()
        placeholder.step()
        real.optimizer.step()
        real.step()

        assert placeholder.get_last_lr() == pytest.approx(real.get_last_lr())

    def test_load_checkpoint_does_not_need_populated_registry(self, tmp_path) -> None:
        model = _tiny_rnn()
        optimizer = torch.optim.AdamW(model.parameters())
        path = tmp_path / "checkpoint_epoch_1.pth"
        save_checkpoint(model, optimizer, 1, path)

        restored = _tiny_rnn()
        with patch.dict(base._MODEL_REGISTRY, clear=True):
            load_checkpoint(path, restored)

        for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
            assert torch.equal(expected, actual)

    def test_loading_from_directory_populates_registry(self) -> None:
        model = _tiny_rnn()
        with patch.dict(base._MODEL_REGISTRY, clear=True):
            loaded = RNN.load_from_checkpoint_dir(
                Path("unused"), {"model_state_dict": model.state_dict()}, "cpu"
            )
        assert isinstance(loaded, RNN)

    def test_load_checkpoint_keeps_tuned_architecture(self, tmp_path) -> None:
        """A non-default architecture must not be rebuilt from config defaults."""
        model = _tiny_rnn(num_layers=1)
        path = tmp_path / "best_model.pth"
        save_checkpoint(model, torch.optim.AdamW(model.parameters()), 1, path)

        load_checkpoint(path, _tiny_rnn(num_layers=1))

    def test_latest_checkpoint_orders_epochs_numerically(self, tmp_path) -> None:
        for epoch in range(1, 12):
            (tmp_path / f"checkpoint_epoch_{epoch}.pth").touch()
        assert latest_checkpoint(tmp_path).name == "checkpoint_epoch_11.pth"

    def test_split_is_reproducible(self, tmp_path) -> None:
        path = tmp_path / "reviews.parquet"
        frame = _reviews(200)
        frame["data"] = frame["data"].apply(list)
        frame.to_parquet(path)

        first, _ = load_train_val_corpus_datasets(str(path), random_state=42)
        second, _ = load_train_val_corpus_datasets(str(path), random_state=42)
        other, _ = load_train_val_corpus_datasets(str(path), random_state=7)

        def rows(dataset):
            return [row.tolist() for row in dataset._x_data]

        assert rows(first) == rows(second)
        assert rows(first) != rows(other)


class _StubBackbone(nn.Module):
    """Stand-in exposing only what HFTransformerModel uses, without a model download."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.config = types.SimpleNamespace(hidden_size=4)

    def gradient_checkpointing_enable(self) -> None:
        pass

    def save_pretrained(self, path: str) -> None:
        torch.save(self.state_dict(), Path(path) / "weights.pt")

    @classmethod
    def from_pretrained(cls, path: str, **_kwargs) -> "_StubBackbone":
        backbone = cls()
        weights = Path(path) / "weights.pt"
        if weights.exists():
            backbone.load_state_dict(torch.load(weights, weights_only=True))
        return backbone


class TestHFCheckpoints:
    @pytest.fixture
    def model(self):
        with patch(
            "sentimentizer.models.hf_base.AutoModel.from_pretrained", _StubBackbone.from_pretrained
        ):
            yield ModernBERT(
                config=ModernBERTConfig(model_name="unused", gradient_checkpointing=False)
            )

    def test_each_checkpoint_keeps_its_own_backbone(self, model, tmp_path) -> None:
        optimizer = torch.optim.AdamW(model.parameters())
        save_checkpoint(model, optimizer, 1, tmp_path / "best_model.pth")
        best = model.backbone.layer.weight.detach().clone()

        with torch.no_grad():
            model.backbone.layer.weight.add_(1.0)
        save_checkpoint(model, optimizer, 2, tmp_path / "checkpoint_epoch_2.pth")
        latest = model.backbone.layer.weight.detach().clone()

        load_checkpoint(tmp_path / "best_model.pth", model)
        assert torch.equal(model.backbone.layer.weight, best)
        load_checkpoint(tmp_path / "checkpoint_epoch_2.pth", model)
        assert torch.equal(model.backbone.layer.weight, latest)

    def test_checkpoint_with_shared_backbone_still_loads(self, model, tmp_path) -> None:
        """Checkpoints written before per-checkpoint directories keep working."""
        metadata = model.save_to_checkpoint_dir(tmp_path)
        torch.save(metadata, tmp_path / "checkpoint_epoch_1.pth")
        load_checkpoint(tmp_path / "checkpoint_epoch_1.pth", model)


class TestSavedWeights:
    @pytest.mark.parametrize(
        ("module", "build"),
        [
            (rnn_module, lambda: RNN(emb_weights=torch.randn(*EMBEDDINGS))),
            (encoder_module, lambda: Encoder(emb_weights=torch.randn(*EMBEDDINGS))),
            (decoder_module, lambda: Decoder(emb_weights=torch.randn(*EMBEDDINGS))),
        ],
    )
    def test_per_model_loader_reads_weights_saved_by_training(
        self, module, build, tmp_path, monkeypatch
    ) -> None:
        model = build()
        path = tmp_path / "weights.pth"
        torch.save(model.save_to_checkpoint_dir(tmp_path), path)
        monkeypatch.setattr(module, "weights_path_for", lambda _model: str(path))

        loaded = module.get_trained_model("cpu")

        for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
            assert torch.equal(expected, actual)

    def test_tuned_architecture_reaches_every_loader(self, tmp_path, monkeypatch) -> None:
        # Two heads load silently with four-head defaults because shapes match.
        config = EncoderConfig(d_model=16, n_heads=2, n_layers=1)
        model = Encoder(emb_weights=torch.randn(*EMBEDDINGS), d_model=16, n_heads=2, n_layers=1)
        source = tmp_path / "best_model_encoder.pth"
        torch.save(model.state_dict(), source)
        (tmp_path / "best_model_encoder_config.json").write_text(
            json.dumps({"model_type": "encoder", **vars(config)})
        )
        destination = tmp_path / "encoder_weights.pth"
        _publish_tuned_weights(source, destination)
        metadata = torch.load(destination, weights_only=True)

        from_directory = Encoder.load_from_checkpoint_dir(tmp_path, metadata, "cpu")
        monkeypatch.setattr(encoder_module, "weights_path_for", lambda _model: str(destination))
        from_module = encoder_module.get_trained_model("cpu")

        for loaded in (from_directory, from_module):
            assert loaded.encoder.layers[0].self_attn.num_heads == 2
            assert len(loaded.encoder.layers) == 1

    def test_tuned_decoder_config_ignores_unknown_keys(self) -> None:
        state, config = base.unpack_saved_weights(
            {"model_state_dict": {"w": 1}, "model_config": {"n_heads": 2, "model_type": "decoder"}},
            DecoderConfig(),
        )
        assert state == {"w": 1}
        assert config.n_heads == 2

    def test_restore_best_checkpoint_replaces_final_weights(self, tmp_path) -> None:
        model = _tiny_rnn()
        optimizer = torch.optim.AdamW(model.parameters())
        save_checkpoint(model, optimizer, 1, tmp_path / "best_model.pth")
        best = [param.detach().clone() for param in model.parameters()]
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        assert restore_best_checkpoint(model, tmp_path)

        for expected, actual in zip(best, model.parameters(), strict=True):
            assert torch.equal(expected, actual)

    @pytest.mark.parametrize("checkpoint_dir", ["", "missing"])
    def test_restore_best_checkpoint_without_checkpoint(self, checkpoint_dir, tmp_path) -> None:
        directory = str(tmp_path / checkpoint_dir) if checkpoint_dir else ""
        assert not restore_best_checkpoint(_tiny_rnn(), directory)


class TestTuning:
    @pytest.fixture
    def trial(self):
        """Run one tuning trial in-process and return (reports, model trainer)."""
        dataset = CorpusDataset(_reviews())
        created = []

        def capture_trainer(*args, **kwargs):
            trainer = new_trainer(*args, **kwargs)
            created.append(trainer)
            return trainer

        def run(**space) -> tuple[list[dict], list]:
            config = {
                "model_type": "rnn",
                "device": "cpu",
                "dict_path": "unused",
                "embeddings_model_name": "unused",
                "embeddings_emb_length": EMBEDDINGS[1],
                "epochs": 2,
                **space,
            }
            with (
                patch("sentimentizer.models.rnn.new_model", lambda **_kwargs: _tiny_rnn()),
                patch("sentimentizer.trainer.new_trainer", capture_trainer),
                patch("ray.tune.report") as report,
            ):
                _trainable_wrapper(config, train_dataset=dataset, val_dataset=dataset)
            return [call.args[0] for call in report.call_args_list], created

        return run

    def test_trial_reports_metrics_for_dict_batches(self, trial) -> None:
        reports, _ = trial()
        assert [report["epoch"] for report in reports] == [1, 2]
        assert "val_macro_f1" in reports[0]

    def test_trial_uses_sampled_hyperparameters(self, trial) -> None:
        _, [trainer] = trial(lr=5e-4, weight_decay=0.03, batch_size=8)
        decay_group = trainer.optimizer.param_groups[0]
        assert decay_group["initial_lr"] == pytest.approx(5e-4)
        assert decay_group["weight_decay"] == pytest.approx(0.03)
        assert trainer.cfg.batch_size == 8

    def test_trial_lr_follows_full_schedule(self, trial) -> None:
        _, [trainer] = trial(epochs=4, batch_size=1)
        # 64 batches per epoch at the default 8 accumulation steps per optimizer step.
        assert trainer.scheduler.total_steps == 4 * 8

    def test_callback_accepts_rays_keyword_dispatch(self) -> None:
        from ray.tune.callback import CallbackList

        callbacks = CallbackList([TunePrometheusCallback("rnn")])
        trial = MagicMock(trial_id="trial-1")
        callbacks.on_trial_start(iteration=1, trials=[trial], trial=trial)
        callbacks.on_trial_result(iteration=1, trials=[trial], trial=trial, result={"epoch": 1})
        callbacks.on_trial_complete(iteration=1, trials=[trial], trial=trial)

    def test_scheduler_budget_is_epochs_not_trials(self) -> None:
        from sentimentizer.tuner import TunerConfig

        scheduler = _get_scheduler(TunerConfig(num_samples=20), max_epochs=4)
        assert scheduler._max_t == 4

    def test_failed_validation_reports_threshold(self) -> None:
        from workflows.stages import tune as tune_stage

        result = MagicMock(validation_passed=False, retry_count=1)
        del result.validation_threshold  # TuningRunResult has no such field.
        options = {
            "mode": "standalone",
            "agent_config": None,
            "tune_samples": 1,
            "tune_max_iterations": 1,
            "tune_output_dir": "unused",
            "no_validate": False,
            "validation_threshold": 0.75,
            "max_retries": 0,
            "save": False,
            "push_to_hub": False,
            "balance_classes": False,
            "balance_seed": 42,
        }
        with (
            patch.object(tune_stage, "_ensure_ray_initialized"),
            patch("sentimentizer.agent.diagnose_model.TuningRun") as run,
        ):
            run.return_value.execute.return_value = result
            tune_stage.run_tune(State(model="rnn", device="cpu", run_type="new"), **options)


class TestDistributedOptions:
    @pytest.mark.parametrize(
        ("run_type", "resume"),
        [("new", True), ("update", False)],
    )
    def test_rejects_options_workers_cannot_honor(self, run_type, resume) -> None:
        state = State(model="rnn", device="cpu", run_type=run_type)
        with pytest.raises(ValueError, match="distributed"):
            validate_train_options(state, distributed=True, resume=resume, checkpoint_dir="ckpt")

    @pytest.mark.parametrize(
        ("run_type", "resume"),
        [("update", True), ("new", False)],
    )
    def test_single_node_keeps_resume_and_update(self, run_type, resume) -> None:
        state = State(model="rnn", device="cpu", run_type=run_type)
        validate_train_options(state, distributed=False, resume=resume, checkpoint_dir="ckpt")

    def test_resume_without_checkpoint_dir_is_rejected(self) -> None:
        state = State(model="rnn", device="cpu", run_type="new")
        with pytest.raises(ValueError, match="checkpoint-dir"):
            validate_train_options(state, distributed=False, resume=True, checkpoint_dir="")

    def test_final_report_keeps_every_persisted_gauge(self, tmp_path, monkeypatch) -> None:
        from workflows.stages import train as train_stage

        monkeypatch.setattr(train_stage, "_METRICS_DIR", tmp_path)
        metrics = ClassificationMetrics(
            neutral_to_positive_rate=0.25,
            neutral_to_negative_rate=0.5,
            pred_neutral_frac=0.75,
            neutral_auc_roc=0.8,
            neutral_avg_precision=0.9,
        )

        _persist_metrics_to_file(_report_metrics(0.1, 0.2, metrics, 1, 1e-3), "rnn", "run-1")

        saved = json.loads((tmp_path / "rnn_metrics.json").read_text())
        assert saved["run_id"] == "run-1"
        assert saved["neutral_to_positive_rate"] == 0.25
        assert saved["neutral_to_negative_rate"] == 0.5
        assert saved["pred_neutral_frac"] == 0.75
        assert saved["neutral_auc_roc"] == 0.8
        assert saved["neutral_avg_precision"] == 0.9


# Workers cannot import this test module, so send its functions by value.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _worker_loop(config: dict) -> None:
    """Run the production worker function with a tiny model and no GloVe download."""
    metrics_publisher._METRICS_DIR = Path(config["metrics_dir"])
    trainer_module.create_model_from_registry = lambda **_kwargs: _tiny_rnn()
    trainer_module._train_func(config)


@pytest.fixture
def ray_cluster():
    ray.shutdown()
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


class TestRealRayRuns:
    def test_two_worker_training_reports_whole_validation_set(self, ray_cluster, tmp_path) -> None:
        frame = _reviews(48)
        train_ds = ray.data.from_pandas(frame)
        val_ds = ray.data.from_pandas(frame.iloc[:24])
        cfg = _config(
            epochs=2,
            ray_workers=2,
            early_stopping_patience=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        # Keep new_ray_trainer's real configuration, but run the test's worker wrapper.
        with patch.object(trainer_module, "TorchTrainer") as captured:
            new_ray_trainer(train_ds, val_ds, cfg, "rnn")
        kwargs = captured.call_args.kwargs
        kwargs["train_loop_per_worker"] = _worker_loop
        kwargs["train_loop_config"]["metrics_dir"] = str(tmp_path / "metrics")

        result = TorchTrainer(**kwargs).fit()

        assert result.error is None
        assert result.metrics["total"] == 24
        best = result.get_best_checkpoint(metric="val_loss", mode="min")
        assert best is not None
        assert str(tmp_path / "checkpoints") in best.path

        with best.as_directory() as directory, open(Path(directory) / "data.pkl", "rb") as handle:
            saved = ray.cloudpickle.load(handle)
        loaded = RNN.load_from_checkpoint_dir(Path(directory), saved, "cpu")
        assert isinstance(loaded, RNN)

    def test_tune_run_with_prometheus_callback(self, ray_cluster, tmp_path) -> None:
        from ray import tune
        from ray.tune import RunConfig

        def trainable(config: dict) -> None:
            metrics_publisher._METRICS_DIR = Path(config["metrics_dir"])
            rnn_module.new_model = lambda **_kwargs: _tiny_rnn()
            dataset = CorpusDataset(_reviews())
            _trainable_wrapper(config, train_dataset=dataset, val_dataset=dataset)

        tuner = tune.Tuner(
            trainable,
            param_space={
                "model_type": "rnn",
                "device": "cpu",
                "dict_path": "unused",
                "embeddings_model_name": "unused",
                "embeddings_emb_length": EMBEDDINGS[1],
                "epochs": 1,
                "batch_size": 4,
                "lr": 1e-3,
                "metrics_dir": str(tmp_path / "metrics"),
            },
            run_config=RunConfig(
                storage_path=str(tmp_path / "tune"),
                callbacks=[TunePrometheusCallback("rnn")],
            ),
        )

        results = tuner.fit()

        assert results.errors == []
        assert results[0].metrics["epoch"] == 1
