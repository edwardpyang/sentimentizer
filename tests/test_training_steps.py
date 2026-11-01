"""Compare production training paths with explicit effective-batch updates."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray.cloudpickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sentimentizer.trainer as training
from sentimentizer.config import NUM_CLASSES, TrainerConfig


class StepModel(nn.Module):
    STEP_SCHEDULER_PER_BATCH = True

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, NUM_CLASSES, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.arange(NUM_CLASSES * 2).reshape(NUM_CLASSES, 2) / 20)

    def forward(self, input_ids):
        return self.projection(input_ids)

    def prepare_batch(self, batch, device):
        return {"input_ids": batch["input_ids"].to(device)}, batch["target"].to(device)

    def save_to_checkpoint_dir(self, _directory):
        return {"model_state_dict": self.state_dict()}


class WrappedModel(nn.Module):
    """Expose the same scheduler-flag boundary as DDP without starting a cluster."""

    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, **inputs):
        return self.module(**inputs)


class Shard:
    def __init__(self, batches):
        self.batches = batches

    def iter_torch_batches(self, **_kwargs):
        return iter(self.batches)


def _batches(count):
    generator = torch.Generator().manual_seed(42)
    return [
        {
            "input_ids": torch.randn(2, 2, generator=generator),
            "target": torch.tensor([index % NUM_CLASSES, (index + 1) % NUM_CLASSES]),
        }
        for index in range(count)
    ]


def _components(model, total_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0)
    scheduler = training._LinearWarmupCosineScheduler(
        optimizer, warmup_steps=1, total_steps=total_steps, eta_min=1e-6
    )
    return optimizer, scheduler


def _effective_batch_reference(batches, epochs, accumulation):
    """Use actual larger batches, independent of production accumulation logic."""
    model = StepModel()
    optimizer, scheduler = _components(model, epochs * math.ceil(len(batches) / accumulation))
    for _ in range(epochs):
        for offset in range(0, len(batches), accumulation):
            window = batches[offset : offset + accumulation]
            optimizer.zero_grad()
            inputs = torch.cat([batch["input_ids"] for batch in window])
            targets = torch.cat([batch["target"] for batch in window])
            nn.functional.cross_entropy(model(inputs), targets).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()
    return model


@pytest.mark.parametrize("path", ["single", "callbacks", "worker"])
@pytest.mark.parametrize("batch_count", [1, 3, 4])
def test_accumulation_matches_effective_batches_across_epochs(path, batch_count, monkeypatch):
    epochs, accumulation = 2, 2
    batches = _batches(batch_count)
    total_steps = epochs * math.ceil(batch_count / accumulation)
    expected = _effective_batch_reference(batches, epochs, accumulation)
    model = StepModel()
    optimizer, scheduler = _components(model, total_steps)
    monkeypatch.setattr(training, "write_batch_snapshot", lambda **_kwargs: None)
    monkeypatch.setattr(training, "publish_epoch_metrics", lambda **_kwargs: None)

    if path == "single":
        trainer = training.Trainer(
            loss_function=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=TrainerConfig(device="cpu", gradient_accumulation_steps=accumulation),
            model_type="rnn",
        )
        dataset = TensorDataset(
            torch.cat([batch["input_ids"] for batch in batches]),
            torch.cat([batch["target"] for batch in batches]),
        )
        for epoch in range(epochs):
            trainer._train_epoch(model, DataLoader(dataset, batch_size=2), epoch + 1)
        saved_scheduler = scheduler.state_dict()
    elif path == "callbacks":
        training._run_training_loop(
            model=model,
            train_iter=batches,
            val_iter=batches,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=nn.CrossEntropyLoss(),
            callbacks=[],
            model_type="rnn",
            device="cpu",
            grad_accum_steps=accumulation,
        )
        saved_scheduler = scheduler.state_dict()
    else:
        monkeypatch.setattr(training, "create_model_from_registry", lambda **_kwargs: model)
        monkeypatch.setattr(training, "prepare_model", WrappedModel)
        monkeypatch.setattr(training, "_get_ray_gauges", lambda *_args: None)
        monkeypatch.setattr(training.train, "get_dataset_shard", lambda _name: Shard(batches))
        monkeypatch.setattr(
            training.train, "get_context", lambda: SimpleNamespace(get_world_rank=lambda: 0)
        )
        checkpoints = []

        def report(_metrics, checkpoint):
            with (
                checkpoint.as_directory() as directory,
                (Path(directory) / "data.pkl").open("rb") as stream,
            ):
                checkpoints.append(ray.cloudpickle.load(stream))

        monkeypatch.setattr(training.train, "report", report)
        training._train_func(
            {
                "epochs": epochs,
                "batch_size": 2,
                "lr": 0.03,
                "betas": [0.9, 0.999],
                "weight_decay": 0,
                "model_type": "rnn",
                "dict_path": "unused",
                "embeddings_model_name": "unused",
                "embeddings_emb_length": 2,
                "gradient_accumulation_steps": accumulation,
                "label_smoothing": 0,
                "use_amp": False,
                "use_8bit_optimizer": False,
                "train_dataset_count": batch_count * 2,
                "num_workers": 1,
            }
        )
        assert len(checkpoints) == epochs
        saved_scheduler = checkpoints[-1]["scheduler_state_dict"]

    assert saved_scheduler["last_epoch"] == total_steps
    assert saved_scheduler["total_steps"] == total_steps
    torch.testing.assert_close(model.projection.weight, expected.projection.weight)


@pytest.mark.parametrize("batch_count,accumulation", [(1, 4), (3, 2), (4, 2), (5, 3)])
def test_scheduler_budget_includes_the_final_incomplete_window(batch_count, accumulation):
    model = StepModel()
    optimizer, scheduler = _components(model, 1)
    trainer = training.Trainer(
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=TrainerConfig(device="cpu", gradient_accumulation_steps=accumulation),
        model_type="rnn",
    )
    trainer.rebuild_scheduler(batch_count, epochs=3)
    assert trainer.scheduler.total_steps == 3 * math.ceil(batch_count / accumulation)


def test_last_batch_train_step_normalizes_an_incomplete_window():
    batch = _batches(1)[0]
    expected = StepModel()
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.03)
    nn.functional.cross_entropy(expected(batch["input_ids"]), batch["target"]).backward()
    nn.utils.clip_grad_norm_(expected.parameters(), max_norm=1)
    reference_optimizer.step()
    model = StepModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    training.train_step(
        model,
        {"input_ids": batch["input_ids"]},
        batch["target"],
        optimizer,
        nn.CrossEntropyLoss(),
        grad_accum_steps=4,
        is_last_batch=True,
        device_type="cpu",
    )
    torch.testing.assert_close(model.projection.weight, expected.projection.weight)


def test_empty_epoch_and_repeated_flush_do_not_step_optimizer_or_scheduler(monkeypatch):
    from unittest.mock import Mock

    model = StepModel()
    optimizer, scheduler = _components(model, 2)
    update = Mock(wraps=optimizer.step)
    monkeypatch.setattr(optimizer, "step", update)
    stepper = training._TrainingStepper(
        model,
        optimizer,
        scheduler,
        nn.CrossEntropyLoss(),
        grad_accum_steps=2,
        use_amp=False,
        device_type="cpu",
    )
    stepper.flush()
    update.assert_not_called()
    assert scheduler.last_epoch == 0
    batch = _batches(1)[0]
    stepper.step({"input_ids": batch["input_ids"]}, batch["target"])
    stepper.flush()
    stepper.flush()
    update.assert_called_once()
    assert scheduler.last_epoch == 1


@pytest.mark.parametrize("accumulation", [0, -1])
def test_nonpositive_accumulation_is_rejected_before_training(accumulation):
    model = StepModel()
    optimizer, scheduler = _components(model, 1)
    with pytest.raises(ValueError, match="positive"):
        training._TrainingStepper(
            model,
            optimizer,
            scheduler,
            nn.CrossEntropyLoss(),
            grad_accum_steps=accumulation,
            use_amp=False,
            device_type="cpu",
        )
