import json
import shutil
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ruff: noqa: E402
ray = pytest.importorskip("ray")
import torch

from sentimentizer.config import DecoderConfig, EncoderConfig, RNNConfig, TrainerConfig
from sentimentizer.extractor import extract_data_ray
from sentimentizer.loader import CorpusDataset, load_train_val_ray_datasets
from sentimentizer.models.decoder import Decoder
from sentimentizer.models.encoder import Encoder
from sentimentizer.models.rnn import RNN, get_trained_model
from sentimentizer.tokenizer import Tokenizer, convert_rating, get_trained_tokenizer, regex_tokenize
from sentimentizer.trainer import new_ray_trainer, new_trainer


@pytest.fixture
def tokenized_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tokens": [
                ["the", "chicken", "never", "showed", "up"],
                ["the", "food", "was", "terrific"],
            ],
            "stars": [2, 5],
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "the chicken never showed up",
                "the food was terrific",
            ],
            "stars": [2, 5],
        }
    )


@pytest.fixture
def processed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data": [
                (1, 2, 3, 4, 5, 6),
                (5, 6, 7, 7, 8, 19),
            ],
            "target": [2, 5],
        }
    )


def test_convert_rating():
    assert convert_rating(5) == 2  # 4-5 stars -> positive (2)
    assert convert_rating(4) == 2  # 4-5 stars -> positive (2)
    assert convert_rating(1) == 0  # 1-2 stars -> negative (0)
    assert convert_rating(2) == 0  # 1-2 stars -> negative (0)
    assert convert_rating(3) == 1  # 3 stars -> neutral (1)


def test_tokenize(raw_df):
    output = regex_tokenize(raw_df.text[0])

    for item in output:
        assert isinstance(item, str)

    assert len(output) > 3


class TestExtractData:
    fname = "artificial-reviews.jsonl"
    stop = 2

    def test_success(self, rel_path, relative_root):
        ray.init(ignore_reinit_error=True)
        ds = extract_data_ray(compressed_file_name=self.fname, file_path=rel_path, stop=self.stop)
        assert isinstance(ds, ray.data.Dataset)

        path = f"{relative_root}/tests/test_data/file.parquet"
        shutil.rmtree(path, ignore_errors=True)
        ds.write_parquet(path)

        df = pd.read_parquet(path)
        assert df.shape == (2, 3)  # text, tokens, stars
        assert df["tokens"].dtype == "object" or isinstance(df["tokens"].dtype, pd.ArrowDtype)
        assert pd.api.types.is_integer_dtype(df["stars"].dtype)

    def test_failure_empty_input(self):
        # todo
        return


class TestDataTokenizer:
    def test_success(self, tokenized_df):
        parser = Tokenizer.from_data(tokenized_df)
        result = parser.transform_dataframe(tokenized_df)
        assert result.shape == (2, 4)

    def test_failure(self):
        # todo
        return


class TestCorpusDataset:
    def test_success(self, processed_df):
        dataset = CorpusDataset(processed_df)
        item = dataset[1]
        assert len(item) == 2
        assert len(dataset) == 2

    def test_failure(self):
        # todo
        return


class TestGetTrainedModel:
    """tests model construction and weight loading"""

    def test_model_construction(self):
        """tests that the RNN model can be constructed with correct architecture"""
        emb_weights = torch.zeros(100, 100)  # small vocab, 100d embeddings
        model = RNN(emb_weights=emb_weights)
        assert isinstance(model, RNN)
        assert model.lstm.bidirectional
        assert model.lstm.batch_first
        assert model.lstm.num_layers == 2

    def test_forward_pass(self):
        """tests that the forward pass produces correct output shape"""
        emb_weights = torch.randn(100, 100)
        model = RNN(emb_weights=emb_weights)
        tokens = torch.randint(0, 100, (2, 10))
        output = model(tokens)
        assert output.shape == (2, 3)  # (batch_size, num_classes)

    def test_missing_weights_file(self):
        """tests that loading from a missing weights file raises FileNotFoundError
        when both local file and Hugging Face Hub download fail"""
        from unittest.mock import patch

        with (
            patch("sentimentizer.models.rnn.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights", return_value=None),
            pytest.raises(FileNotFoundError, match="Weights file not found"),
        ):
            get_trained_model("cpu")

    def test_failure(self):
        # todo
        return


class TestGetTrainedTokenizer:
    """tests if model loads"""

    def test_success(self):
        from unittest.mock import patch

        from gensim import corpora

        # Build a minimal fake dictionary so we don't need the trained file on disk
        fake_dict = corpora.Dictionary()
        fake_dict.token2id = {"good": 0, "bad": 1, "the": 2}
        fake_dict.num_docs = 10
        fake_dict.num_pos = 100
        fake_dict.num_nnz = 30

        with patch("sentimentizer.tokenizer.corpora.Dictionary.load", return_value=fake_dict):
            tokenizer = get_trained_tokenizer()
            assert isinstance(tokenizer, Tokenizer)

    def test_failure(self):
        # todo
        return


class TestTokenize:
    """tests regex"""

    def test_success(self):
        result = regex_tokenize("chicken wasn't good")

        assert len(result) == 3
        assert result[0] == "chicken"
        assert result[1] == "wasn't"

    def test_success_one(self):
        result = regex_tokenize("1st place food")
        assert len(result) == 3
        assert result[0] == "1st"


# ──────────────────────────────────────────────
# Ray Data and Ray Train tests
# ──────────────────────────────────────────────


class TestRayDatasetLoader:
    """tests Ray Dataset loading and splitting"""

    def test_load_ray_datasets(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            assert isinstance(train_ds, ray.data.Dataset)
            assert isinstance(val_ds, ray.data.Dataset)
            total = train_ds.count() + val_ds.count()
            assert total > 0
        except Exception:
            pytest.skip("test parquet data not available")

    def test_load_ray_datasets_default_split(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path)
            # Default test_size is 0.2, so train should be ~80% and val ~20%
            total = train_ds.count() + val_ds.count()
            assert total > 0
        except Exception:
            pytest.skip("test parquet data not available")


class TestNewRayTrainer:
    """tests Ray Train TorchTrainer creation"""

    def test_trainer_creation(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            trainer = new_ray_trainer(
                train_ds=train_ds,
                val_ds=val_ds,
                cfg=cfg,
                model_type="rnn",
            )
            from ray.train.torch import TorchTrainer

            assert isinstance(trainer, TorchTrainer)
        except Exception:
            pytest.skip("test parquet data not available")

    def test_trainer_creation_encoder(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            trainer = new_ray_trainer(
                train_ds=train_ds,
                val_ds=val_ds,
                cfg=cfg,
                model_type="encoder",
            )
            from ray.train.torch import TorchTrainer

            assert isinstance(trainer, TorchTrainer)
        except Exception:
            pytest.skip("test parquet data not available")

    def test_trainer_creation_decoder(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            trainer = new_ray_trainer(
                train_ds=train_ds,
                val_ds=val_ds,
                cfg=cfg,
                model_type="decoder",
            )
            from ray.train.torch import TorchTrainer

            assert isinstance(trainer, TorchTrainer)
        except Exception:
            pytest.skip("test parquet data not available")

    def test_trainer_invalid_model_type(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            with pytest.raises(ValueError, match="no matching model"):
                new_ray_trainer(
                    train_ds=train_ds,
                    val_ds=val_ds,
                    cfg=cfg,
                    model_type="invalid_model",
                )
        except Exception:
            pytest.skip("test parquet data not available")

    def test_train_func_config_serializable(self):
        """test that _train_func config dict is serializable (required for Ray)"""
        config = {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.005,
            "betas": [0.7, 0.99],
            "weight_decay": 1e-4,
            "warmup_ratio": 0.06,
            "scheduler_eta_min": 1e-6,
            "model_type": "rnn",
            "dict_path": "/tmp/test.dict",
            "embeddings_model_name": "glove-wiki-gigaword-100",
            "embeddings_emb_length": 100,
            "input_len": 200,
        }
        serialized = json.dumps(config)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["model_type"] == "rnn"
        assert deserialized["epochs"] == 1

    def test_train_func_config_has_all_keys(self):
        """test that _train_func config contains all required keys"""
        required_keys = {
            "epochs",
            "batch_size",
            "lr",
            "betas",
            "weight_decay",
            "warmup_ratio",
            "scheduler_eta_min",
            "model_type",
            "dict_path",
            "embeddings_model_name",
            "embeddings_emb_length",
            "input_len",
        }
        config = {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.005,
            "betas": [0.7, 0.99],
            "weight_decay": 1e-4,
            "warmup_ratio": 0.06,
            "scheduler_eta_min": 1e-6,
            "model_type": "rnn",
            "dict_path": "/tmp/test.dict",
            "embeddings_model_name": "glove-wiki-gigaword-100",
            "embeddings_emb_length": 100,
            "input_len": 200,
        }
        assert required_keys.issubset(set(config.keys()))


class TestTrainerConfig:
    """tests TrainerConfig with ray_workers for distributed training"""

    def test_default_ray_workers(self):
        cfg = TrainerConfig()
        assert cfg.ray_workers == 1

    def test_custom_ray_workers(self):
        cfg = TrainerConfig(ray_workers=4)
        assert cfg.ray_workers == 4

    def test_cpu_device_config(self):
        cfg = TrainerConfig(device="cpu", ray_workers=1)
        assert cfg.device == "cpu"
        assert cfg.ray_workers == 1


class TestModelConfigs:
    """tests that model configs drive architecture dimensions"""

    def test_rnn_config_defaults(self):
        cfg = RNNConfig()
        assert cfg.hidden_size == 256
        assert cfg.num_layers == 2
        assert cfg.dropout == 0.2

    def test_rnn_custom_config(self):
        """tests that custom RNNConfig changes model architecture"""
        cfg = RNNConfig(hidden_size=128, num_layers=3, dropout=0.3)
        emb_weights = torch.zeros(100, 100)
        model = RNN(
            emb_weights=emb_weights,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        assert model.lstm.hidden_size == 128
        assert model.lstm.num_layers == 3

    def test_encoder_config_defaults(self):
        cfg = EncoderConfig()
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.n_layers == 4
        assert cfg.ff_multiplier == 4

    def test_encoder_custom_config(self):
        """tests that custom EncoderConfig changes model architecture"""
        cfg = EncoderConfig(d_model=128, n_heads=2, n_layers=2, ff_multiplier=2)
        emb_weights = torch.zeros(100, 100)
        model = Encoder(
            emb_weights=emb_weights,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ff_multiplier=cfg.ff_multiplier,
        )
        assert model.d_model == 128
        assert model.encoder.num_layers == 2

    def test_decoder_config_defaults(self):
        cfg = DecoderConfig()
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.n_encoder_layers == 2
        assert cfg.n_decoder_layers == 2
        assert cfg.dropout == 0.3
        assert cfg.ff_multiplier == 4

    def test_decoder_custom_config(self):
        """tests that custom DecoderConfig changes model architecture"""

        cfg = DecoderConfig(d_model=128, n_heads=2, n_encoder_layers=1, n_decoder_layers=2)
        emb_weights = torch.zeros(100, 100)
        model = Decoder(
            emb_weights=emb_weights,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_encoder_layers=cfg.n_encoder_layers,
            n_decoder_layers=cfg.n_decoder_layers,
        )
        assert model.d_model == 128
        assert model.encoder.num_layers == 1
        assert model.decoder.num_layers == 2


class TestSingleTrainer:
    """tests the existing single-node Trainer still works"""

    def test_new_trainer_creates_trainer(self):
        emb_weights = torch.randn(100, 100)
        model = RNN(emb_weights=emb_weights)
        cfg = TrainerConfig(device="cpu")
        trainer = new_trainer(model=model, cfg=cfg, model_type="rnn")
        # Default loss_type is "focal", so the loss function is FocalCrossEntropyLoss
        # (which wraps CrossEntropyLoss internally).
        from sentimentizer.losses import FocalCrossEntropyLoss

        assert isinstance(trainer.loss_function, FocalCrossEntropyLoss)


# ──────────────────────────────────────────────
# Metrics and Monitoring tests
# ──────────────────────────────────────────────


class TestMetricsGauges:
    """Tests specifically for verifying metric updates to Prometheus gauges."""

    def test_train_func_updates_prometheus_gauges(self) -> None:
        """Verifies _train_func updates Prometheus Gauges during distributed training."""
        from sentimentizer.trainer import _train_func

        # 1. Setup minimal config
        config = {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "model_type": "rnn",
            "dict_path": "fake.dict",
            "embeddings_model_name": "fake-glove",
            "embeddings_emb_length": 100,
            "input_len": 200,
            "class_weights": [1.0, 1.0, 1.0],
            "loss_type": "cross_entropy",
            "label_smoothing": 0.0,
        }

        # 2. Mock Ray Data shards
        mock_shard = MagicMock()
        # Simulate one batch of data for both train and val
        mock_shard.iter_torch_batches.return_value = [
            {
                "data": torch.zeros((2, 200), dtype=torch.long),
                "target": torch.ones((2,), dtype=torch.long),
            }
        ]

        # 3. Create mock gauge objects (matching _get_ray_gauges dict keys)
        mock_gauges = {
            "train_loss": MagicMock(),
            "val_loss": MagicMock(),
            "val_accuracy": MagicMock(),
            "val_balanced_accuracy": MagicMock(),
            "val_negative_precision": MagicMock(),
            "val_negative_recall": MagicMock(),
            "val_negative_f1": MagicMock(),
            "val_neutral_precision": MagicMock(),
            "val_neutral_recall": MagicMock(),
            "val_neutral_f1": MagicMock(),
            "val_positive_precision": MagicMock(),
            "val_positive_recall": MagicMock(),
            "val_positive_f1": MagicMock(),
            "val_cohen_kappa": MagicMock(),
            "val_mcc": MagicMock(),
            "val_macro_f1": MagicMock(),
            "val_weighted_f1": MagicMock(),
            "val_neutral_to_positive_rate": MagicMock(),
            "val_neutral_to_negative_rate": MagicMock(),
            "val_pred_neutral_frac": MagicMock(),
            "val_neutral_auc_roc": MagicMock(),
            "val_neutral_avg_precision": MagicMock(),
            "epoch": MagicMock(),
            "lr": MagicMock(),
            "train_loss_ema": MagicMock(),
            "train_loss_avg": MagicMock(),
            "train_batch": MagicMock(),
            "train_grad_norm": MagicMock(),
            "train_throughput": MagicMock(),
        }

        # Setup mock model — parameters() must return a fresh iterator each call
        mock_model = MagicMock()
        p = torch.nn.Parameter(torch.randn(1, 1))
        mock_model.parameters.side_effect = lambda: iter([p])
        mock_model.return_value = torch.randn(2, 3)
        mock_model.state_dict.return_value = {}
        mock_model.prepare_batch.return_value = (torch.randn(2, 2), torch.tensor([1, 1]))
        # MagicMock invents attributes; without this, unwrapping DDP finds an unrelated mock.
        mock_model.module = mock_model

        # 4. Mock dependencies to isolate _train_func
        with (
            patch("ray.train.get_dataset_shard", return_value=mock_shard),
            patch("ray.train.get_context") as mock_get_context,
            patch("ray.train.report"),
            patch("sentimentizer.trainer.train_step", return_value=(0.5, 0.0)),
            patch("sentimentizer.trainer._get_ray_gauges", return_value=mock_gauges),
            patch("sentimentizer.trainer.create_model_from_registry", return_value=mock_model),
            patch("sentimentizer.trainer.prepare_model", side_effect=lambda m: m),
            patch("sentimentizer.exporter.TRAINING_LR", create=True),
        ):
            # Simulate worker rank 0 so metrics are recorded
            mock_get_context.return_value.get_world_rank.return_value = 0

            _train_func(config)

            # 5. Assertions
            assert mock_gauges["val_loss"].set.called, "val_loss gauge was not updated"
            assert mock_gauges["val_accuracy"].set.called, "val_accuracy gauge was not updated"
            assert mock_gauges["lr"].set.called, "lr gauge was not updated"

    def test_trainer_evaluate_updates_gauges(self) -> None:
        """Verifies Trainer.evaluate updates Prometheus Gauges during single-node training."""
        from sentimentizer.trainer import Trainer, TrainerConfig

        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.dataset = [1, 2]
        mock_loader.__iter__.return_value = iter(
            [(torch.zeros(2, 10), torch.ones(2, dtype=torch.long))]
        )

        cfg = TrainerConfig(device="cpu")
        trainer = Trainer(
            loss_function=torch.nn.CrossEntropyLoss(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            cfg=cfg,
            model_type="rnn",
        )

        # Create mock gauge objects (matching _get_ray_gauges dict keys)
        mock_gauges = {
            "train_loss": MagicMock(),
            "val_loss": MagicMock(),
            "val_accuracy": MagicMock(),
            "val_balanced_accuracy": MagicMock(),
            "val_negative_precision": MagicMock(),
            "val_negative_recall": MagicMock(),
            "val_negative_f1": MagicMock(),
            "val_neutral_precision": MagicMock(),
            "val_neutral_recall": MagicMock(),
            "val_neutral_f1": MagicMock(),
            "val_positive_precision": MagicMock(),
            "val_positive_recall": MagicMock(),
            "val_positive_f1": MagicMock(),
            "val_cohen_kappa": MagicMock(),
            "val_mcc": MagicMock(),
            "val_macro_f1": MagicMock(),
            "val_weighted_f1": MagicMock(),
            "val_neutral_to_positive_rate": MagicMock(),
            "val_neutral_to_negative_rate": MagicMock(),
            "val_pred_neutral_frac": MagicMock(),
            "val_neutral_auc_roc": MagicMock(),
            "val_neutral_avg_precision": MagicMock(),
            "epoch": MagicMock(),
            "lr": MagicMock(),
            "train_loss_ema": MagicMock(),
            "train_loss_avg": MagicMock(),
            "train_batch": MagicMock(),
        }

        with (
            patch("sentimentizer.trainer._get_ray_gauges", return_value=mock_gauges),
            # Patch the exporter import inside evaluate() to avoid real Prometheus gauges
            patch("sentimentizer.exporter.TRAINING_EPOCH", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_LOSS", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_ACCURACY", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_COHEN_KAPPA", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_MCC", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_MACRO_F1", create=True),
        ):
            mock_model.return_value = torch.randn(2, 3)
            trainer.evaluate(mock_model, mock_loader, epoch=0)
            assert mock_gauges["val_loss"].set.called, "val_loss gauge was not updated"
            assert mock_gauges["val_accuracy"].set.called, "val_accuracy gauge was not updated"
