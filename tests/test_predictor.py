"""Tests for SentimentPredictor inference, tokenization and metadata paths."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sentimentizer.predictor import SentimentPredictor


class _GloveModel(torch.nn.Module):
    """Minimal GloVe-style model that always returns fixed probabilities."""

    NEEDS_TOKENIZE_STAGE = True

    def __init__(self, probs: list[list[float]]) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 3)
        self.probs = torch.tensor(probs)
        self.tokenizer = MagicMock()
        self.tokenizer.tokenize_text.side_effect = lambda _text: np.zeros((1, 4), dtype=int)
        self.inputs = None

    def predict(self, inputs):
        self.inputs = inputs
        return self.probs


class _HFModel(_GloveModel):
    NEEDS_TOKENIZE_STAGE = False


def _predictor(model_name: str = "encoder", model=None, error: str | None = None):
    tokenizer = MagicMock()
    tokenizer.cfg.max_len = 4
    with (
        patch("sentimentizer.predictor.get_trained_tokenizer", return_value=tokenizer),
        patch.object(SentimentPredictor, "_load_sentiment_model", return_value=(model, error)),
    ):
        return SentimentPredictor(model_name=model_name, device="cpu")


class TestPredictBatch:
    def test_returns_additive_format_in_input_order(self):
        model = _GloveModel([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
        results = _predictor(model=model).predict_batch(["great food", "bad"])

        assert [r["label"] for r in results] == ["positive", "negative"]
        assert results[0]["score"] == pytest.approx(0.7)
        assert results[0]["positive"] == pytest.approx(0.7)
        assert set(results[0]["scores"]) == {"negative", "neutral", "positive"}
        assert [r["token_count"] for r in results] == [2, 1]
        assert {r["model"] for r in results} == {"encoder"}
        assert model.inputs.shape == (2, 4)

    def test_single_predict_returns_first_batch_item(self):
        model = _GloveModel([[0.2, 0.7, 0.1]])
        assert _predictor(model=model).predict("okay")["label"] == "neutral"

    def test_hf_model_uses_its_own_tokenizer_with_configured_limit(self):
        model = _HFModel([[0.7, 0.2, 0.1]])
        model.tokenizer.return_value = {"input_ids": torch.zeros((1, 2))}
        result = _predictor("modernbert", model=model).predict_batch(["bad"])

        assert result[0]["label"] == "negative"
        assert model.tokenizer.call_args.kwargs["max_length"] == 512

    @pytest.mark.parametrize("call", ["predict", "predict_batch"])
    def test_unloaded_model_raises_with_cause(self, call):
        predictor = _predictor(error="no weights")
        argument = "x" if call == "predict" else ["x"]
        with pytest.raises(RuntimeError, match="no weights"):
            getattr(predictor, call)(argument)


class TestTokenize:
    def test_glove_tokenize_works_without_loaded_model(self):
        predictor = _predictor(error="no weights")
        predictor.tokenizer.dictionary.token2id = {"hello": 0}
        predictor.tokenizer.dictionary.__len__ = lambda _self: 1

        result = predictor.tokenize("Hello world")

        assert predictor.can_tokenize
        assert result["tokens"] == ["hello", "world"]
        assert result["token_ids"] == [1, 2, 0, 0]
        assert result["token_count"] == 2

    def test_hf_tokenize_uses_model_tokenizer(self):
        model = _HFModel([[1.0, 0.0, 0.0]])
        model.tokenizer.return_value = {"input_ids": [5, 6]}
        model.tokenizer.convert_ids_to_tokens.return_value = ["a", "b"]

        result = _predictor("modernbert", model=model).tokenize("ab")

        assert result == {"text": "ab", "tokens": ["a", "b"], "token_ids": [5, 6], "token_count": 2}

    def test_hf_model_load_failure_raises_clear_error(self):
        """A failed HF model has no tokenizer at all; this used to be an AttributeError."""
        predictor = _predictor("modernbert", error="no weights")

        assert predictor.tokenizer is None
        assert not predictor.can_tokenize
        with pytest.raises(RuntimeError, match="Tokenizer not available: no weights"):
            predictor.tokenize("hello")


class TestModelInfo:
    def test_loaded_model_reports_parameter_count(self):
        predictor = _predictor(model=_GloveModel([[1.0, 0.0, 0.0]]))
        info = predictor.get_sentiment_model_info()["encoder"]

        assert info["status"] == "loaded"
        assert info["parameters"] == 9
        assert info["max_sequence_length"] == 4

    def test_failed_model_reports_error_status(self):
        info = _predictor(error="boom").get_sentiment_model_info()["encoder"]
        assert info["status"] == "error"
        assert info["parameters"] == 0

    def test_unknown_model_has_no_config(self):
        predictor = _predictor("mystery", error="boom")
        assert predictor.get_sentiment_model_info()["status"] == "unknown_config"


class TestRouterArtifactDownload:
    def _load(self, target, download):
        predictor = _predictor(error="unused")
        with (
            patch("sentimentizer.predictor._ROUTER_AVAILABLE", True),
            patch("sentimentizer.predictor.DEFAULT_ROUTER_MODEL_PATH", str(target)),
            patch("huggingface_hub.snapshot_download", download),
        ):
            return predictor._load_router_model(str(target))

    def test_interrupted_download_leaves_no_artifact_directory(self, tmp_path):
        target = tmp_path / "models" / "router"

        def fail_midway(*, local_dir, **_kwargs):
            Path(local_dir).mkdir(parents=True)
            (Path(local_dir) / "config.json").write_text("{}")
            raise ConnectionError("network lost")

        model, error = self._load(target, fail_midway)

        assert model is None
        assert "not found" in error
        assert not target.exists()
        assert list(target.parent.iterdir()) == []

    def test_complete_download_is_published_and_loaded(self, tmp_path):
        target = tmp_path / "models" / "router"

        def complete(*, local_dir, revision, **_kwargs):
            assert revision
            Path(local_dir).mkdir(parents=True)
            (Path(local_dir) / "config.json").write_text("{}")

        with patch("sentimentizer.predictor.RouterModel.from_pretrained") as load:
            load.return_value.task = "task"
            model, error = self._load(target, complete)

        assert error is None
        assert model is load.return_value
        assert (target / "config.json").exists()
        assert list(target.parent.iterdir()) == [target]
