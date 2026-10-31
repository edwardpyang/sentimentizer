"""Tests for the dense embedding predictor's wire-contract guarantees."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from sentimentizer.embeddings import predictor as predictor_module
from sentimentizer.embeddings.predictor import (
    DENSE_DIM,
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    DenseEmbeddingPredictor,
)


def _predictor(vectors=None):
    model = MagicMock()
    model.encode.return_value = np.zeros((1, DENSE_DIM)) if vectors is None else vectors
    factory = MagicMock(return_value=model)
    return DenseEmbeddingPredictor("id", "rev", device="cpu", model_factory=factory), model, factory


def test_factory_receives_pinned_revision():
    _predictor_obj, _model, factory = _predictor()
    assert factory.call_args.args == ("id",)
    assert factory.call_args.kwargs["revision"] == "rev"
    assert factory.call_args.kwargs["device"] == "cpu"


@pytest.mark.parametrize(
    ("mode", "prefix"), [("query", QUERY_PREFIX), ("document", DOCUMENT_PREFIX)]
)
def test_mode_selects_prefix(mode, prefix):
    predictor, model, _factory = _predictor()
    result = predictor.encode(["text"], mode)

    assert model.encode.call_args.args[0] == [prefix + "text"]
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True
    assert len(result[0]) == DENSE_DIM
    assert isinstance(result[0][0], float)


def test_empty_input_skips_model():
    predictor, model, _factory = _predictor()
    assert predictor.encode([], "query") == []
    model.encode.assert_not_called()


def test_unknown_mode_raises():
    predictor, _model, _factory = _predictor()
    with pytest.raises(ValueError, match="unsupported embedding mode"):
        predictor.encode(["x"], "other")


def test_wrong_dimension_raises():
    predictor, _model, _factory = _predictor(np.zeros((1, 3)))
    with pytest.raises(RuntimeError, match="returned 3 dimensions"):
        predictor.encode(["x"], "query")


def test_missing_dependency_is_reported(monkeypatch):
    monkeypatch.setattr(predictor_module, "SentenceTransformer", None)
    with pytest.raises(RuntimeError, match="embeddings extra"):
        DenseEmbeddingPredictor("id", "rev", device="cpu")
