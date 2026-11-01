"""HTTP-level tests for sentiment and embedding handlers and shared middleware."""

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import Response
from fastapi.testclient import TestClient

from sentimentizer.serve.app import SentimentizerDeployment, create_fastapi_app
from sentimentizer.serve.base import ServiceMetrics
from sentimentizer.serve.embeddings_models import (
    DenseEmbeddingsRequest,
    EmbeddingsRequest,
    VectorBatchRequest,
    VectorRequest,
)
from sentimentizer.serve.models import BatchRequest, PredictRequest, TokenizeRequest

_Deployment = SentimentizerDeployment.func_or_class
_PREDICTION = {
    "label": "positive",
    "score": 0.9,
    "scores": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
    "token_count": 2,
    "model": "encoder",
    "positive": 0.9,
}


def _deployment(**predictor_attrs) -> MagicMock:
    predictor = MagicMock(
        model_name="encoder",
        model_loaded=True,
        model_error=None,
        can_tokenize=True,
        device="cpu",
        version="test",
        router_loaded=False,
        router_error=None,
    )
    predictor.predict_batch.return_value = [_PREDICTION, _PREDICTION]
    predictor.tokenize.return_value = {
        "text": "hi",
        "tokens": ["hi"],
        "token_ids": [1],
        "token_count": 1,
    }
    predictor.configure_mock(**predictor_attrs)
    dep = MagicMock()
    dep.predictor = predictor
    dep._ready = True
    dep._load_error = None
    dep._started_at = 0.0
    dep._sentiment_metrics = ServiceMetrics(prefix="sentimentizer")
    dep._embedding_metrics = ServiceMetrics(prefix="embedding")
    dep.predict_sentiment = AsyncMock(return_value=_PREDICTION)
    dep._format_prediction = _Deployment._format_prediction
    for name in ("_require_ready", "_require_embeddings", "_validate_model", "_observe_embedding"):
        setattr(dep, name, getattr(_Deployment, name).__get__(dep))
    return dep


def _client(dep: MagicMock) -> TestClient:
    api = create_fastapi_app("test", "test")

    @api.post("/predict")
    async def predict(body: PredictRequest, request: Request):
        return await _Deployment.predict(dep, body, request)

    @api.post("/batch")
    async def batch(body: BatchRequest, request: Request):
        return await _Deployment.batch(dep, body, request)

    @api.post("/tokenize")
    async def tokenize(body: TokenizeRequest):
        return await _Deployment.tokenize(dep, body)

    @api.get("/models")
    async def models():
        return await _Deployment.models(dep)

    @api.get("/models/{model_name}")
    async def model_detail(model_name: str):
        return await _Deployment.model_detail(dep, model_name)

    @api.post("/vectors")
    async def vector(body: VectorRequest):
        return await _Deployment.vector(dep, body)

    @api.post("/vectors/batch", response_class=Response)
    async def vectors_batch(body: VectorBatchRequest):
        return await _Deployment.vectors_batch(dep, body)

    @api.post("/embeddings")
    async def embeddings(body: EmbeddingsRequest):
        return await _Deployment.embeddings(dep, body)

    @api.post("/embeddings/dense")
    async def dense(body: DenseEmbeddingsRequest):
        return await _Deployment.dense_embeddings(dep, body)

    @api.get("/ready")
    async def ready():
        return await _Deployment.health_ready(dep)

    @api.get("/boom")
    async def boom():
        raise RuntimeError("secret detail")

    return TestClient(api, raise_server_exceptions=False)


def _embedding_client(vectors) -> tuple[TestClient, MagicMock]:
    dep = _deployment()
    dep._embeddings_handle = MagicMock()
    dep._embeddings_handle.dense.remote = AsyncMock(return_value=vectors)
    return _client(dep), dep


class TestSentimentHandlers:
    def test_predict_returns_contract_fields_and_records_metrics(self):
        dep = _deployment()
        response = _client(dep).post("/predict", json={"text": "great food"})

        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] == {
            "label": "positive",
            "score": 0.9,
            "model": "encoder",
            "token_count": 2,
        }
        assert body["latency_s"] >= 0
        dep.predict_sentiment.assert_awaited_once_with({"text": "great food"})

    def test_predict_failure_is_an_opaque_500_and_counts_as_error(self):
        dep = _deployment()
        dep.predict_sentiment.side_effect = RuntimeError("secret detail")
        response = _client(dep).post("/predict", json={"text": "x"})

        assert response.status_code == 500
        assert "secret" not in response.text
        assert response.json()["error"]["code"] == "internal_error"

    @pytest.mark.parametrize(
        ("path", "payload"), [("/predict", {"text": "x"}), ("/batch", {"texts": ["x"]})]
    )
    def test_unloaded_model_returns_503(self, path, payload):
        dep = _deployment(model_loaded=False, model_error="no weights")
        response = _client(dep).post(path, json=payload)

        assert response.status_code == 503
        assert "no weights" in response.json()["error"]["message"]

    @pytest.mark.parametrize(
        ("path", "payload"),
        [
            ("/predict", {"text": "x", "model": "rnn"}),
            ("/batch", {"texts": ["x"], "model": "rnn"}),
        ],
    )
    def test_wrong_requested_model_returns_400(self, path, payload):
        response = _client(_deployment()).post(path, json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_request"

    def test_batch_returns_all_results(self):
        response = _client(_deployment()).post("/batch", json={"texts": ["a", "b"]})

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert [item["prediction"]["label"] for item in body["results"]] == ["positive"] * 2

    def test_batch_failure_returns_500(self):
        dep = _deployment()
        dep.predictor.predict_batch.side_effect = RuntimeError("boom")
        assert _client(dep).post("/batch", json={"texts": ["a"]}).status_code == 500

    def test_tokenize_returns_tokens(self):
        response = _client(_deployment()).post("/tokenize", json={"text": "hi"})
        assert response.status_code == 200
        assert response.json()["tokens"] == ["hi"]

    def test_tokenize_without_tokenizer_returns_503_not_500(self):
        dep = _deployment(can_tokenize=False, model_error="no weights")
        response = _client(dep).post("/tokenize", json={"text": "hi"})

        assert response.status_code == 503
        assert "no weights" in response.json()["error"]["message"]
        dep.predictor.tokenize.assert_not_called()

    def test_models_reports_default(self):
        dep = _deployment()
        dep.predictor.get_sentiment_model_info.return_value = {"encoder": {"status": "loaded"}}
        body = _client(dep).get("/models").json()
        assert body == {"models": {"encoder": {"status": "loaded"}}, "default": "encoder"}

    def test_model_detail_status_codes(self):
        dep = _deployment()
        dep.predictor.get_sentiment_model_info.return_value = {"encoder": {"status": "loaded"}}
        client = _client(dep)

        assert client.get("/models/encoder").json()["model"] == "encoder"
        assert client.get("/models/mystery").status_code == 400
        assert client.get("/models/rnn").status_code == 404

    def test_health_ready_does_not_name_a_model_that_failed_to_load(self):
        dep = _deployment(model_loaded=False, model_error="no weights")
        response = _client(dep).get("/ready")

        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert response.json()["model_loaded"] is None


class TestEmbeddingHandlers:
    @pytest.fixture(autouse=True)
    def _enabled(self):
        cfg = MagicMock(
            embeddings_enabled=True,
            bge_m3_enabled=True,
            bge_m3_model_id="bge",
            dense_embedding_model_id="nomic",
        )
        # @serve.ingress cloudpickles the class, so handlers have their own module globals.
        with patch.dict(_Deployment.vector.__globals__, {"cfg": cfg}):
            yield cfg

    def test_vector_returns_first_vector(self):
        client, dep = _embedding_client([[0.5, 0.25]])
        assert client.post("/vectors", json={"text": "x"}).json() == {"vector": [0.5, 0.25]}
        assert dep._embedding_metrics.request_count == 1

    def test_vectors_batch_uses_little_endian_wire_format(self):
        client, _dep = _embedding_client([[0.5, 0.25], [1.0, 2.0]])
        response = client.post("/vectors/batch", json={"texts": ["a", "b"]})

        assert response.headers["content-type"] == "application/octet-stream"
        assert struct.unpack("<II", response.content[:8]) == (2, 2)
        assert struct.unpack("<4f", response.content[8:]) == (0.5, 0.25, 1.0, 2.0)

    def test_vectors_batch_rejects_mismatched_dimensions(self):
        client, _dep = _embedding_client([[0.5, 0.25], [1.0]])
        assert client.post("/vectors/batch", json={"texts": ["a", "b"]}).status_code == 500

    def test_dense_embeddings_include_model(self):
        client, _dep = _embedding_client([[0.5]])
        body = client.post("/embeddings/dense", json={"texts": ["a"]}).json()
        assert body == {"model": "nomic", "vectors": [[0.5]]}

    def test_bge_m3_preserves_input_order(self):
        client, dep = _embedding_client([])
        dep._embeddings_handle.bge_m3.remote = AsyncMock(side_effect=lambda text: text)
        body = client.post("/embeddings", json={"texts": ["a", "b"]}).json()
        assert body == {"backend": "flag_embedding", "model": "bge", "vectors": ["a", "b"]}

    def test_embedding_failure_is_counted_as_error(self):
        client, dep = _embedding_client([])
        dep._embeddings_handle.dense.remote = AsyncMock(side_effect=RuntimeError("down"))
        assert client.post("/vectors", json={"text": "x"}).status_code == 500
        assert dep._embedding_metrics.error_count == 1

    def test_disabled_embeddings_return_503(self, _enabled):
        _enabled.embeddings_enabled = False
        client, _dep = _embedding_client([])
        assert client.post("/vectors", json={"text": "x"}).status_code == 503

    def test_disabled_bge_m3_returns_503(self, _enabled):
        _enabled.bge_m3_enabled = False
        client, _dep = _embedding_client([])
        assert client.post("/embeddings", json={"texts": ["x"]}).status_code == 503


class TestMiddleware:
    ORIGIN = {"Origin": "https://app.example"}

    def test_unhandled_500_has_cors_and_request_id_headers(self):
        client = _client(_deployment())
        response = client.get("/boom", headers={**self.ORIGIN, "X-Request-Id": "abc-1"})

        assert response.status_code == 500
        assert response.headers["access-control-allow-origin"] == "*"
        assert response.headers["x-request-id"] == "abc-1"
        assert response.json()["error"]["request_id"] == "abc-1"
        assert "secret" not in response.text

    def test_oversized_body_has_cors_and_request_id_headers(self):
        client = _client(_deployment())
        response = client.post("/predict", content=b"x" * (1024 * 1024 + 1), headers=self.ORIGIN)

        assert response.status_code == 413
        assert response.headers["access-control-allow-origin"] == "*"
        assert response.headers["x-request-id"]

    @pytest.mark.parametrize("supplied", ["a b c", "x" * 129, "bad<script>", ""])
    def test_invalid_request_id_is_replaced(self, supplied):
        response = _client(_deployment()).get("/boom", headers={"X-Request-Id": supplied})
        assert response.headers["x-request-id"] != supplied
        assert len(response.headers["x-request-id"]) == 36

    @pytest.mark.parametrize("chunks,status", [(2, 422), (20, 413)])
    def test_chunked_bodies_are_limited_and_still_readable(self, chunks, status):
        def body():
            yield b'{"not_text":"'
            for _ in range(chunks):
                yield b"a" * 65536
            yield b'"}'

        response = _client(_deployment()).post(
            "/predict", content=body(), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status
        if status == 422:
            # Only reachable if the buffered body was replayed to the route.
            assert response.json()["detail"][0]["loc"] == ["body", "text"]
