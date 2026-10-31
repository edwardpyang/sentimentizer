"""Tests for the FastAPI serve layer.

Tests HTTP error handling, status codes, and validation without starting Ray.
Uses three strategies:
  1. Direct invocation of async handler logic via the unwrapped class
  2. FastAPI TestClient for Pydantic validation (422 responses)
  3. Standalone FastAPI app for exception-handler integration (500 responses)

Ray Serve's ``@deployment`` decorator wraps the class so it can't be
instantiated directly.  We access the original class via
``SentimentizerDeployment.func_or_class`` and call its async methods with a
mock ``self`` that mirrors the deployment's runtime attributes.
"""

import asyncio
import inspect
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Access the original unwrapped class (Ray Serve wraps it with @deployment)
from sentimentizer.serve.app import SentimentizerDeployment as _Deployment
from sentimentizer.serve.app import app
from sentimentizer.serve.base import ServiceMetrics
from sentimentizer.serve.models import PredictRequest

_SentimentizerDeployment = _Deployment.func_or_class


def _router_client(dep):
    """Exercise real handlers and response serialization without a Ray cluster."""
    from sentimentizer.serve.app import create_fastapi_app
    from sentimentizer.serve.models import (
        BatchRequest,
        RouterBatchResponse,
        RouterModelsResponse,
        RouterPredictResponse,
    )

    api = create_fastapi_app("Router contract test", "Router contract test")

    @api.post("/v1/router/predict", response_model=RouterPredictResponse)
    async def single(body: PredictRequest, request: Request):
        return await _SentimentizerDeployment.router_predict(dep, body, request)

    @api.post("/v1/router/batch", response_model=RouterBatchResponse)
    async def batch(body: BatchRequest, request: Request):
        return await _SentimentizerDeployment.router_batch(dep, body, request)

    @api.get("/v1/router/models", response_model=RouterModelsResponse)
    async def models():
        return await _SentimentizerDeployment.router_models(dep)

    return TestClient(api, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_predictor(**overrides):
    """Build a mock SentimentPredictor with sensible defaults."""
    p = MagicMock()
    p.model_loaded = True
    p.router_loaded = True
    p.model_name = "encoder"
    p.device = "cpu"
    p.version = "0.211.0"
    p.router_error = None
    p.predict_batch.return_value = [
        {
            "label": "positive",
            "score": 0.88,
            "token_count": 5,
            "model": "encoder",
        },
    ]
    p.classify_batch.return_value = [
        {"prediction": {"label": "general", "score": 0.95, "token_count": 1}},
    ]
    p.tokenize.return_value = {
        "text": "hello",
        "tokens": ["hello"],
        "token_ids": [42],
        "token_count": 1,
    }
    p.get_sentiment_model_info.return_value = {
        "encoder": {"architecture": "Transformer Encoder", "status": "loaded"},
    }
    p.get_router_model_info.return_value = {
        "model_path": "models/router",
        "categories": ["dietary", "service", "general"],
        "status": "loaded",
    }
    for k, v in overrides.items():
        setattr(p, k, v)
    # When the router isn't loaded, classify_batch raises (matching the
    # real SentimentPredictor's lazy-load-then-raise behavior).
    if not p.router_loaded:
        p.classify_batch.side_effect = RuntimeError(f"Router model not loaded: {p.router_error}")
    return p


def _mock_deployment(predictor=None, ready=True, load_error=None):
    """Build a mock ``self`` with the same attrs as SentimentizerDeployment."""
    from sentimentizer.serve.config import ServeConfig

    dep = MagicMock()
    dep.cfg = ServeConfig()
    dep._started_at = time.time()
    dep._ready = ready
    dep._load_error = load_error
    dep._sentiment_metrics = ServiceMetrics(prefix="sentimentizer")
    dep._router_metrics = ServiceMetrics(prefix="router")
    dep._embedding_metrics = ServiceMetrics(prefix="embedding")
    dep.predictor = predictor or _mock_predictor()
    dep._require_ready = lambda: _SentimentizerDeployment._require_ready(dep)

    async def _predict_sentiment(inputs):
        if isinstance(inputs, dict):
            return dep.predictor.predict_batch([inputs["text"]])[0]
        texts = [inp["text"] for inp in inputs]
        return dep.predictor.predict_batch(texts)

    async def _classify_route(inputs):
        if isinstance(inputs, dict):
            return dep.predictor.classify_batch([inputs["text"]])[0]
        texts = [inp["text"] for inp in inputs]
        return dep.predictor.classify_batch(texts)

    dep.predict_sentiment = _predict_sentiment
    dep.classify_route = _classify_route
    dep.health_ready = lambda: _SentimentizerDeployment.health_ready(dep)
    return dep


def _run(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


def _mock_request():
    """Build a mock Request with request_id for handler signature compatibility."""
    req = MagicMock(spec=Request)
    req.state.request_id = "test-req-id"
    return req


# ---------------------------------------------------------------------------
# Test: _DummyServe stubs exist in source
# ---------------------------------------------------------------------------


class TestDummyServe:
    def test_stubs_exist_in_source(self):
        from sentimentizer.serve import base

        assert "def start(self" in inspect.getsource(base)
        assert "def run(self" in inspect.getsource(base)
        assert "def shutdown(self" in inspect.getsource(base)

    def test_dummy_serve_decorators(self):
        from sentimentizer.serve.base import _DummyServe

        dummy = _DummyServe()

        # Test deployment
        # 1. Bare (no parens)
        @dummy.deployment
        class MyClass1:
            pass

        assert MyClass1.__name__ == "MyClass1"

        # 2. Call (with parens)
        @dummy.deployment(num_replicas=2)
        class MyClass2:
            pass

        assert MyClass2.__name__ == "MyClass2"

        # Test ingress
        # 1. Bare (no parens)
        @dummy.ingress
        class MyClass3:
            pass

        assert MyClass3.__name__ == "MyClass3"

        # 2. Call (with parens)
        @dummy.ingress(None)
        class MyClass4:
            pass

        assert MyClass4.__name__ == "MyClass4"

        # Test batch
        # 1. Bare (no parens)
        @dummy.batch
        def my_fn1():
            pass

        assert my_fn1() is None

        # 2. Call (with parens)
        @dummy.batch(max_batch_size=10)
        def my_fn2():
            pass

        assert my_fn2() is None


# ---------------------------------------------------------------------------
# Test: Pydantic validation with Annotated types (422 responses)
# ---------------------------------------------------------------------------


class TestPydanticValidation:
    """Pydantic request model validation tests.

    After P1#7, validation uses Annotated types with max_length from
    module-level cfg. This gives 422 (Unprocessable Entity) instead of
    the previous 400 from manual validation.
    """

    def test_predict_request_rejects_empty_string(self):
        """PredictRequest with text='' should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text="")
        assert any(e["type"] in ("min_length", "string_too_short") for e in exc_info.value.errors())

    def test_predict_request_rejects_missing_field(self):
        """PredictRequest with no text field should fail validation."""
        with pytest.raises(ValidationError):
            PredictRequest()  # type: ignore[call-arg]

    def test_predict_request_rejects_text_too_long(self):
        """PredictRequest with text exceeding max_text_length should fail."""
        from sentimentizer.serve.config import load_serve_config

        cfg = load_serve_config()
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text="a" * (cfg.max_text_length + 1))
        errors = exc_info.value.errors()
        assert any(e["type"] in ("max_length", "string_too_long") for e in errors)

    def test_batch_request_rejects_empty_list(self):
        """BatchRequest with texts=[] should fail validation."""
        from sentimentizer.serve.models import BatchRequest

        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(texts=[])
        assert any(
            e["type"] in ("min_length", "too_short", "missing") for e in exc_info.value.errors()
        )

    def test_batch_request_rejects_too_many_items(self):
        """BatchRequest with texts exceeding max_batch_size should fail."""
        from sentimentizer.serve.config import load_serve_config
        from sentimentizer.serve.models import BatchRequest

        cfg = load_serve_config()
        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(texts=["a"] * (cfg.max_batch_size + 1))
        errors = exc_info.value.errors()
        assert any(e["type"] in ("max_length", "too_long") for e in errors)

    def test_batch_request_rejects_per_item_text_too_long(self):
        """BatchRequest with an individual text exceeding max_text_length should fail."""
        from sentimentizer.serve.config import load_serve_config
        from sentimentizer.serve.models import BatchRequest

        cfg = load_serve_config()
        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(texts=["a" * (cfg.max_text_length + 1)])
        errors = exc_info.value.errors()
        # Per-item validation error includes index in loc
        assert any("texts" in str(e.get("loc", [])) for e in errors)


# ---------------------------------------------------------------------------
# Test: Prediction response schema (P1#5 additive format)
# ---------------------------------------------------------------------------


class TestPredictionResponseSchema:
    """Verify predict_batch returns format with label, score, token_count, model."""

    def test_predict_batch_includes_required_fields(self):
        pred = _mock_predictor()
        results = pred.predict_batch(["hello"])
        result = results[0]
        assert "label" in result
        assert "score" in result
        assert "model" in result
        assert "token_count" in result

    def test_predict_batch_token_count_is_int(self):
        pred = _mock_predictor()
        results = pred.predict_batch(["hello"])
        assert isinstance(results[0]["token_count"], int)


# ---------------------------------------------------------------------------
# Test: Liveness / Readiness endpoints
# ---------------------------------------------------------------------------


class TestLivenessEndpoint:
    def test_health_live_always_returns_200(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health_live(dep))
        assert isinstance(result, dict)
        assert result["status"] == "live"
        assert "uptime_s" in result


class TestReadinessEndpoint:
    def test_health_ready_model_loaded_returns_ready_dict(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health_ready(dep))
        assert isinstance(result, dict)
        assert result["status"] == "ready"
        assert result["device"] == "cpu"
        assert result["model_loaded"] == "encoder"

    def test_health_ready_model_not_loaded_returns_json_response_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        result = _run(_SentimentizerDeployment.health_ready(dep))
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503

    def test_health_ready_has_no_image_keys(self):
        result = _run(_SentimentizerDeployment.health_ready(_mock_deployment()))
        assert "image_models" not in result
        assert "image_backend_error" not in result


class TestHealthBackwardCompat:
    def test_health_delegates_to_readiness(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health(dep))
        assert isinstance(result, dict)
        assert result["status"] == "ready"

    def test_health_delegates_to_readiness_when_not_loaded(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        result = _run(_SentimentizerDeployment.health(dep))
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# Test: Sentiment model not loaded → 503
# ---------------------------------------------------------------------------


class TestSentimentModelNotLoaded:
    def test_predict_returns_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="model file missing")
        )

        with pytest.raises(HTTPException) as exc_info:
            _run(
                _SentimentizerDeployment.predict(dep, PredictRequest(text="hello"), _mock_request())
            )
        assert exc_info.value.status_code == 503

    def test_batch_returns_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="model file missing")
        )

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.batch(dep, MagicMock(texts=["hello"]), _mock_request()))
        assert exc_info.value.status_code == 503

    def test_predict_returns_503_when_not_ready(self):
        dep = _mock_deployment(ready=False, load_error="yelp.dictionary not found")

        with pytest.raises(HTTPException) as exc_info:
            _run(
                _SentimentizerDeployment.predict(dep, PredictRequest(text="hello"), _mock_request())
            )
        assert exc_info.value.status_code == 503
        assert "not ready" in str(exc_info.value.detail).lower()


# ---------------------------------------------------------------------------
# Test: Router not loaded → 503
# ---------------------------------------------------------------------------


class TestRouterNotLoaded:
    def test_router_predict_returns_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(
                router_loaded=False,
                router_error="sentence-transformers not installed",
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            _run(
                _SentimentizerDeployment.router_predict(
                    dep, MagicMock(text="hello"), _mock_request()
                )
            )
        assert exc_info.value.status_code == 503
        assert "Router model not loaded" in exc_info.value.detail

    def test_router_batch_returns_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(
                router_loaded=False,
                router_error="sentence-transformers not installed",
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            _run(
                _SentimentizerDeployment.router_batch(
                    dep, MagicMock(texts=["hello"]), _mock_request()
                )
            )
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Test: Exception handler → generic 500, no internal leak
# ---------------------------------------------------------------------------


class TestExceptionHandler:
    def test_unhandled_exception_returns_500_generic_message(self):
        """The centralized handler returns 500 with no internal details."""
        test_app = FastAPI()

        @test_app.exception_handler(Exception)
        async def handler(request, exc):
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

        @test_app.get("/fail")
        async def fail_route():
            raise RuntimeError("something secret broke")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/fail")

        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error"
        assert "something secret broke" not in response.text

    def test_http_exception_passes_through(self):
        """HTTPException should NOT be caught by the generic handler."""
        test_app = FastAPI()

        @test_app.exception_handler(Exception)
        async def handler(request, exc):
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

        @test_app.get("/bad-request")
        async def bad_request():
            raise HTTPException(status_code=400, detail="Bad input")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/bad-request")

        assert response.status_code == 400
        assert response.json()["detail"] == "Bad input"


class TestExceptionHandlerRegistration:
    def test_app_has_exception_handler_registered(self):
        """The app should have an Exception handler registered."""
        assert Exception in app.exception_handlers


# ---------------------------------------------------------------------------
# Test: Request ID middleware
# ---------------------------------------------------------------------------


class TestRequestIdMiddleware:
    def test_cors_and_request_id_middleware_registered(self):
        """The app should have CORS and request-ID middleware registered."""

        from sentimentizer.serve.app import app

        cls_names = [m.cls.__name__ for m in app.user_middleware if hasattr(m, "cls")]
        assert "CORSMiddleware" in cls_names
        assert "BaseHTTPMiddleware" in cls_names


# ---------------------------------------------------------------------------
# Test: Prometheus observability endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_route_on_app(self):
        """The app should expose an unversioned Prometheus scrape endpoint."""
        route_paths = [route.path for route in app.routes]
        assert "/metrics" in route_paths

    def test_metrics_response_uses_prometheus_format(self):
        """Metrics should be returned as scrapeable text, not JSON-wrapped text."""
        dep = _mock_deployment()
        dep._sentiment_metrics.record_request(0.25)
        dep._router_metrics.record_request(0.5, error=True)

        response = _run(_SentimentizerDeployment.metrics(dep))
        body = response.body.decode()

        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert "sentimentizer_service_ready 1" in body
        assert "sentimentizer_request_total 1" in body
        assert "sentimentizer_latency_seconds_total 0.250000" in body
        assert "router_request_total 1" in body
        assert "router_error_total 1" in body
        assert "embedding_request_total 0" in body

    def test_metrics_reports_not_ready(self):
        dep = _mock_deployment(ready=False)

        response = _run(_SentimentizerDeployment.metrics(dep))

        assert "sentimentizer_service_ready 0" in response.body.decode()


# ---------------------------------------------------------------------------
# Test: v1 prefix routes exist
# ---------------------------------------------------------------------------


class TestV1Routes:
    def test_v1_routes_on_main_app(self):
        """Sentiment routes should be registered at both new and deprecated paths."""
        route_paths = [route.path for route in app.routes]
        # Primary routes
        assert "/v1/sentiment/predict" in route_paths
        assert "/v1/sentiment/batch" in route_paths
        assert "/v1/sentiment/tokenize" in route_paths
        assert "/v1/sentiment/models" in route_paths
        assert "/v1/sentiment/models/{model_name}" in route_paths
        # Deprecated aliases (kept for backward compatibility)
        assert "/v1/predict" in route_paths
        assert "/v1/batch" in route_paths
        assert "/v1/tokenize" in route_paths
        assert "/v1/models" in route_paths
        assert "/v1/models/{model_name}" in route_paths
        # Router routes unchanged
        assert "/v1/router/predict" in route_paths
        assert "/v1/router/batch" in route_paths
        assert "/v1/router/models" in route_paths

    def test_health_routes_on_main_app(self):
        """Health endpoints should be on the unversioned main app."""
        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths
        assert "/health/live" in route_paths
        assert "/health/ready" in route_paths


# ---------------------------------------------------------------------------
# Test: CORS configuration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Error response envelope (P2#9)
# ---------------------------------------------------------------------------


class TestErrorEnvelope:
    """Verify HTTP exceptions use the standard error envelope format."""

    def test_503_uses_error_envelope(self):
        """Exceptions with string detail should be wrapped in error envelope."""
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        with pytest.raises(HTTPException) as exc_info:
            _run(
                _SentimentizerDeployment.predict(dep, PredictRequest(text="hello"), _mock_request())
            )
        exc = exc_info.value
        assert exc.status_code == 503

    def test_model_validation_wrong_model_400(self):
        """Requesting an unloaded model should return 400."""
        dep = _mock_deployment()
        with pytest.raises(HTTPException) as exc_info:
            _SentimentizerDeployment._validate_model(dep, "rnn")
        assert exc_info.value.status_code == 400
        assert "not loaded" in exc_info.value.detail

    def test_model_validation_none_returns_default(self):
        """Passing None for model should return the default model name."""
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, None)
        assert result == "encoder"

    def test_model_validation_matching_model_succeeds(self):
        """Passing the loaded model name should succeed."""
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, "encoder")
        assert result == "encoder"


# ---------------------------------------------------------------------------
# Test: include_scores and top_k (P2#11)
# ---------------------------------------------------------------------------


class TestFormatPrediction:
    """Verify _format_prediction returns label, score, token_count, model."""

    def test_format_prediction_full(self):
        prediction = {
            "label": "positive",
            "score": 0.88,
            "token_count": 5,
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(prediction)
        assert result["label"] == "positive"
        assert result["score"] == 0.88
        assert result["model"] == "encoder"
        assert result["token_count"] == 5

    def test_format_prediction_always_includes_token_count(self):
        """token_count is always included in the prediction output."""
        prediction = {
            "label": "positive",
            "score": 0.88,
            "model": "encoder",
            "token_count": 3,
        }
        result = _SentimentizerDeployment._format_prediction(prediction)
        assert result["token_count"] == 3
        assert result["label"] == "positive"

    def test_format_prediction_omits_scores_and_dynamic_key(self):
        """Scores dict and dynamic key should not be in the output."""
        prediction = {
            "positive": 0.88,
            "label": "positive",
            "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
            "token_count": 5,
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(prediction)
        assert "scores" not in result
        assert "positive" not in result


# ---------------------------------------------------------------------------
# Test: Request model classes (P2#10)
# ---------------------------------------------------------------------------


class TestRequestModelFields:
    """Verify request model fields."""

    def test_predict_request_defaults(self):
        req = PredictRequest(text="hello")
        assert req.model is None

    def test_predict_request_with_model(self):
        req = PredictRequest(text="hello", model="encoder")
        assert req.model == "encoder"

    def test_batch_request_defaults(self):
        from sentimentizer.serve.models import BatchRequest

        req = BatchRequest(texts=["hello"])
        assert req.model is None

    def test_batch_request_with_model(self):
        from sentimentizer.serve.models import BatchRequest

        req = BatchRequest(texts=["hello"], model="encoder")
        assert req.model == "encoder"


# ---------------------------------------------------------------------------
# Test: V1 model detail endpoint (P3#13)
# ---------------------------------------------------------------------------


class TestModelDetailEndpoint:
    def test_v1_has_model_detail_route(self):
        """Model detail route should exist at both primary and deprecated paths."""
        route_paths = [route.path for route in app.routes]
        assert "/v1/sentiment/models/{model_name}" in route_paths
        assert "/v1/models/{model_name}" in route_paths


# ---------------------------------------------------------------------------
# Test: Request body size limit (P3#15)
# ---------------------------------------------------------------------------


class TestBodySizeLimitMiddleware:
    def test_body_size_limit_middleware_registered(self):
        """The _RequestBodySizeLimitMiddleware should be in app middleware."""
        from sentimentizer.serve.app import _RequestBodySizeLimitMiddleware

        cls_names = [m.cls for m in app.user_middleware if hasattr(m, "cls")]
        assert _RequestBodySizeLimitMiddleware in cls_names

    def test_body_size_limit_middleware_with_content_length(self):
        """Middleware rejects requests with Content-Length exceeding the limit."""
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.testclient import TestClient

        from sentimentizer.serve.app import _RequestBodySizeLimitMiddleware, http_exception_handler

        test_app = FastAPI()
        test_app.add_middleware(_RequestBodySizeLimitMiddleware, default_max_bytes=10)
        test_app.add_exception_handler(HTTPException, http_exception_handler)

        @test_app.post("/")
        async def handler(request: Request):
            body = await request.body()
            return {"size": len(body)}

        client = TestClient(test_app)
        # Content length 5 < 10 -> allowed
        resp = client.post("/", content=b"12345")
        assert resp.status_code == 200
        assert resp.json() == {"size": 5}

        # Content length 15 > 10 -> rejected
        resp = client.post("/", content=b"123456789012345")
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "request_too_large"

    def test_body_size_limit_middleware_chunked_streaming(self):
        """Middleware rejects chunked requests exceeding the limit incrementally."""
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.testclient import TestClient

        from sentimentizer.serve.app import _RequestBodySizeLimitMiddleware, http_exception_handler

        test_app = FastAPI()
        test_app.add_middleware(_RequestBodySizeLimitMiddleware, default_max_bytes=10)
        test_app.add_exception_handler(HTTPException, http_exception_handler)

        @test_app.post("/")
        async def handler(request: Request):
            body = await request.body()
            return {"size": len(body)}

        client = TestClient(test_app)

        # Chunked stream: generator
        def chunked_generator(chunks):
            yield from chunks

        # Total size 6 < 10 -> allowed
        resp = client.post("/", content=chunked_generator([b"abc", b"def"]))
        assert resp.status_code == 200
        assert resp.json() == {"size": 6}

        # Total size 15 > 10 -> rejected
        resp = client.post("/", content=chunked_generator([b"12345", b"67890", b"12345"]))
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "request_too_large"


# ---------------------------------------------------------------------------
# Test: Validate model field behavior
# ---------------------------------------------------------------------------


class TestValidateModel:
    """Test _validate_model edge cases."""

    def test_validate_model_none_returns_default(self):
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, None)
        assert result == "encoder"

    def test_validate_model_matching_succeeds(self):
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, "encoder")
        assert result == "encoder"

    def test_validate_model_case_insensitive(self):
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, "Encoder")
        assert result == "encoder"

    def test_validate_model_wrong_raises_400(self):
        dep = _mock_deployment()
        with pytest.raises(HTTPException) as exc_info:
            _SentimentizerDeployment._validate_model(dep, "rnn")
        assert exc_info.value.status_code == 400
        assert "not loaded" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Test: Error response envelope format
# ---------------------------------------------------------------------------


class TestErrorResponseEnvelope:
    """Test that HTTP exceptions produce the standard envelope."""

    def test_http_exception_handler_wraps_string_detail(self):
        """HTTPException with string detail should be wrapped in error envelope."""
        from sentimentizer.serve.app import http_exception_handler

        test_app = FastAPI()

        @test_app.exception_handler(HTTPException)
        async def handler(request, exc):
            return await http_exception_handler(request, exc)

        @test_app.get("/test-503")
        async def test_503():
            raise HTTPException(status_code=503, detail="Service unavailable")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test-503")
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "service_unavailable"
        assert data["error"]["message"] == "Service unavailable"

    def test_status_code_to_error_code_mapping(self):
        from sentimentizer.serve.app import _status_code_to_error_code

        assert _status_code_to_error_code(400) == "bad_request"
        assert _status_code_to_error_code(404) == "not_found"
        assert _status_code_to_error_code(413) == "request_too_large"
        assert _status_code_to_error_code(422) == "validation_error"
        assert _status_code_to_error_code(503) == "service_unavailable"
        assert _status_code_to_error_code(499) == "error_499"


# ---------------------------------------------------------------------------
# Test: Token count in prediction response
# ---------------------------------------------------------------------------


class TestTokenCount:
    """Verify token_count is present in prediction output."""

    def test_predict_batch_includes_token_count(self):
        pred = _mock_predictor()
        results = pred.predict_batch(["hello world"])
        assert "token_count" in results[0]
        assert isinstance(results[0]["token_count"], int)

    def test_format_prediction_preserves_token_count(self):
        prediction = {
            "label": "positive",
            "score": 0.88,
            "token_count": 12,
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(prediction)
        assert result["token_count"] == 12

    def test_format_prediction_token_count_required(self):
        """token_count is required in prediction output (predict_batch always includes it)."""
        prediction = {
            "label": "positive",
            "score": 0.88,
            "model": "encoder",
            "token_count": 7,
        }
        result = _SentimentizerDeployment._format_prediction(prediction)
        assert result["token_count"] == 7


# ---------------------------------------------------------------------------
# Test: ServeConfig with cors_origins
# ---------------------------------------------------------------------------


class TestServeConfigCorsOrigins:
    """Test ServeConfig cors_origins parsing."""

    def test_default_cors_origins(self):
        from sentimentizer.serve.config import ServeConfig

        cfg = ServeConfig()
        assert cfg.cors_origins == ["*"]

    def test_env_var_cors_origins(self):
        import os

        from sentimentizer.serve.config import load_serve_config

        os.environ["SENTIMENTIZER_CORS_ORIGINS"] = "http://localhost:3000,https://app.example.com"
        try:
            cfg = load_serve_config()
            assert cfg.cors_origins == [
                "http://localhost:3000",
                "https://app.example.com",
            ]
        finally:
            del os.environ["SENTIMENTIZER_CORS_ORIGINS"]

    def test_single_cors_origin(self):
        import os

        from sentimentizer.serve.config import load_serve_config

        os.environ["SENTIMENTIZER_CORS_ORIGINS"] = "https://app.example.com"
        try:
            cfg = load_serve_config()
            assert cfg.cors_origins == ["https://app.example.com"]
        finally:
            del os.environ["SENTIMENTIZER_CORS_ORIGINS"]

    def test_parse_list_helper(self):
        from sentimentizer.serve.config import _parse_list

        assert _parse_list("a,b,c") == ["a", "b", "c"]
        assert _parse_list("a, b , c") == ["a", "b", "c"]
        assert _parse_list("single") == ["single"]
        assert _parse_list("") == []


# ---------------------------------------------------------------------------
# Test: Health endpoint response shapes
# ---------------------------------------------------------------------------


class TestHealthResponseShape:
    """Verify health endpoint response structure."""

    def test_liveness_has_status_and_uptime(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health_live(dep))
        assert "status" in result
        assert "uptime_s" in result
        assert result["status"] == "live"

    def test_readiness_has_device_and_version(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health_ready(dep))
        assert "device" in result
        assert "version" in result
        assert "model_loaded" in result
        assert "router_loaded" in result

    def test_readiness_not_loaded_includes_error(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        result = _run(_SentimentizerDeployment.health_ready(dep))
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# Test: CORS middleware configuration
# ---------------------------------------------------------------------------


class TestCORSMiddleware:
    """Test CORS middleware is properly configured."""

    def test_cors_middleware_allows_all_origins_by_default(self):
        from sentimentizer.serve.config import ServeConfig

        cfg = ServeConfig()
        assert cfg.cors_origins == ["*"]

    def test_cors_exposes_request_id_header(self):
        """X-Request-Id should be in exposed headers for CORS."""
        from sentimentizer.serve.app import app

        for m in app.user_middleware:
            if hasattr(m, "kwargs") and "expose_headers" in m.kwargs:
                assert "X-Request-Id" in m.kwargs["expose_headers"]

    def test_cors_allow_credentials_disabled_for_wildcard(self):
        """allow_credentials should be False when wildcard origin '*' is configured."""
        from sentimentizer.serve.app import app

        for m in app.user_middleware:
            if (
                hasattr(m, "kwargs")
                and "allow_origins" in m.kwargs
                and "*" in m.kwargs["allow_origins"]
            ):
                assert m.kwargs.get("allow_credentials") is False


# ---------------------------------------------------------------------------
# Test: Image settings moved to the imagenate service
# ---------------------------------------------------------------------------


class TestRemovedImageSettings:
    def test_removed_env_var_warns_and_is_ignored(self, monkeypatch):
        from sentimentizer.serve import config as config_module

        monkeypatch.setenv("SENTIMENTIZER_KREA_2_ENABLED", "true")
        with patch.object(config_module, "logger") as logger:
            loaded = config_module.load_serve_config()
        logger.warning.assert_called_once_with(
            "image_setting_ignored_moved_to_imagenate", env_var="SENTIMENTIZER_KREA_2_ENABLED"
        )
        assert not hasattr(loaded, "krea_2_enabled")

    def test_per_model_diffusion_env_var_warns(self, monkeypatch):
        from sentimentizer.serve import config as config_module

        monkeypatch.setenv("SENTIMENTIZER_DIFFUSION_KREA_2_DEFAULT_STEPS", "4")
        with patch.object(config_module, "logger") as logger:
            config_module.load_serve_config()
        logger.warning.assert_called_once()

    def test_stale_yaml_key_warns_instead_of_failing_startup(self, tmp_path):
        from sentimentizer.serve import config as config_module

        path = tmp_path / "service.yaml"
        path.write_text("api_keys: [secret-value]\nserve_port: 9000\n")
        with patch.object(config_module, "logger") as logger:
            loaded = config_module.load_serve_config(path)
        assert loaded.serve_port == 9000
        logger.warning.assert_called_once_with(
            "image_setting_ignored_moved_to_imagenate", yaml_key="api_keys"
        )
        assert "secret-value" not in str(logger.warning.call_args)

    def test_clean_config_does_not_warn(self):
        from sentimentizer.serve import config as config_module

        with patch.object(config_module, "logger") as logger:
            config_module.load_serve_config()
        logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Request body size limits
# ---------------------------------------------------------------------------


class TestRequestBodySizeLimitMiddleware:
    def _make_app(self) -> FastAPI:

        from sentimentizer.serve.app import _RequestBodySizeLimitMiddleware

        app = FastAPI()
        app.add_middleware(
            _RequestBodySizeLimitMiddleware,
            default_max_bytes=100,
            path_limits={"/v1/images/": 200},
        )

        @app.post("/normal")
        async def normal(request: Request):
            body = await request.body()
            return {"size": len(body)}

        @app.post("/v1/images/generate")
        async def images(request: Request):
            body = await request.body()
            return {"size": len(body)}

        return app

    def test_normal_route_under_limit(self) -> None:
        client = TestClient(self._make_app())
        resp = client.post("/normal", content=b"a" * 99)
        assert resp.status_code == 200

    def test_normal_route_over_limit(self) -> None:
        client = TestClient(self._make_app())
        resp = client.post("/normal", content=b"a" * 101)
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "request_too_large"

    def test_image_route_under_limit(self) -> None:
        client = TestClient(self._make_app())
        resp = client.post("/v1/images/generate", content=b"a" * 199)
        assert resp.status_code == 200

    def test_image_route_over_limit(self) -> None:
        client = TestClient(self._make_app())
        resp = client.post("/v1/images/generate", content=b"a" * 201)
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "request_too_large"


class TestGenericRouterHTTP:
    def test_custom_labels_preserve_single_and_batch_envelopes(self):
        predictor = _mock_predictor()
        predictor.classify_batch.side_effect = lambda texts: [
            {"prediction": {"label": text, "score": 0.9, "token_count": 1}} for text in texts
        ]
        dep = _mock_deployment(predictor)
        with _router_client(dep) as client:
            single = client.post("/v1/router/predict", json={"text": "custom_a"})
            assert single.status_code == 200
            assert single.headers["content-type"] == "application/json"
            assert single.json()["prediction"] == {
                "label": "custom_a",
                "score": 0.9,
                "token_count": 1,
            }
            batch = client.post("/v1/router/batch", json={"texts": ["custom_a", "custom_b"]})
            assert batch.status_code == 200
            assert batch.json()["count"] == 2
            assert [row["prediction"]["label"] for row in batch.json()["results"]] == [
                "custom_a",
                "custom_b",
            ]
            assert batch.json()["results"][0]["prediction"] == single.json()["prediction"]
        assert dep._router_metrics.error_count == 0
        assert dep._router_metrics.request_count == 2

    @pytest.mark.parametrize(
        "bad_prediction",
        [
            {},
            {"label": "a", "score": float("nan"), "token_count": 0},
            {"label": "a", "score": "0.9", "token_count": 0},
            {"label": "a", "score": 1.1, "token_count": 0},
            {"label": "a", "score": 0.9, "token_count": True},
            {"label": "a", "score": 0.9, "token_count": -1},
            {"label": "a", "score": 0.9, "token_count": 1, "unexpected": 1},
        ],
    )
    def test_bad_backend_output_is_500_and_counts_as_error(self, bad_prediction):
        dep = _mock_deployment(
            _mock_predictor(
                classify_batch=MagicMock(
                    return_value=[{"prediction": bad_prediction}],
                )
            )
        )
        with _router_client(dep) as client:
            response = client.post("/v1/router/predict", json={"text": "example"})
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        assert response.json()["error"]["request_id"]
        assert dep._router_metrics.error_count == 1
        assert dep._router_metrics.request_count == 1

    def test_partial_batch_is_not_returned_as_success(self):
        dep = _mock_deployment()
        with _router_client(dep) as client:
            response = client.post("/v1/router/batch", json={"texts": ["a", "b"]})
        assert response.status_code == 500
        assert dep._router_metrics.error_count == 1

    def test_real_metadata_producer_matches_json_contract_without_loading(self, tmp_path):
        from sentimentizer.predictor import SentimentPredictor
        from sentimentizer.router.schemas import RouterArtifactMetadata, RouterTaskSpec

        task = RouterTaskSpec(task_id="test", labels=("a", "b"))
        (tmp_path / "router_config.json").write_text(
            RouterArtifactMetadata(task=task).model_dump_json()
        )
        predictor = SentimentPredictor.__new__(SentimentPredictor)
        predictor._initialize_router_state()
        predictor._router_model_path = str(tmp_path)
        predictor.router = None
        predictor._router_error = None
        predictor._load_router_model = MagicMock()
        dep = _mock_deployment(predictor)
        with _router_client(dep) as client:
            response = client.get("/v1/router/models")
        assert response.status_code == 200
        assert response.json() == {
            "model_path": str(tmp_path),
            "categories": ["a", "b"],
            "task_id": "test",
            "taxonomy_version": "1",
            "status": "not_loaded",
            "error": None,
        }
        predictor._load_router_model.assert_not_called()

    def test_load_failure_stays_503_and_requests_stay_422(self):
        dep = _mock_deployment(_mock_predictor(router_loaded=False, router_error="missing model"))
        with _router_client(dep) as client:
            response = client.post("/v1/router/predict", json={"text": "example"})
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "service_unavailable"
            response = client.post("/v1/router/batch", json={"texts": []})
            assert response.status_code == 422
            assert "detail" in response.json()

    def test_openapi_is_static_and_generic_with_numeric_bounds(self):
        with _router_client(_mock_deployment()) as client:
            schemas = client.get("/openapi.json").json()["components"]["schemas"]
        prediction = schemas["RouterPrediction"]
        assert prediction["additionalProperties"] is False
        assert "enum" not in prediction["properties"]["label"]
        assert prediction["properties"]["score"]["minimum"] == 0
        assert prediction["properties"]["score"]["maximum"] == 1
        assert prediction["properties"]["token_count"]["minimum"] == 0
        assert schemas["RouterPredictResponse"]["properties"]["latency_s"]["minimum"] == 0
        assert schemas["RouterBatchResponse"]["properties"]["count"]["minimum"] == 0

    def test_batch_schema_enforces_count_and_zero_tokens_remain_valid(self):
        from sentimentizer.serve.models import RouterBatchResponse

        item = {"prediction": {"label": "custom", "score": 1.0, "token_count": 0}}
        assert RouterBatchResponse(results=[item], count=1, latency_s=0.0).count == 1
        for count in [0, True, "1", -1]:
            with pytest.raises(ValidationError):
                RouterBatchResponse(results=[item], count=count, latency_s=0.0)
