"""Serve the Sentimentizer pipeline via Ray Serve.

Provides a unified REST API for:
  - Sentiment analysis (encoder model by default)
  - Review routing (Dietary, Service, General categories)

Uses ``serve.batch`` to auto-batch individual ``/v1/sentiment/predict`` and
``/v1/router/predict`` requests into efficient forward passes, and
``asyncio.to_thread()`` to keep sync model inference off the event loop.

Uses FastAPI for HTTP routing via ``@serve.ingress(app)`` with
decorator-based route registration.

Configuration is loaded from ``service.yaml`` with environment
variable overrides. See ``ServeConfig`` for all settings and their
corresponding env vars.

Usage:
    ray serve run sentimentizer.serve:app
    # or programmatically:
    python -m sentimentizer.serve

Endpoints:
  Sentiment analysis:
    POST /v1/sentiment/predict           -- Classify a single text (auto-batched)
    POST /v1/sentiment/batch             -- Classify multiple texts (single forward pass)
    POST /v1/sentiment/tokenize          -- Tokenize text without inference
    GET  /v1/sentiment/models            -- Sentiment model metadata (all models)
    GET  /v1/sentiment/models/{name}     -- Single model metadata

  Router (review categorization):
    POST /v1/router/predict   -- Route a single text (auto-batched)
    POST /v1/router/batch     -- Route multiple texts
    GET  /v1/router/models    -- Router model metadata

  Image Generation (if enabled):
    POST /v1/images/generate    -- Generate with Krea 2 or Ideogram 4
    POST /v1/images/jobs        -- Create an async image generation job
    GET  /v1/images/jobs        -- List image generation jobs
    GET  /v1/images/jobs/{id}   -- Get job status/result
    DELETE /v1/images/jobs/{id} -- Cancel a queued job
    GET  /v1/images/models      -- Image model metadata
    GET  /v1/images/models/{name} -- Single image model metadata

  Infrastructure (unversioned):
    GET  /health               -- Backward-compatible alias for /health/ready
    GET  /health/live          -- Liveness probe (always 200)
    GET  /health/ready         -- Readiness probe (503 if model not loaded)
    GET  /metrics              -- Prometheus service and request metrics

  Deprecated (kept for backward compatibility, use /v1/sentiment/* instead):
    POST /v1/predict
    POST /v1/batch
    POST /v1/tokenize
    GET  /v1/models
    GET  /v1/models/{name}
"""

import asyncio
import os
import struct
import threading
import time
import uuid
from importlib.metadata import version as _pkg_version
from typing import Any

# Prevent Ray from creating isolated worker venvs via uv.
# Must be set before Ray imports occur — Ray workers that use uv
# to create a fresh venv will fail with ModuleNotFoundError.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from sentimentizer import logger
from sentimentizer.predictor import _MODEL_CONFIGS, SentimentPredictor
from sentimentizer.serve.base import (
    PROMETHEUS_CONTENT_TYPE,
    ServiceMetrics,
    render_service_metrics,
    serve,
)
from sentimentizer.serve.config import cfg
from sentimentizer.serve.embeddings_models import (
    DenseEmbeddingsRequest,
    DenseEmbeddingsResult,
    EmbeddingsRequest,
    EmbeddingsResult,
    VectorBatchRequest,
    VectorRequest,
)
from sentimentizer.serve.models import (
    BatchRequest,
    BatchResponse,
    HealthLiveResponse,
    ModelDetailResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    RouterBatchResponse,
    RouterBatchResultItem,
    RouterModelsResponse,
    RouterPredictResponse,
    TokenizeRequest,
    TokenizeResponse,
)

# ---------------------------------------------------------------------------
# Load configuration (YAML defaults < env var overrides)
# ---------------------------------------------------------------------------

if cfg.api_keys:
    os.environ.setdefault("SENTIMENTIZER_API_KEYS", ",".join(cfg.api_keys))

# ---------------------------------------------------------------------------
# FastAPI app (module-level for @serve.ingress)
# ---------------------------------------------------------------------------

_VERSION = _pkg_version("sentimentizer")


class _RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with body exceeding max_bytes limit.

    Defense-in-depth: the K8s ingress already enforces
    ``proxy-body-size: "1m"``, but this middleware catches requests
    that bypass the ingress (e.g., port-forward, node port).
    """

    def __init__(
        self,
        app: Any,
        default_max_bytes: int = 1 * 1024 * 1024,
        path_limits: dict[str, int] | None = None,
    ) -> None:
        super().__init__(app)
        self.default_max_bytes = default_max_bytes
        self.path_limits = path_limits or {}

    async def dispatch(
        self,
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if request.method in ("POST", "PUT", "PATCH"):
            max_bytes = self.default_max_bytes
            for prefix, limit in self.path_limits.items():
                if request.url.path.startswith(prefix):
                    max_bytes = limit
                    break

            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    length = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": {
                                "code": "bad_request",
                                "message": "Invalid Content-Length header",
                            }
                        },
                    )

                if length > max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "request_too_large",
                                "message": f"Request body exceeds {max_bytes} bytes",
                            }
                        },
                    )
            else:
                # No Content-Length (e.g. Transfer-Encoding: chunked).
                # Buffer and verify size before passing request downstream.
                original_receive = request._receive
                bytes_received = 0
                chunks = []
                is_too_large = False

                while True:
                    try:
                        message = await original_receive()
                    except Exception:
                        # Client disconnected or read error
                        break

                    if message["type"] == "http.request":
                        body = message.get("body", b"")
                        bytes_received += len(body)
                        chunks.append(body)
                        if bytes_received > max_bytes:
                            is_too_large = True
                            break
                        if not message.get("more_body", False):
                            break
                    elif message["type"] == "http.disconnect":
                        break
                    else:
                        break

                if is_too_large:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "request_too_large",
                                "message": f"Request body exceeds {max_bytes} bytes",
                            }
                        },
                    )

                # Reconstruct receive so downstream can read the cached body
                full_body = b"".join(chunks)
                received_all = False

                async def cached_receive() -> dict[str, Any]:
                    nonlocal received_all
                    if received_all:
                        return {"type": "http.disconnect"}
                    received_all = True
                    return {
                        "type": "http.request",
                        "body": full_body,
                        "more_body": False,
                    }

                request._receive = cached_receive

        return await call_next(request)


def create_fastapi_app(
    title: str,
    description: str,
    path_limits: dict[str, int] | None = None,
) -> FastAPI:
    """Factory to create a FastAPI app with standard middleware and exception handlers."""
    app = FastAPI(
        title=title,
        version=_VERSION,
        description=description,
    )

    app.add_middleware(
        _RequestBodySizeLimitMiddleware,
        default_max_bytes=1 * 1024 * 1024,
        path_limits=path_limits,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials="*" not in cfg.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
    )

    @app.middleware("http")
    async def request_id_middleware(
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        """Add X-Request-Id to every request/response for distributed tracing."""
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Centralized handler for all unhandled exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("unhandled error in request", request_id=request_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "request_id": request_id,
                }
            },
        )

    app.add_exception_handler(HTTPException, http_exception_handler)

    return app


def _status_code_to_error_code(status_code: int) -> str:
    """Map common HTTP status codes to machine-readable error codes."""
    mapping = {
        400: "bad_request",
        401: "invalid_api_key",
        403: "forbidden",
        404: "not_found",
        409: "idempotency_key_conflict",
        413: "request_too_large",
        422: "validation_error",
        429: "rate_limit_exceeded",
        503: "service_unavailable",
    }
    return mapping.get(status_code, f"error_{status_code}")


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Wrap HTTPException in the standard error envelope."""
    detail = exc.detail
    if isinstance(detail, str):
        error_code = _status_code_to_error_code(exc.status_code)
        content = {
            "error": {
                "code": error_code,
                "message": detail,
            }
        }
    else:
        content = (
            {"error": detail} if isinstance(detail, dict) else {"error": {"message": str(detail)}}
        )
    return JSONResponse(status_code=exc.status_code, content=content)


app = create_fastapi_app(
    title="Sentimentizer",
    description="Sentiment analysis and review routing API",
)


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


@serve.deployment(
    # With 1 replica and max_ongoing_requests=20, the only concurrency win
    # is the serve.batch window. Increase num_replicas for CPU-bound load.
    num_replicas=1,
    max_ongoing_requests=20,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
@serve.ingress(app)
class SentimentizerDeployment:
    """Serves sentiment analysis and review routing over HTTP.

    Uses ``serve.ingress`` with FastAPI for HTTP routing.
    ``serve.batch`` auto-batches individual ``/v1/sentiment/predict`` calls into
    efficient forward passes. Sync model inference runs via
    ``asyncio.to_thread()`` to avoid blocking the event loop.
    """

    def __init__(
        self,
        embeddings_handle: Any | None = None,
        image_model_names: list[str] | None = None,
    ) -> None:
        self._started_at = time.time()
        self._ready = False
        self._load_error: str | None = None
        self._sentiment_metrics = ServiceMetrics(prefix="sentimentizer")
        self._router_metrics = ServiceMetrics(prefix="router")
        self._embedding_metrics = ServiceMetrics(prefix="embedding")
        self.predictor: SentimentPredictor | None = None
        self._embeddings_handle = embeddings_handle
        self._image_model_names = image_model_names or []
        self._comfyui_health_client: Any | None = None
        if self._image_model_names:
            from sentimentizer.diffusion.comfyui import ComfyUIClient

            self._comfyui_health_client = ComfyUIClient(
                cfg.comfyui_base_url,
                timeout_s=min(cfg.comfyui_timeout_s, 10.0),
                poll_interval_s=cfg.comfyui_poll_interval_s,
            )

        try:
            self.predictor = SentimentPredictor(
                model_name=cfg.default_model,
                router_model_path=cfg.router_model_path,
            )
            self._ready = True
            logger.info(
                "Sentimentizer ready",
                model=self.predictor.model_name,
                model_loaded=self.predictor.model_loaded,
                router_loaded=self.predictor.router_loaded,
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Failed to load models on startup")
            raise

    def _require_ready(self) -> None:
        if not self._ready or self.predictor is None:
            raise HTTPException(
                status_code=503,
                detail=f"Service not ready: {self._load_error or 'models not loaded'}",
            )

    def _require_embeddings(self, *, bge_m3: bool = False) -> None:
        if self._embeddings_handle is None or not cfg.embeddings_enabled:
            raise HTTPException(status_code=503, detail="Embeddings are disabled")
        if bge_m3 and not cfg.bge_m3_enabled:
            raise HTTPException(status_code=503, detail="BGE-M3 embeddings are disabled")

    async def _observe_embedding(self, result: Awaitable[Any]) -> Any:
        """Await one embedding operation and record its latency and outcome."""
        started_at = time.perf_counter()
        try:
            value = await result
        except Exception:
            self._embedding_metrics.record_request(
                time.perf_counter() - started_at,
                error=True,
            )
            raise
        self._embedding_metrics.record_request(time.perf_counter() - started_at)
        return value

    def _validate_model(self, model_name: str | None) -> str:
        """Validate requested model matches loaded model.

        Returns the validated model name. Raises HTTPException(400)
        if the requested model doesn't match the loaded model.
        """
        if model_name is None:
            return self.predictor.model_name
        if model_name.lower().strip() != self.predictor.model_name:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model_name}' is not loaded. "
                    f"Available model: '{self.predictor.model_name}'"
                ),
            )
        return self.predictor.model_name

    @staticmethod
    def _format_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
        """Format a prediction dict for the API response.

        Extracts label, score, token_count, and model from the
        predictor's output dict.
        """
        return {
            "label": prediction.get("label", "unknown"),
            "score": prediction.get("score", 0.0),
            "model": prediction.get("model", "unknown"),
            "token_count": prediction.get("token_count", 0),
        }

    # ------------------------------------------------------------------
    # Auto-batched endpoints (serve.batch collects individual calls)
    #
    # NOTE: @serve.batch transforms these methods so a single call returns
    # a single dict, not list[dict]. The type hint shows list[dict] because
    # that's the batch signature, but callers receive a single item.
    # ------------------------------------------------------------------

    @serve.batch(
        max_batch_size=cfg.predict_batch_size,
        batch_wait_timeout_s=cfg.predict_batch_wait_s,
    )
    async def predict_sentiment(self, inputs: list[dict]) -> list[dict]:
        """Auto-batched sentiment prediction.

        Collects up to ``predict_batch_size`` individual ``/v1/sentiment/predict``
        requests over a ``predict_batch_wait_s`` window, then
        runs them through a single forward pass.
        """
        texts = [inp["text"] for inp in inputs]
        predictions = await asyncio.to_thread(self.predictor.predict_batch, texts)
        return predictions

    @serve.batch(
        max_batch_size=cfg.classify_batch_size,
        batch_wait_timeout_s=cfg.classify_batch_wait_s,
    )
    async def classify_route(self, inputs: list[dict]) -> list[dict]:
        """Auto-batched route classification.

        Collects up to ``classify_batch_size`` individual
        ``/v1/router/predict`` requests, then classifies them in one batch.
        """
        texts = [inp["text"] for inp in inputs]
        results = await asyncio.to_thread(self.predictor.classify_batch, texts)
        return results

    # ------------------------------------------------------------------
    # Sentiment handlers (v1/sentiment prefix)
    # Old /v1/* paths are kept as deprecated aliases.
    # ------------------------------------------------------------------

    @app.post("/v1/sentiment/predict", response_model=PredictResponse)
    @app.post("/v1/predict", response_model=PredictResponse, deprecated=True)
    async def predict(self, body: PredictRequest, request: Request) -> dict[str, Any]:
        """POST /v1/sentiment/predict -- single text sentiment analysis (auto-batched)."""
        self._require_ready()
        if not self.predictor.model_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Sentiment model not loaded: {self.predictor.model_error}",
            )
        self._validate_model(body.model)

        start = time.perf_counter()
        try:
            prediction = await self.predict_sentiment({"text": body.text})
        except Exception:
            latency = time.perf_counter() - start
            self._sentiment_metrics.record_request(latency, error=True)
            raise
        latency = time.perf_counter() - start

        self._sentiment_metrics.record_request(latency)
        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "prediction completed",
            request_id=request_id,
            input_length=len(body.text),
            latency_s=f"{latency:.4f}",
            model=prediction.get("model", ""),
        )
        formatted = self._format_prediction(prediction)
        return {
            "prediction": formatted,
            "latency_s": round(latency, 4),
        }

    @app.post("/v1/sentiment/batch", response_model=BatchResponse)
    @app.post("/v1/batch", response_model=BatchResponse, deprecated=True)
    async def batch(self, body: BatchRequest, request: Request) -> dict[str, Any]:
        """POST /v1/sentiment/batch -- batch sentiment analysis (explicit forward pass)."""
        self._require_ready()
        if not self.predictor.model_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Sentiment model not loaded: {self.predictor.model_error}",
            )
        self._validate_model(body.model)

        start = time.perf_counter()
        try:
            raw_predictions = await asyncio.to_thread(self.predictor.predict_batch, body.texts)
        except Exception:
            latency = time.perf_counter() - start
            self._sentiment_metrics.record_request(latency, error=True)
            raise
        latency = time.perf_counter() - start

        results = [
            {
                "prediction": self._format_prediction(pred),
            }
            for pred in raw_predictions
        ]
        self._sentiment_metrics.record_request(latency)
        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "batch prediction completed",
            request_id=request_id,
            batch_size=len(body.texts),
            latency_s=f"{latency:.4f}",
        )
        return {
            "results": results,
            "count": len(results),
            "latency_s": round(latency, 4),
        }

    @app.post("/v1/sentiment/tokenize", response_model=TokenizeResponse)
    @app.post("/v1/tokenize", response_model=TokenizeResponse, deprecated=True)
    async def tokenize(self, body: TokenizeRequest) -> dict[str, Any]:
        """POST /v1/sentiment/tokenize -- standalone tokenization without inference."""
        self._require_ready()
        result = await asyncio.to_thread(self.predictor.tokenize, body.text)
        return result

    @app.get("/v1/sentiment/models", response_model=ModelsResponse)
    @app.get("/v1/models", response_model=ModelsResponse, deprecated=True)
    async def models(self) -> dict[str, Any]:
        """GET /v1/sentiment/models -- sentiment model metadata (all models)."""
        self._require_ready()
        models_info = self.predictor.get_sentiment_model_info()
        return {"models": models_info, "default": self.predictor.model_name}

    @app.get("/v1/sentiment/models/{model_name}", response_model=ModelDetailResponse)
    @app.get("/v1/models/{model_name}", response_model=ModelDetailResponse, deprecated=True)
    async def model_detail(
        self,
        model_name: str = Path(..., description="Model name: rnn, encoder, or decoder"),
    ) -> dict[str, Any]:
        """GET /v1/sentiment/models/{model_name} -- single model metadata."""
        self._require_ready()
        if model_name not in _MODEL_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_name}'. Available: {list(_MODEL_CONFIGS.keys())}",
            )
        info = self.predictor.get_sentiment_model_info()
        model_info = info.get(model_name)
        if model_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in model info.",
            )
        return {"model": model_name, "info": model_info}

    # ------------------------------------------------------------------
    # Embedding handlers
    # ------------------------------------------------------------------

    @app.post("/vectors")
    async def vector(self, body: VectorRequest) -> dict[str, list[float]]:
        """Return one legacy dense vector."""
        self._require_embeddings()
        vectors = await self._observe_embedding(
            self._embeddings_handle.dense.remote([body.text], body.mode)
        )
        return {"vector": vectors[0]}

    @app.post("/vectors/batch", response_class=Response)
    async def vectors_batch(self, body: VectorBatchRequest) -> Response:
        """Return dense vectors using the legacy little-endian binary format."""
        self._require_embeddings()
        vectors = await self._observe_embedding(
            self._embeddings_handle.dense.remote(body.texts, body.mode)
        )
        rows = len(vectors)
        dim = len(vectors[0]) if vectors else 0
        payload = bytearray(struct.pack("<II", rows, dim))
        for vector in vectors:
            if len(vector) != dim:
                raise HTTPException(status_code=500, detail="Embedding dimensions differ")
            payload.extend(struct.pack(f"<{dim}f", *vector))
        return Response(content=bytes(payload), media_type="application/octet-stream")

    @app.post("/v1/embeddings", response_model=EmbeddingsResult)
    async def embeddings(self, body: EmbeddingsRequest) -> dict[str, Any]:
        """Return BGE-M3 dense and learned-sparse vectors in input order."""
        self._require_embeddings(bge_m3=True)
        vectors = await self._observe_embedding(
            asyncio.gather(*(self._embeddings_handle.bge_m3.remote(text) for text in body.texts))
        )
        return {
            "backend": "flag_embedding",
            "model": cfg.bge_m3_model_id,
            "vectors": vectors,
        }

    @app.post("/v1/embeddings/dense", response_model=DenseEmbeddingsResult)
    async def dense_embeddings(self, body: DenseEmbeddingsRequest) -> dict[str, Any]:
        """Return nomic dense vectors in input order."""
        self._require_embeddings()
        vectors = await self._observe_embedding(
            self._embeddings_handle.dense.remote(body.texts, body.mode)
        )
        return {"model": cfg.dense_embedding_model_id, "vectors": vectors}

    # ------------------------------------------------------------------
    # Router handlers (v1 prefix)
    # ------------------------------------------------------------------

    @app.post("/v1/router/predict", response_model=RouterPredictResponse)
    async def router_predict(self, body: PredictRequest, request: Request) -> RouterPredictResponse:
        """POST /v1/router/predict -- classify a single text (auto-batched)."""
        self._require_ready()

        start = time.perf_counter()
        try:
            raw_result = await self.classify_route({"text": body.text})
            item = RouterBatchResultItem.model_validate(raw_result)
            latency = time.perf_counter() - start
            response = RouterPredictResponse(
                prediction=item.prediction, latency_s=round(latency, 4)
            )
        except Exception:
            latency = time.perf_counter() - start
            self._router_metrics.record_request(latency, error=True)
            if not self.predictor.router_loaded:
                raise HTTPException(
                    status_code=503,
                    detail=f"Router model not loaded: {self.predictor.router_error}",
                ) from None
            raise
        self._router_metrics.record_request(latency)
        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "router prediction completed",
            request_id=request_id,
            input_length=len(body.text),
            latency_s=f"{latency:.4f}",
            category=response.prediction.label,
        )
        return response

    @app.post("/v1/router/batch", response_model=RouterBatchResponse)
    async def router_batch(self, body: BatchRequest, request: Request) -> RouterBatchResponse:
        """POST /v1/router/batch -- classify multiple texts into routes."""
        self._require_ready()

        start = time.perf_counter()
        try:
            results = await asyncio.to_thread(self.predictor.classify_batch, body.texts)
            if len(results) != len(body.texts):
                raise ValueError("Router batch results must match the number of input texts")
            latency = time.perf_counter() - start
            response = RouterBatchResponse(
                results=results, count=len(results), latency_s=round(latency, 4)
            )
        except Exception:
            latency = time.perf_counter() - start
            self._router_metrics.record_request(latency, error=True)
            if not self.predictor.router_loaded:
                raise HTTPException(
                    status_code=503,
                    detail=f"Router model not loaded: {self.predictor.router_error}",
                ) from None
            raise
        self._router_metrics.record_request(latency)
        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "router batch completed",
            request_id=request_id,
            batch_size=len(body.texts),
            latency_s=f"{latency:.4f}",
        )
        return response

    @app.get("/v1/router/models", response_model=RouterModelsResponse)
    async def router_models(self) -> RouterModelsResponse:
        """GET /v1/router/models -- router model metadata."""
        self._require_ready()
        return RouterModelsResponse.model_validate(self.predictor.get_router_model_info())

    # ------------------------------------------------------------------
    # Infrastructure handlers (unversioned)
    # ------------------------------------------------------------------

    @app.get("/metrics", response_class=Response)
    async def metrics(self) -> Response:
        """GET /metrics -- return per-replica Prometheus metrics."""
        predictor_ready = bool(
            self._ready and self.predictor is not None and self.predictor.model_loaded
        )
        body = render_service_metrics(
            (
                self._sentiment_metrics,
                self._router_metrics,
                self._embedding_metrics,
            ),
            service_prefix="sentimentizer_service",
            ready=predictor_ready,
            uptime_s=time.time() - self._started_at,
        )
        return Response(
            content=body,
            headers={"Content-Type": PROMETHEUS_CONTENT_TYPE},
        )

    @app.get("/health/live", response_model=HealthLiveResponse)
    async def health_live(self) -> dict[str, Any]:
        """GET /health/live -- liveness probe (always returns 200)."""
        return {
            "status": "live",
            "uptime_s": round(time.time() - self._started_at, 1),
        }

    @app.get("/health/ready", response_model=None)
    async def health_ready(self) -> dict[str, Any] | JSONResponse:
        """GET /health/ready -- readiness probe (503 if model not loaded).

        Note: response_model=None disables Pydantic response validation.
        The 503 path returns a raw JSONResponse that bypasses schema anyway,
        so explicit response_model would give a false guarantee.
        """
        if not self._ready or self.predictor is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "uptime_s": round(time.time() - self._started_at, 1),
                    "error": self._load_error or "models not loaded",
                },
            )
        body = {
            "status": "ready" if self.predictor.model_loaded else "not_ready",
            "device": self.predictor.device,
            "version": self.predictor.version,
            "uptime_s": round(time.time() - self._started_at, 1),
            "model_loaded": self.predictor.model_name,
            "router_loaded": self.predictor.router_loaded,
            "router_error": self.predictor.router_error,
        }
        image_backend_ready = True
        if self._image_model_names:
            try:
                await asyncio.to_thread(self._comfyui_health_client.system_stats)
                body["image_models"] = {name: "ready" for name in self._image_model_names}
            except Exception:
                image_backend_ready = False
                body["status"] = "not_ready"
                body["image_models"] = {name: "unavailable" for name in self._image_model_names}
                body["image_backend_error"] = "ComfyUI sidecar unavailable"
        body["embeddings_enabled"] = cfg.embeddings_enabled
        body["bge_m3_enabled"] = cfg.bge_m3_enabled
        if not self.predictor.model_loaded or not image_backend_ready:
            return JSONResponse(status_code=503, content=body)
        return body

    @app.get("/health", response_model=None)
    async def health(self) -> dict[str, Any] | JSONResponse:
        """GET /health -- backward-compatible alias for /health/ready."""
        return await self.health_ready()


def main(host: str | None = None, port: int | None = None, diffusion: bool = False) -> None:
    """Start the Sentimentizer Serve application programmatically.

    Args:
        host: Bind address. Defaults to ``serve_host`` from ``service.yaml``.
        port: Bind port. Defaults to ``serve_port`` from ``service.yaml``.
        diffusion: If True, start the image generation (diffusion) deployment
            alongside sentiment. This requires GPU hardware and model weights.
            When False (default), only sentiment + router endpoints are served.
            ComfyUI runs as a separate headless process and owns the GPU.
            Enable Krea 2 or Ideogram 4 in ``service.yaml`` or with env vars.

    Workers use the current Python executable directly (no isolated venv),
    which avoids the ``ModuleNotFoundError: No module named 'ray'`` issue
    where Ray's worker spawn creates a fresh venv that lacks ``ray``.
    """
    import sys

    from ray.serve.config import HTTPOptions

    host = cfg.serve_host if host is None else host
    port = cfg.serve_port if port is None else port

    # RAY_ENABLE_UV_RUN_RUNTIME_ENV is already set at module level (line 51).
    os.environ.setdefault("RAY_ENABLE_RUNTIME_ENV_HOOK", "1")
    os.environ.setdefault("RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES", "")

    import ray

    ray.init(
        ignore_reinit_error=True,
        namespace="sentimentizer",
        runtime_env={
            "py_executable": sys.executable,
        },
    )
    serve.start(
        http_options=HTTPOptions(
            host=host,
            port=port,
            request_timeout_s=cfg.request_timeout_s,
        )
    )

    start_diffusion = diffusion or cfg.krea_2_enabled or cfg.ideogram_4_enabled
    model_names: list[str] = []
    if cfg.krea_2_enabled:
        model_names.append("krea_2")
    if cfg.ideogram_4_enabled:
        model_names.append("ideogram_4")
    if start_diffusion and not model_names:
        raise ValueError(
            "--diffusion requires an explicitly enabled image model. Set "
            "SENTIMENTIZER_KREA_2_ENABLED=true or SENTIMENTIZER_IDEOGRAM_4_ENABLED=true "
            "and satisfy that model's license requirements."
        )

    embeddings_handle = None
    if cfg.embeddings_enabled:
        from sentimentizer.serve.embeddings_app import EmbeddingsDeployment

        embeddings_handle = EmbeddingsDeployment.bind()

    serve.run(
        SentimentizerDeployment.bind(embeddings_handle, model_names),
        name="sentimentizer",
        route_prefix="/",
    )

    if start_diffusion:
        from sentimentizer.diffusion.job_store import JobStore
        from sentimentizer.serve.diffusion_app import ComfyUIDeployment, ImagesDispatcher

        JobStore.options(
            name="diffusion_job_store",
            lifetime="detached",
            get_if_exists=True,
        ).remote(ttl_s=cfg.job_ttl_s)

        comfyui_handle = ComfyUIDeployment.bind(model_names)

        serve.run(
            ImagesDispatcher.bind(comfyui_handle, model_names),
            name="images",
            route_prefix="/v1/images",
        )

    import signal

    shutdown_event = threading.Event()

    def handle_signal(signum: int, frame: Any) -> None:
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"signal {signum}"
        logger.info("received_shutdown_signal", signal=signame)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Sentimentizer Serve running", host=host, port=port)

    shutdown_event.wait()
    logger.info("Shutting down Serve")
    serve.shutdown()
    ray.shutdown()
