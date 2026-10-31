# Serving and REST API

This document describes how to deploy and interact with the unified Sentimentizer REST API.

## Ray Serve Deployment (Python)

> [!NOTE]
> Serving requires the `ray` extra. You can install it using:
> ```bash
> uv sync --extra ray
> # or if adding to a project:
> uv add "sentimentizer[ray]"
> ```

The `serve` command starts a Ray Serve application with FastAPI routing (featuring interactive Swagger docs at `/docs`). It loads the sentiment model (defaulting to the configuration in `service.yaml`) at startup. The intent router loads lazily on the first classification request; metadata requests do not trigger loading. Headless Krea 2 and Ideogram 4 generation can be enabled through a separate ComfyUI process. All public APIs share the Ray Serve port.

### Starting the Server

```bash
# Start with defaults from sentimentizer/serve/service.yaml
make serve

# Or start via CLI with custom options
sentimentizer serve --host 0.0.0.0 --port 8000
```

By default, the server binds to `0.0.0.0:8000`. Set `serve_host`,
`serve_port`, and `ray_object_store_memory_mb` in
`sentimentizer/serve/service.yaml`; command-line flags and their matching
environment variables override the YAML values.

### Optional embedding service

Install the embeddings extra and enable the deployment explicitly:

```bash
uv sync --extra ray --extra embeddings
SENTIMENTIZER_EMBEDDINGS_ENABLED=1 \
SENTIMENTIZER_BGE_M3_ENABLED=1 \
python -m sentimentizer.serve --embeddings
```

`POST /v1/embeddings` accepts up to 64 non-empty texts and returns BGE-M3
1024-dimensional dense vectors plus sorted learned-sparse token IDs and weights.
The legacy `POST /vectors` and binary `POST /vectors/batch` routes serve the
configured dense model for existing clients. Model loading is disabled by
default because BGE-M3 is a large optional dependency.

For a BGE-M3-only production process, use:

```bash
python -m sentimentizer.serve --bge-m3-only
```

Its replica count, per-replica request concurrency, and Ray CPU/GPU reservations
come from the `bge_m3_num_replicas`, `bge_m3_max_ongoing_requests`,
`bge_m3_num_cpus`, and `bge_m3_num_gpus` YAML settings.

Both the full and BGE-M3-only applications expose `/health/live`,
`/health/ready`, `/health`, and `/metrics` on the serving port.

---

## API Endpoints Reference

> **Route namespacing.** Sentiment routes are canonically namespaced under
> `/v1/sentiment/…`. The older flat forms — `/v1/predict`, `/v1/batch`,
> `/v1/tokenize`, `/v1/models`, `/v1/models/{model_name}` — still work but are
> declared `deprecated=True` in the OpenAPI schema. Prefer the namespaced route
> in new clients.
>
> | Deprecated | Canonical |
> |---|---|
> | `POST /v1/predict` | `POST /v1/sentiment/predict` |
> | `POST /v1/batch` | `POST /v1/sentiment/batch` |
> | `POST /v1/tokenize` | `POST /v1/sentiment/tokenize` |
> | `GET /v1/models` | `GET /v1/sentiment/models` |
> | `GET /v1/models/{model_name}` | `GET /v1/sentiment/models/{model_name}` |
>
> Router (`/v1/router/…`), embeddings (`/v1/embeddings`, `/v1/embeddings/dense`),
> and images (`/v1/images/…`) are already namespaced and have no deprecated form.

### Sentiment Analysis Endpoints

#### Single Sentiment Prediction
- **Route**: `POST /v1/sentiment/predict` (deprecated alias: `POST /v1/predict`)
- **Request Body**:
  ```json
  {
    "text": "the food was terrific"
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/sentiment/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "the food was terrific"}'
  ```
- **Response**:
  ```json
  {
    "prediction": {
      "label": "positive",
      "score": 0.92,
      "token_count": 4,
      "model": "encoder"
    },
    "latency_s": 0.0043
  }
  ```

#### Batch Sentiment Prediction
- **Route**: `POST /v1/sentiment/batch` (deprecated alias: `POST /v1/batch`)
- **Request Body**:
  ```json
  {
    "texts": ["great pizza!", "terrible service"]
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/sentiment/batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["great pizza!", "terrible service"]}'
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "prediction": {
          "label": "positive",
          "score": 0.89,
          "token_count": 2,
          "model": "encoder"
        }
      },
      {
        "prediction": {
          "label": "negative",
          "score": 0.94,
          "token_count": 2,
          "model": "encoder"
        }
      }
    ],
    "count": 2,
    "latency_s": 0.0031
  }
  ```

#### Standalone Tokenization (No Inference)
- **Route**: `POST /v1/sentiment/tokenize` (deprecated alias: `POST /v1/tokenize`)
- **Request Body**:
  ```json
  {
    "text": "the food was terrific"
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/sentiment/tokenize \
    -H "Content-Type: application/json" \
    -d '{"text": "the food was terrific"}'
  ```
- **Response**:
  ```json
  {
    "text": "the food was terrific",
    "tokens": ["the", "food", "was", "terrific"],
    "token_ids": [4, 12, 10, 48],
    "token_count": 4
  }
  ```

#### List All Sentiment Models
- **Route**: `GET /v1/sentiment/models` (deprecated alias: `GET /v1/models`)
- **Command**:
  ```bash
  curl http://localhost:8000/v1/sentiment/models
  ```

#### Single Model Metadata
- **Route**: `GET /v1/sentiment/models/{model_name}` (deprecated alias: `GET /v1/models/{model_name}`)
- **Command**:
  ```bash
  curl http://localhost:8000/v1/sentiment/models/encoder
  ```

---

### Router (Review Categorization) Endpoints

The router uses a SentenceTransformer backbone and LogisticRegression head.
Its default categories are `dietary`, `service`, and `general`; custom artifacts
provide their own vocabulary. Each route
declares a Pydantic response model from `sentimentizer/serve/models.py`;
FastAPI serializes responses as JSON objects. Example scores and timings below
are illustrative.

| Field | Meaning |
| --- | --- |
| `prediction.label` | Predicted route category; route categories are separate from sentiment labels. |
| `prediction.score` | Finite probability assigned to the predicted category, between 0 and 1. |
| `prediction.token_count` | Number of tokens from `regex_tokenize(text)`, rather than backbone subword tokens; can be zero. |
| `latency_s` | Server-side handler duration in seconds, rounded to four decimal places; excludes client/network time. |
| `results` | Batch prediction envelopes in input order, with exactly one result per input. |
| `count` | Number of entries in `results`. |

Prediction objects contain `label`, `score`, and `token_count`. They omit the
input text, a `category` alias, the model name, and all-class probabilities.

Handlers return Pydantic objects, with strict nonnegative integer counts,
finite nonnegative latency, and no unexpected response fields. The shared
classification schema lives in `router/schemas.py`; HTTP envelopes live in
`serve/models.py`. Inference validates label membership against the loaded
artifact's vocabulary, maps probabilities through `head.classes_`, and rejects
malformed or partial batches. Validation occurs before success accounting.
OpenAPI keeps the label type generic; use `/v1/router/models` to discover the
loaded vocabulary. See [task configuration](router.md#reuse-with-external-configurations).

#### Classify Single Review

- **Route**: `POST /v1/router/predict`
- **Request Body**:
  ```json
  {
    "text": "They were so careful with my celiac needs"
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/router/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "They were so careful with my celiac needs"}'
  ```
- **Response** (`RouterPredictResponse`):
  ```json
  {
    "prediction": {
      "label": "dietary",
      "score": 0.95,
      "token_count": 8
    },
    "latency_s": 0.0031
  }
  ```

#### Classify Batch of Reviews

- **Route**: `POST /v1/router/batch`
- **Request Body**:
  ```json
  {
    "texts": ["Great gluten-free options!", "The waiter was rude", "Decent pizza"]
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/router/batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Great gluten-free options!", "The waiter was rude", "Decent pizza"]}'
  ```
- **Response** (`RouterBatchResponse`):
  ```json
  {
    "results": [
      {
        "prediction": {
          "label": "dietary",
          "score": 0.95,
          "token_count": 3
        }
      },
      {
        "prediction": {
          "label": "service",
          "score": 0.89,
          "token_count": 4
        }
      },
      {
        "prediction": {
          "label": "general",
          "score": 0.91,
          "token_count": 2
        }
      }
    ],
    "count": 3,
    "latency_s": 0.0052
  }
  ```

#### Router Model Metadata

- **Route**: `GET /v1/router/models`
- **Command**:
  ```bash
  curl http://localhost:8000/v1/router/models
  ```
- **Declared response contract** (`RouterModelsResponse`):
  ```json
  {
    "model_path": "models/router",
    "categories": ["dietary", "service", "general"],
    "status": "loaded",
    "error": null,
    "task_id": "food_reviews",
    "taxonomy_version": "1"
  }
  ```

`status` is currently produced as `loaded` or `not_loaded`. Before the first
classification, it is `not_loaded` with no load error. A failed load is cached:
status remains `not_loaded` and `error` contains the failure message. Metadata
reads the local sidecar without loading or downloading weights. The example
above describes a newly trained food artifact. Legacy artifacts may have null
task/version fields. Unknown custom artifacts report empty categories and null
task/version; the default food path can identify its food task before loading.
Missing optional inference dependencies do not prevent local metadata inspection.
The response model defaults `error` to `null` when the producer omits it.

#### Router Errors

Invalid request bodies retain Pydantic's HTTP 422 response shape. A failed
router load during classification returns HTTP 503 through the standard
`{"error": {"code": "service_unavailable", "message": "..."}}` envelope.
Unhandled server failures return HTTP 500 with `code: "internal_error"` and a
`request_id` inside `error`. Response validation failures are server errors;
they are separate from request validation. Missing custom model paths never
download the default food model. Correct the artifact or config and recreate
the predictor/service to retry a cached loading failure.

---

### Image Generation Endpoints

> [!NOTE]
> Torch Sentiment requires the `ray` and `diffusion` extras. ComfyUI runs in a
> separate environment, owns the CUDA GPU, and must be started first.
> ```bash
> uv sync --extra ray --extra diffusion
> ```

The Ray deployment reserves no GPU and serializes submissions to the ComfyUI
queue. Startup validates the native node types and checkpoint filenames before
exposing the image service. No ComfyUI custom nodes are required.

#### Headless ComfyUI setup

Use a dedicated checkout and virtual environment. The pinned revision below is
known to contain native ConvRot, Krea 2, and Ideogram 4 support:

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git ../ComfyUI
cd ../ComfyUI
git checkout feca51a
uv venv
uv pip install -r requirements.txt
```

Download the official checkpoints into their corresponding directories:

```bash
curl -L -o models/diffusion_models/krea2_turbo_int8_convrot.safetensors \
  https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_int8_convrot.safetensors
curl -L -o models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors \
  https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
curl -L -o models/vae/qwen_image_vae.safetensors \
  https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors

curl -L -o models/diffusion_models/ideogram4_int8_convrot.safetensors \
  https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_int8_convrot.safetensors
curl -L -o models/diffusion_models/ideogram4_unconditional_int8_convrot.safetensors \
  https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_int8_convrot.safetensors
curl -L -o models/text_encoders/qwen3vl_8b_fp8_scaled.safetensors \
  https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors
curl -L -o models/vae/flux2-vae.safetensors \
  https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors
```

Review the checkpoint licenses before downloading. Ideogram 4 is
non-commercial and requires an explicit operator acknowledgement. Krea 2 uses
the [Krea 2 Community License](https://www.krea.ai/krea-2-licensing); enabling
it requires explicit license acceptance and a fail-closed output moderation
endpoint. Start ComfyUI on loopback so its unauthenticated API is not publicly
exposed:

```bash
uv run python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --temp-directory /tmp/sentimentizer-comfyui
```

Then configure and start Torch Sentiment:

```bash
export SENTIMENTIZER_KREA_2_ENABLED=true
export SENTIMENTIZER_KREA_2_LICENSE_ACCEPTED=true
export SENTIMENTIZER_IDEOGRAM_4_ENABLED=true
export SENTIMENTIZER_IDEOGRAM_4_LICENSE_ACCEPTED=true
export SENTIMENTIZER_IMAGE_MODERATION_URL=http://127.0.0.1:8090/v1/moderate-image
export SENTIMENTIZER_IMAGE_MODERATION_API_KEY=replace-me
export SENTIMENTIZER_COMFYUI_BASE_URL=http://127.0.0.1:8188
export SENTIMENTIZER_COMFYUI_TEMP_DIRECTORY=/tmp/sentimentizer-comfyui/temp
export SENTIMENTIZER_API_KEYS=test-key-123
python -m sentimentizer.serve
```

The moderation endpoint receives JSON containing `image_b64`, `mime_type`,
`model`, and `user`. It must explicitly return `{"safe": true}` before Torch
Sentiment releases an image. It may reject output with `{"safe": false,
"code": "...", "message": "..."}`. Timeouts, connection failures, and malformed
responses fail closed. ComfyUI workflows use `PreviewImage`; Torch Sentiment
reads and then deletes every result from `SENTIMENTIZER_COMFYUI_TEMP_DIRECTORY`.
That path must be the `temp` child beneath ComfyUI's `--temp-directory` value
and must be mounted into both processes if they run in separate containers.

#### Synchronous Image Generation
- **Route**: `POST /v1/images/generate`
- **Auth**: `Authorization: Bearer <api_key>` (required)
- **Request Body**:
  ```json
  {
    "prompt": "a calico cat in a teacup, soft window light",
    "model": "krea_2",
    "width": 1024,
    "height": 1024,
    "output_format": "png"
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/images/generate \
    -H "Authorization: Bearer test-key-123" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "a calico cat in a teacup", "model": "krea_2"}'
  ```
- **Response**:
  ```json
  {
    "id": "img_ABCDEFGHIJKL",
    "created": 1700000000,
    "model": "krea_2",
    "image_b64": "...",
    "format": "png",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "steps": 8,
    "guidance_scale": 1.0,
    "latency_s": 4.8
  }
  ```

Supported parameters are `prompt`, `model` (`"krea_2"` or `"ideogram_4"`),
`steps`, `guidance_scale`, `width`, `height`, `seed`, `output_format`, `user`,
and the `Idempotency-Key` header. Output is returned as `b64_json`. The current
native workflows reject `negative_prompt`, `reference_images`, and URL output
rather than silently ignoring them.

Rate-limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) are
returned on every response. Exceeding the limit returns `429` with a `Retry-After` header.

#### List Image Models
- **Route**: `GET /v1/images/models`
- **Auth**: Required
- **Command**:
  ```bash
  curl http://localhost:8000/v1/images/models \
    -H "Authorization: Bearer test-key-123"
  ```

#### Single Image Model Metadata
- **Route**: `GET /v1/images/models/{name}`
- **Auth**: Required
- **Command**:
  ```bash
  curl http://localhost:8000/v1/images/models/krea_2 \
    -H "Authorization: Bearer test-key-123"
  ```

---

### Shared & Infrastructure Endpoints

#### Liveness Probe
- **Route**: `GET /health/live`
- **Purpose**: Returns `200 OK` (e.g. `{"status": "live", "uptime_s": 12.3}`) to confirm that the server process is running.
- **Command**:
  ```bash
  curl http://localhost:8000/health/live
  ```

#### Readiness Probe
- **Route**: `GET /health/ready`
- **Purpose**: Returns `200 OK` if the models are successfully loaded and ready to serve traffic, otherwise returns `503 Service Unavailable`.
- **Command**:
  ```bash
  curl http://localhost:8000/health/ready
  ```

#### Backward-Compatible Health Check
- **Route**: `GET /health`
- **Purpose**: Delegates to the readiness probe.
- **Command**:
  ```bash
  curl http://localhost:8000/health
  ```

#### Prometheus Metrics
- **Route**: `GET /metrics`
- **Purpose**: Returns Prometheus text exposition for readiness, uptime,
  request totals, error totals, and cumulative inference latency.
- **Command**:
  ```bash
  curl http://localhost:8000/metrics
  ```

Metrics are maintained per Ray Serve replica. The default deployments use one
replica; deployments scaled above one replica should use Ray's native aggregate
metrics rather than treating a single `/metrics` scrape as a cluster total.
Keep this operational endpoint loopback-only or restrict it at the reverse
proxy when the serving port is internet-facing.

#### Interactive API Docs (Swagger UI)
- Open `http://localhost:8000/docs` in your browser to view and interact with the REST API documentation.

---

## Ray Serve Internals

This section documents non-obvious Ray Serve behaviors that affect development and operations. For the full architectural detail, see `docs/serve-api-plan.md` under "Ray Serve Architecture & Pitfalls".

### `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`

**Must be set before any Ray import.** Without it, Ray workers try to create isolated venvs via `uv` that lack the `ray` package, causing `ModuleNotFoundError: No module named 'ray'`. Set in three places:

1. `sentimentizer/serve/app.py` — module-level, before `from ray import serve`
2. `sentimentizer/serve/app.py` — `main()` function, before `ray.init()`
3. `workflows/cli.py` — `serve_cmd()`, before importing `sentimentizer.serve`
4. `workflows/lifecycle.py` — module-level (covers training paths)

If you add a new entry point that uses Ray, add this env var.

### `@serve.batch` — callers get a single dict, not a list

`@serve.batch` collects multiple calls into a batch. The batch function signature is `list[dict] → list[dict]`, but **individual callers** receive only their single item from the result list. Never index into the return (e.g., `result[0]`) — it's already unwrapped.

```python
# Handler: calls batch function with single item
prediction = await self.predict_sentiment({"text": body.text})
# prediction is a dict, NOT a list[dict]

# Batch function: processes a batch
@serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
async def predict_sentiment(self, inputs: list[dict]) -> list[dict]:
    # Must return same length as inputs
    ...
```

**If the return list length doesn't match inputs length, remaining callers hang forever.**

### `@serve.ingress` — no `APIRouter`, no `__call__`

- All routes must be registered on the root `FastAPI` app with explicit `/v1/...` prefixes. `APIRouter` sub-apps don't work with `@serve.ingress`.
- Defining `__call__` on the deployment class raises `RuntimeError`.

### Model loading and readiness

`SentimentizerDeployment.__init__()` loads models synchronously. If loading fails, `self._ready` stays `False` and `/health/ready` returns 503. When image generation is enabled, readiness also performs a live ComfyUI sidecar check and returns 503 if it is unavailable. The process stays alive — it doesn't crash. Check `/health/ready` to confirm the configured backends are available.

### `asyncio.to_thread()` for sync inference

PyTorch model inference is blocking. All inference calls use `asyncio.to_thread()` to run in a thread pool, keeping the event loop responsive. Never call `predictor.predict_batch()` directly from an async handler.

### Deployment configuration

```python
@serve.deployment(
    num_replicas=1,          # Scale for CPU-bound load
    max_ongoing_requests=20,  # Backpressure: 21st request gets 503
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
```

For GPU inference, change `"num_gpus": 1` and set `device="cuda"`.

### `_DummyServe` fallback

`sentimentizer/serve/base.py` provides a `_DummyServe` that lets the serve module be imported without Ray installed. Always import `serve` from `base.py`:

```python
from sentimentizer.serve.base import serve  # grace fallback without ray
# NOT: from ray import serve  # ImportError without ray
```

### Startup sequence

1. `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` — prevents worker venv issues
2. `ray.init(namespace="sentimentizer", runtime_env={"py_executable": sys.executable})` — uses current Python
3. `serve.start(http_options=HTTPOptions(...))` — starts HTTP proxy
4. `serve.run(SentimentizerDeployment.bind(), ...)` — deploys the app
5. Model loading in `__init__()` — `/health/ready` returns 503 until models load

---

## Go CLI Client

A lightweight Go CLI client is included in the project root (`main.go`) to interact directly with the REST endpoints from the command line.

### Compilation and Usage

```bash
# Run with single text
go run main.go -text "the food was terrific"

# Positional arguments (defaults to single prediction)
go run main.go "best restaurant in town"

# Pipe input from stdin
echo "terrible service" | go run main.go

# Output raw JSON responses
go run main.go -raw -text "amazing pasta"

# Point to a custom serve endpoint
go run main.go -host http://remote-host:8000 -text "great coffee"
```

### Example Console Output

The Go client outputs colorized results with emojis indicating predicted sentiment classes:

```
Text:       the food was terrific
Prediction: positive 👍
Scores:     negative=0.03, neutral=0.05, positive=0.92
Latency:    12ms
```
