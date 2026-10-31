# sentimentizer

[![PyPI Latest Release](https://img.shields.io/pypi/v/sentimentizer.svg)](https://pypi.org/project/sentimentizer/)
![GitHub CI](https://github.com/eddiepyang/sentimentizer/actions/workflows/ci.yaml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight PyTorch models for sentiment analysis. Small models can be pretty effective for classification tasks at a much smaller cost to deploy — all models were trained on a single GPU in minutes, and inference requires less than 1GB of memory.

> **Beta release** — API is subject to change.

---

## Install

```bash
# Install local-only version (no Ray dependency)
uv add sentimentizer

# Install with distributed training, tuning, and serving features
uv add "sentimentizer[ray]"

# Install the headless ComfyUI image API client
uv add "sentimentizer[diffusion]"

# Install all features
uv add "sentimentizer[ray,diffusion]"
```

---

## Quick Start
  
Run a pre-trained model locally:

```python
from sentimentizer.predictor import SentimentPredictor

# Load the model
predictor = SentimentPredictor(model_name="encoder")

# Predict sentiment (returns label, score, all-class scores, token count, and model type)
result = predictor.predict("amazing restaurant!")
# >> {"positive": 0.848753035068512,
#     "label": "positive",
#     "score": 0.848753035068512,
#     "scores": {"negative": 0.05746544152498245,
#                "neutral": 0.09378153085708618,
#                "positive": 0.848753035068512},
#     "token_count": 2,
#     "model": "encoder"}

# Batch prediction
results = predictor.predict_batch(["Great food!", "Terrible service."])
# >> [{'positive': 0.848753035068512,
#      'label': 'positive',
#      'score': 0.848753035068512,
#      'scores': {'negative': 0.05746544152498245,
#                 'neutral': 0.09378153085708618,
#                 'positive': 0.848753035068512},
#      'token_count': 2,
#      'model': 'encoder'},
#     {'negative': 0.9586858749389648,
#      'label': 'negative',
#      'score': 0.9586858749389648,
#      'scores': {'negative': 0.9586858749389648,
#                 'neutral': 0.03224344924092293,
#                 'positive': 0.0090706842020154},
#      'token_count': 2,
#      'model': 'encoder'}]
```

Models output **3-class probabilities** (negative, neutral, positive) that sum to 1.0 per sample.

---

## Intent Router API

The intent router classifies reviews as `dietary`, `service`, or `general`.
It uses a SentenceTransformer backbone and LogisticRegression head; the former
SetFit implementation has been replaced. Install serving and router support
with `uv add "sentimentizer[ray,router]"`, then start the API with
`sentimentizer serve`. The router loads on the first classification request.

| Method | Endpoint | Response model |
| --- | --- | --- |
| POST | `/v1/router/predict` | `RouterPredictResponse`: `prediction`, `latency_s` |
| POST | `/v1/router/batch` | `RouterBatchResponse`: `results`, `count`, `latency_s` |
| GET | `/v1/router/models` | `RouterModelsResponse`: `model_path`, `categories`, `status`, `error` |

For example, send `{"text": "They were so careful with my celiac needs"}` to
`/v1/router/predict`. The JSON response has this shape (scores and latency are
illustrative):

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

Batch requests accept `{"texts": ["Great service", "Delicious pizza"]}`;
each entry in `results` uses the same `{"prediction": {...}}` envelope.
`score` represents the predicted category's probability, `token_count` counts
regex tokens, and `latency_s` measures server-side handler time in seconds.
HTTP envelopes live in `sentimentizer/serve/models.py`; shared task and
classification schemas live in `sentimentizer/router/schemas.py`.

See the [complete JSON examples and validation rules](docs/serving.md#router-review-categorization-endpoints)
and the [router guide](docs/router.md#python-and-http-response-contract).
Responses validate finite scores in `[0, 1]`, nonnegative counts/timing, and
complete batches. Confidence columns follow the classifier's saved class order.

The router uses task configuration and artifact-defined labels while retaining
food-review defaults. To train another task, supply an external YAML config,
its labeled data, and optional augmentation seeds. Each trained artifact saves
its task definition; changing YAML alone does not retrain existing weights.
See [reuse with external configurations](docs/router.md#reuse-with-external-configurations).

---

## Image Generation (Krea 2 / Ideogram 4)

Image generation uses a separately managed, headless ComfyUI process. Torch
Sentiment submits native Krea 2 and Ideogram 4 workflows over HTTP; it does not
load image weights or reserve the CUDA device inside Ray. The supported
checkpoints are the official INT8 ConvRot builds.

### Prerequisites

```bash
# Torch Sentiment side
uv sync --extra ray --extra diffusion

# Separate ComfyUI checkout and environment. This revision has native ConvRot,
# Krea 2, and Ideogram 4 nodes; no custom nodes or browser UI are required.
git clone https://github.com/Comfy-Org/ComfyUI.git ../ComfyUI
cd ../ComfyUI
git checkout feca51a
uv venv
uv pip install -r requirements.txt

# Start API-only on localhost. Install the CUDA PyTorch build appropriate for
# the host before this step if requirements did not select it.
uv run python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --temp-directory /tmp/sentimentizer-comfyui
```

### Configure

Put these files under the ComfyUI checkout:

| Model | ComfyUI directory | Checkpoint |
| :--- | :--- | :--- |
| Krea 2 | `models/diffusion_models` | `krea2_turbo_int8_convrot.safetensors` |
| Krea 2 | `models/text_encoders` | `qwen3vl_4b_fp8_scaled.safetensors` |
| Krea 2 | `models/vae` | `qwen_image_vae.safetensors` |
| Ideogram 4 | `models/diffusion_models` | `ideogram4_int8_convrot.safetensors` |
| Ideogram 4 | `models/diffusion_models` | `ideogram4_unconditional_int8_convrot.safetensors` |
| Ideogram 4 | `models/text_encoders` | `qwen3vl_8b_fp8_scaled.safetensors` |
| Ideogram 4 | `models/vae` | `flux2-vae.safetensors` |

The official download URLs are listed in
[`docs/serving.md`](docs/serving.md#headless-comfyui-setup).

Configuration lives in two YAML files:

- **`sentimentizer/serve/service.yaml`** — enabled models, ComfyUI URL, auth, and rate limits.
- **`sentimentizer/diffusion/diffusion_config.yaml`** — checkpoint filenames and generation defaults.

Edit `service.yaml` to enable a model:

```yaml
# Enable one or both models
krea_2_enabled: true
# Required for Krea 2. Review the Community License and confirm eligibility.
krea_2_license_accepted: true
ideogram_4_enabled: true
# Required for Ideogram 4. Review its non-commercial model license first.
ideogram_4_license_accepted: true
# Required for Krea 2. The endpoint must explicitly approve generated output.
image_moderation_url: "http://127.0.0.1:8090/v1/moderate-image"
image_moderation_api_key: "replace-me"
default_image_model: "krea_2"
comfyui_base_url: "http://127.0.0.1:8188"
comfyui_temp_directory: "/tmp/sentimentizer-comfyui/temp"

# Auth — required for image routes (/v1/images/*)
api_keys: ["sk-your-secret-key"]

```

Or via environment variables:

```bash
export SENTIMENTIZER_KREA_2_ENABLED=true
export SENTIMENTIZER_KREA_2_LICENSE_ACCEPTED=true
export SENTIMENTIZER_IDEOGRAM_4_ENABLED=true
export SENTIMENTIZER_IDEOGRAM_4_LICENSE_ACCEPTED=true
export SENTIMENTIZER_IMAGE_MODERATION_URL=http://127.0.0.1:8090/v1/moderate-image
export SENTIMENTIZER_COMFYUI_BASE_URL=http://127.0.0.1:8188
export SENTIMENTIZER_COMFYUI_TEMP_DIRECTORY=/tmp/sentimentizer-comfyui/temp
export SENTIMENTIZER_API_KEYS=sk-your-secret-key
```

Krea 2 uses the [Krea 2 Community License](https://www.krea.ai/krea-2-licensing).
Its output-moderation endpoint receives JSON containing `image_b64`,
`mime_type`, `model`, and `user`, and must return `{"safe": true}` to release
the image. A rejection may return `{"safe": false, "code": "...", "message":
"..."}`. Missing, malformed, or unavailable moderation responses fail closed.

### Run

```bash
# Start Ray Serve; ComfyUI must already be running.
python -m sentimentizer.serve

# Krea 2 generation
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cinematic portrait of an astronaut", "model": "krea_2", "width": 1024, "height": 1024}'

# Ideogram 4 generation; structured JSON prompts give its strongest text/layout results.
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a bold poster reading FRESH", "model": "ideogram_4", "width": 1024, "height": 1024}'

# List available models
curl http://localhost:8000/v1/images/models \
  -H "Authorization: Bearer sk-your-secret-key"

# Async job mode (for long-running requests)
curl -X POST http://localhost:8000/v1/images/jobs \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cinematic portrait of an astronaut", "model": "krea_2"}'

# Poll job status
curl http://localhost:8000/v1/images/jobs/{job_id} \
  -H "Authorization: Bearer sk-your-secret-key"
```

### API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/images/generate` | Required | Synchronous image generation |
| POST | `/v1/images/jobs` | Required | Async job creation (201 + Location) |
| GET | `/v1/images/jobs` | Required | List jobs (paginated, scoped to API key) |
| GET | `/v1/images/jobs/{id}` | Required | Get job status |
| DELETE | `/v1/images/jobs/{id}` | Required | Cancel the targeted ComfyUI prompt |
| GET | `/v1/images/models` | Required | List available image models |
| GET | `/v1/images/models/{name}` | Required | Single model metadata |

---

## Models

Four architectures are available:

| Model | Module | Description |
| :--- | :--- | :--- |
| **ModernBERT** ⭐ | `sentimentizer.models.modernbert` | ModernBERT contextual transformer backbone with mean pooling and layer-wise unfreezing — **best performance** |
| **Encoder** | `sentimentizer.models.encoder` | Transformer encoder with CLS token + positional encoding (4 layers, `d_model=256`) |
| **RNN** | `sentimentizer.models.rnn` | Bidirectional 2-layer LSTM (`hidden=256`) with pre-trained GloVe embeddings — solid baseline |
| **Decoder** | `sentimentizer.models.decoder` | Encoder-Decoder Transformer with learnable query token + cross-attention (2 encoder + 4 decoder layers) |

All models output **3-class logits** `(B, 3)` mapped to: negative (0), neutral (1), positive (2).

---

## Documentation

Detailed guides and implementation details are available in the specialized documentation files:

- 🚀 **[Model Serving Guide](docs/serving.md)**: Ray Serve application deployment, FastAPI endpoints (sentiment/routing/image generation), and the Go CLI client.
- 🎨 **[Headless Image Serving](docs/serving.md#image-generation-endpoints)**: Krea 2 and Ideogram 4 setup, checkpoints, API, and licensing.
- 🏋️ **[Model Training & Checkpointing Guide](docs/training.md)**: Yelp datasets, single-node/distributed commands, training arguments, sleep prevention, and checkpoint resuming.
- ⚙️ **[Model Configuration Reference](docs/configuration.md)**: Configuration dataclasses (`RNNConfig`, `EncoderConfig`, etc.), parameter defaults, and consistency checks.
- 🎛️ **[Hyperparameter Tuning Guide](docs/tuning.md)**: Optuna searches, LangGraph iterative agent tuning (via Ollama GLM 5.1), and validation/retries.
- 🔗 **[Hugging Face Hub Integration](docs/huggingface.md)**: Pre-trained weights synchronization, explicit pull/push, and auto-generated model cards.
- 📈 **[Metrics and Monitoring Pipeline](docs/metrics.md)**: Exporter details, Grafana dashboards, Prometheus scrape targets, NaN handling, and real-time intra-epoch batch metrics.
- 🧭 **[Intent Review Router](docs/router.md)**: Dietary/service/general classification, Python and JSON response contracts, Ollama GLM 5.1 augmentation, training, and evaluation.
- 🛠️ **[Troubleshooting Guide](docs/troubleshooting.md)**: Solutions for common issues like majority-class collapses, vocabulary matches, or scheduling.

---

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (CPU-only PyTorch, no Ray)
uv sync

# Install with Ray distributed features
uv sync --extra ray

# Install dev and test suites
uv sync --extra dev --extra ray

# Install with diffusion (image generation) support
uv sync --extra diffusion

# Full development install
uv sync --extra dev --extra ray --extra diffusion
```

### Local CUDA / GPU development
The locked packages resolve CPU-only PyTorch. To install CUDA-enabled PyTorch locally:
```bash
uv sync --no-sources-package torch
```
*Note: This ignores CPU overrides in `pyproject.toml` and pulls PyTorch from PyPI with CUDA/NVIDIA libraries. Avoid committing changes to `uv.lock`.*

---

## Testing

Ensure local CI tests pass prior to submitting changes:

```bash
# Run all tests
uv run pytest tests/

# Run only Ray Train tests
uv run pytest tests/ -k "Ray"

# Run with coverage report
uv run pytest tests/ --cov=sentimentizer --cov-report=term-missing

# Verbose per-test output (or: make test-verbose)
uv run pytest tests/ -v
```

---

## Project Structure

```
sentimentizer/
├── __init__.py          # Logging and timing utilities
├── compat.py            # Transformers/setfit compatibility shims
├── config.py            # Configuration dataclasses and constants
├── data_source.py       # Unified DataSource protocol (pandas/Ray)
├── device.py            # Device detection (cuda/mps/cpu)
├── env.py               # Environment setup (NVIDIA LD_LIBRARY_PATH)
├── extractor.py          # Ray Data extraction from zip/tar archives
├── exporter.py           # Standalone Prometheus metrics exporter
├── export_onnx.py        # ONNX export, quantization, validation
├── hf.py                # Hugging Face Hub push/pull + model card generation
├── hf_dataset.py        # Dataset wrapper and collation for HF transformers
├── hf_tokenizer.py      # Tokenizer wrapper for HF transformers
├── loader.py             # Data loading utilities
├── losses.py             # FocalCrossEntropyLoss for 3-class training
├── metrics.py            # 3-class classification metrics (per-class P/R/F1, balanced accuracy, MCC)
├── metrics_publisher.py   # Epoch metrics publishing (Prometheus + JSON) + intra-epoch batch snapshots
├── predictor.py           # SentimentPredictor (model loading, inference)
├── safety.py              # Shared prompt safety (NSFW blocklist, injection patterns)
├── serve/                 # Ray Serve deployment: FastAPI + @serve.ingress, /v1/ prefix
│   ├── app.py             # FastAPI route handlers and deployment class
│   ├── base.py            # ServiceMetrics (request/latency tracking), _DummyServe fallback
│   ├── config.py           # Serve deployment configuration (YAML/env var loading, incl. cors_origins)
│   ├── middleware.py       # Auth, rate limiting, idempotency, prompt safety for image routes
│   ├── models.py          # Pydantic request/response models for Swagger docs
│   ├── diffusion_models.py # Pydantic request/response models for image generation (+ Job models)
│   └── diffusion_app.py    # Headless ComfyUI deployment, dispatcher, and job endpoints
├── diffusion/                # ComfyUI workflows and image job support
│   ├── comfyui.py            # HTTP client + Krea 2 / Ideogram 4 native workflows
│   ├── config.py             # ImageModelConfig + YAML/env loading
│   ├── diffusion_config.yaml # ConvRot checkpoint names and sampling defaults
│   └── job_store.py          # JobStoreLogic + Ray actor for async job metadata
├── tokenizer.py           # Text tokenizer with pre-trained support
├── trainer.py             # Training logic
├── tuner.py               # Ray Tune + Optuna hyperparameter search
├── data/                  # Training data (Yelp, GloVe)
├── agent/                 # LLM-guided tuning agent
│   ├── __init__.py       # Package exports
│   ├── config.yaml       # Agent + tuner configuration (YAML)
│   ├── loader.py         # YAML → dataclass config loader
│   ├── models.py         # Pydantic models (AnalysisResult, TuningDecision, etc.)
│   ├── agents.py         # Pydantic AI agents (GLM 5.1 via Ollama)
│   ├── prompts.py        # System prompts for analysis & strategy agents
│   ├── state.py          # LangGraph AgentState TypedDict
│   ├── nodes.py          # LangGraph node functions (analyze, decide, tune, evaluate)
│   ├── graph.py          # LangGraph StateGraph + run_agent_tuning() entry point
│   └── diagnose_model.py # TuningRun workflow (tune → train → validate → retry pipeline)
├── router/                # SentenceTransformer + LogisticRegression intent router
│   ├── __init__.py       # Package exports
│   ├── config.py         # RouterConfig (SetFitConfig alias), food defaults, AugmentConfig
│   ├── schemas.py        # RouterTaskSpec, RouterClassification, artifact metadata
│   ├── model.py          # RouterModel inference, save/load, legacy artifact migration
│   ├── seeds.py          # Golden example utterances per category
│   ├── augment.py        # GLM 5.1 augmentation via Ollama
│   ├── dataset.py        # JSONL dataset loader, train/test split
│   ├── train_router.py   # Contrastive backbone fine-tuning + classifier head fitting
│   └── evaluate.py       # Similarity heatmap, threshold calibration
└── models/
    ├── __init__.py
    ├── base.py            # BaseSentimentModel with predict() and predict_text()
    ├── hf_base.py         # Base class for Hugging Face transformer architectures
    ├── rnn.py            # Bidirectional LSTM (3-class output)
    ├── encoder.py         # Transformer encoder model (3-class output)
    ├── decoder.py         # Encoder-decoder transformer (3-class output)
    └── modernbert.py      # ModernBERT transformer classifier wrapper (3-class output)
```

---

## License

[MIT](LICENSE)
