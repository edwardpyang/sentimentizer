# Configuration Reference

Sentimentizer is designed with modularity in mind. Configuration spans two layers:

- **Training-time model architecture** (this file, below) — Python dataclasses in `sentimentizer/config.py` that define model shape and training hyperparameters.
- **Runtime configuration** (serve / diffusion / router) — Python dataclass + YAML + env-var overrides in the corresponding subpackages. See the [Runtime Configuration](#runtime-configuration) section.

---

## Configuration Architecture

At the root of the configuration hierarchy is `DriverConfig`, which acts as a container for all other configuration classes:

- **FileConfig**: Input dataset and output weights file paths.
- **EmbeddingsConfig**: Configuration for downloading and mapping pre-trained word embeddings.
- **TokenizerConfig**: Parameters governing vocabulary construction, sequence padding length, and rating mapping.
- **TrainerConfig**: Hyperparameters for training loops, loss functions, learning rate schedules, and optimizations.
- **RNNConfig**, **EncoderConfig**, **DecoderConfig**, **ModernBERTConfig**: Architecture-specific parameters for each model type.
- **HuggingFaceConfig**: Hugging Face Hub metadata.

---

## Model Config Parameters

### RNN Architecture (LSTM)
Defined by `RNNConfig`. Uses a bidirectional LSTM model.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `hidden_size` | integer | `256` | Number of features in the LSTM hidden state. |
| `num_layers` | integer | `2` | Number of stacked LSTM layers. |
| `dropout` | float | `0.2` | Dropout probability applied between LSTM layers. |
| `num_classes` | integer | `3` | Classification output shape (3 = negative/neutral/positive). |

### Transformer Encoder Architecture
Defined by `EncoderConfig`. A standard Transformer Encoder classifier.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `d_model` | integer | `256` | Dimensionality of the input projected embeddings and attention layers. |
| `n_heads` | integer | `4` | Number of multi-head attention heads. |
| `n_layers` | integer | `4` | Number of Transformer Encoder layers stacked. |
| `dropout` | float | `0.2` | Dropout probability applied to attention and feedforward layers. |
| `ff_multiplier` | integer | `4` | feedforward dimension expansion factor (e.g. `d_model * ff_multiplier = 1024`). |
| `num_classes` | integer | `3` | Classification output shape. |

### Transformer Decoder Architecture
Defined by `DecoderConfig`. An Encoder-Decoder model structure.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `d_model` | integer | `256` | Dimensionality of the model layers. |
| `n_heads` | integer | `4` | Number of attention heads. |
| `n_encoder_layers` | integer | `2` | Number of Transformer Encoder layers. |
| `n_decoder_layers` | integer | `2` | Number of Transformer Decoder layers. |
| `dropout` | float | `0.3` | Dropout probability (higher to prevent overfitting on decoder cross-attention). |
| `ff_multiplier` | integer | `4` | feedforward dimension expansion factor. |
| `num_classes` | integer | `3` | Classification output shape. |

### ModernBERT Architecture
Defined by `ModernBERTConfig`. Adapts the pre-trained `ModernBERT` transformer.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | string | `"answerdotai/ModernBERT-base"` | Backbone Hugging Face repo name. |
| `dropout` | float | `0.1` | Dropout probability applied in classifier head. |
| `num_classes` | integer | `3` | Classification output shape. |
| `max_seq_length` | integer | `512` | Maximum token sequence length for dynamic padding. |
| `freeze_backbone_epochs`| integer | `1` | Number of epochs to freeze the backbone (training only classifier head). |
| `gradient_checkpointing`| boolean | `True` | Saves ~60% VRAM during backpropagation at ~30% compute overhead. |

---

## Configuration Consistency

Sentimentizer validates configuration consistency automatically prior to initialization via the `validate_config_consistency` function:

- **3-Class Classification Check**: If `TokenizerConfig.include_neutral` is `True`, all models' `num_classes` MUST be exactly `3`.
- **Binary Classification Check**: If `TokenizerConfig.include_neutral` is `False`, all models' `num_classes` MUST be exactly `2`.

If an inconsistency is detected, a `ValueError` is raised, preventing training from starting with invalid shape mismatches.

---

## Runtime Configuration

Three runtime domains follow the same loading pattern: **dataclass defaults < YAML file < environment variables** (highest priority). Each subpackage owns its own `config.py` + YAML pair and exposes a single `load_*_config()` entry point that returns the populated dataclass(es). Type coercion happens at parse time; invalid env-var values raise `ValueError` with the offending env var name.

### Serve (HTTP + Ray Serve)

| | |
| :--- | :--- |
| **Module** | [`sentimentizer/serve/config.py`](../sentimentizer/serve/config.py) |
| **Dataclass** | `ServeConfig` |
| **YAML** | [`sentimentizer/serve/service.yaml`](../sentimentizer/serve/service.yaml) |
| **Loader** | `load_serve_config(path=None) -> ServeConfig` |
| **Env prefix** | `SENTIMENTIZER_*` |

Owns operational concerns: which image models to enable, the headless ComfyUI
connection, auth, rate limits, and CORS.

| Env var | Field | Type | Notes |
| :--- | :--- | :--- | :--- |
| `SENTIMENTIZER_DEFAULT_MODEL` | `default_model` | str | Sentiment model: `rnn`, `encoder`, `decoder`, `modernbert` |
| `SENTIMENTIZER_API_KEYS` | `api_keys` | comma-list | Required for `/v1/images/*` |
| `SENTIMENTIZER_CORS_ORIGINS` | `cors_origins` | comma-list | |
| `SENTIMENTIZER_KREA_2_ENABLED` | `krea_2_enabled` | bool | Native INT8 ConvRot workflow |
| `SENTIMENTIZER_KREA_2_LICENSE_ACCEPTED` | `krea_2_license_accepted` | bool | Required when Krea 2 is enabled |
| `SENTIMENTIZER_IDEOGRAM_4_ENABLED` | `ideogram_4_enabled` | bool | Native INT8 ConvRot workflow |
| `SENTIMENTIZER_IDEOGRAM_4_LICENSE_ACCEPTED` | `ideogram_4_license_accepted` | bool | Required when Ideogram 4 is enabled |
| `SENTIMENTIZER_COMFYUI_BASE_URL` | `comfyui_base_url` | str | Headless ComfyUI URL |
| `SENTIMENTIZER_COMFYUI_TIMEOUT_S` | `comfyui_timeout_s` | float | Generation timeout |
| `SENTIMENTIZER_COMFYUI_POLL_INTERVAL_S` | `comfyui_poll_interval_s` | float | History polling interval |
| `SENTIMENTIZER_COMFYUI_TEMP_DIRECTORY` | `comfyui_temp_directory` | str | Required shared path for deleting fetched PreviewImage files |
| `SENTIMENTIZER_IMAGE_MODERATION_URL` | `image_moderation_url` | str | Required for Krea 2; fail-closed output moderation endpoint |
| `SENTIMENTIZER_IMAGE_MODERATION_API_KEY` | `image_moderation_api_key` | str | Optional moderation bearer credential |
| `SENTIMENTIZER_IMAGE_MODERATION_TIMEOUT_S` | `image_moderation_timeout_s` | float | Moderation timeout |
| `SENTIMENTIZER_DEFAULT_IMAGE_MODEL` | `default_image_model` | str | Used when request body omits `model` |
| `SENTIMENTIZER_RATE_LIMIT_PER_MIN` | `rate_limit_per_min` | int | Per API key |
| `SENTIMENTIZER_RATE_LIMIT_BURST` | `rate_limit_burst` | int | Token bucket burst |
| `SENTIMENTIZER_IDEMPOTENCY_TTL_S` | `idempotency_ttl_s` | int | |
| `SENTIMENTIZER_JOB_TTL_S` | `job_ttl_s` | int | |
| `SENTIMENTIZER_REQUEST_TIMEOUT_S` | `request_timeout_s` | int | HTTP timeout |

The moderation endpoint receives `image_b64`, `mime_type`, `model`, and `user`
as JSON. It must return a JSON object containing a boolean `safe`; any transport
failure or invalid response blocks release of the generated image.

`comfyui_temp_directory` must point to the `temp` child created under
ComfyUI's `--temp-directory` value. When the processes run in separate
containers, mount that directory at the configured path in both containers.

### Diffusion (model internals)

| | |
| :--- | :--- |
| **Module** | [`sentimentizer/diffusion/config.py`](../sentimentizer/diffusion/config.py) |
| **Dataclass** | `ImageModelConfig` (one instance per model) |
| **YAML** | [`sentimentizer/diffusion/diffusion_config.yaml`](../sentimentizer/diffusion/diffusion_config.yaml) |
| **Loader** | `load_diffusion_config(path=None) -> dict[str, ImageModelConfig]` |
| **Env prefix** | `SENTIMENTIZER_DIFFUSION_<MODEL>_*` |

Owns native ComfyUI checkpoint filenames, denoising defaults, max pixels,
dimension alignment, and the reported quantization format. The YAML has one
section each for `krea_2` and `ideogram_4`.

| Env var pattern | Field | Type |
| :--- | :--- | :--- |
| `SENTIMENTIZER_DIFFUSION_{KREA_2,IDEOGRAM_4}_DEFAULT_STEPS` | `default_steps` | int |
| `SENTIMENTIZER_DIFFUSION_{KREA_2,IDEOGRAM_4}_DEFAULT_GUIDANCE` | `default_guidance` | float |
| `SENTIMENTIZER_DIFFUSION_{KREA_2,IDEOGRAM_4}_MAX_PIXELS` | `max_pixels` | int |
| `SENTIMENTIZER_DIFFUSION_{KREA_2,IDEOGRAM_4}_{TRANSFORMER,TEXT_ENCODER,VAE}` | checkpoint fields | str |

### Router (training + augmentation)

| | |
| :--- | :--- |
| **Module** | [`sentimentizer/router/config.py`](../sentimentizer/router/config.py) |
| **Dataclasses** | `RouterConfig`, `AugmentConfig`, `RouteLabels` |
| **YAML** | [`sentimentizer/router/config.yaml`](../sentimentizer/router/config.yaml) |
| **Loader** | `load_router_config(path=None) -> tuple[RouterConfig, AugmentConfig]` |
| **Env prefix** | `SENTIMENTIZER_ROUTER_*` (training), `SENTIMENTIZER_AUGMENT_*` (augmentation) |

| Env var | Field | Section | Type |
| :--- | :--- | :--- | :--- |
| `SENTIMENTIZER_ROUTER_BASE_MODEL` | `base_model` | training | str |
| `SENTIMENTIZER_ROUTER_NUM_ITERATIONS` | `num_iterations` | training | int |
| `SENTIMENTIZER_ROUTER_NUM_EPOCHS` | `num_epochs` | training | int |
| `SENTIMENTIZER_ROUTER_BATCH_SIZE` | `batch_size` | training | int |
| `SENTIMENTIZER_ROUTER_MAX_SEQ_LENGTH` | `max_seq_length` | training | int |
| `SENTIMENTIZER_ROUTER_SEED` | `seed` | training | int |
| `SENTIMENTIZER_ROUTER_OUTPUT_DIR` | `output_dir` | training | str (path) |
| `SENTIMENTIZER_AUGMENT_MODEL` | `model` | augmentation | str |
| `SENTIMENTIZER_AUGMENT_OLLAMA_URL` | `ollama_url` | augmentation | str |
| `SENTIMENTIZER_AUGMENT_VARIATIONS_PER_SEED` | `variations_per_seed` | augmentation | int |
| `SENTIMENTIZER_AUGMENT_OUTPUT_PATH` | `output_path` | augmentation | str |
| `SENTIMENTIZER_AUGMENT_BATCH_SIZE` | `batch_size` | augmentation | int |
| `SENTIMENTIZER_AUGMENT_SEED_PATH` | `seed_path` | augmentation | str (path) |

`RouterConfig.task` is a `RouterTaskSpec` loaded from `training.task` in YAML.
It records `task_id`, `taxonomy_version` (default `"1"`), `description`, ordered
`labels`, optional `label_descriptions`, optional `fallback_label`, and
`input_format` (default `"text-v1"`). Labels must be unique, nonempty strings
with at least two classes; description and fallback references must belong to
the vocabulary. Older YAML without a task section retains the food defaults.
Training saves the task spec with the model; inference uses that artifact
metadata rather than the current training YAML.

`augmentation.seed_path` defaults to `""`, which selects bundled food seeds
only for the default task. A custom task requires an explicit JSONL seed file
for augmentation; relative paths resolve against the config file's directory.
See the [external config example](router.md#reuse-with-external-configurations).

### Loading order in callers

Module-level singletons evaluate the loader once at import time:

```python
# sentimentizer/serve/app.py
from sentimentizer.serve.config import load_serve_config
cfg = load_serve_config()
```

For image generation, model defaults are loaded once and the dispatcher selects
the requested workflow:

```python
# sentimentizer/serve/diffusion_app.py
from sentimentizer.diffusion.config import IMAGE_MODEL_CONFIGS

model_cfg = IMAGE_MODEL_CONFIGS[model_name]
```

This keeps checkpoint filenames and sampling defaults in YAML while operational
enablement and the ComfyUI URL remain in the serve configuration.
