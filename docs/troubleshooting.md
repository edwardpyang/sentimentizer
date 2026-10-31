# Troubleshooting

## 3-Class Migration

### Binary → 3-class changes

The pipeline was migrated from binary (negative/positive, `BCEWithLogitsLoss`) to 3-class (negative/neutral/positive, `CrossEntropyLoss`). Key changes:

| Aspect | Binary (old) | 3-class (current) |
|--------|-------------|-------------------|
| **Classes** | 2 (negative, positive) | 3 (negative, neutral, positive) |
| **Loss** | `BCEWithLogitsLoss` with `pos_weight` | `CrossEntropyLoss` with `class_weights` and `label_smoothing=0.1` |
| **Target dtype** | `torch.float32` | `torch.long` |
| **Model output** | `(B, 1)` logits | `(B, 3)` logits |
| **Final activation** | `sigmoid` | `softmax` |
| **Rating mapping** | 1-2★ → 0, 4-5★ → 1 | 1-2★ → 0, 3★ → 1, 4-5★ → 2 |
| **`predict_text()` return** | `float` (0-1 score) | `dict[str, float]` (e.g., `{"negative": 0.05, "neutral": 0.12, "positive": 0.83}`) |
| **Class balance** | `compute_pos_weight()` → single float | `compute_class_weights()` → 3-element tensor, `weight_smoothing` parameter |
| **NaN replacement** | `0.5` (binary random) | `1/num_classes` ≈ `0.333` (uniform over 3 classes) |
| **Configuration** | `NUM_CLASSES=2`, `pos_weight` | `NUM_CLASSES=3`, `LABEL_NAMES`, `class_weights`, `loss_type`, `focal_gamma`, `label_smoothing`, `weight_smoothing`, `neutral_oversample_ratio` |

### Common issues after migration

#### Old binary weights crash on load

**Symptoms**: `RuntimeError` about tensor shape mismatch (expecting `(3, hidden_dim)` but got `(1, hidden_dim)`).

**Fix**: Old binary weights are incompatible with the 3-class architecture. Delete them and retrain:

```bash
rm models/rnn_sentiment.model
rm models/encoder_sentiment.model
rm models/decoder_sentiment.model
make train MODEL=rnn
```

#### `compute_pos_weight()` raises `ValueError`

**Symptoms**: `ValueError: compute_pos_weight() is deprecated for 3-class classification. Use compute_class_weights() instead.`

**Fix**: Replace all `compute_pos_weight()` calls with `compute_class_weights()`. The old function raises an error for `num_classes > 2`.

#### Target dtype errors in training

**Symptoms**: `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long, but got Float`.

**Fix**: Targets must be `torch.long` for `CrossEntropyLoss`, not `torch.float32`. This applies to both DataLoader and Ray distributed paths. Search for any residual `.float()` casts on targets and replace with `.long()`.

#### `torch.squeeze` removing class dimension

**Symptoms**: Shape `(3,)` instead of `(1, 3)` for batch-of-1 predictions.

**Fix**: `forward()` must always return `(B, num_classes)`, never squeeze. `predict()` uses `torch.softmax(logits, dim=-1)` returning `(B, 3)`.

#### Neutral class has low recall (< 0.60)

**Symptoms**: `neutral_recall` below 0.60, `pred_neutral_frac` below 0.05.

**Fix (escalate through these)**:
1. Default: `CrossEntropyLoss` with `class_weights` (smoothed) + `label_smoothing=0.1`
2. Focal loss: `loss_type="focal"`, `focal_gamma=2.0`
3. Oversampling: `neutral_oversample_ratio=0.20`
4. Reduce `weight_smoothing` toward `0.0` for more aggressive class weighting

## Model Training Issues

### Zero neutral-class recall / Cohen's kappa = 0

**Symptoms**: After training, `neutral_recall` is 0, `cohen_kappa` is 0, and the model predicts the majority class (positive) for every input.

**Root cause**: The model is effectively not learning word meanings. Nearly all tokens are mapped to the out-of-vocabulary (OOV) embedding, so the model trains on random vectors.

**Check**: Look at the GloVe match rate in the logs. If it's below ~50%, the dictionary is corrupted:

```
matched 23/20000 dictionary words to glove-wiki-gigaword-100 vectors (0.1%)
```

A healthy match rate should be 50–80%.

**Common causes and fixes**:

| Cause | Symptoms | Fix |
|-------|----------|-----|
| **Dictionary tokens have wrapping quotes** | `extract_embeddings()` raises `ValueError` for match rate < 50%; dictionary contains `"'the'"` instead of `"the"` | Since the fix, `new_dictionary()` and `_count_vocab_batch()` use `list(doc_tokens)` instead of `str(doc_tokens)` for numpy arrays. If you have a stale `.dictionary` file, delete it and re-run the pipeline with `--run-type new`. |
| **Wrong scheduler configuration** | Loss decays to minimum partway through training; model stops learning before the final epoch | All models use per-batch scheduler stepping with `_LinearWarmupCosineScheduler` and `warmup_ratio=0.06`. The scheduler is rebuilt with real optimizer-step counts at training time. `T_max` and `warmup_epochs` on `SchedulerParams` are legacy fields — they don't control the schedule. See "Learning rate drops to minimum" below. |
| **Zero LR during warmup** | First epoch shows no learning progress | `_LinearWarmupCosineScheduler` uses `(step + 1) / warmup_steps` instead of `step / warmup_steps`. Verify the warmup formula is correct. |
| **Class imbalance not addressed** | Model always predicts positive; `balanced_accuracy` near 0.33 (random) | Use `class_weights` (default with `compute_class_weights()`), `label_smoothing=0.1`, or `loss_type="focal"` with `focal_gamma=2.0`. |

### Dictionary has wrapping quotes (`'the'` instead of `the`)

This occurs when parquet stores the `tokens` column as numpy arrays. The code `isinstance(doc_tokens, list)` returns `False` for numpy arrays, and the fallback `regex_tokenize(str(numpy_array))` stringifies the array representation (`"['if' 'you' ...]"`), causing the regex `[a-z0-9'-]+` to capture the quotes as part of each token.

**Affected files** (all fixed as of the current version):
- `sentimentizer/tokenizer.py` — `new_dictionary()` and `_count_vocab_batch()` now use `list(doc_tokens)` with `TypeError` catch
- `workflows/stages/tokenize.py` — resume path lambda now uses `hasattr(x, '__iter__')` instead of `isinstance(x, list)`

**Recovery**: Delete the dictionary file and re-run the pipeline:

```bash
rm sentimentizer/data/yelp.dictionary
make train   # or make train-distributed
```

### ValueError: GloVe vocabulary match rate is below 50%

`extract_embeddings()` in `sentimentizer/extractor.py` raises a `ValueError` if fewer than 50% of dictionary words match GloVe vectors. This is a guardrail against the wrapping-quotes bug and similar data corruption.

**Fix**: See "Dictionary has wrapping quotes" above. After fixing the dictionary, re-run the pipeline.

## Scheduler Issues

### Learning rate drops to minimum before training finishes

**Symptoms**: Training loss plateaus or increases after ~50% of epochs; final metrics are poor.

**Cause**: The scheduler's `total_steps` (computed dynamically as `epochs * steps_per_epoch`) is too low, causing the cosine decay to complete before training finishes. This can happen if the dataset is smaller than expected or if `gradient_accumulation_steps` reduces the effective step count.

With per-batch scheduler stepping, `total_steps` is computed as `epochs * steps_per_epoch` where `steps_per_epoch = len(train_loader) // grad_accum_steps`. The invariant is that `total_steps >= epochs * steps_per_epoch` — this is trivially true since `total_steps` equals that quantity.

**Fix**:
- Verify that `gradient_accumulation_steps` is correctly configured. If it's too high, `steps_per_epoch` will be too small and the cosine curve will cover too few steps, causing the LR to decay too quickly.
- `_LinearWarmupCosineScheduler` clamps `progress` to `1.0` so stepping past `total_steps` can't bounce the LR back up.
- In Ray distributed training, `total_steps` is estimated from `train_dataset_count / num_workers` — a slightly off estimate means the cosine curve ends ~10% early or late, which is acceptable.

### Zero learning rate on first epoch (warmup)

**Symptoms**: First epoch shows no learning progress; loss barely changes.

**Cause**: The warmup formula `step / warmup_steps` returns 0.0 at `step=0`, making the LR zero for the entire first epoch.

**Fix**: `_LinearWarmupCosineScheduler` uses `(step + 1) / warmup_steps`, which gives `1/warmup_steps` at `step=0`. A test in `TestSchedulerCorrectness.test_warmup_cosine_no_zero_lr` catches regressions.

## Model Loading Issues

### Tuned model fails to load during validation (`load_state_dict` size mismatch)

**Symptoms**: After a tuning run finds a non-default architecture (e.g. `n_heads=8`, `n_layers=6`), `TuningRun._validate_model` raises a `RuntimeError` from `load_state_dict` about mismatched tensor shapes or unexpected/missing keys.

**Cause**: `n_heads` cannot be inferred from weight shapes — `nn.MultiheadAttention` stores `in_proj_weight` as `(3 * d_model, d_model)` regardless of head count. Reconstructing the model from weight-shape inference alone silently rebuilds it with the **default** architecture, which then fails to load tuned weights.

**Fix**: `_train_final_model` writes a sidecar config JSON next to the weights (`best_model_<type>_config.json` beside `best_model_<type>.pth`). `_load_trained_model` reads it to reconstruct the exact architecture. If the sidecar is missing (Hugging Face downloads, older checkpoints — all of which use the default architecture), it logs `model_config_sidecar_missing` and falls back to weight-shape inference. If you hit this with a custom checkpoint, ensure the sidecar JSON is present alongside the `.pth`. Weights published to the serving path embed the same config as `model_config`, because the sidecar does not travel with them to serving or the Hub.

## Data Pipeline Issues

### Class imbalance in Yelp dataset

The Yelp reviews dataset has roughly 3.5:1 positive-to-negative ratio with neutral (~11%) being the smallest class. Without addressing this, the model learns to always predict positive.

**Solutions** (use one or more, in escalation order):
- **Smoothed class weights** (default): `compute_class_weights()` with `weight_smoothing=0.5` produces sqrt-scaled inverse frequency weights. Controlled by `--weight-smoothing` CLI flag.
- **Label smoothing** (default `0.1`): Regularizes the model from making overconfident predictions. Controlled by `--label-smoothing` CLI flag.
- **Focal loss**: `--loss-type focal --focal-gamma 2.0` applies higher loss to hard examples. Useful when class weights alone aren't enough.
- **Neutral oversampling**: `--neutral-oversample-ratio 0.20` targets 20% neutral reviews in training data via duplication.
- **Undersampling**: `--balance-strategy class_weights_only` (default) avoids aggressive undersampling that destroys 89% of data.

All options are available for `train`, `train-distributed`, and `run` commands.

### Stale metrics from previous training run on dashboard

See the [Stale Metrics Reset](#stale-metrics-reset) section in `metrics.md` for details. `_reset_stale_metrics(model_type)` is called automatically at the start of every training run and zeroes **all** model types (rnn, encoder, decoder), not just the current one.

## GPU / CUDA Issues

### Distributed training runs on CPU instead of GPU

**Symptoms**: `make train-distributed MODEL=encoder` trains on CPU; `torch.cuda.is_available()` returns `False`; `resolve_device("auto")` returns `"cpu"`; dashboard shows RNN metrics from a previous run instead of encoder metrics.

**Root cause**: If you previously ran `make setup-ci` or had the CPU-only torch source configured, `torch` resolves from the CPU-only PyTorch wheel index. This gives `torch 2.x+cpu` which has no CUDA support — even when NVIDIA libraries (`nvidia-cublas`, `nvidia-cuda-runtime`, etc.) are installed, `torch.cuda.is_available()` returns `False`.

**Fix**: Install CUDA-enabled torch (the default since `pyproject.toml` no longer pins CPU-only):

```bash
make setup
```

This runs `uv sync --extra ray` which resolves torch from PyPI with CUDA support.

This tells `uv` to ignore the `[tool.uv.sources]` override and resolve torch from PyPI, which includes CUDA support.

**Diagnostic**: `resolve_device("auto")` logs a warning when it detects CPU-only torch with NVIDIA libraries installed:

```
torch 2.11.0+cpu is CPU-only but NVIDIA CUDA libraries are installed.
Distributed and GPU training will use CPU.
Install the CUDA variant with: uv sync --no-sources-package torch
```

## Running Tests

```bash
# All tests
make test

# Skip Ray-dependent tests (faster, no Ray initialization)
uv run pytest tests/ -v -k "not Ray"

# Only run dictionary/tokenizer tests
uv run pytest tests/test_dictionary_lifecycle.py tests/test_loader.py -v

# Only run scheduler correctness tests
uv run pytest tests/test_training.py -v -k "TestSchedulerCorrectness"
```

## ONNX Export Issues

### RNN ONNX validation fails with high max_diff

**Symptoms**: `validate_onnx_export()` reports `max_diff > 1e-2` for RNN models.

**Cause**: The RNN's ONNX-compatible forward path (masked fallback) skips `pack_padded_sequence` and processes padding tokens. Short sequences have more padding, causing larger numeric drift.

**Fix**: This is expected. The tolerance for RNN is intentionally set to `1e-2` (vs `1e-4` for Encoder/Decoder). If validation still fails:
- Ensure the model was exported with `_RNNOnnxWrapper` (calls `forward(onnx_export=True)`)
- Check that `seq_len` in `export_model_to_onnx()` matches the training sequence length
- Verify the model is in `eval()` mode before export

### `onnxruntime` import error during export

**Symptoms**: `ModuleNotFoundError: No module named 'onnxruntime'` when running `sentimentizer export`.

**Fix**: Install the ONNX optional dependency group:

```bash
pip install -e ".[onnx]"
# or with uv:
uv sync --extra onnx
```

### `sentence_transformers` import error during router training

**Symptoms**: `ModuleNotFoundError: No module named 'sentence_transformers'` (or `sklearn`) when running `sentimentizer router train`.

**Fix**: Install the router optional dependency group:

```bash
pip install -e ".[router]"
# or with uv:
uv sync --extra router
```

> If you hit `No module named 'setfit'`, you are running stale code or a stale
> checkout — `setfit` is no longer a dependency of this project. See
> [router.md](router.md).

### Do I need `optimum-onnx`?

No. This project's ONNX export and quantization use `onnxruntime.quantization.quantize_dynamic` directly (see `sentimentizer/export_onnx.py`); the `[onnx]` extra installs only `onnx`, `onnxruntime`, and `onnxscript`. `optimum-onnx` was previously considered for SetFit ONNX export, but SetFit is no longer a dependency.

## Intent Router Issues

### Router model metadata returns 500

**Check**: Current code returns an ordered category list and validates the typed
metadata response. Older releases returned a dictionary where the response
schema required a list. Update stale service code if logs still identify that
shape mismatch. See the [metadata contract](serving.md#router-model-metadata).

Invalid or unreadable local metadata is reported in the response's `error`
field without loading weights. Check `router_config.json` for duplicate labels,
invalid JSON, or conflicts between `labels` and `task.labels`. Metadata uses
`status: "not_loaded"` before first use and after a cached load failure; this
status alone does not indicate an HTTP error.

### Router label looks correct but its confidence does not

**Check**: Current `RouterModel.classify_batch()` selects the winning probability
column through `head.classes_` and the saved vocabulary. Earlier versions
assumed numeric class IDs were column positions. Update stale callers that
combine raw predict/predict_proba results using that assumption.

The saved label order defines integer IDs and must match the fitted head.
Inspect the artifact's `labels`/`task` metadata if an external producer assigned
different meanings to those IDs; retrain or correct the artifact at its source.
A schema can validate a numeric range but cannot prove an incorrectly labeled
training dataset has the right semantics.

### A custom model path fails instead of downloading

Automatic Hub download is reserved for the default `models/router` food path.
Set `ROUTER_MODEL_PATH` to an existing trained custom artifact directory. It
must contain its backbone, classifier head, and label metadata. A missing
custom path never downloads food weights into that directory. Loading failures
are cached; recreate the predictor/service after correcting the path or artifact.

### Ollama API connection failure during augmentation

**Symptoms**: `augment_seeds()` logs `Ollama API request failed` and returns only the original seeds.

**Fix**: Ensure Ollama is running locally on port 11434:

```bash
ollama serve  # Start the Ollama API server
ollama pull glm-5.1:cloud  # Pull the model (cloud variant)
```

The `augment_seeds()` function gracefully handles API failures — it returns the original seeds unchanged, so training can proceed with just the 30 golden examples if augmentation is unavailable.

### Low inter-class similarity (> 0.65) after training

**Symptoms**: `evaluate_router()` reports inter-class similarity above 0.65, meaning the model cannot distinguish between categories well.

**Fix**: This typically means:
1. Too few training examples — run augmentation to expand seeds
2. Hard negatives not represented — ensure `augment.py` generates category-confusing examples
3. Consider upgrading the base model from `BAAI/bge-base-en-v1.5` to `mxbai-embed-large-v1` in `RouterConfig`

## ModernBERT Issues

### Out-of-Memory (OOM) during training

**Symptoms**: CUDA Out-Of-Memory (OOM) error when fine-tuning ModernBERT.

**Fix**: ModernBERT is a large contextual transformer model. By default, it uses an 8-bit AdamW optimizer via `bitsandbytes` to reduce the VRAM footprint. If `bitsandbytes` is not available, training falls back to standard AdamW, which requires substantially more VRAM. Ensure you run `uv sync --extra ray` to install all required dependencies. If OOM still occurs, reduce `batch_size` in your configuration or CLI command.

### Air-gapped / Offline Environment Loading

**Symptoms**: Loading hangs or throws `ConnectionError` when attempting to load the ModernBERT backbone.

**Fix**: The model loader supports offline mode if the Hugging Face model directory is cached locally. Set `HF_HUB_OFFLINE=1` in your environment, or configure the pre-trained weights directory in `config.py` to point directly to a local copy of the model artifacts.

### ONNX Export Limitations

**Symptoms**: Running ONNX export on a ModernBERT model fails.

**Fix**: ModernBERT currently has `SUPPORTS_ONNX = False` configured in its class definition. The ONNX exporter explicitly checks this property and rejects the export request. Use standard PyTorch weights for serving and inference.

### Ray Serve Worker Crashes with ModuleNotFoundError

**Symptoms**: `ModuleNotFoundError: No module named 'ray'` in Ray Serve worker processes.

**Root cause**: Ray 2.x workers try to create isolated venvs using `uv` when `RAY_ENABLE_UV_RUN_RUNTIME_ENV` is not set to `"0"`. These venvs lack the `ray` package.

**Fix**: Ensure `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` is set before any `import ray` in all entry points:
- `sentimentizer/serve/app.py` (module-level and `main()`)
- `workflows/cli.py` (`serve_cmd()`)
- `workflows/lifecycle.py` (module-level)

If you add a new CLI command or entry point that uses Ray, add `os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")` at the top.

### Ray Serve startup hangs

**Symptoms**: `ray.init()` or `serve.start()` times out.

**Root cause**: Stale Ray processes (raylet, gcs_server) from a previous Ctrl-C or crash are holding ports.

**Fix**: The `lifecycle.py` module-level cleanup handles this automatically — it kills stale processes and cleans `/tmp/ray/` before `ray.init()`. If the auto-retry doesn't help:

```bash
pkill -9 -f raylet
pkill -9 -f gcs_server
rm -rf /tmp/ray/
```

Then restart `make serve`.

### Ray Serve fails to serialize the FastAPI app

**Symptoms**: Startup fails at `@serve.ingress(app)` with `TypeError: Failed to
serialize the ASGI app` and `cannot pickle '_thread.lock' object`. The
serializability report may mention `FastAPI.add_api_route` or
`_openapi_routes_version`.

**Root cause**: Ray Serve 2.55.1's class-based FastAPI transformation is
incompatible with FastAPI 0.139.x. This is dependency-version drift, not a lock
created by the deployment instance. A minimal class-based `@serve.ingress`
application reproduces the failure.

**Fix**: Keep FastAPI pinned to 0.136.1 while Ray remains on 2.55.1. Container
builds must install the versions in `uv.lock`; do not resolve the open-ended
project requirements again during `docker build`.

After changing either dependency, verify the serialization boundary before
loading model weights:

```bash
python -c 'import sentimentizer.serve.bge_only_app'
```

For the FODMAP development stack, rebuild the image so an older dependency
layer is not reused:

```bash
docker compose build --no-cache vectorizer
docker compose up vectorizer
```

### `/health/ready` returns 503 permanently

**Symptoms**: The server is running but `/health/ready` always returns 503 with "models not loaded".

**Root cause**: Model loading failed in `SentimentizerDeployment.__init__()`. The error message is in the 503 response body and in the server logs.

**Fix**: Check the server logs for the model loading error. Common causes:
- Missing model weights (`models/encoder_sentiment.model` or `models/router/`)
- Out of memory (especially with GPU inference)
- Wrong `--model-path` for the router (set via `ROUTER_MODEL_PATH` env var or `--model-path` CLI flag)

### Requests hang indefinitely on `/v1/predict`

**Symptoms**: POST requests to `/v1/predict` never return.

**Root cause**: The `@serve.batch` decorator on `predict_sentiment()` collects requests over a `batch_wait_timeout_s` window (default 50ms). If the batch function raises an exception, individual callers hang because `@serve.batch` doesn't propagate errors back to all callers in all Ray versions.

**Check**: On a successful inference, the batch function (`predict_sentiment`,
`classify_route`) must return a list with the same length as `inputs`. Inspect
replica logs for inference exceptions and output-size mismatches. Error
propagation depends on the installed Ray behavior; do not return ad hoc error
dictionaries as router predictions, because they violate the declared success
response schema.

### Ray Serve worker venv issues (uv)

**Symptoms**: Workers crash with `ModuleNotFoundError` for packages that are installed in the main environment.

**Root cause**: Related to `RAY_ENABLE_UV_RUN_RUNTIME_ENV`. Ray may still try to use `uv` even when the env var is set if `RAY_ENABLE_RUNTIME_ENV_HOOK` is not also set.

**Fix**: The `main()` function in `serve/app.py` sets both:
```python
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_ENABLE_RUNTIME_ENV_HOOK", "1")
os.environ.setdefault("RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES", "")
```
And passes `runtime_env={"py_executable": sys.executable}` to `ray.init()`. This forces workers to use the same Python executable as the driver.

### Batch endpoint returns fewer items than input

**Symptoms**: The `/v1/router/batch` endpoint returns fewer items than the number of input texts.

**Check**: Current code rejects partial probability batches, validates the HTTP
request/result count, and requires Pydantic count to equal results length. An
invalid backend response is a server error, rather than a successful partial
batch. If a successful response is short, check for an older deployment or a
custom adapter bypassing these checks. Every input must have one result in the
same order; do not pad a short response with fabricated predictions.

### CORS preflight fails with duplicate headers

**Symptoms**: Browser requests fail with duplicate `Access-Control-Allow-Origin` headers.

**Root cause**: Some Ray Serve versions add their own CORS headers. If the FastAPI `CORSMiddleware` is also adding CORS headers, you get duplicates.

**Fix**: Check the raw response headers with `curl -v`. If duplicates appear, check the Ray Serve proxy configuration. In the current setup, this shouldn't happen because Ray's proxy doesn't add CORS headers by default.

### Long-running Ray Serve dies when training starts

Training's `_kill_stale_ray_processes` (`workflows/lifecycle.py`) runs `pkill` on
every `raylet`, `gcs_server`, and `ray::IDLE` on the host when `ray.init` fails, and
`_cleanup_stale_ray_sessions` deletes `/tmp/ray/session_*` directories older than an
hour, which can include a live session. This affects any Ray service on the host,
including sentiment Serve and imagenate. Run long-lived services in their own
containers, or do not train on that host.

### Ray Train Worker Crashes

**Symptoms**: Ray workers crash with `ModuleNotFoundError: No module named 'ray'` when starting distributed training.

**Fix**: Make sure `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` is exported in the worker subprocess environment before initializing Ray. Without this, Ray workers try to build separate virtual environments via `uv` that lack the `ray` package.

### Progress Bar Spams Log Lines in Distributed Training

**Symptoms**: Progress bars during distributed training output a new line for every step, filling up the console.

**Fix**: Inside the distributed training loop, the code uses `ray.experimental.tqdm_ray.tqdm` instead of the standard `tqdm`. This intercepts the output and routes it back to the Ray driver centrally to update the progress bar in-place.

## Linting and Formatting

```bash
make ci        # Full CI mirror: check + typecheck + test — run before pushing
make check     # ruff format, ruff check --fix, ruff check
make format    # ruff format, then ruff check --fix --select I (import sort)
make typecheck # pyright
```

**Do not run `black`.** It is not a dependency; the project formats with
`ruff format`.

### CI fails with "Would reformat: ..." but `make lint` passed

`make lint` and `make test-lint` run only `ruff check`, which does **not**
verify formatting. CI runs `ruff format --check .` as a separate step and fails
on any drift.

**Fix**: run `make check` (or `make format`) to apply formatting, then `make ci`
to confirm the whole chain. Note CI uses `bash -e`, so a formatting failure
aborts the job before pyright and pytest run — a green format does not mean the
remaining steps passed.
