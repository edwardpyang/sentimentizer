# Agent Guide

Sentimentizer is a PyTorch sentiment-analysis pipeline with four model
architectures (RNN/LSTM, Transformer Encoder, Encoder-Decoder, ModernBERT). It
supports single-node and distributed training via Ray Train, Ray Serve
inference, an intent router, BGE-M3 embeddings, Prometheus metrics, and Grafana
dashboards. Image generation moved to the separate
[imagenate](https://github.com/edwardpyang/imagenate) service.

## Working rules

- Inspect the current branch and worktree before editing. Preserve unrelated
  user changes and work around a dirty worktree.
- Use repository code, tests, configuration, and current local documentation as
  the source of truth. Plans and dated observations can become stale; verify
  them before relying on them.

Code-style and design rules live in `.rules/`. Read the matching files before
implementing:

| Change area | Required rules |
|---|---|
| Any Python implementation | `.rules/python-style.md`, `.rules/dry.md`, `.rules/solid.md` |
| Tuning knobs or model-call constants | `.rules/prompts-and-constants.md` |
| Logging, timeouts, or long-running work | `.rules/observability.md` |
| A feature plan or multi-milestone implementation | `.rules/plan-review.md`, `.rules/testing-workflow.md` |

## Task routing

| Task | Where to look first |
|---|---|
| Plan a feature, refactor, or architectural change | `.skills/plan-feature/SKILL.md` |
| Change threads, locks, shared state, Ray processes, lazy construction, shutdown | `.skills/change-concurrent-code-safely/SKILL.md` |
| Runtime failures, zero negative accuracy, GloVe mismatch, scheduler issues | `docs/troubleshooting.md` |
| Metric definitions, gauges, stale-metric behavior | `docs/metrics.md` |
| Config dataclasses and env vars | `docs/configuration.md` |
| Serving endpoints and API shape | `docs/serving.md` |
| Training behavior and loops | `docs/training.md`, `docs/tuning.md` |
| ONNX export | `docs/onnx.md` |
| Intent router | `docs/router.md` |
| Hugging Face upload/download | `docs/huggingface.md` |

`docs/*_plan.md` files are historical design context, not current behavior.
Treat the code as the source of truth when they disagree.

## Verification

**`make ci` is the completion gate for any code change.** It runs
`check` (ruff format, ruff fix, ruff check), then `typecheck` (pyright), then
`test` (pytest) — the same four gates CI enforces, in the same order.

```bash
make ci        # full local mirror of CI — run before pushing
make check     # format + autofix + lint only (fast inner loop)
make typecheck # pyright
make test      # pytest tests/
```

Rules:

- **Do not run `black`.** It is not a dependency. The project formats with
  `ruff format`; running another formatter produces churn that CI rejects.
- **`make lint` and `make test-lint` do not check formatting.** They run only
  `ruff check`. CI runs `ruff format --check .` as a separate step and fails on
  drift, so a green `make lint` does not mean a green CI. Use `make ci`.
- CI aborts at the first failing step (`bash -e`). A formatting failure means
  pyright and pytest never ran — fix the format, then re-run `make ci` locally
  before assuming the rest passes.
- Run verification **only when code changed**. Documentation- and
  markdown-only changes do not need the suite.
- pyright reports warnings that do not fail the build; only `error` counts.

## Architecture

```
sentimentizer/
  config.py            — Dataclass configs (DriverConfig, RNNConfig, EncoderConfig, *SchedulerParams)
  trainer.py           — Trainer, new_trainer(), _train_func(), Ray gauge logic, scheduler rebuild
  tuner.py             — Ray Tune integration, TunePrometheusCallback
  metrics.py           — ClassificationMetrics dataclass + compute functions (torchmetrics)
  metrics_publisher.py — Per-model-type JSON writes, write_batch_snapshot(), _METRIC_GAUGE_KEYS
  exporter.py          — Standalone Prometheus exporter (port 8081), all gauge definitions
  losses.py            — FocalCrossEntropyLoss
  loader.py            — DataLoader, Ray Data loading, class balancing
  tokenizer.py         — Vocabulary/dictionary build, regex_tokenize, convert_rating
  hf_tokenizer.py      — HFTokenizer for ModernBERT
  hf_dataset.py        — HFDataset / HFCollateFn (dynamic per-batch padding)
  predictor.py         — SentimentPredictor (model loading, inference, lazy router load)
  export_onnx.py       — ONNX export, quantization, validation (_RNNOnnxWrapper)
  extractor.py         — Raw data extraction
  data_source.py       — Dataset source abstraction
  device.py            — resolve_device() / auto_detect_device()
  safety.py            — Prompt-injection patterns for web search results
  env.py               — Environment variable handling
  hf.py                — Hugging Face Hub upload/download
  models/
    base.py            — BaseSentimentModel, predict(), predict_text(), STEP_SCHEDULER_PER_BATCH
    hf_base.py         — HFTransformerModel base (weights + config sidecar + tokenizer dir)
    rnn.py             — Bidirectional LSTM (onnx_export flag)
    encoder.py         — Transformer encoder, mean pooling
    decoder.py         — Encoder-decoder transformer
    modernbert.py      — ModernBERT wrapper (new_modernbert_model)
  serve/
    app.py             — Main Ray Serve deployment: FastAPI + @serve.ingress, /v1 sub-app
    base.py            — ServiceMetrics, _DummyServe fallback (no response builders)
    config.py          — ServeConfig dataclass + YAML/env loading (incl. cors_origins)
    models.py          — Pydantic request/response models
    bge_only_app.py    — BGE-M3-only deployment
    embeddings_app.py  — Embeddings deployment
    embeddings_models.py
    service.yaml       — Serve config
  embeddings/
    bge_m3.py          — BGE-M3 dense/sparse embeddings
    predictor.py
  router/              — Intent router (SentenceTransformer + LogisticRegression)
    schemas.py         — Shared task, classification, and artifact metadata schemas
    config.py          — RouterConfig (alias SetFitConfig), RouteLabels, AugmentConfig
    model.py           — RouterModel: backbone + sklearn head
    seeds.py           — Golden example utterances (10 per category)
    augment.py         — GLM 5.1 augmentation via Ollama
    dataset.py         — JSONL loader, train/test split
    train_router.py    — Contrastive fine-tune + LogisticRegression head fit
    evaluate.py        — Similarity heatmap, threshold calibration
    calibration.py     — Frozen-head probability calibration and acceptance policy selection
    provenance.py      — Dataset fingerprints, used-data hashes, run configuration/versions
    benchmark.py       — TF-IDF, frozen encoder, and contrastive encoder comparison
  agent/
    diagnose_model.py  — Tuning/diagnosis agent
    websearch.py       — Typed web search tool
    agents.py graph.py nodes.py state.py prompts.py loader.py models.py

workflows/
  driver.py            — CLI entry point
  cli.py               — Click command definitions
  lifecycle.py         — State, logger, Ray init, stale-session cleanup, atexit handlers
  helpers.py           — Model loading, config utilities
  stages/
    train.py           — run_train(), _run_fit_single(), _run_fit_distributed(), _reset_stale_metrics()
    tune.py            — Ray Tune orchestration
    tokenize.py        — Dictionary build / resume path
    extract.py export.py diagnose.py hf.py
```

`tests/` mirrors these modules (28 `test_*.py` files plus `conftest.py`). Find
the relevant one with `ls tests/` rather than trusting a list here.

## Metrics Pipeline

- **Training driver** writes `/tmp/sentimentizer_metrics/{model_type}_metrics.json` (one file per model type)
- **Batch snapshots** write `/tmp/sentimentizer_metrics/{model_type}_batch.json` every N batches (10 for ModernBERT, 50 otherwise) for intra-epoch dashboard visibility
- **Standalone exporter** (port 8081) reads both every 10s, serves `sentimentizer_training_*` gauges
- **Ray workers** (port 8080) emit `ray_sentimentizer_live_*` gauges from rank 0
- **Tune process** (port 8082) emits `sentimentizer_tune_*` gauges per trial
- **Grafana** uses `sentimentizer_training_* or ray_sentimentizer_live_*` PromQL fallback

Per-model-type files exist so concurrent training runs never race on a shared
file. `_reset_stale_metrics(model_type)` writes zeroed JSON for **all** model
types (so the exporter can clear stale gauges) but resets **Prometheus gauges
only for the current model type**. Zeroed files carry `_reset: true` and
`_trace.reset_by`; the exporter skips `_reset: true` epoch data so untrained
models do not show `epoch=0`, while still reading batch snapshots.

## Common Tasks

### Dependency management

```bash
uv sync                          # local-only mode
uv sync --extra ray              # Ray distributed features
uv sync --extra dev --extra ray  # dev deps (ruff, pytest, pyright)
uv add <package>                 # add a dependency
uv run <command>                 # run in the venv
```

Extras: `dev`, `ray`, `router` (sentence-transformers, scikit-learn), `onnx`,
`embeddings`. Combos allowed: `uv sync --extra router,onnx`. CI installs
`dev,ray,onnx,router`.

### Running tests

```bash
uv run pytest tests/ --exitfirst --failed-first
uv run pytest tests/ -k "Ray"      # Ray-specific
uv run pytest tests/ -v            # verbose when debugging
```

### Checkpointing

All `make train*` targets checkpoint to `checkpoints/<MODEL>/` every epoch
(`--checkpoint-dir checkpoints/<MODEL> --checkpoint-every 1`), so an
interrupted machine does not lose the run. `TrainerConfig.checkpoint_dir`
defaults to `""` (disabled) — checkpointing is enabled by CLI/Makefile, not by
the config dataclass.

- Resume: `make train-resume MODEL=modernbert` (uses `--resume-train` →
  `latest_checkpoint()` → restores model/optimizer/scheduler state)
- Disable: `make train-no-checkpoint`
- Contents: model weights, optimizer state, scheduler state, epoch, val_loss
- `best_model.pth` written whenever val_loss improves (`--checkpoint-best`, default True)

### Sleep prevention

All `make train*` targets wrap training in `systemd-inhibit --what=sleep` on
Linux when available, detected at Makefile parse time via `INHIBIT_SLEEP`.
Disable with `make train INHIBIT_SLEEP=`. Only sleep is inhibited, not idle
shutdown or lid close.

### Regenerating dashboards

```bash
make start-metrics   # regenerate JSON + restart Grafana + start exporter
```

Grafana reads provisioned dashboards only at **startup**, so a restart is
required after any dashboard change.

### Adding a new model type

1. Config dataclass in `sentimentizer/config.py`
2. Model class in `sentimentizer/models/`
3. Optimization/scheduler params in `config.py` (`_get_opt_params`,
   `_get_sched_params` — include `warmup_ratio`)
4. Model factory import in `trainer.py` (`_train_func`, `new_trainer`)
5. Model import in `workflows/helpers.py` (`_load_model`, `_get_model_config`)

### Adding a new training metric

1. Gauge definition in `exporter.py` (with `model_type` label)
2. Gauge in `_get_ray_gauges()` in `trainer.py`
3. Add to `_METRIC_GAUGE_KEYS` in `metrics_publisher.py`
4. Set in `Trainer.evaluate()`, `_train_func()`, and via `publish_epoch_metrics()`
5. Add to `_reset_stale_metrics()` in `workflows/stages/train.py` (reset to 0)
6. Add to `write_epoch_metrics_to_file()` in `metrics_publisher.py` and
   `_update_training_metrics()` in `exporter.py`
7. Add to dashboard in `scripts/generate_ray_dashboards.py`

## Conventions

### Model and tensor contracts

- **3-class classification.** Logits are `(B, 3)`; labels are 0=negative,
  1=neutral, 2=positive. `LABEL_NAMES = ["negative", "neutral", "positive"]`
  and `NUM_CLASSES = 3` in `config.py` are the single source of truth — import
  them, never duplicate.
- **`forward()` output is always `(B, num_classes)`.** Never squeeze; a
  batch-of-1 returns `(1, 3)`, not `(3,)`.
- **`predict()`** returns `torch.softmax(logits, dim=-1)` — a
  `(B, num_classes)` probability matrix.
- **Loss is `CrossEntropyLoss`**, not `BCEWithLogitsLoss`. **Target dtype is
  `torch.long`** — never call `.float()` on targets, in either the DataLoader
  or the Ray path. `FocalCrossEntropyLoss` (`losses.py`) is available via
  `loss_type="focal"`.
- **`predict_text()`** on `BaseSentimentModel` returns all three scores
  (`{"negative": …, "neutral": …, "positive": …}`) with no `token_count` or
  `model` field. Used by `diagnose_model.py` and `hf.py`.
- **Tuned models save a sidecar config JSON** (`best_model_<type>_config.json`)
  — `n_heads` cannot be inferred from weights, so reconstruction needs it.
- All function signatures need type hints.

### Serving API

- **`predict_batch()` returns the additive v1 format**, and `predict()` is
  `predict_batch([text])[0]`:

  ```python
  {"label": "positive", "score": 0.88,
   "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
   "token_count": 12, "model": "encoder",
   "positive": 0.88}  # deprecated dynamic winning-class key, kept for back-compat
  ```

  The HTTP layer wraps this in `"prediction": {...}` with `"latency_s"`.
  `token_count` comes from `len(regex_tokenize(text))` computed during existing
  tokenization — no extra cost.
- **Router Python and HTTP responses:** `classify_batch()` returns a list of
  `{"prediction": {"label": "dietary", "score": 0.95, "token_count": 8}}`
  dictionaries; `classify()` returns one item. HTTP single responses add
  `latency_s`; HTTP batch responses wrap the list in `results`, with `count`
  and `latency_s`. Prediction objects have no `text`, `category`, or `model`
  field. Shared Pydantic task/classification schemas live in `router/schemas.py`;
  HTTP envelopes in `serve/models.py` reuse them. Router handlers return typed
  response models and validate before success accounting. Scores are finite in
  `[0,1]`; counts and latency are nonnegative; batches must be complete. Labels
  come from the loaded artifact and probabilities follow `head.classes_` order.
  Keep additional task configurations external to the food-focused defaults.
- **Router quality and decisions:** Split original source groups before
  augmentation. Preserve `group_id` through generated variants. Training,
  validation, calibration, policy selection and test data must be disjoint;
  artifact provenance records used text/source hashes and evaluation rejects
  overlap. `RouterModel.decide_batch()` and `/v1/router/decide{,/batch}` expose
  accept/abstain separately from the topic label. Classification responses stay
  compatible. Decision endpoints require a saved acceptance policy. Failed
  operating targets produce abstain-all; no threshold implies no acceptance.
- **FastAPI + `@serve.ingress`, one app per deployment.** There is **no mounted
  `v1` sub-app** — routes declare their full path on the app returned by
  `create_fastapi_app()`, e.g. `@app.post("/v1/sentiment/predict")`. Health
  endpoints stay unversioned (`/health`, `/health/live`, `/health/ready`).
- **Routes are namespaced by domain; the flat forms are deprecated.** Canonical:
  `/v1/sentiment/{predict,batch,tokenize,models,models/{name}}`,
  `/v1/router/{predict,batch,models}`, `/v1/embeddings`, and
  `/v1/embeddings/dense`. `/v1/images/*` belongs to the imagenate service. The unnamespaced `/v1/predict`, `/v1/batch`,
  `/v1/tokenize`, `/v1/models`, `/v1/models/{name}` still exist but are declared
  `deprecated=True` — do not add new callers.
- **`GET /metrics` serves per-replica Prometheus text**, rendered by
  `render_service_metrics()` in `serve/base.py` with prefix
  `sentimentizer_service`. It is per-replica: aggregate across replicas rather
  than treating one scrape as a cluster total.
- **Never define `__call__`** on a deployment class — `@serve.ingress`
  forbids it. `@serve.batch` methods stay internal, called from route handlers.
- **Do not share one `FastAPI` instance across deployments.** Two
  `@serve.ingress()` deployments sharing an app collide on routes and
  middleware. Use a factory such as `create_fastapi_app()` so each deployment
  gets its own instance.
- **Trailing slashes.** `@app.post("")` raises `FastAPIError: Prefix and path
  cannot be both empty`. A deployment with a non-root `route_prefix` must use
  `@app.post("/")`, so clients must send the trailing slash or get a 307. Document
  such endpoints with the trailing slash.
- **`serve.start()` timeouts need an explicit object.** `http_options={"request_timeout_s": 600}`
  is silently ignored; pass
  `ray.serve.config.HTTPOptions(..., request_timeout_s=600)`.
- **Middleware order:** CORS is outermost (added last, processed first) with
  `allow_origins=cfg.cors_origins` (default `["*"]`, configurable via
  `SENTIMENTIZER_CORS_ORIGINS`, comma-separated). Request-ID is inner to CORS:
  `X-Request-Id` is read or generated, set on the response and
  `request.state.request_id`. `_RequestBodySizeLimitMiddleware` rejects
  `Content-Length` > 1 MiB with 413, matching the K8s ingress
  `proxy-body-size: "1m"`.
- **Validation is centralized in Pydantic.** Request models use
  `Annotated[str, Field(min_length=1, max_length=cfg.max_text_length)]`;
  `BatchRequest.texts` validates both per-item length and list size. Do not add
  manual `HTTPException(400)` validation — Pydantic returns 422.
- **Error envelope:** HTTP exceptions return
  `{"error": {"code": ..., "message": ...}}`; unhandled exceptions add
  `request_id`. Pydantic 422 responses keep their default shape.
- **`model` field on requests.** `PredictRequest`/`BatchRequest` accept optional
  `model: str | None`. If given it is validated against the loaded model (400 on
  mismatch); if omitted the default model is used.
- **`GET /v1/models/{model_name}`** returns 400 for unknown names, 404 when the
  model exists but is not loaded.
- **`serve/base.py` holds `ServiceMetrics`, `_DummyServe`, and
  `render_service_metrics()`.** Response dicts are built directly in route
  handlers, except router handlers return typed response models — there are no
  response-builder helpers.
- **Router loads lazily.** `SentimentPredictor.__init__` does not load the
  router model; `_ensure_router_loaded()` runs on the first
  `classify()`/`classify_batch()` and memoizes success *and* failure. Serve
  handlers let `classify_batch` trigger the load and catch failures → 503; they
  must not pre-guard on `router_loaded`. `get_router_model_info()` reports
  `"not_loaded"` before first use.
  A per-predictor condition elects one load owner and publishes terminal state;
  never hold it during construction/network I/O. All exit paths notify waiters.
  `load_router()` preserves memoized failure; `retry_router_load()` explicitly
  retries it or joins an active attempt. POST `/v1/router/load` exposes these
  with strict `retry_failed: bool = false`. GET `/health/router` reads readiness
  without triggering loading and reports decision readiness separately.

### Ray Serve environment

- **`RAY_ENABLE_UV_RUN_RUNTIME_ENV` must be `0`/`false`, set at module level
  before any `import ray`.** Two separate failures ride on this one variable:
  - Left at its default, Ray workers build isolated venvs that lack `ray`,
    producing `ModuleNotFoundError: No module named 'ray'`.
  - It also controls cold start: the uv-run runtime-env hook packages the whole
    CWD (~3 MB) and reinstalls all 172 packages on every cold start, adding
    ~37 s. Disabling it drops BGE-M3 cold start from ~52 s to ~16 s. Workers
    already inherit the project venv via `uv run`, so the packaging is
    redundant.

  It must be **module level** — Ray reads it at `ray_constants.py` import time,
  not at `ray.init()`. Setting it inside `main()` is too late. Currently set in
  `serve/app.py`, `serve/bge_only_app.py`, and `workflows/lifecycle.py` (for
  training). **Any new Serve entry point must add**
  `os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "false")` at module
  level.
- **`auto_detect_device()` requires an argument.** It re-exports
  `resolve_device(device)`. Always call `auto_detect_device("auto")` or
  `resolve_device("auto")`, never bare.

### Metrics and torchmetrics

- **`compute_class_weights()`** replaces `compute_pos_weight()` (the old
  function raises `ValueError` when `num_classes > 2`).
- **`ClassificationMetrics`** is 3-class: per-class fields
  (`negative_precision`, `neutral_recall`, …), `balanced_accuracy`, `macro_f1`,
  `weighted_f1`, `mcc`, `confusion_matrix` (3×3), `neutral_to_positive_rate`,
  `neutral_to_negative_rate`, `pred_neutral_frac`. The binary fields (`tp`,
  `tn`, `fp`, `fn`, `precision`, `recall`, `f1`, `npv`, `positive_accuracy`,
  `negative_accuracy`, plus scalar `auc_roc`/`avg_precision`) no longer exist.
- **Prometheus gauges use per-class names** (`TRAINING_VAL_NEUTRAL_F1`,
  `TRAINING_VAL_NEGATIVE_PRECISION`), not binary ones. Ray gauge dict keys must
  match `_METRIC_GAUGE_KEYS` in `metrics_publisher.py`.
- **`torchmetrics.MulticlassAUROC` silently returns wrong results on NaN
  input** — always call `_replace_nan_probs()` first. It accepts
  `(N, num_classes)`, replaces NaN with `1/num_classes`, and re-normalizes rows
  with partial NaN to sum to 1.0.
- **`torchmetrics.MulticlassCohenKappa` returns `nan` for single-class
  targets** — wrap with `_safe_item()` to coerce `nan → 0.0` for Prometheus
  compatibility. Single-class kappa is therefore `0.0`, a deliberate behavioral
  change from the old custom implementation (which returned `1.0`).
- **torchmetrics crashes on empty input** — keep the `if total == 0` early
  return in `compute_classification_metrics()`.
- **Never create `prometheus_client` gauges at module import time** for Ray
  workers; use lazy init via `_get_ray_gauges()`.
- **`_balance_dataframe()` / `_balance_ray_dataset()`** handle multi-class
  targets (0, 1, 2), not just binary.
- Tuning knobs: `weight_smoothing` (default `0.5`; `1.0` = full inverse
  frequency, `0.0` = uniform), `label_smoothing` (default `0.1`),
  `neutral_oversample_ratio` (default `0.0`; `0.20` oversamples neutral to 20%
  of training data).

### Data handling

- **When iterating DataFrame or batch columns of token lists, use
  `list(doc_tokens)` with a `TypeError` catch — never `str(doc_tokens)`.**
  Numpy arrays from parquet are iterable but fail `isinstance(x, list)`, and
  `str()` produces `"['if' 'you' ...]"`, whose regex tokenization captures the
  wrapping quotes. This previously caused a 99.9% GloVe vocabulary mismatch and
  majority-class collapse.

### Scheduler

- **All models step the scheduler per batch** (`STEP_SCHEDULER_PER_BATCH = True`
  on `BaseSentimentModel`). The scheduler is rebuilt with real optimizer-step
  counts in `Trainer.fit()` / `_train_func()` once the DataLoader length is
  known.
- **`warmup_ratio` (default `0.06`) is the live knob**; `T_max` and
  `warmup_epochs` in `SchedulerParams` are legacy fields from per-epoch
  stepping. `CosineAnnealingLR` is no longer used anywhere — every model uses
  `_LinearWarmupCosineScheduler`.
- **`_LinearWarmupCosineScheduler` must use `(step + 1) / warmup_steps`** (plain
  `step / warmup_steps` gives zero LR for the whole first epoch) and must clamp
  `progress` to `1.0` so stepping past `total_steps` cannot bounce the LR back
  up. It returns a *relative* lambda multiplier (`eta_min / base_lr`) so the LR
  bottoms out exactly at `eta_min`.

### Training paths

- **`_load_model()` runs only in the single-node path.** Distributed training
  (`_run_fit_distributed`) does not load the model in the driver — Ray workers
  build their own via `_train_func`.
- **`_reset_stale_metrics(model_type)`** is called at training start; see
  Metrics Pipeline for its all-types/current-type-only split.

### ONNX export

- **RNN `onnx_export` flag**: `RNN.forward(inputs, onnx_export=True)` bypasses
  `pack_padded_sequence` (ONNX-incompatible) with a masked fallback that pulls
  hidden states from `lstm_out`. The default path is unchanged for training and
  inference.
- **`_RNNOnnxWrapper`** exists because `torch.onnx.export()` calls
  `model(*args)` and cannot pass keyword arguments.
- **Tolerances**: `1e-2` for RNN (masked fallback has padding drift), `1e-4`
  for Encoder/Decoder.
- **Opset 17** — stable. Opset 18+ needs `dynamo_export`, still preview.
- **Quantization**: `onnxruntime.quantization.quantize_dynamic` with
  `QuantType.QInt8` (FP32 activations, INT8 weights — optimal for AVX-512).
- Use **`optimum-onnx[onnxruntime]`**, not `optimum[onnxruntime]` — `optimum`
  v2.0+ split ONNX into a separate package.
- Artifacts go to `onnx_artifacts/` (gitignored) with a metadata JSON beside
  each `.onnx` recording model_type, opset_version, input_shape, dictionary
  path, `num_classes`, `label_names`, and validation results.
- CLI: `sentimentizer export --model rnn --quantize`.
- **`SUPPORTS_ONNX` gates export.** `BaseSentimentModel` sets it `True`;
  `HFTransformerModel` sets it `False`, so ModernBERT and any other HF-backed
  model are rejected cleanly.

### Intent router

- **The router does not use SetFit.** `setfit` is not a dependency and is
  imported nowhere. `RouterModel` (`router/model.py`) is a
  `SentenceTransformer` backbone fine-tuned with contrastive pairs, plus an
  sklearn `LogisticRegression` head fitted on the resulting embeddings. Do not
  reintroduce `setfit`, `SetFitModel`, or a `transformers.default_logdir`
  compat shim — all were removed.
- **`SetFitConfig` is a backward-compatible alias for `RouterConfig`**
  (`router/config.py`). Prefer `RouterConfig` in new code.
- The module is still named `router/` rather than `setfit/`, which also avoids
  shadowing any similarly named package.
- **`RouterConfig` defaults**: `base_model="BAAI/bge-base-en-v1.5"` (109M
  params, 768-dim), `num_iterations=20` (contrastive pairs per example),
  `num_epochs=1`, `batch_size=16`, `max_seq_length=512`, `seed=42`,
  `output_dir=models/router`. Move to `mxbai-embed-large-v1` (335M) only if
  evaluation thresholds are not met.
- **Default food categories** in `RouteLabels`: dietary (0) — allergies, celiac, FODMAP,
  ingredient safety; service (1) — wait times, staff, reservations; general (2)
  — ambiance, price, general food quality.
- **Seeds**: 10 utterances per category in `seeds.py`, expanded by `augment.py`
  (`AugmentConfig`: GLM 5.1 via Ollama at `http://localhost:11434/api/generate`,
  default model `glm-5.1:cloud`, 50 variations per seed).
- **Generic task configuration:** `RouterConfig.task` defines ordered labels,
  descriptions, task/version IDs, input format, and optional fallback. Artifacts
  save that spec; runtime code must not substitute `RouteLabels` for custom
  vocabularies. `RouterModel.classify_batch()` returns `RouterClassification`
  objects without loading sentiment weights. Empty augmentation.seed_path uses
  bundled seeds only for the default food task; custom tasks supply JSONL seeds.
- **Evaluation** reports all saved classes, observed centroid similarities,
  and confidence statistics. The centroid matrix diagonal is self-similarity,
  not an intra-class metric. The optional historical tau heuristic uses the
  configured fallback and is not a calibrated serving policy.
- Pipeline: `make router-augment` → `make router-train` → `make router-evaluate`;
  `make upload-router` pushes to the Hub.
- Router ONNX export is deferred to v2 — the router runs Python inference.

## Ray 2.55 API Conventions

This project uses **Ray 2.55.1**, whose API changed significantly from earlier
versions. Verify against the installed source in
`.venv/lib/python3.12/site-packages/ray/` if unsure.

### Ray Data (`ray.data.Dataset`)

- **`Dataset` objects are NOT iterable.** Never `for row in ds:`. Use
  `ds.iter_rows()`, `ds.iter_batches()`, or `ds.iter_torch_batches()`.
- **Split** with `ds.train_test_split(test_size=0.2, shuffle=True, seed=42)` →
  `(train_ds, val_ds)` tuple of `MaterializedDataset`. `ds.random_split()` does
  not exist.
- **Sample a fraction** with `ds.random_sample(fraction, seed=42)` → a single
  `Dataset`, **not** a tuple.
- **Filter** with `ds.filter(fn=)` or `ds.filter(expr=)` (prefer `expr`).
- **Shuffle** `ds.random_shuffle(seed=42)`; **concatenate** `ds.union(other)`;
  **count** `ds.count()` (materializes).
- **Rich progress bars** are on by default via `DataContext` in `loader.py` and
  env vars in `sentimentizer/__init__.py`
  (`enable_rich_progress_bars = True`, `use_ray_tqdm = False`,
  `RAY_DATA_ENABLE_RICH_PROGRESS_BARS=1`, `RAY_TQDM=0`). Requires `rich`.

### Ray Train (`ray.train`)

- **Checkpoints are directory-based only.** `Checkpoint.from_dict()` /
  `to_dict()` were removed in Ray 2.55+.

```python
# Writing a checkpoint (inside a training function)
import os, tempfile
import ray.cloudpickle as pickle
from ray.train import Checkpoint

checkpoint_data = {
    "model_state_dict": model.module.state_dict(),  # unwrap DDP
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
}
with tempfile.TemporaryDirectory() as checkpoint_dir:
    with open(os.path.join(checkpoint_dir, "data.pkl"), "wb") as fp:
        pickle.dump(checkpoint_data, fp)
    checkpoint = Checkpoint.from_directory(checkpoint_dir)
    tune.report({...}, checkpoint=checkpoint)
```

```python
# Reading a checkpoint (on the driver)
with result.checkpoint.as_directory() as checkpoint_dir:
    with open(os.path.join(checkpoint_dir, "data.pkl"), "rb") as fp:
        checkpoint_data = pickle.load(fp)
    model_state_dict = checkpoint_data["model_state_dict"]
```

- **`train.get_context()`** works only inside a Ray Train worker function
  launched by `trainer.fit()`. Never call it from the driver.
- **`prepare_model()`** wraps a model with DDP — reach the original via
  `model.module`.

### Ray Tune (`ray.tune`)

- `tune.report({...})` to report metrics; `tune.Tuner` (not deprecated
  `tune.run`); `tune.with_parameters()` for large objects such as datasets;
  `tune.with_resources()` for resource requirements.

### Common pitfalls

- **Never iterate a `Dataset` directly** — `for x in ds` raises `TypeError`.
- **Never use `ds.random_split()`** — use `train_test_split()` or
  `random_sample()`.
- **Never use `Checkpoint.from_dict()` / `to_dict()`** — removed.
- **Never call `train.get_context()` outside a worker** — raises `RuntimeError`.
- **`get_dataset_shard` is a standalone function**: `train.get_dataset_shard("train")`,
  NOT `train.get_context().get_dataset_shard("train")`.
- **`random_sample(fraction)` returns one `Dataset`** — do not unpack it.
- **Never create `ray.util.metrics.Gauge`/`Counter`/`Histogram` at module import
  time.** In Ray 2.55+, custom metric objects created in the driver are never
  exported; they must be created inside a worker context to register with that
  worker's metrics agent. Use lazy init — see `_get_ray_gauges()` in
  `trainer.py` for the canonical module-level-cache + factory pattern.
- **Ray session temp files accumulate in `/tmp/ray/` (5+ GB each).** Each
  `ray.init()` creates a session directory cleaned up only by `ray.shutdown()`;
  a crash leaves it behind. `workflows/lifecycle.py` clears stale sessions at
  startup via `_cleanup_stale_ray_sessions()`, can force-clear via
  `_kill_stale_ray_processes()` on a failed init, and registers `_ray_cleanup()`
  and `_cuda_cleanup()` with `atexit`. Always call `ray.shutdown()` in tests and
  scripts.
- **In tests**: `ray.init(ignore_reinit_error=True)`, `ray.shutdown()` in cleanup.

## Web Search Utility

Two interfaces:

1. **Python module** (`sentimentizer/agent/websearch.py`) — typed, secure,
   wired into the tuning agent as an `@agent.tool`. Preferred for code and
   agent use.
2. **Shell script** (`scripts/web_search.sh`) — quick manual queries.

Requires `OLLAMA_API_KEY` in the environment (copy `.env.example` → `.env`).

```python
from sentimentizer.agent.websearch import web_search, WebSearchResult, reset_rate_limit

results: list[WebSearchResult] = web_search("best learning rate for RNN")
for r in results:
    print(r.title, r.url, r.content[:100])

reset_rate_limit()  # call at the start of each agent run
```

Safeguards in the Python module: the API key is read from the env var only and
never passed as a parameter or surfaced in errors; queries are length-capped
(200 chars) and screened for secret patterns; results are truncated (2000
chars) and filtered for prompt-injection patterns (`ignore previous
instructions`, `system:`, `<system>`); errors have Bearer tokens and key values
replaced with `[REDACTED]`; calls are rate-limited to 3 per agent run; requests
time out after 15 s.

Use it for API documentation and library versions not in the codebase, known
issues or breaking changes in dependencies, current best practices, and facts
that need to be up to date.
