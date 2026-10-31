# Intent Router (Review Categorization)

The intent router is a lightweight review classification module that categorizes incoming restaurant reviews into targeted areas. This allows downstream systems to route reviews to appropriate teams or track issues (e.g. flagging gluten-free queries to the kitchen).

> **Not SetFit.** This module was originally built on the `setfit` library, and
> some names still carry that history (`SetFitConfig` is a backward-compatible
> alias for `RouterConfig`). `setfit` is no longer a dependency and is imported
> nowhere. `RouterModel` (`sentimentizer/router/model.py`) is a
> `SentenceTransformer` backbone fine-tuned with contrastive pairs plus an
> sklearn `LogisticRegression` head.

---

## Category Schema

Reviews are classified into three mutually exclusive categories:

- **Dietary**: Mentions of allergies, gluten-free options, celiac requirements, dairy-free alternatives, vegan/vegetarian options, or dietary restrictions.
- **Service**: Mentions of wait times, reservation issues, staff friendliness, ordering issues, table service, customer support, or staff behavior.
- **General**: Ambiance, decor, pricing, menu variety, food taste, overall review quality, location, or general dining experiences.

---

## Python and HTTP Response Contract

`SentimentPredictor.classify(text)` returns a Python dictionary with one
`prediction` envelope:

```python
from sentimentizer.predictor import SentimentPredictor

predictor = SentimentPredictor(model_name="encoder")
route = predictor.classify("They were so careful with my celiac needs")
# Example: {"prediction": {"label": "dietary", "score": 0.95, "token_count": 8}}
```

`classify_batch(texts)` returns a list of those dictionaries. Python results
do not include HTTP timing or a batch-level wrapper. The raw `RouterModel`
methods remain separate: `predict()` returns class labels and
`predict_proba()` returns a NumPy probability matrix. For standalone router
inference, `RouterModel.classify_batch()` returns validated
`RouterClassification` objects using a single encoding pass:

```python
from sentimentizer.router import RouterModel

router = RouterModel.from_pretrained("models/router")
predictions = router.classify_batch(["Great gluten-free options!"])
result = predictions[0].model_dump(mode="json")
# Example: {"label": "dietary", "score": 0.95}
```

This standalone interface does not load sentiment weights or the GloVe dictionary.

The HTTP layer uses these existing Pydantic models from
`sentimentizer/serve/models.py`:

| Endpoint | Response model | JSON envelope |
| --- | --- | --- |
| `POST /v1/router/predict` | `RouterPredictResponse` | `{"prediction": {...}, "latency_s": ...}` |
| `POST /v1/router/batch` | `RouterBatchResponse` | `{"results": [{"prediction": {...}}], "count": ..., "latency_s": ...}` |
| `GET /v1/router/models` | `RouterModelsResponse` | `{"model_path": ..., "categories": [...], "status": ..., "error": ..., "task_id": ..., "taxonomy_version": ...}` |

Each prediction contains `label`, `score`, and `token_count`. The category
vocabulary defaults to `dietary`, `service`, and `general`; `score` represents the winning
category's probability. Token counts use `regex_tokenize`, rather than the
backbone tokenizer. Batch results preserve input order and contain one
prediction per input. See the
[serving guide](serving.md#router-review-categorization-endpoints) for complete
request/response examples, field meanings, and HTTP error responses.

The router loads on the first `classify()` or `classify_batch()` call. Loading
success and failure are cached. Metadata requests do not initiate loading:
`status` is `not_loaded` before first use or after failure, and `loaded` after
success. The declared metadata schema uses an ordered categories array and an
optional error, serialized as `null` when absent. Metadata reads the local
sidecar without loading the backbone. Unknown custom artifacts report empty
categories and null task/version; legacy artifacts may have labels without
task/version metadata. Only the default food path can auto-download the default
food Hub artifact. Missing custom paths fail explicitly.

Shared Pydantic schemas in `router/schemas.py` define the task vocabulary and
bounded classification values. HTTP schemas reuse them and validate counts,
finite nonnegative timing, and unexpected response fields. Inference validates
labels against the artifact and maps probabilities through `head.classes_`.
Invalid output fails the request instead of returning partial results.

## Reuse with External Configurations

The shipped config and seeds remain food-focused. The shared pipeline supports
any configured vocabulary with at least two unique labels. For example, an
external YAML can define another food-related task:

```yaml
training:
  task:
    task_id: food_requests
    taxonomy_version: "1"
    description: Classify restaurant information requests.
    labels: [menu, reservation]
    label_descriptions:
      menu: Questions about dishes and menu options.
      reservation: Questions about booking a table.
    input_format: text-v1
  output_dir: models/food_requests
augmentation:
  seed_path: seeds.jsonl
  output_path: augmented_requests.jsonl
```

Training JSONL uses `text` and integer `label`, where IDs follow the order of
`task.labels`. Every configured class must be represented in training. Boolean,
fractional, and out-of-range IDs are rejected. Extra fields such as `group_id`
are preserved by JSONL saves and inherited by generated variations. Split related
examples together when preparing independent evaluation data; the built-in
training split remains a random row split.

```bash
sentimentizer router augment --config /path/to/requests.yaml
sentimentizer router train --config /path/to/requests.yaml --data augmented_requests.jsonl
sentimentizer router evaluate --model-path models/food_requests --data held_out_requests.jsonl
```

`augmentation.seed_path` is relative to the config file when it is not absolute.
Custom tasks require their own seeds for augmentation. Empty seed_path selects
bundled seeds only for the default food task. CLI overrides retain all other
loaded settings, including task, epochs, sequence limit, and split seed.

Artifacts save task metadata alongside labels/head type in `router_config.json`.
The label order must agree with the head. Old artifacts with saved labels and
legacy SetFit label dictionaries are supported; unknown numeric classes without
label metadata fail. The known default food serving path can supply its explicit
compatibility mapping. Use `ROUTER_MODEL_PATH` to serve another trained artifact.
One artifact is loaded per service instance. Custom input formatting is the
caller's responsibility, used consistently in training and inference and recorded
by `input_format`; YAML does not execute formatter code.

---

## Workflow Stages

### 1. Data Augmentation (GLM 5.1 via Ollama)

Since hand-labeling thousands of reviews is expensive, Sentimentizer uses a few-shot augmentation pipeline. It starts with 30 golden seed examples (10 per category) defined in `sentimentizer/router/seeds.py` and uses **GLM 5.1** via **Ollama** to generate synthetic variations.

```bash
# Run augmentation using default configurations (30 seeds x 50 variations = ~1,500 items)
make router-augment

# Or customize using the CLI
sentimentizer router augment \
  --model "glm-5.1:cloud" \
  --variations 50 \
  --output "augmented_yelp.jsonl"
```

To resume an interrupted generation, add the `--resume` flag to skip already-generated seeds and append new variations.

---

### 2. Model Training

Training follows the SetFit *method* without the library: it fine-tunes a sentence-transformer backbone (`BAAI/bge-base-en-v1.5` by default, 109M params, 768-dim) with contrastive pairs, then fits an sklearn `LogisticRegression` head on the resulting embeddings. `RouterConfig.num_iterations` (default 20) controls contrastive pairs generated per example.

```bash
# Run training on the augmented data
make router-train

# Or train via CLI specifying the data file
sentimentizer router train --data augmented_yelp.jsonl --output-dir models/router
```

---

### 3. Evaluation

`evaluate_router()` reports per-class precision/recall/F1 for the full saved
vocabulary, a centroid cosine-similarity matrix with its observed labels, and
confidence statistics. The matrix diagonal is centroid self-similarity, not an
intra-class quality measurement. No universal similarity thresholds are enforced.

`tau_threshold` is the historical fallback heuristic when `task.fallback_label`
is set (the food task uses `general`); it is null for tasks without a fallback
or without correct evaluation predictions. It is not a calibrated acceptance
guarantee or an automatic serving gate. CLI evaluation prints the report as
JSON and uses every row in the supplied file, which should be independent of training.

```bash
# Evaluate the model
make router-evaluate

# Or run via CLI
sentimentizer router evaluate --model-path models/router --data augmented_yelp.jsonl
```

---

### 4. Push to Hugging Face Hub

Once trained and validated, publish the model directory to the Hugging Face Hub.

```bash
# Push model to the Hub
make upload-router

# Or push via CLI with a custom repo ID
sentimentizer router push --model-path models/router --repo-id your-username/sentimentizer-router
```

Run the whole pipeline (augment → train → evaluate) with `make router-pipeline`.

---

## Environment Variable Overrides

Both `RouterConfig` (training) and `AugmentConfig` (augmentation) support env-var overrides on top of `router/config.yaml`. Loading order (highest priority first):

1. Env var (e.g. `SENTIMENTIZER_ROUTER_NUM_EPOCHS=3`)
2. `sentimentizer/router/config.yaml` value
3. Dataclass default

Training (`SENTIMENTIZER_ROUTER_*`):

| Env var | Field | Type |
| :--- | :--- | :--- |
| `SENTIMENTIZER_ROUTER_BASE_MODEL` | `base_model` | str |
| `SENTIMENTIZER_ROUTER_NUM_ITERATIONS` | `num_iterations` | int |
| `SENTIMENTIZER_ROUTER_NUM_EPOCHS` | `num_epochs` | int |
| `SENTIMENTIZER_ROUTER_BATCH_SIZE` | `batch_size` | int |
| `SENTIMENTIZER_ROUTER_MAX_SEQ_LENGTH` | `max_seq_length` | int |
| `SENTIMENTIZER_ROUTER_SEED` | `seed` | int |
| `SENTIMENTIZER_ROUTER_OUTPUT_DIR` | `output_dir` | str (path) |

Augmentation (`SENTIMENTIZER_AUGMENT_*`):

| Env var | Field | Type |
| :--- | :--- | :--- |
| `SENTIMENTIZER_AUGMENT_MODEL` | `model` | str |
| `SENTIMENTIZER_AUGMENT_OLLAMA_URL` | `ollama_url` | str |
| `SENTIMENTIZER_AUGMENT_VARIATIONS_PER_SEED` | `variations_per_seed` | int |
| `SENTIMENTIZER_AUGMENT_OUTPUT_PATH` | `output_path` | str |
| `SENTIMENTIZER_AUGMENT_BATCH_SIZE` | `batch_size` | int |
| `SENTIMENTIZER_AUGMENT_SEED_PATH` | `seed_path` | str (path) |

Invalid values raise `ValueError` at load time with the offending env var name. See [configuration.md](configuration.md#runtime-configuration) for the full multi-domain reference.

---

## Router CLI Options Reference

### `sentimentizer router augment`
- `--output`: Output JSONL file path.
- `--model`: Ollama model name to use.
- `--variations`: Number of variations to generate per seed utterance.
- `--ollama-url`: Ollama API endpoint.
- `--resume`: Resume from existing output file.
- `--config`: Path to router config YAML (defaults to `router/config.yaml`).

### `sentimentizer router train`
- `--data`: **Required**. Path to the augmented JSONL dataset.
- `--base-model`: Sentence-transformer base model.
- `--output-dir`: Output directory for the trained model.
- `--config`: Path to router config YAML.

### `sentimentizer router evaluate`
- `--model-path`: **Required**. Path to the trained router model directory.
- `--data`: Path to evaluation JSONL data.

### `sentimentizer router push`
- `--model-path`: Path to local model directory.
- `--repo-id`: Hugging Face repository ID.
