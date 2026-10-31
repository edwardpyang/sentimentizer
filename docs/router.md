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
training split keeps source groups and normalized duplicates together. Every class
must have enough independent source groups to appear in each partition. Missing
`group_id` allows only exact-duplicate checks; it cannot establish paraphrase independence.

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

To resume an interrupted generation, add `--resume` to request missing variations
for incomplete seeds. Progress is counted from accepted unique variations.

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
sentimentizer router evaluate --model-path models/router --data data/router/split/test.jsonl
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

Run the whole pipeline (split → augment → train → evaluate) with `make router-pipeline`.

The Makefile pipeline now requires labeled original sources at
`data/router/reviews.jsonl` (`ROUTER_RAW_DATA` overrides this) and writes independent
partitions to `data/router/split` (`ROUTER_DATA_DIR`). It splits **before**
augmentation, augments only training sources, uses validation during training,
and evaluates only on the test file. It stops if augmentation is incomplete.

## Quality, calibration, and decisions

### Collect and reserve independent data

Collect representative real inputs and have people annotate them using a written
policy. Keep their original source/review/conversation ID in `group_id`; all
paraphrases and snippets from the same source inherit it. Record annotation
provenance in additional fields. The 30 bundled seeds are an augmentation starter,
not a production benchmark. Synthetic generation checks labels structurally;
sample and audit their semantic correctness before training.

For the shipped food task, use this annotation policy when building a new dataset:

| Input meaning | Primary label |
|---|---|
| An explicit restriction, allergy, suitability question, or accommodation issue, including staff handling it | dietary |
| Staff, waiting, reservations, or order handling without a dietary issue | service |
| Taste, ambiance, price, or other topics without either issue above | general |

For mixed-topic inputs, dietary takes precedence over service, then general.
Mentioning an ingredient alone is insufficient: "the bread was dry" is general;
"the gluten-free bread was dry" is dietary. "The waiter checked my nut allergy"
is dietary; "I have no allergies; the waiter was rude" is service. Label the
meaning of the whole input, including negation. Keep difficult mixed-topic,
negation, spelling, long-review, and out-of-domain examples as separate review
slices. Resolve disagreements before generating variants. Record a new
`taxonomy_version` if adopting this changes an existing task's annotation policy;
the code does not impose these food rules on external tasks or rewrite old artifacts.
Put the same precedence rule in the external task's `label_descriptions` so the
augmentation prompt and annotation guide agree.

Reserve calibration and policy-selection files in addition to train/validation/test:

```bash
sentimentizer router split --data labeled_reviews.jsonl --output-dir data/router/split \
  --calibration-size 0.1 --policy-size 0.1
sentimentizer router augment --seeds data/router/split/train.jsonl \
  --output data/router/split/train.augmented.jsonl
sentimentizer router train --data data/router/split/train.augmented.jsonl \
  --validation-data data/router/split/validation.jsonl --output-dir models/router
```

The default split reserves 20% each for validation/test; optional calibration and
policy fractions are additional. Fractions are approximate for unequal groups.
The splitter connects source groups linked by duplicate texts, rejects conflicting
duplicate labels, and fails when it cannot retain every class in each split.
If random candidates miss rare classes, a constrained group assignment checks
coverage directly instead of treating missed samples as proof of infeasibility.
The fallback prioritizes the requested row count and then class balance; genuine
infeasibility and solver failure have distinct errors. Optional metadata fields
are normalized across all records before dataset conversion, so later source IDs
are retained even when the first record has none.
Use at least five independent source groups per class for five partitions; more
may be required for the requested fractions or connected groups.

Augmentation adds stable `seed_id`, `group_id`, `is_seed`, and generator metadata.
`--resume` counts unique accepted variations per seed, including after an interrupted
final write. It retries shortfalls up to three attempts per invocation. Exhausted
attempts leave recoverable output and a failing CLI exit code. Old seed-only files
can resume; legacy generated rows without lineage must be regenerated. Use one
writer per output file. Changing the task or seed/source identities requires a new
output file rather than guessing lineage.

### Compare training approaches

```bash
sentimentizer router benchmark --train-data data/router/split/train.augmented.jsonl \
  --validation-data data/router/split/validation.jsonl \
  --test-data data/router/split/test.jsonl --output-dir models/router-comparison
```

This fits TF-IDF + logistic regression, frozen sentence embeddings + logistic
regression, and the contrastively tuned encoder on identical inputs. It saves
the models and `benchmark_report.json`: per-class precision/recall/F1, macro-F1,
confusion matrices, reliability bins/expected calibration error, accuracy versus
coverage, warmed single-request p95 latency, batch throughput, training time,
and whole-process RSS before/after training. Sequential runs share libraries and
caches: RSS is neither isolated model memory nor GPU memory. TF-IDF is saved as a
sklearn pipeline, not a serving-compatible RouterModel. These are measured local
results; do not use unit-test fixtures to claim production model quality.

Training honors the configured seed for pair sampling, PyTorch initialization,
shuffling and the classifier. `training_mode` selects `contrastive` (default) or
`frozen`; `learning_rate`, `warmup_ratio`, and `head_max_iter` are configurable in
YAML or `SENTIMENTIZER_ROUTER_*` environment variables. The native contrastive
training loop avoids the compatibility `fit()` wrapper's default Trainer seed.
Artifacts record configuration, installed package versions, dataset fingerprints,
and the resolved backbone revision when available. GPU bitwise determinism is not
promised. Validation metrics are saved to `validation_report.json`.

### Calibrate and select an operating point

```bash
sentimentizer router calibrate --model-path models/router \
  --calibration-data data/router/split/calibration.jsonl \
  --policy-data data/router/split/policy.jsonl \
  --min-accuracy 0.95 --min-accepted 20 --output-dir models/router-calibrated
sentimentizer router evaluate --model-path models/router-calibrated \
  --data data/router/split/test.jsonl
```

The head stays frozen while sigmoid probability calibration is fitted on the
calibration file. Threshold selection uses the separate policy file and maximizes
observed coverage subject to the requested accuracy and minimum accepted count.
Equal-confidence inputs stay together. If no threshold qualifies, the saved policy
abstains on every input. This is empirical evidence, not an accuracy guarantee or
an out-of-domain detector. Measure final accepted accuracy and coverage on the
untouched test set and review behavior on challenging and out-of-domain inputs.

All these partitions must be disjoint by source and normalized text. Artifact
metadata retains hashes for all data used in training, validation, calibration,
and policy selection; evaluation rejects known overlap. Hashes detect known
overlap rather than certifying independent data collection. Calibration requires
training provenance, so retrain legacy artifacts first. Legacy artifacts can
still be classified/evaluated, with a warning that independence is unverified.

### Serve explicit decisions

Existing `/v1/router/predict` and `/v1/router/batch` still return their original
prediction shapes. To use the acceptance policy, point `ROUTER_MODEL_PATH` at the
calibrated artifact and call the separate decision API:

```json
{"prediction": {"label": "dietary", "score": 0.72, "token_count": 8,
                "decision": "abstain"}, "latency_s": 0.01}
```

`POST /v1/router/decide` accepts `{"text": "..."}` and is auto-batched.
`POST /v1/router/decide/batch` accepts `{"texts": ["...", "..."]}` and returns
`results`, `count`, and `latency_s`. The topic label stays intact even when the
decision is `abstain`; `general` remains a topic. Artifacts without an acceptance
policy return 503 on decision endpoints while classification remains available.
Standalone Python uses `RouterModel.decide_batch()`; the high-level predictor
adds `decide()` and `decide_batch()` prediction envelopes with token counts.

`GET /health/router` is independent of sentiment readiness. It returns 503 until
a router is loaded and reports `router_loading`, `router_error`, and
`decision_ready`; it never loads/downloads a model. A loaded router without a
policy is classification-ready with `decision_ready: false`.

`POST /v1/router/load` with `{}` warms up the router. Failures remain cached;
`{"retry_failed": true}` explicitly retries a failed attempt. Concurrent calls
join the same attempt within each predictor/replica. Construction runs outside
the state condition, and failures/interruption notify waiters. Each replica owns
its own model; this does not share loaded weights across processes. Python
operators can call `load_router()` and `retry_router_load()` directly.

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
