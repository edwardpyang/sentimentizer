# Router Pydantic JSON response plan

**Status:** ✅ Completed — 2026-10-31

_Original status: 📋 Not started — 2026-10-31_

**Implemented direction (2026-10-31):** The
[generic router plan](generic-router-plan.md) implements these response changes
with artifact-defined labels and food-review defaults, replacing the fixed
RouteCategory enum and mandatory three-label mapping below. Response envelopes,
bounds, batch checks, metadata correction, and confidence mapping are implemented
and tested. The final `make ci` passed: 796 tests passed, one skipped, zero
pyright errors, and successful package builds. This document retains the
original review; current contracts are in [the router guide](router.md).

## Goal

Make the intent router's existing JSON responses match explicit, validated
Pydantic v2 models. Preserve the v1 response envelopes and the predictor's
dictionary API while correcting producer/schema mismatches and confidence
selection.

## Current-code review

Verified against the local checkout on 2026-10-31, on `main` with an initially
clean worktree. These are static code findings, not a live deployment audit.

| Priority | Finding | Evidence and consequence |
| --- | --- | --- |
| High | Router metadata violates its response schema when router dependencies are available. | `predictor.py::get_router_model_info()` assigns `RouteLabels.label_names()` (a `dict[int, str]`) to `categories`; `serve/models.py::RouterModelsResponse` requires `list[str]`. FastAPI response validation can turn `GET /v1/router/models` into a 500. The mock in `tests/test_serve.py` already returns a list, hiding the discrepancy. |
| Medium | Confidence selection assumes class IDs equal probability-column positions. | `predictor.py::classify_batch()` indexes `probabilities[i, label_idx]`. Sklearn's installed `LogisticRegression.predict_proba()` documents columns in `head.classes_` order. A string prediction outside the canonical labels silently selects column zero; noncanonical or incomplete heads can produce wrong scores or indexing failures. |
| Medium | A short model result silently truncates the response. | `classify_batch()` uses `zip(texts, predictions, strict=False)` and does not validate probability dimensions. A partial batch can be reported as successful. |
| Medium | Existing response models permit invalid domain values. | `RouterPrediction.label` is any string; score, token count, count, and latency lack bounds. Metadata status is any string. There is no batch-count invariant. |
| Medium | Validation happens after handlers record success. | Router handlers return ad hoc dictionaries, use `.get(..., {})`, and record successful requests before FastAPI applies `response_model`. Malformed output can increment success metrics before becoming an HTTP 500. |

The former SetFit implementation is now `RouterModel`: a SentenceTransformer
backbone and sklearn LogisticRegression head. `SetFitConfig` remains an alias,
and the loader can read legacy SetFit artifact filenames. No SetFit dependency
or replacement classifier is needed for this work.

The single and batch router routes already declare Pydantic response models
and already return JSON. The change is schema enforcement and producer
alignment, rather than adding JSON serialization to an unstructured endpoint.

## Scope

In scope:

- The success responses of `/v1/router/predict`, `/v1/router/batch`, and
  `/v1/router/models`.
- Canonical route labels, numeric constraints, metadata categories, class-column
  mapping, and complete batch output.
- Explicit Pydantic objects returned by the HTTP handlers, with validation
  inside their existing request error-accounting boundary.
- Unit, HTTP serialization, OpenAPI, and lazy-load regression coverage.

Out of scope:

- Retraining, changing categories or thresholds, replacing the backbone/head,
  changing saved artifact formats, or reintroducing the SetFit library.
- Adding all-class scores, `text`, `category`, or `model` fields to prediction
  JSON; changing the sentiment endpoints or their response schemas.
- Changing router request models, their currently accepted `model` field,
  error envelopes, middleware, lazy-load ordering, or concurrency controls.
- Combining `predict()` and `predict_proba()` into one encoder pass. The current
  double encoding is a separate performance concern.

## Architecture

```text
sentimentizer/
  router/config.py           MOD: RouteCategory enum; preserve RouteLabels API
  predictor.py               MOD: metadata list and reliable prediction mapping
  serve/models.py            MOD: strengthen the existing router response models
  serve/app.py               MOD: construct and return validated response objects
tests/
  test_router.py             MOD: label source and model compatibility coverage
  test_router_responses.py   NEW: schema and predictor response contract coverage
  test_serve.py              MOD: real-handler HTTP serialization and failures
  test_hf.py                 existing lazy-load regression tests retained
docs/
  router.md                 MOD: Python and HTTP response contract
  serving.md                MOD: complete single/batch/metadata JSON examples
  troubleshooting.md        MOD: router response validation diagnostics
AGENTS.md                   MOD: document typed router HTTP responses
```

Keep Pydantic schemas in `serve/models.py`, as required by `.rules/dry.md`.
The predictor continues returning dictionaries and does not import the serving
layer or its global `cfg`. `router/config.py` is lightweight and can be imported
by the schema without importing sentence-transformers; `router/__init__.py`
and `router/model.py` currently keep heavyweight model imports lazy.

No dependency, dataclass setting, YAML field, environment override, model
factory, or training loader change is required. Existing serving limits and
batch settings remain authoritative.

## Public interfaces and schemas

### Label source

Add `class RouteCategory(StrEnum)` in `router/config.py` with members
`DIETARY = "dietary"`, `SERVICE = "service"`, and `GENERAL = "general"`.
Derive the existing integer-to-string mapping from these members, preserving
`RouteLabels.label_names(cls) -> dict[int, str]` and IDs 0, 1, 2 exactly.
Derive `RouteLabels.num_classes(cls) -> int` from the canonical mapping.
Neither method's signature changes. Do not mix these route categories with
the sentiment model's negative/neutral/positive labels.

### Pydantic responses

Add a private `class _RouterResponseModel(BaseModel)` in `serve/models.py`
with `model_config = ConfigDict(extra="forbid")` and no fields. All five
existing router response classes inherit it. Preserve their class names,
module paths, descriptions, and examples. The complete field contracts are:

| Model | Fields; fields without an explicit default are required |
| --- | --- |
| `RouterPrediction` | `label: RouteCategory`; `score: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)`; `token_count: int = Field(strict=True, ge=0)` |
| `RouterPredictResponse` | `prediction: RouterPrediction`; `latency_s: float = Field(strict=True, ge=0, allow_inf_nan=False)` |
| `RouterBatchResultItem` | `prediction: RouterPrediction` |
| `RouterBatchResponse` | `results: list[RouterBatchResultItem]`; `count: int = Field(strict=True, ge=0)`; `latency_s: float = Field(strict=True, ge=0, allow_inf_nan=False)` |
| `RouterModelsResponse` | `model_path: str`; `categories: list[RouteCategory]`; `status: Literal["loaded", "not_loaded"]`; `error: str \| None = None` |

Add `RouterBatchResponse._validate_count(self) -> Self`, decorated with
`@model_validator(mode="after")`, raising `ValueError` unless
`count == len(results)`. The HTTP handler separately checks that result count
equals request text count; a response schema alone cannot verify that.

Unknown fields and invalid labels are errors rather than silently dropped
output. Retain zero as a valid token count: whitespace/punctuation can pass
existing request validation but yield no regex tokens. Numeric strings and
booleans must not become counts or scores through coercion. Reject NaN and
infinities. These are response-only constraints; do not change the extra-field
policy of the shared request models.

### Methods and call-site inventory

Counts below are executable production call sites in the current checkout;
docstring examples and tests are recorded separately. Calls passed as arguments
to `asyncio.to_thread` are included.

| Method | Before → after | Production consumers; tests |
| --- | --- | --- |
| `SentimentPredictor.classify(self, text: str)` | `-> dict[str, Any]` unchanged | 0 production calls; 2 calls in `test_hf.py`; 1 docstring example |
| `SentimentPredictor.classify_batch(self, texts: list[str])` | `-> list[dict[str, Any]]` unchanged | 3: `classify`, `classify_route`, `router_batch`; 2 calls in `test_serve.py` helper plus its mocked return value |
| `SentimentPredictor.get_router_model_info(self)` | `-> dict[str, Any]` unchanged | 1: `router_models`; mock in `test_serve.py` |
| `SentimentizerDeployment.classify_route(self, inputs: list[dict])` | `async -> list[dict]` unchanged | 1: `router_predict`; replaced by helper in serving tests |
| `SentimentizerDeployment.router_predict(self, body: PredictRequest, request: Request)` | `async -> dict[str, Any]` → `async -> RouterPredictResponse` | 1 registered FastAPI route; 1 explicit direct test call |
| `SentimentizerDeployment.router_batch(self, body: BatchRequest, request: Request)` | `async -> dict[str, Any]` → `async -> RouterBatchResponse` | 1 registered FastAPI route; 1 explicit direct test call |
| `SentimentizerDeployment.router_models(self)` | `async -> dict[str, Any]` → `async -> RouterModelsResponse` | 1 registered FastAPI route; no explicit direct test call |

No `RouterModel.predict()` or `predict_proba()` signature change is planned;
training and evaluation callers continue receiving arrays.

### Producer behavior

`get_router_model_info()` returns ordered string values from the canonical
mapping, rather than the mapping itself. Preserve `status="not_loaded"` before
first use and after a cached load failure, with the existing optional error.
Preserve `categories=[]` when the router extra is unavailable. Metadata never
triggers a load or download.

For `classify_batch()`:

1. Preserve the existing lazy-load/error check. After that check, return `[]`
   for an empty Python batch without calling the backend; HTTP empty batches
   continue receiving Pydantic 422 errors.
2. Obtain predictions and probabilities through the existing model methods.
   Normalize array/list containers without changing their public APIs.
3. Require one prediction and probability row per input, and probability
   columns matching `router.head.classes_`. Require the head's classes to map
   uniquely to all three canonical categories. Reject missing, duplicate,
   unknown, boolean, or nonintegral class identifiers explicitly.
4. Map integer class IDs through `RouteLabels`; accept canonical string class
   names for legacy heads. Find each predicted class's probability column in
   `head.classes_`, never by integer position or a fallback to column zero.
   Reject predictions absent from the head's classes. Validate any nonempty
   artifact label metadata against the canonical ID order; legacy artifacts
   without label metadata can use the canonical mapping.
5. Reject malformed dimensions and nonfinite/out-of-range probability values;
   do not normalize or repair model output. Assemble exactly one existing
   `{"prediction": {...}}` dictionary per input in order, converting NumPy
   values to native `str`, `float`, and `int`.

These failures represent invalid server/model output. They remain server
errors, not client 422 responses. Model-load failures retain their current 503
behavior. A head trained on only a subset of categories is explicitly
unsupported by this three-category serving contract and must fail rather than
receive fabricated scores. No new public helper functions are required.

### JSON boundary

The single handler validates its auto-batched dictionary using
`RouterBatchResultItem.model_validate(raw_result)`, then constructs a
`RouterPredictResponse`. The batch handler constructs `RouterBatchResponse`
from its results and verifies request/result cardinality. The metadata handler
returns `RouterModelsResponse.model_validate(info)`.

For prediction handlers, perform response construction and validation inside
the existing `try` block, before recording success and logging completion.
Validation failures use existing error metrics and the generic 500 envelope.
Use typed attributes for logging instead of `.get()` fallbacks. Keep error
logging free of request bodies and raw Pydantic input values.

Retain each route's explicit `response_model`. Return the Pydantic object and
let FastAPI serialize JSON; do not return `.model_dump_json()` strings, which
would risk JSON string responses. Python code needing JSON-compatible
dictionaries can call `.model_dump(mode="json")` on a response object.

The existing successful single response remains:

```json
{
  "prediction": {"label": "dietary", "score": 0.95, "token_count": 8},
  "latency_s": 0.0031
}
```

Batch remains `{"results": [{"prediction": {...}}], "count": 1,
"latency_s": 0.0031}`. Metadata uses an ordered categories array and retains
HTTP serialization of `error: null` when no error exists. Do not enable
`response_model_exclude_none` as part of this change.

## Implementation order

1. **Canonical labels, schemas, and metadata correction.** Add `RouteCategory`,
   derive the legacy mapping without changing its API, strengthen the five
   schemas, and correct metadata category production. Add schema and actual
   predictor metadata tests in `test_router_responses.py`, plus mapping
   regression tests in `test_router.py`. End with `make ci`.
2. **Prediction producer correctness.** Correct class-column selection,
   invalid-class handling, dimensions, finite probability checks, cardinality,
   native scalar conversion, and empty Python batch behavior. Add deterministic
   predictor tests with fake heads/backbones, including legacy string classes.
   Do not download models or alter loading synchronization. End with `make ci`.
3. **Typed HTTP responses and documentation.** Return the existing Pydantic
   response classes from the three handlers; include construction in prediction
   error accounting. Add HTTP/OpenAPI tests and update current docs/conventions.
   Preserve and run lazy-load regression coverage in `test_hf.py`. End with
   `make ci`, and update this plan's status in the same implementation change.

Each milestone is a commit-sized unit containing its own tests. Do not proceed
past a failing gate. This planning-only change needs `git diff --check`, not
the Python suite.

## Testing strategy

| Module | Scenarios |
| --- | --- |
| `tests/test_router_responses.py` (new) | Round-trip single/batch/metadata through `model_validate`, `model_dump(mode="json")`, and JSON serialization; all three labels; unknown labels; extras; score endpoints 0/1; NaN/infinity; negative/fractional/bool counts; numeric strings; batch count mismatch; zero tokens and empty internal results. Test actual predictor metadata with router available/unavailable and loaded/not-loaded/cached-error states without constructing real models. |
| `tests/test_router_responses.py` (new) | Predictor results for canonical integers, canonical string classes in sklearn sort order, reordered integer classes, missing/unknown/duplicate classes, incompatible artifact labels, wrong dimensions, short/extra outputs, empty inputs, invalid probabilities, native scalars, and one-output-per-input ordering. Use fake model/head objects and deterministic arrays. |
| `tests/test_router.py` | Preserve IDs and exact `dict[int, str]` mapping; derive enum values consistently; retain existing raw model array/save/load behavior and SetFitConfig alias tests. |
| `tests/test_serve.py` | Invoke the real router handlers through thin TestClient adapters with the actual response models; assert `application/json`, object bodies, single/batch field parity, metadata array, `error: null`, and OpenAPI enum/bounds/required fields. Include malformed backend output and partial batches yielding generic 500 plus error metrics, and unavailable router yielding 503. Use the real metadata producer in at least one integration case rather than only the existing list-valued mock. |
| `tests/test_hf.py` | Existing first-classify download and cached-failure behavior stays green; no eager load or metadata-triggered downloads. |

Use the app factory and actual generic error handler in HTTP failure tests;
`TestClient(..., raise_server_exceptions=False)` is needed to inspect 500
bodies. Direct handler tests alone do not exercise response serialization.
Schema and fake-model tests must run without the router extra. Existing
optional model integration tests may retain their dependency skips. Fake-model
tests must explicitly inject the canonical `RouteLabels` binding as well as
router availability when the optional import guard did not define that binding;
they must not depend on the developer's installed extras.

## Risk review and revision record

Review pass 1 identified that moving schemas into the predictor would import
serving configuration into inference, that enum literals could duplicate the
canonical label source, and that HTTP response validation occurs too late for
correct success metrics. The revised design keeps schemas at the serving
boundary, derives label mappings from one enum, and validates before success
accounting. It also adds request/result cardinality checks beyond the batch
schema's own count invariant.

Review pass 2 checked compatibility, optional imports, legacy heads, metadata
states, and test isolation. It clarified that incomplete-category heads fail
explicitly, strict extra-field handling applies only to responses, and predictor
tests must supply bindings absent behind the optional import guard. A final
re-review found no remaining implementation-blocking gaps. The resulting
decisions are:

| Risk | Resolution |
| --- | --- |
| Existing clients depend on the v1 envelope. | Keep keys, nesting, score meaning, token counting, latency rounding, and nullable metadata error. Do not add output fields. |
| Python callers depend on dictionaries. | Preserve predictor and auto-batch return signatures; only HTTP handlers return Pydantic objects. |
| Strict schemas expose invalid historic artifacts. | Support canonical integer and string classes and missing legacy label metadata. Fail explicitly for incompatible classes/metadata; document retraining or artifact correction. Do not guess scores. |
| Partial outputs violate Ray batching expectations. | Validate full cardinality in the producer and request cardinality at the HTTP boundary; fail the batch through the existing exception path. Never return partial success or error dictionaries that violate the success schema. |
| Schema imports accidentally require heavy optional dependencies. | Import the lightweight enum/config only; verify schema imports without sentence-transformers and preserve model import guards. No new dependencies. |
| Response validation is mistaken for client input validation. | Preserve Pydantic request 422 behavior. Invalid backend data is generic 500; unavailable model remains 503. |
| Metrics report validation failures as successes. | Construct prediction response models inside existing `try` blocks before the success increment; assert exactly one error increment in tests. |
| Concurrency or Ray lifecycle scope expands. | Preserve lazy loading, batching decorators/settings, `asyncio.to_thread`, ingress factory, middleware, module-level Ray environment settings, and all synchronization. No new deployment or lifecycle behavior. |
| Training, sentiment classes, schedulers, metrics files, ONNX, or model registry drift. | These interfaces are unchanged. Route categories stay separate from sentiment classes; full `make ci` remains the regression gate. |

No unresolved implementation dependency requires a new package, model download,
training run, or live service. A live trained artifact was not available under
`models/router` in this checkout, so this review does not claim deployed model
accuracy or artifact compatibility has been measured.

## Assumptions

- The requested “SetFit model” refers to the current intent router and its HTTP
  prediction/metadata responses; the code and current router documentation
  identify that replacement explicitly.
- Existing v1 JSON shape should remain compatible. Invalid backend values may
  now fail deliberately instead of being coerced, truncated, or mis-scored.
- Local installed FastAPI 0.136.1 and Pydantic 2.13.3 support the planned
  response validation/serialization APIs; verified from installed
  `fastapi/routing.py` and `pydantic/main.py` on 2026-10-31. No version upgrade
  is part of the plan.
- Installed sklearn source orders probability columns by `classes_`.
  `router/train_router.py` fits the head on integer labels and saves canonical
  label names. Legacy migration permits string heads, so both need tests.
- Static repository call-site counts exclude external consumers. Preserving
  predictor signatures avoids requiring an external Python migration.

## Convention updates

- `AGENTS.md`: retain the dictionary `classify_batch()` contract; document
  typed router HTTP handlers, bounded scores/counts, and metadata categories
  as an array. Clarify the existing “response dicts built directly” statement
  for these three handlers.
- `docs/router.md`: describe Python dictionaries versus HTTP Pydantic models,
  label vocabulary, confidence semantics, and invalid-artifact failures.
- `docs/serving.md`: add full batch and metadata JSON examples and correct its
  stale statement that the router loads at startup.
- `docs/troubleshooting.md`: document metadata schema mismatches, invalid class
  mappings, and partial output as server errors. Avoid prescribing partial or
  success-shaped fallback output for router batches.
- `.rules/`: no rule change needed; schema ownership remains as documented.
- Keep this plan marked not started until implementation begins; preserve its
  previous status as an italic original-status line when updating it.
