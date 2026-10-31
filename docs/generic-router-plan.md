# Generic router with food-review defaults

**Status:** 📋 Not started — 2026-10-31

## Goal and scope

Make the existing router reusable for different classification tasks while
keeping the repository's shipped configuration, seeds, examples, and default
model food-related. Use the same implementation with separately trained model
artifacts for food reviews and a possible future chemistry workflow.

This is the current implementation direction. It supersedes the fixed router
enum in [the response plan](router-pydantic-response-plan.md) and defers the
chemistry-specific modules/endpoints in
[the episode proposal](chemistry-episode-classification-plan.md). Neither
chemistry training data nor a chemistry configuration is to be shipped now.

In scope: task metadata, arbitrary label counts, generic probability mapping,
configurable augmentation inputs, task-aware training/evaluation, artifact
compatibility, typed responses, and regression tests with a synthetic non-food
task. Out of scope: a mixed-domain trained head, automatic domain detection,
multiple loaded routers per replica, dynamic model registries, chemistry outcome
rules, episode endpoints, and changes to the three-class sentiment models.

## Source review and assumptions

Verified from local code on 2026-10-31 on `extract-imagenate`:

- RouterModel already separates backbone, head, and saved labels. Its raw
  predict/predict_proba APIs can remain compatible.
- Predictor classification and metadata use RouteLabels rather than loaded
  artifact labels. The augmentation prompt assumes exactly three categories,
  training saves food labels unconditionally, and threshold evaluation assumes
  a category named general.
- RouterConfig is loaded from YAML/env, but the training CLI reconstructs it
  with just base_model/output_dir, discarding other loaded fields.
- Dataset saves currently drop fields other than text/label.
- The default missing-model download always uses the food Hub repository,
  including for custom paths. This must not silently supply a food model for a
  custom deployment.
- Existing response models have the shape/numeric/cardinality gaps documented
  in the original response plan. No live model performance has been measured.

The user wants reuse later, rather than a chemistry deployment now. Assume one
configured router artifact per service instance; external consumers can use
RouterModel directly without loading sentiment weights or the GloVe dictionary.

## Architecture

```text
sentimentizer/router/
  schemas.py                 NEW: shared Pydantic task and classification types
  config.py, config.yaml     MOD: task settings; shipped values remain food
  seeds.py                   KEEP: food defaults
  model.py                   MOD: task sidecar and generic classify_batch
  train_router.py            MOD: use configured labels; apply sequence limit
  augment.py                 MOD: task description and N-class prompts
  dataset.py                 MOD: retain extra dataset metadata
  evaluate.py                MOD: artifact labels; optional configured fallback
sentimentizer/prompts/
  augment_router.txt         MOD: generic template filled from food config
sentimentizer/predictor.py    MOD: delegate router classification and metadata
sentimentizer/serve/
  models.py                  MOD: reuse core prediction type; bounded envelopes
  app.py                     MOD: return validated response objects
workflows/cli.py             MOD: preserve full config; external seeds/config
tests/test_router.py         MOD: food compatibility and arbitrary taxonomies
tests/test_router_responses.py NEW: shared types and producer contracts
tests/test_serve.py, test_cli.py MOD: JSON and configuration integration
README.md, docs/, AGENTS.md, .rules/dry.md MOD: ownership and reuse conventions
```

No dependency upgrade, new model factory, Serve deployment, concurrency
primitive, or lazy-load synchronization change is needed. Keep optional model
dependencies guarded. New schemas must import without sentence-transformers,
Ray, or serving configuration.

## Config and schema contracts

New shared models in router/schemas.py use Pydantic v2. RouterTaskSpec is frozen
and forbids extras. RouterClassification forbids extras and bounds its values.

| Type | Complete fields and defaults |
| --- | --- |
| `RouterTaskSpec` | `task_id: str`; `taxonomy_version: str = "1"`; `description: str = ""`; `labels: tuple[str, ...]`; `label_descriptions: dict[str, str] = Field(default_factory=dict)`; `fallback_label: str \| None = None`; `input_format: str = "text-v1"` |
| `RouterClassification` | `label: str = Field(min_length=1)`; `score: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)` |

Require nonempty task/version/input-format IDs, at least two unique nonempty
labels, description keys drawn from labels, and any fallback_label to belong
to labels. The ordered label list defines integer class IDs. Custom formatting
is performed by the caller, consistently for training and serving; input_format
records that contract rather than importing or executing a formatter from YAML.

Add `RouterConfig.task: RouterTaskSpec`, with a default factory returning a
deep copy of the food task spec so its description dictionary is not shared
between config instances. Put the only literal food label list in the owning config;
derive RouteLabels.label_names/num_classes from it. Preserve RouteLabels,
its existing ID attributes, and the SetFitConfig alias for compatibility.
Core inference/training uses the configured/saved task, not RouteLabels.

Add `AugmentConfig.seed_path: str = ""`; empty selects the bundled food seeds
only for the food task. A custom task requires an explicit seed file when
augmentation is requested. Add SENTIMENTIZER_AUGMENT_SEED_PATH to the router
override table; it uses string coercion. Task fields are supplied through the
existing external YAML --config path; no new task-specific environment-variable
matrix is needed.

The shipped YAML adds this under training, keeping all current training and
augmentation settings:

```yaml
task:
  task_id: food_reviews
  taxonomy_version: "1"
  description: Classify restaurant reviews by topic.
  labels: [dietary, service, general]
  label_descriptions:
    dietary: Allergies, dietary restrictions, and ingredient suitability.
    service: Staff, waiting, reservations, and service experience.
    general: Food quality, ambiance, price, and other review topics.
  fallback_label: general
  input_format: text-v1
```

load_router_config keeps its return signature, converts the nested task dict
to RouterTaskSpec, and accepts older food YAML without a task section. The CLI
uses dataclasses.replace for explicit overrides instead of reconstructing a
partial RouterConfig. No production chemistry config or seeds are added.
Document `seed_path: ""` under the shipped augmentation section; existing files
without that field retain the same bundled food seeds.

## Interfaces and call sites

New interfaces:

```python
RouterTaskSpec.validate_task(self) -> Self  # after model_validator
RouterModel.classify_batch(self, texts: list[str]) -> list[RouterClassification]
```

`validate_task` implements the task invariants above. classify_batch has zero
current consumers and will be called by SentimentPredictor.classify_batch and
available directly to external users. It requires explicit task metadata or
unambiguous saved labels; it never invents food labels for unknown artifacts.

Modified signatures (counts are executable current production sites, plus
tests; method registration counts are noted separately):

| Interface | Before → after; call sites |
| --- | --- |
| `RouterModel.__init__(self, backbone: Any, head: Any \| None = None, labels: list[str] \| None = None) -> None` | Add keyword-only `task: RouterTaskSpec \| None = None`; 1 production constructor, 5 in test_router.py; internal cls constructors in loading also propagate metadata. |
| `_build_prompt(seed_text: str, seed_label: int, num_variations: int) -> str` | Add keyword-only `task: RouterTaskSpec \| None = None`; 1 production call, 1 test. None preserves the default food spec. |
| `augment_seeds(seeds: list[dict], model: str = "glm-5.1:cloud", ollama_url: str = "http://localhost:11434/api/generate", variations_per_seed: int = 50, batch_size: int = 5, output_path: str \| None = None, resume: bool = False) -> list[dict]` | Add keyword-only `task: RouterTaskSpec \| None = None`; 1 production call, 5 tests. |
| `_augment_loop(seeds: list[dict], augmented: list[dict], model: str, ollama_url: str, variations_per_seed: int, batch_size: int, out_file: object \| None = None) -> list[dict]` | Add keyword-only `task: RouterTaskSpec \| None = None`; 3 production calls, no direct tests. |
| `calibrate_threshold(model: RouterModel, eval_dataset: dict, label_names: dict[int, str] \| None = None) -> float` | Add keyword-only `fallback_label: str \| None = None`; return `float \| None`. One production caller in evaluate_router. Select the configured fallback; no fallback means no legacy tau recommendation. |
| `router_predict(self, body: PredictRequest, request: Request)` | async return dict[str, Any] → RouterPredictResponse; one HTTP registration, one explicit direct test call. |
| `router_batch(self, body: BatchRequest, request: Request)` | async return dict[str, Any] → RouterBatchResponse; one HTTP registration, one explicit direct test call. |
| `router_models(self)` | async return dict[str, Any] → RouterModelsResponse; one HTTP registration, no explicit direct test call. |

Unchanged signatures with behavior changes: load_router_config (production
train/augment consumers), train_router (one production consumer),
load_router_dataset (two production/two test consumers), save_dataset_to_jsonl
(test coverage), evaluate_router (one production consumer), predictor
classify_batch (three production/two test-helper calls), get_router_model_info
(one production consumer), and raw RouterModel predict/predict_proba/save/load.

## Artifact, prediction, and response behavior

Persist task metadata in router_config.json alongside its existing labels,
model_type, and head_type fields. Keep task.labels and the legacy labels array
consistent: derive labels from task when the constructor omits labels, and reject
an explicitly conflicting list. Publish sidecar writes atomically. Require head.classes_ to cover
the task's classes uniquely and reject conflicting/missing mappings. Generic
classification encodes once, gets predict_proba from the head, chooses its
highest probability, and resolves the column through classes_ and saved labels.
Reject nonfinite/out-of-range probabilities, malformed dimensions, and partial
batches. A loaded empty Python batch returns an empty list without encoding.

Old artifacts with saved labels remain usable; canonical legacy food artifacts
without labels can use an explicit food compatibility path. Unknown integer
classes without label metadata fail clearly. Optional task_id/version metadata
may be absent for legacy artifacts; never silently assign an unrelated task.
Custom missing artifact paths must fail instead of downloading the default food
model. Preserve automatic food Hub download for the default food path only.

RouterPrediction inherits RouterClassification and adds the existing
nonnegative strict-integer token_count. Keep label a string: validate membership
against the loaded artifact in classify_batch instead of a food-only enum.
Keep the existing single/batch JSON envelopes. RouterModelsResponse.categories
is an ordered list from artifact metadata; add optional
`task_id: str | None = None` and `taxonomy_version: str | None = None` fields
for model provenance. These metadata fields are additive; existing prediction
keys remain unchanged.

Before lazy model loading, metadata reads only the local sidecar when present;
it must not load the backbone or initiate a download. If metadata is unknown,
report categories=[] and null task/version rather than guessing food categories.
The configured default food compatibility path can identify the default task
without loading weights. Preserve loaded/not_loaded status and cached errors.

Apply finite/nonnegative latency, count invariants, extra-field rejection, and
typed handler returns from the response plan. Validate before recording success.
Do not mutate global OpenAPI schemas or Pydantic enums on model load. The generic
OpenAPI label is a nonempty string; /v1/router/models supplies the actual labels.

Augmentation receives the task description, label descriptions, and the full
list of competing labels. Remove other_category_1/other_category_2 assumptions.
Preserve dataset metadata, validate labels at training, and avoid embedding
supervised labels in the input text. Reject boolean, fractional, negative, or
out-of-range class IDs before integer coercion. Apply configured max_seq_length. Evaluation
derives explicit label IDs and names from the artifact, including absent test
classes, and does not require a category named general. The retained food tau
helper is a historical heuristic, not a calibrated acceptance guarantee; tasks
without a fallback return tau_threshold=null and still get classification
reports and confidence statistics.

## Future chemistry use

An external chemistry config/data set can train another artifact through the
same API. Its integration formats prompt/actions/results into stable text and
sets an appropriate input_format version. A different label set needs a newly
trained head, and potentially a fine-tuned backbone if evaluation warrants it;
switching YAML does not convert food weights into a chemistry classifier.

For the earlier example, the external task might label the action sequence as
material_substitution. Its integration can attach new_formulation_created using
the creation action and artifact reference. Outcome/evidence schemas remain a
later adapter concern. The generic food response does not acquire chemistry
fields, and this change does not add chemistry modules or routes.

## Implementation order and testing

1. **Task schema, food defaults, and artifacts.** Add shared models, nested
   configuration, compatible artifact loading/saving, and arbitrary-label
   prediction. test_router.py and test_router_responses.py cover old food
   config/artifacts, class order, two/five-label synthetic tasks, duplicate or
   unknown labels, malformed probabilities, and import without optional extras.
   Finish with make ci.
2. **Training, augmentation, and evaluation.** Preserve config overrides,
   propagate task/seed inputs, generalize prompts and label reports, and retain
   dataset metadata. Test N-class prompts, external seeds, missing custom
   seeds, config precedence, sequence limit application, absent evaluation
   classes, no-general tasks, and no food fallback for missing custom artifacts.
   Finish with make ci.
3. **Typed serving and documentation.** Delegate predictor classification,
   return validated response models, fix metadata arrays, and document reuse.
   Test existing food HTTP envelopes, non-food labels from a fake artifact,
   bounded values, complete batches, lazy metadata, cached load failures, and
   errors through TestClient. Keep existing lazy-load regressions in test_hf.py.
   Finish with make ci; update all affected plan statuses with the code.

No classifier service interface changes require new threads or shared
registries. Keep existing lazy initialization and async batching boundaries;
metadata reads observe atomically published sidecars and do not mutate models.
Use deterministic tests if implementation exposes a lifecycle interaction.

## Review, risks, and convention updates

Review 1 removed chemistry-specific endpoints and the fixed enum from the
shared design. Revision made training/runtime labels artifact-owned, kept
food-specific defaults in config, and preserved external JSON prediction shape.
Review 2 identified fallback downloads, partial CLI config overrides, pre-load
metadata guessing, and the general-only evaluator; the above resolutions are
explicit. Re-review also retained legacy model APIs and separated schema
ownership from serving configuration. A further revision clarified deep-copy
defaults, explicit task/labels conflicts, and validation before class-ID
coercion. Final re-review found no remaining implementation-blocking gaps in
the generic core; actual chemistry labeling and evidence rules remain deferred.

| Risk | Resolution |
| --- | --- |
| Configuration changes are mistaken for retrained models. | Persist task metadata with weights; validate class mappings and reject incompatible artifacts. |
| Food behavior regresses. | Keep food YAML, seed examples, default artifact path, IDs, compatibility aliases, and regression coverage. |
| A custom task quietly receives food labels/data. | Artifact-derived labels; explicit custom seeds; no automatic food download for custom paths. |
| Pydantic schema becomes task-specific. | Static generic schema plus model-bound label membership validation and metadata discovery. |
| Single-model reuse grows into a model registry. | One router artifact per instance; external chemistry deployment later. |
| Task refactor affects sentiment, scheduler, metrics, ONNX, or Ray lifecycle. | Keep these contracts unchanged; full make ci at each code milestone. |

During implementation update README.md, router.md, serving.md,
configuration.md, troubleshooting.md, and AGENTS.md with food defaults versus
generic runtime behavior. Extend .rules/dry.md to name router/schemas.py as the
shared task/classification schema owner; HTTP envelopes remain in serve/models.py.
Documentation-only planning verification is git diff --check; no Python suite.
