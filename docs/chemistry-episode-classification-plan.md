# Chemistry agent episode classification

**Status:** 📋 Not started — 2026-10-31

## Goal and user requirements

Classify the chemistry task an agent actually performed, and associate it with
an outcome evidenced by its actions and outputs. The unit is an episode:
prompt, ordered actions/tool results, and optional final response.

The user's example is: replaced a material, searched previous experiments with
that material, generated amounts in a specified probability space, and created
a new formulation. A provisional classification is `material_substitution`
with outcome `new_formulation_created`. The probability-space representation
must be preserved as supplied; no particular distribution is assumed.

This is task/outcome classification. It does not request sentiment analysis,
response-quality grading, or a claim that the new formulation meets physical
performance targets. Creating a formulation is an observable artifact outcome;
performance validation would be a separate outcome with its own evidence.

## Scope and open domain decisions

Initial scope: one primary episode task label, the supplied action sequence,
and an artifact/completion outcome. Keep task type, action type, and outcome
separate. The same substitution task can create a formulation, stop after
retrieval, or fail. Do not train cross-product labels such as
`successful_substitution_with_retrieval_and_sampling`.

Candidate primary tasks for annotation review: `material_substitution`,
`amount_optimization`, `property_targeting`, `experiment_lookup`,
`formulation_generation`, and `other`. These overlap in ordinary language;
the labeling rubric must specify primary-task precedence and mixed-task cases.
The initial action tags are `replace_material`, `search_experiments`,
`sample_amounts`, `create_formulation`, and `other`. Both vocabularies are
provisional until reviewed against real episodes. They are artifact data, not
hard-coded three-class enums.

Initial outcomes: `new_formulation_created`, `candidate_set_created`,
`no_artifact_created`, or `unknown`. Evidence basis is `tool_result`,
`agent_report`, or `none`. A successful tool call without the relevant output
artifact is insufficient for a creation outcome. A failed search or absent
artifact does not establish that a formulation is scientifically infeasible.

Pending before training: the actual agent event format and tool-to-action
mapping, primary-task definitions, what ends an episode, and which tool outputs
prove formulation creation. Do not finalize labels or claim model accuracy
before those are available. No model or implementation change is authorized by
this proposal alone.

Out of scope: replacing the agent, chemistry simulation, numerical formulation
generation, laboratory validation, new chemical structure models, multi-agent
orchestration, automatic model hot reload, and an outcome-prediction model in
the first version.

## Verified repository findings

Verified from local source on 2026-10-31; current branch `extract-imagenate`.
Existing README/API documentation edits are uncommitted and must be preserved.

- `router/model.py::RouterModel` is already a generic SentenceTransformer plus
  sklearn head with saved label metadata. Reuse it with a newly fitted head;
  do not reuse the restaurant head's weights.
- `RouteLabels`, training's saved labels, augmentation seeds/template, and
  evaluation are restaurant-specific. `_build_prompt()` explicitly assumes
  exactly two other categories. `calibrate_threshold()` looks for `general`.
- `dataset.py` randomly splits rows. `save_dataset_to_jsonl()` retains only
  text and integer label, discarding trace/group metadata.
- `train_router()` accepts `eval_dataset` but does not use it. Its backbone
  construction does not apply `RouterConfig.max_seq_length`.
- `workflows/cli.py::router_train()` reconstructs `RouterConfig` using only
  backbone and output directory, losing other loaded training settings.
- The existing predictor has class-column and metadata-shape issues documented
  in [the response plan](router-pydantic-response-plan.md).
- Existing serving accepts text/strings, rather than action trajectories.
  There is no chemistry trace classifier in the inspected source.

The existing three-class sentiment contract remains unchanged. The earlier
response plan's restaurant enum and all-three-categories validation should stay
specific to the legacy restaurant routes; they must not constrain this new
classifier. Share probability and response-validation principles, not its
restaurant label vocabulary.

## Architecture

```text
sentimentizer/router/
  model.py                   REUSE: backbone/head arrays and artifact save/load
  episode_models.py          NEW: shared Pydantic episode/manifest/prediction contracts
  episodes.py                NEW: formatting, outcome resolution, classifier
  episode_training.py        NEW: grouped data, frozen-backbone baseline, evaluation
sentimentizer/serve/
  models.py                  MOD: episode HTTP envelopes using shared core schemas
  config.py, service.yaml    MOD: explicit episode artifact path, disabled by default
  app.py                     MOD: episode routes and one replica-owned classifier
workflows/cli.py             MOD: separate episode train/evaluate subcommands
tests/
  test_router_episodes.py     NEW: contracts, formatting, outcomes, probabilities
  test_episode_training.py    NEW: grouping, fit/evaluation, manifest round trip
  test_serve.py, test_cli.py  MOD: endpoint and CLI contracts
docs/, README.md, AGENTS.md   MOD: documented episode workflow and examples
.rules/dry.md                MOD: recognize shared episode schema ownership
```

Use one shared Pydantic schema module because training and inference both need
the trace contract. Keep HTTP-only envelopes in `serve/models.py`, referencing
the core schemas rather than duplicating them. Core inference must not import
global serving configuration; only the HTTP batch request uses `cfg.max_batch_size`.

## Public interfaces and schema proposal

All new schemas forbid extra fields. Strings identifying episodes, actions,
artifacts, labels, and versions are nonempty. Scores are finite and bounded
`[0, 1]`; latency is finite and nonnegative. Every list default below uses a
factory. Fields without a default are required.

| New schema | Complete fields |
| --- | --- |
| `EpisodeArtifact` | `artifact_id: str`; `kind: Literal["formulation", "candidate_set", "other"]` |
| `EpisodeAction` | `action_id: str`; `action_type: str`; `tool_name: str`; `status: Literal["succeeded", "failed", "pending"]`; `arguments: dict[str, JsonValue] = {}`; `result: dict[str, JsonValue] = {}`; `artifacts: list[EpisodeArtifact] = []`; `depends_on: list[str] = []`; `basis: Literal["tool_result", "agent_report"] = "tool_result"` |
| `EpisodeInput` | `episode_id: str`; `prompt: str`; `actions: list[EpisodeAction]` (at least one); `response: str \| None = None`; `complete: bool = False` |
| `EpisodeOutcome` | `type: Literal["new_formulation_created", "candidate_set_created", "no_artifact_created", "unknown"]`; `basis: Literal["tool_result", "agent_report", "none"]`; `evidence_action_ids: list[str]`; `artifact_ids: list[str]` |
| `EpisodePrediction` | `task_type: str`; `score: float`; `decision: Literal["accepted", "needs_review"]`; `outcome: EpisodeOutcome` |
| `EpisodePredictResponse` | `episode_id: str`; `prediction: EpisodePrediction`; `model: str`; `taxonomy_version: str`; `latency_s: float` |
| `EpisodeBatchRequest` | `episodes: list[EpisodeInput]` (one through existing `cfg.max_batch_size`) |
| `EpisodeBatchResult` | `episode_id: str`; `prediction: EpisodePrediction` |
| `EpisodeBatchResponse` | `results: list[EpisodeBatchResult]`; `count: int`; `model: str`; `taxonomy_version: str`; `latency_s: float` |
| `EpisodeTaxonomy` | `version: str`; `task_labels: list[str]` (unique, at least two, including `other`); `action_types: list[str]` (unique, including `other`) |
| `EpisodeManifest` | `schema_version: Literal["episode-v1"] = "episode-v1"`; `model_id: str`; `taxonomy: EpisodeTaxonomy`; `formatter_version: Literal["episode-text-v1"] = "episode-text-v1"`; `max_input_tokens: int = 512` (positive); `acceptance_threshold: float \| None = None` (finite `[0,1]` when set) |
| `EpisodeTrainingRecord` | `episode: EpisodeInput`; `task_type: str`; `group_id: str`; `parent_seed_id: str \| None = None` |

`JsonValue` is Pydantic's JSON value type. Validate unique action IDs, topological
dependency order, known dependencies, and nonconflicting artifact IDs. Validate
task/action names against the loaded taxonomy. Outcome evidence IDs must resolve
to actions and their artifacts in the supplied episode. Pydantic guarantees
structure; the calling agent's tool adapter supplies evidence provenance. A
caller claiming `tool_result` is not independently verified by this service.

The agent integration should emit normalized actions directly where possible,
retaining tool arguments, result payloads, probability-space constraints, and
artifact references. If a tool is generic or its purpose is unclear, use
`action_type="other"` until an adapter or annotation rule resolves it. Preserve
the original payload rather than inventing an action from its name alone.

New core signatures (existing public method signatures remain unchanged):

```python
def format_episode(episode: EpisodeInput, *, formatter_version: str) -> str: ...
def resolve_episode_outcome(episode: EpisodeInput) -> EpisodeOutcome: ...

class EpisodeClassifier:
    def __init__(self, model: RouterModel, manifest: EpisodeManifest) -> None: ...
    @classmethod
    def from_pretrained(cls, path: str | Path) -> EpisodeClassifier: ...
    def predict_batch(self, episodes: list[EpisodeInput]) -> list[EpisodePrediction]: ...

def load_episode_records(path: str | Path) -> list[EpisodeTrainingRecord]: ...
def split_episode_records(
    records: list[EpisodeTrainingRecord], *, calibration_size: float = 0.2,
    test_size: float = 0.2, seed: int = 42,
) -> tuple[list[EpisodeTrainingRecord], list[EpisodeTrainingRecord], list[EpisodeTrainingRecord]]: ...
def train_episode_classifier(
    config: RouterConfig, records: list[EpisodeTrainingRecord],
    taxonomy: EpisodeTaxonomy,
) -> EpisodeClassifier: ...
def evaluate_episode_classifier(
    classifier: EpisodeClassifier, records: list[EpisodeTrainingRecord],
) -> dict[str, JsonValue]: ...
```

Training initially freezes the backbone, encodes formatted traces, and fits a
new LogisticRegression head. Use the existing RouterConfig backbone,
max-sequence-length, seed, and output directory settings; document that its
contrastive iteration/epoch/batch-size settings do not apply to this baseline.
Calibration/test splitting is separate; the training function receives training
records only. Validate at least two represented task classes and their
taxonomy mapping before fitting. Fine-tuning is a follow-up if baseline errors
justify it, with a separate reviewed configuration change.

Add `router train-episodes --data PATH --taxonomy PATH --config PATH` with an
optional `--acceptance-threshold FLOAT`, and
`router evaluate-episodes --model-path PATH --data PATH`; evaluation uses the
explicit held-out file in full, rather than randomly splitting it again.
Exact Click handler signatures:
`router_train_episodes(ctx: click.Context, data: str, taxonomy: str, config_path: str | None, acceptance_threshold: float | None) -> None`
and `router_evaluate_episodes(ctx: click.Context, model_path: str, data: str) -> None`.
The training CLI saves the train/calibration/test ID manifests; the untouched
test partition is never used for fitting or threshold selection. Default the
acceptance threshold to None. Persist a supplied threshold only with the
calibration report and an explicit reviewed acceptance policy; no automatic
threshold is chosen from the test set. The calibration report should show
coverage/error tradeoffs across candidate thresholds so a later artifact can
carry a justified policy.

Persist the normal RouterModel files plus `episode_manifest.json`. Use its
label-to-ID mapping and `head.classes_` for probability columns; validate class
coverage against this taxonomy rather than RouteLabels. Encode once per batch
and derive task predictions from the head's probability matrix.

`format_episode()` preserves action order, tool status, payload values, and
material identifiers with explicit field boundaries. It excludes supervised
task labels and derived outcome labels. Include prompt and actions, keeping
final agent prose as supplementary context. Use exactly the same formatter in
training and serving. Check the actual backbone token limit and reject oversized
episodes with a clear input error; do not silently drop late creation/failure
actions. A long-trace aggregation design is a follow-up if real traces do not
fit. Sentence Transformers documents truncation beyond its input limit.
[Source](https://sbert.net/docs/package_reference/base/model.html).

Outcome resolution uses completed actions and artifacts, rather than a second
text classifier. A succeeded create action with a formulation artifact yields
`new_formulation_created`; a candidate-set artifact yields
`candidate_set_created`. Prefer tool-result evidence to agent reports and keep
the basis explicit. Pending/contradictory/incomplete evidence yields `unknown`.
Use `no_artifact_created` only for a completed episode with no creation evidence;
the input's `complete` flag distinguishes completed episodes from trace prefixes.
Never equate these outcomes with meeting property targets.

## HTTP and serving integration

Add `POST /v1/router/episodes/predict` and `POST /v1/router/episodes/batch`.
Keep current text-based endpoints and all their response models compatible.
The new structured request uses EpisodeInput directly. Retain generic request
422, unavailable model 503, and internal-output 500 envelopes.

New deployment method signatures:
`async def classify_episodes(self, inputs: list[EpisodeInput]) -> list[EpisodePrediction]`,
`async def episode_predict(self, body: EpisodeInput, request: Request) -> EpisodePredictResponse`,
`async def episode_batch(self, body: EpisodeBatchRequest, request: Request) -> EpisodeBatchResponse`.
Each new handler has one FastAPI registration. Each new classifier/formatter/
loader/trainer/evaluator interface currently has zero repository callers.
Existing train_router, dataset loader, predictor, and restaurant handler
signatures do not change; no caller migration is required.

Add `ServeConfig.episode_model_path: str = ""`, YAML
`episode_model_path: ""`, and `SENTIMENTIZER_EPISODE_MODEL_PATH` in the existing
env override mapping. It is a string, so the existing default string coercion
applies; no non-string `_FIELD_TYPES` entry is needed. Empty disables the new model. Load an explicit
local episode artifact at replica construction when enabled, before publishing
readiness; do not fall back to the restaurant Hub repository. Expose model and
taxonomy version in new responses. No runtime task/model registry is needed.

Use a single @serve.batch method with existing classify batch settings and
`max_concurrent_batches=1`, shared by single and explicit-batch routes. Sync
model inference uses asyncio.to_thread; handlers construct and validate response
models before recording success. Batch results preserve episode order and IDs,
and count must match both results and request length.

Concurrency contract: resource is one episode model per replica; invariant is
that requests cannot observe partial construction or overlap model batches.
Readiness after construction is the publication point; one batch dispatcher
owns inference. Validate ordinary success, load failure, overlapping requests,
and request cancellation while a worker batch completes. Do not release a
serialization boundary while its worker thread is still using the model. No
changes to the existing restaurant lazy loader or synchronization are planned.

For the user's example, a provisional response is below. The score is
illustrative and applies to task classification only. `needs_review` reflects
the default unconfigured acceptance policy; artifact evidence is independently
recorded. The four input actions have IDs a1 through a4, with a4 reporting the
created formulation artifact.

```json
{
  "episode_id": "episode-123",
  "prediction": {
    "task_type": "material_substitution",
    "score": 0.94,
    "decision": "needs_review",
    "outcome": {
      "type": "new_formulation_created",
      "basis": "tool_result",
      "evidence_action_ids": ["a4"],
      "artifact_ids": ["formulation-456"]
    }
  },
  "model": "chemistry-episode-v1",
  "taxonomy_version": "chemistry-v1",
  "latency_s": 0.012
}
```

## Data and evaluation

Start with human-labeled real agent traces covering complete and partial
workflows, failed/retried tools, proposals versus executed actions, and other
chemistry tasks. Include cases where the prompt mentions substitution but
actions only perform retrieval, and where a final response claims creation
without an output artifact. The annotation rubric decides task precedence;
outcome follows observed evidence.

Group train/calibration/test by project/formulation lineage and conversation;
paraphrases and synthetic derivatives inherit the parent's group. Combine
overlapping lineage groups before splitting. Refuse a split that leaves a
training class absent rather than silently inventing training data. Preserve
all metadata in JSONL. Group-based splitting addresses related-example
leakage. [Source](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

Report task macro-F1, per-class precision/recall, confusion matrix, and
acceptance coverage/error rate. Evaluate action/outcome rules separately against
annotated evidence. Choose the acceptance threshold using calibration data and
an agreed tolerated error/coverage tradeoff; `None` means every prediction
requires review until that policy is set. Calibration does not establish
scientific correctness or reliable out-of-domain detection.
[Source](https://scikit-learn.org/stable/modules/calibration.html).

Do not reuse the restaurant `general` fallback, minimum-correct-confidence tau
heuristic, or fixed embedding-similarity targets as acceptance gates. Do not
augment chemistry traces with the Yelp prompt. Introduce domain augmentation
only after annotation review, preserving provenance and validating units,
material identifiers, action ordering, and amount constraints.

## Implementation order and verification

1. **Trace contract and labeling rubric.** Finalize real tool mappings,
   episode completion, taxonomy, Pydantic schemas, formatter, and outcome
   resolver. Add `test_router_episodes.py`: the user's four-action example,
   pending/failed actions, reported-only artifacts, contradictory results,
   references, duplicate IDs, arbitrary JSON payloads, and probability-space
   preservation. Finish with `make ci`.
2. **Training baseline and artifacts.** Add grouped JSONL data and CLI commands,
   frozen-backbone fit, manifest save/load, calibrated acceptance policy, and
   evaluation reports. Add `test_episode_training.py` and CLI tests for grouped
   leakage, unseen/absent classes, deterministic splits, class-column order,
   optional dependencies, one encode per batch, and explicit test data use.
   Finish with `make ci`.
3. **Serving and agent integration.** Add disabled-by-default local loading,
   batched episode endpoints, typed responses, metrics, and trace emission from
   the calling agent. Add HTTP serialization, schema, complete-batch, oversized
   trace, unavailable model, invalid backend, and deterministic concurrency
   tests. Update current docs and this plan's status. Finish with `make ci`.

No training data, live chemistry agent, or live artifact behavior was provided
for empirical evaluation. Performance and final labels are intentionally not
promised. This documentation-only proposal requires `git diff --check`; do not
run the Python suite until implementation changes.

## Review and risk resolutions

First review revised a prompt/response-only design after the user clarified
action-driven task/outcome classification. Second review used the actual
four-action example to distinguish creation evidence from scientific success,
avoid cross-product labels, and remove a premature outcome classifier. Final
review clarified completion flags, dependency references, optional imports,
group leakage, long traces, and inference ownership. An additional revision
separated HTTP-only schemas from the shared core to avoid importing serving
configuration into training, and made the acceptance policy explicit rather
than implicitly selecting an unvalidated threshold. Re-review found no further
architecture blockers; the named domain/annotation decisions remain pending.

| Risk or gap | Resolution |
| --- | --- |
| Task boundaries overlap. | Provisional taxonomy plus primary-task annotation rubric; review real traces before training. |
| Agent prose claims more than tool results show. | Explicit evidence basis, action IDs, artifact IDs, and conservative unknown outcome. |
| Classifier confidence is mistaken for outcome confidence. | Score applies only to task_type; outcome is evidence-resolved and carries provenance. |
| Three-category restaurant assumptions leak into chemistry. | Separate episode interfaces and artifact taxonomy; preserve legacy contracts. |
| Logs expose proprietary formulations. | Use episode/action IDs, counts, versions, latency, and errors; do not log full arguments/results. No external augmentation by default. |
| Model downloads or heavy imports affect existing routes. | Optional dependencies stay guarded; new feature disabled without explicit local artifact. |
| Small or correlated datasets inflate metrics. | Real held-out projects/conversations; parent-group-aware augmentation and separate calibration/test. |
| Long traces lose the action that determines the outcome. | Validate token budget and reject oversized inputs explicitly; evaluate aggregation separately. |
| Sentiment/training/Ray contracts drift. | Existing sentiment classes, schedulers, model factory, metrics files, ONNX capability, and lazy restaurant loading stay unchanged. |

Convention updates during implementation: extend `.rules/dry.md` to recognize
episode_models.py as the shared episode schema owner; document new endpoint and
artifact contracts in AGENTS.md, README.md, router.md, serving.md, and
configuration.md, and add trace/schema/loading diagnostics to troubleshooting.md.
Preserve the previous plan status when updating lifecycle state.
