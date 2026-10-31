# Router reliability, evaluation, and decisions

**Status:** ⏳ Partially implemented — 2026-10-31

_Original status: 📋 Not started — 2026-10-31_

Milestone 1 is implemented. Focused tests: 68 passed. `make ci`: 728 passed,
2 skipped; lint/type checks and wheel/sdist builds passed. Local tests use the
existing environment with PYTHONPATH pointing to this managed worktree.

Milestone 2 is implemented. `make ci`: 743 passed, 1 skipped; lint/type checks
and wheel/sdist builds passed. Tests include real frozen-head calibration,
joblib persistence, CLI calibration, all three benchmark result paths, and a
tiny offline trainable sentence encoder trained twice and reloaded. Source
inspection found fit() resets the training seed through default Trainer args;
the native old_fit() loop preserves the explicitly configured RNG seed.

## Goal and scope

Implement the five router review recommendations on main at d5396bd. Preserve
existing classification responses. Add explicit decision responses, independent
evaluation workflows, and reproducible comparison tooling. Acquiring and labeling
production reviews and choosing a production operating point remain operator work;
the PR must not invent measurements or represent synthetic examples as human data.

## Architecture and public interfaces

- MOD router/dataset.py: group-disjoint splits, normalized duplicate detection,
  dataset fingerprints and overlap validation. Keep load_router_dataset(data_path:
  str = "augmented_yelp.jsonl", test_size: float = 0.2, seed: int = 42) unchanged
  (one production CLI caller; two existing test callers).
- MOD router/augment.py: stable seed lineage, accepted-count resume, bounded
  retries and deduplication. Keep augment_seeds's existing positional arguments;
  add max_attempts: int = 3 as a keyword-only option (one production CLI caller).
- MOD router/train_router.py: use supplied validation data, record reproducible
  training provenance and evaluation results. Preserve train_router(config:
  RouterConfig, train_dataset: Any, eval_dataset: Any = None) -> RouterModel
  (one production CLI caller).
- NEW router/calibration.py: calibrate_router(model: RouterModel,
  calibration_records: list[dict], policy_records: list[dict], *, min_accuracy:
  float = 0.95, min_accepted: int = 20) -> RouterModel. Freeze the trained head,
  fit sigmoid probability calibration, then select a threshold on disjoint policy
  records. A failed operating target means abstain-all, not a relaxed target.
- NEW router/benchmark.py: benchmark_router(config: RouterConfig, train_records:
  list[dict], validation_records: list[dict], test_records: list[dict], *,
  repeats: int = 5) -> dict. Compare TF-IDF, frozen encoder, and contrastive encoder
  on the same data; report F1/recall/confusion, latency, process RSS, artifacts,
  configuration and versions. No inference that RSS is isolated model memory.
- MOD router/model.py and schemas.py: optional provenance and acceptance policy
  in the artifact. Existing classify_batch(texts: list[str]) stays compatible;
  add decide_batch(texts: list[str]) -> list[RouterDecision]. RouterDecision
  extends label/score with decision: Literal["accept", "abstain"]. Uncalibrated
  artifacts reject decision requests explicitly.
- MOD predictor.py: per-instance condition protecting lazy-load state;
  retry_router_load() -> bool is an explicit operator retry. Add decide_batch
  and decide with the same prediction envelopes as classification.
- MOD serve/app.py and models.py: POST /v1/router/decide and /decide/batch,
  GET /health/router (read-only; never initiates a download). Existing routes
  and metadata response remain compatible.
- MOD workflows/cli.py: router split, train --validation-data, augment --seeds,
  calibrate, benchmark; evaluation rejects known training/validation overlap.
- MOD Makefile/docs/router.md/AGENTS.md/tests: independent file workflow,
  annotation policy, compatibility, concurrency, CLI and artifact coverage.

## Implementation order and verification

1. Dataset splitting, augmentation recovery, CLI split and independent pipeline.
   Run focused regressions and make ci before the next milestone.
2. Training validation, provenance, evaluation metrics, calibration/decision
   artifacts, and comparison CLI. Run focused regressions and make ci.
3. Concurrent loading/retry, decision HTTP API, router readiness, final docs.
   Run deterministic race tests alone, HTTP tests, and make ci.

## Concurrency ownership and race review

Resource: each predictor's router/error/loading state. Invariant: exactly one
construction per attempt; waiters observe the same terminal state. Sharing scope:
threads within one predictor/Serve replica. Different replicas intentionally own
different model objects. Hugging Face owns cache/download file locking. The
linearization point is publication of the complete result under the condition.
Never hold the condition during loading or waiting on network operations.

| A | B | Risk | Resolution |
|---|---|---|---|
| first load | first load | duplicate construction | one owner; others wait |
| failed load | waiter | stuck waiter or accidental retry | publish error and notify in all paths |
| retry | first load/retry | reset an in-flight attempt | join active attempt; reset only terminal failure |
| readiness | loading | false ready | readiness checks published model only |

## Review, revision, and risks

First review found two sources of leakage: paraphrase families and exact duplicate
texts under different source IDs. Revised splitting to keep connected groups
together and enforce overlap checks at train/calibrate/benchmark boundaries.
Second review found that fitting calibration and choosing thresholds on final
test data would bias results. Revised the CLI to require separate calibration
and policy-selection files, with training fingerprints retained in artifacts.

| Risk | Resolution |
|---|---|
| Old classification clients | Preserve their exact response shape; separate decision API |
| Few groups/classes | Fail with an actionable error instead of leaking groups |
| Legacy augmentation without lineage | Reject ambiguous resume; regenerate from seeds |
| Insufficient acceptance evidence | Minimum accepted count; abstain-all when unmet |
| Legacy artifacts lack provenance | Reject calibration without retraining provenance; warn on evaluation |
| Optional dependencies | Keep calibration/benchmark imports at command boundaries |
| Empirical claims | Smoke-test with controlled data only; no production quality claim |
| Sentiment/Ray contracts | No changes to sentiment class count, model factories, Ray lifecycle |

## Assumptions and convention updates

Verified 2026-10-31: sklearn in the local environment supports FrozenEstimator
and CalibratedClassifierCV. Existing model artifacts use joblib. No trained router
is present locally. Caller-defined group_id must identify the original source,
not each generated row; missing lineage cannot establish semantic independence.
Update router docs with a food mixed-topic precedence rule, annotation guidance,
calibration caveats and commands. Add regression tests for mixed-topic policy
examples without claiming untrained model accuracy.
