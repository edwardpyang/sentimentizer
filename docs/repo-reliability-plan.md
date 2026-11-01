# Repository reliability improvements

**Status:** ✅ Completed (2026-11-01; implemented and verified locally)

_Original status: ⏳ Partially implemented (2026-11-01)_

## Goal and scope

Implement the six approved improvements: reliable ModernBERT Ray tests, installed
wheel/container smoke checks, coverage after failed tests, reusable release/CI
verification, shared accumulation/scheduler behavior, and enforced types at schema
and configuration boundaries. Public prediction, CLI, and checkpoint formats stay
compatible. A full training orchestration rewrite is outside this change.

## Architecture and interfaces

- MOD `tests/test_modernbert_ray.py`: generate a small parquet fixture rather
  than depending on local data; own Ray cleanup and let assertions fail normally.
- MOD `Dockerfile`, `.dockerignore`: include the `workflows` package.
- NEW `scripts/smoke_installed.py`: `main() -> None`; verify installed imports,
  bundled serving YAML, and CLI/serving help from outside the source checkout.
- NEW `.github/workflows/verify.yaml`: reusable verification with lint, format,
  Pyright, coverage tests, wheel smoke tests, pandas-floor tests, and Docker smoke.
- MOD CI/publish workflows: call the shared workflow; publishing requires its
  success. Reporting runs after failed tests only when coverage data exists.
- NEW `.github/scripts/coverage_summary.py`: `render_summary(test_outcome: str)
  -> str` and `main() -> None`; exercise complete, partial, and missing-data
  reporting through the actual script.
- MOD `trainer.py`: introduce a private per-epoch stepper that calls `train_step`
  and owns optimizer-boundary scheduler stepping and partial-window flushes.
  All three existing `train_step` call sites use it. Existing public function
  signatures stay compatible; accept tensor inputs in `train_step`'s annotation.
- MOD schema/config modules: override the legacy Pyright warning policy with
  module directives for argument, return, call, attribute, optional-member,
  general-type, and import errors in six boundary modules.
- NEW training parity tests: compare real optimizer updates and scheduler counts
  across single-node, callback, and Ray worker execution.

The private training interfaces are:

```python
def _apply_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    gradient_scale: float = 1.0,
) -> float: ...

def _optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int: ...

class _TrainingStepper:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        loss_function: Callable,
        *,
        grad_accum_steps: int,
        use_amp: bool,
        device_type: str,
    ) -> None: ...

    def step(
        self, inputs: dict[str, torch.Tensor] | torch.Tensor, target: torch.Tensor
    ) -> tuple[float, float]: ...

    def flush(self) -> float: ...
```

`Callable` is `collections.abc.Callable`; tensor and optimizer types come from
PyTorch. Three loop call sites now construct a stepper, which calls `train_step`.
No config fields, CLI flags, HTTP endpoints, or checkpoint fields are added.

## Implementation order and verification

1. Test lifecycle, packaging, shared workflows, and coverage reporting. Add tests
   against failure paths and smoke the built wheel. Finish with `make ci`.
2. Shared training stepper, parity/resume regressions, and scoped type enforcement.
   Finish with focused training tests, the new type gate, and `make ci`.
3. Update this plan and contributor documentation to reflect verified behavior.

## Testing strategy

- `tests/test_ci_reporting.py`: run the real summary script against real coverage
  data for passing, failing, cancelled, and missing-data runs.
- `tests/test_modernbert_ray.py`: construct the real trainer with synthetic data;
  scratch reproductions verify assertions fail and setup failures shut Ray down.
- `scripts/smoke_installed.py`: verify installed package origins, bundled YAML,
  console help, and module help in wheel and container environments.
- `tests/test_training_steps.py`: compare actual optimizer updates with explicit
  effective-batch training for all three loop entry points over multiple epochs;
  cover complete/partial windows, scheduler budgets, empty/repeated flushes,
  invalid accumulation, and the standalone last-batch path.
- `tests/test_pipeline_regressions.py`: exercise real two-worker training with
  incomplete windows and checkpoint resume with non-divisible batch counts.
- `tests/test_serve_config.py`: reject non-mapping YAML, unknown fields, and bad
  field types; retain typed environment precedence.
- Pyright: run the full project and verify deliberately invalid argument/return
  probes fail in a temporary boundary-module copy. Actionlint checks workflows.

## Ownership and concurrency review

Resource: gradients, optimizer and scheduler state. Invariant: exactly one
optimizer update and one per-update scheduler step per accumulation window,
including an incomplete final window. Sharing scope: one stepper per epoch per
worker; no shared driver state or locks. Linearization point: `optimizer.step()`,
followed by `scheduler.step()`. DDP still performs backward synchronization on
every microbatch; this change does not introduce `no_sync` or alter collectives.

| Risk/interleaving | Resolution |
| --- | --- |
| Epoch ends with pending gradients | Flush once and reset the epoch-local state. |
| Optimizer replaced after backbone unfreeze | Construct a new stepper each epoch. |
| DDP hides model scheduler flag | Resolve the flag from the unwrapped model. |
| Test raises after Ray starts | Fixture finalizer shuts down on every exit path. |
| Tests fail before coverage exists | Skip report/artifact work without masking test failure. |
| Checkout hides missing installed modules | Run smoke tests from a temporary working directory with source paths removed. |
| Failed verification still allows publishing | Publish explicitly depends on shared verification. |
| Global stricter types create unrelated churn | Enforce selected diagnostics only at schema/config boundaries. |
| Dynamic config dictionaries bypass useful types | Validate through `TypeAdapter(ServeConfig)` and reject malformed YAML/unknown options. |

## Review and revision

First review: a wholesale loop rewrite would couple Ray reporting, checkpointing,
metrics, and early stopping. Revised scope to share only update boundaries while
testing actual entry points. Also require isolated installation checks rather
than help commands from `/app`, where source files could hide packaging defects.
Second review: account for missing coverage data, DDP wrappers, optimizer rebuilds,
and one-shot callback iterators. No new dependencies or external service APIs are
needed by application code. A further implementation review found that partial
windows also under-scaled gradients and scheduler budgets rounded down; normalize
those windows and round budgets up, with observed failing regressions before the
fix. Configuration dictionaries now cross a Pydantic adapter instead of using an
unchecked constructor unpack. CI action configuration is validated separately.

## Assumptions and convention updates

Pre-change audit on 2026-11-01: three training loops called `train_step`; the
callback loop had no production caller; Docker omitted `workflows`; the
ModernBERT Ray test caught assertions; release checks omitted Pyright. Installed
Ray APIs and existing worker integration tests remain the source of truth.
`AGENTS.md`, coverage guidance, configuration docs, and training docs now describe
the implementation. A subsequent user request authorizes a PR containing this
work and the CD tagging follow-up below.

## Verification results (2026-11-01)

- Baseline: 959 passed, one skipped. Coverage data was kept under `/tmp`.
- Milestone 1: `make ci` passed with 963 passed, one skipped, and an isolated
  wheel build/smoke check. The built Docker image also passed its smoke check.
- Regression proof before the training/config fixes: 12 failing tests covered
  the three loop paths, partial scheduler budgets, and malformed configuration.
  A separate SGD reproduction fails against the old `train_step` and passes
  against the current function. Bad trainer assertions previously skipped and
  now fail; dataset setup exceptions execute Ray shutdown.
- Focused training, callback, and configuration suite: 49 passed.
- Final `make ci`: format/lint passed, Pyright zero errors (214 legacy warnings),
  990 tests passed and one skipped, source distribution/wheel built, and installed
  wheel smoke passed. CPU thread counts were bounded to one while an unrelated
  benchmark was running. An earlier unbounded run was interrupted and replaced
  by a clean passing run.
- Coverage using the same configured source/branch scope: lines 75.43% → 76.08%;
  branches 61.59% → 63.06%; combined 72.58% → 73.41%. Ray subprocess tracing remains
  excluded, as before.
- A temporary config-module copy with invalid argument and return types produces
  exactly two Pyright errors; the production boundary modules have no errors.
- Actionlint v1.7.12 accepted all three final workflow files. The final image was
  rebuilt and smoke-tested with networking disabled from `/tmp`; package origins,
  bundled YAML, and both CLI entry points passed.
- The real final coverage data rendered successfully through the summary script,
  including the Markdown missing-lines column. `git diff --check` passed.

GitHub-hosted workflow execution and PyPI publishing were not triggered as part
of this local implementation.

## Follow-up: CD release tags (2026-11-01)

**Status:** ✅ Completed (2026-11-01; implemented and verified locally)

_Original status: 📋 Not started (2026-11-01)_
_Previous status: ⏳ Partially implemented (2026-11-01)_

Goal: after a successful main-branch PyPI publish, create a GitHub release and a
tag equal to `pyproject.toml`'s version, following the existing bare-version tag
convention. The tag must point at the exact published commit. Manual release
events already own their release/tag and do not need this follow-up job.

Implementation: add `.github/scripts/publish_release.py` with `main() -> None`
and `ensure_release(version: str, commit: str) -> None`, called by a new job in
`publish.yaml` that needs `publish` and alone receives `contents: write`.
`_github_request(method: str, path: str, payload: dict[str, object] | None = None)
-> dict[str, object] | None` owns HTTP/authentication and a bounded timeout.
Existing matching lightweight or annotated tags/releases are reused. A tag at a
different commit fails; it is never moved. Generated release notes use GitHub's
API. The package version continues to be bumped in PRs, not rewritten by CD.

Review 1: creating tags before upload could falsely imply deployment succeeded.
Revised the job dependency to require successful publishing. Give only the release
job repository write access. Use `GITHUB_TOKEN` so creating the release does not
recursively launch the existing release-triggered publishing workflow.
Review 2: account for annotated tags, malformed API responses, authorization and
network errors, reruns after partial tag/release creation, concurrent creation,
and a tag already pointing elsewhere. Serialize duplicate release jobs for the
same commit and recheck already-created resources after a create conflict. Use
GitHub's semantic-version latest-release selection for out-of-order completion.

Tests: exercise the actual helper and main entry point with mocked HTTP at the
GitHub boundary; include fresh releases, reruns, lightweight/annotated tags,
conflicts, bad responses, HTTP failures, and version/commit selection. Run focused
tests, actionlint, and `make ci`. No live release is created during verification.

Verification: 27 focused release-automation tests passed, actionlint accepted the
workflows, and the post-bump `make ci` gate passed with 1,017 tests passed, one
skipped, zero Pyright errors (214 legacy warnings), successful distribution
builds, and installed-wheel smoke checks. Version `0.333.0` is a minor bump due
to observable configuration-validation behavior. The user subsequently requested
that the complete change be pushed and opened as a PR against `main`.
