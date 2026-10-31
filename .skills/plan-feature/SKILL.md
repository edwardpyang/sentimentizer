---
name: plan-feature
description: Use ONLY when the user asks to plan a new feature, refactor, or architectural change. Guides a structured research-and-review cycle: discover codebase, identify risks, iterate on the plan, then produce an implementation-ordered plan committed under docs/. Do NOT use for bug fixes, small edits, or single-file changes.
---

# Plan a Feature

You are planning a non-trivial feature, architectural change, or refactor in
this repository — a PyTorch sentiment-analysis pipeline with Ray Train, Ray
Serve, an intent router, embeddings, and metrics. Produce a reviewed,
risk-adjusted plan before writing any code.

> **Do not run `mdformat` on this file or any other `.skills/**/SKILL.md`.** It
> rewrites the `---` frontmatter delimiters into horizontal rules and collapses
> the YAML into a heading.

## When to use this skill

- A multi-step change spanning several files or modules.
- A change touching shared infrastructure (`Trainer`, `SentimentPredictor`,
  the metrics pipeline, `create_fastapi_app()`, Ray session lifecycle).
- The user explicitly asks to "plan" or "review" before implementing.

Do NOT use this for one-off edits, bug fixes, or changes confined to one file.

## Phase 1: Discover

1. **Read `AGENTS.md`** at the repo root. There is no `CONTRIBUTING.md`.
1. **Read the `.rules/` files matching your change area.** AGENTS.md carries
   the routing table. At minimum, any Python implementation requires
   `.rules/python-style.md`, `.rules/dry.md`, and `.rules/solid.md`; a
   multi-milestone plan requires `.rules/plan-review.md` and
   `.rules/testing-workflow.md`. Logging, timeouts, or long-running work also
   requires `.rules/observability.md`. These are mandatory, not background
   reading.
1. **Check `.skills/` for a task-specific workflow.** A concurrency-sensitive
   change has its own skill (`change-concurrent-code-safely`); load it rather
   than improvising.
1. **Map the affected surface area** with `glob`/`grep`/`read`:
   - Every call site of any function whose signature you plan to change. Count
     them; the number goes in the plan.
   - Config dataclasses and the YAML/env overrides that feed them.
   - Model factory and loader sites (`trainer.py::_train_func`,
     `workflows/helpers.py`::_load_model`).
   - Test files covering the affected area.
   - Optional dependency groups in `pyproject.toml`.
1. **Read 2–3 existing implementations end-to-end.** Adding a new model means
   reading `models/rnn.py`, `models/encoder.py`, or `models/modernbert.py`
   completely — not skimming one.
1. **Verify library capability from source.** Check what Ray Train, Ray Serve,
   or Transformers actually support before planning around it. Marketing
   READMEs often overstate coverage.
1. **Verify, don't inherit, dated observations.** Behavior recorded in an
   older plan or doc may be stale. Re-check it, and record the verification
   date next to anything you rely on.

## Phase 2: Draft the plan

The plan is a committed document at `docs/<slug>-plan.md`.

**It must open with a `**Status:**` line carrying a state and a date** —
`📋 Not started`, `⏳ Partially implemented`, or `✅ Completed`.
`.rules/plan-review.md` is the authority on this and on the status lifecycle;
follow it rather than inventing a format. Preserve prior status text as an
italic `_Original status: …_` line instead of deleting it.

### Required sections

1. **Goal** — one or two sentences on what the feature does.
1. **Scope** — what is in, and explicitly what is out.
1. **Architecture** — file tree with `NEW`/`MOD` markers.
1. **Public interfaces** — endpoints, CLI surfaces, model signatures,
   and response shapes. Serve endpoints are namespaced by domain
   (`/v1/sentiment/{...}`, `/v1/router/{...}`, etc.).
1. **Implementation order** — milestones, each a commit-sized unit, each
   ending in a `make ci` gate.
1. **Testing strategy** — named test modules and the scenarios each must
   cover.
1. **Risk review** — a table of risk → resolution (Phase 3).
1. **Assumptions** — every empirical claim the plan rests on.
1. **Convention updates** — changes needed in `AGENTS.md`, `.rules/`,
   `docs/troubleshooting.md`, or relevant doc files.

### For each change, specify

- New classes and functions with complete signatures — no placeholder types. A
  signature naming a type the plan never defines is not decision-complete.
- Modified functions with before/after signatures and the call-site count.
- Config/schema field additions with types and defaults.
- For a new model type: the config dataclass, model class, factory import in
  `_train_func`, and loader import in `_load_model`.

## Phase 3: Risk review

`.rules/plan-review.md` owns the general review cycle — at least one explicit
revise-and-re-review pass. This section adds the risk classes that recur *in
this repository*. Walk them explicitly.

- **3-class contract preservation.** All models output `(B, 3)` logits;
  labels are 0=negative, 1=neutral, 2=positive. Any metric, loss, or serving
  change must respect this. `LABEL_NAMES` and `NUM_CLASSES` in `config.py`
  are the single source of truth — never re-declare.
- **Base class coupling.** If the new subclass overrides `__init__`, does it
  skip parent initialization that sets required state (`STEP_SCHEDULER_PER_BATCH`,
  `SUPPORTS_ONNX`)? Document which parent methods are called/skipped and why.
- **Scheduler contract.** `_LinearWarmupCosineScheduler` uses `(step + 1) /
  warmup_steps` (not `step / warmup_steps`) and clamps `progress` to `1.0`.
  Any scheduler change must preserve these invariants.
- **Metrics pipeline coherence.** Adding a training metric requires changes in
  seven places (see "Adding a new training metric" in AGENTS.md). Missing one
  silently drops the metric or breaks the exporter.
- **Ray lifecycle.** `_train_func` runs on Ray workers — it must not depend on
  driver state loaded before `trainer.fit()`. `train.get_context()` and
  `train.get_dataset_shard()` are standalone functions inside workers only.
  Checkpoints are directory-based (`Checkpoint.from_directory()`), not
  dict-based.
- **Ray Serve contracts.** Never define `__call__` on a deployment class.
  Use `create_fastapi_app()` so each deployment gets its own app instance.
  Middleware order is behavior — CORS is outermost.
- **`RAY_ENABLE_UV_RUN_RUNTIME_ENV` module-level.** Any new Serve entry point
  must set `os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "false")`
  at module level, before any `import ray`.
- **Lazy loading contracts.** `SentimentPredictor` lazy-loads the router;
  `_ensure_router_loaded()` memoizes. Changing load ordering can break the
  contract or add startup cost.
- **Optional heavy dependencies.** Keep them behind import guards so the base
  service imports and serves unaffected routes without them. Requesting an
  unavailable backend must raise a clear capability error, not a deep
  `ImportError`.
- **Import-time side effects.** Does the new optional dep get imported at
  module level? Guard with try/except. A module-level Ray gauge creation
  crashes in a worker-less context; use the lazy `_get_ray_gauges()` pattern.
- **Dependency conflicts.** Does the new dep conflict with existing deps
  (e.g., same library, different versions)? Check `pyproject.toml` and the
  new package's dependencies.
- **Semantic mismatches.** Do config fields from the existing system map
  cleanly to the new backend, or do some fields become meaningless? Document
  which fields are ignored and what happens (warn? error?).
- **Test coverage gaps.** Are there code paths that only execute when the
  optional dep IS installed? These need separate test files or conditional
  skips.
- **Coexistence risk.** Can both backends/models run in the same process? Do
  they compete for GPU/memory? Log a warning if simultaneous use is risky.
- **API divergence.** Does the new library have the same API surface as the
  existing one? Document exact mismatches (e.g., seed handling, return types,
  unsupported features).
- **Registry completeness.** For every model type in the system, is there a
  defined set of available backends and capabilities (`SUPPORTS_ONNX`,
  `STEP_SCHEDULER_PER_BATCH`)? Is there a clear error if someone requests
  an unsupported operation?
- **Return type consistency.** If the existing interface returns a specific
  tuple/dict shape, the new implementation must return the same shape. Wrap
  the library's return type if it differs.
- **ONNX export compatibility.** `SUPPORTS_ONNX` gates export. If the new
  model subclasses `HFTransformerModel`, ONNX export is disabled by default —
  the plan must address whether that's intentional.
- **Metrics file race conditions.** If training runs concurrently or the
  metrics pipeline changes, check for write races in
  `/tmp/sentimentizer_metrics/`. Writes are atomic by design; new write
  paths must not break that.

## Phase 4: Iterate

Present the plan and ask what is actually undecided — not a fixed script.
Useful questions:

1. Does the scope match what you intended?
1. Which risks should I dig into further?
1. Should we adjust the optional dependency boundary?
1. Is any milestone large enough to land as its own change?

Revise and re-review at least once. Do not implement until the user approves.

## Phase 5: Implementation order

After approval, produce a numbered implementation checklist ordered by
dependency (things that must exist before other things can reference them).
Each item should be a single commit-sized unit:

1. Import guard / compat module (no other code depends on it yet)
1. Config changes (new fields, registry, env vars)
1. New implementation classes (behind the import guard)
1. Factory/wiring (routes config → implementation)
1. Serve layer / integration points
1. Optional dep group in pyproject.toml
1. Tests (unit tests for factory, config; integration tests behind skipif)
1. Convention docs (AGENTS.md, `.rules/`, etc.)

Every milestone ends with `make ci` — per `.rules/testing-workflow.md`, not
only at the end. A milestone that cannot pass the full gate is not finished.

Blocking prerequisites come first. If the plan depends on live model behavior
or data, verify it empirically *before* writing tests against assumptions.

## Tips

- **Copy the existing pattern.** If the codebase has an import-guard + factory
  pattern (like `TRANSFORMERS_AVAILABLE` + `new_modernbert_model()`), copy it
  exactly. Consistency matters.
- **Immutable config dataclasses.** If the existing config uses `frozen=True`,
  your new fields must have defaults or use the same pattern for construction.
- **YAML config precedence.** New env vars need corresponding entries in both
  the `_ENV_OVERRIDES` dict and the YAML file. Don't forget the `_FIELD_TYPES`
  dict for non-string types.
- **Return type consistency.** If the existing interface returns a specific
  shape, the new backend must return the same type. Wrap the library's return
  type if it differs.
- **Web-verify library claims.** Before planning around a library's supported
  models or features, check its actual source code, model directory, or API
  docs. The plan's risk review is the right place to flag "supported models"
  vs "assumed models."
- **Report environment-dependent failures** rather than hiding or bypassing
  them.
- **Documentation-only changes** verify with `git diff --check`; skip the
  Python suite.
