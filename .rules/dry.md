## Brief overview

DRY — Don't Repeat Yourself. Every piece of knowledge should have a single,
authoritative representation in the codebase. Applies to logic, constants,
schemas, and configuration alike.

## What to deduplicate

- **Logic**: if the same non-trivial computation appears twice, extract a
  function. A literal third occurrence is a hard signal — refactor before adding it.
- **Constants and magic values**: `LABEL_NAMES`, `NUM_CLASSES`, model type
  strings, gauge names, and config defaults live in `config.py` or the owning
  module, named once. Never inline the same literal in two places.
- **Schema**: Pydantic v2 models in `router/schemas.py` own shared router task,
  classification, and artifact metadata contracts. HTTP envelopes in
  `serve/models.py` reuse these types; `serve/embeddings_models.py` owns embedding
  request/response shapes. Do not restate field shapes in docstrings or ad-hoc
  dicts. Use the model's `.model_dump()` / `.model_json_schema()` instead of
  re-typing.
- **Behavioral knowledge**: the 3-class contract, metrics computation, class
  balancing, the metrics pipeline write path — one implementation, called from
  every path.
- **Metric gauge definitions**: gauge names, labels, and keys are defined in
  `exporter.py` and referenced by `_METRIC_GAUGE_KEYS` — never redeclare.

## What DRY is NOT

- **Not "never type the same token twice."** Two functions that happen to share
  a line today but change for different reasons should stay separate — coupling
  them creates a false abstraction that's worse than the duplication.
- **The test**: would these copies always have to change *together*, for the
  *same* reason? If yes, deduplicate. If they only look alike (incidental
  duplication), leave them — premature extraction is a SOLID/coupling violation.
- Prefer a small amount of duplication over the *wrong* abstraction. It is
  cheaper to merge two similar functions later than to untangle a bad shared one.

## Practice

- Extract shared helpers into the module that *owns* the concept, not a generic
  `utils` grab-bag (see [solid](solid.md) — single responsibility).
- When you find existing logic that does what you need, **reuse it** rather than
  writing a parallel implementation — search first.
- Config/env defaults are declared once in `config.py` or `env.py`; tests
  override via the same surface, never by re-declaring values.
