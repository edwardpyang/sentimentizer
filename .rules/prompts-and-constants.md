## Brief overview

Model-call constants (generation params, schema names, tuning knobs) are
externalized from handler code — never inlined as literals in the call site.
This extends DRY to the model-inference and agent surface.

## Constants at module top

- Magic values passed to model APIs — temperature, top_k, top_p, max_tokens,
  prompt templates, schema names — are named module-level constants
  (`UPPER_SNAKE`) at the top of the owning module, not buried inline in the
  request payload.
- Config dataclass defaults (in `config.py`) are the single source of truth
  for training parameters — never re-declare them in `Makefile` targets or
  CLI flags.
- Reason: these are tuning knobs — they change for different reasons and at a
  different cadence than the call-site logic. Named constants make the knob
  discoverable and keep the call site readable (DRY: single source of truth).

## Current locations

- **Training parameters**: `sentimentizer/config.py` — dataclass defaults for
  scheduler, optimization, data loading.
- **Metrics gauge keys**: `sentimentizer/metrics_publisher.py` —
  `_METRIC_GAUGE_KEYS` maps metric names to gauge definitions.
- **Router config**: `sentimentizer/router/config.py` — RouterConfig /
  AugmentConfig defaults.
- **Agent constants**: `sentimentizer/agent/` — model names, prompt templates,
  search rate limits.
- **Serve config**: `sentimentizer/serve/config.py` — endpoint limits, CORS,
  body-size caps.

## What this is NOT

- Not "every string is a constant." Short, structural strings (route names,
  error codes, log messages) stay inline — only **model-facing and tuning-knob
  values** are externalized.
- Not an env-var thing. Generation params and training knobs are *code-level*
  tuning constants. `config.py` handles env-var overrides for deploy-time
  config (URLs, model ids, device settings).
