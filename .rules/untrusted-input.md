## Brief overview

Web results, HTTP input, and downloaded artifacts are controlled by someone else. Treat
every field as hostile, and decide what happens when one is malformed.

## Content that reaches an LLM

- **Filter every field the model will see**, not only the obvious one. A search result's
  title and URL reach the agent exactly as its content does.
- **Patterns in `safety.py` must tolerate variation.** Use `(?i)` on all of them, and allow
  filler words: real attacks say "ignore *all of your* previous instructions".
- Regex filtering is best effort. Tool docstrings must still tell the model to treat
  results as untrusted data.
- Budgets must have an owner that resets them. A "per run" limit with no reset at the start
  of the run is really a per-process limit. Test the entry point, not only the counter.

## Parsing external responses

- Check the shape before using it. `json.loads` can return a list or string, and `null`
  values defeat `dict.get(key, default)`.
- Raise the exception type callers already handle. An `AttributeError` from a malformed
  response escapes `except (ValueError, RuntimeError, OSError)` and aborts the whole run.

## HTTP input

- Validate request values that get echoed into headers or logs, such as `X-Request-Id`:
  use an allowlist and a length limit, and replace invalid values.
- Keep validation in Pydantic where possible. See the Serving API section of `AGENTS.md`.
- Internal errors never expose exception text in the response body.

## Downloaded artifacts

- `torch.load` always uses `weights_only=True`.
- **Pickle formats execute code on load.** Never call `joblib.load` or `pickle.load` on an
  artifact in a serving or download path. Store numbers in safetensors or JSON, as
  `router/head.py` does. One-time migrations use an allowlisted unpickler, and joblib's
  object-array path must be blocked too because it calls an unrestricted `pickle.load`.
- Pin auto-downloaded artifacts to a commit (`HF_ROUTER_REVISION`). A pin stops the content
  from changing; it does not show that the content is safe. Do not call a pinned commit
  reviewed unless someone reviewed it.
- `trust_remote_code=True` requires a pinned `revision`.
