# Image Service Extraction — Implementation Plan

**Status:** ⏳ Partially implemented (2026-10-31). M1-M4b are done on the `extract-from-sentimentizer` branch in `imagenate`. The image flow has never run in production, so M5 reduces to generating real images once (see M5). M6-M7 are ready on `remove-image-service` and wait on that.

_Original status: 📋 Not started (2026-10-31)_

## Goal

Move headless image generation (Krea 2 / Ideogram 4 via the ComfyUI sidecar)
out of `sentimentizer` into its own repository and deployable service,
**`imagenate`**. Image generation is unrelated to sentiment analysis, so the
new service carries no sentimentizer naming. Client-facing behavior is
**unchanged**: same `/v1/images/*` routes, request/response
shapes, error codes, auth, rate limiting, idempotency, async jobs, licensing
gates, and moderation. The new service keeps Ray Serve for this migration so
that any behavior difference can be attributed to the move itself; removing
Ray is a separate, later change (see [Follow-up](#follow-up-drop-ray-serve)).

## Scope

**In scope**

- New repository containing the image service, created with git history for
  the moved files.
- A standalone entry point, config, health/metrics routes, packaging, CI,
  container image, and Kubernetes manifests for the new service.
- An OpenAPI parity check between the old and new `/v1/images` apps.
- Renaming the service's environment variables from `SENTIMENTIZER_*` to
  `IMAGENATE_*` (see [Environment variable rename](#environment-variable-rename)).
- Ingress cutover of `/v1/images` to the new service.
- Removal of all image code, config, tests, the `diffusion` extra, and docs
  from `sentimentizer` **after** cutover is verified.

**Out of scope**

- Dropping Ray Serve, replacing the in-memory `JobStore`, or any change to
  ComfyUI workflows, checkpoints, models, or licensing gates.
- Moving to a separate hostname (D5). Cutover keeps `/v1/images` on the
  current host; a hostname change is a later, client-visible step.
- Changing sentiment, router, or embeddings behavior. The only sentiment-side
  behavior change is that `/health/ready` no longer reports image backends.
- Fixing pre-existing issues found during discovery (listed under
  [Pre-existing issues](#pre-existing-issues-found-during-discovery)); each is
  noted where it affects the migration.

## Decisions

| ID | Decision | Status |
|---|---|---|
| D1 | Repo and package name | **Decided:** `imagenate` for both the GitHub repo and the Python package. Free on PyPI and in your GitHub account (checked 2026-10-31). |
| D2 | Where the service runs | Recommended: its own container/pod. Required if it shares a host with sentiment training (see risks R1, R2). |
| D3 | Env var prefix | **Decided:** rename `SENTIMENTIZER_*` to `IMAGENATE_*` during the move, with no backward-compatible aliases. Cutover writes fresh deployment config for the new service anyway, and parity is about the client API, not config names. |
| D4 | Layout in the new repo | Recommended: mirror the current paths (`imagenate/diffusion/*`, `imagenate/serve/*`) so Milestone 1 is a mechanical rename. Reorganize later. |
| D5 | Hostname | Recommended: cut over on the current host's `/v1/images` path so clients see no change. Move to a dedicated hostname (e.g. `images.example.com`) in a later change, with a redirect or deprecation window. |

## Current state (verified 2026-10-31 against `main` @ 2d90b50)

### What moves

| Path | Lines | Notes |
|---|---|---|
| `sentimentizer/diffusion/` (`comfyui.py`, `config.py`, `diffusion_config.yaml`, `image_utils.py`, `job_store.py`, `moderation.py`, `__init__.py`) | ~890 | Only imports from `sentimentizer` are `logger` and its own siblings. |
| `sentimentizer/serve/diffusion_app.py` | 654 | `ComfyUIDeployment`, `ImagesDispatcher`, `_backend_http_exception`. |
| `sentimentizer/serve/diffusion_models.py` | 258 | Pydantic request/response models. |
| `sentimentizer/serve/middleware.py` | 305 | `require_api_key`, `rate_limit`, `IdempotencyCache`, `idempotent`, `check_prompt_safety`. **Only `diffusion_app.py` imports it.** |
| `tests/test_comfyui.py` | 25 tests | Imports `ServeConfig` for licensing-gate tests. |
| `tests/test_diffusion_jobs.py` | 22 tests | Includes one Ray integration test (`test_visibility_across_actor_handles`). |
| `tests/test_middleware.py` | 5 of 6 classes | `TestRequireApiKey`, `TestRateLimiter`, `TestIdempotencyCache`, `TestCheckPromptSafety`, `TestErrorEnvelopeIntegration` (it uses `check_prompt_safety`) move. `TestRequestBodySizeLimitMiddleware` tests sentimentizer's own `_RequestBodySizeLimitMiddleware`, so it stays and is also copied to the new repo for `serve/http.py`. |
| `tests/test_safety.py` | 9 tests | Moves with `is_safe`. Sentimentizer keeps only `INJECTION_PATTERNS` and a smaller test. |
| `tests/test_serve.py` | 2 classes + 1 test | `TestGenerateRequestValidation` (4 tests) and `TestRateLimiterAndAuth` move; `TestReadinessEndpoint::test_health_ready_returns_503_when_comfyui_is_unavailable` is deleted. |

### What the moved code depends on in `sentimentizer` (must be copied)

| Dependency | Defined in | Copy as |
|---|---|---|
| `create_fastapi_app()`, `_RequestBodySizeLimitMiddleware`, `http_exception_handler`, `_status_code_to_error_code` | `serve/app.py` (~150 lines) | `imagenate/serve/http.py` |
| `ServiceMetrics`, `render_service_metrics`, `serve` import | `serve/base.py` | `imagenate/serve/base.py` (drop `_DummyServe`; Ray is a hard dependency there) |
| `is_safe`, `NSFW_BLOCKLIST`, `INJECTION_PATTERNS` | `safety.py` (70 lines) | `imagenate/safety.py`. `sentimentizer` keeps only `INJECTION_PATTERNS`, which `agent/websearch.py` uses; `is_safe` and the blocklist have no other caller and are removed in M6. |
| `logger` | `sentimentizer/__init__.py` | Minimal structlog setup in `imagenate/__init__.py` |
| Image fields of `ServeConfig` + validation + env map | `serve/config.py` | `ImagesConfig` (see Public interfaces) |
| API-key env export (`SENTIMENTIZER_API_KEYS`) | `serve/app.py:119-120` (module level) | **Not copied.** It works today only because `diffusion_app.py` imports `serve/app.py`, so each replica runs the export. Copying `create_fastapi_app` cuts that import, so a module-level export in the entry module would never run in a replica. `_load_valid_keys()` reads `load_images_config().api_keys` instead (R6). |

**Why `diffusion_app.py` currently needs torch:** it imports
`create_fastapi_app` from `serve/app.py`, which imports `SentimentPredictor`
and the model stack. Copying `create_fastapi_app` cuts that edge; the new
service's runtime dependencies are FastAPI, Ray Serve, protobuf (capped below 7,
see R17), Pillow, orjson, PyYAML, and structlog. ComfyUI and moderation clients use only `urllib`.

**DRY note.** Copying `create_fastapi_app`, `ServiceMetrics`, and `safety.py`
is deliberate. Per `.rules/dry.md`, deduplicate only when copies must always
change together for the same reason. After the split, each service evolves its
middleware, error envelope, and blocklist independently. A shared third
package would couple two release cycles to save about 300 lines.

### Sentimentizer touchpoints that change

| Location | Change |
|---|---|
| `serve/app.py` `SentimentizerDeployment.__init__(embeddings_handle, image_model_names)` | Drop `image_model_names` and the ComfyUI health client (lines ~375-389). 1 bind site (`app.py` `main`); tests set `_image_model_names` / `_comfyui_health_client` at `tests/test_serve.py:95-96` and `:358-362`. |
| `serve/app.py` `health_ready()` | Remove the image backend block (lines ~810-818). |
| `serve/app.py` `main(host, port, diffusion=False)` | Becomes `main(host, port)`. Remove the diffusion branch (lines ~876-917). 1 call site: `serve/__main__.py:108`. |
| `serve/app.py` module docstring, lines 37-43 | Remove `/v1/images/*` endpoints. |
| `serve/app.py:119-120` | Remove the `SENTIMENTIZER_API_KEYS` export (only image auth reads it). |
| `serve/__main__.py` `--diffusion` / `SENTIMENTIZER_DIFFUSION_ENABLED` | Keep the flag for one release; it exits non-zero with a message pointing to the new service (R7). |
| `serve/models.py` | Remove `ImageModelsStatus` and `HealthReadyResponse.image_models`. |
| `safety.py`, `tests/test_safety.py` | Remove `is_safe` and `NSFW_BLOCKLIST`; keep `INJECTION_PATTERNS`. |
| `serve/config.py` `ServeConfig` | Remove 18 fields: `krea_2_enabled`, `krea_2_license_accepted`, `ideogram_4_enabled`, `ideogram_4_license_accepted`, `comfyui_base_url`, `comfyui_timeout_s`, `comfyui_poll_interval_s`, `comfyui_temp_directory`, `image_moderation_url`, `image_moderation_api_key`, `image_moderation_timeout_s`, `default_image_model`, `api_keys`, `rate_limit_per_min`, `rate_limit_burst`, `idempotency_ttl_s`, `job_ttl_s`, `prompt_blocklist_path`. Plus their validation, `_ENV_OVERRIDES` entries, `_FIELD_TYPES` entries, and docstring. **Keep** `request_timeout_s` and `cors_origins` (sentiment uses both). |
| `serve/service.yaml` | Remove the image and image-middleware blocks. |
| `pyproject.toml` | Remove the `diffusion` extra (`pillow`). Bump version (breaking for `--diffusion` users). |
| `.github/workflows/ci.yaml`, `publish.yaml` | Drop `diffusion` from `--extra` lists. |
| `Makefile` `serve-all` | Remove. |
| `.env.example` | Remove the diffusion block. |
| Docs | `README.md` (install lines 22-26, "Image Generation" section), `docs/serving.md` (line 15 and the "Image Generation Endpoints" section from line 244), `docs/configuration.md` (lines 99-200 image parts), `AGENTS.md` (architecture tree, "Headless image generation" section, image bullets under Serving API), `.rules/observability.md:32`, `.rules/prompts-and-constants.md:30`, `.skills/plan-feature/SKILL.md:10`. |
| Historical plans | `docs/diffusion_serving_plan.md`, `docs/flux2_reference_image_plan.md`, `docs/mlx_diffusion_plan.md` move to the new repo with history. |

## Architecture (new repository)

```
imagenate/
  pyproject.toml                      NEW  deps: fastapi==0.136.1, ray[serve]>=2.55.1,<2.56.0,
                                           pillow, orjson, pyyaml, structlog, pydantic; dev: ruff, pyright, pytest
  Makefile                            NEW  check / typecheck / test / ci / serve (mirrors sentimentizer targets)
  Dockerfile                          NEW  no torch
  Dockerfile.comfyui                  NEW  ComfyUI sidecar at COMFYUI_REVISION; weights are mounted, not baked in
  compose.yaml, compose.cpu.yaml      NEW  local stack with the same shape as the pod; the override checks wiring without a GPU
  AGENTS.md, CLAUDE.md                NEW  full agent guide for this repo (CLAUDE.md is a symlink)
  LICENSE, .gitignore, .dockerignore  NEW  .gitignore is new, not copied: sentimentizer's ignores *.json, which would drop the OpenAPI fixture
  .rules/                             NEW  all seven rule files, rewritten for this repo where they named sentiment code
  .skills/                            NEW  plan-feature, change-concurrent-code-safely (adapted); bump-comfyui-revision, add-image-model (new)
  .github/workflows/ci.yaml           NEW  ruff format --check, ruff check, pyright, pytest; permissions: contents: read
  .github/workflows/comfyui-compat.yaml NEW  runs the startup checks against pinned and latest ComfyUI (M4b)
  COMFYUI_REVISION                    NEW  pinned ComfyUI commit; single source for docs, CI, and the sidecar image
  scripts/check_comfyui_compat.py     NEW  placeholder checkpoints + runtime validation + workflow validation
  k8s/                                NEW  deployment (ComfyUI native sidecar), service, pvc, ingress (path /v1/images, proxy-body-size 4m, 600 s timeouts). No PDB: see R18
  docs/                               MOVED serving (image section), configuration (image section), historical plans; NEW troubleshooting
  imagenate/
    __init__.py                       NEW  structlog logger; sets RAY_ENABLE_UV_RUN_RUNTIME_ENV before any import of ray
    __main__.py                       NEW  CLI: --host, --port, --config
    safety.py                         COPY
    diffusion/                        MOVED (history kept)
      comfyui.py config.py diffusion_config.yaml image_utils.py job_store.py moderation.py
    serve/
      app.py                          NEW  main()
      ops_app.py                      NEW  OpsDeployment
      base.py                         COPY (ServiceMetrics, render_service_metrics)
      http.py                         COPY (create_fastapi_app and friends)
      config.py                       NEW  ImagesConfig + load_images_config()
      service.yaml                    NEW  image keys moved from sentimentizer service.yaml
      middleware.py                   MOVED (history kept)
      diffusion_app.py                MOVED (history kept; imports re-pointed)
      diffusion_models.py             MOVED (history kept)
  tests/
    test_comfyui.py test_diffusion_jobs.py test_safety.py MOVED
    test_middleware.py                                    MOVED (5 classes)
    test_http.py                                          COPY  (TestRequestBodySizeLimitMiddleware, re-pointed at serve/http.py)
    test_generate_request.py test_auth_rate_limit.py      NEW  (classes split out of test_serve.py)
    test_config.py test_ops_routes.py test_openapi_parity.py test_entrypoint.py   NEW
    test_packaging.py test_compat_script.py                                       NEW
    fixtures/images_openapi.json                          NEW  golden snapshot from sentimentizer
```

## Public interfaces

### Unchanged (parity contract)

All routes under `/v1/images` keep their paths, methods, request/response
models, status codes, error codes, and headers:

- `POST /v1/images/generate`
- `POST /v1/images/jobs` (201, `Location: /v1/images/jobs/{id}`)
- `GET /v1/images/jobs`
- `GET /v1/images/jobs/{id}`
- `DELETE /v1/images/jobs/{id}`
- `GET /v1/images/models`
- `GET /v1/images/models/{name}`

Also unchanged: Bearer auth, `Idempotency-Key`, the rate-limit headers, the
4 MiB body limit, `X-Request-Id`, the `{"error": {"code", "message"}}`
envelope. Configuration *values* and semantics are unchanged; only the env var
names change (D3). The images app is still served by Ray Serve with
`route_prefix="/v1/images"`, so `request.scope["root_path"]` (used to build
`Location`) is unchanged.

### New in the image service

**Ops routes.** Today the images app has no health or metrics routes: readiness
comes from sentiment's `/health/ready` probing ComfyUI, and the
`comfyui_images` `ServiceMetrics` collector is never rendered. Add a second,
small Serve app at `route_prefix="/"` with:

| Route | Behavior |
|---|---|
| `GET /health/live` | 200 `{"status": "live", "uptime_s": ...}`, the same shape as sentimentizer's. |
| `GET /health/ready` | 200 when ComfyUI `system_stats` succeeds and the `diffusion_job_store` actor resolves. Otherwise 503 with `image_models` and `image_backend_error`, the same keys sentiment reports today. |
| `GET /health` | Alias for `/health/ready`, as in sentimentizer. |
| `GET /metrics` | `render_service_metrics(..., service_prefix="imagenate")` over the `comfyui_images` collector. |

The dispatcher has no collector, so `/metrics` reads `ComfyUIDeployment`'s, which is in
another process. `OpsDeployment` gets a handle with
`serve.get_deployment_handle("ComfyUIDeployment", app_name="images")` and calls a new
`metrics_snapshot()` method; `ServiceMetrics` is already picklable. It must not `.bind()`
`ComfyUIDeployment`: a second `serve.run()` would start a second replica and a second,
unserialized ComfyUI queue. For the same reason, readiness uses its own short-timeout
`ComfyUIClient`, like sentiment's `/health/ready` does today.

**Entry point.**

```python
# imagenate/__init__.py: every module imports through here, which guarantees the order
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "false")  # before any `import ray`

# imagenate/serve/app.py
def main(host: str | None = None, port: int | None = None) -> None: ...
```

`main()` reproduces the image half of today's `sentimentizer.serve.app.main`:

1. `ray.init(address="local", namespace="imagenate", runtime_env={"py_executable": sys.executable})`.
2. `serve.start(http_options=HTTPOptions(host=..., port=..., request_timeout_s=cfg.request_timeout_s))`. A plain dict would silently drop the timeout.
3. Create the detached `JobStore` actor `diffusion_job_store`.
4. `serve.run(ImagesDispatcher.bind(...), name="images", route_prefix="/v1/images")` and `serve.run(OpsDeployment.bind(...), name="ops", route_prefix="/")`.
5. The same SIGINT/SIGTERM shutdown sequence.

It raises the same `ValueError` when no image model is enabled, now before Ray
starts rather than after. Signal handlers are registered first, because startup takes
long enough for a stop to land inside it.

`address="local"` is required. Without it, `ray.init()` joins a cluster recorded in
`/tmp/ray/ray_current_cluster` on the host, or named by `RAY_ADDRESS` (R1). That file is
written by `ray start`, not by `ray.init()`, so a local cluster is not joined by accident.

CLI: `python -m imagenate [--host H] [--port P] [--config PATH]`.
Default port is **8001**, so it can run beside sentiment on one host during
cutover. There is no `--diffusion` flag; the service only does images.

**Config.**

```python
@dataclass(frozen=True)
class ImagesConfig:
    serve_host: str = "0.0.0.0"
    serve_port: int = 8001
    request_timeout_s: int = 600
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    krea_2_enabled: bool = False
    krea_2_license_accepted: bool = False
    ideogram_4_enabled: bool = False
    ideogram_4_license_accepted: bool = False
    comfyui_base_url: str = "http://127.0.0.1:8188"
    comfyui_timeout_s: float = 600.0
    comfyui_poll_interval_s: float = 0.25
    comfyui_temp_directory: str = ""
    image_moderation_url: str = ""
    image_moderation_api_key: str = ""
    image_moderation_timeout_s: float = 10.0
    default_image_model: str = "krea_2"
    api_keys: list[str] = field(default_factory=list)
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 10
    idempotency_ttl_s: int = 600
    job_ttl_s: int = 3600

def load_images_config(path: str | Path | None = None) -> ImagesConfig: ...
```

Defaults, the `__post_init__` validation (licensing gates, Krea moderation
requirement, temp-directory requirement, default-model-enabled check, URL
scheme, positive timeouts), env var names, and parsing are moved verbatim from
`ServeConfig`; env var names follow the
[rename table](#environment-variable-rename). The config path env var is
`IMAGENATE_SERVE_CONFIG`. `prompt_blocklist_path` is dropped: it is declared but
never read anywhere.

`middleware._get_limiter()` currently calls
`sentimentizer.serve.config.load_serve_config()`. Re-point it to
`load_images_config()`.

### Environment variable rename

Rule: replace the `SENTIMENTIZER_` prefix with `IMAGENATE_`, keeping the rest of
the name. For example, `SENTIMENTIZER_KREA_2_ENABLED` becomes
`IMAGENATE_KREA_2_ENABLED`.

| Today | In imagenate |
|---|---|
| `SENTIMENTIZER_KREA_2_ENABLED`, `_KREA_2_LICENSE_ACCEPTED`, `_IDEOGRAM_4_ENABLED`, `_IDEOGRAM_4_LICENSE_ACCEPTED` | `IMAGENATE_` + same suffix |
| `SENTIMENTIZER_COMFYUI_BASE_URL`, `_COMFYUI_TIMEOUT_S`, `_COMFYUI_POLL_INTERVAL_S`, `_COMFYUI_TEMP_DIRECTORY` | `IMAGENATE_` + same suffix |
| `SENTIMENTIZER_IMAGE_MODERATION_URL`, `_IMAGE_MODERATION_API_KEY`, `_IMAGE_MODERATION_TIMEOUT_S`, `_DEFAULT_IMAGE_MODEL` | `IMAGENATE_` + same suffix |
| `SENTIMENTIZER_API_KEYS`, `_RATE_LIMIT_PER_MIN`, `_RATE_LIMIT_BURST`, `_IDEMPOTENCY_TTL_S`, `_JOB_TTL_S` | `IMAGENATE_` + same suffix |
| `SENTIMENTIZER_SERVE_HOST`, `_SERVE_PORT`, `_CORS_ORIGINS`, `_REQUEST_TIMEOUT_S` (shared with sentiment today) | `IMAGENATE_` + same suffix; sentimentizer keeps its own |
| `SENTIMENTIZER_DIFFUSION_{KREA_2,IDEOGRAM_4}_<FIELD>` (`diffusion/config.py`) | `IMAGENATE_DIFFUSION_{KREA_2,IDEOGRAM_4}_<FIELD>` |
| `SENTIMENTIZER_SERVE_CONFIG` (config file path) | `IMAGENATE_SERVE_CONFIG` |
| `SENTIMENTIZER_LOG_LEVEL` | `IMAGENATE_LOG_LEVEL` |
| `SENTIMENTIZER_DIFFUSION_ENABLED`, `SENTIMENTIZER_PROMPT_BLOCKLIST_PATH` | Not carried over. The service always serves images, and the blocklist path was never read. |

The new `docs/configuration.md` in imagenate carries this table, so operators
can translate existing deployment config at cutover.

### Changed in sentimentizer

| Before | After | Call sites |
|---|---|---|
| `main(host, port, diffusion=False)` | `main(host, port)` | 1 (`serve/__main__.py:108`) |
| `SentimentizerDeployment.__init__(embeddings_handle=None, image_model_names=None)` | `__init__(embeddings_handle=None)` | 1 bind in `main`; 2 test fixtures |
| `ServeConfig` with 18 image fields | Without them | Readers: `diffusion_app.py`, `middleware.py`, `app.py` (all removed or edited above); `tests/test_comfyui.py` (moves) |
| `GET /health/ready` includes `image_models` / `image_backend_error` when images enabled | Never includes them | — |
| `python -m sentimentizer.serve --diffusion` | Exits 2 with "image generation moved to the imagenate service; see docs/serving.md" | `Makefile` `serve-all` (removed) |

## Implementation order

Every milestone ends with `make ci` in the repo it touches. In sentimentizer,
documentation-only steps use `git diff --check` instead, per AGENTS.md.

### M1. Bootstrap the new repo with history (new repo, ~0.5-1 day) — **done**

1. In a fresh clone:
   ```bash
   git filter-repo \
     --path sentimentizer/diffusion/ \
     --path sentimentizer/serve/diffusion_app.py \
     --path sentimentizer/serve/diffusion_models.py \
     --path sentimentizer/serve/middleware.py \
     --path sentimentizer/safety.py \
     --path tests/test_comfyui.py --path tests/test_diffusion_jobs.py --path tests/test_middleware.py \
     --path tests/test_safety.py \
     --path docs/diffusion_serving_plan.md --path docs/flux2_reference_image_plan.md \
     --path docs/mlx_diffusion_plan.md \
     --path-rename sentimentizer/:imagenate/ \
     --replace-message <file with: regex:\(#(\d+)\)==>(sentimentizer#\1)>
   ```
   Then merge the result into `imagenate` with `git merge --allow-unrelated-histories`, because
   the repo already has an initial commit. The message rewrite stops old PR numbers from
   autolinking to unrelated issues in the new repo.
2. Rewrite imports from `sentimentizer.` to `imagenate.`.
3. Add the copied `serve/http.py`, `serve/base.py`, and the logger.
4. Add `pyproject.toml`, `Makefile`, ruff/pyright/pytest config copied from sentimentizer, and CI.
5. Move `TestRequestBodySizeLimitMiddleware` out of the moved `test_middleware.py` into `tests/test_http.py`, re-pointed at `imagenate.serve.http`.
6. Port `TestGenerateRequestValidation` and `TestRateLimiterAndAuth` into new test files.

**Gate:** the moved tests are the same count as in sentimentizer: 91 (25 + 22 + 28 + 9 + 7
ported). `make ci` cannot be green here alone, because the moved modules and tests import the
config from M2, so M1 and M2 share one gate.

### M2. Config and entry point (new repo, ~0.5 day) — **done**

1. Add `ImagesConfig` / `load_images_config()` and `service.yaml`. Move the licensing-gate tests in `test_comfyui.py` from `ServeConfig` to `ImagesConfig`.
2. Re-point `diffusion_app.py` and `middleware.py` from `sentimentizer.serve.config` to the new config.
3. Rename env vars per the [rename table](#environment-variable-rename): `ImagesConfig`'s env map, `diffusion/config.py` (`SENTIMENTIZER_DIFFUSION_` prefix), `middleware.py` (replace the env read with `load_images_config()`), and the moved tests (`test_middleware.py` sets `SENTIMENTIZER_API_KEYS` in 16 places). Add the startup warning for old names (R8).
4. Add `serve/app.py` with `main()`, and `serve/ops_app.py` with the ops deployment.
5. Add `__main__.py`.
6. Add a test that fails if the string `SENTIMENTIZER_` appears in `imagenate/` outside the old-name warning list. A test rather than a CI step, so that `make ci` catches it locally. The torch check in R14 is a test for the same reason.

**Gate:** `make ci` green, including the new `test_config.py`,
`test_ops_routes.py`, and `test_entrypoint.py`.

### M3. Parity check (both repos, ~0.5 day) — **done**

1. In sentimentizer, before anything is removed, dump the images OpenAPI schema with a one-off script (not committed to sentimentizer):
   ```python
   from sentimentizer.serve.diffusion_app import images_app
   images_app.openapi()
   ```
   Commit the output to the new repo as `tests/fixtures/images_openapi.json`.
2. Add `test_openapi_parity.py`, which compares the new `images_app.openapi()` with the snapshot. It ignores `info.title`, `info.version`, and `info.description`, and the new ops routes, which live on a separate app.
3. Smoke test: run sentimentizer `--diffusion` on port 8000 and the new service on 8001 against the same ComfyUI, with the same config values under each service's env names. Send the same request set to both: generate, create/get/list/cancel job, models, auth failure, rate limit, idempotency conflict, oversized body, unsafe prompt. Compare status codes and response keys.

**Gate:** parity test green; smoke results match (image bytes will differ
unless `seed` is fixed, so fix it).

**Result (2026-10-31):** the schema matches, and all 24 smoke scenarios match, against a
stand-in ComfyUI and moderation endpoint with no real weights. The new service took its API
keys from YAML only, which exercises R6. Stopping the stand-in returned 503 from
`/health/ready` while `/health/live` stayed 200. A real generation on a GPU is still needed
in M5.

### M4. Packaging and deployment (new repo, ~0.5-1 day) — **done**

1. `Dockerfile` without torch, and `Dockerfile.comfyui` for the sidecar at `COMFYUI_REVISION`.
2. `k8s/`: a Deployment and Service, with `replicas: 1` and `strategy: Recreate` (R18). No PDB.
   - The ComfyUI sidecar shares `comfyui_temp_directory` through an `emptyDir` volume, because the client unlinks files under that path. Both containers run as UID 1000.
   - ComfyUI is a native sidecar with an exec startup probe. It stays on loopback, which a kubelet `httpGet` could not reach, and imagenate starts only once it is up, because the service refuses to start without its nodes.
   - Every model is off in the manifest. Accepting a license is the operator's decision.
3. An Ingress for `path: /v1/images` (Prefix) with `proxy-body-size: "4m"`, and `proxy-read-timeout` and `proxy-send-timeout` of 600 to match `request_timeout_s`. Today's shared ingress caps all paths at `1m`, below the images app's own 4 MiB limit. It reuses the existing TLS secret with no cert-manager annotation (R19).
4. Readiness probe on `/health/ready` and liveness probe on `/health/live`. Readiness allows 12 s because the health client allows ComfyUI 10 s. A startup probe covers the time before the ops app exists.

**Gate:** `make ci` green; container builds; the probes pass against a
running ComfyUI.

**Result (2026-10-31):** both images build. Run together as a pod with Docker (shared network
namespace and temp volume, UID 1000, CPU ComfyUI with placeholder checkpoints), the startup,
liveness, and readiness checks passed and the service could delete from the shared temp
directory. The manifests pass `kubeconform` for Kubernetes 1.31. They have not been applied to
a cluster, and the GPU build of the sidecar has not been built.

### M4b. ComfyUI compatibility check (new repo, ~0.5 day) — **done**

ComfyUI usually supports new open-weight models within days of release, but
its node names, inputs, and enum values change between releases. Today a
mismatch shows up only when the service starts against a ComfyUI it hasn't
seen. This milestone adds a CI job that runs the same startup checks against
both the pinned and the latest ComfyUI release, without a GPU or real weights.

1. **Pin in one place.** Move the ComfyUI commit pinned in the README and
   `docs/serving.md` (`feca51a`) into a `COMFYUI_REVISION` file, as the full SHA, because
   `actions/checkout` cannot fetch a short one. Docs, CI, and
   the sidecar image read it from there.
2. **Make the startup check reusable.** Extract
   `ComfyUIDeployment._validate_runtime()` into a plain function in
   `diffusion/comfyui.py`, with no behavior change:
   ```python
   def validate_comfyui_runtime(
       client: ComfyUIClient, models: dict[str, ImageModelConfig]
   ) -> None: ...
   ```
   Move `_BASE_NODES`, `_MODEL_NODES`, and `_choices()` with it.
   `ComfyUIDeployment.__init__` calls the function, so the deployment and CI
   share one implementation.
3. **`scripts/check_comfyui_compat.py`** does four things, against a running ComfyUI:
   - Creates empty placeholder files named after every checkpoint in `diffusion_config.yaml` (via `load_diffusion_config()`), under ComfyUI's `models/diffusion_models`, `models/text_encoders`, and `models/vae`. These make the files appear in `/object_info` choices without downloading any weights.
   - Calls `validate_comfyui_runtime()` for every model. This catches missing or renamed node classes and checkpoint-loader changes.
   - Builds each model's workflow with `build_krea_2_workflow` / `build_ideogram_4_workflow` and submits it to `POST /prompt`. ComfyUI validates the graph (input names, types, enum values) before queueing, so a renamed or removed input fails here with `node_errors`, which the node-class check alone would miss.
   - Cancels each accepted prompt immediately, since the placeholder weights can't load.
4. **`.github/workflows/comfyui-compat.yaml`:**
   - Checks out ComfyUI at either the pinned revision or the latest release tag (from the GitHub releases API).
   - Installs CPU-only torch, torchvision, and torchaudio from one index, then ComfyUI's requirements. ComfyUI requires `torchaudio`, and the PyPI build fails to load against CPU torch. Then starts ComfyUI with `--cpu --listen 127.0.0.1`.
   - Runs the script.
   - Triggers:
     - The **pinned** run on pull requests that touch `diffusion/`, `COMFYUI_REVISION`, or the workflow file, so our own workflow-builder mistakes are caught.
     - The **latest** run weekly and on manual dispatch, so upstream breakage shows up before anyone bumps the pin.
   - `permissions: contents: read`. A failure shows as a failed scheduled run. Opening an issue automatically would need `issues: write`; leave that out unless wanted.
5. **Bumping the pin** is a PR that edits `COMFYUI_REVISION`. That PR's pinned run is the gate.

**Gate:** both runs green against the current pin and the current latest
release; `make ci` green.

**Result (2026-10-31):** the script passes against a real CPU ComfyUI at `feca51a8` and at
v0.43.0, following the steps in the workflow file. As a negative control, a bad `sampler_name`
returns HTTP 400 with `node_errors`. The workflow file itself runs for the first time on the
first push.

### M5. Cutover (ops, no code) — **not done**

**Update (2026-10-31): the image flow has never been run against ComfyUI or deployed**, confirmed by
the owner and consistent with the repo: the `k8s/` manifests and Dockerfile omit the `diffusion`
extra, set no image env, and have no sidecar. There are no clients to cut over, jobs to lose, or
ingress rule to swap, and the soak period protects nothing. The gate for M6 becomes this: **one
real generation per model that will be used, on a GPU, through `make docker-up` in imagenate, with
the output image inspected.** The steps below apply only if a deployment is added first.

1. Deploy the new service with the current image config translated to `IMAGENATE_*` names per the rename table, and the same secret values (API keys, moderation key).
2. Add the `/v1/images` ingress rule. NGINX ingress picks the longest matching Prefix path, so it takes precedence over `/`.
3. Verify through the public host: `/v1/images/models` and one `generate` per model with real weights on a GPU. Nothing so far has done this.
4. Disable images in sentimentizer by unsetting `SENTIMENTIZER_KREA_2_ENABLED` / `SENTIMENTIZER_IDEOGRAM_4_ENABLED` and `--diffusion`, then redeploy it.

**Rollback:** remove the ingress rule and re-enable images in sentimentizer.
Nothing in sentimentizer has been deleted at this point.

**Known loss:** async jobs created before the cutover are not visible to the
new service. `JobStore` is in-memory per Ray cluster, so `GET /v1/images/jobs/{id}`
for them returns 404 until clients stop polling. Any restart loses them today
too. Cut over in a quiet window.

### M6. Remove image code from sentimentizer (sentimentizer, ~0.5 day) — **ready, do not merge before M5**

Only after M5 has run cleanly for an agreed soak period. The change is on the
`remove-image-service` branch, so merging it is the step that waits.

1. Code: every row of "Sentimentizer touchpoints that change" above.
   - `--diffusion` becomes a hard error.
   - `load_serve_config()` logs one warning per removed image env var that is still set, so a stale deployment does not lose images silently (R7).
2. Tests:
   - Delete the moved tests.
   - Move `TestRequestBodySizeLimitMiddleware` from `test_middleware.py` into `test_serve.py` (next to `TestBodySizeLimitMiddleware`), then delete `test_middleware.py`, whose top-level import of `serve/middleware.py` would otherwise break.
   - Update the `test_serve.py` fixtures.
   - Add tests that `--diffusion` exits non-zero with the message, that a legacy image env var produces a warning, and that `/health/ready` no longer has image keys.
3. Remove the `diffusion` extra.
4. Run `uv lock` and review the diff. Run `uv lock --check`. If the lock churns unrelated markers, hand-edit as in #74.
5. Bump the version.

**Gate:** `make ci` green.

### M7. Documentation (sentimentizer, docs only) — **ready, do not merge before M5**

1. Update `AGENTS.md`, README, `docs/serving.md`, `docs/configuration.md`, `.rules/observability.md`, `.rules/prompts-and-constants.md`, and `.skills/plan-feature/SKILL.md` as listed above. Leave a short pointer to the new repo in `docs/serving.md`.
2. Set this plan's status to `✅ Completed` in the same change.

**Gate:** `git diff --check`.

## Testing strategy

| Module (repo) | Covers |
|---|---|
| `test_comfyui.py` (new, moved) | Workflow builders, client polling/cancel/temp-file cleanup, moderation client, `_backend_http_exception`, licensing gates (now via `ImagesConfig`). |
| `test_diffusion_jobs.py` (new, moved) | `JobStoreLogic` lifecycle, TTL reaping, pagination, auth scoping, Ray actor visibility. |
| `test_middleware.py` (new, moved) | Auth, token bucket, idempotency cache, prompt safety, error-envelope integration. |
| `test_http.py` (new, copied) | Per-path body-size limits in the copied `create_fastapi_app`. |
| `test_generate_request.py`, `test_auth_rate_limit.py` (new) | Classes ported from `test_serve.py`. |
| `test_config.py` (new) | Every `ImagesConfig` validation error. Env override and type coercion for each field. `IMAGENATE_SERVE_CONFIG` path precedence. Old `SENTIMENTIZER_*` names are ignored and produce one warning each. |
| `test_ops_routes.py` (new) | `/health/live`; `/health/ready` 200 and 503 (ComfyUI down; job store missing) with the expected keys; `/metrics` prefix and counters. |
| `test_packaging.py` (new) | torch is not installed; `SENTIMENTIZER_` appears only in the legacy warning list. |
| `test_compat_script.py` (new) | Placeholders land in the right directories and never overwrite real weights; each failure is reported. |
| `test_entrypoint.py` (new) | `RAY_ENABLE_UV_RUN_RUNTIME_ENV` set at package import; a fresh process that never imports `serve/app.py` accepts a key set only in YAML; signal handlers before Ray; `main()` calls `ray.init(address="local", ...)` and `HTTPOptions(request_timeout_s=...)` (mock `ray`/`serve`); `ValueError` when no model is enabled. |
| `test_openapi_parity.py` (new) | `/v1/images` OpenAPI schema equals the snapshot taken from sentimentizer. |
| `test_comfyui.py` additions (new) | `validate_comfyui_runtime()` with a mocked client: missing node, missing checkpoint, and all present. These cover the logic the compatibility job exercises against a real ComfyUI. |
| `comfyui-compat.yaml` (CI, new) | Real ComfyUI, pinned and latest: node classes, checkpoint choices, and `/prompt` graph validation for every enabled model's workflow. |
| `test_serve.py` (sentimentizer) | Fixtures without image attributes; `/health/ready` without image keys. |
| `test_bge_only_serve.py`, next to the existing `build_parser` tests (sentimentizer) | `--diffusion` and `SENTIMENTIZER_DIFFUSION_ENABLED` exit non-zero with the pointer message. |
| `test_serve_config` cases in `test_serve.py` (sentimentizer) | Warning when a removed image env var is set. `ServeConfig` no longer accepts image keys from YAML. Decide whether stale keys in a custom YAML are an error or a warning; recommend warning, to match env. |

## Risk review

| # | Risk | Resolution |
|---|---|---|
| R1 | **Ray cluster collision on a shared host.** `ray.init()` with no address joins the cluster recorded in `/tmp/ray/ray_current_cluster` (verified in Ray 2.55.1). `ray start` writes that file and `ray.init()` does not, so this applies only if someone ran `ray start`. The image service would then attach to it, and its `serve.start()` would hit an existing Serve instance whose HTTP options are already fixed. Confirmed in M3: sentimentizer and imagenate ran side by side as two clusters. | `ray.init(address="local")`, its own namespace, and a test asserting the call. Prefer its own container (D2). |
| R2 | **Sentimentizer training kills co-located Ray processes.** `workflows/lifecycle.py::_kill_stale_ray_processes` runs `pkill` on every `raylet`, `gcs_server` and `ray::IDLE` on the host. `_cleanup_stale_ray_sessions` deletes `/tmp/ray/session_*` dirs whose mtime is more than 1 h old, which can include a live long-running session. | Run the image service in its own container (separate PID namespace and `/tmp`). On a bare host, pass `_temp_dir` to `ray.init` outside `/tmp/ray` and document that training must not share the host. This hazard already exists for a long-running sentiment Serve process; note it in `docs/troubleshooting.md`. |
| R3 | **Readiness gap.** Today image readiness is reported only through sentiment's `/health/ready`; the images app has no health routes. | New ops app (M2), probes (M4). |
| R4 | **API drift during the move.** | Golden OpenAPI snapshot (M3) plus a side-by-side smoke run. |
| R5 | **Body-size mismatch at the edge.** The images app accepts 4 MiB, but the shared ingress caps all paths at `1m`. | The new ingress sets `4m` for `/v1/images` (M4). This is a pre-existing mismatch, so the new limit is deliberately the app's intended one. Confirm with the owner. |
| R6 | **Auth breaks silently.** `require_api_key` reads the API-key env var from the process env, which `serve/app.py` populates from config at import time. Replicas get that only because `diffusion_app.py` imports `serve/app.py` today, and copying `create_fastapi_app` removes that import. | Remove the export. `_load_valid_keys()` reads the config in each process. `test_entrypoint.py` proves it in a fresh process, and the M3 smoke run proved it in real replicas. |
| R7 | **Images disappear silently after M6.** Operators still setting `SENTIMENTIZER_KREA_2_ENABLED` or passing `--diffusion` would start sentiment without images and no error. | `--diffusion` exits non-zero with a pointer. Legacy image env vars warn at startup. Keep both for one release. |
| R8 | **Old env names are ignored.** An operator who copies the current deployment env to the new service unchanged gets every image model disabled. `main()` then refuses to start with the existing "requires an explicitly enabled image model" error, which doesn't point at the rename. | The rename is listed in the cutover steps (M5). At startup, before that check, `imagenate` logs one warning per old `SENTIMENTIZER_*` name from the rename table that is set, for one release, so the cause is obvious. Separate prefixes also mean the two services can't pick up each other's host or port settings on a shared host. |
| R9 | **`Location` header prefix.** `create_job` builds `Location` from `request.scope["root_path"]`. | Keep `route_prefix="/v1/images"` on Ray Serve; covered by the smoke run. |
| R10 | **Lost async jobs at cutover.** | Accept and document (M5). |
| R11 | **Request timeout dropped.** `http_options={"request_timeout_s": ...}` is silently ignored. | Use `HTTPOptions` (asserted in `test_entrypoint.py`). |
| R12 | **Dependency drift.** `fastapi==0.136.1` is pinned because of a Ray Serve 2.55.1 pickling issue. | Copy the pin and its comment. Upgrade only together with Ray. |
| R13 | **Lockfile churn when removing the `diffusion` extra.** | Review the `uv lock` diff; hand-edit if unrelated markers change (as in #74). |
| R14 | **Optional heavy deps.** | The new service needs no torch. Add a CI step that asserts `import torch` fails in its environment, so the dependency can't creep back through a transitive import. The compatibility job's CPU torch is installed only into ComfyUI's own environment. |
| R15 | **ComfyUI drift.** New releases rename or remove nodes, inputs, or enum values, and today that surfaces only at startup. | M4b: weekly run against the latest release, a pinned run on relevant PRs, and one pin file. |
| R16 | **Compatibility job gives false confidence.** It proves the graph is accepted, not that real weights load or images look right. | State this in the job's docs. Bumping the pin still needs one real generation per model on a GPU host before release. |
| R17 | **Fresh lockfile breaks Ray.** The new repo resolves dependencies from scratch, and sentimentizer's lock had been holding them back. protobuf 7 removed `FieldDescriptor.label`, so `serve.run()` fails in the controller. All unit tests pass, because they mock Ray. | Cap `protobuf<7` with the reason beside it. Found by the M3 real start, which is why `AGENTS.md` requires one for changes to wiring or dependencies. |
| R18 | **More than one pod breaks clients.** Jobs, rate limits, and idempotency keys are in memory, so a job created on one pod returns 404 from another. Copying sentimentizer's `replicas: 2` would do this. A PDB with `minAvailable: 1` on one replica would block node drains. | `replicas: 1`, `strategy: Recreate`, no PDB. Rollouts cause a brief outage, which is acceptable until the follow-up replaces the state. |
| R19 | **Two ingresses fight over one certificate.** A second ingress with the cert-manager annotation for the same host and secret creates a competing Certificate. | The new ingress references the secret and has no annotation. It must be in the same namespace as the sentimentizer ingress. |
| R20 | **Proxy timeout cuts off generation.** NGINX's default read timeout is 60 s, far below `request_timeout_s` of 600. | Set `proxy-read-timeout` and `proxy-send-timeout` on the new ingress. |

### Repo-specific risk classes from `.skills/plan-feature` that do not apply

3-class contract, base model classes, the scheduler, the training metrics
pipeline, the lazy router, ONNX export, and metrics-file races: the moved code
touches none of them. Sentiment `/health/ready` changes shape only by losing
optional keys.

## Assumptions

| Assumption | Verified |
|---|---|
| Nothing outside `diffusion_app.py` imports `serve/middleware.py`. | `grep`, 2026-10-31 |
| Nothing outside `serve/app.py`, `serve/__main__.py`, `serve/config.py`, `safety.py` and the moved files references diffusion or ComfyUI code. | `grep`, 2026-10-31 |
| `ray.init(address="local")` always starts a new local instance. | Ray 2.55.1 source and the M3 run, 2026-10-31 |
| NGINX ingress prefers the longest Prefix match, so `/v1/images` beats `/`. | Documented NGINX ingress behavior; confirm on the target cluster during M5. |
| ComfyUI's `POST /prompt` validates the graph (node types, input names, enum values) and returns `node_errors` before queueing. | Real ComfyUI at `feca51a8` and v0.43.0, 2026-10-31 (negative control rejected) |
| Empty placeholder files appear in `/object_info` loader choices, and ComfyUI starts with `--cpu`. | Real ComfyUI at `feca51a8` and v0.43.0, 2026-10-31. Run with CUDA hidden, not on a GPU-less runner; the first CI run confirms that. |
| The ComfyUI and moderation clients need no third-party HTTP library. | Both use `urllib`; read 2026-10-31 |
| `prompt_blocklist_path` is unused. | `grep`, 2026-10-31 |
| The `comfyui_images` metrics collector is never rendered. | `grep` for `to_prometheus` / `/metrics`, 2026-10-31 |

## Convention updates

- **sentimentizer `AGENTS.md`:**
  - Remove the "Headless image generation" section and the `diffusion/` and `serve/diffusion_*` entries from the architecture tree.
  - Remove `/v1/images/*` and `route_prefix="/v1/images"` from the Serving API bullets, and the image example from the trailing-slash bullet.
  - Drop `diffusion` from the extras list and CI install line.
  - Add a one-line pointer to the new repo.
- **`.rules/observability.md`:** drop ComfyUI from the external-call examples.
- **`.rules/prompts-and-constants.md`:** remove the diffusion constants entry.
- **New repo `AGENTS.md`:** states the `IMAGENATE_*` env prefix and carries the moved "Headless image generation" section, the relevant Serving API bullets (`HTTPOptions`, trailing slashes, separate FastAPI app per deployment, no `__call__`), the `RAY_ENABLE_UV_RUN_RUNTIME_ENV` rule, and R1/R2 as operating rules.
- **`docs/troubleshooting.md` (sentimentizer):** note R2 (training's Ray cleanup kills other Ray processes on the host).

## Pre-existing issues found during discovery

None of these are fixed by this plan. Each is listed with how the plan handles it.

- **Shared ingress body limit (1m) is below the images app limit (4 MiB).** Addressed only for the new ingress (R5).
- **`comfyui_images` metrics are collected but never exposed.** The new `/metrics` exposes them through a handle call (see Ops routes). The dispatcher has no collector of its own.
- **The `k8s/` manifests and Dockerfile do not serve images today.** See M5.
- **`prompt_blocklist_path` is a dead config field.** Dropped in the new config; removed from sentimentizer in M6.
- **`.rules/plan-review.md` asks for a revise-and-review pass; the review log below records it.**

## Review log

**Pass 1 (first draft).** The draft moved the files, copied
`create_fastapi_app`, and switched ingress. Review found:

- (a) The images app has no health or metrics routes, because readiness comes from sentiment's `/health/ready`. Added the ops app (R3).
- (b) `ray.init()` with no address would join sentiment's cluster on a shared host (R1).
- (c) Training's Ray cleanup can kill or delete a co-located cluster (R2).
- (d) `SENTIMENTIZER_API_KEYS` is exported at import time in `serve/app.py`; missing it makes every request 401 (R6).
- (e) Removing image code before cutover leaves no rollback. Reordered so removal (M6) follows cutover (M5).
- (f) Operators with stale image env would lose images silently (R7).
- (g) No objective parity check. Added the OpenAPI snapshot and smoke run (M3).
- (h) The ingress body limit is below the app limit (R5).

**Pass 2 (after revision).** Re-checked the moved-file list against imports.
`TestErrorEnvelopeIntegration` uses `check_prompt_safety`, so it moves; only
`TestRequestBodySizeLimitMiddleware` stays, and it needs a new home in
sentimentizer because `test_middleware.py` imports the deleted module.
`middleware._get_limiter()` imports `load_serve_config`, so it was added to M2.
`test_comfyui.py` imports `ServeConfig`, so its licensing-gate tests now go
through `ImagesConfig` in M2. Corrected the `/health/live` shape to match
sentimentizer's (`"live"` plus `uptime_s`). No unresolved gaps remain that would block implementation.

**Pass 3 (naming decisions, 2026-10-31).** D1 settled on `imagenate`, and D3
changed from keeping `SENTIMENTIZER_*` to renaming during the move, because the
service is unrelated to sentiment. Review of that change:

- (a) The API-key env name is read in `middleware.py` as well as exported in `serve/app.py`; both must change together (R6).
- (b) `diffusion/config.py` builds `SENTIMENTIZER_DIFFUSION_*` names, which are easy to miss with a field-by-field rename. Covered by the CI string check in M2.
- (c) Copying the old env unchanged would leave every model disabled, and startup would fail with an error that doesn't mention the rename. Added the old-name startup warning and the cutover translation step (R8, M5).
- (d) The old R8 (shared host/port names colliding on one host) no longer applies once the prefixes differ.
- (e) Added D5 so the hostname move stays out of the cutover.

**Pass 4 (ComfyUI compatibility check, 2026-10-31).** Added M4b. Review:

- (a) The node-class check alone misses renamed inputs, so the job also submits each built workflow to `/prompt`.
- (b) Copying `_validate_runtime` into a script would create a second implementation that drifts; extracting it into a function keeps one (`.rules/dry.md`).
- (c) The pin lived in two docs; moved to `COMFYUI_REVISION`.
- (d) The job can't prove real weights work, so R16 keeps a manual GPU check when bumping the pin.
- (e) Two ComfyUI behaviors it relies on are unverified and listed as assumptions.

**Pass 5 (execution, 2026-10-31).** Running the plan found these. Each is now in the sections above.

- (a) The module-level API-key export would not have run in replicas once `create_fastapi_app` moved. Every request would have returned 401 unless the key came from env inherited through the raylet (R6).
- (b) `/metrics` over "the dispatcher's collectors" had nothing to render, and the obvious fix of binding `ComfyUIDeployment` to the ops app would have broken queue serialization.
- (c) A fresh lockfile picks up protobuf 7, which stops Serve from starting (R17). Only a real start showed this.
- (d) The k8s section would have copied `replicas: 2` and a PDB onto in-memory state and a single GPU (R18). It also omitted the sidecar image, the second certificate conflict (R19), and the proxy timeout (R20).
- (e) M1 could not pass its own gate before M2's config existed. The repo's existing initial commit also needed an unrelated-histories merge.
- (f) Copying sentimentizer's `.gitignore` would have dropped the JSON fixture, since it ignores `*.json`.
- (g) Discovery missed `test_safety.py`, `ImageModelsStatus`, the dead `is_safe` after M6, and the `LOG_LEVEL` rename. R1 was broader than Ray's behavior: `ray.init()` does not record its cluster.
- (h) **Every ComfyUI failure returned 500 and stranded async jobs in `processing`.** Serve pickles a deployment class when its decorator runs, so `_backend_http_exception`, defined below the classes, raised `NameError` in the replica. This is a pre-existing bug that sentimentizer still has. The M3 comparison missed it because the stand-in never failed a workflow; running against a real ComfyUI found it. Fixed in imagenate with a regression test, so failures now differ from sentimentizer by design: 502 instead of 500.
- (i) The compatibility job left `torchaudio` to PyPI, which fails to load against CPU torch. Running the job's steps found this before it reached CI.

## Follow-up: drop Ray Serve

After the new service has run in production, this becomes a separate plan.
Replace the two single-replica deployments with one FastAPI/uvicorn process:

- `JobStoreLogic` is already Ray-free (`job_store.py:46`); use it in-process, optionally backed by SQLite so jobs survive restarts.
- An `asyncio.Lock` (already present in `ComfyUIDeployment`) serializes ComfyUI submissions.
- Cancellation becomes `asyncio.Task.cancel()` plus the existing ComfyUI `/api/jobs/{id}/cancel`.

R1, R2, R11 and R12 go away with it. Estimated at 3-4 days.

## Effort summary

| Milestone | Repo | Estimate |
|---|---|---|
| M1 Bootstrap with history | new | 0.5-1 day |
| M2 Config, env rename and entry point | new | 0.5-0.75 day |
| M3 Parity check | both | 0.5 day |
| M4 Packaging and deployment | new | 0.5-1 day |
| M4b ComfyUI compatibility check | new | 0.5 day |
| M5 Cutover | ops | under 1 hour plus soak time |
| M6 Remove from sentimentizer | sentimentizer | 0.5 day |
| M7 Docs | sentimentizer | 1-2 hours |
| **Total** | | **about 3.5-4.5 days of work**, plus soak time |
