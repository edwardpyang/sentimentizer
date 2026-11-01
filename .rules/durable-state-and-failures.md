## Brief overview

Other processes, later runs, and shell scripts all rely on files and exit codes. A crash
midway must not look like success, and neither should a failure that was only logged.

## Publish files atomically

- Write to a temporary file in the **same directory**, then `Path.replace()` it onto the
  destination. A different filesystem makes the rename non-atomic or fails it.
- Metrics JSON goes through `write_json_atomic()` in `metrics_publisher.py`. Do not call
  `path.write_text()` on a file that the exporter or another process reads.
- Set permissions on temporary files that another user may read:
  `NamedTemporaryFile` creates them as `0600`.

## Existence must mean complete

If code skips work because a path exists, that path must only exist once the work is done.

- Download a directory to a staging location, then rename it into place
  (`SentimentPredictor._download_router_artifact`).
- Write the marker last. Weights are the marker for a model download, so the backbone is
  fetched before the weights (`download_weights`).
- Separate "optional file is absent" from "download failed". Only the first may be skipped.

## Failures must be visible

- A CLI stage that fails raises, so the process exits nonzero. `logger.error()` followed by
  a normal return makes `make upload-*` and CI report success.
- Library functions that log instead of raising return a result the caller must check.
- Isolate failures per item in loops over independent inputs. One corrupt metrics file must
  not stop the other model types from reporting, especially after gauges were cleared.
- Do not log a success message on a path where steps may have failed.

## Degraded states

When a dependency is optional or failed to load, trace every handler that uses it, not only
the main one. For example, an HF model owns its tokenizer, so a failed load also breaks
`/v1/sentiment/tokenize`. A known degraded state returns 503, not an `AttributeError` 500.

## Middleware order

`app.add_middleware` and `@app.middleware` insert at the outside, so the last one added runs
first. The order is part of the contract in `AGENTS.md`; assert it through response headers
in a test. Starlette runs `exception_handler(Exception)` outside all user middleware, so
convert unhandled errors inside the request-ID middleware or they lose their CORS headers.
