## Brief overview

Rule for observability: what to log, how to log it, and what a timeout/error
must carry so a user staring at a stack trace can tell *where the time went*
without re-running with a debugger. Codifies the lessons from training runs,
Ray Serve cold starts, and metrics pipeline debugging.

## Logging conventions

- **Structured logging via the project logger.** Workflows get a logger from
  `workflows/lifecycle.py`. Ray workers use the `ray.util.logger` module.
  No bare `print` for service/training code.
- **Key/value fields, not f-strings.** Pass structured data as extra fields or
  format strings that include labeled values, not bare concatenations.
  `logger.info("training epoch complete", extra={"epoch": 3, "val_loss": 0.42,
  "duration_s": 12.3})` — fields are first-class and grep-able.
- **One logger per module, named `__name__`.** Training, serving, and metrics
  each get their own named loggers.

## What to log at each boundary

- **Training boundary (epoch end).** Log epoch number, train/val loss and
  metrics, duration, and learning rate. The metrics files in
  `/tmp/sentimentizer_metrics/` are the durable artifact; logs are for
  diagnosis.
- **Ray Serve boundary (request).** Log the endpoint hit, model used,
  latency, and any error. Ray Serve emits access logs; application handlers
  log domain events only ("prediction complete", "router loaded").
- **Ray Train boundary (worker lifecycle).** Log worker start/end with rank,
  world size, and device. A hung driver must be distinguishable from a hung
  worker.
- **External call boundary (Hugging Face).** Before calling an
  external service, log what's about to happen (model name, URL, endpoint).
  After, log the outcome (bytes downloaded, status, duration_s).
- **Metrics pipeline boundary.** Log when metric files are written, reset, or
  skipped. A stale gauge must be traceable to the file write that should have
  cleared it.

## Timeout and budget errors must carry context

A bare `TimeoutError("training did not complete")` is useless — the user can't
tell whether the time went to data loading, a slow epoch, or a checkpoint write.
**Every timeout or budget-exceeded error MUST include:**

- **What was being done** (model type, epoch, batch number, endpoint).
- **How far it got** (batches processed, samples seen, phase reached).
- **A hint about the likely culprit** when the timeout wraps a multi-stage
  operation (data loading, GPU contention, checkpoint I/O).

## Duration fields

- **Always `duration_s` (float) or `duration_ms` (int).** Field name is the
  contract — grep/aggregation depends on it. Compute as
  `time.monotonic() - start` or `int((time.monotonic() - start) * 1000)`.
- **Use `time.monotonic()` for durations, never `time.time()`.** Wall-clock
  time can jump (NTP adjustments, DST); monotonic time only moves forward.
- Log the duration at the *end* of the operation, not the start.

## What NOT to log

- **Secrets.** API keys, HuggingFace tokens, model paths that contain
  credentials. Log the operation, not the key.
- **Full request/response bodies.** Log shapes (`batch_size=16`,
  `seq_len=128`), not contents. Training data and predictions are the durable
  artifacts; logs are for diagnosis, not replay.
- **High-frequency noise.** A per-batch log line in a 500-batch epoch is noise.
  Log every N batches (the batch snapshot cadence: 10 for ModernBERT, 50
  otherwise) and at epoch end.
- **Pydantic validation errors at INFO.** A 422 from a missing param is a
  client bug, not a service event — log at WARNING, and only the field names,
  not the full input.

## Ray-specific notes

- **Ray session logs** live in `/tmp/ray/session_latest/logs/`. When Ray fails
  to start or a worker crashes, check those before re-running.
- **Ray gauge updates** in `trainer.py` are the observability surface during
  training. The gauges are Prometheus-facing; do not add per-step logging that
  duplicates gauge data.
