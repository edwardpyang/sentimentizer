"""Standalone Prometheus metrics exporter for Sentimentizer.

Exposes system, GPU, and application metrics on a configurable port
for Prometheus scraping. This ensures the Grafana dashboard always has
data to display, even when Ray is not running.

Usage:
    uv run python sentimentizer/exporter.py [--port 8081] [--interval 10]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from prometheus_client import Gauge, Info, start_http_server

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Persisted metrics directory (per-model-type files)
# ──────────────────────────────────────────────

_METRICS_DIR = Path("/tmp/sentimentizer_metrics")


def get_training_metrics_path(model_type: str) -> Path:
    """Return the per-model-type path for persisted training metrics.

    Each model type (rnn, encoder, decoder, modernbert) writes to its own JSON file to
    eliminate race conditions when multiple training processes run concurrently.
    """
    return _METRICS_DIR / f"{model_type}_metrics.json"


# ──────────────────────────────────────────────
# System metrics
# ──────────────────────────────────────────────

SYSTEM_CPU_PERCENT = Gauge(
    "sentimentizer_system_cpu_percent",
    "System CPU usage percentage",
)

SYSTEM_MEMORY_PERCENT = Gauge(
    "sentimentizer_system_memory_percent",
    "System memory usage percentage",
)

SYSTEM_MEMORY_AVAILABLE_BYTES = Gauge(
    "sentimentizer_system_memory_available_bytes",
    "Available system memory in bytes",
)

SYSTEM_MEMORY_TOTAL_BYTES = Gauge(
    "sentimentizer_system_memory_total_bytes",
    "Total system memory in bytes",
)

SYSTEM_DISK_PERCENT = Gauge(
    "sentimentizer_system_disk_percent",
    "System disk usage percentage for root partition",
)

SYSTEM_DISK_FREE_BYTES = Gauge(
    "sentimentizer_system_disk_free_bytes",
    "Free disk space in bytes for root partition",
)

SYSTEM_DISK_TOTAL_BYTES = Gauge(
    "sentimentizer_system_disk_total_bytes",
    "Total disk space in bytes for root partition",
)

SYSTEM_INFO = Info(
    "sentimentizer_system",
    "System information",
)

# ──────────────────────────────────────────────
# GPU metrics
# ──────────────────────────────────────────────

GPU_UTILIZATION = Gauge(
    "sentimentizer_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_index", "gpu_name"],
)

GPU_MEMORY_USED_BYTES = Gauge(
    "sentimentizer_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_index", "gpu_name"],
)

GPU_MEMORY_TOTAL_BYTES = Gauge(
    "sentimentizer_gpu_memory_total_bytes",
    "GPU memory total in bytes",
    ["gpu_index", "gpu_name"],
)

GPU_TEMPERATURE_CELSIUS = Gauge(
    "sentimentizer_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_index", "gpu_name"],
)

# ──────────────────────────────────────────────
# Ray Tune trial metrics
# ──────────────────────────────────────────────

TUNE_TRIAL_VAL_ACCURACY = Gauge(
    "sentimentizer_tune_val_accuracy",
    "Validation accuracy reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_LOSS = Gauge(
    "sentimentizer_tune_val_loss",
    "Validation loss reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_TRAIN_LOSS = Gauge(
    "sentimentizer_tune_train_loss",
    "Training loss reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_F1 = Gauge(
    "sentimentizer_tune_val_f1",
    "Validation F1 score reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_COHEN_KAPPA = Gauge(
    "sentimentizer_tune_val_cohen_kappa",
    "Validation Cohen's kappa reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_BALANCED_ACCURACY = Gauge(
    "sentimentizer_tune_val_balanced_accuracy",
    "Validation balanced accuracy reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_NEGATIVE_F1 = Gauge(
    "sentimentizer_tune_val_negative_f1",
    "Validation negative-class F1 reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_NEUTRAL_F1 = Gauge(
    "sentimentizer_tune_val_neutral_f1",
    "Validation neutral-class F1 reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_POSITIVE_F1 = Gauge(
    "sentimentizer_tune_val_positive_f1",
    "Validation positive-class F1 reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_MACRO_F1 = Gauge(
    "sentimentizer_tune_val_macro_f1",
    "Validation macro-averaged F1 reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_WEIGHTED_F1 = Gauge(
    "sentimentizer_tune_val_weighted_f1",
    "Validation weighted-averaged F1 reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_MCC = Gauge(
    "sentimentizer_tune_val_mcc",
    "Validation Matthews correlation coefficient reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_EPOCH = Gauge(
    "sentimentizer_tune_epoch",
    "Current training epoch for the Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_BEST_VAL_ACCURACY = Gauge(
    "sentimentizer_tune_best_val_accuracy",
    "Best validation accuracy across all Ray Tune trials",
    ["model_type"],
)

TUNE_BEST_VAL_LOSS = Gauge(
    "sentimentizer_tune_best_val_loss",
    "Best validation loss across all Ray Tune trials",
    ["model_type"],
)

TUNE_BEST_VAL_F1 = Gauge(
    "sentimentizer_tune_best_val_f1",
    "Best validation F1 score across all Ray Tune trials",
    ["model_type"],
)

TUNE_TRIAL_COUNT = Gauge(
    "sentimentizer_tune_trial_count",
    "Total number of Ray Tune trials",
    ["model_type"],
)

TUNE_TRIAL_COMPLETED_COUNT = Gauge(
    "sentimentizer_tune_trial_completed_count",
    "Number of completed Ray Tune trials",
    ["model_type"],
)

# ──────────────────────────────────────────────
# Training model performance metrics
# ──────────────────────────────────────────────

TRAINING_TRAIN_LOSS = Gauge(
    "sentimentizer_training_train_loss",
    "Training loss from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_LOSS = Gauge(
    "sentimentizer_training_val_loss",
    "Validation loss from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_ACCURACY = Gauge(
    "sentimentizer_training_val_accuracy",
    "Validation accuracy from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_COHEN_KAPPA = Gauge(
    "sentimentizer_training_val_cohen_kappa",
    "Validation Cohen's kappa from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_MCC = Gauge(
    "sentimentizer_training_val_mcc",
    "Validation Matthews correlation coefficient from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_BALANCED_ACCURACY = Gauge(
    "sentimentizer_training_val_balanced_accuracy",
    "Validation balanced accuracy from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEGATIVE_PRECISION = Gauge(
    "sentimentizer_training_val_negative_precision",
    "Validation negative-class precision from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEGATIVE_RECALL = Gauge(
    "sentimentizer_training_val_negative_recall",
    "Validation negative-class recall from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEGATIVE_F1 = Gauge(
    "sentimentizer_training_val_negative_f1",
    "Validation negative-class F1 from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEUTRAL_PRECISION = Gauge(
    "sentimentizer_training_val_neutral_precision",
    "Validation neutral-class precision from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEUTRAL_RECALL = Gauge(
    "sentimentizer_training_val_neutral_recall",
    "Validation neutral-class recall from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEUTRAL_F1 = Gauge(
    "sentimentizer_training_val_neutral_f1",
    "Validation neutral-class F1 from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_POSITIVE_PRECISION = Gauge(
    "sentimentizer_training_val_positive_precision",
    "Validation positive-class precision from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_POSITIVE_RECALL = Gauge(
    "sentimentizer_training_val_positive_recall",
    "Validation positive-class recall from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_POSITIVE_F1 = Gauge(
    "sentimentizer_training_val_positive_f1",
    "Validation positive-class F1 from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_MACRO_F1 = Gauge(
    "sentimentizer_training_val_macro_f1",
    "Validation macro-averaged F1 from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_WEIGHTED_F1 = Gauge(
    "sentimentizer_training_val_weighted_f1",
    "Validation weighted-averaged F1 from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEUTRAL_AUC_ROC = Gauge(
    "sentimentizer_training_val_neutral_auc_roc",
    "Validation neutral-class AUC-ROC from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEUTRAL_AVG_PRECISION = Gauge(
    "sentimentizer_training_val_neutral_avg_precision",
    "Validation neutral-class average precision from the current training run",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEUTRAL_TO_POSITIVE_RATE = Gauge(
    "sentimentizer_training_val_neutral_to_positive_rate",
    "Fraction of true neutral reviews misclassified as positive",
    ["model_type", "run_id"],
)

TRAINING_VAL_NEUTRAL_TO_NEGATIVE_RATE = Gauge(
    "sentimentizer_training_val_neutral_to_negative_rate",
    "Fraction of true neutral reviews misclassified as negative",
    ["model_type", "run_id"],
)

TRAINING_VAL_PRED_NEUTRAL_FRAC = Gauge(
    "sentimentizer_training_val_pred_neutral_frac",
    "Fraction of predictions that are neutral",
    ["model_type", "run_id"],
)

TRAINING_EPOCH = Gauge(
    "sentimentizer_training_epoch",
    "Current training epoch number",
    ["model_type", "run_id"],
)

TRAINING_LR = Gauge(
    "sentimentizer_training_lr",
    "Current learning rate",
    ["model_type", "run_id"],
)

TRAINING_TRAIN_LOSS_EMA = Gauge(
    "sentimentizer_training_train_loss_ema",
    "Fast-moving EMA training loss (updated every N batches during training)",
    ["model_type", "run_id"],
)

TRAINING_TRAIN_LOSS_AVG = Gauge(
    "sentimentizer_training_train_loss_avg",
    "Slow-moving epoch-average training loss (updated every N batches during training)",
    ["model_type", "run_id"],
)

TRAINING_BATCH = Gauge(
    "sentimentizer_training_batch",
    "Current batch number within the epoch during training",
    ["model_type", "run_id"],
)

TRAINING_GRAD_NORM = Gauge(
    "sentimentizer_training_grad_norm",
    "Gradient norm before clipping (updated every N batches)",
    ["model_type", "run_id"],
)

TRAINING_THROUGHPUT = Gauge(
    "sentimentizer_training_throughput",
    "Samples processed per second (updated every N batches)",
    ["model_type", "run_id"],
)

# ──────────────────────────────────────────────
# Ray health metrics
# ──────────────────────────────────────────────

RAY_AVAILABLE = Gauge(
    "sentimentizer_ray_available",
    "Whether Ray cluster is reachable (1=up, 0=down)",
)

RAY_NODE_COUNT = Gauge(
    "sentimentizer_ray_node_count",
    "Number of Ray nodes detected from metrics",
)

RAY_METRIC_COUNT = Gauge(
    "sentimentizer_ray_metric_count",
    "Number of Ray metric series found",
)

RAY_CONTROLLER_STATE = Gauge(
    "sentimentizer_ray_controller_state",
    "Ray controller state",
)

RAY_CONTROLLER_OPERATION_TIME = Gauge(
    "sentimentizer_ray_controller_operation_time_s",
    "Ray controller operation time",
)


def _update_system_metrics() -> None:
    """Update system metric gauges from psutil."""
    SYSTEM_CPU_PERCENT.set(psutil.cpu_percent(interval=0))

    mem = psutil.virtual_memory()
    SYSTEM_MEMORY_PERCENT.set(mem.percent)
    SYSTEM_MEMORY_AVAILABLE_BYTES.set(mem.available)
    SYSTEM_MEMORY_TOTAL_BYTES.set(mem.total)

    disk = psutil.disk_usage("/")
    SYSTEM_DISK_PERCENT.set(disk.percent)
    SYSTEM_DISK_FREE_BYTES.set(disk.free)
    SYSTEM_DISK_TOTAL_BYTES.set(disk.total)


def _update_gpu_metrics() -> None:
    """Update GPU metric gauges from nvidia-smi if available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            gpu_index = parts[0]
            gpu_name = parts[1]
            try:
                utilization = float(parts[2]) if parts[2] != "[N/A]" else 0
                mem_used_mib = float(parts[3]) if parts[3] != "[N/A]" else 0
                mem_total_mib = float(parts[4]) if parts[4] != "[N/A]" else 0
                temperature = float(parts[5]) if parts[5] != "[N/A]" else 0
            except ValueError:
                continue

            GPU_UTILIZATION.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(utilization)
            GPU_MEMORY_USED_BYTES.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(
                mem_used_mib * 1024 * 1024
            )
            GPU_MEMORY_TOTAL_BYTES.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(
                mem_total_mib * 1024 * 1024
            )
            GPU_TEMPERATURE_CELSIUS.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(temperature)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _update_ray_metrics(ray_url: str) -> None:
    """Check Ray metrics endpoint and update Ray health gauges."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{ray_url}/metrics", timeout=5) as resp:
            text = resp.read().decode("utf-8")
            metric_lines = [
                line for line in text.split("\n") if line.strip() and not line.startswith("#")
            ]
            RAY_AVAILABLE.set(1)
            RAY_METRIC_COUNT.set(len(metric_lines))

            # Count unique nodes by looking for ray_node_cpu_percent
            node_instances = set()
            for line in metric_lines:
                if line.startswith("ray_node_cpu_percent") and 'instance="' in line:
                    start = line.index('instance="') + len('instance="')
                    end = line.index('"', start)
                    node_instances.add(line[start:end])
                elif line.startswith(
                    (
                        "ray_serve_controller_state",
                        "ray_controller_state",
                    )
                ):
                    with contextlib.suppress(ValueError):
                        RAY_CONTROLLER_STATE.set(float(line.split()[-1]))
                elif line.startswith(
                    (
                        "ray_serve_controller_operation_time",
                        "ray_controller_operation_time",
                    )
                ):
                    with contextlib.suppress(ValueError):
                        RAY_CONTROLLER_OPERATION_TIME.set(float(line.split()[-1]))
            RAY_NODE_COUNT.set(len(node_instances))
    except (urllib.error.URLError, OSError):
        RAY_AVAILABLE.set(0)
        RAY_NODE_COUNT.set(0)
        RAY_METRIC_COUNT.set(0)
        RAY_CONTROLLER_STATE.set(0)
        RAY_CONTROLLER_OPERATION_TIME.set(0)


_TRAINING_GAUGES = (
    TRAINING_TRAIN_LOSS,
    TRAINING_VAL_LOSS,
    TRAINING_VAL_ACCURACY,
    TRAINING_VAL_BALANCED_ACCURACY,
    TRAINING_VAL_NEGATIVE_PRECISION,
    TRAINING_VAL_NEGATIVE_RECALL,
    TRAINING_VAL_NEGATIVE_F1,
    TRAINING_VAL_NEUTRAL_PRECISION,
    TRAINING_VAL_NEUTRAL_RECALL,
    TRAINING_VAL_NEUTRAL_F1,
    TRAINING_VAL_POSITIVE_PRECISION,
    TRAINING_VAL_POSITIVE_RECALL,
    TRAINING_VAL_POSITIVE_F1,
    TRAINING_VAL_MACRO_F1,
    TRAINING_VAL_WEIGHTED_F1,
    TRAINING_VAL_COHEN_KAPPA,
    TRAINING_VAL_MCC,
    TRAINING_VAL_NEUTRAL_TO_POSITIVE_RATE,
    TRAINING_VAL_NEUTRAL_TO_NEGATIVE_RATE,
    TRAINING_VAL_PRED_NEUTRAL_FRAC,
    TRAINING_VAL_NEUTRAL_AUC_ROC,
    TRAINING_VAL_NEUTRAL_AVG_PRECISION,
    TRAINING_EPOCH,
    TRAINING_LR,
    TRAINING_TRAIN_LOSS_EMA,
    TRAINING_TRAIN_LOSS_AVG,
    TRAINING_BATCH,
    TRAINING_GRAD_NORM,
    TRAINING_THROUGHPUT,
)


def _update_training_metrics() -> None:
    """Read persisted training metrics from per-model JSON files and update gauges.

    Each model type (rnn, encoder, decoder, modernbert) writes to its own JSON file in
    ``/tmp/sentimentizer_metrics/{model_type}_metrics.json`` to eliminate
    race conditions when multiple training processes run concurrently.  The
    exporter discovers all files and zeroes out Prometheus gauges for
    any model type whose file is missing or stale.
    """
    try:
        # Clear all previous training metric timeseries to purge old run_ids
        for g in _TRAINING_GAUGES:
            g.clear()

        _METRICS_DIR.mkdir(parents=True, exist_ok=True)
        for model_type in ("rnn", "encoder", "decoder", "modernbert"):
            path = get_training_metrics_path(model_type)

            if not path.exists():
                # File absent -> zero out gauges under empty run_id so baseline series exist
                lbl = {"model_type": model_type, "run_id": ""}
                for g in _TRAINING_GAUGES:
                    g.labels(**lbl).set(0)
                continue

            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                # Skip only this model; gauges were already cleared for this pass.
                logger.warning("Skipping unreadable metrics file %s: %s", path, exc)
                continue
            if not isinstance(data, dict):
                continue

            run_id = data.get("run_id", "")
            lbl = {"model_type": model_type, "run_id": run_id}

            # Skip reset placeholder files — they contain zeros from
            # _reset_stale_metrics() and are not real training data.
            # But still check the batch snapshot file for intra-epoch data.
            if data.get("_reset"):
                for g in _TRAINING_GAUGES:
                    # Clear/zero the epoch gauges, but avoid resetting batch gauges since they
                    # might be updated by the batch snapshot file.
                    if g not in (
                        TRAINING_TRAIN_LOSS_EMA,
                        TRAINING_TRAIN_LOSS_AVG,
                        TRAINING_BATCH,
                        TRAINING_GRAD_NORM,
                        TRAINING_THROUGHPUT,
                    ):
                        g.labels(**lbl).set(0)

            # Sanity check -- the file name should match _written_by trace field
            written_by = data.get("_written_by", model_type)
            if written_by != model_type:
                logger.warning(
                    "metrics file mismatch: expected %s but found _written_by=%s",
                    model_type,
                    written_by,
                )

            TRAINING_TRAIN_LOSS.labels(**lbl).set(float(data.get("train_loss", 0)))
            TRAINING_VAL_LOSS.labels(**lbl).set(float(data.get("val_loss", 0)))
            TRAINING_VAL_ACCURACY.labels(**lbl).set(float(data.get("accuracy", 0)))
            TRAINING_VAL_BALANCED_ACCURACY.labels(**lbl).set(
                float(data.get("balanced_accuracy", 0))
            )
            TRAINING_VAL_NEGATIVE_PRECISION.labels(**lbl).set(
                float(data.get("negative_precision", 0))
            )
            TRAINING_VAL_NEGATIVE_RECALL.labels(**lbl).set(float(data.get("negative_recall", 0)))
            TRAINING_VAL_NEGATIVE_F1.labels(**lbl).set(float(data.get("negative_f1", 0)))
            TRAINING_VAL_NEUTRAL_PRECISION.labels(**lbl).set(
                float(data.get("neutral_precision", 0))
            )
            TRAINING_VAL_NEUTRAL_RECALL.labels(**lbl).set(float(data.get("neutral_recall", 0)))
            TRAINING_VAL_NEUTRAL_F1.labels(**lbl).set(float(data.get("neutral_f1", 0)))
            TRAINING_VAL_POSITIVE_PRECISION.labels(**lbl).set(
                float(data.get("positive_precision", 0))
            )
            TRAINING_VAL_POSITIVE_RECALL.labels(**lbl).set(float(data.get("positive_recall", 0)))
            TRAINING_VAL_POSITIVE_F1.labels(**lbl).set(float(data.get("positive_f1", 0)))
            TRAINING_VAL_MACRO_F1.labels(**lbl).set(float(data.get("macro_f1", 0)))
            TRAINING_VAL_WEIGHTED_F1.labels(**lbl).set(float(data.get("weighted_f1", 0)))
            TRAINING_VAL_COHEN_KAPPA.labels(**lbl).set(float(data.get("cohen_kappa", 0)))
            TRAINING_VAL_MCC.labels(**lbl).set(float(data.get("mcc", 0)))
            TRAINING_VAL_NEUTRAL_TO_POSITIVE_RATE.labels(**lbl).set(
                float(data.get("neutral_to_positive_rate", 0))
            )
            TRAINING_VAL_NEUTRAL_TO_NEGATIVE_RATE.labels(**lbl).set(
                float(data.get("neutral_to_negative_rate", 0))
            )
            TRAINING_VAL_PRED_NEUTRAL_FRAC.labels(**lbl).set(
                float(data.get("pred_neutral_frac", 0))
            )
            neutral_auc = data.get("neutral_auc_roc")
            if neutral_auc is not None:
                TRAINING_VAL_NEUTRAL_AUC_ROC.labels(**lbl).set(float(neutral_auc))
            else:
                TRAINING_VAL_NEUTRAL_AUC_ROC.labels(**lbl).set(0)
            neutral_ap = data.get("neutral_avg_precision")
            if neutral_ap is not None:
                TRAINING_VAL_NEUTRAL_AVG_PRECISION.labels(**lbl).set(float(neutral_ap))
            else:
                TRAINING_VAL_NEUTRAL_AVG_PRECISION.labels(**lbl).set(0)
            TRAINING_EPOCH.labels(**lbl).set(int(data.get("epoch", 0)))
            lr = data.get("lr")
            TRAINING_LR.labels(**lbl).set(0.0 if lr is None else float(lr))

            # Intra-epoch batch metrics: read from the lightweight batch snapshot file
            # (written every N batches during training for real-time dashboard visibility).
            # The epoch-end JSON no longer contains batch_metrics — the snapshot file
            # is the sole source for intra-epoch gauges to avoid duplicate data points
            # on the Grafana dashboard.
            # Skip batch gauges for reset/stale model types — they have no live data,
            # and creating zero-valued label series pollutes the dashboard with
            # duplicate "emitter" lines under an empty run_id.
            if data.get("_reset"):
                continue

            batch_snapshot_path = _METRICS_DIR / f"{model_type}_batch.json"
            if batch_snapshot_path.exists():
                try:
                    batch_data = json.loads(batch_snapshot_path.read_text())
                    if isinstance(batch_data, dict) and not batch_data.get("_reset"):
                        batch_run_id = batch_data.get("run_id", run_id)
                        batch_lbl = {"model_type": model_type, "run_id": batch_run_id}
                        TRAINING_TRAIN_LOSS_EMA.labels(**batch_lbl).set(
                            float(batch_data.get("loss_ema", 0))
                        )
                        TRAINING_TRAIN_LOSS_AVG.labels(**batch_lbl).set(
                            float(batch_data.get("avg_loss", 0))
                        )
                        TRAINING_BATCH.labels(**batch_lbl).set(int(batch_data.get("batch", 0)))
                        TRAINING_GRAD_NORM.labels(**batch_lbl).set(
                            float(batch_data.get("grad_norm", 0))
                        )
                        TRAINING_THROUGHPUT.labels(**batch_lbl).set(
                            float(batch_data.get("throughput", 0))
                        )
                        if "epoch" in batch_data:
                            TRAINING_EPOCH.labels(**batch_lbl).set(int(batch_data["epoch"]))
                        if "lr" in batch_data:
                            TRAINING_LR.labels(**batch_lbl).set(float(batch_data["lr"]))
                    else:
                        # Batch snapshot has _reset flag or is not a dict — skip
                        pass
                except (json.JSONDecodeError, ValueError):
                    # Corrupt batch snapshot — skip rather than emit zero-valued
                    # series that would duplicate live metrics on the dashboard.
                    pass
            # No batch snapshot file for this model type — the epoch-level
            # gauges (set above) are sufficient.  Do not create zero-valued
            # batch gauge series with a stale run_id, as they produce
            # duplicate "emitter" lines on the Grafana dashboard.
    except Exception as e:
        logger.warning("Error updating training metrics from file: %s", e)


def _metrics_loop(interval: int, ray_url: str) -> None:
    """Background thread that periodically updates metric gauges."""
    while True:
        try:
            _update_system_metrics()
        except Exception as e:
            logger.warning("Error updating system metrics: %s", e)
        try:
            _update_gpu_metrics()
        except Exception as e:
            logger.warning("Error updating GPU metrics: %s", e)
        try:
            _update_ray_metrics(ray_url)
        except Exception as e:
            logger.warning("Error updating Ray metrics: %s", e)
        try:
            _update_training_metrics()
        except Exception as e:
            logger.warning("Error updating training metrics: %s", e)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentimentizer Prometheus metrics exporter")
    parser.add_argument(
        "--port", type=int, default=8081, help="Port to serve metrics on (default: 8081)"
    )
    parser.add_argument(
        "--addr",
        default="127.0.0.1",
        help="Address to serve metrics on (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Metrics collection interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--ray-url",
        default="http://localhost:8080",
        help="Ray metrics URL (default: http://localhost:8080)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Set static system info
    SYSTEM_INFO.info(
        {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "cpu_count": str(os.cpu_count() or 0),
        }
    )

    # Initial collection before starting server
    _update_system_metrics()
    _update_gpu_metrics()
    _update_ray_metrics(args.ray_url)
    _update_training_metrics()

    # Start background collection thread
    collector = threading.Thread(
        target=_metrics_loop,
        args=(args.interval, args.ray_url),
        daemon=True,
    )
    collector.start()

    # Start Prometheus HTTP server
    start_http_server(args.port, addr=args.addr)
    logger.info("Sentimentizer metrics exporter started on port %d", args.port)
    logger.info("Collecting metrics every %d seconds", args.interval)
    logger.info("Ray metrics URL: %s", args.ray_url)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter")


if __name__ == "__main__":
    main()
