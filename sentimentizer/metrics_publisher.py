"""Centralized metrics publishing for training and evaluation.

Consolidates the gauge-setting, JSON persistence, and logging logic that
was previously duplicated across Trainer.evaluate() and _train_func().

Instead of 4 near-identical blocks of gauge-setting code, both callers
now use a single publish_epoch_metrics() function.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL
from sentimentizer.metrics import ClassificationMetrics

logger = new_logger(DEFAULT_LOG_LEVEL)

# ──────────────────────────────────────────────
# Metric field definitions
# ──────────────────────────────────────────────

# Mapping from ClassificationMetrics attribute name → Prometheus/Ray gauge key name.
# This single source of truth replaces the 4× duplicated field-mapping blocks.
_METRIC_GAUGE_KEYS: list[tuple[str, str, bool]] = [
    # (metrics_attr, gauge_key, is_optional_none)
    # gauge_key matches both Ray gauge dict keys and Prometheus gauge variable prefixes
    ("accuracy", "val_accuracy", False),
    ("balanced_accuracy", "val_balanced_accuracy", False),
    ("negative_precision", "val_negative_precision", False),
    ("negative_recall", "val_negative_recall", False),
    ("negative_f1", "val_negative_f1", False),
    ("neutral_precision", "val_neutral_precision", False),
    ("neutral_recall", "val_neutral_recall", False),
    ("neutral_f1", "val_neutral_f1", False),
    ("positive_precision", "val_positive_precision", False),
    ("positive_recall", "val_positive_recall", False),
    ("positive_f1", "val_positive_f1", False),
    ("macro_f1", "val_macro_f1", False),
    ("weighted_f1", "val_weighted_f1", False),
    ("cohen_kappa", "val_cohen_kappa", False),
    ("mcc", "val_mcc", False),
    ("neutral_to_positive_rate", "val_neutral_to_positive_rate", False),
    ("neutral_to_negative_rate", "val_neutral_to_negative_rate", False),
    ("pred_neutral_frac", "val_pred_neutral_frac", False),
    ("neutral_auc_roc", "val_neutral_auc_roc", True),
    ("neutral_avg_precision", "val_neutral_avg_precision", True),
]

# ──────────────────────────────────────────────
# JSON metrics persistence
# ──────────────────────────────────────────────

_METRICS_DIR = Path("/tmp/sentimentizer_metrics")


def get_metrics_dir() -> Path:
    """Return the directory used for per-model metrics JSON files."""
    return _METRICS_DIR


def write_json_atomic(path: Path, data: dict[str, Any], *, indent: int | None = None) -> None:
    """Replace *path* in one step so the exporter never reads a partial file."""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}-", delete=False
    ) as target:
        temporary_path = Path(target.name)
        try:
            target.write(json.dumps(data, indent=indent))
            target.close()
            # NamedTemporaryFile is 0600; the exporter may run as another user.
            os.chmod(temporary_path, 0o644)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def write_batch_snapshot(
    *,
    model_type: str,
    run_id: str = "",
    epoch: int,
    batch: int,
    loss_ema: float,
    avg_loss: float,
    lr: float,
    grad_norm: float | None = None,
    throughput: float | None = None,
) -> None:
    """Write a lightweight batch-level snapshot for real-time dashboard visibility.

    Written every N batches during training (controlled by ``ray_update_every``).
    The standalone exporter reads this file alongside the epoch metrics file to
    provide intra-epoch gauge values on the Grafana dashboard.

    This is intentionally a small file (only 6 fields) — it is the sole
    source for intra-epoch gauges on the Grafana dashboard.

    Args:
        model_type: Model type label.
        run_id: Unique training run ID.
        epoch: Current epoch number.
        batch: Current batch number within the epoch.
        loss_ema: Fast-moving EMA loss value.
        avg_loss: Slow-moving epoch-average loss value.
        lr: Current learning rate.
    """
    import contextlib

    metrics_dir = get_metrics_dir()
    with contextlib.suppress(OSError):
        metrics_dir.mkdir(parents=True, exist_ok=True)

    path = metrics_dir / f"{model_type}_batch.json"

    data: dict[str, Any] = {
        "epoch": int(epoch),
        "batch": int(batch),
        "loss_ema": float(loss_ema),
        "avg_loss": float(avg_loss),
        "lr": float(lr),
        "run_id": run_id,
        "_written_by": model_type,
        "_written_at": time.time(),
    }

    if grad_norm is not None:
        data["grad_norm"] = float(grad_norm)
    if throughput is not None:
        data["throughput"] = float(throughput)

    with contextlib.suppress(OSError):
        write_json_atomic(path, data)


def write_epoch_metrics_to_file(
    *,
    model_type: str,
    run_id: str = "",
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: ClassificationMetrics,
    lr: float | None = None,
) -> None:
    """Write current epoch metrics to the per-model JSON file.

    Each model type writes to its own file
    (``/tmp/sentimentizer_metrics/{model_type}_metrics.json``) so concurrent
    training processes never race on a shared JSON file.

    Args:
        model_type: Model type label.
        run_id: Unique training run ID.
        epoch: Current epoch number.
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch.
        metrics: Computed classification metrics.
        lr: Current learning rate.
    """
    import contextlib

    metrics_dir = get_metrics_dir()
    with contextlib.suppress(OSError):
        metrics_dir.mkdir(parents=True, exist_ok=True)

    path = metrics_dir / f"{model_type}_metrics.json"

    data: dict[str, Any] = {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "accuracy": float(metrics.accuracy),
        "balanced_accuracy": float(metrics.balanced_accuracy),
        "negative_precision": float(metrics.negative_precision),
        "negative_recall": float(metrics.negative_recall),
        "negative_f1": float(metrics.negative_f1),
        "neutral_precision": float(metrics.neutral_precision),
        "neutral_recall": float(metrics.neutral_recall),
        "neutral_f1": float(metrics.neutral_f1),
        "positive_precision": float(metrics.positive_precision),
        "positive_recall": float(metrics.positive_recall),
        "positive_f1": float(metrics.positive_f1),
        "macro_f1": float(metrics.macro_f1),
        "weighted_f1": float(metrics.weighted_f1),
        "cohen_kappa": float(metrics.cohen_kappa),
        "mcc": float(metrics.mcc),
        "neutral_to_positive_rate": float(metrics.neutral_to_positive_rate),
        "neutral_to_negative_rate": float(metrics.neutral_to_negative_rate),
        "pred_neutral_frac": float(metrics.pred_neutral_frac),
        "neutral_auc_roc": (
            float(metrics.neutral_auc_roc) if metrics.neutral_auc_roc is not None else None
        ),
        "neutral_avg_precision": (
            float(metrics.neutral_avg_precision)
            if metrics.neutral_avg_precision is not None
            else None
        ),
        "epoch": int(epoch),
        "lr": float(lr) if lr is not None else None,
        "run_id": run_id,
        "_written_by": model_type,
        "_written_at": time.time(),
    }

    with contextlib.suppress(OSError):
        write_json_atomic(path, data, indent=2)


# ──────────────────────────────────────────────
# Ray gauge publishing
# ──────────────────────────────────────────────


def _set_ray_gauges(
    gauges: dict[str, Any],
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: ClassificationMetrics,
    lr: float,
) -> None:
    """Set all Ray custom metric gauges for a training epoch.

    Args:
        gauges: Dict of gauge name → Ray Gauge object (from _get_ray_gauges).
        epoch: Current epoch number.
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch.
        metrics: Computed classification metrics.
        lr: Current learning rate.
    """
    gauges["train_loss"].set(float(train_loss))
    gauges["val_loss"].set(float(val_loss))
    gauges["lr"].set(float(lr))
    gauges["epoch"].set(epoch)

    for attr, key, is_optional_none in _METRIC_GAUGE_KEYS:
        if key not in gauges:
            continue  # skip keys not in Ray gauges
        value = getattr(metrics, attr)
        if is_optional_none and value is None:
            continue
        gauges[key].set(float(value))


# ──────────────────────────────────────────────
# Prometheus exporter gauge publishing
# ──────────────────────────────────────────────


def _set_prometheus_gauges(
    model_type: str,
    run_id: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: ClassificationMetrics,
    lr: float,
) -> None:
    """Set all standalone Prometheus exporter gauges for a training epoch.

    Imports the gauge objects lazily since the exporter module may not be
    available in all contexts (e.g., Ray workers).
    """
    try:
        from sentimentizer.exporter import (
            TRAINING_EPOCH,
            TRAINING_LR,
            TRAINING_TRAIN_LOSS,
            TRAINING_VAL_ACCURACY,
            TRAINING_VAL_BALANCED_ACCURACY,
            TRAINING_VAL_COHEN_KAPPA,
            TRAINING_VAL_LOSS,
            TRAINING_VAL_MACRO_F1,
            TRAINING_VAL_MCC,
            TRAINING_VAL_NEGATIVE_F1,
            TRAINING_VAL_NEGATIVE_PRECISION,
            TRAINING_VAL_NEGATIVE_RECALL,
            TRAINING_VAL_NEUTRAL_AUC_ROC,
            TRAINING_VAL_NEUTRAL_AVG_PRECISION,
            TRAINING_VAL_NEUTRAL_F1,
            TRAINING_VAL_NEUTRAL_PRECISION,
            TRAINING_VAL_NEUTRAL_RECALL,
            TRAINING_VAL_NEUTRAL_TO_NEGATIVE_RATE,
            TRAINING_VAL_NEUTRAL_TO_POSITIVE_RATE,
            TRAINING_VAL_POSITIVE_F1,
            TRAINING_VAL_POSITIVE_PRECISION,
            TRAINING_VAL_POSITIVE_RECALL,
            TRAINING_VAL_PRED_NEUTRAL_FRAC,
            TRAINING_VAL_WEIGHTED_F1,
        )
    except ImportError:
        return

    lbl = {"model_type": model_type, "run_id": run_id}
    TRAINING_TRAIN_LOSS.labels(**lbl).set(float(train_loss))
    TRAINING_VAL_LOSS.labels(**lbl).set(float(val_loss))
    TRAINING_EPOCH.labels(**lbl).set(epoch)
    TRAINING_LR.labels(**lbl).set(float(lr))

    # Map metrics attributes to Prometheus gauge objects
    _PROM_GAUGE_MAP: dict[str, Any] = {
        "accuracy": TRAINING_VAL_ACCURACY,
        "balanced_accuracy": TRAINING_VAL_BALANCED_ACCURACY,
        "negative_precision": TRAINING_VAL_NEGATIVE_PRECISION,
        "negative_recall": TRAINING_VAL_NEGATIVE_RECALL,
        "negative_f1": TRAINING_VAL_NEGATIVE_F1,
        "neutral_precision": TRAINING_VAL_NEUTRAL_PRECISION,
        "neutral_recall": TRAINING_VAL_NEUTRAL_RECALL,
        "neutral_f1": TRAINING_VAL_NEUTRAL_F1,
        "positive_precision": TRAINING_VAL_POSITIVE_PRECISION,
        "positive_recall": TRAINING_VAL_POSITIVE_RECALL,
        "positive_f1": TRAINING_VAL_POSITIVE_F1,
        "macro_f1": TRAINING_VAL_MACRO_F1,
        "weighted_f1": TRAINING_VAL_WEIGHTED_F1,
        "cohen_kappa": TRAINING_VAL_COHEN_KAPPA,
        "mcc": TRAINING_VAL_MCC,
        "neutral_to_positive_rate": TRAINING_VAL_NEUTRAL_TO_POSITIVE_RATE,
        "neutral_to_negative_rate": TRAINING_VAL_NEUTRAL_TO_NEGATIVE_RATE,
        "pred_neutral_frac": TRAINING_VAL_PRED_NEUTRAL_FRAC,
        "neutral_auc_roc": TRAINING_VAL_NEUTRAL_AUC_ROC,
        "neutral_avg_precision": TRAINING_VAL_NEUTRAL_AVG_PRECISION,
    }

    for attr, gauge in _PROM_GAUGE_MAP.items():
        value = getattr(metrics, attr)
        if value is None:
            continue
        gauge.labels(**lbl).set(float(value))


# ──────────────────────────────────────────────
# Unified publishing entry point
# ──────────────────────────────────────────────


def publish_epoch_metrics(
    *,
    model_type: str,
    run_id: str = "",
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: ClassificationMetrics,
    lr: float,
    ray_gauges: dict[str, Any] | None = None,
) -> None:
    """Publish training metrics to all backends: Ray gauges, Prometheus, JSON file, and logger.

    This is the single entry point that replaces the 4 duplicated metric-publishing
    blocks that were previously in Trainer.evaluate() and _train_func().

    Args:
        model_type: Model type label (e.g. "rnn", "encoder", "decoder", "modernbert").
        run_id: Unique training run ID.
        epoch: Current training epoch number.
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch.
        metrics: Computed classification metrics.
        lr: Current learning rate.
        ray_gauges: Optional dict of Ray Gauge objects (from _get_ray_gauges).
            If None, Ray gauge publishing is skipped (e.g., in single-node mode
            where Ray is not running).
    """
    # 1. Ray custom metrics (only if gauges are provided — skip if not in Ray context)
    if ray_gauges is not None:
        _set_ray_gauges(
            gauges=ray_gauges,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            lr=lr,
        )

    # 2. Standalone Prometheus exporter gauges (best-effort, skip if not importable)
    _set_prometheus_gauges(
        model_type=model_type,
        run_id=run_id,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        metrics=metrics,
        lr=lr,
    )

    # 3. Persist to JSON for the standalone exporter
    write_epoch_metrics_to_file(
        model_type=model_type,
        run_id=run_id,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        metrics=metrics,
        lr=lr,
    )

    # 4. Structured logging
    neutral_auc_str = (
        f"{metrics.neutral_auc_roc:.4f}" if metrics.neutral_auc_roc is not None else "N/A"
    )
    neutral_ap_str = (
        f"{metrics.neutral_avg_precision:.4f}"
        if metrics.neutral_avg_precision is not None
        else "N/A"
    )
    logger.info(
        f"[{model_type}] [epoch {epoch}] evaluation complete - "
        f"val_loss={val_loss:.4f} accuracy={metrics.accuracy:.4f} "
        f"balanced_accuracy={metrics.balanced_accuracy:.4f} "
        f"macro_f1={metrics.macro_f1:.4f} weighted_f1={metrics.weighted_f1:.4f} "
        f"neg_f1={metrics.negative_f1:.4f} neu_f1={metrics.neutral_f1:.4f} "
        f"pos_f1={metrics.positive_f1:.4f} cohen_kappa={metrics.cohen_kappa:.4f} "
        f"mcc={metrics.mcc:.4f} "
        f"neutral_auc_roc={neutral_auc_str} neutral_avg_precision={neutral_ap_str}"
    )
