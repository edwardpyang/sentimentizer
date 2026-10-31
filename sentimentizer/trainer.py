from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

try:
    import ray
    from ray import train
    from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer, prepare_model
except ImportError:
    pass

from sentimentizer import new_logger
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    DecoderOptimizationParams,
    DecoderSchedulerParams,
    DriverConfig,
    EncoderOptimizationParams,
    EncoderSchedulerParams,
    ModernBERTOptimizationParams,
    ModernBERTSchedulerParams,
    OptimizationParams,
    RNNSchedulerParams,
    SchedulerParams,
    TrainerConfig,
    default_dataloader_workers,
    default_epochs,
)
from sentimentizer.metrics import ClassificationMetrics
from sentimentizer.metrics_publisher import (
    publish_epoch_metrics,
    write_batch_snapshot,
    write_epoch_metrics_to_file,
)

# Backward-compatible alias — existing code imports _write_epoch_metrics_to_file
# from this module.  The canonical location is now metrics_publisher.
_write_epoch_metrics_to_file = write_epoch_metrics_to_file


# ---------------------------------------------------------------------------
# Ray custom metrics (lazy initialization)
#
# In Ray 2.55+, ray.util.metrics.Gauge objects must be created inside a
# Ray worker context (task or actor) to be exported via the Ray metrics
# endpoint.  Creating them at module-import time in the driver process
# silently fails — the gauges are never pushed to the metrics agent.
#
# We therefore lazily create gauge instances on first use and cache them
# per model_type so that each worker gets its own gauge that is properly
# registered with that worker's metrics agent.
# ---------------------------------------------------------------------------

_RAY_GAUGES: dict[tuple[str, str], dict[str, Any]] = {}
"""Cache of Ray Gauge dicts keyed by (model_type, run_id)."""


def _get_ray_gauges(model_type: str, run_id: str = "") -> dict[str, Any] | None:
    """Return (and lazily create) Ray Gauge instances for *model_type*.

    Gauges are created once per (model_type, run_id) and cached.  On the first call
    this imports ``ray.util.metrics.Gauge`` and builds all the gauge
    objects.  If Ray is not available, returns ``None``.
    """
    key = (model_type, run_id)
    if key in _RAY_GAUGES:
        return _RAY_GAUGES[key]

    try:
        from ray.util.metrics import Gauge
    except ImportError:
        return None

    tag_keys = ("model_type", "run_id")
    gauges: dict[str, Any] = {
        "train_loss": Gauge(
            "sentimentizer_live_train_loss",
            description="Live training loss",
            tag_keys=tag_keys,
        ),
        "val_loss": Gauge(
            "sentimentizer_live_val_loss",
            description="Live validation loss",
            tag_keys=tag_keys,
        ),
        "val_accuracy": Gauge(
            "sentimentizer_live_val_accuracy",
            description="Live validation accuracy",
            tag_keys=tag_keys,
        ),
        "val_balanced_accuracy": Gauge(
            "sentimentizer_live_val_balanced_accuracy",
            description="Live validation balanced accuracy",
            tag_keys=tag_keys,
        ),
        "val_negative_precision": Gauge(
            "sentimentizer_live_val_negative_precision",
            description="Live validation negative-class precision",
            tag_keys=tag_keys,
        ),
        "val_negative_recall": Gauge(
            "sentimentizer_live_val_negative_recall",
            description="Live validation negative-class recall",
            tag_keys=tag_keys,
        ),
        "val_negative_f1": Gauge(
            "sentimentizer_live_val_negative_f1",
            description="Live validation negative-class F1",
            tag_keys=tag_keys,
        ),
        "val_neutral_precision": Gauge(
            "sentimentizer_live_val_neutral_precision",
            description="Live validation neutral-class precision",
            tag_keys=tag_keys,
        ),
        "val_neutral_recall": Gauge(
            "sentimentizer_live_val_neutral_recall",
            description="Live validation neutral-class recall",
            tag_keys=tag_keys,
        ),
        "val_neutral_f1": Gauge(
            "sentimentizer_live_val_neutral_f1",
            description="Live validation neutral-class F1",
            tag_keys=tag_keys,
        ),
        "val_positive_precision": Gauge(
            "sentimentizer_live_val_positive_precision",
            description="Live validation positive-class precision",
            tag_keys=tag_keys,
        ),
        "val_positive_recall": Gauge(
            "sentimentizer_live_val_positive_recall",
            description="Live validation positive-class recall",
            tag_keys=tag_keys,
        ),
        "val_positive_f1": Gauge(
            "sentimentizer_live_val_positive_f1",
            description="Live validation positive-class F1",
            tag_keys=tag_keys,
        ),
        "val_cohen_kappa": Gauge(
            "sentimentizer_live_val_cohen_kappa",
            description="Live validation Cohen's kappa",
            tag_keys=tag_keys,
        ),
        "val_mcc": Gauge(
            "sentimentizer_live_val_mcc",
            description="Live validation Matthews correlation coefficient",
            tag_keys=tag_keys,
        ),
        "val_macro_f1": Gauge(
            "sentimentizer_live_val_macro_f1",
            description="Live validation macro-averaged F1 score",
            tag_keys=tag_keys,
        ),
        "val_weighted_f1": Gauge(
            "sentimentizer_live_val_weighted_f1",
            description="Live validation weighted-averaged F1 score",
            tag_keys=tag_keys,
        ),
        "val_neutral_to_positive_rate": Gauge(
            "sentimentizer_live_val_neutral_to_positive_rate",
            description="Live validation neutral-to-positive misclassification rate",
            tag_keys=tag_keys,
        ),
        "val_neutral_to_negative_rate": Gauge(
            "sentimentizer_live_val_neutral_to_negative_rate",
            description="Live validation neutral-to-negative misclassification rate",
            tag_keys=tag_keys,
        ),
        "val_pred_neutral_frac": Gauge(
            "sentimentizer_live_val_pred_neutral_frac",
            description="Live validation fraction of neutral predictions",
            tag_keys=tag_keys,
        ),
        "val_neutral_auc_roc": Gauge(
            "sentimentizer_live_val_neutral_auc_roc",
            description="Live validation neutral-class AUC-ROC",
            tag_keys=tag_keys,
        ),
        "val_neutral_avg_precision": Gauge(
            "sentimentizer_live_val_neutral_avg_precision",
            description="Live validation neutral-class average precision",
            tag_keys=tag_keys,
        ),
        "epoch": Gauge(
            "sentimentizer_live_epoch",
            description="Live training epoch",
            tag_keys=tag_keys,
        ),
        "lr": Gauge(
            "sentimentizer_live_lr",
            description="Live learning rate",
            tag_keys=tag_keys,
        ),
        "train_loss_ema": Gauge(
            "sentimentizer_live_train_loss_ema",
            description="Live fast-moving EMA training loss",
            tag_keys=tag_keys,
        ),
        "train_loss_avg": Gauge(
            "sentimentizer_live_train_loss_avg",
            description="Live slow-moving epoch-average training loss",
            tag_keys=tag_keys,
        ),
        "train_batch": Gauge(
            "sentimentizer_live_train_batch",
            description="Live batch number within the current epoch",
            tag_keys=tag_keys,
        ),
        "train_grad_norm": Gauge(
            "sentimentizer_live_train_grad_norm",
            description="Live gradient norm before clipping",
            tag_keys=tag_keys,
        ),
        "train_throughput": Gauge(
            "sentimentizer_live_train_throughput",
            description="Live samples processed per second",
            tag_keys=tag_keys,
        ),
    }
    # Set default tags so callers can do gauge.set(value) without
    # repeating the tag on every call.
    for g in gauges.values():
        g.set_default_tags({"model_type": model_type, "run_id": run_id})

    _RAY_GAUGES[key] = gauges
    return gauges


logger = new_logger(DEFAULT_LOG_LEVEL)

BEST_CHECKPOINT_NAME = "best_model.pth"


# ──────────────────────────────────────────────
# Shared training primitives
# ──────────────────────────────────────────────


def train_step(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    accum_step_idx: int = 0,
    is_last_batch: bool = False,
    device_type: str = "cuda",
) -> tuple[float, float]:
    """Single training step supporting gradient accumulation, AMP, and dict inputs.

    Returns a tuple of (loss_value, grad_norm).
    """
    is_accum_start = (accum_step_idx % grad_accum_steps) == 0
    is_window_end = (accum_step_idx % grad_accum_steps) == (grad_accum_steps - 1)
    is_accum_end = is_window_end or is_last_batch

    if is_accum_start:
        optimizer.zero_grad()

    # AMP only benefits CUDA (saves VRAM via bfloat16); on CPU it's a no-op
    # that produces warnings if enabled with float32 dtype.
    amp_enabled = use_amp and device_type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    with torch.amp.autocast(device_type, dtype=amp_dtype, enabled=amp_enabled):
        output = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        loss = loss_function(output, target) / grad_accum_steps

    loss.backward()

    grad_norm = 0.0
    if is_accum_end:
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm, norm_type=2
        )
        if isinstance(norm, torch.Tensor):
            grad_norm = float(norm.item())
        elif norm is not None:
            grad_norm = float(norm)
        optimizer.step()

    return loss.item() * grad_accum_steps, grad_norm


def val_step(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor] | torch.Tensor,
    target: torch.Tensor,
    loss_function: Callable,
) -> float:
    """Single validation step supporting dict inputs."""
    with torch.no_grad():
        output = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        loss = loss_function(output, target)
    return loss.item()


@dataclass
class EpochResult:
    """Per-epoch training result container."""

    epoch: int
    train_loss: float
    val_loss: float
    metrics: ClassificationMetrics
    lr: float


@dataclass
class TrainingState:
    """Mutable training state container."""

    val_loss: float = float("inf")
    latest_train_loss: float = 0.0
    latest_epoch: int = 0
    latest_metrics: ClassificationMetrics | None = None
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    running_loss_mean: float = 0.0
    steps: int = 0


class TrainingCallback:
    """Protocol for hooking into the training loop."""

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        """Called after each epoch. Return True to stop training.

        Args:
            result: The current epoch's results.
            state: Mutable training state.

        Returns:
            True if training should stop, False otherwise.
        """
        return False

    def on_train_begin(self, model_type: str, device: str, epochs: int) -> None:
        """Called before training starts."""

    def on_train_end(self, state: TrainingState) -> None:
        """Called after training ends."""


class MetricsCallback(TrainingCallback):
    """Publishes metrics to Prometheus/JSON. Only active on rank 0."""

    def __init__(
        self,
        model_type: str,
        rank: int = 0,
        ray_gauges: dict[str, Any] | None = None,
    ) -> None:
        self.model_type = model_type
        self.rank = rank
        self.ray_gauges = ray_gauges

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if self.rank != 0:
            return False
        publish_epoch_metrics(
            model_type=self.model_type,
            epoch=result.epoch,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            metrics=result.metrics,
            lr=result.lr,
            ray_gauges=self.ray_gauges,
        )
        return False


class CheckpointCallback(TrainingCallback):
    """Saves periodic and best model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        checkpoint_every: int,
        checkpoint_best: bool,
        best_val_loss: float = float("inf"),
        patience_counter: int = 0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = checkpoint_every
        self.checkpoint_best = checkpoint_best
        self._best_val_loss = best_val_loss
        self.patience_counter = patience_counter

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if self.checkpoint_dir is None:
            return False

        # Periodic checkpoint
        if self.checkpoint_every > 0 and result.epoch % self.checkpoint_every == 0:
            ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{result.epoch}.pth"
            # Note: actual save happens in _run_training_loop which has model/optimizer refs
            state._pending_checkpoint_path = str(ckpt_path)  # type: ignore[attr-defined]

        # Best model checkpoint
        if self.checkpoint_best and result.val_loss < self._best_val_loss:
            self._best_val_loss = result.val_loss
            best_path = self.checkpoint_dir / "best_model.pth"
            state._pending_best_checkpoint_path = str(best_path)  # type: ignore[attr-defined]

        return False


class UnfreezeBackboneCallback(TrainingCallback):
    """Callback to unfreeze the model backbone at a specific epoch."""

    def __init__(self, freeze_epochs: int, model_type: str) -> None:
        self.freeze_epochs = freeze_epochs
        self.model_type = model_type
        self._unfrozen = False

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if self._unfrozen or result.epoch < self.freeze_epochs:
            return False
        # Signal the training loop to rebuild optimizer + scheduler
        state._pending_unfreeze = True  # type: ignore[attr-defined]
        self._unfrozen = True
        logger.info(
            f"[{self.model_type}] unfreezing backbone at epoch {result.epoch}, "
            f"rebuilding optimizer with all trainable parameters"
        )
        return False


class EarlyStoppingCallback(TrainingCallback):
    """Stops training when validation loss doesn't improve."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._best_val_loss = float("inf")
        self.patience_counter = 0

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if result.val_loss < self._best_val_loss:
            self._best_val_loss = result.val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(
                    f"early stopping at epoch {result.epoch}, "
                    f"val_loss hasn't improved for {self.patience_counter} epochs"
                )
                return True
        return False


class LoggingCallback(TrainingCallback):
    """Structured logging per epoch. Only active on rank 0."""

    def __init__(self, model_type: str, rank: int = 0) -> None:
        self.model_type = model_type
        self.rank = rank

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if self.rank != 0:
            return False
        logger.info(
            f"[{self.model_type}] [epoch {result.epoch}] completed, "
            f"val_loss={result.val_loss:.4f}, train_loss={result.train_loss:.4f}, "
            f"accuracy={result.metrics.accuracy:.4f}, lr={result.lr:.6f}"
        )
        return False

    def on_train_begin(self, model_type: str, device: str, epochs: int) -> None:
        if self.rank != 0:
            return
        logger.info(f"[{model_type}] fitting model...")
        logger.info(f"[{model_type}] epochs={epochs}, device={device}")


class RayReportCallback(TrainingCallback):
    """Reports metrics and checkpoints to Ray Train. All ranks must call."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        try:
            import tempfile

            import ray.cloudpickle as pickle
            from ray import train
            from ray.train import Checkpoint

            # Note: checkpoint data must be populated by the loop
            checkpoint_data = getattr(state, "_ray_checkpoint_data", None)
            if checkpoint_data is not None:
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, "data.pkl"), "wb") as fp:
                        pickle.dump(checkpoint_data, fp)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report(
                        {
                            "train_loss": result.train_loss,
                            "val_loss": result.val_loss,
                            "accuracy": result.metrics.accuracy,
                            "balanced_accuracy": result.metrics.balanced_accuracy,
                            "negative_precision": result.metrics.negative_precision,
                            "negative_recall": result.metrics.negative_recall,
                            "negative_f1": result.metrics.negative_f1,
                            "neutral_precision": result.metrics.neutral_precision,
                            "neutral_recall": result.metrics.neutral_recall,
                            "neutral_f1": result.metrics.neutral_f1,
                            "positive_precision": result.metrics.positive_precision,
                            "positive_recall": result.metrics.positive_recall,
                            "positive_f1": result.metrics.positive_f1,
                            "macro_f1": result.metrics.macro_f1,
                            "weighted_f1": result.metrics.weighted_f1,
                            "cohen_kappa": result.metrics.cohen_kappa,
                            "mcc": result.metrics.mcc,
                            "total": result.metrics.total,
                            "epoch": result.epoch,
                            "lr": result.lr,
                        },
                        checkpoint=checkpoint,
                    )
            else:
                train.report(
                    {
                        "train_loss": result.train_loss,
                        "val_loss": result.val_loss,
                        "accuracy": result.metrics.accuracy,
                        "balanced_accuracy": result.metrics.balanced_accuracy,
                        "macro_f1": result.metrics.macro_f1,
                        "epoch": result.epoch,
                        "lr": result.lr,
                    }
                )
        except (ImportError, RuntimeError):
            pass
        return False


def compute_epoch_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    model_type: str,
    num_classes: int = 3,
) -> ClassificationMetrics:
    """Compute classification metrics from probabilities and targets.

    Handles NaN replacement in probabilities before computing metrics.
    This is the shared post-processing logic that was previously duplicated
    in both Trainer.evaluate() and _train_func().

    Args:
        probabilities: Softmax probability matrix, shape (N, num_classes).
            May contain NaN values.
        targets: Ground truth class indices, shape (N,).
        model_type: Model type label for logging.
        num_classes: Number of classification classes (default 3).

    Returns:
        ClassificationMetrics with all standard classification metrics.
    """
    from sentimentizer.config import NUM_CLASSES as _NUM_CLASSES

    if num_classes <= 1:
        num_classes = _NUM_CLASSES

    # Replace NaN probabilities with uniform distribution
    nan_mask = np.isnan(probabilities)
    if nan_mask.any():
        nan_count = int(nan_mask.sum())
        logger.warning(
            f"[{model_type}] nan_in_probabilities",
            message=f"Found {nan_count} NaN values in probabilities, replacing with uniform",
        )
        probabilities = np.where(nan_mask, 1.0 / num_classes, probabilities)

    predictions = probabilities.argmax(axis=1)

    from sentimentizer.metrics import compute_classification_metrics

    return compute_classification_metrics(
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
        num_classes=num_classes,
    )


def _prepare_batch_fallback(
    model: torch.nn.Module, batch: Any, device: str
) -> tuple[Any, torch.Tensor]:
    if hasattr(model, "prepare_batch"):
        try:
            res = model.prepare_batch(batch, device)
            if isinstance(res, (tuple, list)) and len(res) == 2:
                # If it's a mock returning a tuple, or real model, accept it
                return res
        except Exception:
            pass

    # Fallback logic — maps legacy 'data' key to 'input_ids' for forward() compat
    if isinstance(batch, dict):
        target = batch["target"].to(device) if "target" in batch else torch.tensor(0)
        inputs = {}
        for k, v in batch.items():
            if k == "target":
                continue
            key = "input_ids" if k == "data" else k
            inputs[key] = v.to(device) if hasattr(v, "to") else v
        return inputs, target
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        inputs, target = batch
        if isinstance(inputs, dict):
            inputs = {
                ("input_ids" if k == "data" else k): v.to(device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }
        elif hasattr(inputs, "to"):
            inputs = inputs.to(device)
        if hasattr(target, "to"):
            target = target.to(device)
        return inputs, target
    else:
        return batch, torch.tensor(0)


def _iter_batches(
    model: torch.nn.Module,
    data_source: DataLoader | Any,
    batch_size: int,
    device: str,
) -> Iterator[tuple[Any, torch.Tensor]]:
    """Yield (inputs_dict, target) regardless of source type. The model owns
    the batch→inputs transformation — no model_type dispatch lives here.
    """
    is_dl = isinstance(data_source, DataLoader)
    if not is_dl and hasattr(data_source, "iterable"):
        is_dl = isinstance(data_source.iterable, DataLoader)

    if is_dl:
        for batch in data_source:
            yield _prepare_batch_fallback(model, batch, device)
    elif hasattr(data_source, "iter_torch_batches"):
        # Ray dataset shard
        for batch in data_source.iter_torch_batches(batch_size=batch_size):
            yield _prepare_batch_fallback(model, batch, device)
    else:
        # Generic generator/iterator yielding batches
        for batch in data_source:
            yield _prepare_batch_fallback(model, batch, device)


def create_model_from_registry(
    model_type: str,
    dict_path: str,
    embeddings_config: Any,
    model_config: Any | None = None,
    freeze_embeddings: bool = True,
) -> torch.nn.Module:
    """Create a model using the MODEL_REGISTRY.

    Replaces inline if/elif blocks for model creation in distributed
    and tuning paths with a unified registry lookup.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder', 'modernbert'.
        dict_path: Path to the dictionary file.
        embeddings_config: EmbeddingsConfig instance.
        model_config: Optional model-specific config (for tuning).
        freeze_embeddings: If True, freezes backbone layers initially.

    Returns:
        The created model instance.

    Raises:
        ValueError: If model_type is not in the registry.
    """
    from sentimentizer.models.base import get_model_registry

    registry = get_model_registry()
    if model_type not in registry:
        raise ValueError(f"no matching model for {model_type}")

    entry = registry[model_type]
    new_model_func = entry["new_model"]

    kwargs: dict[str, Any] = {
        "dict_path": dict_path,
        "embeddings_config": embeddings_config,
        "freeze_embeddings": freeze_embeddings,
    }
    if model_config is not None:
        kwargs["model_config"] = model_config

    return new_model_func(**kwargs)


def _create_training_components(
    model: torch.nn.Module,
    model_type: str,
    device: str,
    cfg: TrainerConfig | None = None,
    lr: float | None = None,
    betas: tuple[float, float] | None = None,
    weight_decay: float | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, Callable]:
    """Create optimizer, scheduler, and loss function for a model.

    Uses CrossEntropyLoss (or FocalCrossEntropyLoss) for 3-class
    classification, replacing the binary BCEWithLogitsLoss.

    Args:
        model: The model to create components for.
        model_type: Model type string.
        device: Device to place tensors on.
        cfg: TrainerConfig for loss_type/label_smoothing/focal_gamma settings.
        lr: Optional learning rate override.
        betas: Optional AdamW betas override.
        weight_decay: Optional weight decay override.
        class_weights: Optional per-class weight tensor for CrossEntropyLoss.

    Returns:
        Tuple of (optimizer, scheduler, loss_function).
    """
    from sentimentizer.losses import FocalCrossEntropyLoss

    opt_params = _get_opt_params(model_type)
    sched_params = _get_sched_params(model_type)

    # Use overrides if provided, otherwise use defaults
    lr = lr if lr is not None else opt_params.lr
    betas = betas if betas is not None else opt_params.betas
    weight_decay = weight_decay if weight_decay is not None else opt_params.weight_decay

    # Exclude 1-D params (biases, LayerNorm scale/shift) from weight decay.
    # Applying WD to these interferes with learned offsets and normalization
    # scales — standard practice since BERT/GPT-2.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    use_8bit = cfg.use_8bit_optimizer if cfg else False

    try:
        import bitsandbytes as bnb

        _bnb_available = True
    except ImportError:
        _bnb_available = False

    # bitsandbytes 8-bit AdamW requires CUDA parameters.
    params_on_cuda = False
    if decay:
        params_on_cuda = decay[0].is_cuda
    elif no_decay:
        params_on_cuda = no_decay[0].is_cuda

    if use_8bit and _bnb_available and params_on_cuda:
        optimizer = bnb.optim.AdamW8bit(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )
    else:
        if use_8bit:
            reason = "bitsandbytes or CUDA device not available"
            logger.warning(
                f"[{model_type}] 8-bit optimizer requested but {reason}, "
                f"falling back to standard AdamW"
            )
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )

    # Placeholder scheduler — rebuilt with real step counts in Trainer.fit() /
    # _train_func() before training starts. Uses rough epoch-based defaults here
    # that get replaced once DataLoader length is known.
    _placeholder_total_steps = max(1, sched_params.T_max)
    _placeholder_warmup = max(1, round(sched_params.warmup_ratio * _placeholder_total_steps))
    scheduler = _LinearWarmupCosineScheduler(
        optimizer,
        warmup_steps=_placeholder_warmup,
        total_steps=_placeholder_total_steps,
        eta_min=sched_params.eta_min,
    )

    # Determine loss type and label smoothing from config
    loss_type = cfg.loss_type if cfg else "cross_entropy"
    label_smoothing = cfg.label_smoothing if cfg else 0.1
    focal_gamma = cfg.focal_gamma if cfg else 2.0

    # Move class weights to device if provided
    weights_on_device = class_weights.to(device) if class_weights is not None else None

    if loss_type == "focal":
        loss_function = FocalCrossEntropyLoss(
            weight=weights_on_device,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    else:
        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights_on_device,
            label_smoothing=label_smoothing,
        )

    return optimizer, scheduler, loss_function


def _rebuild_optimizer_after_unfreeze(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Rebuild optimizer and fast-forward scheduler after backbone unfreeze.

    Shared by Trainer.fit(), _run_training_loop(), and _train_func() so that
    the unfreeze logic doesn't drift out of sync across three callsites.

    Returns the new optimizer and (optionally fast-forwarded) scheduler.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    is_8bit = "8bit" in type(optimizer).__name__
    lr = optimizer.param_groups[0]["lr"]
    betas = optimizer.param_groups[0]["betas"]
    weight_decay = optimizer.param_groups[0].get("weight_decay", 1e-4)

    if is_8bit:
        import bitsandbytes as bnb

        new_optimizer: torch.optim.Optimizer = bnb.optim.AdamW8bit(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )
    else:
        new_optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )

    new_scheduler = None
    if scheduler is not None:
        old_last_epoch = scheduler.last_epoch
        new_scheduler = _LinearWarmupCosineScheduler(
            new_optimizer,
            warmup_steps=scheduler.warmup_steps,
            total_steps=scheduler.total_steps,
            eta_min=scheduler.eta_min,
        )
        for _ in range(old_last_epoch):
            new_scheduler.step()

    return new_optimizer, new_scheduler


def _new_loaders(
    train_data: torch.utils.data.Dataset, val_data: torch.utils.data.Dataset, cfg: TrainerConfig
) -> tuple[DataLoader, DataLoader]:
    # Resolve auto-detect dataloader_workers (-1 means auto)
    workers = cfg.dataloader_workers
    if workers == -1:
        workers = default_dataloader_workers(cfg.device)

    pin_mem = cfg.memory if cfg.device != "mps" else False

    collate_fn = None
    # Use isinstance so this works correctly for HFDataset subclasses too
    try:
        from sentimentizer.hf_dataset import HFDataset

        if isinstance(train_data, HFDataset):
            from sentimentizer.hf_dataset import HFCollateFn

            collate_fn = HFCollateFn(pad_token_id=0)
    except ImportError:
        pass

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        num_workers=workers,
        pin_memory=pin_mem,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        num_workers=workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


@dataclass
class Trainer:
    """Trainer class helps with creating the data loader,
    tracking the torch optimizer and model fitting"""

    loss_function: Callable
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    cfg: TrainerConfig
    model_type: str
    val_loss: float = float("inf")
    latest_train_loss: float = 0.0
    latest_epoch: int = 0
    latest_metrics: ClassificationMetrics | None = None

    def rebuild_scheduler(self, steps_per_epoch: int, epochs: int) -> None:
        """Replace the placeholder scheduler with one sized for the real training length."""
        sched_params = _get_sched_params(self.model_type)
        grad_accum = max(1, self.cfg.gradient_accumulation_steps)
        total_steps = epochs * max(1, steps_per_epoch // grad_accum)
        warmup_ratio = getattr(sched_params, "warmup_ratio", 0.06)
        warmup_steps = max(1, round(warmup_ratio * total_steps))
        self.scheduler = _LinearWarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=sched_params.eta_min,
        )
        logger.info(
            f"[{self.model_type}] per-step scheduler: "
            f"total_steps={total_steps}, warmup_steps={warmup_steps}"
        )

    def _train_epoch(self, model: torch.nn.Module, train_loader: DataLoader, epoch: int) -> None:
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        loss_ema = 0.0
        update_every = self.cfg.ray_update_every if self.cfg.ray_update_every > 0 else 50

        grad_accum_steps = self.cfg.gradient_accumulation_steps
        use_amp = self.cfg.use_amp
        accum_step_idx = 0
        last_unflushed = False
        batch_start_time = time.time()
        grad_norm = 0.0

        with logging_redirect_tqdm():
            pbar = tqdm(train_loader, desc=f"[{self.model_type}] epoch {epoch}", leave=False)
            batches = _iter_batches(model, pbar, self.cfg.batch_size, self.cfg.device)
            for i, (inputs, target) in enumerate(batches):
                batch_size = target.size(0)
                loss_val, step_grad_norm = train_step(
                    model=model,
                    inputs=inputs,
                    target=target,
                    optimizer=self.optimizer,
                    loss_function=self.loss_function,
                    use_amp=use_amp,
                    grad_accum_steps=grad_accum_steps,
                    accum_step_idx=accum_step_idx,
                    is_last_batch=False,
                    device_type="cuda" if "cuda" in self.cfg.device else "cpu",
                )

                if (accum_step_idx % grad_accum_steps) == (grad_accum_steps - 1):
                    grad_norm = step_grad_norm

                now = time.time()
                elapsed = now - batch_start_time
                throughput = batch_size / elapsed if elapsed > 0 else 0.0
                batch_start_time = now

                accum_step_idx += 1
                last_unflushed = (accum_step_idx % grad_accum_steps) != 0

                # Per-batch scheduler step: fire after each completed accumulation window
                if (
                    getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False)
                    and self.scheduler is not None
                    and (accum_step_idx % grad_accum_steps) == 0
                ):
                    self.scheduler.step()

                epoch_loss_sum += loss_val * batch_size
                epoch_sample_count += batch_size
                loss_ema = 0.9 * loss_ema + 0.1 * loss_val if i > 0 else loss_val
                if i % update_every == 0:
                    avg_loss = (
                        epoch_loss_sum / epoch_sample_count if epoch_sample_count > 0 else loss_ema
                    )
                    pbar.set_postfix(
                        loss=f"{loss_ema:.4f}",
                        avg_loss=f"{avg_loss:.4f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    )
                    write_batch_snapshot(
                        model_type=self.model_type,
                        run_id=self.cfg.run_id,
                        epoch=epoch,
                        batch=i,
                        loss_ema=loss_ema,
                        avg_loss=avg_loss,
                        lr=self.optimizer.param_groups[0]["lr"],
                        throughput=throughput,
                        grad_norm=grad_norm,
                    )

        if grad_accum_steps > 1 and last_unflushed:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Flush the partial accumulation window's scheduler step too
            if (
                getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False)
                and self.scheduler is not None
            ):
                self.scheduler.step()

        # Store per-epoch weighted average for evaluate() to consume
        self._epoch_loss_sum = epoch_loss_sum  # type: ignore[attr-defined]
        self._epoch_sample_count = epoch_sample_count  # type: ignore[attr-defined]

    def fit(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        patience_counter: int = 0,
    ) -> None:
        train_loader, val_loader = _new_loaders(train_data, val_data, self.cfg)
        model.to(self.cfg.device)
        start = time.time()

        # Resolve default epochs if auto (-1)
        epochs = self.cfg.epochs
        if epochs == -1:
            epochs = default_epochs(self.model_type)

        is_resuming = start_epoch > 1

        # For per-batch-stepping models (e.g. HF transformers), rebuild the
        # scheduler now that we know the DataLoader length. The placeholder
        # built in _create_training_components used epoch-based counts; here
        # we compute real optimizer-step counts and re-initialize the schedule.
        # On resume, the scheduler was restored from checkpoint with the correct
        # step count, so we skip the rebuild — unless the checkpoint had no
        # scheduler state (old-format checkpoint), in which case we must rebuild
        # and fast-forward past the already-completed epochs.
        step_per_batch = getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False)
        scheduler_needs_rebuild = step_per_batch and self.scheduler is not None
        completed_steps = 0
        if is_resuming:
            # If the checkpoint had scheduler state, load_checkpoint already
            # restored it — no rebuild needed.  Otherwise, rebuild and
            # fast-forward to approximate where training left off.
            has_scheduler_state = (
                hasattr(self.scheduler, "last_epoch") and self.scheduler.last_epoch > 0
            )
            if not has_scheduler_state:
                scheduler_needs_rebuild = True
                # Approximate the number of optimizer steps already completed
                # so we can fast-forward the rebuilt scheduler.
                steps_per_epoch_est = len(train_loader)
                grad_accum = max(1, self.cfg.gradient_accumulation_steps)
                completed_steps = (start_epoch - 1) * max(1, steps_per_epoch_est // grad_accum)
                logger.warning(
                    f"[{self.model_type}] checkpoint had no scheduler state, "
                    f"rebuilding scheduler and fast-forwarding {completed_steps} steps "
                    f"(LR may not match original schedule exactly)"
                )
            else:
                scheduler_needs_rebuild = False
        if scheduler_needs_rebuild:
            self.rebuild_scheduler(len(train_loader), epochs)
            # Fast-forward past already-completed epochs so the LR is
            # approximately where it would have been at this point.
            if completed_steps > 0:
                for _ in range(completed_steps):
                    self.scheduler.step()

        if is_resuming:
            # For per-epoch schedulers (RNN/Encoder/Decoder), fast-forward
            # past already-completed epochs so the LR is correct.
            if (
                self.scheduler is not None
                and not step_per_batch
                and hasattr(self.scheduler, "last_epoch")
                and self.scheduler.last_epoch < (start_epoch - 1)
            ):
                logger.info(
                    f"[{self.model_type}] fast-forwarding per-epoch scheduler "
                    f"from step {self.scheduler.last_epoch} to step {start_epoch - 1}"
                )
                for _ in range(start_epoch - 1 - self.scheduler.last_epoch):
                    self.scheduler.step()
            logger.info(f"[{self.model_type}] resuming from epoch {start_epoch}")

            # Emit validation metrics from the checkpoint so the dashboard
            # shows something immediately instead of waiting for the first
            # resumed epoch to complete.
            self.evaluate(model, val_loader, start_epoch - 1)

        logger.info(
            f"[{self.model_type}] fitting model...",
        )
        logger.info(
            f"[{self.model_type}] config: model={model.__class__.__name__}, "
            f"epochs={epochs}, device={self.cfg.device}"
        )

        checkpoint_dir = self.cfg.checkpoint_dir
        checkpoint_every = self.cfg.checkpoint_every
        checkpoint_best = self.cfg.checkpoint_best

        # Create checkpoint directory if checkpointing is enabled
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        try:
            for epoch in range(start_epoch, epochs + 1):
                self._train_epoch(model, train_loader, epoch)
                self.evaluate(model, val_loader, epoch)

                if self.scheduler and not getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False):
                    self.scheduler.step()

                # Handle backbone unfreezing for HF models or configured models
                config = getattr(model, "config", None)
                freeze_epochs = getattr(config, "freeze_backbone_epochs", 0) if config else 0
                if freeze_epochs > 0 and epoch == freeze_epochs:
                    logger.info(
                        f"[{self.model_type}] unfreezing backbone at epoch {epoch}, "
                        f"rebuilding optimizer with all trainable parameters"
                    )
                    inner_model = model.module if hasattr(model, "module") else model
                    inner_model.unfreeze_backbone()
                    self.optimizer, self.scheduler = _rebuild_optimizer_after_unfreeze(
                        model, self.optimizer, self.scheduler
                    )

                # Update best_val_loss and patience before checkpointing
                improved = self.val_loss < best_val_loss
                if improved:
                    best_val_loss = self.val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Save periodic checkpoint
                if checkpoint_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
                    ckpt_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
                    save_checkpoint(
                        model,
                        self.optimizer,
                        epoch,
                        ckpt_path,
                        scheduler=self.scheduler,
                        val_loss=self.val_loss,
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                    )
                    logger.info(f"[{self.model_type}] saved checkpoint: {ckpt_path}")

                # Save best model checkpoint
                if checkpoint_dir and checkpoint_best and improved:
                    best_path = Path(checkpoint_dir) / BEST_CHECKPOINT_NAME
                    save_checkpoint(
                        model,
                        self.optimizer,
                        epoch,
                        best_path,
                        scheduler=self.scheduler,
                        val_loss=self.val_loss,
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                    )
                    logger.info(
                        f"[{self.model_type}] saved best model checkpoint "
                        f"(val_loss={self.val_loss:.4f}): {best_path}"
                    )

                # Early stopping based on validation loss
                if (
                    self.cfg.early_stopping_patience > 0
                    and patience_counter >= self.cfg.early_stopping_patience
                ):
                    logger.info(
                        f"[{self.model_type}] early stopping at epoch {epoch}, "
                        f"val_loss hasn't improved for {patience_counter} epochs"
                    )
                    break

                logger.info(
                    f"[{self.model_type}] [epoch {epoch}] completed, val_loss={self.val_loss:.4f}"
                )
        finally:
            # Release CUDA resources even on Ctrl-C or exception.
            # Prevents error 804 ("forward compatibility was attempted on
            # non supported HW") caused by orphaned GPU contexts.
            if self.cfg.device in ("cuda", "mps") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        logger.info(
            f"[{self.model_type}] model fitting completed, {time.time() - start:.0f} seconds passed"
        )

    def evaluate(self, model: torch.nn.Module, val_loader: DataLoader, epoch: int) -> None:
        logger.info(f"[{self.model_type}] [epoch {epoch}] evaluating predictions...")
        model.to(self.cfg.device)
        model.eval()

        all_probs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        val_loss_sum = 0.0
        val_sample_count = 0

        with logging_redirect_tqdm():
            pbar = tqdm(val_loader, desc=f"[{self.model_type}] eval {epoch}", leave=False)
            with torch.no_grad():
                batches = _iter_batches(model, pbar, self.cfg.batch_size, self.cfg.device)
                for inputs, target in batches:
                    batch_size = target.size(0)
                    logits = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                    loss_val = self.loss_function(logits, target)
                    val_loss_sum += loss_val.item() * batch_size
                    val_sample_count += batch_size

                    all_probs.append(torch.softmax(logits, dim=-1).cpu())
                    all_targets.append(target.cpu())

        probabilities = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        # Free CPU memory from per-batch tensor lists now that we have
        # the concatenated numpy arrays.  Also release any GPU cache
        # held by validation intermediates before the next training epoch.
        del all_probs, all_targets
        if "cuda" in self.cfg.device and torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = compute_epoch_metrics(probabilities, targets, self.model_type)

        self.val_loss = val_loss_sum / val_sample_count if val_sample_count > 0 else 0.0
        # Per-epoch weighted train loss from _train_epoch
        train_loss_sum = getattr(self, "_epoch_loss_sum", 0.0)
        train_sample_count = getattr(self, "_epoch_sample_count", 0)
        self.latest_train_loss = (
            train_loss_sum / train_sample_count if train_sample_count > 0 else 0.0
        )
        self.latest_epoch = epoch
        self.latest_metrics = metrics

        # Publish metrics to all backends (Ray gauges, Prometheus, JSON, logger)
        publish_epoch_metrics(
            model_type=self.model_type,
            run_id=self.cfg.run_id,
            epoch=epoch,
            train_loss=self.latest_train_loss,
            val_loss=self.val_loss,
            metrics=metrics,
            lr=self.optimizer.param_groups[0]["lr"],
            ray_gauges=_get_ray_gauges(self.model_type, self.cfg.run_id),
        )


def _run_training_loop(
    model: torch.nn.Module,
    train_iter: Any,
    val_iter: Any,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_function: Callable,
    callbacks: list[TrainingCallback],
    model_type: str,
    device: str,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    batch_size: int = 32,
    update_every: int = 50,
) -> TrainingState:
    """Unified training loop used by all three paths (single-node, distributed, tuning).

    Args:
        model: The model to train.
        train_iter: Iterator yielding batches for training.
        val_iter: Iterator yielding batches for validation.
        epochs: Number of epochs to train.
        optimizer: Optimizer for training.
        scheduler: Optional LR scheduler.
        loss_function: Loss function.
        callbacks: List of TrainingCallback instances.
        model_type: Model type string for logging.
        device: Device string.
        use_amp: Whether to use mixed precision (AMP).
        grad_accum_steps: Number of gradient accumulation steps.
        batch_size: Batch size.

    Returns:
        TrainingState with final training state.
    """
    state = TrainingState()

    for cb in callbacks:
        cb.on_train_begin(model_type, device, epochs)

    device_type = "cuda" if "cuda" in device else "cpu"
    accum_step_idx = 0
    last_unflushed = False

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        loss_ema = 0.0
        current_update_every = update_every if update_every > 0 else 50

        with logging_redirect_tqdm():
            pbar = tqdm(train_iter, desc=f"[{model_type}] epoch {epoch}", leave=False)
            batches = _iter_batches(model, pbar, batch_size, device)
            for i, (inputs, target) in enumerate(batches):
                curr_batch_size = target.size(0)
                loss_val, _ = train_step(
                    model=model,
                    inputs=inputs,
                    target=target,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    use_amp=use_amp,
                    grad_accum_steps=grad_accum_steps,
                    accum_step_idx=accum_step_idx,
                    is_last_batch=False,
                    device_type=device_type,
                )
                accum_step_idx += 1
                last_unflushed = (accum_step_idx % grad_accum_steps) != 0

                # Per-batch scheduler step (all models)
                if (
                    getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False)
                    and scheduler is not None
                    and (accum_step_idx % grad_accum_steps) == 0
                ):
                    scheduler.step()

                epoch_loss_sum += loss_val * curr_batch_size
                epoch_sample_count += curr_batch_size
                state.running_loss_mean = (
                    (state.running_loss_mean * state.steps + loss_val) / (state.steps + 1)
                    if state.steps > 0
                    else loss_val
                )
                state.steps += 1
                loss_ema = 0.9 * loss_ema + 0.1 * loss_val if i > 0 else loss_val
                if i % current_update_every == 0:
                    avg_loss = (
                        epoch_loss_sum / epoch_sample_count if epoch_sample_count > 0 else loss_ema
                    )
                    pbar.set_postfix(
                        loss=f"{loss_ema:.4f}",
                        avg_loss=f"{avg_loss:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                    )

        # Flush partial accumulation window left over at epoch boundary
        if grad_accum_steps > 1 and last_unflushed:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False) and scheduler is not None:
                scheduler.step()

        train_loss = epoch_loss_sum / epoch_sample_count if epoch_sample_count > 0 else 0.0
        state.latest_train_loss = train_loss

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            batches = _iter_batches(model, val_iter, batch_size, device)
            for inputs, target in batches:
                curr_batch_size = target.size(0)
                logits = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                loss_val = loss_function(logits, target)
                val_loss_sum += loss_val.item() * curr_batch_size
                val_sample_count += curr_batch_size

                all_probs.append(torch.softmax(logits, dim=-1).cpu())
                all_targets.append(target.cpu())

        val_loss = val_loss_sum / val_sample_count if val_sample_count > 0 else 0.0
        probabilities = torch.cat(all_probs).numpy() if all_probs else np.array([])
        targets = torch.cat(all_targets).numpy() if all_targets else np.array([])

        # Free GPU memory held by validation intermediates before next training epoch
        del all_probs, all_targets
        if device_type == "cuda":
            torch.cuda.empty_cache()

        metrics = compute_epoch_metrics(probabilities, targets, model_type)

        lr = optimizer.param_groups[0]["lr"]
        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            lr=lr,
        )

        state.val_loss = val_loss
        state.latest_epoch = epoch
        state.latest_metrics = metrics

        # Track best val loss and patience for checkpoint resume
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            state.patience_counter = 0
        else:
            state.patience_counter += 1

        if scheduler is not None and not getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False):
            scheduler.step()

        # Callbacks
        should_stop = False
        for cb in callbacks:
            if cb.on_epoch_end(result, state):
                should_stop = True
                break

        # Handle unfreezing backbone signaled by UnfreezeBackboneCallback
        if getattr(state, "_pending_unfreeze", False):
            inner_model = model.module if hasattr(model, "module") else model
            inner_model.unfreeze_backbone()
            optimizer, scheduler = _rebuild_optimizer_after_unfreeze(model, optimizer, scheduler)
            state._pending_unfreeze = False

        # Handle checkpointing signaled by CheckpointCallback
        if hasattr(state, "_pending_checkpoint_path"):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                state._pending_checkpoint_path,
                scheduler=scheduler,
                val_loss=val_loss,
                best_val_loss=state.best_val_loss,
                patience_counter=state.patience_counter,
            )
            delattr(state, "_pending_checkpoint_path")

        if hasattr(state, "_pending_best_checkpoint_path"):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                state._pending_best_checkpoint_path,
                scheduler=scheduler,
                val_loss=val_loss,
                best_val_loss=state.best_val_loss,
                patience_counter=state.patience_counter,
            )
            delattr(state, "_pending_best_checkpoint_path")

        if should_stop:
            break

    for cb in callbacks:
        cb.on_train_end(state)

    return state


def _get_opt_params(
    model_type: str,
) -> (
    OptimizationParams
    | EncoderOptimizationParams
    | DecoderOptimizationParams
    | ModernBERTOptimizationParams
):
    """Return optimization params appropriate for the model type."""
    if model_type == "decoder":
        return DecoderOptimizationParams()
    if model_type == "encoder":
        return EncoderOptimizationParams()
    if model_type == "rnn":
        return OptimizationParams()
    if model_type == "modernbert":
        return ModernBERTOptimizationParams()
    raise ValueError(f"no matching model: {model_type}")


def _get_sched_params(
    model_type: str,
) -> (
    SchedulerParams
    | RNNSchedulerParams
    | EncoderSchedulerParams
    | DecoderSchedulerParams
    | ModernBERTSchedulerParams
):
    """Return scheduler params appropriate for the model type."""
    if model_type == "encoder":
        return EncoderSchedulerParams()
    if model_type == "decoder":
        return DecoderSchedulerParams()
    if model_type == "rnn":
        return RNNSchedulerParams()
    if model_type == "modernbert":
        return ModernBERTSchedulerParams()
    raise ValueError(f"no matching model: {model_type}")


class _LinearWarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine decay.

    During warmup, LR increases linearly from base_lr/warmup_steps to base_lr.
    After warmup, follows cosine decay so that the final LR equals ``eta_min``.
    Used by all model types for per-batch scheduler stepping.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        # LambdaLR multiplies base_lr by the lambda's return value, so to land
        # at an absolute eta_min we have to feed in eta_min / base_lr.  Without
        # this rescaling, the multiplier itself decays to eta_min and the
        # effective LR bottoms out at base_lr * eta_min (e.g. 1e-4 * 1e-6 = 1e-10).
        base_lrs = [g["lr"] for g in optimizer.param_groups]

        # The lambdas read the plan from self rather than capturing it:
        # load_state_dict() restores these attributes but cannot restore closures,
        # so captured values would keep a placeholder scheduler's step counts on resume.
        def make_lambda(base_lr: float) -> Callable[[int], float]:
            def lr_lambda(step: int) -> float:
                if step < self.warmup_steps:
                    return (step + 1) / max(1, self.warmup_steps)
                relative_eta_min = self.eta_min / base_lr if base_lr > 0 else 0.0
                progress = min(
                    1.0,
                    (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps),
                )
                return relative_eta_min + (1.0 - relative_eta_min) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )

            return lr_lambda

        super().__init__(optimizer, [make_lambda(lr) for lr in base_lrs])


def new_trainer(
    model: torch.nn.Module,
    cfg: TrainerConfig,
    model_type: str,
    lr: float | None = None,
    weight_decay: float | None = None,
) -> Trainer:
    # Build class_weights tensor from config if provided
    class_weights_tensor = None
    if cfg.class_weights is not None:
        class_weights_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32)

    optimizer, scheduler, loss_function = _create_training_components(
        model=model,
        model_type=model_type,
        device=cfg.device,
        cfg=cfg,
        lr=lr,
        weight_decay=weight_decay,
        class_weights=class_weights_tensor,
    )

    trainer = Trainer(
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        model_type=model_type,
    )

    return trainer


# ──────────────────────────────────────────────
# Ray Train distributed training
# ──────────────────────────────────────────────


def _gather_validation(
    loss_sum: float, sample_count: int, probabilities: np.ndarray, targets: np.ndarray
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Combine validation results across workers, so every rank sees the same numbers.

    Each worker validates only its own shard. Without this, rank 0 would report metrics
    for part of the data, and workers could disagree about early stopping and leave DDP
    waiting on a rank that already stopped.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return loss_sum, sample_count, probabilities, targets
    gathered: list[Any] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, (loss_sum, sample_count, probabilities, targets))
    return (
        sum(part[0] for part in gathered),
        sum(part[1] for part in gathered),
        np.concatenate([part[2] for part in gathered]),
        np.concatenate([part[3] for part in gathered]),
    )


def _report_metrics(
    train_loss: float, val_loss: float, metrics: ClassificationMetrics, epoch: int, lr: float
) -> dict[str, Any]:
    """Build the per-epoch metrics dict reported to Ray Train.

    The driver persists the final report for the exporter, so anything missing here
    would overwrite that gauge's real value with zero when the run ends.
    """
    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "negative_precision": metrics.negative_precision,
        "negative_recall": metrics.negative_recall,
        "negative_f1": metrics.negative_f1,
        "neutral_precision": metrics.neutral_precision,
        "neutral_recall": metrics.neutral_recall,
        "neutral_f1": metrics.neutral_f1,
        "positive_precision": metrics.positive_precision,
        "positive_recall": metrics.positive_recall,
        "positive_f1": metrics.positive_f1,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "cohen_kappa": metrics.cohen_kappa,
        "mcc": metrics.mcc,
        "neutral_to_positive_rate": metrics.neutral_to_positive_rate,
        "neutral_to_negative_rate": metrics.neutral_to_negative_rate,
        "pred_neutral_frac": metrics.pred_neutral_frac,
        "neutral_auc_roc": metrics.neutral_auc_roc,
        "neutral_avg_precision": metrics.neutral_avg_precision,
        "total": metrics.total,
        "epoch": epoch,
        "lr": lr,
    }


def _train_func(config: dict) -> None:
    """Training function executed on each Ray Train worker.

    Creates the model on the worker, wraps it with DDP via prepare_model(),
    and runs the training loop with dataset shards from Ray Data.
    Reports metrics and checkpoints back to Ray Train.
    """

    # Suppress smart_open verbose logging in Ray workers.
    # smart_open submodules (s3, http, etc.) log INFO messages during
    # initialization which flood Ray worker output. Setting the parent
    # logger suppresses all children that propagate.
    import logging

    logging.getLogger("smart_open").setLevel(logging.WARNING)
    # Also suppress any child loggers that may already exist
    for _name in list(logging.root.manager.loggerDict):
        if _name.startswith("smart_open"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    # Unpack training config (resolve -1 to model-specific default)
    epochs = config["epochs"]
    if epochs == -1:
        epochs = default_epochs(config["model_type"])
    batch_size = config["batch_size"]
    lr = config["lr"]
    betas = tuple(config["betas"])
    weight_decay = config["weight_decay"]
    warmup_ratio = config.get("warmup_ratio", 0.06)
    scheduler_eta_min = config.get("scheduler_eta_min", 1e-6)

    # Unpack model config
    model_type = config["model_type"]
    run_id = config.get("run_id", "")
    dict_path = config["dict_path"]
    embeddings_model_name = config["embeddings_model_name"]
    embeddings_emb_length = config["embeddings_emb_length"]
    freeze_embeddings = config.get("freeze_embeddings", True)
    # input_len is no longer needed by new_model() — removed from config

    # Create model on this worker
    from sentimentizer.config import EmbeddingsConfig

    embeddings_config = EmbeddingsConfig(
        model_name=embeddings_model_name,
        emb_length=embeddings_emb_length,
    )

    model = create_model_from_registry(
        model_type=model_type,
        dict_path=dict_path,
        embeddings_config=embeddings_config,
        freeze_embeddings=freeze_embeddings,
    )

    # Prepare model for distributed training (DDP) and move to correct device
    model = prepare_model(model)

    # Determine device from model (set by prepare_model)
    device = next(model.parameters()).device

    # Set up optimizer — exclude 1-D params from weight decay (same as single-node path).
    _decay, _no_decay = [], []
    for _name, _param in model.named_parameters():
        if not _param.requires_grad:
            continue
        if _param.ndim == 1 or _name.endswith(".bias") or "embed" in _name:
            _no_decay.append(_param)
        else:
            _decay.append(_param)
    use_8bit = config.get("use_8bit_optimizer", False)
    try:
        import bitsandbytes as bnb

        _bnb_available = True
    except ImportError:
        _bnb_available = False

    # bitsandbytes 8-bit AdamW requires CUDA parameters.
    params_on_cuda = False
    if _decay:
        params_on_cuda = _decay[0].is_cuda
    elif _no_decay:
        params_on_cuda = _no_decay[0].is_cuda

    if use_8bit and _bnb_available and params_on_cuda:
        optimizer = bnb.optim.AdamW8bit(
            [
                {"params": _decay, "weight_decay": weight_decay},
                {"params": _no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )
    else:
        if use_8bit:
            reason = "bitsandbytes or CUDA device not available"
            logger.warning(
                f"[{model_type}] 8-bit optimizer requested but {reason}, "
                f"falling back to standard AdamW (distributed worker)"
            )
        optimizer = torch.optim.AdamW(
            [
                {"params": _decay, "weight_decay": weight_decay},
                {"params": _no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )

    # Placeholder scheduler — rebuilt with real step counts below for
    # per-batch models. All models use _LinearWarmupCosineScheduler now.
    _sched_params = _get_sched_params(model_type)
    _placeholder_total_steps = max(1, epochs)
    _placeholder_warmup = max(1, round(warmup_ratio * _placeholder_total_steps))
    scheduler = _LinearWarmupCosineScheduler(
        optimizer,
        warmup_steps=_placeholder_warmup,
        total_steps=_placeholder_total_steps,
        eta_min=scheduler_eta_min,
    )
    # Use CrossEntropyLoss for 3-class classification
    loss_type = config.get("loss_type", "cross_entropy")
    label_smoothing = config.get("label_smoothing", 0.1)

    # Build class weights tensor from config if provided
    class_weights_list = config.get("class_weights")
    class_weights_tensor = None
    if class_weights_list is not None:
        class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)

    if loss_type == "focal":
        from sentimentizer.losses import FocalCrossEntropyLoss

        focal_gamma = config.get("focal_gamma", 2.0)
        loss_function = FocalCrossEntropyLoss(
            weight=class_weights_tensor,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    else:
        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=label_smoothing,
        )

    # Get dataset shards for this worker
    # In Ray 2.55+, get_dataset_shard is a standalone function, not a method on get_context()
    # See https://docs.ray.io/en/2.55.1/train/api/doc/ray.train.get_dataset_shard.html
    train_shard = train.get_dataset_shard("train")
    val_shard = train.get_dataset_shard("val")

    start = time.time()
    logger.info(
        f"[{model_type}] fitting model (distributed)...",
    )
    logger.info(
        f"[{model_type}] config: model={model.__class__.__name__}, "
        f"epochs={epochs}, lr={lr}, device={device}, "
        f"warmup_ratio={warmup_ratio}, scheduler_eta_min={scheduler_eta_min}"
    )

    is_rank_0 = train.get_context().get_world_rank() == 0
    ray_gauges = _get_ray_gauges(model_type, run_id)

    # For per-batch-stepping models, rebuild scheduler with real optimizer-step
    # counts before training starts. Uses the same dataset-count estimate that
    # drives the tqdm total_batches calculation below.
    # DDP does not forward custom methods such as prepare_batch to the wrapped model.
    inner_model = model.module if hasattr(model, "module") else model
    if getattr(type(inner_model), "STEP_SCHEDULER_PER_BATCH", False) and scheduler is not None:
        _train_dataset_count = config.get("train_dataset_count", 0)
        _num_workers = config.get("num_workers", 1)
        _grad_accum = max(1, config.get("gradient_accumulation_steps", 1))
        if _train_dataset_count > 0:
            _per_worker_rows = math.ceil(_train_dataset_count / max(1, _num_workers))
            _steps_per_epoch = max(1, math.ceil(_per_worker_rows / batch_size) // _grad_accum)
        else:
            _steps_per_epoch = 100  # conservative fallback; logged as warning
            logger.warning(
                f"[{model_type}] train_dataset_count not available; "
                f"using steps_per_epoch={_steps_per_epoch} for scheduler init"
            )
        _total_steps = epochs * _steps_per_epoch
        _sched_params = _get_sched_params(model_type)
        _warmup_ratio = getattr(_sched_params, "warmup_ratio", 0.06)
        _warmup_steps = max(1, round(_warmup_ratio * _total_steps))
        scheduler = _LinearWarmupCosineScheduler(
            optimizer,
            warmup_steps=_warmup_steps,
            total_steps=_total_steps,
            eta_min=_sched_params.eta_min,
        )
        logger.info(
            f"[{model_type}] per-step scheduler (distributed): "
            f"total_steps={_total_steps}, warmup_steps={_warmup_steps}"
        )

    ray_best_val_loss = float("inf")
    ray_patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        ray_loss_ema = 0.0
        # Reduce log frequency to avoid console spam in distributed training.
        ray_update_every = config.get("ray_update_every", -1)
        if ray_update_every <= 0:
            ray_update_every = 100 if model_type == "modernbert" else 500

        # Compute total_batches from driver-provided dataset count.
        # Ray streaming shards don't expose .count(), so we pass the
        # full dataset size from the driver and divide by world size.
        total_batches = None
        train_dataset_count = config.get("train_dataset_count")
        num_workers = config.get("num_workers", 1)
        if train_dataset_count is not None and train_dataset_count > 0:
            per_worker_rows = math.ceil(train_dataset_count / max(1, num_workers))
            total_batches = math.ceil(per_worker_rows / batch_size)

        import contextlib

        # HF models store ragged input_ids/attention_mask columns that Ray
        # cannot auto-convert to torch tensors.  Provide a collate_fn that
        # pads variable-length sequences so iter_torch_batches succeeds.
        _ray_collate_fn = None
        if model_type == "modernbert":
            from sentimentizer.hf_dataset import ray_hf_collate_fn

            _ray_collate_fn = ray_hf_collate_fn

        if is_rank_0:
            try:
                from ray.experimental.tqdm_ray import tqdm as tqdm_ray
            except ImportError:
                tqdm_ray = tqdm

            if tqdm_ray is tqdm:
                ctx = logging_redirect_tqdm()
                train_pbar = tqdm(
                    train_shard.iter_torch_batches(
                        batch_size=batch_size, collate_fn=_ray_collate_fn
                    ),
                    desc=f"[{model_type}] epoch {epoch}",
                    total=total_batches,
                    leave=False,
                )
            else:
                ctx = contextlib.nullcontext()
                train_pbar = tqdm_ray(
                    train_shard.iter_torch_batches(
                        batch_size=batch_size, collate_fn=_ray_collate_fn
                    ),
                    desc=f"[{model_type}] epoch {epoch}",
                    total=total_batches,
                )
        else:
            ctx = contextlib.nullcontext()
            train_pbar = train_shard.iter_torch_batches(
                batch_size=batch_size, collate_fn=_ray_collate_fn
            )

        use_amp = config.get("use_amp", False)
        grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        accum_step_idx_epoch = 0
        last_unflushed_epoch = False
        batch_start_time = time.time()
        grad_norm = 0.0

        with ctx:
            for i, batch in enumerate(train_pbar):
                inputs, target = inner_model.prepare_batch(batch, device)
                bs = target.size(0)
                loss_val, step_grad_norm = train_step(
                    model,
                    inputs=inputs,
                    target=target,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    use_amp=use_amp,
                    grad_accum_steps=grad_accum_steps,
                    accum_step_idx=accum_step_idx_epoch,
                    is_last_batch=False,
                    device_type=device_type,
                )

                if (accum_step_idx_epoch % grad_accum_steps) == (grad_accum_steps - 1):
                    grad_norm = step_grad_norm

                now = time.time()
                elapsed = now - batch_start_time
                throughput = bs / elapsed if elapsed > 0 else 0.0
                batch_start_time = now

                accum_step_idx_epoch += 1
                last_unflushed_epoch = (accum_step_idx_epoch % grad_accum_steps) != 0

                # Per-batch scheduler step (all models)
                if (
                    getattr(type(inner_model), "STEP_SCHEDULER_PER_BATCH", False)
                    and scheduler is not None
                    and (accum_step_idx_epoch % grad_accum_steps) == 0
                ):
                    scheduler.step()

                epoch_loss_sum += loss_val * bs
                epoch_sample_count += bs
                ray_loss_ema = 0.9 * ray_loss_ema + 0.1 * loss_val if i > 0 else loss_val
                if i % ray_update_every == 0:
                    avg_loss = (
                        epoch_loss_sum / epoch_sample_count
                        if epoch_sample_count > 0
                        else ray_loss_ema
                    )
                    if hasattr(train_pbar, "set_postfix"):
                        train_pbar.set_postfix(
                            loss=f"{ray_loss_ema:.4f}",
                            avg_loss=f"{avg_loss:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                        )
                    if is_rank_0:
                        logger.info(
                            f"[{model_type}] epoch {epoch} batch {i}: "
                            f"loss={ray_loss_ema:.4f}, "
                            f"avg_loss={avg_loss:.4f}, "
                            f"lr={optimizer.param_groups[0]['lr']:.6f}"
                        )
                    # Update Ray gauges for real-time Prometheus visibility
                    if ray_gauges is not None:
                        ray_gauges["train_loss_ema"].set(ray_loss_ema)
                        ray_gauges["train_loss_avg"].set(avg_loss)
                        ray_gauges["train_batch"].set(i)
                        ray_gauges["train_throughput"].set(throughput)
                        ray_gauges["train_grad_norm"].set(grad_norm)
                        ray_gauges["epoch"].set(epoch)
                        ray_gauges["lr"].set(optimizer.param_groups[0]["lr"])

                    # Write lightweight batch snapshot for dashboard visibility
                    # (picked up by standalone exporter within 10s)
                    if is_rank_0:
                        write_batch_snapshot(
                            model_type=model_type,
                            run_id=run_id,
                            epoch=epoch,
                            batch=i,
                            loss_ema=ray_loss_ema,
                            avg_loss=avg_loss,
                            lr=optimizer.param_groups[0]["lr"],
                            throughput=throughput,
                            grad_norm=grad_norm,
                        )

        # Flush partial gradient accumulation window at epoch boundary
        if grad_accum_steps > 1 and last_unflushed_epoch:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if (
                getattr(type(inner_model), "STEP_SCHEDULER_PER_BATCH", False)
                and scheduler is not None
            ):
                scheduler.step()

        # Backbone unfreezing for HF models (e.g. ModernBERT)
        freeze_backbone_epochs = config.get("freeze_backbone_epochs", 0)
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
            logger.info(
                f"[{model_type}] unfreezing backbone at epoch {epoch} (distributed), "
                f"rebuilding optimizer"
            )
            inner_model = model.module if hasattr(model, "module") else model
            inner_model.unfreeze_backbone()
            optimizer, scheduler = _rebuild_optimizer_after_unfreeze(model, optimizer, scheduler)

        # Validation
        logger.info(f"[{model_type}] [epoch {epoch}] evaluating predictions...")
        val_loss_sum = 0.0
        val_sample_count = 0
        all_probs = []
        all_targets = []
        model.eval()
        with torch.no_grad():
            for batch in val_shard.iter_torch_batches(
                batch_size=batch_size, collate_fn=_ray_collate_fn
            ):
                inputs, target = inner_model.prepare_batch(batch, device)
                bs = target.size(0)
                logits = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                loss_val = loss_function(logits, target)
                val_loss_sum += loss_val.item() * bs
                val_sample_count += bs

                all_probs.append(torch.softmax(logits, dim=-1).cpu())
                all_targets.append(target.cpu())

        train_loss = epoch_loss_sum / epoch_sample_count if epoch_sample_count > 0 else 0.0

        val_loss_sum, val_sample_count, probabilities, targets = _gather_validation(
            val_loss_sum,
            val_sample_count,
            torch.cat(all_probs).numpy(),
            torch.cat(all_targets).numpy(),
        )
        val_loss = val_loss_sum / val_sample_count if val_sample_count > 0 else 0.0

        # Free GPU memory held by validation intermediates before next training epoch
        del all_probs, all_targets
        if device_type == "cuda":
            torch.cuda.empty_cache()

        metrics = compute_epoch_metrics(probabilities, targets, model_type)

        if val_loss < ray_best_val_loss:
            ray_best_val_loss = val_loss
            ray_patience_counter = 0
        else:
            ray_patience_counter += 1

        if scheduler and not getattr(type(inner_model), "STEP_SCHEDULER_PER_BATCH", False):
            scheduler.step()

        # Publish metrics from rank 0 to prevent worker collisions
        if train.get_context().get_world_rank() == 0:
            publish_epoch_metrics(
                model_type=model_type,
                run_id=run_id,
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                metrics=metrics,
                lr=optimizer.param_groups[0]["lr"],
                ray_gauges=_get_ray_gauges(model_type, run_id),
            )

        # Report metrics and checkpoint to Ray Train
        # Ray 2.55+ requires directory-based checkpoints (from_dict removed)
        import tempfile

        import ray.cloudpickle as pickle

        # prepare_model() wraps the model in DDP, which prefixes every key
        # in state_dict with "module.".  Loading those keys into a non-DDP
        # model (e.g. via get_trained_model() for serving) raises
        # "Missing key(s) in state_dict".  Unwrap to .module here so the
        # saved checkpoint matches the bare model architecture.
        inner_model = model.module if hasattr(model, "module") else model
        # NOTE: train.report() with checkpoint= reads the checkpoint files
        # from disk when called, so the temp directory must still exist.
        # Keep train.report() INSIDE the with-block to avoid FileNotFoundError
        # from pyarrow.fs.copy_files when the directory is auto-deleted.
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            ckpt_dir_path = Path(checkpoint_dir)
            checkpoint_data = inner_model.save_to_checkpoint_dir(ckpt_dir_path)
            checkpoint_data.update(
                {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "best_val_loss": float(ray_best_val_loss),
                    "patience_counter": ray_patience_counter,
                }
            )
            if scheduler is not None:
                checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
            with open(os.path.join(checkpoint_dir, "data.pkl"), "wb") as fp:
                pickle.dump(checkpoint_data, fp)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                _report_metrics(
                    train_loss, val_loss, metrics, epoch, optimizer.param_groups[0]["lr"]
                ),
                checkpoint=checkpoint,
            )

        # Safe to stop here: every rank computed the same val_loss in _gather_validation.
        early_stopping_patience = config.get("early_stopping_patience", 0)
        if 0 < early_stopping_patience <= ray_patience_counter:
            logger.info(
                f"[{model_type}] early stopping at epoch {epoch}, "
                f"val_loss hasn't improved for {ray_patience_counter} epochs"
            )
            break

    logger.info(f"[{model_type}] model fitting completed, {time.time() - start:.0f} seconds passed")


# ──────────────────────────────────────────────
# Checkpoint save / load
# ──────────────────────────────────────────────


def _checkpoint_artifact_dir(path: str | Path) -> Path:
    """Return the directory holding on-disk artifacts for one checkpoint file.

    Each checkpoint needs its own directory: models that keep weights outside the
    metadata file (HF backbones) would otherwise overwrite every earlier checkpoint.
    """
    return Path(path).with_suffix("")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    val_loss: float | None = None,
    best_val_loss: float | None = None,
    patience_counter: int | None = None,
    tokenizer: Any = None,
) -> None:
    """Save a training checkpoint. Model decides its own on-disk layout.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer to checkpoint.
        epoch: Current epoch number (1-based).
        path: File path to save the checkpoint (.pth).
        scheduler: Optional LR scheduler to include in the checkpoint.
        val_loss: Optional validation loss to include in the checkpoint.
        best_val_loss: Optional best validation loss seen so far.
        patience_counter: Optional early-stopping patience counter.
        tokenizer: Optional tokenizer to include for transformer models.
    """
    inner = model.module if hasattr(model, "module") else model
    if hasattr(inner, "save_to_checkpoint_dir"):
        metadata = inner.save_to_checkpoint_dir(_checkpoint_artifact_dir(path), tokenizer=tokenizer)
    else:
        metadata = {"model_state_dict": inner.state_dict()}
    metadata.update(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
    )
    if scheduler is not None:
        metadata["scheduler_state_dict"] = scheduler.state_dict()
    if val_loss is not None:
        metadata["val_loss"] = val_loss
    if best_val_loss is not None:
        metadata["best_val_loss"] = best_val_loss
    if patience_counter is not None:
        metadata["patience_counter"] = patience_counter

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically and last, so a checkpoint file that exists is always complete
    # and its artifacts are already on disk.
    tmp_path = Path(f"{path}.tmp")
    torch.save(metadata, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"checkpoint saved: {path} (epoch={epoch})")

    if getattr(type(inner), "NEEDS_TOKENIZE_STAGE", True) is False and tokenizer is None:
        logger.warning(
            f"{type(inner).__name__} checkpoint saved without tokenizer — "
            f"air-gapped K8s deployments will fail at predict time"
        )


def _move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    """Move all optimizer state tensors to the target device.

    ``optimizer.load_state_dict`` does not respect ``map_location`` —
    momentum buffers and other per-parameter state tensors can end up on
    CPU even when the model parameters are on GPU.  This helper moves
    every tensor inside the optimizer state to the correct device.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint from disk. Restores model weights,
    optimizer state, and scheduler state.

    Args:
        path: File path to the checkpoint (.pth).
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map tensors to when loading.

    Returns:
        Dict with checkpoint metadata. Always includes 'epoch' and
        'optimizer_state_dict'. May also include 'scheduler_state_dict',
        'val_loss', 'best_val_loss', and 'patience_counter' if they
        were saved in the checkpoint.
    """
    # weights_only=True is safe: checkpoints now serialize configs as plain
    # dicts (not dataclass objects), so no arbitrary pickle deserialization.
    # Legacy checkpoints that stored the config dataclass object directly
    # will fail here — retrain the model to get a new-format checkpoint.
    metadata = torch.load(path, map_location=device, weights_only=True)

    inner = model.module if hasattr(model, "module") else model
    if "model_state_dict" in metadata:
        # The caller's model already has the right architecture, which may be tuned;
        # rebuilding one from config defaults could not hold these weights.
        inner.load_state_dict(metadata["model_state_dict"])
    else:
        artifact_dir = _checkpoint_artifact_dir(path)
        # Checkpoints from before per-checkpoint directories share the parent directory.
        ckpt_dir = artifact_dir if artifact_dir.is_dir() else Path(path).parent
        loaded_model = type(inner).load_from_checkpoint_dir(ckpt_dir, metadata, device)
        inner.load_state_dict(loaded_model.state_dict())

    if optimizer is not None and "optimizer_state_dict" in metadata:
        try:
            optimizer.load_state_dict(metadata["optimizer_state_dict"])
            _move_optimizer_to_device(optimizer, device)
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                f"optimizer state dict could not be loaded (likely param count "
                f"mismatch due to frozen/unfrozen backbone change): {exc}. "
                f"Starting optimizer from scratch — momentum will rebuild over "
                f"a few epochs."
            )

    if scheduler is not None and "scheduler_state_dict" in metadata:
        try:
            scheduler.load_state_dict(metadata["scheduler_state_dict"])
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                f"scheduler state dict could not be loaded: {exc}. "
                f"Scheduler will be rebuilt from scratch."
            )

    logger.info(f"checkpoint loaded: {path} (epoch={metadata.get('epoch', '?')})")

    return metadata


def list_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """List all checkpoint files in a directory, sorted by epoch.

    Returns:
        List of Path objects for each .pth checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []
    return sorted(
        ckpt_dir.glob("checkpoint_epoch_*.pth"),
        key=lambda ckpt: int(ckpt.stem.rsplit("_", 1)[-1]),
    )


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a directory.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None


def restore_best_checkpoint(
    model: torch.nn.Module, checkpoint_dir: str | Path | None, device: str = "cpu"
) -> bool:
    """Load the lowest-val-loss weights into *model* so callers save those, not the last epoch.

    Returns:
        True if best weights were restored. False if none exist (checkpointing disabled),
        in which case *model* keeps its final-epoch weights.
    """
    best_path = Path(checkpoint_dir) / BEST_CHECKPOINT_NAME if checkpoint_dir else None
    if best_path is None or not best_path.exists():
        logger.warning("no best checkpoint found, keeping final-epoch weights")
        return False
    load_checkpoint(best_path, model, device=device)
    logger.info(f"restored best weights from {best_path}")
    return True


# ──────────────────────────────────────────────
# Ray Train distributed training
# ──────────────────────────────────────────────


def new_ray_trainer(
    train_ds: ray.data.Dataset,
    val_ds: ray.data.Dataset,
    cfg: TrainerConfig,
    model_type: str,
    driver_config: type[DriverConfig] = DriverConfig,
    freeze_embeddings: bool = True,
) -> TorchTrainer:
    """Factory function to create a Ray Train TorchTrainer for distributed training.

    See Ray Train docs:
    https://docs.ray.io/en/2.55.1/train/api/doc/ray.train.torch.TorchTrainer.html

    Args:
        train_ds: Ray Dataset for training
        val_ds: Ray Dataset for validation
        cfg: TrainerConfig with training hyperparameters
        model_type: Model type ("rnn", "encoder", "decoder", or "modernbert")
        driver_config: DriverConfig class with file paths and model config

    Returns:
        Configured TorchTrainer ready to call .fit()
    """

    # Get model-specific optimization and scheduler params
    opt = _get_opt_params(model_type)
    sched = _get_sched_params(model_type)

    # Build train_loop_config with everything the worker needs to
    # create the model and run training (models cannot be passed across workers)
    train_loop_config = {
        "run_id": cfg.run_id,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": opt.lr,
        "betas": list(opt.betas),
        "weight_decay": opt.weight_decay,
        "warmup_ratio": getattr(sched, "warmup_ratio", 0.06),
        "scheduler_eta_min": sched.eta_min,
        "model_type": model_type,
        "dict_path": driver_config.files.dictionary_file_path,
        "embeddings_model_name": driver_config.embeddings.model_name,
        "embeddings_emb_length": driver_config.embeddings.emb_length,
        "freeze_embeddings": freeze_embeddings,
        "loss_type": cfg.loss_type,
        "label_smoothing": cfg.label_smoothing,
        "focal_gamma": cfg.focal_gamma,
        "class_weights": cfg.class_weights,
        "use_8bit_optimizer": cfg.use_8bit_optimizer,
        "use_amp": cfg.use_amp,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        # freeze_backbone_epochs: read from model-specific config for HF models
        "freeze_backbone_epochs": getattr(
            getattr(driver_config, model_type, None), "freeze_backbone_epochs", 0
        ),
        "train_dataset_count": train_ds.count(),
        "num_workers": cfg.ray_workers,
        "ray_update_every": cfg.ray_update_every,
        "early_stopping_patience": cfg.early_stopping_patience,
    }

    use_gpu = cfg.device in ("cuda", "mps")

    trainer = TorchTrainer(
        train_loop_per_worker=_train_func,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(
            num_workers=cfg.ray_workers,
            use_gpu=use_gpu,
        ),
        # Keep only the best checkpoint by val_loss; Ray also always keeps the latest.
        run_config=RunConfig(
            storage_path=str(Path(cfg.checkpoint_dir).resolve()) if cfg.checkpoint_dir else None,
            name=cfg.run_id,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
        datasets={"train": train_ds, "val": val_ds},
    )

    return trainer
