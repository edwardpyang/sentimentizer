"""Train stage: fit the model (single-node or distributed via Ray Train).

Heavy imports (torch, ray, config) are DEFERRED to function bodies to avoid
importing the ML stack at module level. Do NOT add module-level imports
of torch, ray, gensim, or sentimentizer.config here.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from workflows.helpers import _load_model
from workflows.lifecycle import State, _cuda_cleanup, _ensure_ray_initialized, logger

# Per-model-type metrics file base directory.  Each model writes to its own
# file so concurrent training processes never race on a shared JSON file.
_METRICS_DIR = Path("/tmp/sentimentizer_metrics")


def _metrics_path(model_type: str) -> Path:
    """Return the JSON file path for *model_type* metrics."""
    return _METRICS_DIR / f"{model_type}_metrics.json"


_ALL_MODEL_TYPES = ("rnn", "encoder", "decoder", "modernbert")


def _reset_stale_metrics(model_type: str, run_id: str = "") -> None:
    """Clear stale persisted metrics and reset Prometheus gauges.

    When a new training run starts for *model_type*, metrics from previous runs
    are stale.  We zero out all per-model JSON files (so the exporter can
    clear stale gauge values) but only reset Prometheus gauges for the *current*
    model type.  Other model types' gauges are left untouched — the exporter
    will handle them when it reads their ``_reset: true`` JSON files and skips
    them, leaving their gauges at whatever the last real training run set.
    """
    import time

    # --- 1. Persisted JSON files (zero all model types) ---
    try:
        _METRICS_DIR.mkdir(parents=True, exist_ok=True)
        for mt in _ALL_MODEL_TYPES:
            zeroed_metrics: dict[str, Any] = {
                "train_loss": 0.0,
                "val_loss": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "negative_precision": 0.0,
                "negative_recall": 0.0,
                "negative_f1": 0.0,
                "neutral_precision": 0.0,
                "neutral_recall": 0.0,
                "neutral_f1": 0.0,
                "positive_precision": 0.0,
                "positive_recall": 0.0,
                "positive_f1": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "cohen_kappa": 0.0,
                "mcc": 0.0,
                "neutral_to_positive_rate": 0.0,
                "neutral_to_negative_rate": 0.0,
                "pred_neutral_frac": 0.0,
                "neutral_auc_roc": None,
                "neutral_avg_precision": None,
                "epoch": 0,
                "lr": 0.0,
                "run_id": run_id if mt == model_type else "",
                "_reset": True,
                "_trace": {
                    "reset_by": model_type,
                    "reset_at": time.time(),
                },
            }
            path = _metrics_path(mt)
            path.write_text(json.dumps(zeroed_metrics, indent=2))

            # Also remove stale batch snapshot files so the exporter
            # doesn't serve intra-epoch data from a previous run.
            batch_path = _METRICS_DIR / f"{mt}_batch.json"
            with contextlib.suppress(OSError):
                batch_path.unlink(missing_ok=True)

        logger.info(
            "stale_metrics_cleared",
            model_type=model_type,
            cleared_types=list(_ALL_MODEL_TYPES),
        )
    except OSError as exc:
        logger.warning("failed_to_clear_stale_metrics_file: %s", exc)

    # --- 2. prometheus_client gauges (zero current model type only) ---
    try:
        from sentimentizer.exporter import (
            TRAINING_BATCH,
            TRAINING_EPOCH,
            TRAINING_GRAD_NORM,
            TRAINING_LR,
            TRAINING_THROUGHPUT,
            TRAINING_TRAIN_LOSS,
            TRAINING_TRAIN_LOSS_AVG,
            TRAINING_TRAIN_LOSS_EMA,
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

        lbl = {"model_type": model_type, "run_id": run_id}
        for gauge in (
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
            TRAINING_VAL_COHEN_KAPPA,
            TRAINING_VAL_MCC,
            TRAINING_VAL_MACRO_F1,
            TRAINING_VAL_WEIGHTED_F1,
            TRAINING_VAL_NEUTRAL_TO_POSITIVE_RATE,
            TRAINING_VAL_NEUTRAL_TO_NEGATIVE_RATE,
            TRAINING_VAL_PRED_NEUTRAL_FRAC,
            TRAINING_EPOCH,
            TRAINING_LR,
            TRAINING_TRAIN_LOSS_EMA,
            TRAINING_TRAIN_LOSS_AVG,
            TRAINING_BATCH,
            TRAINING_GRAD_NORM,
            TRAINING_THROUGHPUT,
        ):
            gauge.labels(**lbl).set(0)
        TRAINING_VAL_NEUTRAL_AUC_ROC.labels(**lbl).set(0)
        TRAINING_VAL_NEUTRAL_AVG_PRECISION.labels(**lbl).set(0)
    except ImportError:
        pass

    # --- 3. Ray _RAY_GAUGES cache (invalidate all) ---
    try:
        from sentimentizer.trainer import _RAY_GAUGES

        _RAY_GAUGES.clear()
    except ImportError:
        pass


def validate_train_options(
    state: State, *, distributed: bool, resume: bool, checkpoint_dir: str
) -> None:
    """Reject option combinations that would otherwise be silently ignored.

    Without a checkpoint directory, resuming would quietly restart from scratch. Ray
    workers build a fresh model in ``_train_func``, so neither a checkpoint nor existing
    weights would be used. Call before extract/tokenize so the failure is immediate.
    """
    if resume and not distributed and not checkpoint_dir:
        raise ValueError("Resuming requires --checkpoint-dir")
    if not distributed:
        return
    if resume:
        raise ValueError("Resuming is not supported with --distributed; train without --resume")
    if state.run_type == "update":
        raise ValueError(
            "--run-type update is not supported with --distributed; "
            "workers always build a new model"
        )


def run_train(
    state: State,
    *,
    distributed: bool,
    num_workers: int,
    save: bool,
    checkpoint_dir: str,
    checkpoint_every: int,
    resume: bool,
    balance_classes: bool,
    balance_seed: int,
    class_weights: list[float] | None = None,
    num_classes: int = 3,
    include_neutral: bool = True,
    loss_type: str = "cross_entropy",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    weight_smoothing: float = 0.5,
    neutral_oversample_ratio: float = 0.0,
    balance_strategy: str = "class_weights_only",
    freeze_embeddings: bool = True,
    push_to_hub: bool = False,
    pull_from_hub: bool = False,
    hf_repo: str | None = None,
    run_id: str = "",
    ray_update_every: int = -1,
) -> None:
    """Fit the model (single-node or distributed)."""
    from sentimentizer.device import resolve_device

    validate_train_options(
        state, distributed=distributed, resume=resume, checkpoint_dir=checkpoint_dir
    )

    # Validate that include_neutral and num_classes are consistent
    if include_neutral and num_classes != 3:
        raise ValueError(f"include_neutral=True requires num_classes=3, got {num_classes}")
    if not include_neutral and num_classes != 2:
        raise ValueError(f"include_neutral=False requires num_classes=2, got {num_classes}")

    device = resolve_device(state.device)

    from sentimentizer.config import DriverConfig, default_epochs, weights_path_for

    # Handle --pull-from-hub before training (preserves old behavior)
    if pull_from_hub:
        from sentimentizer.hf import download_weights

        weights_path = weights_path_for(state.model)
        repo_id = hf_repo if hf_repo is not None else DriverConfig.hf.repo_id
        # Use the per-model repo if hf_repo matches the default
        if hf_repo is None:
            from sentimentizer.config import HF_WEIGHTS_REPOS

            repo_id = HF_WEIGHTS_REPOS.get(state.model)

        result_path = download_weights(
            model_type=state.model,
            local_path=weights_path,
            repo_id=repo_id,
            dict_path=DriverConfig.files.dictionary_file_path,
        )
        if result_path:
            logger.info(f"Pulled {state.model} weights from HF Hub. Forcing run type to 'update'.")
        else:
            logger.error("Failed to pull weights from HF Hub. Proceeding with original run type.")

    epochs = default_epochs(state.model)

    if distributed:
        from workflows.lifecycle import is_ray_available

        if not is_ray_available():
            raise ImportError(
                "Ray is required for distributed training. Install with: pip install '.[ray]'"
            )
        _run_fit_distributed(
            state=state,
            device=device,
            epochs=epochs,
            num_workers=num_workers,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            balance_classes=balance_classes,
            balance_seed=balance_seed,
            class_weights=class_weights,
            num_classes=num_classes,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            weight_smoothing=weight_smoothing,
            neutral_oversample_ratio=neutral_oversample_ratio,
            balance_strategy=balance_strategy,
            freeze_embeddings=freeze_embeddings,
            save=save,
            push_to_hub=push_to_hub,
            hf_repo=hf_repo,
            run_id=run_id,
            ray_update_every=ray_update_every,
        )
    else:
        model = _load_model(state, device, freeze_embeddings=freeze_embeddings)
        _run_fit_single(
            state=state,
            device=device,
            model=model,
            epochs=epochs,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            resume=resume,
            balance_classes=balance_classes,
            balance_seed=balance_seed,
            class_weights=class_weights,
            num_classes=num_classes,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            weight_smoothing=weight_smoothing,
            neutral_oversample_ratio=neutral_oversample_ratio,
            balance_strategy=balance_strategy,
            save=save,
            push_to_hub=push_to_hub,
            hf_repo=hf_repo,
            run_id=run_id,
            ray_update_every=ray_update_every,
        )


def _run_fit_single(
    state: State,
    device: str,
    model: Any,
    epochs: int,
    checkpoint_dir: str,
    checkpoint_every: int,
    resume: bool,
    balance_classes: bool,
    balance_seed: int,
    class_weights: list[float] | None,
    num_classes: int,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
    weight_smoothing: float,
    neutral_oversample_ratio: float,
    balance_strategy: str,
    save: bool,
    push_to_hub: bool,
    hf_repo: str | None,
    run_id: str = "",
    ray_update_every: int = -1,
) -> None:
    """Single-node training using the existing Trainer class."""
    import torch

    from sentimentizer.config import DriverConfig, default_dataloader_workers, weights_path_for
    from sentimentizer.loader import compute_class_weights, load_train_val_corpus_datasets
    from sentimentizer.trainer import new_trainer

    if state.model == "modernbert":
        from sentimentizer.hf_dataset import load_train_val_hf_datasets

        train_dataset, val_dataset = load_train_val_hf_datasets(
            data_path=DriverConfig.files.hf_processed_reviews_file_path,
            balance_classes=balance_classes,
            random_state=balance_seed,
        )
    else:
        train_dataset, val_dataset = load_train_val_corpus_datasets(
            data_path=DriverConfig.files.processed_reviews_file_path,
            balance_classes=balance_classes,
            random_state=balance_seed,
        )

    # Auto-compute class_weights from training data if not explicitly set
    if class_weights is None:
        import pandas as pd

        cw_path = (
            DriverConfig.files.hf_processed_reviews_file_path
            if state.model == "modernbert"
            else DriverConfig.files.processed_reviews_file_path
        )
        train_df = pd.read_parquet(cw_path)
        class_weights_tensor = compute_class_weights(
            train_df, num_classes=num_classes, smoothing=weight_smoothing
        )
        del train_df
    else:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    cfg = DriverConfig.trainer(
        run_id=run_id,
        epochs=epochs,
        device=device,
        dataloader_workers=default_dataloader_workers(device),
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        class_weights=class_weights_tensor.tolist(),
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        ray_update_every=ray_update_every,
    )

    # Clear stale metrics using the auto-generated run_id from TrainerConfig
    # (if run_id was empty, __post_init__ generated one). This ensures the
    # current run_id appears in the Grafana dashboard dropdown.
    _reset_stale_metrics(state.model, cfg.run_id)

    trainer = new_trainer(
        model=model,
        cfg=cfg,
        model_type=state.model,
    )

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    if resume:
        from sentimentizer.trainer import latest_checkpoint, load_checkpoint

        ckpt_path = latest_checkpoint(checkpoint_dir)
        if ckpt_path is None:
            logger.info("no checkpoint found, starting from scratch")
        else:
            checkpoint = load_checkpoint(
                ckpt_path, model, trainer.optimizer, trainer.scheduler, device=device
            )
            ckpt_epoch = checkpoint.get("epoch", 0)
            start_epoch = ckpt_epoch + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            if best_val_loss == float("inf") and "val_loss" in checkpoint:
                best_val_loss = checkpoint["val_loss"]
            patience_counter = checkpoint.get("patience_counter", 0)
            logger.info(
                f"resumed from checkpoint: {ckpt_path}, "
                f"epoch={ckpt_epoch}, start_epoch={start_epoch}, "
                f"best_val_loss={best_val_loss:.4f}, patience={patience_counter}"
            )

    try:
        trainer.fit(
            model,
            train_data=train_dataset,
            val_data=val_dataset,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
        )
    finally:
        _cuda_cleanup()

    if save:
        from sentimentizer.trainer import restore_best_checkpoint

        restore_best_checkpoint(model, checkpoint_dir, device=device)
        weights_path = weights_path_for(state.model)
        inner = model.module if hasattr(model, "module") else model
        if hasattr(inner, "save_to_checkpoint_dir"):
            metadata = inner.save_to_checkpoint_dir(Path(weights_path).parent)
            torch.save(metadata, weights_path)
        else:
            torch.save(model.state_dict(), weights_path)
        logger.info(f"model weights saved to: {weights_path}")

        if push_to_hub:
            from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig
            from sentimentizer.hf import push_model_to_hub

            # Use per-model repo if hf_repo is not explicitly provided
            repo_id = hf_repo if hf_repo is not None else HF_WEIGHTS_REPOS.get(state.model)

            push_model_to_hub(
                local_path=weights_path,
                model_type=state.model,
                repo_id=repo_id,
                dict_path=(
                    DriverConfig.files.dictionary_file_path if state.model != "modernbert" else None
                ),
                backbone_path=(
                    str(Path(weights_path).parent / "backbone")
                    if hasattr(inner, "save_to_checkpoint_dir")
                    else None
                ),
            )


def _run_fit_distributed(
    state: State,
    device: str,
    epochs: int,
    num_workers: int,
    checkpoint_dir: str,
    checkpoint_every: int,
    balance_classes: bool,
    balance_seed: int,
    class_weights: list[float] | None,
    num_classes: int,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
    weight_smoothing: float,
    neutral_oversample_ratio: float,
    balance_strategy: str,
    freeze_embeddings: bool,
    save: bool,
    push_to_hub: bool,
    hf_repo: str | None,
    run_id: str = "",
    ray_update_every: int = -1,
) -> None:
    """Distributed training using Ray Train TorchTrainer.

    The model is NOT loaded in the driver process -- Ray workers create their
    own model from scratch in ``_train_func``.  Loading the model here would
    waste GPU memory (for ``run_type="update"``) since the driver-side model
    is never used for training.
    """
    import torch

    from sentimentizer.config import DriverConfig, weights_path_for
    from sentimentizer.loader import compute_class_weights, load_train_val_ray_datasets
    from sentimentizer.trainer import new_ray_trainer

    _ensure_ray_initialized()

    # Auto-compute class_weights from full dataset if not explicitly set
    if class_weights is None:
        import pandas as pd

        cw_path = (
            DriverConfig.files.hf_processed_reviews_file_path
            if state.model == "modernbert"
            else DriverConfig.files.processed_reviews_file_path
        )
        full_df = pd.read_parquet(cw_path)
        class_weights_tensor = compute_class_weights(
            full_df, num_classes=num_classes, smoothing=weight_smoothing
        )
        del full_df
    else:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    if state.model == "modernbert":
        train_ds, val_ds = load_train_val_ray_datasets(
            DriverConfig.files.hf_processed_reviews_file_path,
            balance_classes=balance_classes,
            random_state=balance_seed,
        )
    else:
        train_ds, val_ds = load_train_val_ray_datasets(
            DriverConfig.files.processed_reviews_file_path,
            balance_classes=balance_classes,
            random_state=balance_seed,
        )

    cfg = DriverConfig.trainer(
        run_id=run_id,
        epochs=epochs,
        device=device,
        ray_workers=num_workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        class_weights=class_weights_tensor.tolist(),
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        ray_update_every=ray_update_every,
    )

    # Clear stale metrics using the auto-generated run_id from TrainerConfig
    # (if run_id was empty, __post_init__ generated one). This ensures the
    # current run_id appears in the Grafana dashboard dropdown.
    _reset_stale_metrics(state.model, cfg.run_id)

    ray_trainer = new_ray_trainer(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg,
        model_type=state.model,
        freeze_embeddings=freeze_embeddings,
    )

    try:
        result = ray_trainer.fit()
    finally:
        _cuda_cleanup()

    # Publish final metrics from distributed training to the driver-level
    # Prometheus gauges so they are visible on the /metrics endpoint.
    _publish_distributed_metrics(result, state.model, cfg.run_id)

    logger.info(  # type: ignore[call-arg]
        "distributed training completed",
        best_checkpoint=result.checkpoint,
        metrics=result.metrics,
    )

    if save:
        import os
        import shutil

        import ray.cloudpickle as pickle

        weights_path = weights_path_for(state.model)
        best_checkpoint = (
            result.get_best_checkpoint(metric="val_loss", mode="min") or result.checkpoint
        )
        with best_checkpoint.as_directory() as checkpoint_dir_path:
            with open(os.path.join(checkpoint_dir_path, "data.pkl"), "rb") as fp:
                checkpoint_data = pickle.load(fp)

            # Check if there is a backbone subdirectory (HF model format)
            backbone_src = Path(checkpoint_dir_path) / "backbone"
            if backbone_src.exists():
                backbone_dst = Path(weights_path).parent / "backbone"
                if backbone_dst.exists():
                    shutil.rmtree(backbone_dst)
                shutil.copytree(backbone_src, backbone_dst)
            elif state.model == "modernbert":
                # HF models require a backbone directory — its absence means
                # the Ray checkpoint is corrupt or incomplete. Raise immediately
                # rather than saving a weights file that can't be loaded.
                raise FileNotFoundError(
                    f"Expected backbone/ directory inside Ray checkpoint at "
                    f"{checkpoint_dir_path!r} for model_type={state.model!r}, "
                    f"but it was not found. The checkpoint is likely corrupt. "
                    f"Re-run training to produce a valid checkpoint."
                )

            torch.save(
                checkpoint_data,
                weights_path,
            )
        logger.info(f"model weights saved to: {weights_path}")

        if push_to_hub:
            from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig
            from sentimentizer.hf import push_model_to_hub

            repo_id = hf_repo if hf_repo is not None else HF_WEIGHTS_REPOS.get(state.model)

            push_model_to_hub(
                local_path=weights_path,
                model_type=state.model,
                repo_id=repo_id,
                dict_path=DriverConfig.files.dictionary_file_path,
            )


def _publish_distributed_metrics(result: Any, model_type: str, run_id: str = "") -> None:
    """Persist final distributed training metrics to the per-model JSON file.

    During distributed training, ``_train_func`` runs inside Ray workers that
    write per-epoch metrics to the JSON file via ``_write_epoch_metrics_to_file``
    and push live gauges from the worker process.  After ``ray_trainer.fit()``
    returns, the driver has the final metrics in ``result.metrics``.

    The driver process does not run a Prometheus HTTP server, so setting
    ``prometheus_client`` gauges here would have no effect — only the
    standalone exporter (port 8081) serves those gauges, and it reads from
    the JSON file.  We therefore persist the final metrics to JSON and log
    a summary, but do not push to driver-level gauges.
    """
    metrics = result.metrics
    logger.info(
        "distributed_metrics_published",
        model_type=model_type,
        accuracy=round(metrics.get("accuracy", 0), 4),
        val_loss=round(metrics.get("val_loss", 0), 4),
        train_loss=round(metrics.get("train_loss", 0), 4),
    )

    _persist_metrics_to_file(result.metrics, model_type, run_id)


def _persist_metrics_to_file(metrics: dict, model_type: str, run_id: str = "") -> None:
    """Write final training metrics to a per-model JSON file for the standalone exporter.

    Each model type writes to its own file
    (``/tmp/sentimentizer_metrics/{model_type}_metrics.json``) so concurrent
    training processes never race on a shared JSON file.  The standalone
    exporter discovers all files and zeroes out Prometheus gauges for
    any model type whose file is missing or stale.
    """
    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = _metrics_path(model_type)

    def _float(key: str, fallback: str | None = None) -> float:
        val = metrics.get(key)
        if val is None and fallback:
            val = metrics.get(fallback)
        return float(val) if val is not None else 0.0

    def _opt_float(key: str, fallback: str | None = None) -> float | None:
        val = metrics.get(key)
        if val is None and fallback:
            val = metrics.get(fallback)
        return float(val) if val is not None else None

    data: dict[str, Any] = {
        "train_loss": float(metrics.get("train_loss", 0)),
        "val_loss": float(metrics.get("val_loss", 0)),
        "accuracy": float(metrics.get("accuracy", 0)),
        "balanced_accuracy": _float("balanced_accuracy"),
        "negative_precision": _float("negative_precision", "val_negative_precision"),
        "negative_recall": _float("negative_recall", "val_negative_recall"),
        "negative_f1": _float("negative_f1", "val_negative_f1"),
        "neutral_precision": _float("neutral_precision", "val_neutral_precision"),
        "neutral_recall": _float("neutral_recall", "val_neutral_recall"),
        "neutral_f1": _float("neutral_f1", "val_neutral_f1"),
        "positive_precision": _float("positive_precision", "val_positive_precision"),
        "positive_recall": _float("positive_recall", "val_positive_recall"),
        "positive_f1": _float("positive_f1", "val_positive_f1"),
        "cohen_kappa": float(metrics.get("cohen_kappa", 0)),
        "mcc": float(metrics.get("mcc", 0)),
        "macro_f1": float(metrics.get("macro_f1", 0)),
        "weighted_f1": _float("weighted_f1"),
        "neutral_to_positive_rate": _float("neutral_to_positive_rate"),
        "neutral_to_negative_rate": _float("neutral_to_negative_rate"),
        "pred_neutral_frac": _float("pred_neutral_frac"),
        "neutral_auc_roc": _opt_float("neutral_auc_roc", "val_neutral_auc_roc"),
        "neutral_avg_precision": _opt_float("neutral_avg_precision", "val_neutral_avg_precision"),
        "epoch": int(metrics.get("epoch", 0)),
        "lr": float(metrics.get("lr", 0.0)),
        "run_id": run_id,
        "_written_by": model_type,
        "_written_at": __import__("time").time(),
    }

    path.write_text(json.dumps(data, indent=2))
    logger.info("training_metrics_persisted", path=str(path), model_type=model_type)
