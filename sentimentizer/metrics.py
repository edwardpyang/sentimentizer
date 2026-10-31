"""Classification metrics for sentiment model evaluation.

Provides comprehensive metrics for 3-class sentiment classification
(negative/neutral/positive), including per-class precision/recall/F1,
balanced accuracy, macro/weighted F1, Cohen's kappa, MCC, per-class
AUC-ROC and average precision (one-vs-rest), confusion matrix,
neutral detection rate, and prediction distribution.

Uses torchmetrics for metric computation, which provides GPU-native
tensor operations and batch accumulation. Edge cases (NaN probabilities,
single-class targets, empty arrays) are handled with explicit guards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
)

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL, LABEL_NAMES, NUM_CLASSES

logger = new_logger(DEFAULT_LOG_LEVEL)


@dataclass
class ClassificationMetrics:
    """Comprehensive classification metrics for 3-class sentiment models.

    Attributes:
        accuracy: Overall accuracy (correct / total).
        balanced_accuracy: Mean of per-class recalls.
        negative_precision: Precision for the negative class.
        negative_recall: Recall for the negative class.
        negative_f1: F1 score for the negative class.
        neutral_precision: Precision for the neutral class.
        neutral_recall: Recall for the neutral class (= neutral detection rate).
        neutral_f1: F1 score for the neutral class.
        positive_precision: Precision for the positive class.
        positive_recall: Recall for the positive class.
        positive_f1: F1 score for the positive class.
        macro_f1: Macro-averaged F1 (mean of per-class F1, weights classes equally).
        weighted_f1: Weighted-averaged F1 (weighted by class frequency).
        confusion_matrix: 3×3 confusion matrix as list of lists.
        neutral_to_positive_rate: Fraction of true neutral misclassified as positive.
        neutral_to_negative_rate: Fraction of true neutral misclassified as negative.
        pred_negative_frac: Fraction of predictions that are negative.
        pred_neutral_frac: Fraction of predictions that are neutral.
        pred_positive_frac: Fraction of predictions that are positive.
        cohen_kappa: Cohen's kappa coefficient.
        mcc: Matthews correlation coefficient.
        negative_auc_roc: AUC-ROC for negative class (one-vs-rest).
        neutral_auc_roc: AUC-ROC for neutral class (one-vs-rest).
        positive_auc_roc: AUC-ROC for positive class (one-vs-rest).
        negative_avg_precision: Average precision for negative class (one-vs-rest).
        neutral_avg_precision: Average precision for neutral class (one-vs-rest).
        positive_avg_precision: Average precision for positive class (one-vs-rest).
        total: Total number of samples.
    """

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    negative_precision: float = 0.0
    negative_recall: float = 0.0
    negative_f1: float = 0.0
    neutral_precision: float = 0.0
    neutral_recall: float = 0.0
    neutral_f1: float = 0.0
    positive_precision: float = 0.0
    positive_recall: float = 0.0
    positive_f1: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    confusion_matrix: list[list[int]] | None = None
    neutral_to_positive_rate: float = 0.0
    neutral_to_negative_rate: float = 0.0
    pred_negative_frac: float = 0.0
    pred_neutral_frac: float = 0.0
    pred_positive_frac: float = 0.0
    cohen_kappa: float = 0.0
    mcc: float = 0.0
    negative_auc_roc: float | None = None
    neutral_auc_roc: float | None = None
    positive_auc_roc: float | None = None
    negative_avg_precision: float | None = None
    neutral_avg_precision: float | None = None
    positive_avg_precision: float | None = None
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary suitable for JSON serialization."""
        return {
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "negative_precision": round(self.negative_precision, 4),
            "negative_recall": round(self.negative_recall, 4),
            "negative_f1": round(self.negative_f1, 4),
            "neutral_precision": round(self.neutral_precision, 4),
            "neutral_recall": round(self.neutral_recall, 4),
            "neutral_f1": round(self.neutral_f1, 4),
            "positive_precision": round(self.positive_precision, 4),
            "positive_recall": round(self.positive_recall, 4),
            "positive_f1": round(self.positive_f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "confusion_matrix": self.confusion_matrix,
            "neutral_to_positive_rate": round(self.neutral_to_positive_rate, 4),
            "neutral_to_negative_rate": round(self.neutral_to_negative_rate, 4),
            "pred_negative_frac": round(self.pred_negative_frac, 4),
            "pred_neutral_frac": round(self.pred_neutral_frac, 4),
            "pred_positive_frac": round(self.pred_positive_frac, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "mcc": round(self.mcc, 4),
            "negative_auc_roc": (
                round(self.negative_auc_roc, 4) if self.negative_auc_roc is not None else None
            ),
            "neutral_auc_roc": (
                round(self.neutral_auc_roc, 4) if self.neutral_auc_roc is not None else None
            ),
            "positive_auc_roc": (
                round(self.positive_auc_roc, 4) if self.positive_auc_roc is not None else None
            ),
            "negative_avg_precision": (
                round(self.negative_avg_precision, 4)
                if self.negative_avg_precision is not None
                else None
            ),
            "neutral_avg_precision": (
                round(self.neutral_avg_precision, 4)
                if self.neutral_avg_precision is not None
                else None
            ),
            "positive_avg_precision": (
                round(self.positive_avg_precision, 4)
                if self.positive_avg_precision is not None
                else None
            ),
            "total": self.total,
        }


def _to_long_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert predictions or targets to a 1-D long tensor on CPU."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().long().flatten()
    return torch.as_tensor(arr, dtype=torch.long).flatten()


def _to_float_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert probabilities to a float32 tensor on CPU."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().float()
    return torch.as_tensor(arr, dtype=torch.float32)


def _replace_nan_probs(probabilities: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Replace NaN values in probabilities with 1/num_classes.

    For 2D (N, C) matrices: zero NaN positions and re-normalize rows.
    For 1D vectors: replace NaN with 1/num_classes.
    """
    nan_mask = torch.isnan(probabilities)
    nan_count = int(nan_mask.sum().item())
    if nan_count > 0:
        logger.warning(
            "nan_in_probabilities",
            message=f"Found {nan_count} NaN values in probabilities, replacing with uniform",
        )
        if probabilities.dim() == 2:
            # For 2D: zero NaN positions and re-normalize rows to sum to 1.0
            probabilities = torch.where(
                nan_mask, torch.tensor(0.0, dtype=probabilities.dtype), probabilities
            )
            row_sums = probabilities.sum(dim=1, keepdim=True).clamp(min=1e-8)
            probabilities = probabilities / row_sums
        else:
            probabilities = torch.where(
                nan_mask,
                torch.tensor(1.0 / NUM_CLASSES, dtype=probabilities.dtype),
                probabilities,
            )
    return probabilities, nan_count


def _safe_item(value: torch.Tensor | float) -> float:
    """Extract a Python float from a scalar tensor, converting NaN to 0.0.

    For multi-element tensors, use ``_safe_item_list`` instead.
    """
    if isinstance(value, torch.Tensor):
        value = value.item()
    if math.isnan(value):
        return 0.0
    return value


def _safe_item_list(value: torch.Tensor) -> list[float]:
    """Convert a tensor to a list of Python floats, converting NaN to 0.0."""
    if value.numel() == 0:
        return []
    return [0.0 if math.isnan(v) else v for v in value.tolist()]


def compute_classification_metrics(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    probabilities: np.ndarray | torch.Tensor | None = None,
    num_classes: int = NUM_CLASSES,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics from predictions and targets.

    Args:
        predictions: Class index predictions (0, 1, 2), shape (N,).
        targets: Ground truth class indices (0, 1, 2), shape (N,).
        probabilities: Probability matrix of shape (N, num_classes).
            Required for AUC-ROC and average precision. If None, those
            metrics will be None.
        num_classes: Number of classes (default 3).

    Returns:
        ClassificationMetrics with all computed metrics.
    """
    # Convert to tensors
    preds_t = _to_long_tensor(predictions)
    targets_t = _to_long_tensor(targets)

    total = targets_t.shape[0]
    if total == 0:
        return ClassificationMetrics(total=0)

    # Compute confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets_t, preds_t, strict=False):
        cm[t.item(), p.item()] += 1

    # Overall accuracy
    accuracy = _safe_item(preds_t.eq(targets_t).float().mean())

    # Per-class metrics via torchmetrics Multiclass*
    per_class_precision = _safe_item_list(
        MulticlassPrecision(num_classes=num_classes, average=None)(preds_t, targets_t)
    )
    per_class_recall = _safe_item_list(
        MulticlassRecall(num_classes=num_classes, average=None)(preds_t, targets_t)
    )
    per_class_f1 = _safe_item_list(
        MulticlassF1Score(num_classes=num_classes, average=None)(preds_t, targets_t)
    )

    # Pad if needed (edge case: not all classes present)
    while len(per_class_precision) < num_classes:
        per_class_precision.append(0.0)
    while len(per_class_recall) < num_classes:
        per_class_recall.append(0.0)
    while len(per_class_f1) < num_classes:
        per_class_f1.append(0.0)

    # Balanced accuracy = mean of per-class recalls
    balanced_accuracy = float(np.mean(per_class_recall))

    # Aggregate F1 scores
    macro_f1 = _safe_item(
        MulticlassF1Score(num_classes=num_classes, average="macro")(preds_t, targets_t)
    )
    weighted_f1 = _safe_item(
        MulticlassF1Score(num_classes=num_classes, average="weighted")(preds_t, targets_t)
    )

    # Cohen's kappa and MCC
    cohen_kappa = _safe_item(MulticlassCohenKappa(num_classes=num_classes)(preds_t, targets_t))
    mcc = _safe_item(MulticlassMatthewsCorrCoef(num_classes=num_classes)(preds_t, targets_t))

    # Confusion matrix as list of lists
    cm_list = cm.tolist()

    # Inter-class confusion rates from confusion matrix
    neutral_to_positive_rate = 0.0
    neutral_to_negative_rate = 0.0
    neutral_total = cm[1].sum().item()
    if neutral_total > 0:
        neutral_to_positive_rate = cm[1, 2].item() / neutral_total
        neutral_to_negative_rate = cm[1, 0].item() / neutral_total

    # Prediction distribution
    pred_counts = np.bincount(preds_t.numpy(), minlength=num_classes)
    pred_negative_frac = float(pred_counts[0] / max(total, 1))
    pred_neutral_frac = float(pred_counts[1] / max(total, 1)) if num_classes > 2 else 0.0
    pred_positive_frac = float(pred_counts[2] / max(total, 1)) if num_classes > 2 else 0.0

    # AUC-ROC and Average Precision per class (one-vs-rest)
    negative_auc_roc: float | None = None
    neutral_auc_roc: float | None = None
    positive_auc_roc: float | None = None
    negative_avg_precision: float | None = None
    neutral_avg_precision: float | None = None
    positive_avg_precision: float | None = None

    if probabilities is not None:
        probs_t = _to_float_tensor(probabilities)
        if probs_t.dim() == 1:
            # Binary compatibility: reshape to (N, 1) won't work for multiclass AUC.
            # Skip AUC computation for 1D probability arrays.
            pass
        else:
            # Handle 2D (N, num_classes) probabilities
            probs_t, _nan_count = _replace_nan_probs(probs_t)

            # Check if all classes are present
            unique_targets = torch.unique(targets_t)
            if len(unique_targets) < 2:
                logger.warning("auc_roc_few_classes", message="AUC-ROC requires at least 2 classes")
                negative_auc_roc = 0.0
                neutral_auc_roc = 0.0
                positive_auc_roc = 0.0
                negative_avg_precision = 0.0
                neutral_avg_precision = 0.0
                positive_avg_precision = 0.0
            else:
                try:
                    auc_roc_per_class = _safe_item_list(
                        MulticlassAUROC(num_classes=num_classes, average=None)(probs_t, targets_t)
                    )
                    avg_prec_per_class = _safe_item_list(
                        MulticlassAveragePrecision(num_classes=num_classes, average=None)(
                            probs_t, targets_t
                        )
                    )
                    auc_list = list(auc_roc_per_class)
                    ap_list = list(avg_prec_per_class)
                    while len(auc_list) < num_classes:
                        auc_list.append(0.0)
                    while len(ap_list) < num_classes:
                        ap_list.append(0.0)
                    negative_auc_roc = float(auc_list[0])
                    neutral_auc_roc = float(auc_list[1]) if num_classes > 2 else 0.0
                    positive_auc_roc = float(auc_list[2]) if num_classes > 2 else float(auc_list[1])
                    negative_avg_precision = float(ap_list[0])
                    neutral_avg_precision = float(ap_list[1]) if num_classes > 2 else 0.0
                    positive_avg_precision = (
                        float(ap_list[2]) if num_classes > 2 else float(ap_list[1])
                    )
                except (ValueError, RuntimeError):
                    logger.warning("auc_roc_failed", message="AUC-ROC computation failed")

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        negative_precision=float(per_class_precision[0]),
        negative_recall=float(per_class_recall[0]),
        negative_f1=float(per_class_f1[0]),
        neutral_precision=float(per_class_precision[1]) if num_classes > 2 else 0.0,
        neutral_recall=float(per_class_recall[1]) if num_classes > 2 else 0.0,
        neutral_f1=float(per_class_f1[1]) if num_classes > 2 else 0.0,
        positive_precision=(
            float(per_class_precision[2]) if num_classes > 2 else float(per_class_precision[1])
        ),
        positive_recall=(
            float(per_class_recall[2]) if num_classes > 2 else float(per_class_recall[1])
        ),
        positive_f1=(float(per_class_f1[2]) if num_classes > 2 else float(per_class_f1[1])),
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix=cm_list,
        neutral_to_positive_rate=neutral_to_positive_rate,
        neutral_to_negative_rate=neutral_to_negative_rate,
        pred_negative_frac=pred_negative_frac,
        pred_neutral_frac=pred_neutral_frac,
        pred_positive_frac=pred_positive_frac,
        cohen_kappa=cohen_kappa,
        mcc=mcc,
        negative_auc_roc=negative_auc_roc,
        neutral_auc_roc=neutral_auc_roc,
        positive_auc_roc=positive_auc_roc,
        negative_avg_precision=negative_avg_precision,
        neutral_avg_precision=neutral_avg_precision,
        positive_avg_precision=positive_avg_precision,
        total=total,
    )


def compute_metrics_from_examples(
    results: list[dict[str, Any]],
    num_classes: int = NUM_CLASSES,
) -> ClassificationMetrics:
    """Compute classification metrics from a list of validation example results.

    Each result dict should have 'expected' ('negative', 'neutral', or 'positive')
    and 'scores' (dict mapping label names to probabilities), as produced by
    the 3-class predict_text() method.

    Args:
        results: List of dicts with 'expected' and 'scores' keys.
        num_classes: Number of classes (default 3).

    Returns:
        ClassificationMetrics with all computed metrics.
    """
    total = len(results)
    if total == 0:
        return ClassificationMetrics(total=0)

    label_to_idx = {name: idx for idx, name in enumerate(LABEL_NAMES)}

    # Build tensors from results
    targets_list: list[int] = []
    preds_list: list[int] = []
    probs_list: list[list[float]] = []

    for r in results:
        expected = r.get("expected", "neutral")
        scores = r.get("scores", {})

        target = label_to_idx.get(expected, 1)

        # Get predicted class from scores dict
        if isinstance(scores, dict) and scores:
            predicted = max(scores, key=scores.get)  # type: ignore[arg-type]
            pred = label_to_idx.get(predicted, 1)
            prob_row = [scores.get(name, 1.0 / num_classes) for name in LABEL_NAMES]
        else:
            # Fallback: use 'correct' field if no scores
            correct = r.get("correct", False)
            pred = target if correct else ((target + 1) % num_classes)
            prob_row = [1.0 / num_classes] * num_classes

        targets_list.append(target)
        preds_list.append(pred)
        probs_list.append(prob_row)

    targets_t = torch.tensor(targets_list, dtype=torch.long)
    preds_t = torch.tensor(preds_list, dtype=torch.long)
    probabilities = np.array(probs_list, dtype=np.float32)

    return compute_classification_metrics(
        predictions=preds_t,
        targets=targets_t,
        probabilities=probabilities,
        num_classes=num_classes,
    )
