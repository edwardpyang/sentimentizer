"""Evaluate task-specific router labels and historical fallback thresholds."""

from __future__ import annotations

import logging
from collections import defaultdict
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from sentimentizer.router.config import FOOD_TASK_SPEC
from sentimentizer.router.dataset import assert_disjoint_keys, record_keys
from sentimentizer.router.provenance import dataset_records

if TYPE_CHECKING:
    from sentimentizer.router.model import RouterModel
    from sentimentizer.router.schemas import RouterClassification

logger = logging.getLogger(__name__)

CALIBRATION_BINS = 10
COVERAGE_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def coverage_metrics(scores: list[float], correct: list[bool], threshold: float | None) -> dict:
    """Describe accepted accuracy and coverage, including abstain-all policies."""
    accepted = (
        np.asarray(scores) >= threshold
        if threshold is not None
        else np.zeros(len(scores), dtype=bool)
    )
    return {
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "accepted": int(accepted.sum()),
        "accuracy": float(np.asarray(correct)[accepted].mean()) if accepted.any() else None,
    }


def prediction_metrics(
    targets: list[str], predicted: list[str], scores: list[float], labels: list[str]
) -> dict:
    """Report classification quality, confidence reliability and accuracy/coverage."""
    if not targets or not (len(targets) == len(predicted) == len(scores)):
        raise ValueError("Metrics require nonempty aligned predictions and targets")
    confidence = np.asarray(scores)
    correct = np.asarray(targets) == np.asarray(predicted)
    calibration_error = 0.0
    bins = []
    for index in range(CALIBRATION_BINS):
        lower, upper = index / CALIBRATION_BINS, (index + 1) / CALIBRATION_BINS
        mask = (confidence >= lower) & (
            (confidence < upper) if index < CALIBRATION_BINS - 1 else (confidence <= upper)
        )
        count = int(mask.sum())
        if count:
            accuracy = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            calibration_error += count / len(targets) * abs(accuracy - mean_confidence)
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "accuracy": accuracy,
                    "mean_confidence": mean_confidence,
                }
            )
    coverage = []
    for threshold in COVERAGE_THRESHOLDS:
        coverage.append(coverage_metrics(scores, correct.tolist(), threshold))
    return {
        "classification_report": classification_report(
            targets,
            predicted,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=labels).tolist(),
        "confusion_labels": labels,
        "accuracy": float(correct.mean()),
        "expected_calibration_error": float(calibration_error),
        "reliability_bins": bins,
        "accuracy_coverage": coverage,
    }


def _label_names(model: RouterModel) -> dict[int, str]:
    """Read the saved vocabulary, including legacy string-class heads."""
    labels = list(model.task.labels) if model.task is not None else model.labels
    return dict(enumerate(labels or model.class_labels()))


def _target_names(values: list, label_names: dict[int, str]) -> list[str]:
    """Resolve evaluation targets without silently coercing invalid IDs."""
    names = []
    for value in values:
        if isinstance(value, str) and value in label_names.values():
            names.append(value)
        elif isinstance(value, Integral) and not isinstance(value, bool) and value in label_names:
            names.append(label_names[int(value)])
        else:
            raise ValueError("Evaluation label is outside the artifact vocabulary")
    return names


def _fallback_label(model: RouterModel) -> str | None:
    """Keep the historical food fallback only for identifiable food artifacts."""
    if model.task is not None:
        return model.task.fallback_label
    return FOOD_TASK_SPEC.fallback_label if tuple(model.labels) == FOOD_TASK_SPEC.labels else None


def compute_similarity_matrix(
    model: RouterModel, texts_by_label: dict[int, list[str]]
) -> np.ndarray:
    """Compute centroid cosine similarities for the observed classes.

    The diagonal is centroid self-similarity, not an intra-class quality metric.
    """
    label_names = _label_names(model)
    labels = sorted(texts_by_label)
    centroids = [np.mean(model.model_encode(texts_by_label[label]), axis=0) for label in labels]
    matrix = np.zeros((len(labels), len(labels)))
    for i, first in enumerate(centroids):
        for j, second in enumerate(centroids):
            norm = np.linalg.norm(first) * np.linalg.norm(second)
            matrix[i, j] = np.dot(first, second) / norm if norm > 0 else 0.0
            if i < j:
                logger.info(
                    "Router centroid similarity",
                    extra={
                        "first_label": label_names[labels[i]],
                        "second_label": label_names[labels[j]],
                        "similarity": float(matrix[i, j]),
                    },
                )
    return matrix


def _threshold_recommendation(
    predictions: list[RouterClassification], targets: list[str], fallback_label: str | None
) -> float | None:
    """Compute the legacy heuristic, which is not a calibrated acceptance policy."""
    if fallback_label is None:
        return None
    correct = []
    fallback_errors = []
    for prediction, target in zip(predictions, targets, strict=True):
        if prediction.label == target:
            correct.append(prediction.score)
        elif prediction.label == fallback_label:
            fallback_errors.append(prediction.score)
    if not correct:
        logger.warning("No correct predictions for historical threshold recommendation")
        return None
    threshold = min(correct)
    if fallback_errors:
        threshold = (threshold + max(fallback_errors)) / 2
    return float(threshold)


def calibrate_threshold(
    model: RouterModel,
    eval_dataset: Any,
    label_names: dict[int, str] | None = None,
    *,
    fallback_label: str | None = None,
) -> float | None:
    """Return a historical fallback heuristic, or None for tasks without one."""
    label_names = _label_names(model) if label_names is None else label_names
    if label_names != _label_names(model):
        raise ValueError("Evaluation label names conflict with the artifact vocabulary")
    fallback_label = _fallback_label(model) if fallback_label is None else fallback_label
    if fallback_label is not None and fallback_label not in label_names.values():
        raise ValueError("Fallback label is outside the artifact vocabulary")
    if fallback_label is None:
        return None
    targets = _target_names(list(eval_dataset["label"]), label_names)
    predictions = model.classify_batch(list(eval_dataset["text"]))
    return _threshold_recommendation(predictions, targets, fallback_label)


def evaluate_router(
    model: RouterModel,
    eval_dataset: Any,
    label_names: dict[int, str] | None = None,
    *,
    check_independence: bool = True,
) -> dict:
    """Report all configured classes, observed centroids, and confidence statistics."""
    label_names = _label_names(model) if label_names is None else label_names
    if label_names != _label_names(model):
        raise ValueError("Evaluation label names conflict with the artifact vocabulary")
    records = dataset_records(eval_dataset)
    if check_independence:
        if model.provenance is not None:
            assert_disjoint_keys(model.provenance.used_keys, record_keys(records))
        else:
            logger.warning(
                "Legacy artifact has no data provenance; evaluation independence is unverified"
            )
    texts = [row["text"] for row in records]
    targets = _target_names([row["label"] for row in records], label_names)
    if not texts or len(texts) != len(targets):
        raise ValueError("Evaluation requires a nonempty, aligned text/label dataset")
    predictions = model.classify_batch(texts)
    predicted_labels = [prediction.label for prediction in predictions]
    names = list(label_names.values())
    logger.info("Router evaluation complete", extra={"examples": len(texts), "classes": len(names)})
    texts_by_label = defaultdict(list)
    for text, target in zip(texts, targets, strict=True):
        texts_by_label[names.index(target)].append(text)
    scores = [prediction.score for prediction in predictions]
    return {
        **prediction_metrics(targets, predicted_labels, scores, names),
        "similarity_matrix": compute_similarity_matrix(model, texts_by_label),
        "similarity_labels": [label_names[label] for label in sorted(texts_by_label)],
        "tau_threshold": _threshold_recommendation(predictions, targets, _fallback_label(model)),
        "confidence_statistics": {
            "min": min(scores),
            "max": max(scores),
            "mean": float(np.mean(scores)),
        },
        "acceptance_policy": model.acceptance_policy.model_dump()
        if model.acceptance_policy
        else None,
        "policy_result": coverage_metrics(
            scores,
            [
                prediction.label == target
                for prediction, target in zip(predictions, targets, strict=True)
            ],
            model.acceptance_policy.threshold,
        )
        if model.acceptance_policy is not None
        else None,
    }
