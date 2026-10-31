"""Evaluate task-specific router labels and historical fallback thresholds."""

from __future__ import annotations

import logging
from collections import defaultdict
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import classification_report

from sentimentizer.router.config import FOOD_TASK_SPEC

if TYPE_CHECKING:
    from sentimentizer.router.model import RouterModel
    from sentimentizer.router.schemas import RouterClassification

logger = logging.getLogger(__name__)


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
    model: RouterModel, eval_dataset: Any, label_names: dict[int, str] | None = None
) -> dict:
    """Report all configured classes, observed centroids, and confidence statistics."""
    label_names = _label_names(model) if label_names is None else label_names
    if label_names != _label_names(model):
        raise ValueError("Evaluation label names conflict with the artifact vocabulary")
    texts = list(eval_dataset["text"])
    targets = _target_names(list(eval_dataset["label"]), label_names)
    if not texts or len(texts) != len(targets):
        raise ValueError("Evaluation requires a nonempty, aligned text/label dataset")
    predictions = model.classify_batch(texts)
    predicted_labels = [prediction.label for prediction in predictions]
    names = list(label_names.values())
    report = classification_report(
        targets,
        predicted_labels,
        labels=names,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    logger.info("Router evaluation complete", extra={"examples": len(texts), "classes": len(names)})
    texts_by_label = defaultdict(list)
    for text, target in zip(texts, targets, strict=True):
        texts_by_label[names.index(target)].append(text)
    scores = [prediction.score for prediction in predictions]
    return {
        "classification_report": report,
        "similarity_matrix": compute_similarity_matrix(model, texts_by_label),
        "similarity_labels": [label_names[label] for label in sorted(texts_by_label)],
        "tau_threshold": _threshold_recommendation(predictions, targets, _fallback_label(model)),
        "confidence_statistics": {
            "min": min(scores),
            "max": max(scores),
            "mean": float(np.mean(scores)),
        },
    }
