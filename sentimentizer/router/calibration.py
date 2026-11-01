"""Fit held-out probability calibration and select an explicit acceptance policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sentimentizer.router.dataset import assert_disjoint, assert_disjoint_keys, record_keys
from sentimentizer.router.provenance import add_dataset
from sentimentizer.router.schemas import (
    DEFAULT_MIN_ACCEPTED,
    DEFAULT_MIN_ACCURACY,
    RouterAcceptancePolicy,
)

if TYPE_CHECKING:
    from sentimentizer.router.model import RouterModel


def select_acceptance_policy(
    scores: list[float],
    correct: list[bool],
    *,
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
    min_accepted: int = DEFAULT_MIN_ACCEPTED,
) -> RouterAcceptancePolicy:
    """Maximize observed coverage subject to accuracy and minimum evidence.

    This is a measured operating point, not a statistical accuracy guarantee.
    Equal-score examples are always accepted/rejected together.
    """
    if not scores or len(scores) != len(correct):
        raise ValueError("Policy selection requires aligned nonempty predictions")
    if not 0 < min_accuracy <= 1 or min_accepted < 1:
        raise ValueError("Invalid policy accuracy or minimum accepted count")
    confidence = np.asarray(scores)
    if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("Policy confidence must be finite and bounded")
    for threshold in sorted(set(scores)):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        accuracy = float(np.asarray(correct)[accepted].mean())
        if count >= min_accepted and accuracy >= min_accuracy:
            return RouterAcceptancePolicy(
                threshold=threshold,
                min_accuracy=min_accuracy,
                min_accepted=min_accepted,
                examples=len(scores),
                accepted=count,
                accuracy=accuracy,
            )
    return RouterAcceptancePolicy(
        min_accuracy=min_accuracy, min_accepted=min_accepted, examples=len(scores), accepted=0
    )


def calibrate_router(
    model: RouterModel,
    calibration_records: list[dict],
    policy_records: list[dict],
    *,
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
    min_accepted: int = DEFAULT_MIN_ACCEPTED,
) -> RouterModel:
    """Calibrate a frozen head, then select its operating point on separate data."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    from sentimentizer.router.model import RouterModel

    if model.provenance is None or not model.provenance.used_keys.get("texts"):
        raise ValueError("Calibration requires training provenance; retrain legacy artifacts first")
    if model.task is None:
        raise ValueError("Calibration requires an artifact with a configured task")
    if model.provenance.calibration_method is not None:
        raise ValueError("Artifact is already calibrated; calibrate the uncalibrated artifact")
    if not calibration_records or not policy_records:
        raise ValueError("Calibration and policy datasets must both be nonempty")
    assert_disjoint(calibration_records, policy_records)
    assert_disjoint_keys(
        model.provenance.used_keys, record_keys(calibration_records), record_keys(policy_records)
    )
    classes = dict(zip(model.class_labels(), model.head.classes_.tolist(), strict=True))
    calibration_labels = [
        model.task.labels[model.task.label_id(row["label"])] for row in calibration_records
    ]
    targets = [model.task.labels[model.task.label_id(row["label"])] for row in policy_records]
    if set(calibration_labels) != set(model.task.labels):
        raise ValueError("Calibration data must contain every configured class")
    # The estimator is frozen and was fit on disjoint training data. Use all
    # calibration rows for its fixed predictions, even for fewer than five
    # examples per class; the default CV would unnecessarily require five.
    indices = np.arange(len(calibration_records))
    calibrated_head = CalibratedClassifierCV(
        FrozenEstimator(model.head), method="sigmoid", cv=[(indices, indices)], ensemble=False
    )
    calibrated_head.fit(
        model.model_encode([row["text"] for row in calibration_records]),
        [classes[label] for label in calibration_labels],
    )
    provenance = model.provenance.model_copy(deep=True)
    provenance.calibration_method = "sigmoid"
    add_dataset(provenance, "calibration", calibration_records)
    add_dataset(provenance, "policy", policy_records)
    result = RouterModel(model.backbone, calibrated_head, task=model.task, provenance=provenance)
    predictions = result.classify_batch([row["text"] for row in policy_records])
    result.acceptance_policy = select_acceptance_policy(
        [prediction.score for prediction in predictions],
        [
            prediction.label == target
            for prediction, target in zip(predictions, targets, strict=True)
        ],
        min_accuracy=min_accuracy,
        min_accepted=min_accepted,
    )
    return result
