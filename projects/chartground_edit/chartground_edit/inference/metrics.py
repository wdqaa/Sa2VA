"""Dependency-light binary segmentation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .mask_processing import MaskProcessingError, normalize_binary_values


def intersection_over_union(prediction: Any, ground_truth: Any) -> float:
    """Compute binary IoU; two empty masks have score 1.0 by convention."""
    predicted, target = _validated_pair(prediction, ground_truth)
    intersection = int(np.logical_and(predicted, target).sum())
    union = int(np.logical_or(predicted, target).sum())
    return 1.0 if union == 0 else intersection / union


def dice_score(prediction: Any, ground_truth: Any) -> float:
    """Compute binary Dice; two empty masks have score 1.0 by convention."""
    predicted, target = _validated_pair(prediction, ground_truth)
    intersection = int(np.logical_and(predicted, target).sum())
    denominator = int(predicted.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def empty_prediction(prediction: Any | None) -> bool:
    """Return true for ``None`` or a valid mask containing no foreground."""
    if prediction is None:
        return True
    return not bool(normalize_binary_values(prediction, name="prediction").any())


def inference_success(
    prediction: Any | None,
    *,
    failure_reason: str | None = None,
) -> bool:
    """A successful inference has no failure code and a nonempty valid mask."""
    if failure_reason is not None or prediction is None:
        return False
    return not empty_prediction(prediction)


def _validated_pair(prediction: Any, ground_truth: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        predicted = normalize_binary_values(prediction, name="prediction")
        target = normalize_binary_values(ground_truth, name="ground_truth")
    except MaskProcessingError:
        raise
    if predicted.shape != target.shape:
        raise ValueError(
            f"prediction/ground-truth size mismatch: {predicted.shape} != {target.shape}"
        )
    return predicted, target
