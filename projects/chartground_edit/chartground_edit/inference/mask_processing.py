"""Strict post-processing for masks returned by Sa2VA ``predict_forward``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


class MaskProcessingError(ValueError):
    """Raised when an upstream mask has an unsupported shape or value domain."""

    def __init__(
        self,
        message: str,
        *,
        raw_mask_shapes: list[list[int]] | None = None,
        raw_mask_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_mask_shapes = raw_mask_shapes or []
        self.raw_mask_metadata = raw_mask_metadata or []


@dataclass
class MaskProcessingResult:
    """Normalized first mask plus metadata for every upstream mask."""

    mask: np.ndarray | None
    num_masks: int
    raw_mask_shapes: list[list[int]] = field(default_factory=list)
    raw_mask_metadata: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None
    resized: bool = False
    binarization: str = "accepted bool/0-1/0-255; nonzero values are foreground"


def process_prediction_masks(
    prediction_masks: Any,
    *,
    image_size: tuple[int, int],
) -> MaskProcessingResult:
    """Select and normalize the first mask without consulting ground truth.

    The upstream InternVL path returns a list whose elements are normally
    ``(1, H, W)`` bool arrays. Only leading singleton batch/frame dimensions
    are removed. Any non-singleton extra dimension is rejected explicitly.
    A spatial mismatch is restored with nearest-neighbor interpolation.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise MaskProcessingError("image_size dimensions must be positive")
    if prediction_masks is None:
        return MaskProcessingResult(
            mask=None,
            num_masks=0,
            failure_reason="missing_prediction_masks",
        )
    if not isinstance(prediction_masks, (list, tuple)):
        raise MaskProcessingError(
            "prediction_masks must be a list or tuple, got "
            f"{type(prediction_masks).__name__}"
        )

    raw_metadata = [_describe_raw_mask(item, index) for index, item in enumerate(prediction_masks)]
    shapes = [item["shape"] for item in raw_metadata]
    if not prediction_masks:
        return MaskProcessingResult(
            mask=None,
            num_masks=0,
            raw_mask_shapes=shapes,
            raw_mask_metadata=raw_metadata,
            failure_reason="empty_prediction_masks",
        )

    try:
        selected = _as_numpy(prediction_masks[0])
        while selected.ndim > 2 and selected.shape[0] == 1:
            selected = selected[0]
        if selected.ndim != 2:
            raise MaskProcessingError(
                "first prediction mask must reduce to (H, W) by removing only "
                f"leading singleton dimensions, got shape {selected.shape}"
            )
        binary = normalize_binary_values(selected, name="first prediction mask")
    except MaskProcessingError as exc:
        raise MaskProcessingError(
            str(exc),
            raw_mask_shapes=shapes,
            raw_mask_metadata=raw_metadata,
        ) from exc
    resized = binary.shape != (height, width)
    if resized:
        mask_image = Image.fromarray(binary.astype(np.uint8) * 255)
        binary = np.asarray(
            mask_image.resize((width, height), Image.Resampling.NEAREST)
        ) == 255
    binary = np.asarray(binary, dtype=bool)
    if binary.shape != (height, width):
        raise MaskProcessingError(
            f"normalized mask shape {binary.shape} does not match {(height, width)}"
        )
    failure_reason = None if binary.any() else "empty_prediction_mask"
    return MaskProcessingResult(
        mask=binary,
        num_masks=len(prediction_masks),
        raw_mask_shapes=shapes,
        raw_mask_metadata=raw_metadata,
        failure_reason=failure_reason,
        resized=resized,
    )


def normalize_binary_values(mask: Any, *, name: str = "mask") -> np.ndarray:
    """Convert bool, 0/1, or 0/255 two-dimensional data to bool."""
    array = _as_numpy(mask)
    if array.ndim != 2:
        raise MaskProcessingError(f"{name} must be two-dimensional, got {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise MaskProcessingError(f"{name} dimensions must be positive")
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise MaskProcessingError(f"{name} has unsupported dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise MaskProcessingError(f"{name} must not contain NaN or infinite values")
    values = set(np.unique(array).tolist())
    if not (values.issubset({0, 1}) or values.issubset({0, 255})):
        raise MaskProcessingError(
            f"{name} must contain only bool, 0/1, or 0/255 values; got {sorted(values)}"
        )
    return np.asarray(array != 0, dtype=bool)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
    try:
        return np.asarray(value)
    except Exception as exc:
        raise MaskProcessingError(
            f"mask cannot be converted to a NumPy array: {exc}"
        ) from exc


def _describe_raw_mask(value: Any, index: int) -> dict[str, Any]:
    try:
        array = _as_numpy(value)
        finite = bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None
        minimum = _python_scalar(array.min()) if array.size else None
        maximum = _python_scalar(array.max()) if array.size else None
        return {
            "index": index,
            "python_type": type(value).__name__,
            "shape": [int(item) for item in array.shape],
            "dtype": str(array.dtype),
            "min": minimum,
            "max": maximum,
            "finite": finite,
        }
    except Exception as exc:
        return {
            "index": index,
            "python_type": type(value).__name__,
            "shape": [],
            "dtype": None,
            "min": None,
            "max": None,
            "finite": None,
            "inspection_error": str(exc),
        }


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
