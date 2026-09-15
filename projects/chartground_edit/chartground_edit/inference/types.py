"""Shared result types for segmentation inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PredictionResult:
    """One model prediction without any ground-truth-derived fields.

    ``mask`` is either ``None`` or a two-dimensional bool array in original
    image coordinates. ``raw_masks`` deliberately preserves the object
    returned by the upstream model and is not intended for JSON serialization.
    """

    mask: np.ndarray | None
    raw_masks: Any
    text_output: str | None
    model_name: str
    checkpoint_path: str
    instruction: str
    prompt: str
    num_masks: int
    inference_time_ms: float | None
    model_load_time_ms: float | None
    peak_gpu_memory_mb: float | None
    success: bool
    failure_reason: str | None
    raw_mask_shapes: list[list[int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
