"""Model-facing inference boundary for ChartGround-Edit."""

from .mask_processing import (
    MaskProcessingError,
    MaskProcessingResult,
    process_prediction_masks,
)
from .metrics import (
    dice_score,
    empty_prediction,
    inference_success,
    intersection_over_union,
)
from .sa2va_backend import (
    PROMPT_TEMPLATE,
    Sa2VAError,
    Sa2VAInternVL3Backend,
    Sa2VALoadError,
    build_prompt,
)
from .split_evaluation import (
    PredictionAttempt,
    SplitEvaluationError,
    binary_pixel_counts,
    gallery_status_label,
    is_global_failure,
    run_prediction_sequence,
    select_split_samples,
    summarize_results,
    write_summary_files,
)
from .types import PredictionResult

__all__ = [
    "MaskProcessingError",
    "MaskProcessingResult",
    "PROMPT_TEMPLATE",
    "PredictionAttempt",
    "PredictionResult",
    "Sa2VAError",
    "Sa2VAInternVL3Backend",
    "Sa2VALoadError",
    "SplitEvaluationError",
    "binary_pixel_counts",
    "build_prompt",
    "dice_score",
    "empty_prediction",
    "gallery_status_label",
    "inference_success",
    "intersection_over_union",
    "is_global_failure",
    "process_prediction_masks",
    "run_prediction_sequence",
    "select_split_samples",
    "summarize_results",
    "write_summary_files",
]
