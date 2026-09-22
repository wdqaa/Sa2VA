"""Deterministic split selection and aggregate segmentation metrics."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .mask_processing import normalize_binary_values


class SplitEvaluationError(ValueError):
    """Raised when a requested manifest split cannot be evaluated as declared."""


@dataclass(frozen=True)
class PredictionAttempt:
    """A prediction or an explicit record that a sample was not run."""

    sample: Any
    prediction: Any | None
    not_run_reason: str | None = None


def select_split_samples(
    dataset: Sequence[Any], split: str, *, expected_count: int | None = None
) -> list[Any]:
    """Select a split in manifest order and enforce an optional exact count."""
    if type(split) is not str or not split:
        raise SplitEvaluationError("split must be a non-empty string")
    samples = [sample for sample in dataset if sample.annotation.get("split") == split]
    if not samples:
        raise SplitEvaluationError(f"manifest contains no samples for split={split!r}")
    if expected_count is not None and len(samples) != expected_count:
        raise SplitEvaluationError(
            f"expected {expected_count} samples for split={split!r}, found {len(samples)}"
        )
    return samples


def is_global_failure(prediction: Any) -> bool:
    """Identify failures after which reusing the model process is unsafe."""
    if prediction is None:
        return False
    if prediction.failure_reason == "model_load_failed":
        return True
    return bool(prediction.metadata.get("cuda_oom", False))


def run_prediction_sequence(
    samples: Sequence[Any],
    backend: Any,
    *,
    continue_on_sample_error: bool,
) -> list[PredictionAttempt]:
    """Call one backend instance once per sample, without retries or parallelism."""
    attempts: list[PredictionAttempt] = []
    stop_reason: str | None = None
    for sample in samples:
        if stop_reason is not None:
            attempts.append(PredictionAttempt(sample, None, stop_reason))
            continue
        try:
            prediction = backend.predict_mask(
                sample.image, sample.annotation["instruction"]
            )
        except Exception as exc:
            attempts.append(
                PredictionAttempt(
                    sample,
                    None,
                    f"backend_exception:{type(exc).__name__}:{exc}",
                )
            )
            if not continue_on_sample_error:
                stop_reason = "not_run_after_sample_error"
            continue
        attempts.append(PredictionAttempt(sample, prediction))
        if is_global_failure(prediction):
            stop_reason = "not_run_after_global_error"
        elif not prediction.success and not continue_on_sample_error:
            stop_reason = "not_run_after_sample_error"
    return attempts


def binary_pixel_counts(prediction: Any, ground_truth: Any) -> dict[str, int]:
    """Return foreground, intersection and union counts for a binary pair."""
    predicted = normalize_binary_values(prediction, name="prediction")
    target = normalize_binary_values(ground_truth, name="ground_truth")
    if predicted.shape != target.shape:
        raise ValueError(
            f"prediction/ground-truth size mismatch: {predicted.shape} != {target.shape}"
        )
    predicted_pixels = int(predicted.sum())
    target_pixels = int(target.sum())
    intersection = int(np.logical_and(predicted, target).sum())
    union = predicted_pixels + target_pixels - intersection
    return {
        "predicted_foreground_pixels": predicted_pixels,
        "gt_foreground_pixels": target_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
    }


def summarize_results(
    results: Sequence[dict[str, Any]],
    *,
    model_load_time_ms: float | None,
    model_load_attempts: int | None,
) -> dict[str, Any]:
    """Aggregate every requested sample, including failed and empty predictions."""
    rows = list(results)
    sample_count = len(rows)
    if not rows:
        raise SplitEvaluationError("cannot summarize an empty result set")

    ious = [float(row["iou"]) for row in rows if row.get("iou") is not None]
    dices = [float(row["dice"]) for row in rows if row.get("dice") is not None]
    inference_times = [
        float(row["inference_time_ms"])
        for row in rows
        if row.get("inference_time_ms") is not None
    ]
    intersection = sum(int(row["intersection_pixels"]) for row in rows)
    union = sum(int(row["union_pixels"]) for row in rows)
    predicted = sum(int(row["predicted_foreground_pixels"]) for row in rows)
    target = sum(int(row["gt_foreground_pixels"]) for row in rows)
    successful = sum(bool(row.get("success")) for row in rows)
    empty = sum(bool(row.get("empty_prediction")) for row in rows)
    disjoint = sum(bool(row.get("nonempty_disjoint")) for row in rows)
    overlap = sum(int(row["intersection_pixels"]) > 0 for row in rows)
    peaks = [
        float(row["peak_gpu_memory_mb"])
        for row in rows
        if row.get("peak_gpu_memory_mb") is not None
    ]
    null_reasons: dict[str, str] = {}
    if len(ious) != sample_count:
        null_reasons["macro_metrics"] = (
            f"{sample_count - len(ious)} sample(s) lacked computable overlap metrics"
        )
    if len(inference_times) != sample_count:
        null_reasons["inference_time"] = (
            f"{sample_count - len(inference_times)} sample(s) did not record inference time"
        )

    return {
        "sample_count": sample_count,
        "successful_inference_count": successful,
        "inference_success_rate": successful / sample_count,
        "empty_prediction_count": empty,
        "empty_prediction_rate": empty / sample_count,
        "nonempty_disjoint_count": disjoint,
        "nonempty_disjoint_rate": disjoint / sample_count,
        "overlap_count": overlap,
        "overlap_rate": overlap / sample_count,
        "mean_iou": statistics.fmean(ious) if len(ious) == sample_count else None,
        "median_iou": statistics.median(ious) if len(ious) == sample_count else None,
        "mean_dice": statistics.fmean(dices) if len(dices) == sample_count else None,
        "median_dice": statistics.median(dices) if len(dices) == sample_count else None,
        "micro_iou": intersection / union if union else None,
        "micro_dice": 2 * intersection / (predicted + target)
        if predicted + target
        else None,
        "total_intersection_pixels": intersection,
        "total_union_pixels": union,
        "total_predicted_foreground_pixels": predicted,
        "total_gt_foreground_pixels": target,
        "timed_inference_count": len(inference_times),
        "mean_inference_time_ms": statistics.fmean(inference_times)
        if inference_times
        else None,
        "total_inference_time_ms": sum(inference_times) if inference_times else None,
        "model_load_time_ms": model_load_time_ms,
        "model_load_attempts": model_load_attempts,
        "peak_gpu_memory_mb": max(peaks) if peaks else None,
        "multiple_mask_count": sum(int(row.get("num_masks", 0)) > 1 for row in rows),
        "null_metric_reasons": null_reasons,
    }


def gallery_status_label(result: dict[str, Any]) -> str:
    """Produce an explicit, testable state label for gallery rows."""
    if result.get("empty_prediction"):
        reason = result.get("failure_reason")
        return f"EMPTY ({reason})" if reason else "EMPTY"
    if not result.get("success"):
        return f"FAILED ({result.get('failure_reason') or 'unknown'})"
    if result.get("nonempty_disjoint"):
        return "SUCCESS / NONEMPTY_DISJOINT"
    return "SUCCESS"


def write_summary_files(
    output_dir: str | Path,
    results: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Write deterministic JSONL, JSON and two-column CSV summaries."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "results.jsonl"
    summary_path = root / "summary.json"
    csv_path = root / "summary.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in summary.items():
            serialized = (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
            writer.writerow((key, serialized))
    return jsonl_path, summary_path, csv_path
