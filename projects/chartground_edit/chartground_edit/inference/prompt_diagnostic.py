"""Pure statistics and planning helpers for the Phase 2C diagnostic."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .prompt_variants import PROMPT_VARIANT_ORDER
from .split_evaluation import SplitEvaluationError, summarize_results


SPLIT_ORDER = ("train", "val", "test")
CHART_ORDER = ("line", "bar", "scatter", "confidence_band")
REFERRING_ORDER = ("category", "appearance", "legend", "trend")
ACTION_ORDER = ("highlight", "recolor", "extract", "remove")
SYNTHETIC_V1_SPLIT_PER_COMBINATION = {"train": 12, "val": 4, "test": 4}


def validate_diagnostic_split(split: str) -> str:
    """Allow val only and explicitly protect the already-used test split."""
    if split == "test":
        raise ValueError(
            "test split is frozen after Phase 2B-2 and cannot be used for Prompt diagnosis"
        )
    if split != "val":
        raise ValueError("Phase 2C Prompt diagnosis only permits split='val'")
    return split


def audit_split_distribution(annotations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cross-tabulate synthetic-v0 with explicit zeros for missing coverage."""
    rows = list(annotations)
    if not rows:
        raise SplitEvaluationError("cannot audit an empty annotation sequence")
    split_counts = Counter(row["split"] for row in rows)
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "split_counts": {split: split_counts[split] for split in SPLIT_ORDER},
        "split_chart_type": {},
        "split_referring_type": {},
        "split_edit_action": {},
        "split_chart_referring": {},
        "missing_chart_referring": {},
    }
    for split in SPLIT_ORDER:
        selected = [row for row in rows if row["split"] == split]
        charts = Counter(row["chart_type"] for row in selected)
        referring = Counter(row["referring_type"] for row in selected)
        actions = Counter(row["edit_action"] for row in selected)
        combinations = Counter(
            (row["chart_type"], row["referring_type"]) for row in selected
        )
        combination_counts = {
            f"{chart}/{reference}": combinations[(chart, reference)]
            for chart in CHART_ORDER
            for reference in REFERRING_ORDER
        }
        result["split_chart_type"][split] = {
            value: charts[value] for value in CHART_ORDER
        }
        result["split_referring_type"][split] = {
            value: referring[value] for value in REFERRING_ORDER
        }
        result["split_edit_action"][split] = {
            value: actions[value] for value in ACTION_ORDER
        }
        result["split_chart_referring"][split] = combination_counts
        result["missing_chart_referring"][split] = [
            key for key, count in combination_counts.items() if count == 0
        ]
    return result


def summarize_prompt_variants(
    results: Sequence[dict[str, Any]],
    variants: Sequence[str] = PROMPT_VARIANT_ORDER,
) -> dict[str, dict[str, Any]]:
    """Calculate the registered metrics separately for every Prompt variant."""
    summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        selected = [row for row in results if row["prompt_variant"] == variant]
        if not selected:
            raise SplitEvaluationError(f"no results for Prompt variant {variant!r}")
        base = summarize_results(
            selected,
            model_load_time_ms=None,
            model_load_attempts=None,
        )
        seg_count = sum(
            isinstance(row.get("text_output"), str)
            and "[SEG]" in row["text_output"]
            for row in selected
        )
        summaries[variant] = {
            "sample_count": base["sample_count"],
            "successful_inference_count": base["successful_inference_count"],
            "inference_success_rate": base["inference_success_rate"],
            "seg_output_count": seg_count,
            "seg_output_rate": seg_count / len(selected),
            "empty_prediction_count": base["empty_prediction_count"],
            "empty_prediction_rate": base["empty_prediction_rate"],
            "nonempty_disjoint_count": base["nonempty_disjoint_count"],
            "nonempty_disjoint_rate": base["nonempty_disjoint_rate"],
            "overlap_count": base["overlap_count"],
            "overlap_rate": base["overlap_rate"],
            "mean_iou": base["mean_iou"],
            "median_iou": base["median_iou"],
            "mean_dice": base["mean_dice"],
            "median_dice": base["median_dice"],
            "micro_iou": base["micro_iou"],
            "micro_dice": base["micro_dice"],
            "mean_inference_time_ms": base["mean_inference_time_ms"],
        }
    return summaries


def build_paired_results(
    results: Sequence[dict[str, Any]],
    variants: Sequence[str] = PROMPT_VARIANT_ORDER,
) -> list[dict[str, Any]]:
    """Align P0/P1/P2 by sample without ranking them by ground truth."""
    sample_order: list[str] = []
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        sample_id = row["sample_id"]
        variant = row["prompt_variant"]
        if sample_id not in grouped:
            grouped[sample_id] = {}
            sample_order.append(sample_id)
        if variant in grouped[sample_id]:
            raise SplitEvaluationError(
                f"duplicate result for sample={sample_id!r}, variant={variant!r}"
            )
        grouped[sample_id][variant] = row
    paired: list[dict[str, Any]] = []
    for sample_id in sample_order:
        missing = [variant for variant in variants if variant not in grouped[sample_id]]
        if missing:
            raise SplitEvaluationError(
                f"sample {sample_id!r} is missing variants: {', '.join(missing)}"
            )
        variant_rows = grouped[sample_id]
        metrics = {
            variant: {
                "iou": variant_rows[variant]["iou"],
                "dice": variant_rows[variant]["dice"],
                "predicted_foreground_pixels": variant_rows[variant][
                    "predicted_foreground_pixels"
                ],
                "empty_prediction": variant_rows[variant]["empty_prediction"],
                "nonempty_disjoint": variant_rows[variant]["nonempty_disjoint"],
                "overlap": variant_rows[variant]["intersection_pixels"] > 0,
                "seg_output": isinstance(variant_rows[variant].get("text_output"), str)
                and "[SEG]" in variant_rows[variant]["text_output"],
            }
            for variant in variants
        }
        baseline = metrics[variants[0]]
        paired.append(
            {
                "sample_id": sample_id,
                "chart_type": variant_rows[variants[0]]["chart_type"],
                "referring_type": variant_rows[variants[0]]["referring_type"],
                "variants": metrics,
                "transitions_from_full_instruction": {
                    variant: {
                        "disjoint_to_overlap": bool(
                            baseline["nonempty_disjoint"] and metrics[variant]["overlap"]
                        ),
                        "mask_to_empty": bool(
                            not baseline["empty_prediction"]
                            and metrics[variant]["empty_prediction"]
                        ),
                        "seg_to_no_seg": bool(
                            baseline["seg_output"] and not metrics[variant]["seg_output"]
                        ),
                        "foreground_pixel_delta": int(
                            metrics[variant]["predicted_foreground_pixels"]
                            - baseline["predicted_foreground_pixels"]
                        ),
                    }
                    for variant in variants[1:]
                },
            }
        )
    return paired


def organize_gallery_rows(
    results: Sequence[dict[str, Any]],
    variants: Sequence[str] = PROMPT_VARIANT_ORDER,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Return deterministic complete rows for GT/P0/P1/P2 rendering."""
    paired = build_paired_results(results, variants)
    lookup = {
        (row["sample_id"], row["prompt_variant"]): row for row in results
    }
    return [
        (
            pair["sample_id"],
            [lookup[(pair["sample_id"], variant)] for variant in variants],
        )
        for pair in paired
    ]


def validate_synthetic_v1_plan(
    *,
    chart_types: int = 4,
    referring_types: int = 4,
    split_per_combination: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate the proposed 20-sample-per-combination balanced split."""
    split_counts = dict(split_per_combination or SYNTHETIC_V1_SPLIT_PER_COMBINATION)
    if chart_types != 4 or referring_types != 4:
        raise ValueError("synthetic-v1 requires 4 chart and 4 referring types")
    if set(split_counts) != set(SPLIT_ORDER):
        raise ValueError("synthetic-v1 split plan must define train, val and test")
    if any(type(value) is not int or value <= 0 for value in split_counts.values()):
        raise ValueError("per-combination split counts must be positive integers")
    combination_count = chart_types * referring_types
    per_combination = sum(split_counts.values())
    if per_combination < 20:
        raise ValueError("synthetic-v1 requires at least 20 samples per combination")
    totals = {
        split: combination_count * split_counts[split] for split in SPLIT_ORDER
    }
    return {
        "chart_type_count": chart_types,
        "referring_type_count": referring_types,
        "combination_count": combination_count,
        "per_combination": per_combination,
        "split_per_combination": split_counts,
        "split_totals": totals,
        "total": sum(totals.values()),
    }
