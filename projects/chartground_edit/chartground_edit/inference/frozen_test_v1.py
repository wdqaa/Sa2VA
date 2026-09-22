"""Frozen-test validation, aggregation, bootstrap, gallery, and edit helpers."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PIL import Image

from chartground_edit.editing import edit

from .prompt_benchmark_v1 import result_metric_semantics
from .prompt_variants import TARGET_ONLY_ZH
from .split_evaluation import SplitEvaluationError


EXPECTED_TEST_SAMPLES = 64
EXPECTED_TEST_ATTEMPTS = 64
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_ITERATIONS = 10_000
CHART_TYPES = ("line", "bar", "scatter", "confidence_band")
REFERRING_TYPES = ("category", "appearance", "legend", "trend")
EDIT_ACTIONS = ("highlight", "recolor", "extract", "remove")


def validate_frozen_test_split(split: str) -> str:
    """Permit test only, naming train and val as explicit protocol violations."""
    if split == "train":
        raise ValueError("synthetic_v1 train inference is forbidden in Phase 3C")
    if split == "val":
        raise ValueError("synthetic_v1 val must not be rerun in Phase 3C")
    if split != "test":
        raise ValueError("Phase 3C only permits split='test'")
    return split


def validate_frozen_prompt_variant(variant: str) -> str:
    """Permit only the globally frozen P2 registry entry."""
    if variant != TARGET_ONLY_ZH:
        raise ValueError(
            "Phase 3C only permits frozen P2 prompt variant 'target_only_zh'"
        )
    return variant


def validate_balanced_test_annotations(
    annotations: Sequence[dict[str, Any]],
    *,
    split: str = "test",
    expected_samples: int = EXPECTED_TEST_SAMPLES,
) -> list[dict[str, Any]]:
    """Select and audit the frozen 64-sample balanced test split."""
    validate_frozen_test_split(split)
    if expected_samples != EXPECTED_TEST_SAMPLES:
        raise SplitEvaluationError(
            f"Phase 3C requires expected_samples={EXPECTED_TEST_SAMPLES}"
        )
    selected = [row for row in annotations if row.get("split") == "test"]
    if len(selected) != EXPECTED_TEST_SAMPLES:
        raise SplitEvaluationError(
            f"expected {EXPECTED_TEST_SAMPLES} synthetic_v1 test samples, "
            f"found {len(selected)}"
        )
    sample_ids = [str(row.get("sample_id")) for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise SplitEvaluationError("test sample_id values must be unique")

    expected_groups = {
        (chart, referring) for chart in CHART_TYPES for referring in REFERRING_TYPES
    }
    groups = Counter(
        (row.get("chart_type"), row.get("referring_type")) for row in selected
    )
    if set(groups) != expected_groups:
        missing = sorted(expected_groups - set(groups))
        extra = sorted(set(groups) - expected_groups)
        raise SplitEvaluationError(
            f"test chart/referring coverage mismatch; missing={missing}, extra={extra}"
        )
    wrong_counts = {
        f"{chart}/{referring}": groups[(chart, referring)]
        for chart, referring in sorted(expected_groups)
        if groups[(chart, referring)] != 4
    }
    if wrong_counts:
        raise SplitEvaluationError(
            f"each test chart/referring group must contain 4 samples: {wrong_counts}"
        )
    expected_actions = Counter({action: 1 for action in EDIT_ACTIONS})
    for chart, referring in sorted(expected_groups):
        actions = Counter(
            row.get("edit_action")
            for row in selected
            if (row.get("chart_type"), row.get("referring_type"))
            == (chart, referring)
        )
        if actions != expected_actions:
            raise SplitEvaluationError(
                f"test group {chart}/{referring} must contain each edit action "
                f"once; got {dict(actions)}"
            )
    return selected


def test_sample_id_list_sha256(annotations: Sequence[dict[str, Any]]) -> str:
    """Hash manifest-order test IDs with an LF after every ID."""
    selected = validate_balanced_test_annotations(annotations)
    payload = "".join(f"{row['sample_id']}\n" for row in selected).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_frozen_test_plan(
    annotations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one and only one P2 attempt per test sample."""
    selected = validate_balanced_test_annotations(annotations)
    plan = [
        {
            "attempt_index": index,
            "sample_id": row["sample_id"],
            "prompt_variant": TARGET_ONLY_ZH,
        }
        for index, row in enumerate(selected)
    ]
    if len(plan) != EXPECTED_TEST_ATTEMPTS:
        raise AssertionError(
            f"expected {EXPECTED_TEST_ATTEMPTS} attempts, got {len(plan)}"
        )
    return plan


def validate_frozen_test_results(
    results: Sequence[dict[str, Any]],
    *,
    expected_samples: int = EXPECTED_TEST_SAMPLES,
) -> None:
    """Require exactly one P2 record for every one of 64 unique test IDs."""
    if expected_samples != EXPECTED_TEST_SAMPLES:
        raise SplitEvaluationError(
            f"Phase 3C requires expected_samples={EXPECTED_TEST_SAMPLES}"
        )
    rows = list(results)
    if len(rows) != EXPECTED_TEST_ATTEMPTS:
        raise SplitEvaluationError(
            f"expected {EXPECTED_TEST_ATTEMPTS} result records, found {len(rows)}"
        )
    ids: list[str] = []
    for row in rows:
        if row.get("split") != "test":
            raise SplitEvaluationError("frozen-test results contain a non-test record")
        validate_frozen_prompt_variant(str(row.get("prompt_variant")))
        ids.append(str(row.get("sample_id")))
    duplicates = sorted(key for key, count in Counter(ids).items() if count != 1)
    if duplicates or len(set(ids)) != EXPECTED_TEST_SAMPLES:
        raise SplitEvaluationError(
            "every test sample must appear exactly once; "
            f"unique={len(set(ids))}, duplicate_or_invalid={duplicates}"
        )


def frozen_result_metric_semantics(
    *,
    prediction_returned: bool,
    failure_reason: str | None,
    error_type: str | None,
    mask_contract_valid: bool,
    predicted_foreground_pixels: int,
    intersection_pixels: int,
) -> dict[str, bool]:
    """Expose Phase 3B semantics without the deprecated legacy success field."""
    semantics = result_metric_semantics(
        prediction_returned=prediction_returned,
        failure_reason=failure_reason,
        error_type=error_type,
        mask_contract_valid=mask_contract_valid,
        predicted_foreground_pixels=predicted_foreground_pixels,
        intersection_pixels=intersection_pixels,
    )
    semantics["segmentation_token_present"] = False
    return semantics


def summarize_frozen_test(
    results: Sequence[dict[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Aggregate every test record, including execution and contract failures."""
    rows = list(results)
    validate_frozen_test_results(rows)
    grouped = {
        "chart_referring": _summaries_by(
            rows, lambda row: f"{row['chart_type']}/{row['referring_type']}"
        ),
        "chart_type": _summaries_by(rows, lambda row: str(row["chart_type"])),
        "referring_type": _summaries_by(
            rows, lambda row: str(row["referring_type"])
        ),
        "edit_action": _summaries_by(rows, lambda row: str(row["edit_action"])),
        "difficulty": _summaries_by(rows, lambda row: str(row["difficulty"])),
        "distractor_count": _summaries_by(
            rows, lambda row: str(row["distractor_count"])
        ),
    }
    base = _aggregate_rows(rows)
    group_values = grouped["chart_referring"]
    group_names = sorted(group_values)
    base.update(
        {
            "group_macro_iou": statistics.fmean(
                group_values[name]["mean_iou"] for name in group_names
            ),
            "group_macro_dice": statistics.fmean(
                group_values[name]["mean_dice"] for name in group_names
            ),
            "worst_group": min(
                group_names,
                key=lambda name: (group_values[name]["mean_iou"], name),
            ),
            "worst_group_mean_iou": min(
                group_values[name]["mean_iou"] for name in group_names
            ),
            "groups": grouped,
            "bootstrap_16_group_macro": bootstrap_group_macro_ci(
                group_values,
                seed=bootstrap_seed,
                iterations=bootstrap_iterations,
            ),
            "worst_10_samples": [
                {
                    "sample_id": row["sample_id"],
                    "chart_type": row["chart_type"],
                    "referring_type": row["referring_type"],
                    "edit_action": row["edit_action"],
                    "difficulty": row["difficulty"],
                    "iou": row["iou"],
                    "dice": row["dice"],
                    "execution_success": row["execution_success"],
                    "mask_contract_valid": row["mask_contract_valid"],
                    "empty_prediction": row["empty_prediction"],
                    "nonempty_disjoint": row["nonempty_disjoint"],
                    "failure_reason": row.get("failure_reason"),
                }
                for row in sorted(
                    rows,
                    key=lambda item: (
                        float(item["iou"]),
                        float(item["dice"]),
                        item["sample_id"],
                    ),
                )[:10]
            ],
        }
    )
    return {
        "metric_semantics": {
            "execution_success": (
                "predict_forward completed without a runtime exception"
            ),
            "mask_contract_valid": (
                "prediction_masks normalized to an original-size binary mask; "
                "an empty mask remains valid"
            ),
            "segmentation_token_present": "generated text contains [SEG]",
            "nonempty_prediction": "normalized mask contains foreground pixels",
            "empty_prediction": "logical inverse of nonempty_prediction",
            "overlapping_prediction": "normalized prediction intersects the GT mask",
            "nonempty_disjoint": (
                "normalized prediction is nonempty and has zero GT intersection"
            ),
        },
        "sample_count": len(rows),
        "attempt_count": len(rows),
        "prompt_variant": TARGET_ONLY_ZH,
        "metrics": base,
    }


def bootstrap_group_macro_ci(
    group_summaries: dict[str, dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Bootstrap the 16 group means for IoU and Dice deterministically."""
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    names = sorted(group_summaries)
    if len(names) != 16:
        raise SplitEvaluationError(
            f"group bootstrap requires 16 groups, found {len(names)}"
        )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(iterations, len(names)))
    output: dict[str, Any] = {
        "seed": seed,
        "iterations": iterations,
        "unit": "chart_type_x_referring_type_group",
    }
    for metric, source in (("iou", "mean_iou"), ("dice", "mean_dice")):
        values = np.asarray(
            [float(group_summaries[name][source]) for name in names],
            dtype=np.float64,
        )
        samples = values[indices].mean(axis=1)
        lower, upper = np.quantile(samples, (0.025, 0.975))
        output[metric] = {
            "estimate": float(values.mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
        }
    return output


def select_frozen_test_gallery_sample_ids(
    annotations: Sequence[dict[str, Any]],
) -> list[str]:
    """Choose the lexicographically first test ID in each fixed group."""
    selected = validate_balanced_test_annotations(annotations)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in selected:
        grouped[(row["chart_type"], row["referring_type"])].append(row["sample_id"])
    return [
        min(grouped[(chart, referring)])
        for chart in CHART_TYPES
        for referring in REFERRING_TYPES
    ]


def normalize_v1_edit_parameters(
    action: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Translate the frozen v1 action schema to the existing v0 editor API."""
    if action == "highlight" and set(parameters) == {"strength"}:
        return {"strength": parameters["strength"]}
    if action == "recolor" and set(parameters) == {"color"}:
        return {"color": parameters["color"]}
    if action == "extract" and parameters == {"background": "transparent"}:
        return {}
    if (
        action == "remove"
        and set(parameters) == {"fill_mode", "color"}
        and parameters["fill_mode"] == "background_color"
    ):
        return {"fill_mode": "color", "color": parameters["color"]}
    raise ValueError(f"unsupported frozen v1 edit parameters for {action!r}")


def apply_predicted_mask_edit(
    image: Image.Image,
    predicted_mask: np.ndarray,
    *,
    edit_action: str,
    edit_parameters: dict[str, Any],
    editor: Callable[..., Image.Image] = edit,
) -> Image.Image:
    """Edit using only the explicit predicted mask passed by the caller."""
    parameters = normalize_v1_edit_parameters(edit_action, edit_parameters)
    return editor(image, predicted_mask, edit_action, parameters)


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    ious = [float(row["iou"]) for row in rows]
    dices = [float(row["dice"]) for row in rows]
    times = [
        float(row["latency_ms"])
        for row in rows
        if row.get("latency_ms") is not None
    ]
    intersection = sum(int(row["intersection_pixels"]) for row in rows)
    union = sum(int(row["union_pixels"]) for row in rows)
    predicted = sum(int(row["predicted_foreground_pixels"]) for row in rows)
    target = sum(int(row["gt_foreground_pixels"]) for row in rows)
    edits_attempted = sum(row.get("edit_execution_success") is not None for row in rows)
    edits_succeeded = sum(row.get("edit_execution_success") is True for row in rows)
    return {
        "execution_success_count": sum(bool(row["execution_success"]) for row in rows),
        "execution_success_rate": sum(bool(row["execution_success"]) for row in rows)
        / count,
        "mask_contract_valid_count": sum(
            bool(row["mask_contract_valid"]) for row in rows
        ),
        "mask_contract_valid_rate": sum(
            bool(row["mask_contract_valid"]) for row in rows
        )
        / count,
        "segmentation_token_count": sum(
            bool(row["segmentation_token_present"]) for row in rows
        ),
        "segmentation_token_rate": sum(
            bool(row["segmentation_token_present"]) for row in rows
        )
        / count,
        "nonempty_prediction_count": sum(
            bool(row["nonempty_prediction"]) for row in rows
        ),
        "nonempty_prediction_rate": sum(
            bool(row["nonempty_prediction"]) for row in rows
        )
        / count,
        "empty_prediction_count": sum(bool(row["empty_prediction"]) for row in rows),
        "empty_prediction_rate": sum(bool(row["empty_prediction"]) for row in rows)
        / count,
        "overlapping_prediction_count": sum(
            bool(row["overlapping_prediction"]) for row in rows
        ),
        "overlapping_prediction_rate": sum(
            bool(row["overlapping_prediction"]) for row in rows
        )
        / count,
        "nonempty_disjoint_count": sum(
            bool(row["nonempty_disjoint"]) for row in rows
        ),
        "nonempty_disjoint_rate": sum(
            bool(row["nonempty_disjoint"]) for row in rows
        )
        / count,
        "sample_macro_iou": statistics.fmean(ious),
        "sample_macro_dice": statistics.fmean(dices),
        "median_iou": statistics.median(ious),
        "median_dice": statistics.median(dices),
        "micro_iou": intersection / union if union else 0.0,
        "micro_dice": 2.0 * intersection / (predicted + target)
        if predicted + target
        else 0.0,
        "mean_latency_ms": statistics.fmean(times) if times else None,
        "median_latency_ms": statistics.median(times) if times else None,
        "timed_attempt_count": len(times),
        "total_intersection_pixels": intersection,
        "total_union_pixels": union,
        "total_predicted_foreground_pixels": predicted,
        "total_gt_foreground_pixels": target,
        "edit_attempt_count": edits_attempted,
        "edit_execution_success_count": edits_succeeded,
        "edit_execution_success_rate_among_attempted": (
            edits_succeeded / edits_attempted if edits_attempted else None
        ),
        "edit_skipped_empty_count": sum(
            bool(row["edit_skipped_empty"]) for row in rows
        ),
    }


def _summaries_by(
    rows: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return {
        name: {
            "count": len(values),
            "mean_iou": statistics.fmean(float(row["iou"]) for row in values),
            "mean_dice": statistics.fmean(float(row["dice"]) for row in values),
            "empty_rate": sum(bool(row["empty_prediction"]) for row in values)
            / len(values),
            "nonempty_disjoint_rate": sum(
                bool(row["nonempty_disjoint"]) for row in values
            )
            / len(values),
        }
        for name, values in sorted(grouped.items())
    }
