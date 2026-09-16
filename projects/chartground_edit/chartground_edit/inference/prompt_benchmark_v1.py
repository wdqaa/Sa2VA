"""Pure validation, aggregation, selection, and bootstrap for Phase 3B."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .prompt_variants import PROMPT_TEMPLATES, PROMPT_VARIANT_ORDER
from .split_evaluation import SplitEvaluationError


EXPECTED_VAL_SAMPLES = 64
EXPECTED_ATTEMPTS = EXPECTED_VAL_SAMPLES * len(PROMPT_VARIANT_ORDER)
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_ITERATIONS = 10_000
CHART_TYPES = ("line", "bar", "scatter", "confidence_band")
REFERRING_TYPES = ("category", "appearance", "legend", "trend")
EDIT_ACTIONS = ("highlight", "recolor", "extract", "remove")
RUNTIME_FAILURE_REASONS = frozenset(
    {
        "model_load_failed",
        "model_inference_failed",
    }
)


def result_metric_semantics(
    *,
    prediction_returned: bool,
    failure_reason: str | None,
    error_type: str | None,
    mask_contract_valid: bool,
    predicted_foreground_pixels: int,
    intersection_pixels: int,
) -> dict[str, bool]:
    """Return orthogonal execution, contract, and prediction outcome flags.

    ``failure_reason`` may describe a valid empty model outcome. It therefore
    must not be used as a blanket synonym for an execution failure.
    """
    del error_type  # Preserved in the API for auditable caller context.
    reason = str(failure_reason or "")
    runtime_failure = (
        not prediction_returned
        or reason in RUNTIME_FAILURE_REASONS
        or reason.startswith("backend_exception:")
        or reason.startswith("not_run_after_")
    )
    contract_valid = bool(mask_contract_valid)
    nonempty = contract_valid and int(predicted_foreground_pixels) > 0
    return {
        "execution_success": not runtime_failure,
        "mask_contract_valid": contract_valid,
        "nonempty_prediction": nonempty,
        "empty_prediction": not nonempty,
        "overlapping_prediction": contract_valid and int(intersection_pixels) > 0,
    }


def ensure_result_metric_semantics(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with unambiguous flags, including legacy-row support."""
    output = dict(row)
    mask_contract_valid = bool(
        output.get(
            "mask_contract_valid",
            output.get("predicted_mask_source") == "model"
            and isinstance(output.get("predicted_mask_shape"), list)
            and len(output["predicted_mask_shape"]) == 2,
        )
    )
    reason = output.get("failure_reason")
    prediction_returned = bool(
        output.get(
            "execution_success",
            not str(reason or "").startswith("backend_exception:")
            and not str(reason or "").startswith("not_run_after_")
            and reason not in RUNTIME_FAILURE_REASONS,
        )
    )
    output.update(
        result_metric_semantics(
            prediction_returned=prediction_returned,
            failure_reason=reason,
            error_type=output.get("error_type"),
            mask_contract_valid=mask_contract_valid,
            predicted_foreground_pixels=int(
                output.get("predicted_foreground_pixels", 0)
            ),
            intersection_pixels=int(output.get("intersection_pixels", 0)),
        )
    )
    output["segmentation_token_present"] = bool(
        output.get("segmentation_token_present", False)
    )
    output["inference_success_legacy_deprecated"] = True
    return output


def validate_benchmark_split(split: str) -> str:
    """Enforce the validation-only boundary and name frozen splits explicitly."""
    if split == "test":
        raise ValueError("synthetic_v1 test is frozen and forbidden in Phase 3B")
    if split == "train":
        raise ValueError("synthetic_v1 train is forbidden in Phase 3B")
    if split != "val":
        raise ValueError("Phase 3B only permits split='val'")
    return split


def validate_prompt_set(
    variants: Sequence[str] = PROMPT_VARIANT_ORDER,
) -> tuple[str, ...]:
    """Require exactly the existing P0/P1/P2 registry in canonical order."""
    received = tuple(variants)
    if received != PROMPT_VARIANT_ORDER:
        raise ValueError(
            "Phase 3B requires exactly the canonical Prompt order: "
            + ",".join(PROMPT_VARIANT_ORDER)
        )
    return received


def validate_balanced_val_annotations(
    annotations: Sequence[dict[str, Any]],
    *,
    split: str = "val",
    expected_samples: int = EXPECTED_VAL_SAMPLES,
) -> list[dict[str, Any]]:
    """Select val without inspecting another split's model outputs and audit balance."""
    validate_benchmark_split(split)
    selected = [row for row in annotations if row.get("split") == "val"]
    if len(selected) != expected_samples:
        raise SplitEvaluationError(
            f"expected {expected_samples} synthetic_v1 val samples, found {len(selected)}"
        )
    combinations = Counter(
        (row.get("chart_type"), row.get("referring_type")) for row in selected
    )
    expected_combinations = {
        (chart, referring) for chart in CHART_TYPES for referring in REFERRING_TYPES
    }
    missing = sorted(expected_combinations - set(combinations))
    extra = sorted(set(combinations) - expected_combinations)
    if missing or extra:
        raise SplitEvaluationError(
            f"val chart/referring coverage mismatch; missing={missing}, extra={extra}"
        )
    wrong_counts = {
        f"{chart}/{referring}": combinations[(chart, referring)]
        for chart, referring in sorted(expected_combinations)
        if combinations[(chart, referring)] != 4
    }
    if wrong_counts:
        raise SplitEvaluationError(
            f"each val chart/referring group must contain 4 samples: {wrong_counts}"
        )
    for combination in sorted(expected_combinations):
        actions = Counter(
            row.get("edit_action")
            for row in selected
            if (row.get("chart_type"), row.get("referring_type")) == combination
        )
        expected_actions = Counter({action: 1 for action in EDIT_ACTIONS})
        if actions != expected_actions:
            raise SplitEvaluationError(
                f"val group {combination[0]}/{combination[1]} must contain each "
                f"edit action once; got {dict(actions)}"
            )
    sample_ids = [str(row.get("sample_id")) for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise SplitEvaluationError("val sample_id values must be unique")
    return selected


def rotated_prompt_order(
    sample_index: int,
    variants: Sequence[str] = PROMPT_VARIANT_ORDER,
) -> tuple[str, ...]:
    """Rotate P0/P1/P2 left by sample_index modulo three."""
    canonical = validate_prompt_set(variants)
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    offset = sample_index % len(canonical)
    return canonical[offset:] + canonical[:offset]


def build_attempt_plan(
    annotations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return exactly 192 sample-major attempts with balanced order positions."""
    selected = validate_balanced_val_annotations(annotations)
    plan: list[dict[str, Any]] = []
    for sample_index, annotation in enumerate(selected):
        for position, variant in enumerate(rotated_prompt_order(sample_index)):
            plan.append(
                {
                    "sample_index": sample_index,
                    "sample_id": annotation["sample_id"],
                    "prompt_variant": variant,
                    "prompt_order_position": position,
                }
            )
    if len(plan) != EXPECTED_ATTEMPTS:
        raise AssertionError(f"expected {EXPECTED_ATTEMPTS} attempts, got {len(plan)}")
    return plan


def prompt_template_hashes() -> dict[str, Any]:
    """Hash the imported Phase 2C templates without duplicating their strings."""
    templates = {name: PROMPT_TEMPLATES[name] for name in PROMPT_VARIANT_ORDER}
    encoded_registry = json.dumps(
        templates, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "variants": {
            name: hashlib.sha256(templates[name].encode("utf-8")).hexdigest()
            for name in PROMPT_VARIANT_ORDER
        },
        "registry_sha256": hashlib.sha256(encoded_registry).hexdigest(),
    }


def validate_result_matrix(
    results: Sequence[dict[str, Any]],
    *,
    expected_samples: int = EXPECTED_VAL_SAMPLES,
) -> None:
    """Ensure every val sample has exactly one P0/P1/P2 scalar record."""
    rows = list(results)
    expected_attempts = expected_samples * len(PROMPT_VARIANT_ORDER)
    if len(rows) != expected_attempts:
        raise SplitEvaluationError(
            f"expected {expected_attempts} result records, found {len(rows)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "val":
            raise SplitEvaluationError("result matrix contains a non-val record")
        if row.get("prompt_variant") not in PROMPT_VARIANT_ORDER:
            raise SplitEvaluationError(
                f"unregistered Prompt variant: {row.get('prompt_variant')!r}"
            )
        grouped[str(row.get("sample_id"))].append(row)
    if len(grouped) != expected_samples:
        raise SplitEvaluationError(
            f"expected {expected_samples} unique sample IDs, found {len(grouped)}"
        )
    for sample_id, sample_rows in grouped.items():
        variants = Counter(row["prompt_variant"] for row in sample_rows)
        if variants != Counter({variant: 1 for variant in PROMPT_VARIANT_ORDER}):
            raise SplitEvaluationError(
                f"sample {sample_id!r} does not have exactly one P0/P1/P2 result"
            )


def summarize_benchmark(
    results: Sequence[dict[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Compute all preregistered metrics, paired comparisons, and selection."""
    rows = [ensure_result_metric_semantics(row) for row in results]
    validate_result_matrix(rows)
    variants: dict[str, dict[str, Any]] = {}
    for variant in PROMPT_VARIANT_ORDER:
        selected = [row for row in rows if row["prompt_variant"] == variant]
        grouped = {
            "chart_referring": _summaries_by(
                selected, lambda row: f"{row['chart_type']}/{row['referring_type']}"
            ),
            "chart_type": _summaries_by(selected, lambda row: str(row["chart_type"])),
            "referring_type": _summaries_by(
                selected, lambda row: str(row["referring_type"])
            ),
            "edit_action": _summaries_by(
                selected, lambda row: str(row["edit_action"])
            ),
            "difficulty": _summaries_by(
                selected, lambda row: str(row["difficulty"])
            ),
            "distractor_count": _summaries_by(
                selected, lambda row: str(row["distractor_count"])
            ),
        }
        base = _aggregate_rows(selected)
        group_values = list(grouped["chart_referring"].items())
        base.update(
            {
                "group_macro_iou": statistics.fmean(
                    value["mean_iou"] for _, value in group_values
                ),
                "group_macro_dice": statistics.fmean(
                    value["mean_dice"] for _, value in group_values
                ),
                "worst_group": min(
                    group_values,
                    key=lambda item: (item[1]["mean_iou"], item[0]),
                )[0],
                "worst_group_mean_iou": min(
                    value["mean_iou"] for _, value in group_values
                ),
                "groups": grouped,
                "worst_10_samples": [
                    {
                        "sample_id": row["sample_id"],
                        "chart_type": row["chart_type"],
                        "referring_type": row["referring_type"],
                        "edit_action": row["edit_action"],
                        "difficulty": row["difficulty"],
                        "iou": row["iou"],
                        "dice": row["dice"],
                        "failure_reason": row.get("failure_reason"),
                        "empty_prediction": row["empty_prediction"],
                        "nonempty_disjoint": row["nonempty_disjoint"],
                    }
                    for row in sorted(
                        selected,
                        key=lambda item: (
                            float(item["iou"]),
                            float(item["dice"]),
                            item["sample_id"],
                        ),
                    )[:10]
                ],
            }
        )
        variants[variant] = base

    selection = select_global_prompt(variants)
    paired = _paired_analysis(rows)
    selected_variant = selection["selected_prompt"]
    selection["paired_bootstrap"] = paired_bootstrap_group_differences(
        variants,
        selected_variant=selected_variant,
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )
    ranked = sorted(
        PROMPT_VARIANT_ORDER,
        key=lambda name: variants[name]["group_macro_iou"],
        reverse=True,
    )
    selection["best_minus_second_group_macro_iou"] = (
        variants[ranked[0]]["group_macro_iou"]
        - variants[ranked[1]]["group_macro_iou"]
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
            "inference_success": (
                "deprecated legacy field; retained with its historical value and "
                "never displayed as execution success"
            ),
        },
        "attempt_count": len(rows),
        "sample_count": len({row["sample_id"] for row in rows}),
        "prompt_variants": list(PROMPT_VARIANT_ORDER),
        "variant_summaries": variants,
        "paired_analysis": paired,
        "selection": selection,
        "p0_vs_p2_chart_type_delta": {
            chart: {
                "p2_minus_p0_mean_iou": (
                    variants["target_only_zh"]["groups"]["chart_type"][chart][
                        "mean_iou"
                    ]
                    - variants["full_instruction"]["groups"]["chart_type"][chart][
                        "mean_iou"
                    ]
                ),
                "p2_minus_p0_mean_dice": (
                    variants["target_only_zh"]["groups"]["chart_type"][chart][
                        "mean_dice"
                    ]
                    - variants["full_instruction"]["groups"]["chart_type"][chart][
                        "mean_dice"
                    ]
                ),
            }
            for chart in ("line", "scatter", "confidence_band")
        },
        "full_instruction_action_deltas": {
            action: {
                "p0_minus_p1_mean_iou": (
                    variants["full_instruction"]["groups"]["edit_action"][action][
                        "mean_iou"
                    ]
                    - variants["target_only_en"]["groups"]["edit_action"][action][
                        "mean_iou"
                    ]
                ),
                "p0_minus_p2_mean_iou": (
                    variants["full_instruction"]["groups"]["edit_action"][action][
                        "mean_iou"
                    ]
                    - variants["target_only_zh"]["groups"]["edit_action"][action][
                        "mean_iou"
                    ]
                ),
            }
            for action in EDIT_ACTIONS
        },
        "per_group_oracle_analysis_only": {
            group: max(
                PROMPT_VARIANT_ORDER,
                key=lambda variant: variants[variant]["groups"]["chart_referring"][
                    group
                ]["mean_iou"],
            )
            for group in variants["full_instruction"]["groups"][
                "chart_referring"
            ]
        },
    }


def select_global_prompt(
    variant_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered global selection hierarchy exactly."""
    validate_prompt_set(tuple(variant_summaries))
    candidates = list(PROMPT_VARIANT_ORDER)
    best_iou = max(round(float(variant_summaries[name]["group_macro_iou"]), 6) for name in candidates)
    candidates = [
        name
        for name in candidates
        if round(float(variant_summaries[name]["group_macro_iou"]), 6) == best_iou
    ]
    rule = "highest_16_group_macro_iou"
    if len(candidates) > 1:
        best_dice = max(
            round(float(variant_summaries[name]["group_macro_dice"]), 6)
            for name in candidates
        )
        candidates = [
            name
            for name in candidates
            if round(float(variant_summaries[name]["group_macro_dice"]), 6)
            == best_dice
        ]
        rule = "group_macro_dice_tiebreak"
    if len(candidates) > 1:
        best_disjoint = min(
            round(float(variant_summaries[name]["nonempty_disjoint_rate"]), 6)
            for name in candidates
        )
        candidates = [
            name
            for name in candidates
            if round(float(variant_summaries[name]["nonempty_disjoint_rate"]), 6)
            == best_disjoint
        ]
        rule = "nonempty_disjoint_rate_tiebreak"
    if len(candidates) > 1:
        selected = (
            "full_instruction"
            if "full_instruction" in candidates
            else next(name for name in PROMPT_VARIANT_ORDER if name in candidates)
        )
        rule = "p0_or_canonical_final_tiebreak"
    else:
        selected = candidates[0]
    return {
        "selected_prompt": selected,
        "selection_rule_applied": rule,
        "remaining_candidates": candidates,
        "selected_on_validation": True,
        "latency_used_for_selection": False,
    }


def paired_bootstrap_group_differences(
    variant_summaries: dict[str, dict[str, Any]],
    *,
    selected_variant: str,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Bootstrap 16 paired group means for selected-minus-other IoU."""
    if selected_variant not in PROMPT_VARIANT_ORDER:
        raise ValueError("selected_variant is not registered")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    group_names = sorted(
        variant_summaries[selected_variant]["groups"]["chart_referring"]
    )
    if len(group_names) != 16:
        raise SplitEvaluationError(
            f"paired bootstrap requires 16 groups, found {len(group_names)}"
        )
    selected = np.asarray(
        [
            variant_summaries[selected_variant]["groups"]["chart_referring"][name][
                "mean_iou"
            ]
            for name in group_names
        ],
        dtype=np.float64,
    )
    output: dict[str, Any] = {}
    for other in PROMPT_VARIANT_ORDER:
        if other == selected_variant:
            continue
        comparison = np.asarray(
            [
                variant_summaries[other]["groups"]["chart_referring"][name][
                    "mean_iou"
                ]
                for name in group_names
            ],
            dtype=np.float64,
        )
        differences = selected - comparison
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(group_names), size=(iterations, len(group_names)))
        bootstrap = differences[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap, (0.025, 0.975))
        output[other] = {
            "selected_minus_other_group_macro_iou": float(differences.mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "ci_contains_zero": bool(lower <= 0.0 <= upper),
            "iterations": iterations,
            "seed": seed,
            "unit": "chart_type_x_referring_type_group",
        }
    return output


def select_gallery_sample_ids(
    annotations: Sequence[dict[str, Any]],
) -> list[str]:
    """Select lexicographically smallest val sample in each of 16 groups."""
    selected = validate_balanced_val_annotations(annotations)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in selected:
        grouped[(row["chart_type"], row["referring_type"])].append(row["sample_id"])
    return [
        min(grouped[(chart, referring)])
        for chart in CHART_TYPES
        for referring in REFERRING_TYPES
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise SplitEvaluationError("cannot aggregate an empty row group")
    ious = [float(row["iou"]) for row in rows]
    dices = [float(row["dice"]) for row in rows]
    times = [
        float(row["inference_time_ms"])
        for row in rows
        if row.get("inference_time_ms") is not None
    ]
    intersection = sum(int(row["intersection_pixels"]) for row in rows)
    union = sum(int(row["union_pixels"]) for row in rows)
    predicted = sum(int(row["predicted_foreground_pixels"]) for row in rows)
    target = sum(int(row["gt_foreground_pixels"]) for row in rows)
    legacy_successful = sum(bool(row["inference_success"]) for row in rows)
    execution_successful = sum(bool(row["execution_success"]) for row in rows)
    contract_valid = sum(bool(row["mask_contract_valid"]) for row in rows)
    seg = sum(bool(row["segmentation_token_present"]) for row in rows)
    nonempty = sum(bool(row["nonempty_prediction"]) for row in rows)
    empty = sum(bool(row["empty_prediction"]) for row in rows)
    disjoint = sum(bool(row["nonempty_disjoint"]) for row in rows)
    overlap = sum(bool(row["overlapping_prediction"]) for row in rows)
    return {
        "attempt_count": count,
        "execution_success_count": execution_successful,
        "execution_success_rate": execution_successful / count,
        "mask_contract_valid_count": contract_valid,
        "mask_contract_valid_rate": contract_valid / count,
        "segmentation_token_count": seg,
        "segmentation_token_rate": seg / count,
        "nonempty_prediction_count": nonempty,
        "nonempty_prediction_rate": nonempty / count,
        "empty_prediction_count": empty,
        "empty_prediction_rate": empty / count,
        "nonempty_disjoint_count": disjoint,
        "nonempty_disjoint_rate": disjoint / count,
        "overlapping_prediction_count": overlap,
        "overlapping_prediction_rate": overlap / count,
        "overlap_count": overlap,
        "overlap_rate": overlap / count,
        "successful_inference_count": legacy_successful,
        "inference_success_rate": legacy_successful / count,
        "inference_success_legacy_deprecated": True,
        "sample_macro_iou": statistics.fmean(ious),
        "sample_macro_dice": statistics.fmean(dices),
        "median_iou": statistics.median(ious),
        "median_dice": statistics.median(dices),
        "worst_sample_iou": min(ious),
        "micro_iou": intersection / union if union else 0.0,
        "micro_dice": 2.0 * intersection / (predicted + target)
        if predicted + target
        else 0.0,
        "mean_inference_time_ms": statistics.fmean(times) if times else None,
        "median_inference_time_ms": statistics.median(times) if times else None,
        "timed_attempt_count": len(times),
        "total_intersection_pixels": intersection,
        "total_union_pixels": union,
        "total_predicted_foreground_pixels": predicted,
        "total_gt_foreground_pixels": target,
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


def _paired_analysis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["sample_id"]][row["prompt_variant"]] = row
    paired_rows: list[dict[str, Any]] = []
    wins = Counter({variant: 0 for variant in PROMPT_VARIANT_ORDER})
    fully_identical = 0
    iou_ties = 0
    for sample_id in sorted(grouped):
        variants = grouped[sample_id]
        ious = {name: float(variants[name]["iou"]) for name in PROMPT_VARIANT_ORDER}
        best = max(ious.values())
        winners = [name for name, value in ious.items() if value == best]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            iou_ties += 1
        mask_hashes = {
            variants[name].get("predicted_mask_sha256") for name in PROMPT_VARIANT_ORDER
        }
        if len(mask_hashes) == 1:
            fully_identical += 1
        baseline = variants["full_instruction"]
        paired_rows.append(
            {
                "sample_id": sample_id,
                "chart_type": baseline["chart_type"],
                "referring_type": baseline["referring_type"],
                "edit_action": baseline["edit_action"],
                "difficulty": baseline["difficulty"],
                "distractor_count": baseline["distractor_count"],
                "metrics": {
                    name: {
                        "iou": variants[name]["iou"],
                        "dice": variants[name]["dice"],
                        "predicted_foreground_pixels": variants[name][
                            "predicted_foreground_pixels"
                        ],
                    }
                    for name in PROMPT_VARIANT_ORDER
                },
                "delta_from_p0": {
                    name: {
                        "iou": float(variants[name]["iou"])
                        - float(baseline["iou"]),
                        "dice": float(variants[name]["dice"])
                        - float(baseline["dice"]),
                        "predicted_foreground_pixels": int(
                            variants[name]["predicted_foreground_pixels"]
                        )
                        - int(baseline["predicted_foreground_pixels"]),
                    }
                    for name in PROMPT_VARIANT_ORDER[1:]
                },
                "iou_winners": winners,
                "prediction_masks_all_identical": len(mask_hashes) == 1,
            }
        )
    return {
        "sample_wins_by_iou": dict(wins),
        "iou_tie_sample_count": iou_ties,
        "fully_identical_prediction_sample_count": fully_identical,
        "samples": paired_rows,
    }
