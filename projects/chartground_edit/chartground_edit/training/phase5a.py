"""Frozen Phase 5A validation aggregation and checkpoint selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from .overfit32 import summarize_checkpoint_rows


CHECKPOINT_NAMES = (
    "baseline",
    "step192",
    "step576",
    "step960",
    "step1344",
    "step1920",
)
EXPECTED_BASELINE = {
    "group_macro_iou": 0.18219856304270052,
    "group_macro_dice": 0.21622086057617812,
    "empty_rate": 0.4375,
    "nonempty_count": 36,
}


def validate_phase3b_baseline(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stop before trained checkpoints unless frozen P2 baseline reproduces."""
    summary = summarize_checkpoint_rows(
        rows, expected_sample_count=64, expected_per_group=4
    )
    mismatches = {}
    for name in ("group_macro_iou", "group_macro_dice", "empty_rate"):
        if not np.isclose(summary[name], EXPECTED_BASELINE[name], rtol=0.0, atol=1e-9):
            mismatches[name] = {
                "actual": summary[name],
                "expected": EXPECTED_BASELINE[name],
            }
    if summary["nonempty_count"] != EXPECTED_BASELINE["nonempty_count"]:
        mismatches["nonempty_count"] = {
            "actual": summary["nonempty_count"],
            "expected": EXPECTED_BASELINE["nonempty_count"],
        }
    if mismatches:
        raise ValueError(f"Phase 3B P2 baseline reproduction failed: {mismatches}")
    return summary


def summarize_phase5a_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate val-only checkpoints and apply the preregistered selection rule."""
    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "val":
            raise ValueError("Phase 5A metrics must be val-only")
        by_checkpoint[str(row["checkpoint"])].append(row)
    if tuple(by_checkpoint) != CHECKPOINT_NAMES or set(by_checkpoint) != set(
        CHECKPOINT_NAMES
    ):
        raise ValueError(
            f"expected checkpoint order {CHECKPOINT_NAMES}, got {tuple(by_checkpoint)}"
        )

    summaries = {}
    for name in CHECKPOINT_NAMES:
        summary = summarize_checkpoint_rows(
            by_checkpoint[name], expected_sample_count=64, expected_per_group=4
        )
        chart_referring = summary.pop("groups")
        summary["groups"] = {
            "chart_referring": chart_referring,
            "chart_type": _summaries_by(by_checkpoint[name], "chart_type"),
            "referring_type": _summaries_by(by_checkpoint[name], "referring_type"),
            "difficulty": _summaries_by(by_checkpoint[name], "difficulty"),
        }
        summaries[name] = summary

    baseline = summaries["baseline"]
    for summary in summaries.values():
        summary["delta_vs_baseline"] = {
            metric: summary[metric] - baseline[metric]
            for metric in (
                "group_macro_iou",
                "group_macro_dice",
                "sample_macro_iou",
                "sample_macro_dice",
                "micro_iou",
                "micro_dice",
                "empty_rate",
            )
        }

    trained = CHECKPOINT_NAMES[1:]
    best = max(
        trained,
        key=lambda name: (
            summaries[name]["group_macro_iou"],
            summaries[name]["group_macro_dice"],
            -summaries[name]["empty_rate"],
            -trained.index(name),
        ),
    )
    baseline_groups = baseline["groups"]["chart_referring"]
    best_groups = summaries[best]["groups"]["chart_referring"]
    improved = [
        name
        for name in sorted(baseline_groups)
        if best_groups[name]["mean_iou"] > baseline_groups[name]["mean_iou"]
    ]
    declined = [
        name
        for name in sorted(baseline_groups)
        if best_groups[name]["mean_iou"] < baseline_groups[name]["mean_iou"]
    ]
    tied = sorted(set(baseline_groups) - set(improved) - set(declined))
    return {
        "checkpoints": summaries,
        "best_checkpoint": best,
        "selection_rule": [
            "16-group Macro IoU",
            "Macro Dice",
            "lower empty rate",
            "earlier checkpoint",
        ],
        "best_group_comparison": {
            "improved": improved,
            "declined": declined,
            "tied": tied,
        },
    }


def _summaries_by(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        name: {
            "count": len(items),
            "mean_iou": float(np.mean([item["iou"] for item in items])),
            "mean_dice": float(np.mean([item["dice"] for item in items])),
            "empty_rate": float(np.mean([item["empty_prediction"] for item in items])),
            "nonempty_disjoint_rate": float(
                np.mean([item["nonempty_disjoint"] for item in items])
            ),
        }
        for name, items in sorted(grouped.items())
    }
