"""Fixed Phase 4C schedule and train-set metric aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
import torch


EXPECTED_SAMPLE_COUNT = 32
EXPECTED_EPOCHS = 10
EXPECTED_STEPS = 320
CHECKPOINT_NAMES = ("baseline", "step32", "step128", "step320")


def build_epoch_schedule(
    sample_ids: Sequence[str],
    *,
    epochs: int,
    seed: int,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    if len(sample_ids) != expected_sample_count or len(set(sample_ids)) != len(
        sample_ids
    ):
        raise ValueError(
            f"projection training requires {expected_sample_count} unique sample IDs"
        )
    if epochs != EXPECTED_EPOCHS:
        raise ValueError("Phase 4C requires exactly 10 epochs")
    schedule = []
    for epoch_index in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch_index)
        for index in torch.randperm(len(sample_ids), generator=generator).tolist():
            schedule.append(
                {
                    "step": len(schedule) + 1,
                    "epoch": epoch_index + 1,
                    "sample_index": index,
                    "sample_id": sample_ids[index],
                }
            )
    counts = Counter(item["sample_id"] for item in schedule)
    expected_steps = expected_sample_count * epochs
    if len(schedule) != expected_steps or set(counts.values()) != {epochs}:
        raise RuntimeError(f"invalid projection-training schedule counts: {counts}")
    return schedule


def select_gallery_sample_ids(
    annotations: Sequence[dict[str, Any]], *, expected_per_group: int = 2
) -> list[str]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in annotations:
        groups[(row["chart_type"], row["referring_type"])].append(
            row["sample_id"]
        )
    if len(groups) != 16 or any(
        len(ids) != expected_per_group for ids in groups.values()
    ):
        raise ValueError(
            "gallery selection requires 16 groups with "
            f"{expected_per_group} samples each"
        )
    return [min(groups[group]) for group in sorted(groups)]


def summarize_overfit_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Phase 4C metrics must be train-only")
        by_checkpoint[row["checkpoint"]].append(row)
    if set(by_checkpoint) != set(CHECKPOINT_NAMES):
        raise ValueError(
            f"expected checkpoints {CHECKPOINT_NAMES}, got {sorted(by_checkpoint)}"
        )
    summaries = {
        name: summarize_checkpoint_rows(by_checkpoint[name])
        for name in CHECKPOINT_NAMES
    }
    baseline = summaries["baseline"]
    for name, summary in summaries.items():
        summary["delta_vs_baseline"] = {
            "group_macro_iou": summary["group_macro_iou"]
            - baseline["group_macro_iou"],
            "group_macro_dice": summary["group_macro_dice"]
            - baseline["group_macro_dice"],
            "sample_macro_iou": summary["sample_macro_iou"]
            - baseline["sample_macro_iou"],
            "sample_macro_dice": summary["sample_macro_dice"]
            - baseline["sample_macro_dice"],
        }
    trained = CHECKPOINT_NAMES[1:]
    best = max(
        trained,
        key=lambda name: (
            summaries[name]["group_macro_iou"],
            -trained.index(name),
        ),
    )
    return {"checkpoints": summaries, "best_checkpoint": best}


def summarize_checkpoint_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    expected_per_group: int = 2,
) -> dict[str, Any]:
    if len(rows) != expected_sample_count:
        raise ValueError(
            f"each checkpoint requires exactly {expected_sample_count} rows"
        )
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != expected_sample_count:
        raise ValueError("each sample must appear exactly once per checkpoint")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['chart_type']}/{row['referring_type']}"].append(row)
    if len(groups) != 16 or any(
        len(items) != expected_per_group for items in groups.values()
    ):
        raise ValueError(
            "checkpoint metrics require 16 groups with "
            f"{expected_per_group} samples each"
        )
    group_metrics = {}
    for name, items in sorted(groups.items()):
        group_metrics[name] = {
            "count": len(items),
            "mean_iou": float(np.mean([item["iou"] for item in items])),
            "mean_dice": float(np.mean([item["dice"] for item in items])),
            "empty_rate": float(np.mean([item["empty_prediction"] for item in items])),
        }
    intersection = sum(int(row["intersection_pixels"]) for row in rows)
    union = sum(int(row["union_pixels"]) for row in rows)
    pred = sum(int(row["predicted_foreground_pixels"]) for row in rows)
    gt = sum(int(row["gt_foreground_pixels"]) for row in rows)
    return {
        "sample_count": len(rows),
        "group_macro_iou": float(
            np.mean([group["mean_iou"] for group in group_metrics.values()])
        ),
        "group_macro_dice": float(
            np.mean([group["mean_dice"] for group in group_metrics.values()])
        ),
        "sample_macro_iou": float(np.mean([row["iou"] for row in rows])),
        "sample_macro_dice": float(np.mean([row["dice"] for row in rows])),
        "micro_iou": float(intersection / union) if union else 1.0,
        "micro_dice": float(2 * intersection / (pred + gt)) if pred + gt else 1.0,
        "empty_rate": float(np.mean([row["empty_prediction"] for row in rows])),
        "empty_count": sum(bool(row["empty_prediction"]) for row in rows),
        "nonempty_count": sum(not bool(row["empty_prediction"]) for row in rows),
        "nonempty_disjoint_rate": float(
            np.mean([row["nonempty_disjoint"] for row in rows])
        ),
        "overlap_rate": float(
            np.mean([row["overlapping_prediction"] for row in rows])
        ),
        "overlap_count": sum(bool(row["overlapping_prediction"]) for row in rows),
        "groups": group_metrics,
    }
