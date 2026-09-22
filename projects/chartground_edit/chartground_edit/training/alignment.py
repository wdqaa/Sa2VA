"""Strict token-to-mask alignment checks for ChartGround training batches."""

from __future__ import annotations

from typing import Any, Sequence

import torch


SEG_TOKEN_SELECTION = "supervised_labels"
OBJECT_COUNT_POLICY = "strict_one_to_one"
EXPECTED_MASKS_PER_SAMPLE = 1
IGNORE_INDEX = -100


def select_supervised_seg_tokens(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    seg_token_idx: int,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return ``[SEG]`` positions that are inside the supervised label span."""
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
        )
    if labels.shape != input_ids.shape:
        raise ValueError(
            "labels shape must match input_ids shape: "
            f"{tuple(labels.shape)} != {tuple(input_ids.shape)}"
        )
    if labels.device != input_ids.device:
        raise ValueError(
            "labels and input_ids must be on the same device: "
            f"{labels.device} != {input_ids.device}"
        )
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise TypeError(
            "labels and input_ids must both use torch.long: "
            f"labels={labels.dtype}, input_ids={input_ids.dtype}"
        )
    if not isinstance(seg_token_idx, int) or isinstance(seg_token_idx, bool):
        raise TypeError("seg_token_idx must be an int")
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise TypeError("ignore_index must be an int")
    return (input_ids == seg_token_idx) & (labels != ignore_index)


def validate_strict_one_to_one(
    supervised_seg_mask: torch.Tensor,
    gt_masks: Sequence[torch.Tensor],
    sample_ids: Sequence[str],
    *,
    expected_masks_per_sample: int = EXPECTED_MASKS_PER_SAMPLE,
    object_count_policy: str = OBJECT_COUNT_POLICY,
) -> list[dict[str, Any]]:
    """Validate every batch item without truncating, padding, or repeating it."""
    if object_count_policy != OBJECT_COUNT_POLICY:
        raise ValueError(
            f"unsupported ChartGround object_count_policy={object_count_policy!r}"
        )
    if expected_masks_per_sample != 1:
        raise ValueError("strict_one_to_one requires expected_masks_per_sample=1")
    if not isinstance(supervised_seg_mask, torch.Tensor):
        raise TypeError("supervised_seg_mask must be a torch.Tensor")
    if supervised_seg_mask.dtype != torch.bool or supervised_seg_mask.ndim != 2:
        raise TypeError("supervised_seg_mask must be bool [batch, sequence]")
    batch_size = supervised_seg_mask.shape[0]
    if len(gt_masks) != batch_size or len(sample_ids) != batch_size:
        raise ValueError(
            "strict alignment batch-size mismatch: "
            f"token_batch={batch_size}; mask_batch={len(gt_masks)}; "
            f"sample_id_batch={len(sample_ids)}; "
            f"object_count_policy={object_count_policy}"
        )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (sample_id, sample_mask) in enumerate(zip(sample_ids, gt_masks)):
        positions = torch.nonzero(
            supervised_seg_mask[index], as_tuple=False
        ).flatten().tolist()
        supervised_count = len(positions)
        gt_count: int | str
        if not isinstance(sample_mask, torch.Tensor) or sample_mask.ndim != 3:
            gt_count = "invalid"
        else:
            gt_count = int(sample_mask.shape[0])
        record = {
            "sample_id": str(sample_id),
            "supervised_seg_count": supervised_count,
            "gt_mask_count": gt_count,
            "token_positions": positions,
            "object_count_policy": object_count_policy,
        }
        records.append(record)
        if (
            supervised_count != expected_masks_per_sample
            or gt_count != expected_masks_per_sample
            or supervised_count != gt_count
        ):
            failures.append(record)
    if failures:
        details = "; ".join(
            "sample_id={sample_id}; supervised_seg_count={supervised_seg_count}; "
            "gt_mask_count={gt_mask_count}; token_positions={token_positions}; "
            "object_count_policy={object_count_policy}".format(**record)
            for record in failures
        )
        raise ValueError(f"strict alignment failed: {details}")
    return records
