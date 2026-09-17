#!/usr/bin/env python3
"""Reproduce Phase 4B token/mask alignment without constructing a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from xtuner.utils import PROMPT_TEMPLATE

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "projects" / "chartground_edit"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.training.alignment import select_supervised_seg_tokens
from chartground_edit.training.sa2va_adapter import (
    ChartGroundPhase4Dataset,
    chartground_sa2va_collect_fn,
)
from projects.sa2va.models import DirectResize
from projects.sa2va.models.sa2va import Sa2VAModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args(argv)

    tokenizer_cfg = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=str(args.checkpoint),
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    dataset = ChartGroundPhase4Dataset(
        manifest_path=args.manifest,
        selection_path=args.selection,
        tokenizer=tokenizer_cfg,
        prompt_template=PROMPT_TEMPLATE.qwen_chat,
        special_tokens=["[SEG]", "<p>", "</p>", "<vp>", "</vp>"],
        extra_image_processor=dict(type=DirectResize, target_length=1024),
        max_length=8192,
        arch_type="intern_vl",
    )
    instance = dataset.prepare_data(0)
    batch = chartground_sa2va_collect_fn([instance])
    data = batch["data"]
    input_ids = data["input_ids"]
    labels = data["labels"]
    seg_token_idx = int(instance["seg_token_idx"])
    all_seg_mask = input_ids == seg_token_idx
    supervised_seg_mask = select_supervised_seg_tokens(
        input_ids, labels, seg_token_idx=seg_token_idx
    )
    all_positions = torch.nonzero(all_seg_mask[0], as_tuple=False).flatten().tolist()
    supervised_positions = torch.nonzero(
        supervised_seg_mask[0], as_tuple=False
    ).flatten().tolist()

    position_records = []
    for position in all_positions:
        label = int(labels[0, position])
        position_records.append(
            {
                "position": position,
                "role": "user" if label == -100 else "assistant",
                "label": label,
                "label_is_ignore_index": label == -100,
            }
        )

    before_embeddings = torch.arange(
        len(all_positions), dtype=torch.float32
    ).unsqueeze(-1)
    before_masks = data["masks"][0]
    fixed_embeddings, fixed_masks = Sa2VAModel.check_obj_number(
        None, [before_embeddings], [before_masks], fix_number=5
    )

    source_sample = dataset.source[0]
    report = {
        "sample_id": source_sample.sample_id,
        "prompt_sha256": hashlib.sha256(
            source_sample.prompt.encode("utf-8")
        ).hexdigest(),
        "assistant_target": source_sample.assistant_target,
        "seg_token_idx": seg_token_idx,
        "sequence_length": int(input_ids.shape[1]),
        "seg_positions": position_records,
        "legacy_selected_positions": all_positions,
        "labels_aware_selected_positions": supervised_positions,
        "gt_mask_count": int(before_masks.shape[0]),
        "legacy_fix_number_5": {
            "before": {
                "token_count": len(all_positions),
                "mask_count": int(before_masks.shape[0]),
            },
            "after": {
                "token_count": len(fixed_embeddings[0]),
                "mask_count": len(fixed_masks[0]),
            },
        },
        "attention_tokens": int(data["attention_mask"][0].sum()),
        "pixel_values_shape": list(instance["pixel_values"].shape),
        "grounding_pixel_values_shape": list(instance["g_pixel_values"].shape),
        "gt_mask_shape": list(before_masks.shape),
        "gt_mask_dtype": str(before_masks.dtype),
        "gt_mask_values": sorted(int(value) for value in torch.unique(before_masks)),
        "gt_mask_nonzero": int(torch.count_nonzero(before_masks)),
        "collator_alignment_records": data["alignment_records"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
