"""Official-preprocessing bridge for the frozen ChartGround Phase 4 samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from projects.sa2va.datasets.base import Sa2VABaseDataset
from projects.sa2va.datasets.data_utils import sa2va_collect_fn

from .alignment import (
    EXPECTED_MASKS_PER_SAMPLE,
    IGNORE_INDEX,
    OBJECT_COUNT_POLICY,
    select_supervised_seg_tokens,
    validate_strict_one_to_one,
)
from .data_adapter import Phase4TrainDataset


class ChartGroundPhase4Dataset(Sa2VABaseDataset):
    """Train-only adapter using Sa2VA's real tokenizer and image preprocessors."""

    def __init__(
        self,
        manifest_path: str | Path,
        selection_path: str | Path,
        tokenizer,
        prompt_template,
        special_tokens=None,
        extra_image_processor=None,
        max_length: int = 8192,
        arch_type: str = "intern_vl",
        single_image_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        self.source = Phase4TrainDataset(manifest_path, selection_path)
        self.single_image_mode = single_image_mode
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            max_length=max_length,
            arch_type=arch_type,
            repeats=1,
            name="ChartGroundPhase4Smoke1",
            **kwargs,
        )
        self.seg_token_idx = self.tokenizer(
            "[SEG]", add_special_tokens=False
        ).input_ids[0]

    def real_len(self) -> int:
        return len(self.source)

    @property
    def modality_length(self) -> list[int]:
        return [self._get_modality_length_default() for _ in range(len(self))]

    def prepare_data(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        if sample.image.size != (sample.image_width, sample.image_height):
            raise ValueError(f"{sample.sample_id}: image metadata size mismatch")
        if sample.mask.shape != (1, sample.image_height, sample.image_width):
            raise ValueError(f"{sample.sample_id}: GT mask must be [1,H,W]")
        if sample.mask.dtype != np.uint8 or not np.isin(sample.mask, (0, 1)).all():
            raise ValueError(f"{sample.sample_id}: GT mask must be binary uint8")
        if not np.any(sample.mask):
            raise ValueError(f"{sample.sample_id}: GT mask must be non-empty")

        image_data = self._process_single_image(
            sample.image, self.single_image_mode
        )
        image_token_str = self._create_image_token_string(
            image_data["num_image_tokens"]
        )
        conversation = self._process_conversations_for_encoding(
            list(sample.conversation), image_token_str
        )
        token_dict = self.get_inputid_labels(conversation)
        return {
            **image_data,
            **token_dict,
            "masks": torch.from_numpy(sample.mask.copy()),
            "sample_id": sample.sample_id,
            "seg_token_idx": self.seg_token_idx,
            "original_size": (sample.image_height, sample.image_width),
        }


def chartground_sa2va_collect_fn(
    instances: Sequence[dict[str, Any]],
    *,
    ignore_index: int = IGNORE_INDEX,
    object_count_policy: str = OBJECT_COUNT_POLICY,
    expected_masks_per_sample: int = EXPECTED_MASKS_PER_SAMPLE,
) -> dict[str, Any]:
    """Use the upstream collator, retain IDs, and fail closed on misalignment."""
    if not instances:
        raise ValueError("cannot collate an empty ChartGround batch")
    sample_ids = [str(instance["sample_id"]) for instance in instances]
    seg_token_ids = {int(instance["seg_token_idx"]) for instance in instances}
    if len(seg_token_ids) != 1:
        raise ValueError(f"batch has inconsistent [SEG] token IDs: {seg_token_ids}")
    upstream_instances = [
        {
            key: value
            for key, value in instance.items()
            if key not in {"sample_id", "seg_token_idx", "original_size"}
        }
        for instance in instances
    ]
    batch = sa2va_collect_fn(upstream_instances)
    data = batch["data"]
    data["sample_ids"] = sample_ids
    supervised_mask = select_supervised_seg_tokens(
        data["input_ids"],
        data["labels"],
        seg_token_idx=seg_token_ids.pop(),
        ignore_index=ignore_index,
    )
    data["alignment_records"] = validate_strict_one_to_one(
        supervised_mask,
        data["masks"],
        sample_ids,
        expected_masks_per_sample=expected_masks_per_sample,
        object_count_policy=object_count_policy,
    )
    return batch

