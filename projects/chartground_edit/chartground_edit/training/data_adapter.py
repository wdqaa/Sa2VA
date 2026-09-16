"""Pure-data Phase 4 contract; this module never imports or builds Sa2VA."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from chartground_edit.datasets.schema_v1 import validate_jsonl_v1
from chartground_edit.inference.prompt_variants import TARGET_ONLY_ZH, build_prompt_variant


EXPECTED_MANIFEST_SHA256 = (
    "ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82"
)
ASSISTANT_TARGET = "Sure, [SEG]."
AUDIT_ONLY_FIELDS = frozenset(
    {
        "chart_type",
        "referring_type",
        "edit_action",
        "edit_parameters",
        "target_attributes",
        "difficulty",
        "distractor_count",
        "scene_id",
        "content_id",
    }
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_output_dir(
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    data_root: str | Path,
) -> Path:
    """Reject locations that could overwrite source or versioned dataset files."""
    output = Path(output_dir).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    if output == repo:
        raise ValueError("training output directory must not be the repository root")
    if output == data or data in output.parents:
        raise ValueError("training output directory must not be inside the data directory")
    return output


@dataclass(frozen=True)
class Phase4DataSample:
    """The fields permitted at the future model boundary."""

    sample_id: str
    image: Image.Image
    mask: np.ndarray
    image_width: int
    image_height: int
    prompt: str
    assistant_target: str

    @property
    def conversation(self) -> tuple[dict[str, str], dict[str, str]]:
        return (
            {"from": "human", "value": self.prompt},
            {"from": "gpt", "value": self.assistant_target},
        )


class Phase4TrainDataset(Sequence[Phase4DataSample]):
    """Read a frozen train-only ID list without tokenizer or model construction."""

    def __init__(self, manifest_path: str | Path, selection_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.selection_path = Path(selection_path)
        manifest_hash = file_sha256(self.manifest_path)
        if manifest_hash != EXPECTED_MANIFEST_SHA256:
            raise ValueError(
                "manifest SHA-256 mismatch: "
                f"expected {EXPECTED_MANIFEST_SHA256}, got {manifest_hash}"
            )
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        if selection.get("manifest_sha256") != manifest_hash:
            raise ValueError("selection manifest hash does not match source manifest")
        if selection.get("split") != "train":
            raise ValueError("Phase 4 selections must be train-only")
        ids = selection.get("sample_ids")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ValueError("selection sample_ids must be a non-empty unique list")

        records = validate_jsonl_v1(self.manifest_path, check_files=True)
        by_id = {record["sample_id"]: record for record in records}
        missing = sorted(set(ids) - set(by_id))
        if missing:
            raise ValueError(f"selection contains unknown sample IDs: {missing}")
        self.records: list[dict[str, Any]] = [by_id[sample_id] for sample_id in ids]
        if any(record["split"] != "train" for record in self.records):
            raise ValueError("selection contains a non-train sample")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | slice) -> Phase4DataSample | list[Phase4DataSample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        record = self.records[index]
        root = self.manifest_path.parent
        with Image.open(root / record["image_path"]) as source:
            image = source.convert("RGB").copy()
        with Image.open(root / record["mask_path"]) as source:
            raw_mask = np.asarray(source.convert("L"))
        mask = (raw_mask == 255).astype(np.uint8, copy=False)[None, ...]
        prompt = build_prompt_variant(
            TARGET_ONLY_ZH,
            instruction=record["full_instruction"],
            referring_expression=record["referring_expression"],
        )
        return Phase4DataSample(
            sample_id=record["sample_id"],
            image=image,
            mask=mask,
            image_width=record["image_width"],
            image_height=record["image_height"],
            prompt=prompt,
            assistant_target=ASSISTANT_TARGET,
        )
