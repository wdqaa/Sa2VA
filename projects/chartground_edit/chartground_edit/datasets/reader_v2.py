"""Dependency-light reader for synthetic v2 without altering the v1 API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

from .schema_v2 import validate_jsonl_v2


@dataclass(frozen=True)
class ChartGroundV2Sample:
    annotation: dict[str, Any]
    image: Image.Image
    mask: Image.Image


class ChartGroundV2Dataset(Sequence[ChartGroundV2Sample]):
    def __init__(self, manifest_path: str | Path, *, validate_files: bool = True):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.annotations = validate_jsonl_v2(self.manifest_path, check_files=validate_files)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int | slice) -> ChartGroundV2Sample | list[ChartGroundV2Sample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        annotation = self.annotations[index]
        with Image.open(self.root / annotation["image_path"]) as source:
            image = source.convert("RGB").copy()
        with Image.open(self.root / annotation["mask_path"]) as source:
            mask = source.convert("L").copy()
        return ChartGroundV2Sample(dict(annotation), image, mask)

    def __iter__(self) -> Iterator[ChartGroundV2Sample]:
        for index in range(len(self)):
            yield self[index]
