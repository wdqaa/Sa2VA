"""Dependency-light reader for ChartGround-Edit JSONL v1 manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

from .schema_v1 import validate_jsonl_v1


@dataclass(frozen=True)
class ChartGroundV1Sample:
    annotation: dict[str, Any]
    image: Image.Image
    mask: Image.Image

    @property
    def full_instruction(self) -> str:
        return self.annotation["full_instruction"]

    @property
    def referring_expression(self) -> str:
        return self.annotation["referring_expression"]


class ChartGroundV1Dataset(Sequence[ChartGroundV1Sample]):
    """Validated v1 reader; referring expressions are read, never reconstructed."""

    def __init__(self, manifest_path: str | Path, *, validate_files: bool = True):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.annotations = validate_jsonl_v1(
            self.manifest_path, check_files=validate_files
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(
        self, index: int | slice
    ) -> ChartGroundV1Sample | list[ChartGroundV1Sample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        annotation = self.annotations[index]
        with Image.open(self.root / annotation["image_path"]) as source:
            image = source.convert("RGB").copy()
        with Image.open(self.root / annotation["mask_path"]) as source:
            mask = source.copy()
        return ChartGroundV1Sample(dict(annotation), image, mask)

    def __iter__(self) -> Iterator[ChartGroundV1Sample]:
        for index in range(len(self)):
            yield self[index]
