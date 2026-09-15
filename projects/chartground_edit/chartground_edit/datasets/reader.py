"""Minimal reader for ChartGround-Edit JSONL manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

from .schema import validate_jsonl


@dataclass(frozen=True)
class ChartGroundSample:
    """One loaded image/mask pair and its validated annotation."""

    annotation: dict[str, Any]
    image: Image.Image
    mask: Image.Image


class ChartGroundDataset(Sequence[ChartGroundSample]):
    """Validated, dependency-light dataset reader with lazy image loading."""

    def __init__(self, manifest_path: str | Path, *, validate_files: bool = True):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.annotations = validate_jsonl(
            self.manifest_path, check_files=validate_files
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int | slice) -> ChartGroundSample | list[ChartGroundSample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        annotation = self.annotations[index]
        with Image.open(self._path(annotation["image_path"])) as image_file:
            image = image_file.convert("RGB").copy()
        with Image.open(self._path(annotation["mask_path"])) as mask_file:
            mask = mask_file.copy()
        return ChartGroundSample(dict(annotation), image, mask)

    def __iter__(self) -> Iterator[ChartGroundSample]:
        for index in range(len(self)):
            yield self[index]

