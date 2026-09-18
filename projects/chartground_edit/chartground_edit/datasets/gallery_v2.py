"""High-resolution 16-combination synthetic v2 gallery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .schema_v2 import validate_jsonl_v2
from .synthetic_v2 import CHART_ORDER_V2, REFERRING_ORDER_V2


GALLERY_SIZE_V2 = (2560, 1920)
GALLERY_DPI_V2 = 200


def select_gallery_records_v2(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for index, (chart, referring) in enumerate(
        (chart, referring) for chart in CHART_ORDER_V2 for referring in REFERRING_ORDER_V2
    ):
        candidates = sorted(
            [row for row in records if row["split"] == "train" and row["chart_type"] == chart and row["referring_type"] == referring],
            key=lambda row: row["sample_id"],
        )
        if not candidates:
            raise ValueError(f"missing gallery group {chart}/{referring}")
        selected.append(candidates[(index * 7) % len(candidates)])
    return selected


def create_synthetic_v2_gallery(manifest_path: str | Path, output_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    records = validate_jsonl_v2(manifest, check_files=True, expected_count=1600)
    selected = select_gallery_records_v2(records)
    sheet = Image.new("RGB", GALLERY_SIZE_V2, (238, 240, 243))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(27)
    body_font = _font(19)
    small_font = _font(15)
    cell_width, cell_height = 640, 480
    for index, row in enumerate(selected):
        grid_row, column = divmod(index, 4)
        x0, y0 = column * cell_width, grid_row * cell_height
        with Image.open(manifest.parent / row["image_path"]) as source:
            image = source.convert("RGB").copy()
        with Image.open(manifest.parent / row["mask_path"]) as source:
            mask = source.convert("L").copy()
        overlay = _overlay(image, mask)
        sheet.paste(_fit(image, (300, 330)), (x0 + 15, y0 + 92))
        sheet.paste(_fit(overlay, (300, 330)), (x0 + 325, y0 + 92))
        draw.text((x0 + 16, y0 + 13), f"{row['chart_type']} / {row['referring_type']}", fill=(22, 31, 43), font=title_font)
        diversity = row["diversity_metadata"]
        detail = f"{diversity['theme']} · {diversity['aspect_family']} · legend {diversity['legend_position']} · {row['difficulty']}"
        draw.text((x0 + 16, y0 + 51), detail, fill=(66, 75, 87), font=body_font)
        draw.text((x0 + 16, y0 + 431), "original", fill=(48, 57, 68), font=small_font)
        draw.text((x0 + 325, y0 + 431), "target mask overlay", fill=(48, 57, 68), font=small_font)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", dpi=(GALLERY_DPI_V2, GALLERY_DPI_V2), optimize=True)
    return output


def _overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    binary = np.asarray(mask.convert("L")) > 0
    array[binary] = 0.48 * array[binary] + 0.52 * np.asarray((255, 43, 170))
    output = np.clip(array, 0, 255).astype(np.uint8)
    mask_image = Image.fromarray(binary.astype(np.uint8) * 255)
    contour = np.logical_xor(
        np.asarray(mask_image.filter(ImageFilter.MaxFilter(9))) > 0,
        np.asarray(mask_image.filter(ImageFilter.MinFilter(9))) > 0,
    )
    output[contour] = (255, 255, 255)
    return Image.fromarray(output)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()
