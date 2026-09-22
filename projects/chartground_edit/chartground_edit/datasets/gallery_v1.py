"""Compact deterministic synthetic_v1 inspection gallery."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from chartground_edit.visualization.render import mask_overlay

from .schema_v1 import validate_jsonl_v1
from .synthetic_v1 import (
    ACTION_ORDER_V1,
    CHART_ORDER_V1,
    DIFFICULTY_ORDER_V1,
    REFERRING_ORDER_V1,
)


def create_synthetic_v1_gallery(
    manifest_path: str | Path, output_path: str | Path
) -> Path:
    """Select one deterministic example per chart/referring combination."""
    manifest = Path(manifest_path)
    records = validate_jsonl_v1(manifest, check_files=True, expected_count=320)
    selected = select_gallery_records(records)

    cell_width, cell_height = 430, 205
    sheet = Image.new("RGB", (cell_width * 4, cell_height * 4), (232, 232, 232))
    font = ImageFont.load_default()
    for index, record in enumerate(selected):
        row, column = divmod(index, 4)
        x0, y0 = column * cell_width, row * cell_height
        image_path = manifest.parent / str(record["image_path"])
        mask_path = manifest.parent / str(record["mask_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        with Image.open(mask_path) as source:
            mask = source.convert("L").copy()
        original = _fit(image, (205, 137))
        overlay = _fit(mask_overlay(image, mask), (205, 137))
        sheet.paste(original, (x0 + 7, y0 + 40))
        sheet.paste(overlay, (x0 + 218, y0 + 40))
        draw = ImageDraw.Draw(sheet)
        title = f"{record['chart_type']} / {record['referring_type']}"
        details = (
            f"{record['split']} | {record['edit_action']} | {record['difficulty']}"
        )
        draw.text((x0 + 8, y0 + 7), title, fill=(20, 20, 20), font=font)
        draw.text((x0 + 8, y0 + 22), details, fill=(45, 45, 45), font=font)
        draw.text((x0 + 7, y0 + 181), "original", fill=(25, 25, 25), font=font)
        draw.text((x0 + 218, y0 + 181), "mask overlay", fill=(25, 25, 25), font=font)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    return output


def select_gallery_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return a deterministic 16-cell selection covering required dimensions."""
    selected: list[dict[str, object]] = []
    for combination_index, (chart, referring) in enumerate(
        (chart, referring)
        for chart in CHART_ORDER_V1
        for referring in REFERRING_ORDER_V1
    ):
        action = ACTION_ORDER_V1[combination_index % 4]
        difficulty = DIFFICULTY_ORDER_V1[combination_index % 3]
        candidates = [
            record
            for record in records
            if record["chart_type"] == chart
            and record["referring_type"] == referring
            and record["edit_action"] == action
            and record["difficulty"] == difficulty
        ]
        if not candidates:
            raise ValueError(
                f"no gallery candidate for {chart}/{referring}/{action}/{difficulty}"
            )
        selected.append(sorted(candidates, key=lambda item: item["sample_id"])[0])
    return selected


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas
