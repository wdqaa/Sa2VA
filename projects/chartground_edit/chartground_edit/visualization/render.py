"""Pillow-based original/mask/overlay/crop visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chartground_edit.datasets.reader import ChartGroundSample


_PANEL_LABELS = ("original", "mask", "overlay", "target crop")


def mask_overlay(
    image: Image.Image,
    mask: Image.Image,
    *,
    color: tuple[int, int, int] = (255, 36, 36),
    alpha: float = 0.48,
) -> Image.Image:
    """Overlay a solid color only where the binary mask is foreground."""
    if image.size != mask.size:
        raise ValueError("image and mask must have the same size")
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    active = np.asarray(mask.convert("L"), dtype=np.uint8) == 255
    result = base.copy()
    result[active] = (1.0 - alpha) * result[active] + alpha * np.asarray(color)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def target_crop(
    image: Image.Image, mask: Image.Image, *, padding: int = 8
) -> Image.Image:
    """Crop the image to the foreground mask bounding box with padding."""
    bbox = mask.convert("L").getbbox()
    if bbox is None:
        raise ValueError("cannot crop an empty mask")
    left, top, right, bottom = bbox
    box = (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    )
    return image.convert("RGB").crop(box)


def render_sample_panel(
    sample: ChartGroundSample,
    output_path: str | Path,
    *,
    panel_size: tuple[int, int] = (240, 160),
) -> Path:
    """Write a four-column original/mask/overlay/crop inspection panel."""
    image = sample.image.convert("RGB")
    mask_rgb = sample.mask.convert("RGB")
    panels = [image, mask_rgb, mask_overlay(image, sample.mask), target_crop(image, sample.mask)]
    header_height = 32
    footer_height = 24
    output = Image.new(
        "RGB",
        (panel_size[0] * 4, header_height + panel_size[1] + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    title = (
        f"{sample.annotation['sample_id']} | {sample.annotation['chart_type']} | "
        f"{sample.annotation['referring_type']} | {sample.annotation['edit_action']}"
    )
    draw.text((8, 10), title, fill="black", font=font)
    for index, (label, panel) in enumerate(zip(_PANEL_LABELS, panels)):
        thumb = _contain_on_white(panel, panel_size)
        x = index * panel_size[0]
        output.paste(thumb, (x, header_height))
        draw.text((x + 6, header_height + panel_size[1] + 5), label, fill="black", font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path, format="PNG")
    return output_path


def create_contact_sheet(
    panel_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    columns: int = 2,
    thumbnail_width: int = 720,
) -> Path:
    """Combine all sample panels into one PNG contact sheet."""
    paths = [Path(path) for path in panel_paths]
    if not paths:
        raise ValueError("at least one panel is required")
    if columns <= 0:
        raise ValueError("columns must be positive")
    thumbnails: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as panel_file:
            panel = panel_file.convert("RGB").copy()
        height = max(1, round(panel.height * thumbnail_width / panel.width))
        thumbnails.append(
            panel.resize((thumbnail_width, height), Image.Resampling.LANCZOS)
        )
    cell_height = max(image.height for image in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    gutter = 8
    sheet = Image.new(
        "RGB",
        (
            columns * thumbnail_width + (columns + 1) * gutter,
            rows * cell_height + (rows + 1) * gutter,
        ),
        (225, 225, 225),
    )
    for index, thumbnail in enumerate(thumbnails):
        row, column = divmod(index, columns)
        x = gutter + column * (thumbnail_width + gutter)
        y = gutter + row * (cell_height + gutter)
        sheet.paste(thumbnail, (x, y))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path


def _contain_on_white(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = image.convert("RGB").copy()
    contained.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2)
    canvas.paste(contained, offset)
    return canvas
