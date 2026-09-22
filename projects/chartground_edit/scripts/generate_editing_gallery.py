#!/usr/bin/env python3
"""Build the versioned four-chart Phase 1B editing gallery from synthetic_v0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets.reader import ChartGroundDataset, ChartGroundSample
from chartground_edit.editing import edit
from chartground_edit.visualization.render import mask_overlay


SELECTIONS = (
    ("line", "highlight", {"strength": 0.72}),
    ("bar", "recolor", {"color": "#E63946"}),
    ("scatter", "extract", {}),
    (
        "confidence_band",
        "remove",
        {"fill_mode": "neighbor", "color": "#FFFFFF", "neighbor_radius": 5},
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("projects/chartground_edit/data/synthetic_v0/annotations.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("projects/chartground_edit/assets/editing_v0_gallery.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise SystemExit(
            f"error: manifest does not exist: {args.manifest}; generate synthetic_v0 first"
        )
    if args.output.suffix.lower() != ".png":
        raise SystemExit("error: --output must end with .png")
    dataset = ChartGroundDataset(args.manifest)
    samples = [_select_sample(dataset, chart_type, action) for chart_type, action, _ in SELECTIONS]
    gallery = _render_gallery(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(args.output, format="PNG")
    print(f"rows={len(samples)}")
    print(f"output={args.output}")


def _select_sample(
    dataset: ChartGroundDataset, chart_type: str, action: str
) -> ChartGroundSample:
    for sample in dataset:
        annotation = sample.annotation
        if annotation["chart_type"] == chart_type and annotation["edit_action"] == action:
            return sample
    raise RuntimeError(f"no sample matches chart_type={chart_type}, action={action}")


def _render_gallery(samples: list[ChartGroundSample]) -> Image.Image:
    panel_size = (360, 240)
    labels = ("original", "GT mask overlay", "edited result")
    margin = 18
    header = 54
    row_gap = 14
    row_height = header + panel_size[1]
    width = margin * 2 + panel_size[0] * 3
    height = margin * 2 + row_height * len(samples) + row_gap * (len(samples) - 1)
    gallery = Image.new("RGB", (width, height), (232, 235, 239))
    draw = ImageDraw.Draw(gallery)
    font = ImageFont.load_default()

    for row, (sample, selection) in enumerate(zip(samples, SELECTIONS)):
        chart_type, action, parameters = selection
        edited = edit(sample.image, sample.mask, action, parameters)
        panels = (
            sample.image,
            mask_overlay(sample.image, sample.mask),
            _display_image(edited),
        )
        y = margin + row * (row_height + row_gap)
        draw.rectangle(
            (margin, y, width - margin, y + row_height), fill="white", outline=(205, 209, 214)
        )
        title = f"{chart_type} | {action} | {sample.annotation['sample_id']}"
        draw.text((margin + 10, y + 8), title, fill=(20, 24, 29), font=font)
        for column, (label, panel) in enumerate(zip(labels, panels)):
            x = margin + column * panel_size[0]
            draw.text((x + 10, y + 28), label, fill=(65, 70, 76), font=font)
            gallery.paste(_contain(panel, panel_size), (x, y + header))
    return gallery


def _display_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    checker = Image.new("RGB", image.size, "white")
    draw = ImageDraw.Draw(checker)
    tile = 12
    for top in range(0, image.height, tile):
        for left in range(0, image.width, tile):
            if (left // tile + top // tile) % 2:
                draw.rectangle(
                    (left, top, left + tile - 1, top + tile - 1), fill=(218, 218, 218)
                )
    checker.paste(image, mask=image.getchannel("A"))
    return checker


def _contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumbnail = image.convert("RGB").copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - thumbnail.width) // 2, (size[1] - thumbnail.height) // 2)
    canvas.paste(thumbnail, offset)
    return canvas


if __name__ == "__main__":
    main()
