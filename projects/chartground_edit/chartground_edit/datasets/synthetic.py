"""Deterministic 32-sample synthetic dataset for the Phase 1A data loop."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chartground_edit.visualization.render import create_contact_sheet, render_sample_panel

from .reader import ChartGroundDataset
from .schema import CHART_TYPES, REFERRING_TYPES, validate_jsonl


GENERATOR_VERSION = "synthetic-v0"
DEFAULT_SEED = 20260915
IMAGE_SIZE = (480, 320)
PLOT_BOX = (52, 30, 360, 270)

CHART_ORDER = ("line", "bar", "scatter", "confidence_band")
REFERRING_ORDER = ("category", "appearance", "legend", "trend")
EDIT_ACTIONS = ("highlight", "recolor", "extract", "remove")
COLORS = ("#2878b5", "#d4553d", "#3a923a")
SERIES_NAMES = ("accuracy", "latency", "throughput")


def generate_dataset(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    count_per_combination: int = 2,
    clean: bool = False,
) -> Path:
    """Generate balanced chart/referring pairs and return the JSONL path.

    ``clean`` only removes the generator-owned image, mask, and visualization
    directories plus its manifest inside the exact output directory.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if count_per_combination <= 0:
        raise ValueError("count_per_combination must be positive")
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    visualization_dir = output_dir / "visualizations"
    manifest_path = output_dir / "annotations.jsonl"
    summary_path = output_dir / "summary.json"
    gallery_path = output_dir / "gallery.png"
    if clean:
        for owned_dir in (image_dir, mask_dir, visualization_dir):
            if owned_dir.exists():
                shutil.rmtree(owned_dir)
        for owned_file in (manifest_path, summary_path, gallery_path):
            if owned_file.exists():
                owned_file.unlink()
    for directory in (image_dir, mask_dir, visualization_dir):
        directory.mkdir(parents=True, exist_ok=True)

    annotations: list[dict[str, Any]] = []
    global_index = 0
    for chart_type in CHART_ORDER:
        for referring_type in REFERRING_ORDER:
            for repetition in range(count_per_combination):
                sample_seed = seed + global_index
                sample_id = f"cge_{chart_type}_{referring_type}_{repetition:02d}"
                image, mask, attributes = _render_chart(
                    chart_type, sample_seed, repetition
                )
                image_relative = Path("images") / f"{sample_id}.png"
                mask_relative = Path("masks") / f"{sample_id}.png"
                image.save(output_dir / image_relative, format="PNG")
                mask.save(output_dir / mask_relative, format="PNG")
                action = EDIT_ACTIONS[global_index % len(EDIT_ACTIONS)]
                annotation = {
                    "sample_id": sample_id,
                    "image_path": image_relative.as_posix(),
                    "mask_path": mask_relative.as_posix(),
                    "width": IMAGE_SIZE[0],
                    "height": IMAGE_SIZE[1],
                    "chart_type": chart_type,
                    "referring_type": referring_type,
                    "instruction": _instruction(
                        chart_type, referring_type, action, attributes
                    ),
                    "target_type": _target_type(chart_type),
                    "target_attributes": attributes,
                    "edit_action": action,
                    "split": _split(global_index),
                    "generator_seed": sample_seed,
                    "metadata": {
                        "schema_version": "chartground-edit-v0",
                        "generator_version": GENERATOR_VERSION,
                        "source": "synthetic",
                        "render_backend": "Pillow",
                        "mask_format": "binary_png",
                    },
                }
                annotations.append(annotation)
                global_index += 1

    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for annotation in annotations:
            handle.write(json.dumps(annotation, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    validate_jsonl(
        manifest_path,
        check_files=True,
        expected_count=len(CHART_ORDER) * len(REFERRING_ORDER) * count_per_combination,
    )
    dataset = ChartGroundDataset(manifest_path)
    panel_paths = []
    for sample in dataset:
        panel_path = visualization_dir / f"{sample.annotation['sample_id']}.png"
        render_sample_panel(sample, panel_path)
        panel_paths.append(panel_path)
    create_contact_sheet(panel_paths, gallery_path, columns=2)
    _write_summary(output_dir, annotations, seed)
    return manifest_path


def dataset_fingerprint(output_dir: str | Path) -> dict[str, str]:
    """Return SHA-256 hashes for the manifest and masks, for repeatability checks."""
    output_dir = Path(output_dir)
    manifest = output_dir / "annotations.jsonl"
    hashes = {"annotations.jsonl": _sha256(manifest)}
    for path in sorted((output_dir / "masks").glob("*.png")):
        hashes[f"masks/{path.name}"] = _sha256(path)
    return hashes


def _render_chart(
    chart_type: str, seed: int, repetition: int
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    renderers: dict[
        str, Callable[[np.random.Generator, int], tuple[Image.Image, Image.Image, dict[str, Any]]]
    ] = {
        "line": _render_line,
        "bar": _render_bar,
        "scatter": _render_scatter,
        "confidence_band": _render_confidence_band,
    }
    if chart_type not in renderers:
        raise ValueError(f"unsupported chart_type: {chart_type}")
    return renderers[chart_type](np.random.default_rng(seed), repetition)


def _base_chart(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw, Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    mask = Image.new("L", IMAGE_SIZE, 0)
    mask_draw = ImageDraw.Draw(mask)
    font = ImageFont.load_default()
    left, top, right, bottom = PLOT_BOX
    for fraction in (0.25, 0.5, 0.75):
        y = round(top + fraction * (bottom - top))
        draw.line((left, y, right, y), fill=(224, 228, 232), width=1)
    draw.line((left, top, left, bottom), fill=(30, 30, 30), width=2)
    draw.line((left, bottom, right, bottom), fill=(30, 30, 30), width=2)
    draw.text((left, 8), title, fill=(20, 20, 20), font=font)
    draw.text((190, 292), "epoch", fill=(45, 45, 45), font=font)
    draw.text((8, 140), "value", fill=(45, 45, 45), font=font)
    return image, draw, mask, mask_draw


def _render_line(
    rng: np.random.Generator, repetition: int
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    image, draw, mask, mask_draw = _base_chart("Line chart")
    x_values = np.linspace(0.0, 1.0, 8)
    noise = rng.normal(0.0, 0.018, size=(3, len(x_values)))
    series = np.stack(
        (
            0.18 + 0.64 * x_values,
            0.82 - 0.50 * x_values,
            0.36 + 0.18 * np.sin(np.pi * x_values),
        )
    ) + noise
    target_index = repetition % 2
    for index, values in enumerate(series):
        points = _data_points(x_values, values)
        draw.line(points, fill=COLORS[index], width=4)
        for point in points:
            _circle(draw, point, 4, COLORS[index])
        if index == target_index:
            mask_draw.line(points, fill=255, width=4)
            for point in points:
                _circle(mask_draw, point, 4, 255)
    _draw_legend(draw, tuple(zip(SERIES_NAMES, COLORS)), kind="line")
    trend = "increasing" if target_index == 0 else "decreasing"
    return image, mask, {
        "series_name": SERIES_NAMES[target_index],
        "category": f"series_{target_index + 1}",
        "color": COLORS[target_index],
        "trend": trend,
        "target_index": target_index,
        "includes_markers": True,
        "includes_confidence_band": False,
    }


def _render_bar(
    rng: np.random.Generator, repetition: int
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    image, draw, mask, mask_draw = _base_chart("Bar chart")
    values = np.asarray([0.40, 0.72, 0.55, 0.88, 0.62]) + rng.normal(0, 0.02, 5)
    categories = ("A", "B", "C", "D", "E")
    target_index = 3 if repetition == 0 else 1
    left, top, right, bottom = PLOT_BOX
    step = (right - left) / len(values)
    font = ImageFont.load_default()
    for index, value in enumerate(values):
        x0 = round(left + index * step + 11)
        x1 = round(left + (index + 1) * step - 11)
        y0 = round(bottom - float(value) * (bottom - top))
        box = (x0, y0, x1, bottom - 1)
        draw.rectangle(box, fill=COLORS[index % 3], outline=(35, 35, 35), width=2)
        draw.text((round((x0 + x1) / 2) - 3, bottom + 6), categories[index], fill="black", font=font)
        if index == target_index:
            mask_draw.rectangle(box, fill=255, outline=255, width=2)
    _draw_legend(
        draw,
        tuple((f"category {name}", COLORS[index % 3]) for index, name in enumerate(categories[:3])),
        kind="box",
    )
    rank = int(np.argsort(np.argsort(values))[target_index]) + 1
    return image, mask, {
        "series_name": f"category {categories[target_index]}",
        "category": categories[target_index],
        "color": COLORS[target_index % 3],
        "trend": "highest" if target_index == int(np.argmax(values)) else "second_highest",
        "height_rank_ascending": rank,
        "target_index": target_index,
        "includes_fill": True,
        "includes_border": True,
    }


def _render_scatter(
    rng: np.random.Generator, repetition: int
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    image, draw, mask, mask_draw = _base_chart("Scatter chart")
    target_index = repetition % 2
    slopes = (0.55, -0.48, 0.12)
    x_values = np.linspace(0.08, 0.92, 10)
    for index, slope in enumerate(slopes):
        y_values = 0.5 + slope * (x_values - 0.5) + rng.normal(0, 0.055, len(x_values))
        points = _data_points(x_values, y_values)
        for point in points:
            _circle(draw, point, 5, COLORS[index])
            if index == target_index:
                _circle(mask_draw, point, 5, 255)
    _draw_legend(draw, tuple(zip(SERIES_NAMES, COLORS)), kind="marker")
    trend = "positive" if slopes[target_index] > 0 else "negative"
    return image, mask, {
        "series_name": SERIES_NAMES[target_index],
        "category": f"group_{target_index + 1}",
        "color": COLORS[target_index],
        "trend": f"{trend}_correlation",
        "target_index": target_index,
        "marker_count": len(x_values),
    }


def _render_confidence_band(
    rng: np.random.Generator, repetition: int
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    image, draw, mask, mask_draw = _base_chart("Confidence bands")
    x_values = np.linspace(0.0, 1.0, 9)
    means = np.stack((0.22 + 0.55 * x_values, 0.77 - 0.44 * x_values))
    means += rng.normal(0, 0.012, means.shape)
    widths = (0.10, 0.075)
    target_index = repetition % 2
    band_colors = ("#b8d7eb", "#efc3ba")
    for index, mean in enumerate(means):
        upper = np.clip(mean + widths[index], 0, 1)
        lower = np.clip(mean - widths[index], 0, 1)
        upper_points = _data_points(x_values, upper)
        lower_points = _data_points(x_values, lower)
        polygon = upper_points + list(reversed(lower_points))
        draw.polygon(polygon, fill=band_colors[index])
        if index == target_index:
            mask_draw.polygon(polygon, fill=255)
    for index, mean in enumerate(means):
        draw.line(_data_points(x_values, mean), fill=COLORS[index], width=3)
    _draw_legend(draw, tuple(zip(SERIES_NAMES[:2], COLORS[:2])), kind="line")
    trend = "increasing" if target_index == 0 else "decreasing"
    return image, mask, {
        "series_name": SERIES_NAMES[target_index],
        "category": f"band_{target_index + 1}",
        "color": band_colors[target_index],
        "trend": trend,
        "target_index": target_index,
        "band_half_width": widths[target_index],
        "independent_from_curve": True,
    }


def _data_points(x_values: np.ndarray, y_values: np.ndarray) -> list[tuple[int, int]]:
    left, top, right, bottom = PLOT_BOX
    return [
        (
            round(left + float(x) * (right - left)),
            round(bottom - float(np.clip(y, 0, 1)) * (bottom - top)),
        )
        for x, y in zip(x_values, y_values)
    ]


def _circle(draw: ImageDraw.ImageDraw, point: tuple[int, int], radius: int, fill: Any) -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    entries: tuple[tuple[str, str], ...],
    *,
    kind: str,
) -> None:
    font = ImageFont.load_default()
    x0, y0 = 374, 46
    draw.rectangle((368, 36, 474, 54 + len(entries) * 24), outline=(150, 150, 150), fill="white")
    for index, (label, color) in enumerate(entries):
        y = y0 + index * 24
        if kind == "line":
            draw.line((x0, y + 5, x0 + 20, y + 5), fill=color, width=3)
            _circle(draw, (x0 + 10, y + 5), 3, color)
        elif kind == "marker":
            _circle(draw, (x0 + 10, y + 5), 4, color)
        else:
            draw.rectangle((x0 + 4, y, x0 + 16, y + 10), fill=color, outline=(40, 40, 40))
        draw.text((x0 + 25, y), label, fill=(25, 25, 25), font=font)


def _instruction(
    chart_type: str,
    referring_type: str,
    action: str,
    attributes: dict[str, Any],
) -> str:
    noun = {
        "line": "曲线",
        "bar": "柱子",
        "scatter": "散点序列",
        "confidence_band": "置信区间",
    }[chart_type]
    if referring_type == "category":
        reference = f"类别 {attributes['category']} 对应的{noun}"
    elif referring_type == "appearance":
        reference = f"颜色为 {attributes['color']} 的{noun}"
    elif referring_type == "legend":
        reference = f"图例中 {attributes['series_name']} 对应的{noun}"
    else:
        reference = f"趋势为 {attributes['trend']} 的{noun}"
    action_text = {
        "highlight": "高亮",
        "recolor": "改成红色",
        "extract": "提取到透明背景",
        "remove": "用背景色移除",
    }[action]
    return f"请将{reference}{action_text}"


def _target_type(chart_type: str) -> str:
    return {
        "line": "curve",
        "bar": "bar",
        "scatter": "scatter_series",
        "confidence_band": "confidence_band",
    }[chart_type]


def _split(index: int) -> str:
    remainder = index % 8
    if remainder == 0:
        return "val"
    if remainder == 1:
        return "test"
    return "train"


def _write_summary(output_dir: Path, annotations: list[dict[str, Any]], seed: int) -> None:
    combinations = Counter(
        (item["chart_type"], item["referring_type"]) for item in annotations
    )
    summary = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "sample_count": len(annotations),
        "chart_type_counts": dict(Counter(item["chart_type"] for item in annotations)),
        "referring_type_counts": dict(Counter(item["referring_type"] for item in annotations)),
        "edit_action_counts": dict(Counter(item["edit_action"] for item in annotations)),
        "split_counts": dict(Counter(item["split"] for item in annotations)),
        "combination_counts": {
            f"{chart_type}/{referring_type}": combinations[(chart_type, referring_type)]
            for chart_type in CHART_ORDER
            for referring_type in REFERRING_ORDER
        },
        "supported_chart_types": sorted(CHART_TYPES),
        "supported_referring_types": sorted(REFERRING_TYPES),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
