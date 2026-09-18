"""Deterministic 1,600-sample synthetic v2 generator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .render_v2 import (
    AXIS_SCALES,
    FONT_FAMILIES,
    GRID_MODES,
    LAYOUT_DENSITIES,
    LEGEND_POSITIONS,
    NUMERIC_FORMATS,
    SIZE_FAMILIES,
    THEMES,
    render_scene_v2,
)
from .schema_v1 import MASK_SEMANTICS_VERSION
from .schema_v2 import GENERATOR_VERSION_V2, SCHEMA_VERSION_V2, validate_jsonl_v2


DEFAULT_SEED_V2 = 20260917
CHART_ORDER_V2 = ("line", "bar", "scatter", "confidence_band")
REFERRING_ORDER_V2 = ("category", "appearance", "legend", "trend")
SPLIT_ORDER_V2 = ("train", "val", "test")
ACTION_ORDER_V2 = ("highlight", "recolor", "extract", "remove")
DIFFICULTY_ORDER_V2 = ("easy", "medium", "hard")
SPLIT_COUNTS_V2 = {"train": 60, "val": 20, "test": 20}
ACTION_COUNTS_V2 = {"train": 15, "val": 5, "test": 5}
_DPI_VALUES = (96, 120, 144, 180, 220)
_JPEG_QUALITIES = (100, 96, 90, 84)
_BLUR_RADII = (0.0, 0.0, 0.35, 0.65)
_SCREENSHOT_SCALES = (1.0, 1.0, 0.88, 0.76)
_CURVE_FAMILIES = ("monotonic", "periodic", "piecewise", "plateau", "noisy", "exponential")
_CROSSING_FAMILIES = ("none", "single", "multiple")


def build_generation_plan_v2(seed: int = DEFAULT_SEED_V2) -> list[dict[str, Any]]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    plan: list[dict[str, Any]] = []
    for chart_index, chart_type in enumerate(CHART_ORDER_V2):
        for referring_index, referring_type in enumerate(REFERRING_ORDER_V2):
            combination = chart_index * 4 + referring_index
            for split_index, split in enumerate(SPLIT_ORDER_V2):
                split_items: list[dict[str, Any]] = []
                for slot in range(SPLIT_COUNTS_V2[split]):
                    identity = f"{seed}|v2|{chart_type}|{referring_type}|{split}|{slot}"
                    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                    sample_seed = int(digest[:15], 16)
                    style_index = slot + combination * 7 + split_index * 11
                    width, height, aspect = SIZE_FAMILIES[style_index % len(SIZE_FAMILIES)]
                    distractor_count = 1 + ((slot * 3 + combination + split_index) % 5)
                    theme = THEMES[(style_index // 2) % len(THEMES)]
                    layout = LAYOUT_DENSITIES[(style_index // 3) % len(LAYOUT_DENSITIES)]
                    legend_position = LEGEND_POSITIONS[style_index % len(LEGEND_POSITIONS)]
                    legend_columns = 1 + ((style_index // 2) % 3)
                    crossing_index = (style_index // 2) % len(_CROSSING_FAMILIES)
                    target_area_proxy = {"line": 0.82, "bar": 0.42, "scatter": 0.76, "confidence_band": 0.18}[chart_type]
                    difficulty_proxy = sum(
                        (
                            ((style_index + combination) % 3) / 2.0,
                            crossing_index / 2.0,
                            (distractor_count - 1) / 4.0,
                            (style_index % 5) / 4.0,
                            min(1.0, (distractor_count + 1) / (legend_columns * 3.0)),
                            target_area_proxy,
                            1.0 - ((style_index % 4) / 3.0),
                            crossing_index / 2.0 if chart_type == "confidence_band" else 0.0,
                        )
                    ) / 8.0
                    split_items.append(
                        {
                            "chart_type": chart_type,
                            "referring_type": referring_type,
                            "split": split,
                            "slot": slot,
                            "edit_action": ACTION_ORDER_V2[slot % 4],
                            "difficulty": "pending",
                            "difficulty_plan_score": round(difficulty_proxy, 6),
                            "distractor_count": distractor_count,
                            "seed": sample_seed,
                            "sample_id": f"cgev2_{chart_type}_{referring_type}_{digest[15:27]}",
                            "scene_id": f"scene_v2_{digest[27:47]}",
                            "style_family": f"v2_{split}_{theme}_{layout}_{style_index % 17:02d}",
                            "instruction_template_family": f"v2_{split}_template_{(slot + combination) % 7:02d}",
                            "canvas_size": (width, height),
                            "aspect_family": aspect,
                            "dpi": _DPI_VALUES[style_index % len(_DPI_VALUES)],
                            "legend_position": legend_position,
                            "legend_columns": legend_columns,
                            "theme": theme,
                            "grid_mode": GRID_MODES[(style_index // 2) % len(GRID_MODES)],
                            "font_family": FONT_FAMILIES[(style_index // 4) % len(FONT_FAMILIES)],
                            "font_size": (10, 12, 14, 16)[style_index % 4],
                            "axis_scale": AXIS_SCALES[(style_index // 3) % len(AXIS_SCALES)],
                            "numeric_format": NUMERIC_FORMATS[(style_index // 5) % len(NUMERIC_FORMATS)],
                            "layout_density": layout,
                            "show_title": style_index % 5 != 0,
                            "show_subtitle": style_index % 4 == 0,
                            "show_axis_labels": style_index % 3 != 0,
                            "jpeg_quality": _JPEG_QUALITIES[(style_index // 3) % len(_JPEG_QUALITIES)],
                            "blur_radius": _BLUR_RADII[(style_index // 5) % len(_BLUR_RADII)],
                            "screenshot_scale": _SCREENSHOT_SCALES[(style_index // 7) % len(_SCREENSHOT_SCALES)],
                            "antialias": (1, 2)[(style_index // 3) % 2],
                            "curve_family": _CURVE_FAMILIES[style_index % len(_CURVE_FAMILIES)],
                            "crossing_family": _CROSSING_FAMILIES[(style_index // 2) % len(_CROSSING_FAMILIES)],
                            "family_offset": style_index % 12,
                        }
                    )
                ranked = sorted(split_items, key=lambda row: (row["difficulty_plan_score"], row["sample_id"]))
                for rank, item in enumerate(ranked):
                    item["difficulty"] = DIFFICULTY_ORDER_V2[min(2, rank * 3 // len(ranked))]
                plan.extend(sorted(split_items, key=lambda row: row["slot"]))
    return plan


def generate_synthetic_v2(
    output_dir: str | Path, *, seed: int = DEFAULT_SEED_V2, clean: bool = False,
) -> Path:
    output = Path(output_dir)
    if clean:
        _validate_clean_target(output)
        _clean_owned_outputs(output)
    image_dir, mask_dir = output / "images", output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in build_generation_plan_v2(seed):
        image, mask, attributes, metadata, diversity, content_id = render_scene_v2(item)
        image_relative = Path("images") / f"{item['sample_id']}.png"
        mask_relative = Path("masks") / f"{item['sample_id']}.png"
        image.save(output / image_relative, format="PNG", dpi=(item["dpi"], item["dpi"]), optimize=False)
        mask.save(output / mask_relative, format="PNG", optimize=False)
        expression = _referring_expression(item["chart_type"], item["referring_type"], attributes)
        parameters = _edit_parameters(item["edit_action"], attributes["color"], item["seed"])
        width, height = image.size
        records.append(
            {
                "schema_version": SCHEMA_VERSION_V2,
                "sample_id": item["sample_id"],
                "image_path": image_relative.as_posix(),
                "mask_path": mask_relative.as_posix(),
                "split": item["split"],
                "chart_type": item["chart_type"],
                "referring_type": item["referring_type"],
                "full_instruction": _full_instruction(expression, item["edit_action"], parameters, item["instruction_template_family"]),
                "referring_expression": expression,
                "edit_action": item["edit_action"],
                "edit_parameters": parameters,
                "target_type": {"line": "curve", "bar": "bar", "scatter": "scatter_series", "confidence_band": "confidence_band"}[item["chart_type"]],
                "target_attributes": attributes,
                "mask_semantics_version": MASK_SEMANTICS_VERSION,
                "generator_version": GENERATOR_VERSION_V2,
                "seed": item["seed"],
                "scene_id": item["scene_id"],
                "content_id": content_id,
                "style_family": item["style_family"],
                "instruction_template_family": item["instruction_template_family"],
                "difficulty": item["difficulty"],
                "distractor_count": item["distractor_count"],
                "image_width": width,
                "image_height": height,
                "generation_metadata": metadata,
                "diversity_metadata": diversity,
            }
        )
    manifest = output / "annotations.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    _write_summary(output, records, seed)
    validate_jsonl_v2(manifest, check_files=True, expected_count=1600)
    return manifest


def dataset_fingerprint_v2(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    result = {"annotations.jsonl": _sha256(root / "annotations.jsonl")}
    for directory in ("images", "masks"):
        for path in sorted((root / directory).glob("*.png")):
            result[f"{directory}/{path.name}"] = _sha256(path)
    return result


def _referring_expression(chart: str, referring: str, attributes: dict[str, Any]) -> str:
    noun = {"line": "曲线", "bar": "柱体序列", "scatter": "散点序列", "confidence_band": "置信区间带"}[chart]
    if referring == "category":
        return f"类别 {attributes['category']} 对应的{noun}"
    if referring == "appearance":
        return f"颜色为 {attributes['color']} 的{noun}"
    if referring == "legend":
        return f"图例标签 {attributes['legend_label']} 对应的{noun}"
    return f"趋势为“{attributes['trend']}”的{noun}"


def _edit_parameters(action: str, target_color: str, seed: int) -> dict[str, Any]:
    if action == "highlight":
        return {"strength": (0.55, 0.65, 0.75)[seed % 3]}
    if action == "recolor":
        choices = ("#E63946", "#00A896", "#F4A261", "#7B2CBF")
        color = choices[seed % len(choices)]
        return {"color": color if color.lower() != target_color.lower() else choices[(seed + 1) % len(choices)]}
    if action == "extract":
        return {"background": "transparent"}
    return {"fill_mode": "background_color", "color": "#FFFFFF"}


def _full_instruction(expression: str, action: str, parameters: dict[str, Any], family: str) -> str:
    if action == "highlight":
        operation = f"高亮该元素（强度 {parameters['strength']:.2f}）"
    elif action == "recolor":
        operation = f"将该元素改为 {parameters['color']}"
    elif action == "extract":
        operation = "提取该元素并使用透明背景"
    else:
        operation = f"移除该元素并使用确定性背景色 {parameters['color']} 填充"
    prefix = ("请定位", "找出", "在图中识别", "精确分割")[_stable_index(family, 4)]
    return f"{prefix}{expression}，然后{operation}。"


def _stable_index(value: str, modulo: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % modulo


def _write_summary(output: Path, records: list[dict[str, Any]], seed: int) -> None:
    summary = {
        "schema_version": SCHEMA_VERSION_V2,
        "generator_version": GENERATOR_VERSION_V2,
        "seed": seed,
        "sample_count": len(records),
        "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "chart_referring_counts": dict(sorted(Counter(f"{row['chart_type']}/{row['referring_type']}" for row in records).items())),
        "action_counts": dict(sorted(Counter(row["edit_action"] for row in records).items())),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_clean_target(path: Path) -> None:
    if path.name != "synthetic_v2" or path.resolve() in {Path.cwd().resolve(), Path("/")}:
        raise ValueError("unsafe clean target; directory must be named synthetic_v2")


def _clean_owned_outputs(path: Path) -> None:
    for name in ("annotations.jsonl", "summary.json", "audit.json"):
        target = path / name
        if target.is_file():
            target.unlink()
    for directory in (path / "images", path / "masks"):
        if directory.is_dir():
            for image in directory.glob("*.png"):
                image.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
