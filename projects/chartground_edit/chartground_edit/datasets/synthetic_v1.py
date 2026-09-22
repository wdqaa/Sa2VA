"""Deterministic, explicitly balanced 320-sample synthetic v1 generator."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schema_v1 import (
    GENERATOR_VERSION_V1,
    MASK_SEMANTICS_VERSION,
    SCHEMA_VERSION_V1,
    validate_jsonl_v1,
)


DEFAULT_SEED_V1 = 20260916
IMAGE_SIZE_V1 = (480, 320)
PLOT_BOX_V1 = (54, 32, 356, 270)
LEGEND_BOX_V1 = (366, 38, 474, 174)
CHART_ORDER_V1 = ("line", "bar", "scatter", "confidence_band")
REFERRING_ORDER_V1 = ("category", "appearance", "legend", "trend")
SPLIT_ORDER_V1 = ("train", "val", "test")
ACTION_ORDER_V1 = ("highlight", "recolor", "extract", "remove")
DIFFICULTY_ORDER_V1 = ("easy", "medium", "hard")
SPLIT_COUNTS_V1 = {"train": 12, "val": 4, "test": 4}

STYLE_FAMILIES_V1 = {
    "train": ("amber", "birch", "cedar"),
    "val": ("dune", "ember", "fjord"),
    "test": ("grove", "harbor", "iris"),
}
TEMPLATE_FAMILIES_V1 = {
    "train": ("atlas", "boreal", "cobalt"),
    "val": ("delta", "equinox", "fable"),
    "test": ("granite", "helix", "indigo"),
}

_COLORS = {
    "easy": ("#2369BD", "#D94841", "#278A45", "#8A52B8", "#E08B26"),
    "medium": ("#376FA6", "#4C7FB0", "#C75D52", "#D57265", "#577F4B"),
    "hard": ("#496D91", "#527494", "#5B7B97", "#64829A", "#6D899D"),
}
_SERIES_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon")
_CATEGORIES = ("A", "B", "C", "D", "E")
_TREND_NAMES = ("明显下降", "缓慢下降", "基本平稳", "缓慢上升", "明显上升")
_CORRELATION_NAMES = ("强负相关", "弱负相关", "无明显相关", "弱正相关", "强正相关")


def build_generation_plan(seed: int = DEFAULT_SEED_V1) -> list[dict[str, Any]]:
    """Build the full matrix before rendering; split is never inferred from index."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    plan: list[dict[str, Any]] = []
    for chart_index, chart_type in enumerate(CHART_ORDER_V1):
        for ref_index, referring_type in enumerate(REFERRING_ORDER_V1):
            combination_index = chart_index * len(REFERRING_ORDER_V1) + ref_index
            for split_index, split in enumerate(SPLIT_ORDER_V1):
                count = SPLIT_COUNTS_V1[split]
                actions = (
                    ACTION_ORDER_V1 * 3 if split == "train" else ACTION_ORDER_V1
                )
                for slot, action in enumerate(actions):
                    if split == "train":
                        difficulty = DIFFICULTY_ORDER_V1[
                            (slot + combination_index) % len(DIFFICULTY_ORDER_V1)
                        ]
                    else:
                        extra = (combination_index + split_index) % 3
                        difficulty = (
                            DIFFICULTY_ORDER_V1[slot]
                            if slot < 3
                            else DIFFICULTY_ORDER_V1[extra]
                        )
                    identity = (
                        f"{seed}|{chart_type}|{referring_type}|{split}|{slot}"
                    )
                    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                    sample_seed = int(digest[:15], 16)
                    plan.append(
                        {
                            "chart_type": chart_type,
                            "referring_type": referring_type,
                            "split": split,
                            "slot": slot,
                            "edit_action": action,
                            "difficulty": difficulty,
                            "seed": sample_seed,
                            "sample_id": (
                                f"cgev1_{chart_type}_{referring_type}_{digest[15:25]}"
                            ),
                            "scene_id": f"scene_{digest[25:41]}",
                            "style_family": STYLE_FAMILIES_V1[split][
                                (slot + combination_index) % 3
                            ],
                            "instruction_template_family": TEMPLATE_FAMILIES_V1[split][
                                (slot + combination_index) % 3
                            ],
                        }
                    )
    return plan


def generate_synthetic_v1(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED_V1,
    clean: bool = False,
) -> Path:
    """Generate v1 and return its manifest path."""
    output_dir = Path(output_dir)
    if clean:
        _validate_clean_target(output_dir)
        _clean_owned_outputs(output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    annotations: list[dict[str, Any]] = []
    for plan_item in build_generation_plan(seed):
        image, mask, attributes, metadata, content_id = _render_scene(plan_item)
        sample_id = plan_item["sample_id"]
        image_relative = Path("images") / f"{sample_id}.png"
        mask_relative = Path("masks") / f"{sample_id}.png"
        image.save(output_dir / image_relative, format="PNG", optimize=False)
        mask.save(output_dir / mask_relative, format="PNG", optimize=False)
        expression = _referring_expression(
            plan_item["chart_type"], plan_item["referring_type"], attributes
        )
        parameters = _edit_parameters(
            plan_item["edit_action"], attributes["color"], plan_item["seed"]
        )
        annotation = {
            "schema_version": SCHEMA_VERSION_V1,
            "sample_id": sample_id,
            "image_path": image_relative.as_posix(),
            "mask_path": mask_relative.as_posix(),
            "split": plan_item["split"],
            "chart_type": plan_item["chart_type"],
            "referring_type": plan_item["referring_type"],
            "full_instruction": _full_instruction(
                expression,
                plan_item["edit_action"],
                parameters,
                plan_item["instruction_template_family"],
            ),
            "referring_expression": expression,
            "edit_action": plan_item["edit_action"],
            "edit_parameters": parameters,
            "target_type": _target_type(plan_item["chart_type"]),
            "target_attributes": attributes,
            "mask_semantics_version": MASK_SEMANTICS_VERSION,
            "generator_version": GENERATOR_VERSION_V1,
            "seed": plan_item["seed"],
            "scene_id": plan_item["scene_id"],
            "content_id": content_id,
            "style_family": plan_item["style_family"],
            "instruction_template_family": plan_item[
                "instruction_template_family"
            ],
            "difficulty": plan_item["difficulty"],
            "distractor_count": metadata["entity_count"] - 1,
            "image_width": IMAGE_SIZE_V1[0],
            "image_height": IMAGE_SIZE_V1[1],
            "generation_metadata": metadata,
        }
        annotations.append(annotation)

    manifest = output_dir / "annotations.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for annotation in annotations:
            handle.write(json.dumps(annotation, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    _write_summary(output_dir, annotations, seed)
    validate_jsonl_v1(manifest, check_files=True, expected_count=320)
    return manifest


def dataset_fingerprint_v1(output_dir: str | Path) -> dict[str, str]:
    """Hash manifest and every generated image/mask for byte-level comparison."""
    root = Path(output_dir)
    result = {"annotations.jsonl": _sha256(root / "annotations.jsonl")}
    for directory in ("images", "masks"):
        for path in sorted((root / directory).glob("*.png")):
            result[f"{directory}/{path.name}"] = _sha256(path)
    return result


def _render_scene(
    plan: dict[str, Any],
) -> tuple[Image.Image, Image.Image, dict[str, Any], dict[str, Any], str]:
    rng = np.random.default_rng(plan["seed"])
    difficulty = plan["difficulty"]
    count = {"easy": 3, "medium": 4, "hard": 5}[difficulty]
    target_index = int(rng.integers(0, count))
    image, draw, mask, mask_draw = _base_chart(
        plan["chart_type"], plan["style_family"]
    )
    renderer = {
        "line": _render_line,
        "bar": _render_bar,
        "scatter": _render_scatter,
        "confidence_band": _render_confidence_band,
    }[plan["chart_type"]]
    entities, geometry, content_array = renderer(
        draw, mask_draw, rng, difficulty, count, target_index
    )
    _draw_legend(draw, entities, plan["chart_type"])
    target = entities[target_index]
    key = {
        "category": "category",
        "appearance": "color",
        "legend": "legend_label",
        "trend": "trend",
    }[plan["referring_type"]]
    value = target[key]
    matching = sum(entity[key] == value for entity in entities)
    if matching != 1:
        raise RuntimeError(
            f"referring expression is ambiguous for {plan['sample_id']}: "
            f"{key}={value!r} matches {matching} entities"
        )
    content_digest = hashlib.sha256()
    content_digest.update(plan["chart_type"].encode("ascii"))
    content_digest.update(np.asarray(content_array, dtype="<f8").tobytes())
    content_id = f"content_{content_digest.hexdigest()[:20]}"
    attributes = {
        "target_index": target_index,
        "category": target["category"],
        "color": target["color"],
        "legend_label": target["legend_label"],
        "trend": target["trend"],
        "includes_legend_proxy": False,
        "includes_markers": plan["chart_type"] in {"line", "scatter"},
    }
    metadata = {
        "render_backend": "Pillow",
        "entity_count": count,
        "target_index": target_index,
        "reference_key": key,
        "reference_value": value,
        "reference_match_count": matching,
        "entity_descriptors": entities,
        "plot_box": list(PLOT_BOX_V1),
        "legend_box": list(LEGEND_BOX_V1),
        "target_geometry": geometry,
        "content_sha256": content_digest.hexdigest(),
    }
    return image, mask, attributes, metadata, content_id


def _base_chart(
    chart_type: str, style_family: str
) -> tuple[Image.Image, ImageDraw.ImageDraw, Image.Image, ImageDraw.ImageDraw]:
    style_index = sum(ord(char) for char in style_family) % 3
    backgrounds = ((255, 255, 255), (250, 252, 249), (252, 250, 248))
    grids = ((222, 226, 230), (217, 226, 220), (228, 220, 216))
    image = Image.new("RGB", IMAGE_SIZE_V1, backgrounds[style_index])
    draw = ImageDraw.Draw(image)
    mask = Image.new("L", IMAGE_SIZE_V1, 0)
    mask_draw = ImageDraw.Draw(mask)
    left, top, right, bottom = PLOT_BOX_V1
    for fraction in (0.2, 0.4, 0.6, 0.8):
        y = round(top + fraction * (bottom - top))
        draw.line((left, y, right, y), fill=grids[style_index], width=1)
    draw.line((left, top, left, bottom), fill=(35, 35, 35), width=2)
    draw.line((left, bottom, right, bottom), fill=(35, 35, 35), width=2)
    font = ImageFont.load_default()
    draw.text((left, 9), chart_type.replace("_", " ").title(), fill=(25, 25, 25), font=font)
    draw.text((190, 292), "x", fill=(50, 50, 50), font=font)
    draw.text((18, 145), "y", fill=(50, 50, 50), font=font)
    return image, draw, mask, mask_draw


def _render_line(
    draw: ImageDraw.ImageDraw,
    mask_draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    difficulty: str,
    count: int,
    target_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    point_count = {"easy": 8, "medium": 10, "hard": 12}[difficulty]
    x = np.linspace(0.02, 0.98, point_count)
    slopes = np.linspace(-0.64, 0.64, count)
    phase = rng.uniform(0, np.pi, count)
    values = np.stack(
        [
            0.50
            + slope * (x - 0.5)
            + 0.035 * np.sin(2 * np.pi * x + phase[index])
            + rng.normal(0, 0.008 if difficulty == "easy" else 0.016, point_count)
            for index, slope in enumerate(slopes)
        ]
    )
    values = np.clip(values, 0.08, 0.92)
    width = {"easy": 5, "medium": 4, "hard": 2}[difficulty]
    radius = {"easy": 5, "medium": 4, "hard": 3}[difficulty]
    entities = _entities(count, difficulty, slopes, scatter=False)
    target_points: list[tuple[int, int]] = []
    for index in range(count):
        points = _data_points(x, values[index])
        draw.line(points, fill=entities[index]["color"], width=width)
        for point in points:
            _circle(draw, point, radius, entities[index]["color"])
        if index == target_index:
            target_points = points
            mask_draw.line(points, fill=255, width=width)
            for point in points:
                _circle(mask_draw, point, radius, 255)
    return entities, {"marker_centers": target_points, "line_width": width}, values


def _render_bar(
    draw: ImageDraw.ImageDraw,
    mask_draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    difficulty: str,
    count: int,
    target_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    raw = rng.uniform(0.22, 0.90, count)
    raw += np.arange(count) * 0.003
    order = np.argsort(raw)
    ranks = np.empty(count, dtype=int)
    ranks[order] = np.arange(1, count + 1)
    colors = _COLORS[difficulty][:count]
    left, top, right, bottom = PLOT_BOX_V1
    step = (right - left) / count
    jitter = int(rng.integers(-3, 4))
    font = ImageFont.load_default()
    entities: list[dict[str, Any]] = []
    target_box: tuple[int, int, int, int] | None = None
    for index, value in enumerate(raw):
        x0 = round(left + index * step + 8 + jitter)
        width_adjustment = index % 3
        x1 = round(left + (index + 1) * step - 8 + jitter + width_adjustment)
        y0 = round(bottom - float(value) * (bottom - top))
        box = (x0, y0, x1, bottom - 1)
        draw.rectangle(box, fill=colors[index], outline=(40, 40, 40), width=2)
        draw.text((round((x0 + x1) / 2) - 3, bottom + 6), _CATEGORIES[index], fill="black", font=font)
        if index == target_index:
            target_box = box
            mask_draw.rectangle(box, fill=255, outline=255, width=2)
        rank_from_top = count - int(ranks[index]) + 1
        entities.append(
            {
                "category": _CATEGORIES[index],
                "color": colors[index],
                "legend_label": f"value-{_SERIES_NAMES[index]}",
                "trend": f"第{rank_from_top}高",
            }
        )
    assert target_box is not None
    return entities, {"bar_box": list(target_box), "includes_border": True}, raw


def _render_scatter(
    draw: ImageDraw.ImageDraw,
    mask_draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    difficulty: str,
    count: int,
    target_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    point_count = {"easy": 9, "medium": 12, "hard": 16}[difficulty]
    x = np.linspace(0.06, 0.94, point_count)
    slopes = np.linspace(-0.72, 0.72, count)
    values = np.stack(
        [
            np.clip(
                0.50
                + slope * (x - 0.5)
                + rng.normal(0, 0.035 + 0.012 * (difficulty == "hard"), point_count),
                0.06,
                0.94,
            )
            for slope in slopes
        ]
    )
    radius = {"easy": 6, "medium": 5, "hard": 3}[difficulty]
    entities = _entities(count, difficulty, slopes, scatter=True)
    target_points: list[tuple[int, int]] = []
    for index in range(count):
        points = _data_points(x, values[index])
        for point in points:
            _circle(draw, point, radius, entities[index]["color"])
            if index == target_index:
                _circle(mask_draw, point, radius, 255)
        if index == target_index:
            target_points = points
    return entities, {"marker_centers": target_points, "marker_radius": radius}, values


def _render_confidence_band(
    draw: ImageDraw.ImageDraw,
    mask_draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    difficulty: str,
    count: int,
    target_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    point_count = {"easy": 9, "medium": 11, "hard": 13}[difficulty]
    x = np.linspace(0.0, 1.0, point_count)
    slopes = np.linspace(-0.56, 0.56, count)
    means = np.stack(
        [
            np.clip(
                0.5
                + slope * (x - 0.5)
                + 0.025 * np.sin(2 * np.pi * x + rng.uniform(0, np.pi)),
                0.15,
                0.85,
            )
            for slope in slopes
        ]
    )
    half_widths = rng.uniform(0.055, 0.105, count)
    entities = _entities(count, difficulty, slopes, scatter=False)
    target_polygon: list[tuple[int, int]] = []
    for index in range(count):
        upper = np.clip(means[index] + half_widths[index], 0.02, 0.98)
        lower = np.clip(means[index] - half_widths[index], 0.02, 0.98)
        polygon = _data_points(x, upper) + list(reversed(_data_points(x, lower)))
        fill = _hex_blend(entities[index]["color"], "#FFFFFF", 0.58)
        draw.polygon(polygon, fill=fill)
        if index == target_index:
            target_polygon = polygon
            mask_draw.polygon(polygon, fill=255)
    for index in range(count):
        draw.line(_data_points(x, means[index]), fill=entities[index]["color"], width=2)
    content = np.concatenate((means, half_widths[:, None]), axis=1)
    return entities, {"band_polygon": target_polygon, "mask_source": "filled_band_polygon"}, content


def _entities(
    count: int, difficulty: str, slopes: np.ndarray, *, scatter: bool
) -> list[dict[str, Any]]:
    trend_names = _CORRELATION_NAMES if scatter else _TREND_NAMES
    indexes = np.linspace(0, len(trend_names) - 1, count).round().astype(int)
    return [
        {
            "category": f"group-{_CATEGORIES[index]}",
            "color": _COLORS[difficulty][index],
            "legend_label": f"series-{_SERIES_NAMES[index]}",
            "trend": trend_names[int(indexes[index])],
            "slope": round(float(slopes[index]), 6),
        }
        for index in range(count)
    ]


def _draw_legend(
    draw: ImageDraw.ImageDraw, entities: list[dict[str, Any]], chart_type: str
) -> None:
    left, top, right, bottom = LEGEND_BOX_V1
    draw.rectangle((left, top, right, bottom), outline=(145, 145, 145), fill="white")
    font = ImageFont.load_default()
    for index, entity in enumerate(entities):
        y = top + 10 + index * 23
        color = entity["color"]
        if chart_type in {"line", "confidence_band"}:
            draw.line((left + 8, y + 5, left + 28, y + 5), fill=color, width=3)
            _circle(draw, (left + 18, y + 5), 3, color)
        elif chart_type == "scatter":
            _circle(draw, (left + 18, y + 5), 4, color)
        else:
            draw.rectangle((left + 11, y, left + 25, y + 11), fill=color, outline="#333333")
        draw.text((left + 33, y), entity["legend_label"], fill="#222222", font=font)


def _data_points(x_values: np.ndarray, y_values: np.ndarray) -> list[tuple[int, int]]:
    left, top, right, bottom = PLOT_BOX_V1
    return [
        (
            round(left + float(x) * (right - left)),
            round(bottom - float(np.clip(y, 0, 1)) * (bottom - top)),
        )
        for x, y in zip(x_values, y_values)
    ]


def _circle(
    draw: ImageDraw.ImageDraw, point: tuple[int, int], radius: int, fill: Any
) -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _referring_expression(
    chart_type: str, referring_type: str, attributes: dict[str, Any]
) -> str:
    noun = {
        "line": "曲线",
        "bar": "柱体",
        "scatter": "散点序列",
        "confidence_band": "置信区间带",
    }[chart_type]
    if referring_type == "category":
        return f"类别 {attributes['category']} 对应的{noun}"
    if referring_type == "appearance":
        return f"颜色为 {attributes['color']} 的{noun}"
    if referring_type == "legend":
        return f"图例标签 {attributes['legend_label']} 对应的{noun}"
    return f"趋势为“{attributes['trend']}”的{noun}"


def _full_instruction(
    expression: str,
    action: str,
    parameters: dict[str, Any],
    template_family: str,
) -> str:
    action_text = {
        "highlight": "高亮显示",
        "recolor": f"重新着色为 {parameters.get('color')}",
        "extract": "提取到透明背景",
        "remove": f"使用背景色 {parameters.get('color')} 移除",
    }[action]
    wrappers = {
        "atlas": "请将{target}{action}。",
        "boreal": "对{target}执行以下编辑：{action}。",
        "cobalt": "需要编辑{target}，操作是{action}。",
        "delta": "请定位{target}并{action}。",
        "equinox": "图中{target}需要被{action}。",
        "fable": "本次任务针对{target}：{action}。",
        "granite": "找出{target}，随后{action}。",
        "helix": "编辑对象是{target}；请{action}。",
        "indigo": "仅处理{target}，要求{action}。",
    }
    return wrappers[template_family].format(target=expression, action=action_text)


def _edit_parameters(action: str, current_color: str, seed: int) -> dict[str, Any]:
    if action == "highlight":
        return {"strength": 0.65}
    if action == "extract":
        return {"background": "transparent"}
    if action == "remove":
        return {"fill_mode": "background_color", "color": "#FFFFFF"}
    candidates = ("#E63946", "#2A9D8F", "#7B2CBF", "#F4A261")
    start = seed % len(candidates)
    color = next(
        candidates[(start + offset) % len(candidates)]
        for offset in range(len(candidates))
        if candidates[(start + offset) % len(candidates)].lower()
        != current_color.lower()
    )
    return {"color": color}


def _target_type(chart_type: str) -> str:
    return {
        "line": "curve",
        "bar": "bar",
        "scatter": "scatter_series",
        "confidence_band": "confidence_band",
    }[chart_type]


def _hex_blend(first: str, second: str, second_weight: float) -> str:
    first_rgb = tuple(int(first[index : index + 2], 16) for index in (1, 3, 5))
    second_rgb = tuple(int(second[index : index + 2], 16) for index in (1, 3, 5))
    result = tuple(
        round((1 - second_weight) * left + second_weight * right)
        for left, right in zip(first_rgb, second_rgb)
    )
    return "#" + "".join(f"{value:02X}" for value in result)


def _validate_clean_target(output_dir: Path) -> None:
    if not str(output_dir).strip():
        raise ValueError("--clean output directory must not be empty")
    resolved = output_dir.expanduser().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden or len(resolved.parts) < 3:
        raise ValueError(f"refusing unsafe --clean target: {resolved}")
    if "synthetic_v1" not in resolved.name:
        raise ValueError(
            "refusing --clean target whose final directory name does not contain "
            "'synthetic_v1'"
        )


def _clean_owned_outputs(output_dir: Path) -> None:
    for directory in (output_dir / "images", output_dir / "masks"):
        if directory.exists():
            shutil.rmtree(directory)
    for filename in ("annotations.jsonl", "summary.json", "audit.json"):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _write_summary(
    output_dir: Path, annotations: list[dict[str, Any]], seed: int
) -> None:
    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(item[field] for item in annotations).items()))

    summary = {
        "generator_version": GENERATOR_VERSION_V1,
        "command": (
            "python projects/chartground_edit/scripts/generate_synthetic_v1.py "
            f"--output-dir projects/chartground_edit/data/synthetic_v1 --seed {seed} --clean"
        ),
        "seed": seed,
        "sample_count": len(annotations),
        "split_counts": counts("split"),
        "chart_type_counts": counts("chart_type"),
        "referring_type_counts": counts("referring_type"),
        "edit_action_counts": counts("edit_action"),
        "difficulty_counts": counts("difficulty"),
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
