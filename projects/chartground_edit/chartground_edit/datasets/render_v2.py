"""Pillow renderer for the visually diverse synthetic v2 protocol."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


SIZE_FAMILIES = (
    (720, 360, "wide"), (672, 448, "landscape"), (560, 420, "classic"),
    (500, 500, "square"), (448, 560, "portrait"),
)
LEGEND_POSITIONS = ("top", "bottom", "left", "right", "inside")
THEMES = ("light", "dark")
GRID_MODES = ("none", "major", "major_minor")
LAYOUT_DENSITIES = ("compact", "standard", "whitespace")
NUMERIC_FORMATS = ("standard", "scientific", "percent", "negative")
AXIS_SCALES = (("linear", "linear"), ("log", "linear"), ("linear", "log"))
FONT_FAMILIES = ("DejaVu Sans", "Lato", "Nimbus Sans")
MARKERS = ("circle", "square", "diamond", "triangle", "none")

_FONT_PATHS = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/lato/Lato-Regular.ttf"),
    Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"),
)
_DISTINCT = ("#2774B8", "#D94F43", "#2B9348", "#8B5CF6", "#F59E0B", "#0891B2")
_SIMILAR = ("#426B88", "#4D728C", "#587A91", "#638297", "#6E8A9C", "#7992A2")
_DARK_DISTINCT = ("#54A6FF", "#FF7167", "#62D48B", "#B895FF", "#FFC857", "#4DD9E8")
_NAMES = ("Orion", "Lyra", "Cygnus", "Vela", "Draco", "Aquila")
_CATEGORIES = ("Group A", "Group B", "Group C", "Group D", "Group E", "Group F")
_TREND_LABELS = (
    "steady increase", "steady decrease", "periodic", "piecewise rise",
    "plateau", "noisy exponential",
)


def render_scene_v2(
    plan: dict[str, Any],
) -> tuple[Image.Image, Image.Image, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    rng = np.random.default_rng(plan["seed"])
    width, height = plan["canvas_size"]
    antialias = plan["antialias"]
    render_width, render_height = width * antialias, height * antialias
    scale = float(antialias)
    colors = _palette(plan["difficulty"], plan["theme"])
    entity_count = plan["distractor_count"] + 1
    target_index = int(rng.integers(0, entity_count))
    boxes = _layout_boxes(render_width, render_height, plan["legend_position"], plan["layout_density"])
    image, font, background, foreground = _base_canvas(plan, boxes, scale)
    mask = Image.new("L", (render_width, render_height), 0)
    renderer = {
        "line": _render_line,
        "bar": _render_bar,
        "scatter": _render_scatter,
        "confidence_band": _render_band,
    }[plan["chart_type"]]
    image, entities, geometry, content, signals = renderer(
        image, mask, boxes["data"], rng, plan, entity_count, target_index,
        colors, background, scale,
    )
    # Keep the semantic target strictly inside the plotted data region. This
    # clips only anti-aliased glyph fringes at axes; legend proxies are never
    # drawn into the mask.
    mask_array = np.asarray(mask).copy()
    data_left, data_top, data_right, data_bottom = boxes["data"]
    clipped = np.zeros_like(mask_array)
    clipped[data_top : data_bottom + 1, data_left : data_right + 1] = mask_array[
        data_top : data_bottom + 1, data_left : data_right + 1
    ]
    mask = Image.fromarray(clipped)
    legend_box, glyph_boxes = _draw_legend(
        image, boxes["legend"], entities, plan, font, foreground, background, scale
    )
    if antialias > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        mask_array = np.asarray(mask.resize((width, height), Image.Resampling.LANCZOS)) >= 128
        mask = Image.fromarray(mask_array.astype(np.uint8) * 255)
        geometry = _scale_geometry(geometry, 1.0 / antialias)
        boxes = {key: _scale_box(value, 1.0 / antialias) for key, value in boxes.items()}
        legend_box = _scale_box(legend_box, 1.0 / antialias)
        glyph_boxes = [_scale_box(box, 1.0 / antialias) for box in glyph_boxes]
    image, mask = _apply_degradation(image, mask, plan)
    target = entities[target_index]
    reference_key = {
        "category": "category", "appearance": "color",
        "legend": "legend_label", "trend": "trend",
    }[plan["referring_type"]]
    reference_value = target[reference_key]
    match_count = sum(entity[reference_key] == reference_value for entity in entities)
    if match_count != 1:
        raise RuntimeError(f"ambiguous v2 reference for {plan['sample_id']}")
    content_digest = hashlib.sha256()
    content_digest.update(plan["chart_type"].encode("ascii"))
    content_digest.update(np.asarray(content, dtype="<f8").tobytes())
    content_digest.update(str(plan["seed"]).encode("ascii"))
    content_sha = content_digest.hexdigest()
    foreground_ratio = float((np.asarray(mask) > 0).mean())
    color_distance = _minimum_color_distance(target["color"], [entity["color"] for entity in entities if entity is not target])
    difficulty_factors = {
        "color_similarity": round(1.0 - color_distance, 6),
        "geometry_overlap": round(float(signals["occlusion_ratio"]), 6),
        "series_density": round((entity_count - 2) / 4.0, 6),
        "marker_similarity": round(float(signals["marker_similarity"]), 6),
        "legend_complexity": round(min(1.0, entity_count / max(plan["legend_columns"], 1) / 5.0), 6),
        "small_target": round(max(0.0, 1.0 - foreground_ratio / 0.06), 6),
        "thin_glyph": round(float(signals["thin_glyph"]), 6),
        "band_overlap": round(float(signals["band_overlap"]), 6),
    }
    difficulty_score = float(np.mean(list(difficulty_factors.values())))
    attributes = {
        "target_index": target_index,
        "category": target["category"],
        "color": target["color"],
        "legend_label": target["legend_label"],
        "trend": target["trend"],
        "includes_legend_proxy": False,
        "includes_markers": bool(geometry.get("marker_centers")),
    }
    generation_metadata = {
        "render_backend": "Pillow",
        "entity_count": entity_count,
        "target_index": target_index,
        "reference_key": reference_key,
        "reference_value": reference_value,
        "reference_match_count": match_count,
        "entity_descriptors": entities,
        "plot_box": boxes["plot"],
        "data_box": boxes["data"],
        "legend_box": legend_box,
        "legend_glyph_boxes": glyph_boxes,
        "target_geometry": geometry,
        "content_sha256": content_sha,
    }
    diversity = {
        "canvas_size": [width, height],
        "dpi": plan["dpi"],
        "aspect_family": plan["aspect_family"],
        "legend_position": plan["legend_position"],
        "legend_columns": plan["legend_columns"],
        "theme": plan["theme"],
        "grid_mode": plan["grid_mode"],
        "font_family": plan["font_family"],
        "font_size": plan["font_size"],
        "axis_scale": list(plan["axis_scale"]),
        "numeric_format": plan["numeric_format"],
        "layout_density": plan["layout_density"],
        "title": plan["show_title"],
        "subtitle": plan["show_subtitle"],
        "axis_labels": plan["show_axis_labels"],
        "series_count": entity_count,
        "color_distance": round(color_distance, 6),
        "line_width": round(float(signals["line_width"]), 3),
        "marker_size": round(float(signals["marker_size"]), 3),
        "marker_family": signals["marker_family"],
        "crossing_count": int(signals["crossing_count"]),
        "occlusion_ratio": round(float(signals["occlusion_ratio"]), 6),
        "foreground_ratio": round(foreground_ratio, 8),
        "degradation": {
            "jpeg_quality": plan["jpeg_quality"],
            "blur_radius": plan["blur_radius"],
            "screenshot_scale": plan["screenshot_scale"],
            "antialias_factor": antialias,
        },
        "difficulty_score": round(difficulty_score, 6),
        "difficulty_plan_score": plan["difficulty_plan_score"],
        "difficulty_factors": difficulty_factors,
        "data_signature": content_sha,
        "crossing_family": plan["crossing_family"],
        "curve_family": plan["curve_family"],
    }
    return image, mask, attributes, generation_metadata, diversity, f"content_{content_sha[:20]}"


def _base_canvas(
    plan: dict[str, Any], boxes: dict[str, list[int]], scale: float
) -> tuple[Image.Image, ImageFont.ImageFont, tuple[int, int, int], tuple[int, int, int]]:
    dark = plan["theme"] == "dark"
    split_backgrounds = {
        "train": ((250, 251, 252), (27, 31, 38)),
        "val": ((240, 237, 226), (39, 27, 32)),
        "test": ((226, 241, 250), (23, 38, 43)),
    }
    background = split_backgrounds[plan["split"]][1 if dark else 0]
    foreground = (231, 235, 240) if dark else (31, 39, 49)
    grid = (67, 74, 84) if dark else (215, 220, 226)
    minor = (49, 55, 64) if dark else (234, 237, 240)
    image = Image.new("RGB", (boxes["canvas"][2], boxes["canvas"][3]), background)
    draw = ImageDraw.Draw(image)
    font = _font(plan["font_family"], round(plan["font_size"] * scale))
    left, top, right, bottom = boxes["plot"]
    if plan["grid_mode"] != "none":
        fractions = (0.25, 0.5, 0.75) if plan["grid_mode"] == "major" else (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
        for index, fraction in enumerate(fractions):
            color = grid if index % 2 or plan["grid_mode"] == "major" else minor
            x = round(left + fraction * (right - left))
            y = round(top + fraction * (bottom - top))
            draw.line((x, top, x, bottom), fill=color, width=max(1, round(scale)))
            draw.line((left, y, right, y), fill=color, width=max(1, round(scale)))
    axis_width = max(2, round(1.5 * scale))
    draw.line((left, top, left, bottom), fill=foreground, width=axis_width)
    draw.line((left, bottom, right, bottom), fill=foreground, width=axis_width)
    if plan["split"] == "val":
        draw.line((left, top, right, top), fill=foreground, width=axis_width)
    elif plan["split"] == "test":
        draw.line((left, top, right, top), fill=foreground, width=axis_width)
        draw.line((right, top, right, bottom), fill=foreground, width=axis_width)
    if plan["show_title"]:
        draw.text((left, max(2, top - round(34 * scale))), plan["chart_type"].replace("_", " ").title(), fill=foreground, font=font)
    if plan["show_subtitle"]:
        draw.text((left, max(2, top - round(17 * scale))), "synthetic measurement summary", fill=foreground, font=_font(plan["font_family"], max(8, round((plan["font_size"] - 3) * scale))))
    if plan["show_axis_labels"]:
        draw.text((round((left + right) / 2), bottom + round(6 * scale)), "measurement", fill=foreground, font=font, anchor="ma")
        draw.text((max(2, left - round(9 * scale)), round((top + bottom) / 2)), "value", fill=foreground, font=font, anchor="rm")
    tick_font = _font(plan["font_family"], max(7, round((plan["font_size"] - 3) * scale)))
    x_labels = ("10⁰", "10¹", "10²") if plan["axis_scale"][0] == "log" else ("0", "50", "100")
    if plan["numeric_format"] == "scientific":
        y_labels = ("1e−3", "1e−2", "1e−1")
    elif plan["numeric_format"] == "percent":
        y_labels = ("0%", "50%", "100%")
    elif plan["numeric_format"] == "negative":
        y_labels = ("−50", "0", "50")
    elif plan["axis_scale"][1] == "log":
        y_labels = ("10⁰", "10¹", "10²")
    else:
        y_labels = ("0", "0.5", "1.0")
    for index, label in enumerate(x_labels):
        x = round(left + index * (right - left) / 2)
        draw.text((x, bottom + round(2 * scale)), label, fill=foreground, font=tick_font, anchor="ma")
    for index, label in enumerate(reversed(y_labels)):
        y = round(top + index * (bottom - top) / 2)
        draw.text((left - round(4 * scale), y), label, fill=foreground, font=tick_font, anchor="rm")
    return image, font, background, foreground


def _layout_boxes(width: int, height: int, legend_position: str, density: str) -> dict[str, list[int]]:
    margin_factor = {"compact": 0.065, "standard": 0.09, "whitespace": 0.13}[density]
    left, right = round(width * margin_factor), round(width * (1 - margin_factor))
    top, bottom = round(height * (margin_factor + 0.04)), round(height * (1 - margin_factor))
    legend_width, legend_height = round(width * 0.22), round(height * 0.19)
    if legend_position == "right":
        plot = [left, top, right - legend_width - round(width * 0.025), bottom]
        legend = [plot[2] + round(width * 0.02), top, right, top + legend_height]
    elif legend_position == "left":
        plot = [left + legend_width + round(width * 0.025), top, right, bottom]
        legend = [left, top, left + legend_width, top + legend_height]
    elif legend_position == "top":
        plot = [left, top + legend_height + round(height * 0.025), right, bottom]
        legend = [left, top, right, top + legend_height]
    elif legend_position == "bottom":
        plot = [left, top, right, bottom - legend_height - round(height * 0.025)]
        legend = [left, plot[3] + round(height * 0.02), right, bottom]
    else:
        plot = [left, top, right, bottom]
        legend = [right - legend_width, top + round(height * 0.025), right, top + round(height * 0.025) + legend_height]
    data = list(plot)
    if legend_position == "inside":
        data[2] = legend[0] - round(width * 0.025)
    return {"canvas": [0, 0, width, height], "plot": plot, "data": data, "legend": legend}


def _render_line(
    image: Image.Image, mask: Image.Image, box: list[int], rng: np.random.Generator,
    plan: dict[str, Any], count: int, target: int, colors: tuple[str, ...],
    background: tuple[int, int, int], scale: float,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any], np.ndarray, dict[str, Any]]:
    draw, mask_draw = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    x = np.linspace(0.035, 0.965, 32)
    curves = []
    for index in range(count):
        family = (index + plan["family_offset"]) % 6
        if family == 0:
            values = 0.16 + 0.66 * x
        elif family == 1:
            values = 0.82 - 0.62 * x
        elif family == 2:
            values = 0.5 + 0.24 * np.sin((2 + index % 2) * np.pi * x + 0.5 * index)
        elif family == 3:
            values = np.where(x < 0.45, 0.2 + 0.7 * x, 0.515 + 0.2 * (x - 0.45))
        elif family == 4:
            values = 0.22 + 0.55 * np.minimum(x / 0.55, 1.0)
        else:
            values = 0.18 + 0.11 * np.exp(1.8 * x) + rng.normal(0, 0.025, len(x))
        if plan["crossing_family"] == "none":
            values = 0.12 + 0.12 * (values - values.min()) / max(np.ptp(values), 1e-6) + index * (0.7 / max(count, 2))
        elif plan["crossing_family"] == "single":
            cross = 0.25 + 0.12 * (plan["family_offset"] % 4)
            slope = (index - (count - 1) / 2) * 0.28
            values = 0.5 + slope * (x - cross) + (index % 2) * 0.03
        else:
            values = 0.5 + 0.22 * np.sin((index % 3 + 1.5) * 2 * np.pi * x + index * 0.9)
        values = values + rng.normal(0.0, 0.009, len(x))
        curves.append(np.clip(values, 0.055, 0.945))
    curves_array = np.asarray(curves)
    widths = [max(1, round(value * scale)) for value in (1.2, 1.8, 2.5, 3.2)]
    width = widths[(plan["family_offset"] + target) % len(widths)]
    marker = MARKERS[(plan["family_offset"] + target) % len(MARKERS)]
    marker_radius = max(2, round((2.5 + (plan["family_offset"] % 4) * 1.2) * scale))
    dash_styles = ("solid", "dash", "dot_dash")
    entities = _entities(count, colors)
    target_centers: list[list[int]] = []
    target_points: list[tuple[int, int]] = []
    for index, values in enumerate(curves_array):
        points = _points(box, x, values)
        style = dash_styles[(index + plan["family_offset"]) % 3]
        _polyline(draw, points, entities[index]["color"], width, style)
        series_marker = MARKERS[(index + plan["family_offset"]) % len(MARKERS)]
        if series_marker != "none":
            for point in points[::4]:
                _marker(draw, point, marker_radius, series_marker, entities[index]["color"])
        if index == target:
            target_points = points
            _polyline(mask_draw, points, 255, max(width, round(2 * scale)), style)
            if series_marker != "none":
                for point in points[::4]:
                    _marker(mask_draw, point, marker_radius, series_marker, 255)
                    target_centers.append([point[0], point[1]])
            marker = series_marker
    crossings = _crossing_count(curves_array[target], [curve for i, curve in enumerate(curves_array) if i != target])
    occlusion = _curve_occlusion(curves_array, target)
    signals = _signals(width / scale, marker_radius / scale, marker, crossings, occlusion)
    return image, entities, {"marker_centers": target_centers, "polyline": [[x, y] for x, y in target_points], "line_width": max(width, round(2 * scale)) / scale}, curves_array, signals


def _render_bar(
    image: Image.Image, mask: Image.Image, box: list[int], rng: np.random.Generator,
    plan: dict[str, Any], count: int, target: int, colors: tuple[str, ...],
    background: tuple[int, int, int], scale: float,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any], np.ndarray, dict[str, Any]]:
    draw, mask_draw = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    category_count = 4 + plan["family_offset"] % 4
    values = rng.uniform(0.22, 0.92, (count, category_count))
    for index in range(count):
        mode = index % 4
        if mode == 0:
            values[index] = np.linspace(0.25, 0.88, category_count)
        elif mode == 1:
            values[index] = np.linspace(0.85, 0.28, category_count)
        elif mode == 2:
            values[index] = 0.35 + 0.42 * np.sin(np.linspace(0, np.pi, category_count))
    values += rng.normal(0.0, 0.012, values.shape)
    values = np.clip(values, 0.08, 0.96)
    negative = plan["numeric_format"] == "negative"
    if negative:
        values[:, ::2] *= -1
    left, top, right, bottom = box
    zero_y = round((top + bottom) / 2) if negative else bottom
    draw.line((left, zero_y, right, zero_y), fill=(95, 101, 110), width=max(1, round(scale)))
    group_width = (right - left) / category_count
    gap_fraction = (0.08, 0.18, 0.3)[plan["family_offset"] % 3]
    bar_width = max(2, group_width * (1 - gap_fraction) / count)
    entities = _entities(count, colors)
    target_boxes: list[list[int]] = []
    for category in range(category_count):
        group_left = left + category * group_width + group_width * gap_fraction / 2
        for index in range(count):
            x0 = round(group_left + index * bar_width)
            x1 = max(x0 + 1, round(x0 + bar_width * 0.9))
            value = float(values[index, category])
            if negative:
                y = round(zero_y - value * (bottom - top) * 0.46)
                y0, y1 = sorted((zero_y, y))
            else:
                y0, y1 = round(bottom - value * (bottom - top)), bottom
            draw.rectangle((x0, y0, x1, y1), fill=entities[index]["color"], outline=(35, 39, 44), width=max(1, round(scale)))
            if index == target:
                mask_draw.rectangle((x0, y0, x1, y1), fill=255)
                target_boxes.append([x0, y0, x1, y1])
    font = _font(plan["font_family"], max(7, round((plan["font_size"] - 2) * scale)))
    labels = ("North sector", "Long category label", "Central", "South", "Auxiliary", "Reference", "Tail")
    for category in range(category_count):
        x = round(left + (category + 0.5) * group_width)
        draw.text((x, bottom + round(3 * scale)), labels[category], fill=(90, 96, 105), font=font, anchor="ma")
    overlap = max(0.0, (count * bar_width - group_width * 0.92) / max(group_width, 1))
    signals = _signals(bar_width / scale, 0.0, "none", 0, overlap)
    return image, entities, {"bar_boxes": target_boxes}, values, signals


def _render_scatter(
    image: Image.Image, mask: Image.Image, box: list[int], rng: np.random.Generator,
    plan: dict[str, Any], count: int, target: int, colors: tuple[str, ...],
    background: tuple[int, int, int], scale: float,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any], np.ndarray, dict[str, Any]]:
    draw, mask_draw = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    points_per_series = (14, 24, 42, 70)[plan["family_offset"] % 4]
    entities = _entities(count, colors)
    all_values, target_centers = [], []
    target_binary = Image.new("1", image.size, 0)
    other_binary = Image.new("1", image.size, 0)
    for index in range(count):
        x = np.sort(rng.uniform(0.04, 0.96, points_per_series))
        slope = np.linspace(-0.62, 0.62, count)[index]
        cluster_shift = (index - (count - 1) / 2) * (0.025 if plan["difficulty"] == "hard" else 0.09)
        y = 0.5 + slope * (x - 0.5) + cluster_shift + rng.normal(0, 0.035 + 0.012 * (index % 3), points_per_series)
        if index == (plan["family_offset"] % count):
            y[-1] = min(0.95, y[-1] + 0.28)
        y = np.clip(y, 0.04, 0.96)
        values = np.stack([x, y], axis=1)
        all_values.append(values)
        centers = _points(box, x, y)
        radius = max(2, round((2.4 + ((index + plan["family_offset"]) % 4) * 1.2) * scale))
        marker = MARKERS[(index + plan["family_offset"]) % 4]
        blended = _blend_hex(entities[index]["color"], background, (0.45, 0.62, 0.78, 0.92)[(index + plan["family_offset"]) % 4])
        for point in centers:
            _marker(draw, point, radius, marker, blended)
            _marker(target_binary if index == target else other_binary, point, radius, marker, 1)
            if index == target:
                _marker(mask_draw, point, radius, marker, 255)
                target_centers.append([point[0], point[1]])
    target_arr, other_arr = np.asarray(target_binary), np.asarray(other_binary)
    occlusion = float(np.logical_and(target_arr, other_arr).sum() / max(target_arr.sum(), 1))
    target_radius = 2.4 + ((target + plan["family_offset"]) % 4) * 1.2
    target_marker = MARKERS[(target + plan["family_offset"]) % 4]
    signals = _signals(target_radius * 2, target_radius, target_marker, 0, occlusion)
    return image, entities, {"marker_centers": target_centers}, np.concatenate(all_values), signals


def _render_band(
    image: Image.Image, mask: Image.Image, box: list[int], rng: np.random.Generator,
    plan: dict[str, Any], count: int, target: int, colors: tuple[str, ...],
    background: tuple[int, int, int], scale: float,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any], np.ndarray, dict[str, Any]]:
    x = np.linspace(0.02, 0.98, 36)
    entities = _entities(count, colors)
    arrays, target_polygon = [], []
    target_binary = Image.new("1", image.size, 0)
    other_binary = Image.new("1", image.size, 0)
    for index in range(count):
        center = 0.22 + index * 0.55 / max(count - 1, 1)
        center_values = center + 0.08 * np.sin(2 * np.pi * x + index * 0.8)
        if plan["crossing_family"] != "none":
            center_values = 0.5 + (index - (count - 1) / 2) * 0.22 * (x - (0.3 + 0.1 * (plan["family_offset"] % 4)))
            if plan["crossing_family"] == "multiple":
                center_values += 0.11 * np.sin((index % 3 + 1) * 2 * np.pi * x + index)
        center_values += rng.normal(0.0, 0.007, len(x))
        base_width = (0.045, 0.075, 0.115, 0.16)[(index + plan["family_offset"]) % 4] * float(rng.uniform(0.94, 1.06))
        lower_width = base_width * (0.7 + 0.25 * np.sin(2 * np.pi * x + index))
        upper_width = base_width * (1.15 + 0.2 * np.cos(2 * np.pi * x + index * 0.4))
        lower = np.clip(center_values - lower_width, 0.02, 0.98)
        upper = np.clip(center_values + upper_width, 0.02, 0.98)
        top_points = _points(box, x, upper)
        bottom_points = _points(box, x[::-1], lower[::-1])
        polygon = top_points + bottom_points
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        alpha = (70, 105, 145, 180)[(index + plan["family_offset"]) % 4]
        rgb = _hex_rgb(entities[index]["color"])
        layer_draw.polygon(polygon, fill=(*rgb, alpha))
        line_alpha = 90 if (index + plan["family_offset"]) % 3 == 0 else 240
        line_points = _points(box, x, center_values)
        layer_draw.line(line_points, fill=(*rgb, line_alpha), width=max(1, round((1.2 + index % 2) * scale)))
        image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
        destination = target_binary if index == target else other_binary
        ImageDraw.Draw(destination).polygon(polygon, fill=1)
        if index == target:
            ImageDraw.Draw(mask).polygon(polygon, fill=255)
            target_polygon = [[px, py] for px, py in polygon]
        arrays.append(np.stack([center_values, lower, upper], axis=1))
    target_arr, other_arr = np.asarray(target_binary), np.asarray(other_binary)
    overlap = float(np.logical_and(target_arr, other_arr).sum() / max(target_arr.sum(), 1))
    centers = np.asarray([item[:, 0] for item in arrays])
    crossings = _crossing_count(centers[target], [curve for i, curve in enumerate(centers) if i != target])
    width_value = float(np.mean(arrays[target][:, 2] - arrays[target][:, 1]))
    signals = _signals(1.5, 0.0, "none", crossings, overlap, band_overlap=overlap)
    return image, entities, {"band_polygon": target_polygon, "mean_band_width": width_value}, np.concatenate(arrays), signals


def _entities(count: int, colors: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {
            "category": _CATEGORIES[index], "color": colors[index],
            "legend_label": f"series-{_NAMES[index].lower()}",
            "trend": _TREND_LABELS[index],
        }
        for index in range(count)
    ]


def _draw_legend(
    image: Image.Image, box: list[int], entities: list[dict[str, str]],
    plan: dict[str, Any], font: ImageFont.ImageFont, foreground: tuple[int, int, int],
    background: tuple[int, int, int], scale: float,
) -> tuple[list[int], list[list[int]]]:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=max(2, round(5 * scale)), fill=background, outline=(125, 130, 138), width=max(1, round(scale)))
    columns = min(plan["legend_columns"], len(entities))
    rows = math.ceil(len(entities) / columns)
    cell_width = max(1, (right - left - round(12 * scale)) / columns)
    cell_height = max(round(13 * scale), (bottom - top - round(8 * scale)) / rows)
    glyph_boxes: list[list[int]] = []
    legend_font = _font(plan["font_family"], max(7, round((plan["font_size"] - 2) * scale)))
    for index, entity in enumerate(entities):
        row, column = divmod(index, columns)
        x = round(left + 6 * scale + column * cell_width)
        y = round(top + 4 * scale + row * cell_height + cell_height / 2)
        glyph = [x, y - max(2, round(3 * scale)), x + max(9, round(15 * scale)), y + max(2, round(3 * scale))]
        draw.rectangle(glyph, fill=entity["color"])
        glyph_boxes.append(glyph)
        draw.text((glyph[2] + round(4 * scale), y), entity["legend_label"], fill=foreground, font=legend_font, anchor="lm")
    return list(box), glyph_boxes


def _apply_degradation(image: Image.Image, mask: Image.Image, plan: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    scale = float(plan["screenshot_scale"])
    if scale != 1.0:
        intermediate = (max(32, round(image.width * scale)), max(32, round(image.height * scale)))
        image = image.resize(intermediate, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
        mask = mask.resize(intermediate, Image.Resampling.NEAREST).resize(mask.size, Image.Resampling.NEAREST)
    blur = float(plan["blur_radius"])
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    quality = int(plan["jpeg_quality"])
    if quality < 100:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as source:
            image = source.convert("RGB").copy()
    binary = np.asarray(mask.convert("L")) > 0
    return image, Image.fromarray(binary.astype(np.uint8) * 255)


def _points(box: list[int], x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    return [(round(left + float(px) * (right - left)), round(bottom - float(py) * (bottom - top))) for px, py in zip(x, y)]


def _polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], fill: Any, width: int, style: str) -> None:
    if style == "solid":
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    on, off = ((10, 6) if style == "dash" else (3, 4))
    for start, end in zip(points[:-1], points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1.0, math.hypot(dx, dy))
        position = 0.0
        while position < length:
            next_position = min(length, position + on)
            p0 = (round(start[0] + dx * position / length), round(start[1] + dy * position / length))
            p1 = (round(start[0] + dx * next_position / length), round(start[1] + dy * next_position / length))
            draw.line((p0, p1), fill=fill, width=width)
            position += on + off


def _marker(draw: Any, point: tuple[int, int], radius: int, marker: str, fill: Any) -> None:
    if isinstance(draw, Image.Image):
        draw = ImageDraw.Draw(draw)
    x, y = point
    if marker == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    elif marker == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=fill)
    elif marker == "diamond":
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), fill=fill)
    elif marker == "triangle":
        draw.polygon(((x, y - radius), (x + radius, y + radius), (x - radius, y + radius)), fill=fill)


def _font(family: str, size: int) -> ImageFont.ImageFont:
    index = FONT_FAMILIES.index(family) if family in FONT_FAMILIES else 0
    path = _FONT_PATHS[index]
    return ImageFont.truetype(str(path), size=max(6, size)) if path.is_file() else ImageFont.load_default()


def _palette(difficulty: str, theme: str) -> tuple[str, ...]:
    if difficulty == "hard":
        return _SIMILAR
    if theme == "dark":
        return _DARK_DISTINCT
    if difficulty == "medium":
        return (_DISTINCT[0], "#3D83BC", _DISTINCT[1], "#E06B60", _DISTINCT[2], "#51A66A")
    return _DISTINCT


def _signals(line_width: float, marker_size: float, marker: str, crossings: int, occlusion: float, *, band_overlap: float = 0.0) -> dict[str, Any]:
    return {
        "line_width": line_width, "marker_size": marker_size,
        "marker_family": marker, "crossing_count": crossings,
        "occlusion_ratio": min(1.0, max(0.0, occlusion)),
        "marker_similarity": 1.0 if marker != "none" and marker in {"circle", "square"} else 0.35,
        "thin_glyph": min(1.0, max(0.0, 1.0 - line_width / 5.0)),
        "band_overlap": min(1.0, max(0.0, band_overlap)),
    }


def _crossing_count(target: np.ndarray, others: list[np.ndarray]) -> int:
    total = 0
    for other in others:
        difference = target - other
        signs = np.sign(difference)
        signs[signs == 0] = 1
        total += int(np.count_nonzero(signs[1:] != signs[:-1]))
    return total


def _curve_occlusion(curves: np.ndarray, target: int) -> float:
    distances = np.abs(curves - curves[target])
    nearby = np.any(np.delete(distances, target, axis=0) < 0.045, axis=0)
    return float(nearby.mean())


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def _blend_hex(value: str, background: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    foreground = _hex_rgb(value)
    return tuple(round(alpha * fg + (1 - alpha) * bg) for fg, bg in zip(foreground, background))


def _minimum_color_distance(target: str, others: list[str]) -> float:
    target_rgb = np.asarray(_hex_rgb(target), dtype=np.float64)
    return min(float(np.linalg.norm(target_rgb - np.asarray(_hex_rgb(other))) / math.sqrt(3 * 255**2)) for other in others)


def _scale_box(box: list[int], factor: float) -> list[int]:
    return [round(value * factor) for value in box]


def _scale_geometry(value: Any, factor: float) -> Any:
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value):
            return [round(float(item) * factor) for item in value]
        return [_scale_geometry(item, factor) for item in value]
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key == "line_width":
                output[key] = item * factor
            elif key == "mean_band_width":
                output[key] = item
            else:
                output[key] = _scale_geometry(item, factor)
        return output
    return value
