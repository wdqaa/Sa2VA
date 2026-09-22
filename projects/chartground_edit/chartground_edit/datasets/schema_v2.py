"""Strict, v1-compatible validation for ChartGround-Edit synthetic v2."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .schema import SchemaValidationError
from .schema_v1 import (
    CHART_TYPES_V1,
    DIFFICULTIES_V1,
    EDIT_ACTIONS_V1,
    MASK_SEMANTICS_VERSION,
    REFERRING_TYPES_V1,
    SPLITS_V1,
    TARGET_TYPES_V1,
)


SCHEMA_VERSION_V2 = "chartground-edit-v2"
GENERATOR_VERSION_V2 = "synthetic-v2.0.0"
REQUIRED_FIELDS_V2 = (
    "schema_version", "sample_id", "image_path", "mask_path", "split",
    "chart_type", "referring_type", "full_instruction", "referring_expression",
    "edit_action", "edit_parameters", "target_type", "target_attributes",
    "mask_semantics_version", "generator_version", "seed", "scene_id",
    "content_id", "style_family", "instruction_template_family", "difficulty",
    "distractor_count", "image_width", "image_height", "generation_metadata",
    "diversity_metadata",
)

_TARGET_BY_CHART = {
    "line": "curve", "bar": "bar", "scatter": "scatter_series",
    "confidence_band": "confidence_band",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_DIVERSITY_REQUIRED = {
    "canvas_size", "dpi", "aspect_family", "legend_position", "legend_columns",
    "theme", "grid_mode", "font_family", "font_size", "axis_scale",
    "numeric_format", "layout_density", "title", "subtitle", "axis_labels",
    "series_count", "color_distance", "line_width", "marker_size",
    "marker_family", "crossing_count", "occlusion_ratio", "degradation",
    "difficulty_score", "difficulty_factors", "data_signature",
}


def validate_annotation_v2(
    annotation: Mapping[str, Any], *, base_dir: str | Path | None = None,
    check_files: bool = False,
) -> None:
    if not isinstance(annotation, Mapping):
        raise SchemaValidationError("annotation must be a JSON object")
    missing = sorted(set(REQUIRED_FIELDS_V2) - set(annotation))
    unknown = sorted(set(annotation) - set(REQUIRED_FIELDS_V2))
    if missing:
        raise SchemaValidationError(f"missing required fields: {', '.join(missing)}")
    if unknown:
        raise SchemaValidationError(f"unknown fields: {', '.join(unknown)}")
    for field in REQUIRED_FIELDS_V2:
        if field in {"edit_parameters", "target_attributes", "generation_metadata", "diversity_metadata", "seed", "distractor_count", "image_width", "image_height"}:
            continue
        if type(annotation[field]) is not str or not annotation[field].strip():
            raise SchemaValidationError(f"{field!r} must be a nonempty string")
    if annotation["schema_version"] != SCHEMA_VERSION_V2:
        raise SchemaValidationError(f"schema_version must be {SCHEMA_VERSION_V2!r}")
    if annotation["generator_version"] != GENERATOR_VERSION_V2:
        raise SchemaValidationError(f"generator_version must be {GENERATOR_VERSION_V2!r}")
    if annotation["mask_semantics_version"] != MASK_SEMANTICS_VERSION:
        raise SchemaValidationError(f"mask_semantics_version must be {MASK_SEMANTICS_VERSION!r}")
    enums = {
        "chart_type": CHART_TYPES_V1, "referring_type": REFERRING_TYPES_V1,
        "edit_action": EDIT_ACTIONS_V1, "target_type": TARGET_TYPES_V1,
        "split": SPLITS_V1, "difficulty": DIFFICULTIES_V1,
    }
    for field, allowed in enums.items():
        if annotation[field] not in allowed:
            raise SchemaValidationError(f"{field!r} must be one of {sorted(allowed)}")
    if annotation["target_type"] != _TARGET_BY_CHART[annotation["chart_type"]]:
        raise SchemaValidationError("target_type does not match chart_type")
    for field in ("seed", "distractor_count", "image_width", "image_height"):
        if type(annotation[field]) is not int or annotation[field] < 0:
            raise SchemaValidationError(f"{field!r} must be a non-negative integer")
    if annotation["distractor_count"] < 1:
        raise SchemaValidationError("distractor_count must be at least one")
    if min(annotation["image_width"], annotation["image_height"]) <= 0:
        raise SchemaValidationError("image dimensions must be positive")
    for field in ("sample_id", "scene_id", "content_id"):
        if _SAFE_ID.fullmatch(annotation[field]) is None:
            raise SchemaValidationError(f"unsafe identifier in {field!r}")
    for field in ("image_path", "mask_path"):
        path = Path(annotation[field])
        if path.is_absolute() or ".." in path.parts:
            raise SchemaValidationError(f"{field!r} must be a safe relative path")
    if annotation["image_path"] == annotation["mask_path"]:
        raise SchemaValidationError("image_path and mask_path must differ")
    for field in ("edit_parameters", "target_attributes", "generation_metadata", "diversity_metadata"):
        if type(annotation[field]) is not dict or not annotation[field]:
            raise SchemaValidationError(f"{field!r} must be a nonempty object")
    _validate_edit_parameters(annotation["edit_action"], annotation["edit_parameters"])
    if annotation["referring_expression"] not in annotation["full_instruction"]:
        raise SchemaValidationError("referring_expression must occur in full_instruction")
    diversity = annotation["diversity_metadata"]
    diversity_missing = sorted(_DIVERSITY_REQUIRED - set(diversity))
    if diversity_missing:
        raise SchemaValidationError(f"diversity_metadata missing: {', '.join(diversity_missing)}")
    if diversity["series_count"] != annotation["distractor_count"] + 1:
        raise SchemaValidationError("series_count and distractor_count disagree")
    if diversity["canvas_size"] != [annotation["image_width"], annotation["image_height"]]:
        raise SchemaValidationError("canvas_size and image dimensions disagree")
    if not check_files:
        return
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    image_path, mask_path = root / annotation["image_path"], root / annotation["mask_path"]
    if not image_path.is_file() or not mask_path.is_file():
        raise SchemaValidationError("referenced image or mask file is missing")
    with Image.open(image_path) as image:
        image_size = image.size
    with Image.open(mask_path) as source:
        mask_size, mask_mode, mask_format = source.size, source.mode, source.format
        mask = np.asarray(source)
    expected = (annotation["image_width"], annotation["image_height"])
    if image_size != expected or mask_size != expected:
        raise SchemaValidationError("image/mask size does not match annotation")
    if mask_mode != "L" or mask_format != "PNG" or mask.ndim != 2:
        raise SchemaValidationError("mask must be a single-channel PNG")
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}) or not np.any(mask == 255):
        raise SchemaValidationError("mask must be binary and nonempty")


def validate_jsonl_v2(
    path: str | Path, *, check_files: bool = False, expected_count: int | None = None,
) -> list[dict[str, Any]]:
    manifest = Path(path)
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SchemaValidationError(f"blank line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"invalid JSON at line {line_number}: {exc}") from exc
            try:
                validate_annotation_v2(record, base_dir=manifest.parent, check_files=check_files)
            except SchemaValidationError as exc:
                raise SchemaValidationError(f"line {line_number}: {exc}") from exc
            records.append(record)
    if expected_count is not None and len(records) != expected_count:
        raise SchemaValidationError(f"expected {expected_count} records, got {len(records)}")
    return records


def _validate_edit_parameters(action: str, parameters: dict[str, Any]) -> None:
    expected = {
        "highlight": {"strength"}, "recolor": {"color"},
        "extract": {"background"}, "remove": {"fill_mode", "color"},
    }[action]
    if set(parameters) != expected:
        raise SchemaValidationError(f"invalid edit_parameters for {action}")
    if action == "highlight" and (
        type(parameters["strength"]) not in (int, float)
        or isinstance(parameters["strength"], bool)
        or not 0 <= float(parameters["strength"]) <= 1
    ):
        raise SchemaValidationError("highlight strength must be numeric in [0, 1]")
    if action == "extract" and parameters["background"] != "transparent":
        raise SchemaValidationError("extract background must be transparent")
    if action in {"recolor", "remove"} and _HEX_COLOR.fullmatch(str(parameters["color"])) is None:
        raise SchemaValidationError("edit color must be an RGB hex string")
    if action == "remove" and parameters["fill_mode"] != "background_color":
        raise SchemaValidationError("remove fill_mode must be background_color")
