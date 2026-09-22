"""Strict validation for the versioned ChartGround-Edit synthetic v1 protocol."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .schema import SchemaValidationError


SCHEMA_VERSION_V1 = "chartground-edit-v1"
MASK_SEMANTICS_VERSION = "chartground-edit-mask-v0"
GENERATOR_VERSION_V1 = "synthetic-v1.0.0"

CHART_TYPES_V1 = frozenset({"line", "bar", "scatter", "confidence_band"})
REFERRING_TYPES_V1 = frozenset({"category", "appearance", "legend", "trend"})
EDIT_ACTIONS_V1 = frozenset({"highlight", "recolor", "extract", "remove"})
TARGET_TYPES_V1 = frozenset({"curve", "bar", "scatter_series", "confidence_band"})
SPLITS_V1 = frozenset({"train", "val", "test"})
DIFFICULTIES_V1 = frozenset({"easy", "medium", "hard"})

REQUIRED_FIELDS_V1 = (
    "schema_version",
    "sample_id",
    "image_path",
    "mask_path",
    "split",
    "chart_type",
    "referring_type",
    "full_instruction",
    "referring_expression",
    "edit_action",
    "edit_parameters",
    "target_type",
    "target_attributes",
    "mask_semantics_version",
    "generator_version",
    "seed",
    "scene_id",
    "content_id",
    "style_family",
    "instruction_template_family",
    "difficulty",
    "distractor_count",
    "image_width",
    "image_height",
    "generation_metadata",
)

_TARGET_BY_CHART = {
    "line": "curve",
    "bar": "bar",
    "scatter": "scatter_series",
    "confidence_band": "confidence_band",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _exact_type(value: Any, expected: type, field: str) -> None:
    if type(value) is not expected:
        raise SchemaValidationError(
            f"{field!r} must be {expected.__name__}, got {type(value).__name__}"
        )


def _nonempty_string(record: Mapping[str, Any], field: str) -> str:
    value = record[field]
    _exact_type(value, str, field)
    if not value.strip():
        raise SchemaValidationError(f"{field!r} must not be empty")
    return value


def _validate_relative_path(value: str, field: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SchemaValidationError(
            f"{field!r} must be a safe relative path without '..', got {value!r}"
        )


def _validate_edit_parameters(action: str, parameters: Any) -> None:
    _exact_type(parameters, dict, "edit_parameters")
    expected_keys = {
        "highlight": {"strength"},
        "recolor": {"color"},
        "extract": {"background"},
        "remove": {"fill_mode", "color"},
    }[action]
    if set(parameters) != expected_keys:
        raise SchemaValidationError(
            f"edit_parameters for {action!r} must contain exactly "
            f"{sorted(expected_keys)}, got {sorted(parameters)}"
        )
    if action == "highlight":
        strength = parameters["strength"]
        if type(strength) not in (int, float) or isinstance(strength, bool):
            raise SchemaValidationError("edit_parameters.strength must be numeric")
        if not 0.0 <= float(strength) <= 1.0:
            raise SchemaValidationError("edit_parameters.strength must be in [0, 1]")
    elif action == "extract":
        if parameters["background"] != "transparent":
            raise SchemaValidationError(
                "extract edit_parameters.background must be 'transparent'"
            )
    else:
        color = parameters["color"]
        if type(color) is not str or _HEX_COLOR.fullmatch(color) is None:
            raise SchemaValidationError(
                "edit_parameters.color must be an RGB hex string such as '#E63946'"
            )
        if action == "remove" and parameters["fill_mode"] != "background_color":
            raise SchemaValidationError(
                "remove edit_parameters.fill_mode must be 'background_color'"
            )


def validate_annotation_v1(
    annotation: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    check_files: bool = False,
) -> None:
    """Validate one v1 record and, optionally, its referenced image and mask."""
    if not isinstance(annotation, Mapping):
        raise SchemaValidationError("annotation must be a JSON object")
    missing = [field for field in REQUIRED_FIELDS_V1 if field not in annotation]
    if missing:
        raise SchemaValidationError(f"missing required fields: {', '.join(missing)}")
    unknown = sorted(set(annotation) - set(REQUIRED_FIELDS_V1))
    if unknown:
        raise SchemaValidationError(f"unknown fields: {', '.join(unknown)}")

    for field in (
        "schema_version",
        "sample_id",
        "image_path",
        "mask_path",
        "split",
        "chart_type",
        "referring_type",
        "full_instruction",
        "referring_expression",
        "edit_action",
        "target_type",
        "mask_semantics_version",
        "generator_version",
        "scene_id",
        "content_id",
        "style_family",
        "instruction_template_family",
        "difficulty",
    ):
        _nonempty_string(annotation, field)

    if annotation["schema_version"] != SCHEMA_VERSION_V1:
        raise SchemaValidationError(
            f"schema_version must be {SCHEMA_VERSION_V1!r}"
        )
    if annotation["mask_semantics_version"] != MASK_SEMANTICS_VERSION:
        raise SchemaValidationError(
            f"mask_semantics_version must be {MASK_SEMANTICS_VERSION!r}"
        )
    if annotation["generator_version"] != GENERATOR_VERSION_V1:
        raise SchemaValidationError(
            f"generator_version must be {GENERATOR_VERSION_V1!r}"
        )

    enum_fields = {
        "chart_type": CHART_TYPES_V1,
        "referring_type": REFERRING_TYPES_V1,
        "edit_action": EDIT_ACTIONS_V1,
        "target_type": TARGET_TYPES_V1,
        "split": SPLITS_V1,
        "difficulty": DIFFICULTIES_V1,
    }
    for field, allowed in enum_fields.items():
        if annotation[field] not in allowed:
            raise SchemaValidationError(
                f"{field!r} must be one of {sorted(allowed)}, got {annotation[field]!r}"
            )

    expected_target = _TARGET_BY_CHART[annotation["chart_type"]]
    if annotation["target_type"] != expected_target:
        raise SchemaValidationError(
            f"target_type {annotation['target_type']!r} does not match "
            f"chart_type {annotation['chart_type']!r}; expected {expected_target!r}"
        )

    for field in ("seed", "distractor_count", "image_width", "image_height"):
        _exact_type(annotation[field], int, field)
        if annotation[field] < 0:
            raise SchemaValidationError(f"{field!r} must be non-negative")
    if annotation["image_width"] == 0 or annotation["image_height"] == 0:
        raise SchemaValidationError("image dimensions must be greater than zero")
    if annotation["distractor_count"] < 1:
        raise SchemaValidationError("distractor_count must be at least one")

    for field in ("sample_id", "scene_id", "content_id"):
        if _SAFE_ID.fullmatch(annotation[field]) is None:
            raise SchemaValidationError(
                f"{field!r} may contain only letters, numbers, '.', '_' and '-'"
            )
    for field in ("image_path", "mask_path"):
        _validate_relative_path(annotation[field], field)
    if annotation["image_path"] == annotation["mask_path"]:
        raise SchemaValidationError("image_path and mask_path must differ")

    _exact_type(annotation["target_attributes"], dict, "target_attributes")
    _exact_type(annotation["generation_metadata"], dict, "generation_metadata")
    if not annotation["target_attributes"]:
        raise SchemaValidationError("target_attributes must not be empty")
    if not annotation["generation_metadata"]:
        raise SchemaValidationError("generation_metadata must not be empty")
    _validate_edit_parameters(annotation["edit_action"], annotation["edit_parameters"])

    expression = annotation["referring_expression"]
    if expression not in annotation["full_instruction"]:
        raise SchemaValidationError(
            "referring_expression must occur verbatim in full_instruction"
        )

    if not check_files:
        return
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    image_path = root / annotation["image_path"]
    mask_path = root / annotation["mask_path"]
    for label, path in (("image", image_path), ("mask", mask_path)):
        if not path.is_file():
            raise SchemaValidationError(f"{label} file does not exist: {path}")
    try:
        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as mask:
            mask_format = mask.format
            mask_mode = mask.mode
            mask_size = mask.size
            mask_array = np.asarray(mask)
    except (OSError, ValueError) as exc:
        raise SchemaValidationError(f"could not read sample files: {exc}") from exc
    expected_size = (annotation["image_width"], annotation["image_height"])
    if image_size != expected_size or mask_size != expected_size:
        raise SchemaValidationError(
            f"declared size {expected_size} does not match image {image_size} "
            f"and mask {mask_size}"
        )
    if mask_format != "PNG" or mask_mode != "L":
        raise SchemaValidationError(
            f"mask must be single-channel lossless PNG; got format={mask_format!r}, "
            f"mode={mask_mode!r}"
        )
    values = set(np.unique(mask_array).tolist())
    if not values.issubset({0, 255}):
        raise SchemaValidationError(
            f"mask contains values outside 0 and 255: {sorted(values)}"
        )
    foreground = int(np.count_nonzero(mask_array))
    if foreground == 0:
        raise SchemaValidationError("mask must not be empty")
    if foreground == mask_array.size:
        raise SchemaValidationError("mask must not cover the full image")


def validate_jsonl_v1(
    manifest_path: str | Path,
    *,
    check_files: bool = True,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Read v1 JSONL and enforce record plus manifest-level uniqueness."""
    path = Path(manifest_path)
    annotations: list[dict[str, Any]] = []
    unique_fields = {
        field: set()
        for field in (
            "sample_id",
            "image_path",
            "mask_path",
            "seed",
            "scene_id",
            "content_id",
        )
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise SchemaValidationError(f"line {line_number} is blank")
            try:
                annotation = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    f"line {line_number} is invalid JSON: {exc.msg}"
                ) from exc
            try:
                validate_annotation_v1(
                    annotation, base_dir=path.parent, check_files=check_files
                )
            except SchemaValidationError as exc:
                raise SchemaValidationError(f"line {line_number}: {exc}") from exc
            for field, seen in unique_fields.items():
                value = annotation[field]
                if value in seen:
                    raise SchemaValidationError(
                        f"line {line_number}: duplicate {field} {value!r}"
                    )
                seen.add(value)
            annotations.append(annotation)
    if expected_count is not None and len(annotations) != expected_count:
        raise SchemaValidationError(
            f"expected {expected_count} annotations, found {len(annotations)}"
        )
    return annotations
