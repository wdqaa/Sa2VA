"""Strict validation for the ChartGround-Edit JSONL v0 protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


SCHEMA_VERSION = "chartground-edit-v0"
CHART_TYPES = frozenset({"line", "bar", "scatter", "confidence_band"})
REFERRING_TYPES = frozenset({"category", "appearance", "legend", "trend"})
EDIT_ACTIONS = frozenset({"highlight", "recolor", "extract", "remove"})
TARGET_TYPES = frozenset({"curve", "bar", "scatter_series", "confidence_band"})
SPLITS = frozenset({"train", "val", "test"})

REQUIRED_FIELDS = (
    "sample_id",
    "image_path",
    "mask_path",
    "width",
    "height",
    "chart_type",
    "referring_type",
    "instruction",
    "target_type",
    "target_attributes",
    "edit_action",
    "split",
    "generator_seed",
    "metadata",
)

_TARGET_BY_CHART = {
    "line": "curve",
    "bar": "bar",
    "scatter": "scatter_series",
    "confidence_band": "confidence_band",
}


class SchemaValidationError(ValueError):
    """Raised when an annotation or one of its referenced files is invalid."""


def _require_exact_type(value: Any, expected: type, field: str) -> None:
    if type(value) is not expected:
        raise SchemaValidationError(
            f"{field!r} must be {expected.__name__}, got {type(value).__name__}"
        )


def _require_nonempty_string(annotation: Mapping[str, Any], field: str) -> None:
    value = annotation[field]
    _require_exact_type(value, str, field)
    if not value.strip():
        raise SchemaValidationError(f"{field!r} must not be empty")


def _resolve_path(path_value: str, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def validate_annotation(
    annotation: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    check_files: bool = False,
) -> None:
    """Validate one annotation and optionally its image/mask files.

    Raises:
        SchemaValidationError: if any v0 schema or file invariant is violated.
    """
    if not isinstance(annotation, Mapping):
        raise SchemaValidationError("annotation must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in annotation]
    if missing:
        raise SchemaValidationError(f"missing required fields: {', '.join(missing)}")
    unknown = sorted(set(annotation) - set(REQUIRED_FIELDS))
    if unknown:
        raise SchemaValidationError(f"unknown fields: {', '.join(unknown)}")

    for field in ("sample_id", "image_path", "mask_path", "instruction"):
        _require_nonempty_string(annotation, field)

    for field in ("width", "height"):
        value = annotation[field]
        _require_exact_type(value, int, field)
        if value <= 0:
            raise SchemaValidationError(f"{field!r} must be greater than zero")

    generator_seed = annotation["generator_seed"]
    _require_exact_type(generator_seed, int, "generator_seed")
    if generator_seed < 0:
        raise SchemaValidationError("'generator_seed' must be non-negative")

    enum_fields = {
        "chart_type": CHART_TYPES,
        "referring_type": REFERRING_TYPES,
        "target_type": TARGET_TYPES,
        "edit_action": EDIT_ACTIONS,
        "split": SPLITS,
    }
    for field, allowed in enum_fields.items():
        value = annotation[field]
        _require_exact_type(value, str, field)
        if value not in allowed:
            raise SchemaValidationError(
                f"{field!r} must be one of {sorted(allowed)}, got {value!r}"
            )

    expected_target = _TARGET_BY_CHART[annotation["chart_type"]]
    if annotation["target_type"] != expected_target:
        raise SchemaValidationError(
            f"target_type {annotation['target_type']!r} does not match "
            f"chart_type {annotation['chart_type']!r}; expected {expected_target!r}"
        )

    for field in ("target_attributes", "metadata"):
        _require_exact_type(annotation[field], dict, field)

    metadata = annotation["metadata"]
    for field in ("schema_version", "generator_version", "source"):
        if field not in metadata:
            raise SchemaValidationError(f"metadata is missing required field {field!r}")
        if type(metadata[field]) is not str or not metadata[field].strip():
            raise SchemaValidationError(f"metadata.{field} must be a non-empty string")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"metadata.schema_version must be {SCHEMA_VERSION!r}"
        )

    if not check_files:
        return

    root = Path(base_dir) if base_dir is not None else None
    image_path = _resolve_path(annotation["image_path"], root)
    mask_path = _resolve_path(annotation["mask_path"], root)
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
            mask_values = set(np.unique(np.asarray(mask)).tolist())
    except (OSError, ValueError) as exc:
        raise SchemaValidationError(f"could not read sample files: {exc}") from exc

    expected_size = (annotation["width"], annotation["height"])
    if image_size != expected_size or mask_size != expected_size:
        raise SchemaValidationError(
            f"declared size {expected_size} does not match image {image_size} "
            f"and mask {mask_size}"
        )
    if mask_format != "PNG":
        raise SchemaValidationError(f"mask must be PNG, got {mask_format!r}")
    if mask_mode != "L":
        raise SchemaValidationError(f"mask must use mode 'L', got {mask_mode!r}")
    if not mask_values.issubset({0, 255}):
        raise SchemaValidationError(
            f"mask contains values outside 0 and 255: {sorted(mask_values)}"
        )
    if 255 not in mask_values:
        raise SchemaValidationError("mask must not be empty")


def validate_jsonl(
    manifest_path: str | Path,
    *,
    check_files: bool = True,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Read and validate a JSONL manifest, including unique sample IDs."""
    manifest_path = Path(manifest_path)
    annotations: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
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
                validate_annotation(
                    annotation,
                    base_dir=manifest_path.parent,
                    check_files=check_files,
                )
            except SchemaValidationError as exc:
                raise SchemaValidationError(f"line {line_number}: {exc}") from exc
            sample_id = annotation["sample_id"]
            if sample_id in sample_ids:
                raise SchemaValidationError(
                    f"line {line_number}: duplicate sample_id {sample_id!r}"
                )
            sample_ids.add(sample_id)
            annotations.append(annotation)

    if expected_count is not None and len(annotations) != expected_count:
        raise SchemaValidationError(
            f"expected {expected_count} annotations, found {len(annotations)}"
        )
    return annotations

