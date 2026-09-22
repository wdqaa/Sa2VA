"""Independent hard-constraint and leakage audit for synthetic_v1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .schema import SchemaValidationError
from .schema_v1 import validate_annotation_v1
from .synthetic_v1 import (
    ACTION_ORDER_V1,
    CHART_ORDER_V1,
    DIFFICULTY_ORDER_V1,
    REFERRING_ORDER_V1,
    SPLIT_COUNTS_V1,
    SPLIT_ORDER_V1,
)


DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.01


def audit_synthetic_v1(
    manifest_path: str | Path,
    *,
    expected_count: int = 320,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    """Audit raw records and files without trusting generator assertions."""
    if not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in [0, 1]")
    manifest = Path(manifest_path)
    records, parse_errors = _read_raw_jsonl(manifest)
    hard_failures = list(parse_errors)
    schema_errors: list[str] = []
    for index, record in enumerate(records, start=1):
        try:
            validate_annotation_v1(record, base_dir=manifest.parent, check_files=True)
        except (SchemaValidationError, KeyError, TypeError) as exc:
            schema_errors.append(f"line {index}: {exc}")
    hard_failures.extend(schema_errors)

    distributions = _distributions(records)
    hard_failures.extend(_check_exact_distribution(records, expected_count))
    duplicates = {
        field: _duplicate_groups(records, field)
        for field in (
            "sample_id",
            "image_path",
            "mask_path",
            "seed",
            "scene_id",
            "content_id",
        )
    }
    for field, groups in duplicates.items():
        if groups:
            hard_failures.append(f"duplicate {field}: {len(groups)} group(s)")
    content_hash_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        try:
            content_hash_groups[record["generation_metadata"]["content_sha256"]].append(
                record["sample_id"]
            )
        except (KeyError, TypeError):
            continue
    content_hash_duplicates = [
        {"sha256": digest, "sample_ids": sample_ids}
        for digest, sample_ids in sorted(content_hash_groups.items())
        if len(sample_ids) > 1
    ]
    if content_hash_duplicates:
        hard_failures.append(
            "duplicate underlying numeric content hash: "
            f"{len(content_hash_duplicates)} group(s)"
        )

    family_leakage = {
        "style_family": _cross_split_values(records, "style_family"),
        "instruction_template_family": _cross_split_values(
            records, "instruction_template_family"
        ),
    }
    for field, overlaps in family_leakage.items():
        if overlaps:
            hard_failures.append(f"cross-split {field} leakage: {overlaps}")

    semantic_errors: list[str] = []
    mask_stats: list[dict[str, Any]] = []
    image_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mask_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    valid_images: list[tuple[dict[str, Any], Path]] = []
    mask_issue_counts = {
        "empty_count": 0,
        "full_count": 0,
        "dimension_error_count": 0,
        "non_binary_count": 0,
        "format_or_mode_error_count": 0,
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        sample_id = str(record.get("sample_id", "<missing>"))
        try:
            metadata = record["generation_metadata"]
            entities = metadata["entity_descriptors"]
            key = metadata["reference_key"]
            value = metadata["reference_value"]
            match_count = sum(entity.get(key) == value for entity in entities)
            if match_count != 1 or metadata.get("reference_match_count") != 1:
                semantic_errors.append(
                    f"{sample_id}: referring expression matches {match_count} entities"
                )
            if record["distractor_count"] != len(entities) - 1:
                semantic_errors.append(f"{sample_id}: distractor_count is not actual")
        except (KeyError, TypeError) as exc:
            semantic_errors.append(f"{sample_id}: invalid generation metadata: {exc}")
            continue
        image_path = manifest.parent / str(record.get("image_path", ""))
        mask_path = manifest.parent / str(record.get("mask_path", ""))
        if not image_path.is_file() or not mask_path.is_file():
            continue
        image_hashes[_sha256(image_path)].append(record)
        mask_hashes[_sha256(mask_path)].append(record)
        valid_images.append((record, image_path))
        try:
            with Image.open(mask_path) as source:
                mask_format = source.format
                mask_mode = source.mode
                mask_size = source.size
                mask = np.asarray(source)
            expected_size = (
                record.get("image_width"),
                record.get("image_height"),
            )
            if mask_size != expected_size:
                mask_issue_counts["dimension_error_count"] += 1
            if mask_format != "PNG" or mask_mode != "L":
                mask_issue_counts["format_or_mode_error_count"] += 1
            values = set(np.unique(mask).tolist())
            if not values.issubset({0, 255}):
                mask_issue_counts["non_binary_count"] += 1
            if mask.ndim != 2:
                semantic_errors.append(
                    f"{sample_id}: mask must be single-channel for semantic audit"
                )
                continue
            foreground = mask == 255
            count = int(foreground.sum())
            ratio = count / int(mask.size)
            if count == 0:
                mask_issue_counts["empty_count"] += 1
            if count == mask.size:
                mask_issue_counts["full_count"] += 1
            mask_stats.append(
                {
                    "sample_id": sample_id,
                    "foreground_pixels": count,
                    "foreground_ratio": ratio,
                    "total_pixels": int(mask.size),
                }
            )
            legend = metadata["legend_box"]
            left, top, right, bottom = map(int, legend)
            if np.any(foreground[top : bottom + 1, left : right + 1]):
                semantic_errors.append(f"{sample_id}: legend proxy enters mask")
            plot_left, plot_top, plot_right, plot_bottom = map(
                int, metadata["plot_box"]
            )
            outside = foreground.copy()
            outside[plot_top : plot_bottom + 1, plot_left : plot_right + 1] = False
            if outside.any():
                semantic_errors.append(f"{sample_id}: mask leaves plot region")
            if record.get("chart_type") == "confidence_band" and ratio < 0.01:
                semantic_errors.append(
                    f"{sample_id}: confidence band foreground ratio {ratio:.6f} "
                    "is below the non-line threshold 0.01"
                )
            geometry = metadata["target_geometry"]
            if record.get("chart_type") in {"line", "scatter"}:
                for x, y in geometry.get("marker_centers", []):
                    if not foreground[int(y), int(x)]:
                        semantic_errors.append(
                            f"{sample_id}: target marker center ({x}, {y}) is absent"
                        )
                        break
        except (OSError, ValueError, KeyError, TypeError) as exc:
            semantic_errors.append(f"{sample_id}: semantic mask check failed: {exc}")
    hard_failures.extend(semantic_errors)

    exact_image_duplicates = _hash_groups(image_hashes)
    exact_mask_duplicates = _hash_groups(mask_hashes)
    if exact_image_duplicates:
        hard_failures.append(
            f"exact duplicate image content: {len(exact_image_duplicates)} group(s)"
        )
    if exact_mask_duplicates:
        hard_failures.append(
            f"exact duplicate mask content: {len(exact_mask_duplicates)} group(s)"
        )

    near_duplicates = _find_cross_split_near_duplicates(
        valid_images, near_duplicate_threshold
    )
    foreground_counts = [item["foreground_pixels"] for item in mask_stats]
    foreground_ratios = [item["foreground_ratio"] for item in mask_stats]
    return {
        "passed": not hard_failures,
        "manifest": str(manifest),
        "expected_count": expected_count,
        "sample_count": len(records),
        "schema_error_count": len(schema_errors),
        "schema_errors": schema_errors,
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "distributions": distributions,
        "duplicate_groups": duplicates,
        "content_hash_duplicate_groups": content_hash_duplicates,
        "family_leakage": family_leakage,
        "exact_file_duplicates": {
            "images": exact_image_duplicates,
            "masks": exact_mask_duplicates,
        },
        "near_duplicate_check": {
            "method": "RGB 24x16 plot-area mean absolute difference / 255",
            "threshold": near_duplicate_threshold,
            "scope": "cross-split pairs only",
            "candidate_count": len(near_duplicates),
            "candidates": near_duplicates[:200],
            "truncated": len(near_duplicates) > 200,
            "is_hard_failure": False,
        },
        "mask_audit": {
            "sample_count": len(mask_stats),
            **mask_issue_counts,
            "foreground_pixels": _numeric_summary(foreground_counts),
            "foreground_ratio": _numeric_summary(foreground_ratios),
            "semantic_error_count": len(semantic_errors),
            "semantic_errors": semantic_errors,
        },
    }


def write_audit_json(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _read_raw_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        return [], [f"cannot open manifest: {exc}"]
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append(f"line {line_number} is blank")
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number} is invalid JSON: {exc.msg}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number} is not a JSON object")
                continue
            records.append(value)
    return records, errors


def _distributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count(keys: tuple[str, ...]) -> dict[str, int]:
        values: Counter[str] = Counter()
        for record in records:
            if all(key in record for key in keys):
                values[" / ".join(str(record[key]) for key in keys)] += 1
        return dict(sorted(values.items()))

    return {
        "split": count(("split",)),
        "split_x_chart_type": count(("split", "chart_type")),
        "split_x_referring_type": count(("split", "referring_type")),
        "split_x_edit_action": count(("split", "edit_action")),
        "split_x_difficulty": count(("split", "difficulty")),
        "chart_x_referring_x_split": count(
            ("chart_type", "referring_type", "split")
        ),
        "chart_x_referring_x_split_x_action": count(
            ("chart_type", "referring_type", "split", "edit_action")
        ),
        "style_family": count(("style_family",)),
        "instruction_template_family": count(("instruction_template_family",)),
        "distractor_count": count(("distractor_count",)),
        "difficulty": count(("difficulty",)),
    }


def _check_exact_distribution(
    records: list[dict[str, Any]], expected_count: int
) -> list[str]:
    errors: list[str] = []
    if len(records) != expected_count:
        errors.append(f"expected {expected_count} records, found {len(records)}")
    split_counts = Counter(record.get("split") for record in records)
    global_split_counts = {
        split: per_combination * len(CHART_ORDER_V1) * len(REFERRING_ORDER_V1)
        for split, per_combination in SPLIT_COUNTS_V1.items()
    }
    if split_counts != Counter(global_split_counts):
        errors.append(
            f"split counts mismatch: expected {global_split_counts}, "
            f"got {dict(split_counts)}"
        )
    for chart in CHART_ORDER_V1:
        for referring in REFERRING_ORDER_V1:
            for split in SPLIT_ORDER_V1:
                subset = [
                    record
                    for record in records
                    if record.get("chart_type") == chart
                    and record.get("referring_type") == referring
                    and record.get("split") == split
                ]
                expected = SPLIT_COUNTS_V1[split]
                if len(subset) != expected:
                    errors.append(
                        f"{chart}/{referring}/{split}: expected {expected}, got {len(subset)}"
                    )
                actions = Counter(record.get("edit_action") for record in subset)
                expected_actions = {
                    action: 3 if split == "train" else 1
                    for action in ACTION_ORDER_V1
                }
                if actions != Counter(expected_actions):
                    errors.append(
                        f"{chart}/{referring}/{split}: action counts {dict(actions)}"
                    )
                difficulties = {record.get("difficulty") for record in subset}
                if difficulties != set(DIFFICULTY_ORDER_V1):
                    errors.append(
                        f"{chart}/{referring}/{split}: missing difficulty coverage"
                    )
    global_difficulties = Counter(record.get("difficulty") for record in records)
    if global_difficulties and max(global_difficulties.values()) - min(
        global_difficulties.values()
    ) > 1:
        errors.append(f"global difficulty imbalance: {dict(global_difficulties)}")
    return errors


def _duplicate_groups(
    records: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    groups: dict[Any, list[str]] = defaultdict(list)
    for record in records:
        if field in record:
            groups[record[field]].append(str(record.get("sample_id", "<missing>")))
    return [
        {"value": value, "sample_ids": sample_ids}
        for value, sample_ids in sorted(groups.items(), key=lambda item: str(item[0]))
        if len(sample_ids) > 1
    ]


def _cross_split_values(
    records: list[dict[str, Any]], field: str
) -> dict[str, list[str]]:
    splits_by_value: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if field in record and "split" in record:
            splits_by_value[str(record[field])].add(str(record["split"]))
    return {
        value: sorted(splits)
        for value, splits in sorted(splits_by_value.items())
        if len(splits) > 1
    }


def _hash_groups(
    groups: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    result = []
    for digest, records in sorted(groups.items()):
        if len(records) > 1:
            result.append(
                {
                    "sha256": digest,
                    "sample_ids": [record["sample_id"] for record in records],
                    "splits": sorted({record["split"] for record in records}),
                }
            )
    return result


def _find_cross_split_near_duplicates(
    items: list[tuple[dict[str, Any], Path]], threshold: float
) -> list[dict[str, Any]]:
    fingerprints: list[tuple[dict[str, Any], np.ndarray]] = []
    for record, path in items:
        with Image.open(path) as source:
            image = source.convert("RGB")
            plot_box = tuple(record["generation_metadata"]["plot_box"])
            fingerprint = np.asarray(
                image.crop(plot_box).resize((24, 16), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        fingerprints.append((record, fingerprint))
    candidates: list[dict[str, Any]] = []
    for left_index, (left_record, left_fp) in enumerate(fingerprints):
        for right_record, right_fp in fingerprints[left_index + 1 :]:
            if left_record["split"] == right_record["split"]:
                continue
            distance = float(np.mean(np.abs(left_fp - right_fp)) / 255.0)
            if distance <= threshold:
                candidates.append(
                    {
                        "left_sample_id": left_record["sample_id"],
                        "left_split": left_record["split"],
                        "right_sample_id": right_record["sample_id"],
                        "right_split": right_record["split"],
                        "distance": round(distance, 8),
                    }
                )
    candidates.sort(key=lambda item: (item["distance"], item["left_sample_id"]))
    return candidates


def _numeric_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
