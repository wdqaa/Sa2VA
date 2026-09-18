"""Independent distribution, semantic, leakage, and diversity audit for v2."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter

from .schema import SchemaValidationError
from .schema_v1 import validate_jsonl_v1
from .schema_v2 import validate_annotation_v2
from .synthetic_v2 import (
    ACTION_COUNTS_V2, ACTION_ORDER_V2, CHART_ORDER_V2, REFERRING_ORDER_V2,
    SPLIT_COUNTS_V2, SPLIT_ORDER_V2,
)


def audit_synthetic_v2(
    manifest_path: str | Path, *, v1_manifest_path: str | Path | None = None,
    expected_count: int = 1600,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    records, parse_errors = _read_jsonl(manifest)
    hard_failures = list(parse_errors)
    schema_errors: list[str] = []
    for index, record in enumerate(records, start=1):
        try:
            validate_annotation_v2(record, base_dir=manifest.parent, check_files=True)
        except (SchemaValidationError, KeyError, TypeError) as exc:
            schema_errors.append(f"line {index}: {exc}")
    hard_failures.extend(schema_errors)
    hard_failures.extend(_distribution_failures(records, expected_count))

    duplicate_fields: dict[str, list[dict[str, Any]]] = {}
    for field in ("sample_id", "image_path", "mask_path", "seed", "scene_id", "content_id"):
        groups = _duplicates(records, field)
        duplicate_fields[field] = groups
        if groups:
            hard_failures.append(f"duplicate {field}: {len(groups)} group(s)")
    family_leakage = {
        field: _cross_split_values(records, field)
        for field in ("style_family", "instruction_template_family")
    }
    for field, values in family_leakage.items():
        if values:
            hard_failures.append(f"cross-split {field} leakage: {values}")
    data_signature_leakage = _signature_leakage(records)
    if data_signature_leakage:
        hard_failures.append(f"cross-split data signature reuse: {len(data_signature_leakage)}")
    action_only_duplicates = _action_only_duplicates(records)
    if action_only_duplicates:
        hard_failures.append(f"action-only cross-split near duplicates: {len(action_only_duplicates)}")

    semantic_errors: list[str] = []
    mask_stats: list[dict[str, Any]] = []
    image_hash_groups: dict[str, list[str]] = defaultdict(list)
    mask_hash_groups: dict[str, list[str]] = defaultdict(list)
    fingerprints: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record.get("sample_id", "<missing>"))
        image_path = manifest.parent / str(record.get("image_path", ""))
        mask_path = manifest.parent / str(record.get("mask_path", ""))
        if not image_path.is_file() or not mask_path.is_file():
            continue
        image_hash_groups[_sha256(image_path)].append(sample_id)
        mask_hash_groups[_sha256(mask_path)].append(sample_id)
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB").copy()
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L"))
            foreground = mask == 255
            values = set(np.unique(mask).tolist())
            if not values.issubset({0, 255}):
                semantic_errors.append(f"{sample_id}: mask is not binary")
            if not foreground.any():
                semantic_errors.append(f"{sample_id}: empty mask")
            if foreground.all():
                semantic_errors.append(f"{sample_id}: full-frame mask")
            if image.size != (mask.shape[1], mask.shape[0]):
                semantic_errors.append(f"{sample_id}: image/mask dimension mismatch")
            metadata = record["generation_metadata"]
            entities = metadata["entity_descriptors"]
            key, reference = metadata["reference_key"], metadata["reference_value"]
            if sum(entity[key] == reference for entity in entities) != 1:
                semantic_errors.append(f"{sample_id}: ambiguous referring expression")
            if record["distractor_count"] != len(entities) - 1:
                semantic_errors.append(f"{sample_id}: distractor_count mismatch")
            for left, top, right, bottom in metadata["legend_glyph_boxes"]:
                if foreground[top : bottom + 1, left : right + 1].any():
                    semantic_errors.append(f"{sample_id}: legend glyph enters target mask")
                    break
            plot_left, plot_top, plot_right, plot_bottom = metadata["plot_box"]
            outside = foreground.copy()
            outside[plot_top : plot_bottom + 1, plot_left : plot_right + 1] = False
            if outside.any():
                semantic_errors.append(f"{sample_id}: target mask leaves plot area")
            geometry = metadata["target_geometry"]
            for x, y in geometry.get("marker_centers", []):
                if not _has_foreground_near(foreground, x, y, radius=2):
                    semantic_errors.append(f"{sample_id}: target marker center missing")
                    break
            if record["chart_type"] == "line":
                points = geometry.get("polyline", [])
                coverage = np.mean([_has_foreground_near(foreground, x, y, radius=3) for x, y in points]) if points else 0.0
                if coverage < 0.85:
                    semantic_errors.append(f"{sample_id}: incomplete visible target line ({coverage:.3f})")
            elif record["chart_type"] == "bar":
                for left, top, right, bottom in geometry.get("bar_boxes", []):
                    inset = min(2, max(0, (right - left) // 5), max(0, (bottom - top) // 5))
                    crop = foreground[top + inset : bottom - inset + 1, left + inset : right - inset + 1]
                    vertical_continuity = float(crop.any(axis=1).mean()) if crop.size else 0.0
                    if crop.size == 0 or vertical_continuity < 0.95:
                        semantic_errors.append(f"{sample_id}: incomplete target bar series")
                        break
            elif record["chart_type"] == "confidence_band" and foreground.mean() < 0.004:
                semantic_errors.append(f"{sample_id}: confidence band is not a full region")
            mask_stats.append({"sample_id": sample_id, "foreground_pixels": int(foreground.sum()), "foreground_ratio": float(foreground.mean())})
            fingerprints.append(_fingerprint(record, image, foreground))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            semantic_errors.append(f"{sample_id}: semantic audit failed: {exc}")
    hard_failures.extend(semantic_errors)
    exact_image_duplicates = _hash_duplicates(image_hash_groups)
    exact_mask_duplicates = _hash_duplicates(mask_hash_groups)
    if exact_image_duplicates:
        hard_failures.append(f"exact duplicate images: {len(exact_image_duplicates)} group(s)")
    if exact_mask_duplicates:
        hard_failures.append(f"exact duplicate masks: {len(exact_mask_duplicates)} group(s)")

    near_duplicates = _find_joint_near_duplicates(fingerprints)
    if near_duplicates:
        hard_failures.append(f"joint-feature cross-split near duplicates: {len(near_duplicates)}")

    v1_comparison: dict[str, Any] | None = None
    cross_version_exact: list[dict[str, Any]] = []
    if v1_manifest_path is not None:
        v1_manifest = Path(v1_manifest_path)
        v1_records = validate_jsonl_v1(v1_manifest, check_files=True, expected_count=320)
        cross_version_exact = _cross_version_exact_duplicates(v1_manifest, v1_records, image_hash_groups)
        if cross_version_exact:
            hard_failures.append(f"exact image duplicates with synthetic_v1: {len(cross_version_exact)}")
        v1_comparison = _diversity_comparison(v1_manifest, v1_records, records, fingerprints)

    difficulty_pairs: dict[int, set[str]] = defaultdict(set)
    for record in records:
        difficulty_pairs[int(record["distractor_count"])].add(str(record["difficulty"]))
    difficulty_independent = len(difficulty_pairs) > 1 and all(len(values) >= 2 for values in difficulty_pairs.values())
    if not difficulty_independent:
        hard_failures.append("difficulty remains one-to-one with distractor_count")

    return {
        "passed": not hard_failures,
        "manifest": str(manifest),
        "sample_count": len(records),
        "expected_count": expected_count,
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "schema_error_count": len(schema_errors),
        "schema_errors": schema_errors,
        "distributions": _distributions(records),
        "difficulty_distractor_independent": difficulty_independent,
        "difficulty_by_distractor": {str(key): sorted(values) for key, values in sorted(difficulty_pairs.items())},
        "duplicate_groups": duplicate_fields,
        "family_leakage": family_leakage,
        "data_signature_leakage": data_signature_leakage,
        "action_only_cross_split_duplicates": action_only_duplicates,
        "exact_file_duplicates": {"images": exact_image_duplicates, "masks": exact_mask_duplicates},
        "cross_version_exact_image_duplicates": cross_version_exact,
        "near_duplicate_check": {
            "method": "joint dHash Hamming + 16-bin RGB histogram L1 + edge thumbnail + target-mask thumbnail + aspect ratio",
            "thresholds": {"dhash_hamming": 4, "histogram_l1": 0.08, "edge_mae": 0.025, "mask_mae": 0.01, "foreground_ratio_delta": 0.002, "aspect_delta": 0.02},
            "scope": "cross-split pairs only",
            "candidate_count": len(near_duplicates),
            "candidates": near_duplicates[:100],
            "is_hard_failure": True,
        },
        "mask_audit": {
            "sample_count": len(mask_stats),
            "semantic_error_count": len(semantic_errors),
            "semantic_errors": semantic_errors,
            "foreground_pixels": _numeric_summary([item["foreground_pixels"] for item in mask_stats]),
            "foreground_ratio": _numeric_summary([item["foreground_ratio"] for item in mask_stats]),
        },
        "v1_v2_diversity_comparison": v1_comparison,
    }


def write_audit_json_v2(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _distribution_failures(records: list[dict[str, Any]], expected: int) -> list[str]:
    failures = []
    if len(records) != expected:
        failures.append(f"expected {expected} records, got {len(records)}")
    split_counts = Counter(row.get("split") for row in records)
    if split_counts != Counter({"train": 960, "val": 320, "test": 320}):
        failures.append(f"incorrect split distribution: {dict(split_counts)}")
    for chart in CHART_ORDER_V2:
        for referring in REFERRING_ORDER_V2:
            for split in SPLIT_ORDER_V2:
                subset = [row for row in records if row.get("chart_type") == chart and row.get("referring_type") == referring and row.get("split") == split]
                if len(subset) != SPLIT_COUNTS_V2[split]:
                    failures.append(f"incorrect count for {chart}/{referring}/{split}: {len(subset)}")
                    continue
                actions = Counter(row["edit_action"] for row in subset)
                expected_actions = Counter({action: ACTION_COUNTS_V2[split] for action in ACTION_ORDER_V2})
                if actions != expected_actions:
                    failures.append(f"incorrect action balance for {chart}/{referring}/{split}: {dict(actions)}")
    return failures


def _distributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("split", "chart_type", "referring_type", "edit_action", "difficulty", "distractor_count")
    output = {field: dict(sorted(Counter(str(row[field]) for row in records).items())) for field in fields}
    output["chart_referring"] = dict(sorted(Counter(f"{row['chart_type']}/{row['referring_type']}" for row in records).items()))
    output["split_action"] = dict(sorted(Counter(f"{row['split']}/{row['edit_action']}" for row in records).items()))
    for field in ("aspect_family", "legend_position", "theme", "grid_mode", "layout_density", "numeric_format"):
        output[field] = dict(sorted(Counter(str(row["diversity_metadata"][field]) for row in records).items()))
    return output


def _signature_leakage(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["diversity_metadata"]["data_signature"]].append(row)
    return [
        {"data_signature": signature, "sample_ids": [row["sample_id"] for row in rows], "splits": sorted({row["split"] for row in rows})}
        for signature, rows in groups.items() if len({row["split"] for row in rows}) > 1
    ]


def _action_only_duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (row["chart_type"], row["referring_expression"], row["diversity_metadata"]["data_signature"])
        groups[key].append(row)
    return [
        {"sample_ids": [row["sample_id"] for row in rows], "splits": sorted({row["split"] for row in rows}), "actions": sorted({row["edit_action"] for row in rows})}
        for rows in groups.values() if len({row["split"] for row in rows}) > 1 and len({row["edit_action"] for row in rows}) > 1
    ]


def _fingerprint(record: dict[str, Any], image: Image.Image, foreground: np.ndarray) -> dict[str, Any]:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    gray_array = np.asarray(gray, dtype=np.int16)
    bits = gray_array[:, 1:] > gray_array[:, :-1]
    dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
    rgb = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[..., channel], bins=16, range=(0, 256), density=False)
        histograms.extend((hist / hist.sum()).tolist())
    edge_source = np.asarray(image.convert("L").resize((20, 20), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    horizontal = np.abs(np.diff(edge_source, axis=1, prepend=edge_source[:, :1]))
    vertical = np.abs(np.diff(edge_source, axis=0, prepend=edge_source[:1, :]))
    edge = horizontal + vertical
    return {
        "sample_id": record["sample_id"], "split": record["split"],
        "aspect": image.width / image.height, "dhash": dhash,
        "histogram": np.asarray(histograms, dtype=np.float32),
        "edge": edge.flatten(), "foreground_ratio": float(foreground.mean()),
        "mask_thumbnail": np.asarray(
            Image.fromarray(foreground.astype(np.uint8) * 255).resize(
                (32, 32), Image.Resampling.BOX
            ),
            dtype=np.float32,
        ).flatten() / 255.0,
    }


def _find_joint_near_duplicates(fingerprints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_aspect: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in fingerprints:
        by_aspect[round(item["aspect"] * 20)].append(item)
    candidates = []
    for bucket in by_aspect.values():
        for left_index, left in enumerate(bucket):
            for right in bucket[left_index + 1 :]:
                if left["split"] == right["split"] or abs(left["aspect"] - right["aspect"]) > 0.02:
                    continue
                hamming = (left["dhash"] ^ right["dhash"]).bit_count()
                if hamming > 4:
                    continue
                histogram_l1 = float(np.abs(left["histogram"] - right["histogram"]).sum() / 3.0)
                if histogram_l1 > 0.08:
                    continue
                edge_mae = float(np.mean(np.abs(left["edge"] - right["edge"])))
                if edge_mae > 0.025:
                    continue
                mask_mae = float(np.mean(np.abs(left["mask_thumbnail"] - right["mask_thumbnail"])))
                if mask_mae > 0.01 or abs(left["foreground_ratio"] - right["foreground_ratio"]) > 0.002:
                    continue
                candidates.append({"sample_ids": [left["sample_id"], right["sample_id"]], "splits": [left["split"], right["split"]], "dhash_hamming": hamming, "histogram_l1": histogram_l1, "edge_mae": edge_mae, "mask_mae": mask_mae})
    return candidates


def _diversity_comparison(v1_manifest: Path, v1_records: list[dict[str, Any]], v2_records: list[dict[str, Any]], v2_fingerprints: list[dict[str, Any]]) -> dict[str, Any]:
    def color_distance(row: dict[str, Any]) -> float:
        entities = row["generation_metadata"]["entity_descriptors"]
        target = entities[row["generation_metadata"]["target_index"]]["color"]
        target_rgb = np.asarray(_hex_rgb(target))
        return min(float(np.linalg.norm(target_rgb - np.asarray(_hex_rgb(entity["color"]))) / math.sqrt(3 * 255**2)) for index, entity in enumerate(entities) if index != row["generation_metadata"]["target_index"])
    v1_fingerprints = []
    v1_foreground = []
    for row in v1_records:
        with Image.open(v1_manifest.parent / row["image_path"]) as source:
            image = source.convert("RGB").copy()
        with Image.open(v1_manifest.parent / row["mask_path"]) as source:
            mask = np.asarray(source.convert("L")) > 0
        v1_fingerprints.append(_fingerprint(row, image, mask))
        v1_foreground.append(float(mask.mean()))
    def summary(records: list[dict[str, Any]], *, v2: bool) -> dict[str, Any]:
        diversity = [row.get("diversity_metadata", {}) for row in records]
        line_records = [row for row in records if row["chart_type"] == "line"]
        marker_records = [row for row in records if row["chart_type"] in {"line", "scatter"}]
        if v2:
            line_widths = [float(row["diversity_metadata"]["line_width"]) for row in line_records]
            marker_sizes = [float(row["diversity_metadata"]["marker_size"]) for row in marker_records]
        else:
            line_widths = [float(row["generation_metadata"]["target_geometry"]["line_width"]) for row in line_records]
            marker_sizes = [
                float(row["generation_metadata"]["target_geometry"].get(
                    "marker_radius", {"easy": 5, "medium": 4, "hard": 3}[row["difficulty"]]
                ))
                for row in marker_records
            ]
        return {
            "sample_count": len(records),
            "aspect_ratio": _numeric_summary([row["image_width"] / row["image_height"] for row in records]),
            "unique_canvas_sizes": len({(row["image_width"], row["image_height"]) for row in records}),
            "foreground_ratio": _numeric_summary([float(item.get("foreground_ratio", 0.0)) for item in diversity] if v2 else v1_foreground),
            "series_count": _numeric_summary([row["distractor_count"] + 1 for row in records]),
            "difficulty": dict(sorted(Counter(row["difficulty"] for row in records).items())),
            "distractor_count": dict(sorted(Counter(str(row["distractor_count"]) for row in records).items())),
            "color_distance": _numeric_summary([float(item["color_distance"]) for item in diversity] if v2 else [color_distance(row) for row in records]),
            "line_width": _numeric_summary(line_widths),
            "marker_size": _numeric_summary(marker_sizes),
            "legend_position": dict(sorted(Counter(item["legend_position"] for item in diversity).items())) if v2 else {"right": len(records)},
            "theme": dict(sorted(Counter(item["theme"] for item in diversity).items())) if v2 else {"light": len(records)},
            "crossing_count": _numeric_summary([float(item["crossing_count"]) for item in diversity]) if v2 else {"status": "not recorded in v1"},
            "occlusion_ratio": _numeric_summary([float(item["occlusion_ratio"]) for item in diversity]) if v2 else {"status": "not recorded in v1"},
        }
    return {
        "v1": summary(v1_records, v2=False),
        "v2": summary(v2_records, v2=True),
        "image_fingerprint_distance": {
            "method": "deterministic pair sample over dHash Hamming normalized to [0,1]",
            "v1_within": _fingerprint_distance_summary(v1_fingerprints, seed=11),
            "v2_within": _fingerprint_distance_summary(v2_fingerprints, seed=17),
            "v1_to_v2": _cross_fingerprint_distance_summary(v1_fingerprints, v2_fingerprints, seed=23),
        },
    }


def _fingerprint_distance_summary(items: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(min(5000, len(items) * 5)):
        left, right = rng.choice(len(items), size=2, replace=False)
        distances.append((items[int(left)]["dhash"] ^ items[int(right)]["dhash"]).bit_count() / 64.0)
    return _numeric_summary(distances)


def _cross_fingerprint_distance_summary(left: list[dict[str, Any]], right: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(5000):
        a, b = left[int(rng.integers(0, len(left)))], right[int(rng.integers(0, len(right)))]
        distances.append((a["dhash"] ^ b["dhash"]).bit_count() / 64.0)
    return _numeric_summary(distances)


def _cross_version_exact_duplicates(v1_manifest: Path, v1_records: list[dict[str, Any]], v2_hashes: dict[str, list[str]]) -> list[dict[str, Any]]:
    duplicates = []
    for row in v1_records:
        digest = _sha256(v1_manifest.parent / row["image_path"])
        if digest in v2_hashes:
            duplicates.append({"sha256": digest, "v1_sample_id": row["sample_id"], "v2_sample_ids": v2_hashes[digest]})
    return duplicates


def _has_foreground_near(mask: np.ndarray, x: int, y: int, radius: int) -> bool:
    top, bottom = max(0, y - radius), min(mask.shape[0], y + radius + 1)
    left, right = max(0, x - radius), min(mask.shape[1], x + radius + 1)
    return bool(mask[top:bottom, left:right].any())


def _duplicates(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[str]] = defaultdict(list)
    for row in records:
        groups[row.get(field)].append(str(row.get("sample_id", "<missing>")))
    return [{"value": value, "sample_ids": ids} for value, ids in groups.items() if len(ids) > 1]


def _cross_split_values(records: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in records:
        values[str(row.get(field))].add(str(row.get("split")))
    return {value: sorted(splits) for value, splits in values.items() if len(splits) > 1}


def _hash_duplicates(groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    return [{"sha256": digest, "sample_ids": ids} for digest, ids in groups.items() if len(ids) > 1]


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records, errors = [], []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("record is not an object")
                records.append(value)
            except (json.JSONDecodeError, TypeError) as exc:
                errors.append(f"line {index}: {exc}")
    return records, errors


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size), "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "mean": float(array.mean()), "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
