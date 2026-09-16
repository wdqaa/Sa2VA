"""Deterministic train-only subset selection for Phase 4 smoke/overfit work."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from chartground_edit.datasets.schema_v1 import validate_jsonl_v1
from chartground_edit.datasets.synthetic_v1 import (
    ACTION_ORDER_V1,
    CHART_ORDER_V1,
    REFERRING_ORDER_V1,
)

from .data_adapter import EXPECTED_MANIFEST_SHA256, file_sha256


DIFFICULTY_ORDER = ("easy", "medium", "hard")
SELECTION_VERSION = "phase4-overfit-subsets-v1"


def _load_frozen_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    digest = file_sha256(manifest_path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "manifest SHA-256 mismatch: "
            f"expected {EXPECTED_MANIFEST_SHA256}, got {digest}"
        )
    return validate_jsonl_v1(manifest_path, check_files=True, expected_count=320)


def _base_payload(kind: str, sample_ids: list[str]) -> dict[str, Any]:
    return {
        "selection_version": SELECTION_VERSION,
        "name": kind,
        "source_manifest": "projects/chartground_edit/data/synthetic_v1/annotations.jsonl",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "split": "train",
        "prompt_variant": "target_only_zh",
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
    }


def build_smoke1_manifest(
    records: list[dict[str, Any]], manifest_root: Path
) -> dict[str, Any]:
    candidates = sorted(
        (
            record
            for record in records
            if record["split"] == "train"
            and record["chart_type"] == "bar"
            and record["referring_type"] == "category"
        ),
        key=lambda record: record["sample_id"],
    )
    if not candidates:
        raise ValueError("no train bar/category candidate for smoke1")
    ratios: list[tuple[float, dict[str, Any]]] = []
    for record in candidates:
        with Image.open(manifest_root / record["mask_path"]) as source:
            mask = np.asarray(source.convert("L"))
        ratios.append((float(np.count_nonzero(mask) / mask.size), record))
    median_ratio = float(np.median([ratio for ratio, _ in ratios]))
    ratio, selected = min(
        ratios,
        key=lambda item: (abs(item[0] - median_ratio), item[1]["sample_id"]),
    )
    payload = _base_payload("smoke1", [selected["sample_id"]])
    payload.update(
        {
            "selection_rule": (
                "Among train bar/category records, choose the mask foreground ratio "
                "closest to that candidate set's median; break ties by sample_id."
            ),
            "audit": {
                "chart_type": selected["chart_type"],
                "referring_type": selected["referring_type"],
                "mask_foreground_ratio": ratio,
                "candidate_median_foreground_ratio": median_ratio,
            },
        }
    )
    return payload


def _balance_score(
    pair: tuple[dict[str, Any], dict[str, Any]],
    action_counts: Counter[str],
    difficulty_counts: Counter[str],
) -> tuple[Any, ...]:
    left, right = pair
    next_actions = action_counts.copy()
    next_actions.update((left["edit_action"], right["edit_action"]))
    next_difficulties = difficulty_counts.copy()
    next_difficulties.update((left["difficulty"], right["difficulty"]))
    action_values = [next_actions[action] for action in ACTION_ORDER_V1]
    difficulty_values = [next_difficulties[item] for item in DIFFICULTY_ORDER]
    action_mean = sum(action_values) / len(action_values)
    difficulty_mean = sum(difficulty_values) / len(difficulty_values)
    return (
        max(action_values) - min(action_values),
        sum((value - action_mean) ** 2 for value in action_values),
        int(left["edit_action"] == right["edit_action"]),
        max(difficulty_values) - min(difficulty_values),
        sum((value - difficulty_mean) ** 2 for value in difficulty_values),
        int(left["difficulty"] == right["difficulty"]),
        -abs(left["distractor_count"] - right["distractor_count"]),
        left["sample_id"],
        right["sample_id"],
    )


def build_overfit32_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    train = [record for record in records if record["split"] == "train"]
    selected: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    for chart_type in CHART_ORDER_V1:
        for referring_type in REFERRING_ORDER_V1:
            group = sorted(
                (
                    record
                    for record in train
                    if record["chart_type"] == chart_type
                    and record["referring_type"] == referring_type
                ),
                key=lambda record: record["sample_id"],
            )
            if len(group) != 12:
                raise ValueError(
                    f"expected 12 train records for {chart_type}/{referring_type}, "
                    f"found {len(group)}"
                )
            pair = min(
                itertools.combinations(group, 2),
                key=lambda item: _balance_score(
                    item, action_counts, difficulty_counts
                ),
            )
            selected.extend(pair)
            action_counts.update(record["edit_action"] for record in pair)
            difficulty_counts.update(record["difficulty"] for record in pair)

    sample_ids = [record["sample_id"] for record in selected]
    payload = _base_payload("overfit32", sample_ids)
    payload.update(
        {
            "selection_rule": (
                "Visit chart/referring groups in generator canonical order. For each "
                "group choose two train records by a frozen lexicographic objective: "
                "minimize global action imbalance and variance, duplicate action, "
                "difficulty imbalance and variance, duplicate difficulty; prefer wider "
                "distractor separation; finally break ties by sample_id."
            ),
            "distribution": {
                "chart_referring": {
                    f"{chart}/{referring}": sum(
                        record["chart_type"] == chart
                        and record["referring_type"] == referring
                        for record in selected
                    )
                    for chart in CHART_ORDER_V1
                    for referring in REFERRING_ORDER_V1
                },
                "edit_action": dict(sorted(action_counts.items())),
                "difficulty": dict(sorted(difficulty_counts.items())),
                "distractor_count": dict(
                    sorted(
                        Counter(
                            str(record["distractor_count"]) for record in selected
                        ).items()
                    )
                ),
            },
        }
    )
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_phase4_subsets(
    manifest_path: str | Path,
    smoke_path: str | Path,
    overfit_path: str | Path,
) -> dict[str, Any]:
    """Independently verify frozen counts, balance, and held-out separation."""
    manifest = Path(manifest_path)
    records = _load_frozen_manifest(manifest)
    by_id = {record["sample_id"]: record for record in records}
    smoke = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
    overfit = json.loads(Path(overfit_path).read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, payload, expected in (
        ("smoke1", smoke, 1),
        ("overfit32", overfit, 32),
    ):
        ids = payload.get("sample_ids", [])
        if payload.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
            failures.append(f"{name}: manifest hash mismatch")
        if payload.get("split") != "train":
            failures.append(f"{name}: split is not train")
        if len(ids) != expected or len(ids) != len(set(ids)):
            failures.append(f"{name}: expected {expected} unique IDs, found {len(ids)}")
        if any(sample_id not in by_id for sample_id in ids):
            failures.append(f"{name}: unknown sample ID")
        if any(
            by_id[sample_id]["split"] != "train"
            for sample_id in ids
            if sample_id in by_id
        ):
            failures.append(f"{name}: non-train sample")
        if any(key in payload for key in ("scene_id", "content_id")):
            failures.append(f"{name}: identity leakage field in selection payload")

    selected = [
        by_id[sample_id]
        for sample_id in overfit.get("sample_ids", [])
        if sample_id in by_id
    ]
    groups = Counter(
        (record["chart_type"], record["referring_type"]) for record in selected
    )
    expected_groups = {
        (chart, referring): 2
        for chart in CHART_ORDER_V1
        for referring in REFERRING_ORDER_V1
    }
    if groups != expected_groups:
        failures.append("overfit32: chart/referring distribution is not 16 groups x 2")
    held_out = [record for record in records if record["split"] in {"val", "test"}]
    for field in ("sample_id", "scene_id", "content_id"):
        if {record[field] for record in selected} & {record[field] for record in held_out}:
            failures.append(f"overfit32: {field} overlaps val/test")

    return {
        "passed": not failures,
        "failures": failures,
        "manifest_sha256": file_sha256(manifest),
        "smoke1_count": len(smoke.get("sample_ids", [])),
        "overfit32_count": len(selected),
        "chart_referring_counts": {
            f"{chart}/{referring}": groups[(chart, referring)]
            for chart in CHART_ORDER_V1
            for referring in REFERRING_ORDER_V1
        },
        "edit_action_counts": dict(
            sorted(Counter(record["edit_action"] for record in selected).items())
        ),
        "difficulty_counts": dict(
            sorted(Counter(record["difficulty"] for record in selected).items())
        ),
        "distractor_counts": dict(
            sorted(
                Counter(str(record["distractor_count"]) for record in selected).items()
            )
        ),
    }


def prepare_phase4_subsets(
    manifest_path: str | Path,
    smoke_output: str | Path,
    overfit_output: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = Path(manifest_path)
    records = _load_frozen_manifest(manifest)
    smoke = build_smoke1_manifest(records, manifest.parent)
    overfit = build_overfit32_manifest(records)
    _write_json(Path(smoke_output), smoke)
    _write_json(Path(overfit_output), overfit)
    return smoke, overfit
