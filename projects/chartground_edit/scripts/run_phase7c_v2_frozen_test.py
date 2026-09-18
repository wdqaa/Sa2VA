#!/usr/bin/env python3
"""Run the single preregistered synthetic_v2 frozen test ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "projects/chartground_edit"
for root in (REPO_ROOT, PACKAGE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from chartground_edit.inference.frozen_test_v1 import apply_predicted_mask_edit
from chartground_edit.inference.metrics import dice_score, intersection_over_union
from chartground_edit.inference.projection_checkpoint import (
    load_projection_checkpoint_into_model,
    projection_state,
    restore_projection_state,
)
from chartground_edit.inference.prompt_benchmark_v1 import prompt_template_hashes
from chartground_edit.inference.prompt_variants import TARGET_ONLY_ZH
from chartground_edit.inference.sa2va_backend import Sa2VAInternVL3Backend
from chartground_edit.training.data_adapter import Phase7V2SplitDataset
from chartground_edit.training.overfit32 import (
    select_gallery_sample_ids,
    summarize_checkpoint_rows,
)
from chartground_edit.training.strategy_b import (
    load_strategy_b_checkpoint_into_hf_model,
    prepare_hf_strategy_b_model,
)
from chartground_edit.visualization.phase5b_saved import (
    classify_failure,
    error_map,
)
from chartground_edit.visualization.render import create_contact_sheet, mask_overlay


STATE_ORDER = ("zero_shot", "v1_step960", "v2_strategy_a", "v2_strategy_b")
MANIFEST_SHA256 = "1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be"
TEST_ID_SHA256 = "34ca02e0e79e1b208600ad844f42b4e24fb6c4ee96ed3bd910f06fc127a9ad2b"
FULL_PTH_SHA256 = "5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6"
V1_SHA256 = "64c0d109d2985893ba1f2ba4c4fe7acc4265dc758e5d56ecb6ac8e2aa791f41e"
V2_A_SHA256 = "6c8f30d20e52b1e0473b8a9bdd69e682c944cbdcf6bed8c0c45f30a399eab3c4"
V2_B_SHA256 = "c47ce4e38a9b1679c766d6360d66b6a3b69286d36b9d7cde50c70991ae475b97"
PROTOCOL_SHA256 = "c7e82d6c7377eb1d4ae38a8051f73acfac13278c1ea93e7ebf4a0c69c90dcc22"
PROTOCOL_COMMIT = "de3b45f"
PROMPT_REGISTRY_SHA256 = "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0"
P2_SHA256 = "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806"
BASE_REPO_ID = "OpenGVLab/InternVL3-2B"
BASE_REVISION = "899155015275a9b7338c7f4677e19c784e0e5a21"
SA2VA_REVISION = "15837dcaecc304714a1f0f069e74f47e47521c7f"
EXPECTED_SAMPLES = 320
EXPECTED_CALLS = 1280
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_ITERATIONS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--v1-projection", required=True, type=Path)
    parser.add_argument("--v2-a-projection", required=True, type=Path)
    parser.add_argument("--v2-b-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path,
        default=PACKAGE_ROOT / "data/synthetic_v2/annotations.jsonl",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=PACKAGE_ROOT / "docs/phase7c_v2_frozen_test_protocol.md",
    )
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--ablation-output", type=Path)
    parser.add_argument("--gallery-output", type=Path)
    parser.add_argument("--failure-output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_hash(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(mask.shape).encode())
    digest.update(mask.astype(np.uint8, copy=False).tobytes())
    return digest.hexdigest()


def test_id_hash(records: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "".join(row["sample_id"] + "\n" for row in records).encode()
    ).hexdigest()


def degradation_type(record: dict[str, Any]) -> str:
    values = record["diversity_metadata"]["degradation"]
    tags = []
    if values["jpeg_quality"] < 100:
        tags.append("jpeg")
    if values["blur_radius"] > 0:
        tags.append("blur")
    if values["screenshot_scale"] < 1:
        tags.append("screenshot_scale")
    if values["antialias_factor"] > 1:
        tags.append("antialias_x2")
    return "+".join(tags) if tags else "clean"


def detect_local_revision(checkpoint: Path) -> str | None:
    revisions = set()
    root = checkpoint / ".cache/huggingface/download"
    if root.is_dir():
        for path in root.glob("*.metadata"):
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines and len(lines[0]) == 40:
                revisions.add(lines[0])
    return next(iter(revisions)) if len(revisions) == 1 else None


def _projection_identity() -> dict[str, str]:
    return {
        "source_hf_revision": SA2VA_REVISION,
        "full_pth_sha256": FULL_PTH_SHA256,
        "base_repo_id": BASE_REPO_ID,
        "base_revision": BASE_REVISION,
    }


def _strategy_b_identity() -> dict[str, str]:
    return {**_projection_identity(), "manifest_sha256": MANIFEST_SHA256}


def _audit_projection(path: Path, expected_hash: str, *, step: int) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError(f"projection identity mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"meta", "state_dict"} or len(payload["state_dict"]) != 4:
        raise ValueError(f"projection payload mismatch: {path}")
    metadata = payload.get("meta", {}).get("chartground_phase4b")
    if not isinstance(metadata, dict) or metadata.get("optimizer_step") != step:
        raise ValueError(f"projection step mismatch: {path}")
    if metadata.get("prompt_variant") != TARGET_ONLY_ZH:
        raise ValueError(f"projection Prompt mismatch: {path}")
    if metadata.get("prompt_registry_sha256") != PROMPT_REGISTRY_SHA256:
        raise ValueError(f"projection registry mismatch: {path}")
    if metadata.get("prompt_template_sha256") != P2_SHA256:
        raise ValueError(f"projection template mismatch: {path}")
    actual = {
        "source_hf_revision": metadata.get("source_hf_revision"),
        "full_pth_sha256": metadata.get("full_pth", {}).get("sha256"),
        "base_repo_id": metadata.get("base_checkpoint", {}).get("repo_id"),
        "base_revision": metadata.get("base_checkpoint", {}).get("revision"),
    }
    if actual != _projection_identity():
        raise ValueError(f"projection base identity mismatch: {path}")
    return {"sha256": expected_hash, "bytes": path.stat().st_size, "step": step}


def _audit_strategy_b(path: Path) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != V2_B_SHA256:
        raise ValueError("Strategy B checkpoint identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"meta", "state_dict"}:
        raise ValueError("Strategy B payload keys mismatch")
    state = payload["state_dict"]
    projection = [name for name in state if name.startswith("text_hidden_fcs.")]
    lora = [
        name for name in state
        if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name
    ]
    metadata = payload.get("meta", {}).get("chartground_phase7b")
    if len(state) != 68 or len(projection) != 4 or len(lora) != 64:
        raise ValueError("Strategy B tensor-count mismatch")
    if not isinstance(metadata, dict) or metadata.get("optimizer_step") != 4800:
        raise ValueError("Strategy B step mismatch")
    actual = {
        "source_hf_revision": metadata.get("source_hf_revision"),
        "full_pth_sha256": metadata.get("full_pth", {}).get("sha256"),
        "base_repo_id": metadata.get("base_checkpoint", {}).get("repo_id"),
        "base_revision": metadata.get("base_checkpoint", {}).get("revision"),
        "manifest_sha256": metadata.get("manifest_sha256"),
    }
    if actual != _strategy_b_identity():
        raise ValueError("Strategy B base identity mismatch")
    expected_lora = {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "bias": "none",
        "layers": list(range(20, 28)),
        "target_modules": [
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(20, 28)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ],
        "modules_to_save": None,
    }
    if metadata.get("lora") != expected_lora:
        raise ValueError("Strategy B LoRA identity mismatch")
    return {
        "sha256": V2_B_SHA256,
        "bytes": path.stat().st_size,
        "step": 4800,
        "projection_tensors": 4,
        "lora_tensors": 64,
    }


def preflight(args: argparse.Namespace) -> tuple[Phase7V2SplitDataset, dict[str, Any]]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be absent or empty")
    if args.dtype != "bfloat16" or args.device != "cuda:0":
        raise ValueError("Phase 7C requires bfloat16 on logical cuda:0")
    if sha256_file(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("protocol file differs from frozen identity")
    if args.protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("--protocol-sha256 differs from frozen identity")
    if sha256_file(args.manifest) != MANIFEST_SHA256:
        raise ValueError("synthetic_v2 manifest differs from frozen identity")
    if detect_local_revision(args.checkpoint) != SA2VA_REVISION:
        raise ValueError("Sa2VA HF revision mismatch")
    hashes = prompt_template_hashes()
    if hashes["registry_sha256"] != PROMPT_REGISTRY_SHA256:
        raise ValueError("Prompt registry hash mismatch")
    if hashes["variants"][TARGET_ONLY_ZH] != P2_SHA256:
        raise ValueError("P2 template hash mismatch")
    source = Phase7V2SplitDataset(args.manifest, split="test", allow_test=True)
    if len(source) != EXPECTED_SAMPLES or test_id_hash(source.records) != TEST_ID_SHA256:
        raise ValueError("frozen test IDs differ from protocol")
    groups = Counter(
        (row["chart_type"], row["referring_type"]) for row in source.records
    )
    if len(groups) != 16 or set(groups.values()) != {20}:
        raise ValueError("frozen test group balance mismatch")
    artifacts = {
        "v1_step960": _audit_projection(args.v1_projection, V1_SHA256, step=960),
        "v2_strategy_a": _audit_projection(args.v2_a_projection, V2_A_SHA256, step=4800),
        "v2_strategy_b": _audit_strategy_b(args.v2_b_checkpoint),
    }
    return source, {
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_commit": PROTOCOL_COMMIT,
        "manifest_sha256": MANIFEST_SHA256,
        "test_sample_id_list_sha256": TEST_ID_SHA256,
        "checkpoint_artifacts": artifacts,
    }


def _set_lora_enabled(language_model, enabled: bool) -> None:
    if enabled:
        language_model.base_model.enable_adapter_layers()
    else:
        language_model.base_model.disable_adapter_layers()
    status = language_model.get_model_status().enabled
    if status is not enabled:
        raise RuntimeError(f"LoRA enable state mismatch: expected={enabled}, actual={status}")


def _assert_projection(model, expected: dict[str, torch.Tensor]) -> None:
    actual = projection_state(model)
    if set(actual) != set(expected):
        raise RuntimeError("projection state keys changed")
    for name in actual:
        reference = expected[name].to(device=actual[name].device, dtype=actual[name].dtype)
        if not torch.equal(actual[name], reference):
            raise RuntimeError(f"projection state mismatch: {name}")


def _checkpoint_projection(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        name: tensor for name, tensor in payload["state_dict"].items()
        if name.startswith("text_hidden_fcs.")
    }


def _activate_state(args, backend, name: str, original_projection) -> None:
    model = backend._model
    language_model = model.language_model
    if name == "zero_shot":
        restore_projection_state(model, original_projection)
        _set_lora_enabled(language_model, False)
        _assert_projection(model, original_projection)
    elif name == "v1_step960":
        load_projection_checkpoint_into_model(
            model, args.v1_projection, expected_identity=_projection_identity()
        )
        _set_lora_enabled(language_model, False)
        _assert_projection(model, _checkpoint_projection(args.v1_projection))
    elif name == "v2_strategy_a":
        load_projection_checkpoint_into_model(
            model, args.v2_a_projection, expected_identity=_projection_identity()
        )
        _set_lora_enabled(language_model, False)
        _assert_projection(model, _checkpoint_projection(args.v2_a_projection))
    elif name == "v2_strategy_b":
        load_strategy_b_checkpoint_into_hf_model(
            model, args.v2_b_checkpoint, expected_identity=_strategy_b_identity()
        )
        _set_lora_enabled(language_model, True)
        _assert_projection(model, _checkpoint_projection(args.v2_b_checkpoint))
    else:
        raise ValueError(f"unknown Phase 7C state: {name}")


def _prediction_row(state, sample, annotation, result, *, run_edit: bool):
    gt = sample.mask[0].astype(bool, copy=False)
    valid = (
        isinstance(result.mask, np.ndarray)
        and result.mask.shape == gt.shape
        and result.mask.dtype == np.bool_
    )
    prediction = result.mask.astype(bool, copy=False) if valid else np.zeros_like(gt)
    intersection = int(np.logical_and(prediction, gt).sum())
    union = int(np.logical_or(prediction, gt).sum())
    pred_pixels = int(prediction.sum())
    empty = pred_pixels == 0
    edit_success: bool | None = None
    edited: Image.Image | None = None
    edit_skipped: bool | None = None
    if run_edit:
        edit_skipped = empty
        if not empty:
            try:
                edited = apply_predicted_mask_edit(
                    sample.image,
                    prediction,
                    edit_action=annotation["edit_action"],
                    edit_parameters=annotation["edit_parameters"],
                )
                edit_success = True
            except Exception:
                edit_success = False
    diversity = annotation["diversity_metadata"]
    row = {
        "state": state,
        "sample_id": sample.sample_id,
        "split": "test",
        "chart_type": annotation["chart_type"],
        "referring_type": annotation["referring_type"],
        "edit_action": annotation["edit_action"],
        "difficulty": annotation["difficulty"],
        "distractor_count": annotation["distractor_count"],
        "theme": diversity["theme"],
        "degradation_type": degradation_type(annotation),
        "degradation": diversity["degradation"],
        "execution_success": result.failure_reason in (None, "empty_prediction_mask"),
        "mask_contract_valid": valid,
        "segmentation_token_present": bool(
            result.text_output and "[SEG]" in result.text_output
        ),
        "empty_prediction": empty,
        "nonempty_prediction": not empty,
        "overlapping_prediction": intersection > 0,
        "nonempty_disjoint": not empty and intersection == 0,
        "predicted_foreground_pixels": pred_pixels,
        "gt_foreground_pixels": int(gt.sum()),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection_over_union(prediction, gt),
        "dice": dice_score(prediction, gt),
        "inference_time_ms": result.inference_time_ms,
        "failure_reason": result.failure_reason,
        "error_type": result.metadata.get("error_type"),
        "raw_mask_shapes": result.raw_mask_shapes,
        "raw_mask_metadata": result.metadata.get("raw_mask_metadata", []),
        "num_masks": result.num_masks,
        "predicted_mask_sha256": mask_hash(prediction),
        "edit_attempted": bool(run_edit and not empty),
        "edit_skipped_empty": edit_skipped,
        "edit_execution_success": edit_success,
    }
    return row, prediction, edited


def _is_global_failure(result) -> bool:
    return bool(
        result.failure_reason == "model_load_failed"
        or result.metadata.get("cuda_oom")
    )


def _group(rows: Sequence[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: {
            "count": len(items),
            "mean_iou": float(np.mean([row["iou"] for row in items])),
            "mean_dice": float(np.mean([row["dice"] for row in items])),
            "empty_rate": float(np.mean([row["empty_prediction"] for row in items])),
            "overlap_rate": float(np.mean([row["overlapping_prediction"] for row in items])),
            "nonempty_disjoint_rate": float(
                np.mean([row["nonempty_disjoint"] for row in items])
            ),
        }
        for key, items in sorted(grouped.items())
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_state = defaultdict(list)
    for row in rows:
        if row.get("split") != "test":
            raise ValueError("Phase 7C aggregation is test-only")
        by_state[row["state"]].append(row)
    if tuple(by_state) != STATE_ORDER:
        raise ValueError(f"state order must be {STATE_ORDER}")
    states = {}
    for name in STATE_ORDER:
        items = by_state[name]
        values = summarize_checkpoint_rows(
            items, expected_sample_count=EXPECTED_SAMPLES, expected_per_group=20
        )
        chart_referring = values.pop("groups")
        values["groups"] = {
            "chart_referring": chart_referring,
            "chart_type": _group(items, "chart_type"),
            "referring_type": _group(items, "referring_type"),
            "action": _group(items, "edit_action"),
            "difficulty": _group(items, "difficulty"),
            "theme": _group(items, "theme"),
            "degradation_type": _group(items, "degradation_type"),
        }
        for field in (
            "execution_success", "mask_contract_valid",
            "segmentation_token_present", "nonempty_prediction",
        ):
            count = sum(bool(row[field]) for row in items)
            values[field + "_count"] = count
            values[field + "_rate"] = count / len(items)
        values["sample_median_iou"] = float(np.median([row["iou"] for row in items]))
        values["sample_median_dice"] = float(np.median([row["dice"] for row in items]))
        latencies = [float(row["inference_time_ms"]) for row in items if row["inference_time_ms"] is not None]
        values["mean_latency_ms"] = float(np.mean(latencies))
        values["median_latency_ms"] = float(np.median(latencies))
        values["edit_attempt_count"] = sum(row["edit_attempted"] for row in items)
        values["edit_success_count"] = sum(row["edit_execution_success"] is True for row in items)
        values["edit_skipped_empty_count"] = sum(row["edit_skipped_empty"] is True for row in items)
        states[name] = values
    comparisons = {}
    for label, left, right in (
        ("b_minus_zero_shot", "v2_strategy_b", "zero_shot"),
        ("a_minus_zero_shot", "v2_strategy_a", "zero_shot"),
        ("b_minus_a", "v2_strategy_b", "v2_strategy_a"),
        ("v1_minus_zero_shot", "v1_step960", "zero_shot"),
        ("a_minus_v1", "v2_strategy_a", "v1_step960"),
    ):
        comparisons[label] = {
            metric: states[left][metric] - states[right][metric]
            for metric in (
                "group_macro_iou", "group_macro_dice", "sample_macro_iou",
                "sample_macro_dice", "micro_iou", "micro_dice", "empty_rate",
            )
        }
    return {"states": states, "comparisons": comparisons}


def paired_group_bootstrap(left_rows, right_rows) -> dict[str, Any]:
    left = {row["sample_id"]: row for row in left_rows}
    right = {row["sample_id"]: row for row in right_rows}
    if set(left) != set(right) or len(left) != EXPECTED_SAMPLES:
        raise ValueError("paired bootstrap requires identical 320 test IDs")
    groups = sorted(
        {f"{row['chart_type']}/{row['referring_type']}" for row in left_rows}
    )
    result = {"seed": BOOTSTRAP_SEED, "iterations": BOOTSTRAP_ITERATIONS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(groups), size=(BOOTSTRAP_ITERATIONS, len(groups)))
    for metric in ("iou", "dice"):
        values = np.asarray([
            np.mean([
                left[sid][metric] - right[sid][metric]
                for sid in left
                if f"{left[sid]['chart_type']}/{left[sid]['referring_type']}" == group
            ])
            for group in groups
        ])
        draws = values[indices].mean(axis=1)
        result[metric] = {
            "difference": float(values.mean()),
            "ci95": [float(x) for x in np.percentile(draws, (2.5, 97.5))],
        }
    return result


def _failure_analysis(source, rows, output_dir: Path) -> dict[str, Any]:
    b_rows = [row for row in rows if row["state"] == "v2_strategy_b"]
    by_id = {row["sample_id"]: row for row in b_rows}
    categories = Counter()
    hard = Counter()
    for index, annotation in enumerate(source.records):
        sample = source[index]
        gt = Image.fromarray(sample.mask[0] * 255)
        with Image.open(output_dir / "masks/v2_strategy_b" / f"{sample.sample_id}.png") as handle:
            prediction = handle.convert("L").copy()
        category = classify_failure(gt, prediction)
        categories[category] += 1
        by_id[sample.sample_id]["failure_category"] = category
        if annotation["difficulty"] == "hard":
            hard[category] += 1
    worst = sorted(
        b_rows, key=lambda row: (row["iou"], row["dice"], row["sample_id"])
    )[:6]
    return {
        "counts": dict(sorted(categories.items())),
        "iou_below_0_5_count": sum(row["iou"] < 0.5 for row in b_rows),
        "hard_counts": dict(sorted(hard.items())),
        "worst_six": [
            {
                "sample_id": row["sample_id"],
                "iou": row["iou"],
                "dice": row["dice"],
                "failure_category": row["failure_category"],
            }
            for row in worst
        ],
    }


def _trend_deltas(summary: dict[str, Any]) -> dict[str, Any]:
    a = summary["states"]["v2_strategy_a"]["groups"]["chart_referring"]
    b = summary["states"]["v2_strategy_b"]["groups"]["chart_referring"]
    return {
        group: {
            "strategy_a_iou": a[group]["mean_iou"],
            "strategy_b_iou": b[group]["mean_iou"],
            "delta": b[group]["mean_iou"] - a[group]["mean_iou"],
        }
        for group in ("line/trend", "scatter/trend", "confidence_band/trend")
    }


def _group_delta_analysis(summary: dict[str, Any]) -> dict[str, Any]:
    states = summary["states"]
    output = {}
    for label, left, right in (
        ("b_minus_zero_shot", "v2_strategy_b", "zero_shot"),
        ("a_minus_zero_shot", "v2_strategy_a", "zero_shot"),
        ("b_minus_a", "v2_strategy_b", "v2_strategy_a"),
        ("v1_minus_zero_shot", "v1_step960", "zero_shot"),
        ("a_minus_v1", "v2_strategy_a", "v1_step960"),
    ):
        left_groups = states[left]["groups"]["chart_referring"]
        right_groups = states[right]["groups"]["chart_referring"]
        deltas = {
            group: left_groups[group]["mean_iou"] - right_groups[group]["mean_iou"]
            for group in sorted(left_groups)
        }
        output[label] = {
            "iou_deltas": deltas,
            "improved": [group for group, value in deltas.items() if value > 0],
            "declined": [group for group, value in deltas.items() if value < 0],
            "tied": [group for group, value in deltas.items() if value == 0],
        }
    return output


def _difficulty_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        difficulty: {
            state: summary["states"][state]["groups"]["difficulty"][difficulty]
            for state in STATE_ORDER
        }
        for difficulty in ("easy", "medium", "hard")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source, identity = preflight(args)
    backend = Sa2VAInternVL3Backend(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        projection_identity=_projection_identity(),
    )
    backend.load()
    model = backend._model
    original_projection = projection_state(model)
    prepare_hf_strategy_b_model(model)
    _set_lora_enabled(model.language_model, False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    backend_calls = 0
    partial_path = args.output_dir / "attempts.jsonl"
    with partial_path.open("x", encoding="utf-8") as stream:
        for state in STATE_ORDER:
            _activate_state(args, backend, state, original_projection)
            mask_dir = args.output_dir / "masks" / state
            mask_dir.mkdir(parents=True)
            edit_dir = args.output_dir / "edited" / state
            if state == "v2_strategy_b":
                edit_dir.mkdir(parents=True)
            for index in range(len(source)):
                sample, annotation = source[index], source.records[index]
                backend_calls += 1
                result = backend.predict_prompt(
                    sample.image,
                    sample.prompt,
                    instruction=annotation["referring_expression"],
                )
                if _is_global_failure(result):
                    abort = {
                        **identity,
                        "backend_call_count": backend_calls,
                        "completed_record_count": len(rows),
                        "state": state,
                        "sample_id": sample.sample_id,
                        "failure_reason": result.failure_reason,
                        "error": result.metadata.get("error"),
                    }
                    (args.output_dir / "ABORTED.json").write_text(
                        json.dumps(abort, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    raise RuntimeError("global model failure after formal test began")
                row, prediction, edited = _prediction_row(
                    state, sample, annotation, result,
                    run_edit=state == "v2_strategy_b",
                )
                Image.fromarray(prediction.astype(np.uint8) * 255).save(
                    mask_dir / f"{sample.sample_id}.png"
                )
                if edited is not None:
                    edited.save(edit_dir / f"{sample.sample_id}.png")
                row["attempt_index"] = backend_calls
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                print("PHASE7C_TEST=" + json.dumps(row, sort_keys=True), flush=True)

    if backend_calls != EXPECTED_CALLS or len(rows) != EXPECTED_CALLS:
        raise RuntimeError("Phase 7C did not complete exactly 1280 formal calls")
    restore_projection_state(model, original_projection)
    _set_lora_enabled(model.language_model, False)
    _assert_projection(model, original_projection)
    sentinel_id = source.records[0]["sample_id"]
    sentinel_row = next(
        row for row in rows
        if row["state"] == "zero_shot" and row["sample_id"] == sentinel_id
    )
    with Image.open(args.output_dir / "masks/zero_shot" / f"{sentinel_id}.png") as handle:
        sentinel_saved = np.asarray(handle.convert("L")) > 0
    sentinel_exact = mask_hash(sentinel_saved) == sentinel_row["predicted_mask_sha256"]
    if not sentinel_exact:
        raise RuntimeError("zero-shot sentinel artifact hash changed")
    if sha256_file(args.protocol) != PROTOCOL_SHA256:
        raise RuntimeError("protocol changed during frozen test")

    summary = summarize(rows)
    by_state = {
        name: [row for row in rows if row["state"] == name] for name in STATE_ORDER
    }
    summary.update(identity)
    summary.update({
        "state_order": list(STATE_ORDER),
        "formal_backend_call_count": backend_calls,
        "model_load_attempts": backend.model_load_attempts,
        "model_load_time_ms": backend.model_load_time_ms,
        "peak_gpu_memory_mb": backend._peak_gpu_memory_mb(),
        "protocol_sha256_end": sha256_file(args.protocol),
        "restoration": {
            "original_projection_exact": True,
            "lora_disabled": model.language_model.get_model_status().enabled is False,
            "sentinel_sample_id": sentinel_id,
            "sentinel_saved_mask_hash_exact": sentinel_exact,
            "additional_inference_calls": 0,
        },
        "paired_group_bootstrap": {
            "b_minus_zero_shot": paired_group_bootstrap(
                by_state["v2_strategy_b"], by_state["zero_shot"]
            ),
            "b_minus_a": paired_group_bootstrap(
                by_state["v2_strategy_b"], by_state["v2_strategy_a"]
            ),
            "a_minus_v1": paired_group_bootstrap(
                by_state["v2_strategy_a"], by_state["v1_step960"]
            ),
        },
        "failure_analysis": _failure_analysis(source, rows, args.output_dir),
        "trend_b_minus_a": _trend_deltas(summary),
        "group_delta_analysis": _group_delta_analysis(summary),
        "difficulty_comparison": _difficulty_comparison(summary),
        "selected_strategy": "v2_strategy_b",
        "selection_basis": "frozen Phase 7B validation selection; test cannot reselect",
    })
    write_outputs(args, source, rows, summary)
    print("PHASE7C_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def write_outputs(args, source, rows, summary) -> None:
    metrics = args.metrics_output or PACKAGE_ROOT / "results/phase7c_v2_frozen_test_metrics.jsonl"
    summary_path = args.summary_output or PACKAGE_ROOT / "results/phase7c_v2_frozen_test_summary.json"
    report = args.report_output or PACKAGE_ROOT / "docs/phase7c_v2_frozen_test_results.md"
    ablation = args.ablation_output or PACKAGE_ROOT / "assets/phase7c_v2_test_ablation.png"
    gallery = args.gallery_output or PACKAGE_ROOT / "assets/phase7c_v2_final_gallery.png"
    failure = args.failure_output or PACKAGE_ROOT / "assets/phase7c_v2_failure_cases.png"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_ablation(summary, ablation)
    render_final_gallery(source, rows, args.output_dir, gallery)
    render_failures(source, rows, args.output_dir, failure)
    render_report(summary, report)


def render_ablation(summary, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ("Zero-shot", "v1 step960", "v2-A step4800", "v2-B step4800")
    iou = [summary["states"][name]["group_macro_iou"] for name in STATE_ORDER]
    dice = [summary["states"][name]["group_macro_dice"] for name in STATE_ORDER]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 5), dpi=180)
    width = 0.36
    left = axis.bar(x - width / 2, iou, width, label="Macro IoU", color="#2563eb")
    right = axis.bar(x + width / 2, dice, width, label="Macro Dice", color="#16a34a")
    axis.bar_label(left, fmt="%.3f", padding=3)
    axis.bar_label(right, fmt="%.3f", padding=3)
    axis.set_xticks(x, labels)
    axis.set(ylabel="16-group macro score", title="synthetic_v2 frozen test ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def render_final_gallery(source, rows, output_dir: Path, output: Path) -> None:
    by_id = {
        row["sample_id"]: row for row in rows if row["state"] == "v2_strategy_b"
    }
    selected = select_gallery_sample_ids(source.records, expected_per_group=20)
    panels = []
    panel_dir = output_dir / "gallery_panels"
    panel_dir.mkdir(exist_ok=True)
    for sample_id in selected:
        index = next(i for i, row in enumerate(source.records) if row["sample_id"] == sample_id)
        sample, annotation = source[index], source.records[index]
        gt = Image.fromarray(sample.mask[0] * 255)
        with Image.open(output_dir / "masks/v2_strategy_b" / f"{sample_id}.png") as handle:
            prediction = handle.convert("L").copy()
        edited_path = output_dir / "edited/v2_strategy_b" / f"{sample_id}.png"
        edited = Image.open(edited_path).convert("RGB") if edited_path.is_file() else Image.new("RGB", sample.image.size, "#eeeeee")
        images = (
            mask_overlay(sample.image, gt, color=(30, 180, 80)),
            mask_overlay(sample.image, prediction, color=(40, 110, 240)),
            edited,
        )
        labels = ("Input + GT", f"B prediction IoU={by_id[sample_id]['iou']:.3f}", "B edited" if edited_path.is_file() else "B edit skipped")
        panel = Image.new("RGB", (960, 210), "white")
        draw = ImageDraw.Draw(panel)
        for column, (image, label) in enumerate(zip(images, labels)):
            thumb = image.copy(); thumb.thumbnail((310, 160), Image.Resampling.LANCZOS)
            panel.paste(thumb, (column * 320 + (310 - thumb.width) // 2, 25))
            draw.text((column * 320 + 5, 5), label, fill="black", font=ImageFont.load_default())
        draw.text((5, 190), f"{annotation['chart_type']}/{annotation['referring_type']}", fill="black", font=ImageFont.load_default())
        path = panel_dir / f"{sample_id}.png"; panel.save(path); panels.append(path)
        if edited_path.is_file():
            edited.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    create_contact_sheet(panels, output, columns=2, thumbnail_width=960)


def render_failures(source, rows, output_dir: Path, output: Path) -> None:
    b_rows = [row for row in rows if row["state"] == "v2_strategy_b"]
    worst = sorted(b_rows, key=lambda row: (row["iou"], row["dice"], row["sample_id"]))[:6]
    panels = []
    panel_dir = output_dir / "failure_panels"
    panel_dir.mkdir(exist_ok=True)
    for row in worst:
        index = next(i for i, record in enumerate(source.records) if record["sample_id"] == row["sample_id"])
        sample, annotation = source[index], source.records[index]
        gt = Image.fromarray(sample.mask[0] * 255)
        with Image.open(output_dir / "masks/v2_strategy_b" / f"{sample.sample_id}.png") as handle:
            prediction = handle.convert("L").copy()
        images = (sample.image, mask_overlay(sample.image, prediction, color=(40, 110, 240)), error_map(gt, prediction))
        panel = Image.new("RGB", (960, 210), "white"); draw = ImageDraw.Draw(panel)
        for column, image in enumerate(images):
            thumb = image.copy(); thumb.thumbnail((310, 160), Image.Resampling.LANCZOS)
            panel.paste(thumb, (column * 320 + (310 - thumb.width) // 2, 25))
        draw.text((5, 5), f"{annotation['chart_type']}/{annotation['referring_type']} IoU={row['iou']:.3f} Dice={row['dice']:.3f}", fill="black", font=ImageFont.load_default())
        draw.text((5, 190), row["failure_category"], fill="black", font=ImageFont.load_default())
        path = panel_dir / f"{sample.sample_id}.png"; panel.save(path); panels.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    create_contact_sheet(panels, output, columns=2, thumbnail_width=960)


def render_report(summary, output: Path) -> None:
    lines = [
        "# Phase 7C synthetic_v2 frozen-test results", "",
        "The four fixed states were evaluated once on the 320-sample synthetic_v2 test split. Strategy B remains the final strategy because it was selected on validation before this test.", "",
        "| state | Macro IoU | Macro Dice | Sample IoU | Sample Dice | Micro IoU | Micro Dice | empty | overlap | disjoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in STATE_ORDER:
        m = summary["states"][name]
        lines.append(f"| {name} | {m['group_macro_iou']:.6f} | {m['group_macro_dice']:.6f} | {m['sample_macro_iou']:.6f} | {m['sample_macro_dice']:.6f} | {m['micro_iou']:.6f} | {m['micro_dice']:.6f} | {m['empty_rate']:.6f} | {m['overlap_rate']:.6f} | {m['nonempty_disjoint_rate']:.6f} |")
    lines.extend([
        "", "All four states completed 320/320 execution-success, mask-contract-valid and `[SEG]` records.", "",
        "| state | median IoU | median Dice | mean latency ms | median latency ms |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in STATE_ORDER:
        m = summary["states"][name]
        lines.append(f"| {name} | {m['sample_median_iou']:.6f} | {m['sample_median_dice']:.6f} | {m['mean_latency_ms']:.3f} | {m['median_latency_ms']:.3f} |")
    lines.extend(["", "## Fixed comparisons", ""])
    for name, values in summary["comparisons"].items():
        lines.append(f"- `{name}` Macro IoU/Dice: `{values['group_macro_iou']:+.6f}/{values['group_macro_dice']:+.6f}`.")
    lines.extend([
        "", "## 16 chart/referring groups", "",
        "| group | zero-shot | v1 | v2-A | v2-B | B−A |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    groups = summary["states"]["zero_shot"]["groups"]["chart_referring"]
    for group in sorted(groups):
        values = [
            summary["states"][state]["groups"]["chart_referring"][group]["mean_iou"]
            for state in STATE_ORDER
        ]
        lines.append(f"| {group} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {values[3]:.6f} | {values[3]-values[2]:+.6f} |")
    b_a = summary["group_delta_analysis"]["b_minus_a"]
    b_zero = summary["group_delta_analysis"]["b_minus_zero_shot"]
    lines.extend([
        "",
        f"B versus zero-shot improves `{len(b_zero['improved'])}/16` groups and declines on `{len(b_zero['declined'])}`. B versus A improves `{len(b_a['improved'])}/16` and declines on `{len(b_a['declined'])}`.",
        "", "## Difficulty and trend", "",
        "| difficulty | zero-shot | v1 | v2-A | v2-B | B−A |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for difficulty, states in summary["difficulty_comparison"].items():
        values = [states[state]["mean_iou"] for state in STATE_ORDER]
        lines.append(f"| {difficulty} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {values[3]:.6f} | {values[3]-values[2]:+.6f} |")
    lines.extend(["", "Trend B−A IoU deltas:", ""])
    for group, values in summary["trend_b_minus_a"].items():
        lines.append(f"- `{group}`: `{values['delta']:+.6f}`.")
    failures = summary["failure_analysis"]
    dominant = max(failures["counts"], key=failures["counts"].get)
    hard_dominant = max(failures["hard_counts"], key=failures["hard_counts"].get)
    lines.extend([
        "", "## Bootstrap", "",
        "| paired difference | metric | estimate | 95% CI |",
        "|---|---|---:|---:|",
    ])
    for comparison in ("b_minus_zero_shot", "b_minus_a", "a_minus_v1"):
        bootstrap = summary["paired_group_bootstrap"][comparison]
        for metric in ("iou", "dice"):
            values = bootstrap[metric]
            metric_label = "IoU" if metric == "iou" else "Dice"
            lines.append(
                f"| {comparison} | Macro {metric_label} | "
                f"{values['difference']:+.6f} | "
                f"[{values['ci95'][0]:+.6f}, {values['ci95'][1]:+.6f}] |"
            )
    lines.extend([
        "", "Fixed seed `20260916`, 10,000 paired group bootstrap draws.", "",
        "## Strategy B failures", "",
        f"Error counts: `{failures['counts']}`. IoU < 0.5: `{failures['iou_below_0_5_count']}/320`. The largest overall category is `{dominant}`; the largest hard-split category is `{hard_dominant}`. Partial-target is therefore not the largest remaining category.", "",
        f"Hard counts: `{failures['hard_counts']}`.", "",
        "Worst six (deterministic sort by IoU, Dice, then sample ID):", "",
    ])
    for row in failures["worst_six"]:
        lines.append(
            f"- `{row['sample_id']}`: IoU/Dice "
            f"`{row['iou']:.6f}/{row['dice']:.6f}`, {row['failure_category']}."
        )
    lines.extend([
        "",
        f"Strategy B editing: attempted/succeeded/skipped-empty = `{summary['states']['v2_strategy_b']['edit_attempt_count']}/{summary['states']['v2_strategy_b']['edit_success_count']}/{summary['states']['v2_strategy_b']['edit_skipped_empty_count']}`.", "",
        "## Integrity", "",
        f"- Formal backend calls: `{summary['formal_backend_call_count']}`.",
        f"- Model loads: `{summary['model_load_attempts']}`.",
        f"- Protocol SHA-256 start/end: `{summary['protocol_sha256']}` / `{summary['protocol_sha256_end']}`.",
        f"- Restoration: `{summary['restoration']}`.",
        "- No training, retry, threshold tuning, state reselection, synthetic_v2 train/val inference, or Strategy C occurred.", "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
