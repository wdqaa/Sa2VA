#!/usr/bin/env python3
"""Run the frozen P0/P1/P2 benchmark on balanced synthetic_v1 val only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets import ChartGroundV1Dataset
from chartground_edit.datasets.schema_v1 import validate_jsonl_v1
from chartground_edit.inference import (
    PROMPT_VARIANT_ORDER,
    Sa2VAInternVL3Backend,
    binary_pixel_counts,
    build_prompt_variant,
    dice_score,
    intersection_over_union,
    is_global_failure,
)
from chartground_edit.inference.prompt_benchmark_v1 import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    EXPECTED_ATTEMPTS,
    EXPECTED_VAL_SAMPLES,
    build_attempt_plan,
    prompt_template_hashes,
    result_metric_semantics,
    rotated_prompt_order,
    select_gallery_sample_ids,
    sha256_file,
    summarize_benchmark,
    validate_balanced_val_annotations,
    validate_benchmark_split,
    validate_result_matrix,
)
from chartground_edit.visualization.render import mask_overlay


MODEL_NAME = "ByteDance/Sa2VA-InternVL3-2B"
CHECKPOINT_REVISION = "15837dcaecc304714a1f0f069e74f47e47521c7f"
EXPECTED_MANIFEST_SHA256 = (
    "ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82"
)
EXPECTED_PROTOCOL_SHA256 = (
    "5de44dcf8e21845aaf3d8c5355faedd62b80899cc54b698e7fe2731aa839ef1d"
)
EXPECTED_PROMPT_REGISTRY_SHA256 = (
    "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0"
)
DEFAULT_PROTOCOL = Path(
    "projects/chartground_edit/docs/phase3b_benchmark_protocol.md"
)
DEFAULT_REPOSITORY_RESULTS = Path("projects/chartground_edit/results")
DEFAULT_GALLERY = Path(
    "projects/chartground_edit/assets/phase3b_balanced_val_prompt_benchmark.png"
)


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def existing_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def val_only_split(value: str) -> str:
    try:
        return validate_benchmark_split(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def cuda_device(value: str) -> str:
    if not value.startswith("cuda:") or not value[5:].isdigit():
        raise argparse.ArgumentTypeError(
            "device must use logical CUDA syntax such as cuda:0"
        )
    return value


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=existing_directory)
    parser.add_argument("--manifest", required=True, type=existing_file)
    parser.add_argument("--split", default="val", type=val_only_split)
    parser.add_argument("--device", default="cuda:0", type=cuda_device)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument("--expected-samples", required=True, type=positive_integer)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--repository-results-dir",
        type=Path,
        default=DEFAULT_REPOSITORY_RESULTS,
    )
    parser.add_argument("--gallery-output", type=Path, default=DEFAULT_GALLERY)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        preflight = _preflight(args)
    except (OSError, ValueError) as exc:
        print(f"preflight_error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_path = args.output_dir / "run_metadata.json"
    _write_json(
        run_metadata_path,
        {
            **preflight,
            "started_at": started_at,
            "status": "running",
            "expected_attempts": EXPECTED_ATTEMPTS,
        },
    )

    dataset = ChartGroundV1Dataset(args.manifest)
    val_annotations = validate_balanced_val_annotations(
        dataset.annotations,
        split=args.split,
        expected_samples=args.expected_samples,
    )
    samples = [sample for sample in dataset if sample.annotation["split"] == "val"]
    if [sample.annotation["sample_id"] for sample in samples] != [
        row["sample_id"] for row in val_annotations
    ]:
        raise RuntimeError("val Reader order differs from validated manifest order")

    backend = Sa2VAInternVL3Backend(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    results: list[dict[str, Any]] = []
    partial_path = args.output_dir / "results.partial.jsonl"
    global_stop_reason: str | None = None
    with partial_path.open("w", encoding="utf-8", buffering=1) as partial:
        for sample_index, sample in enumerate(samples):
            sample_dir = args.output_dir / "samples" / sample.annotation["sample_id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            original_path = sample_dir / "original.png"
            gt_path = sample_dir / "gt_mask.png"
            sample.image.convert("RGB").save(original_path, format="PNG")
            sample.mask.convert("L").save(gt_path, format="PNG")
            for prompt_order_position, variant in enumerate(
                rotated_prompt_order(sample_index)
            ):
                attempt_index = len(results)
                exact_prompt = build_prompt_variant(
                    variant,
                    instruction=sample.annotation["full_instruction"],
                    referring_expression=sample.annotation["referring_expression"],
                )
                prediction = None
                failure_reason = global_stop_reason
                if global_stop_reason is None:
                    try:
                        prediction = backend.predict_prompt(
                            sample.image,
                            exact_prompt,
                            instruction=sample.annotation["full_instruction"],
                        )
                        failure_reason = prediction.failure_reason
                        if is_global_failure(prediction):
                            global_stop_reason = (
                                "not_run_after_global_failure:"
                                f"{prediction.failure_reason}"
                            )
                    except Exception as exc:
                        failure_reason = (
                            f"backend_exception:{type(exc).__name__}:{exc}"
                        )
                        global_stop_reason = "not_run_after_backend_exception"
                row = _save_attempt(
                    sample=sample,
                    prediction=prediction,
                    exact_prompt=exact_prompt,
                    variant=variant,
                    prompt_order_position=prompt_order_position,
                    attempt_index=attempt_index,
                    failure_reason=failure_reason,
                    output_dir=args.output_dir,
                    checkpoint=args.checkpoint,
                    checkpoint_revision=preflight["checkpoint_revision"],
                    protocol_sha256=preflight["protocol_sha256_start"],
                    manifest_sha256=preflight["manifest_sha256"],
                    prompt_hashes=preflight["prompt_template_hashes"],
                    code_commit=preflight["code_commit"],
                    dtype=args.dtype,
                    device=args.device,
                )
                results.append(row)
                partial.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )

    validate_result_matrix(results, expected_samples=args.expected_samples)
    protocol_sha256_end = sha256_file(args.protocol)
    if protocol_sha256_end != preflight["protocol_sha256_start"]:
        _write_json(
            run_metadata_path,
            {
                **preflight,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "invalid_protocol_changed_during_run",
                "protocol_sha256_end": protocol_sha256_end,
            },
        )
        print("error: protocol changed during benchmark; run is invalid", file=sys.stderr)
        return 3

    benchmark = summarize_benchmark(results)
    verification = _verify_saved_outputs(
        results,
        annotations=val_annotations,
        manifest_root=args.manifest.parent,
        manifest_sha256=preflight["manifest_sha256"],
        protocol_path=args.protocol,
        protocol_sha256=preflight["protocol_sha256_start"],
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    common = {
        **preflight,
        "protocol_sha256_end": protocol_sha256_end,
        "protocol_unchanged": True,
        "started_at": started_at,
        "finished_at": finished_at,
        "physical_gpu_mapping": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": args.device,
        "dtype": args.dtype,
        "backend_instance_count": 1,
        "model_load_attempts": backend.model_load_attempts,
        "model_load_time_ms": backend.model_load_time_ms,
        "pre_load_free_gpu_memory_mb": backend.pre_load_free_gpu_memory_mb,
        "peak_gpu_memory_mb": max(
            (
                float(row["peak_gpu_memory_mb"])
                for row in results
                if row["peak_gpu_memory_mb"] is not None
            ),
            default=None,
        ),
        "global_stop_reason": global_stop_reason,
        "attempt_count": len(results),
        "actual_backend_call_count": sum(
            not str(row.get("failure_reason") or "").startswith("not_run_after_")
            for row in results
        ),
        "verification": verification,
    }
    summary = {**common, **benchmark}
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    _write_jsonl(results_path, results)
    _write_json(summary_path, summary)
    _write_json(
        run_metadata_path,
        {
            **common,
            "status": "complete" if global_stop_reason is None else "invalid_global_stop",
        },
    )
    args.repository_results_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        args.repository_results_dir / "phase3b_balanced_val_metrics.jsonl",
        results,
    )
    _write_json(
        args.repository_results_dir / "phase3b_balanced_val_summary.json", summary
    )
    _render_gallery(
        samples,
        results,
        val_annotations,
        args.gallery_output,
    )

    load_attempts = backend.model_load_attempts
    del backend
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"samples={len(samples)}")
    print(f"attempts={len(results)}")
    print(f"actual_backend_calls={common['actual_backend_call_count']}")
    print(f"model_load_attempts={load_attempts}")
    print(f"selected_prompt={benchmark['selection']['selected_prompt']}")
    print(f"protocol_sha256={protocol_sha256_end}")
    print(f"summary={summary_path}")
    print(f"gallery={args.gallery_output}")
    return 0 if global_stop_reason is None and verification["passed"] else 1


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_samples != EXPECTED_VAL_SAMPLES:
        raise ValueError(
            f"Phase 3B requires --expected-samples {EXPECTED_VAL_SAMPLES}"
        )
    validate_benchmark_split(args.split)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            f"output directory must be absent or empty for a single formal run: "
            f"{args.output_dir}"
        )
    if not args.protocol.is_file():
        raise ValueError(f"protocol file does not exist: {args.protocol}")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("TRANSFORMERS_OFFLINE=1 is required")
    manifest_sha256 = sha256_file(args.manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"manifest SHA-256 mismatch: {manifest_sha256} != "
            f"{EXPECTED_MANIFEST_SHA256}"
        )
    protocol_sha256 = sha256_file(args.protocol)
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            f"protocol SHA-256 mismatch: {protocol_sha256} != "
            f"{EXPECTED_PROTOCOL_SHA256}"
        )
    prompt_hashes = prompt_template_hashes()
    if prompt_hashes["registry_sha256"] != EXPECTED_PROMPT_REGISTRY_SHA256:
        raise ValueError("Prompt registry SHA-256 differs from preregistration")
    annotations = validate_jsonl_v1(
        args.manifest, check_files=True, expected_count=320
    )
    validate_balanced_val_annotations(
        annotations, split=args.split, expected_samples=args.expected_samples
    )
    plan = build_attempt_plan(annotations)
    if len(plan) != EXPECTED_ATTEMPTS:
        raise ValueError(f"attempt plan must contain {EXPECTED_ATTEMPTS} entries")
    revision = _detect_local_revision(args.checkpoint)
    if revision != CHECKPOINT_REVISION:
        raise ValueError(
            f"checkpoint revision mismatch: {revision!r} != {CHECKPOINT_REVISION!r}"
        )
    return {
        "model_name": MODEL_NAME,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_revision": revision,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "protocol_path": str(args.protocol),
        "protocol_sha256_start": protocol_sha256,
        "prompt_template_hashes": prompt_hashes,
        "expected_samples": args.expected_samples,
        "expected_attempts": EXPECTED_ATTEMPTS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "code_commit": _git_commit(),
        "code_worktree_dirty_at_start": _git_dirty(),
        "gpu_name": _gpu_name(args.device),
    }


def _save_attempt(
    *,
    sample: Any,
    prediction: Any,
    exact_prompt: str,
    variant: str,
    prompt_order_position: int,
    attempt_index: int,
    failure_reason: str | None,
    output_dir: Path,
    checkpoint: Path,
    checkpoint_revision: str,
    protocol_sha256: str,
    manifest_sha256: str,
    prompt_hashes: dict[str, Any],
    code_commit: str | None,
    dtype: str,
    device: str,
) -> dict[str, Any]:
    annotation = sample.annotation
    attempt_dir = output_dir / "samples" / annotation["sample_id"] / variant
    attempt_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(sample.mask.convert("L"))
    mask_contract_valid = False
    if prediction is not None and prediction.mask is not None:
        candidate = np.asarray(prediction.mask)
        mask_contract_valid = bool(
            candidate.dtype == np.bool_
            and candidate.ndim == 2
            and candidate.shape == (sample.image.height, sample.image.width)
        )
    if mask_contract_valid:
        predicted = np.asarray(prediction.mask, dtype=bool)
        mask_source = "model"
    else:
        predicted = np.zeros((sample.image.height, sample.image.width), dtype=bool)
        mask_source = "empty_failure_placeholder"
    predicted_image = Image.fromarray(predicted.astype(np.uint8) * 255)
    predicted_path = attempt_dir / "predicted_mask.png"
    overlay_path = attempt_dir / "prediction_overlay.png"
    prompt_path = attempt_dir / "prompt.txt"
    text_path = attempt_dir / "generated_text.txt"
    result_path = attempt_dir / "result.json"
    predicted_image.save(predicted_path, format="PNG")
    mask_overlay(sample.image, predicted_image).save(overlay_path, format="PNG")
    prompt_path.write_text(exact_prompt, encoding="utf-8")
    generated_text = prediction.text_output if prediction is not None else None
    text_path.write_text(generated_text or "", encoding="utf-8")
    counts = binary_pixel_counts(predicted, gt)
    empty = counts["predicted_foreground_pixels"] == 0
    metadata = prediction.metadata if prediction is not None else {}
    semantics = result_metric_semantics(
        prediction_returned=prediction is not None,
        failure_reason=failure_reason,
        error_type=metadata.get("error_type"),
        mask_contract_valid=mask_contract_valid,
        predicted_foreground_pixels=counts["predicted_foreground_pixels"],
        intersection_pixels=counts["intersection_pixels"],
    )
    row = {
        "attempt_index": attempt_index,
        "sample_id": annotation["sample_id"],
        "split": annotation["split"],
        "chart_type": annotation["chart_type"],
        "referring_type": annotation["referring_type"],
        "edit_action": annotation["edit_action"],
        "difficulty": annotation["difficulty"],
        "distractor_count": annotation["distractor_count"],
        "full_instruction": annotation["full_instruction"],
        "referring_expression": annotation["referring_expression"],
        "prompt_variant": variant,
        "prompt_order_position": prompt_order_position,
        "exact_prompt": exact_prompt,
        "inference_success": bool(prediction is not None and prediction.success),
        "inference_success_legacy_deprecated": True,
        "segmentation_token_present": bool(
            isinstance(generated_text, str) and "[SEG]" in generated_text
        ),
        "generated_text": generated_text,
        "num_masks": prediction.num_masks if prediction is not None else 0,
        "raw_mask_shapes": prediction.raw_mask_shapes if prediction is not None else [],
        "raw_mask_metadata": metadata.get("raw_mask_metadata", []),
        "prediction_masks_python_type": metadata.get("prediction_masks_type"),
        **counts,
        **semantics,
        "iou": intersection_over_union(predicted, gt),
        "dice": dice_score(predicted, gt),
        "nonempty_disjoint": bool(
            not empty
            and counts["gt_foreground_pixels"] > 0
            and counts["intersection_pixels"] == 0
        ),
        "resized": bool(metadata.get("mask_resized_nearest", False)),
        "inference_time_ms": (
            prediction.inference_time_ms if prediction is not None else None
        ),
        "peak_gpu_memory_mb": (
            prediction.peak_gpu_memory_mb if prediction is not None else None
        ),
        "failure_reason": failure_reason,
        "error_type": metadata.get("error_type"),
        "error_message": metadata.get("error")
        or metadata.get("mask_processing_error"),
        "predicted_mask_source": mask_source,
        "predicted_mask_shape": list(predicted.shape),
        "gt_mask_shape": list(gt.shape),
        "predicted_mask_sha256": sha256_file(predicted_path),
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint),
        "checkpoint_revision": checkpoint_revision,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "prompt_template_sha256": prompt_hashes["variants"][variant],
        "prompt_registry_sha256": prompt_hashes["registry_sha256"],
        "code_commit": code_commit,
        "dtype": dtype,
        "device": device,
        "output_files": {
            "predicted_mask": str(predicted_path),
            "prediction_overlay": str(overlay_path),
            "prompt": str(prompt_path),
            "generated_text": str(text_path),
            "result": str(result_path),
        },
    }
    _write_json(result_path, row)
    return row


def _verify_saved_outputs(
    results: list[dict[str, Any]],
    *,
    annotations: list[dict[str, Any]],
    manifest_root: Path,
    manifest_sha256: str,
    protocol_path: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        validate_result_matrix(results)
    except ValueError as exc:
        errors.append(str(exc))
    if sha256_file(protocol_path) != protocol_sha256:
        errors.append("protocol hash changed")
    manifest_path = manifest_root / "annotations.jsonl"
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha256:
        errors.append("manifest hash changed")
    annotations_by_id = {row["sample_id"]: row for row in annotations}
    checked_masks = 0
    for row in results:
        annotation = annotations_by_id[row["sample_id"]]
        predicted_path = Path(row["output_files"]["predicted_mask"])
        gt_path = manifest_root / annotation["mask_path"]
        try:
            with Image.open(predicted_path) as source:
                predicted = np.asarray(source.convert("L"))
                predicted_size = source.size
            with Image.open(gt_path) as source:
                gt = np.asarray(source.convert("L"))
                gt_size = source.size
        except OSError as exc:
            errors.append(f"{row['sample_id']}/{row['prompt_variant']}: {exc}")
            continue
        if predicted_size != gt_size or predicted.shape != gt.shape:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: size mismatch"
            )
            continue
        if not set(np.unique(predicted).tolist()).issubset({0, 255}):
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: non-binary saved mask"
            )
            continue
        counts = binary_pixel_counts(predicted, gt)
        recalculated_iou = intersection_over_union(predicted, gt)
        recalculated_dice = dice_score(predicted, gt)
        for key, value in counts.items():
            if int(row[key]) != value:
                errors.append(
                    f"{row['sample_id']}/{row['prompt_variant']}: {key} mismatch"
                )
        if abs(float(row["iou"]) - recalculated_iou) > 1e-12:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: IoU mismatch"
            )
        if abs(float(row["dice"]) - recalculated_dice) > 1e-12:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: Dice mismatch"
            )
        expected_semantics = {
            "mask_contract_valid": True,
            "nonempty_prediction": counts["predicted_foreground_pixels"] > 0,
            "empty_prediction": counts["predicted_foreground_pixels"] == 0,
            "overlapping_prediction": counts["intersection_pixels"] > 0,
        }
        for key, value in expected_semantics.items():
            if row.get(key) is not value:
                errors.append(
                    f"{row['sample_id']}/{row['prompt_variant']}: {key} mismatch"
                )
        if sha256_file(predicted_path) != row["predicted_mask_sha256"]:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: mask hash mismatch"
            )
        checked_masks += 1
    return {
        "passed": not errors,
        "result_record_count": len(results),
        "checked_mask_count": checked_masks,
        "independent_metric_recalculation_count": checked_masks,
        "errors": errors,
    }


def _render_gallery(
    samples: list[Any],
    results: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    output_path: Path,
) -> None:
    selected_ids = select_gallery_sample_ids(annotations)
    sample_lookup = {sample.annotation["sample_id"]: sample for sample in samples}
    result_lookup = {
        (row["sample_id"], row["prompt_variant"]): row for row in results
    }
    panel_size = (230, 154)
    header_height = 48
    footer_height = 30
    gutter = 5
    columns = 5
    row_height = header_height + panel_size[1] + footer_height
    sheet = Image.new(
        "RGB",
        (
            columns * panel_size[0] + (columns + 1) * gutter,
            len(selected_ids) * row_height + (len(selected_ids) + 1) * gutter,
        ),
        (225, 225, 225),
    )
    font = ImageFont.load_default()
    for row_index, sample_id in enumerate(selected_ids):
        sample = sample_lookup[sample_id]
        y = gutter + row_index * (row_height + gutter)
        row_canvas = Image.new("RGB", (columns * panel_size[0], row_height), "white")
        draw = ImageDraw.Draw(row_canvas)
        annotation = sample.annotation
        draw.text(
            (7, 5),
            f"{sample_id} | {annotation['chart_type']}/{annotation['referring_type']} | "
            f"{annotation['edit_action']} | {annotation['difficulty']}",
            fill="black",
            font=font,
        )
        iou_text = " | ".join(
            f"{_short_name(variant)}={result_lookup[(sample_id, variant)]['iou']:.4f}"
            for variant in PROMPT_VARIANT_ORDER
        )
        draw.text((7, 22), f"IoU: {iou_text}", fill="black", font=font)
        gt_overlay = mask_overlay(sample.image, sample.mask)
        panels: list[tuple[str, Image.Image]] = [
            ("original", sample.image.convert("RGB")),
            ("GT overlay", gt_overlay),
        ]
        for variant in PROMPT_VARIANT_ORDER:
            result = result_lookup[(sample_id, variant)]
            with Image.open(result["output_files"]["predicted_mask"]) as source:
                predicted = source.convert("L").copy()
            panels.append(
                (
                    f"{_short_name(variant)} IoU={result['iou']:.4f}",
                    mask_overlay(sample.image, predicted),
                )
            )
        for column, (label, image) in enumerate(panels):
            thumbnail = image.copy()
            thumbnail.thumbnail(panel_size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", panel_size, "white")
            canvas.paste(
                thumbnail,
                (
                    (panel_size[0] - thumbnail.width) // 2,
                    (panel_size[1] - thumbnail.height) // 2,
                ),
            )
            x = column * panel_size[0]
            row_canvas.paste(canvas, (x, header_height))
            draw.text(
                (x + 4, header_height + panel_size[1] + 5),
                label,
                fill="black",
                font=font,
            )
        sheet.paste(row_canvas, (gutter, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def _detect_local_revision(checkpoint: Path) -> str | None:
    metadata_dir = checkpoint / ".cache/huggingface/download"
    revisions: set[str] = set()
    if metadata_dir.is_dir():
        for path in metadata_dir.glob("*.metadata"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            if lines and len(lines[0]) == 40:
                revisions.add(lines[0])
    return next(iter(revisions)) if len(revisions) == 1 else None


def _gpu_name(device: str) -> str | None:
    try:
        import torch

        return str(torch.cuda.get_device_name(int(device.split(":", 1)[1])))
    except Exception:
        return None


def _git_commit() -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ("git", "status", "--short"), capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _short_name(variant: str) -> str:
    return {
        "full_instruction": "P0",
        "target_only_en": "P1",
        "target_only_zh": "P2",
    }[variant]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
