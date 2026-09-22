#!/usr/bin/env python3
"""Run a one-shot frozen P2 evaluation on balanced synthetic_v1 test."""

from __future__ import annotations

import argparse
import gc
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
    Sa2VAInternVL3Backend,
    binary_pixel_counts,
    build_prompt_variant,
    dice_score,
    intersection_over_union,
    is_global_failure,
)
from chartground_edit.inference.frozen_test_v1 import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    EXPECTED_TEST_ATTEMPTS,
    EXPECTED_TEST_SAMPLES,
    apply_predicted_mask_edit,
    build_frozen_test_plan,
    select_frozen_test_gallery_sample_ids,
    summarize_frozen_test,
    test_sample_id_list_sha256,
    validate_balanced_test_annotations,
    validate_frozen_prompt_variant,
    validate_frozen_test_results,
    validate_frozen_test_split,
)
from chartground_edit.inference.prompt_benchmark_v1 import (
    prompt_template_hashes,
    result_metric_semantics,
    sha256_file,
)
from chartground_edit.inference.prompt_variants import TARGET_ONLY_ZH
from chartground_edit.inference.phase5b import (
    SELECTED_PROJECTION_SHA256,
    compare_with_saved_zero_shot,
    projection_identity,
    validate_one_shot_records,
    validate_selected_projection,
)
from chartground_edit.visualization.render import mask_overlay


MODEL_NAME = "ByteDance/Sa2VA-InternVL3-2B"
CHECKPOINT_REVISION = "15837dcaecc304714a1f0f069e74f47e47521c7f"
EXPECTED_MANIFEST_SHA256 = (
    "ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82"
)
EXPECTED_TEST_SAMPLE_ID_LIST_SHA256 = (
    "3061387f0012b13c2fd998d81819df0e7648e8e47ae9ad86ff06490ecfe2e5c4"
)
EXPECTED_PROMPT_REGISTRY_SHA256 = (
    "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0"
)
EXPECTED_P2_TEMPLATE_SHA256 = (
    "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806"
)
EXPECTED_PROTOCOL_SHA256 = (
    "3e2aa95825b369746db5bd667912bbfd74d0288c5c9a5ddb9c75ce3bb3593478"
)
DEFAULT_PROTOCOL = Path("projects/chartground_edit/docs/phase3c_frozen_test_protocol.md")
DEFAULT_REPOSITORY_RESULTS = Path("projects/chartground_edit/results")
DEFAULT_GALLERY = Path(
    "projects/chartground_edit/assets/phase3c_frozen_test_gallery.png"
)
DEFAULT_REPORT = Path("projects/chartground_edit/docs/phase3c_frozen_test_results.md")
DEFAULT_VAL_SUMMARY = Path(
    "projects/chartground_edit/results/phase3b_balanced_val_summary.json"
)
DEFAULT_ZERO_SHOT_SUMMARY = Path(
    "projects/chartground_edit/results/phase3c_frozen_test_summary.json"
)
DEFAULT_ZERO_SHOT_METRICS = Path(
    "projects/chartground_edit/results/phase3c_frozen_test_metrics.jsonl"
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


def test_only_split(value: str) -> str:
    try:
        return validate_frozen_test_split(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def p2_only_variant(value: str) -> str:
    try:
        return validate_frozen_prompt_variant(value)
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


def logical_cuda_zero(value: str) -> str:
    if value != "cuda:0":
        raise argparse.ArgumentTypeError(
            "Phase 3C requires the single visible physical GPU to map to cuda:0"
        )
    return value


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=("phase3c", "phase5b"), default="phase3c"
    )
    parser.add_argument("--checkpoint", required=True, type=existing_directory)
    parser.add_argument("--projection-checkpoint", type=existing_file)
    parser.add_argument("--manifest", required=True, type=existing_file)
    parser.add_argument("--split", required=True, type=test_only_split)
    parser.add_argument("--prompt-variant", required=True, type=p2_only_variant)
    parser.add_argument("--device", default="cuda:0", type=logical_cuda_zero)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument("--expected-samples", required=True, type=positive_integer)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--repository-results-dir", type=Path, default=DEFAULT_REPOSITORY_RESULTS
    )
    parser.add_argument("--gallery-output", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VAL_SUMMARY)
    parser.add_argument("--zero-shot-summary", type=Path, default=DEFAULT_ZERO_SHOT_SUMMARY)
    parser.add_argument("--zero-shot-metrics", type=Path, default=DEFAULT_ZERO_SHOT_METRICS)
    parser.add_argument("--protocol-sha256")
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
    metadata_path = args.output_dir / "run_metadata.json"
    _write_json(
        metadata_path,
        {
            **preflight,
            "started_at": started_at,
            "status": "running",
            "expected_attempts": EXPECTED_TEST_ATTEMPTS,
        },
    )

    dataset = ChartGroundV1Dataset(args.manifest)
    annotations = validate_balanced_test_annotations(
        dataset.annotations,
        split=args.split,
        expected_samples=args.expected_samples,
    )
    samples = [sample for sample in dataset if sample.annotation["split"] == "test"]
    if [sample.annotation["sample_id"] for sample in samples] != [
        row["sample_id"] for row in annotations
    ]:
        raise RuntimeError("test Reader order differs from validated manifest order")

    backend_kwargs: dict[str, Any] = {}
    if args.experiment == "phase5b":
        backend_kwargs = {
            "projection_checkpoint": args.projection_checkpoint,
            "projection_identity": projection_identity(),
        }
    backend = Sa2VAInternVL3Backend(
        args.checkpoint, device=args.device, dtype=args.dtype, **backend_kwargs
    )
    results: list[dict[str, Any]] = []
    backend_call_count = 0
    partial_path = args.output_dir / "results.partial.jsonl"
    with partial_path.open("w", encoding="utf-8", buffering=1) as partial:
        for attempt_index, sample in enumerate(samples):
            exact_prompt = build_prompt_variant(
                TARGET_ONLY_ZH,
                instruction=sample.annotation["referring_expression"],
                referring_expression=sample.annotation["referring_expression"],
            )
            prediction = None
            failure_reason: str | None = None
            backend_call_count += 1
            try:
                prediction = backend.predict_prompt(
                    sample.image,
                    exact_prompt,
                    instruction=sample.annotation["referring_expression"],
                )
                failure_reason = prediction.failure_reason
            except Exception as exc:
                failure_reason = f"backend_exception:{type(exc).__name__}:{exc}"

            if prediction is not None and is_global_failure(prediction):
                _write_json(
                    metadata_path,
                    {
                        **preflight,
                        "started_at": started_at,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "status": "aborted_hardware_or_model_load_failure",
                        "backend_call_count": backend_call_count,
                        "completed_record_count": len(results),
                        "abort_reason": prediction.failure_reason,
                        "abort_error": prediction.metadata.get("error"),
                    },
                )
                print(
                    "hardware/model-load failure invalidated the entire run; "
                    "no partial formal results were published",
                    file=sys.stderr,
                )
                return 4

            row = _save_attempt(
                sample=sample,
                prediction=prediction,
                exact_prompt=exact_prompt,
                attempt_index=attempt_index,
                failure_reason=failure_reason,
                output_dir=args.output_dir,
                checkpoint=args.checkpoint,
                preflight=preflight,
                dtype=args.dtype,
                device=args.device,
                projection_checkpoint=args.projection_checkpoint,
            )
            results.append(row)
            partial.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    validate_frozen_test_results(results, expected_samples=args.expected_samples)
    if args.experiment == "phase5b":
        validate_one_shot_records(results)
    if backend_call_count != EXPECTED_TEST_ATTEMPTS:
        raise RuntimeError(
            f"expected {EXPECTED_TEST_ATTEMPTS} backend calls, got {backend_call_count}"
        )
    protocol_sha256_end = sha256_file(args.protocol)
    if protocol_sha256_end != preflight["protocol_sha256_start"]:
        _write_json(
            metadata_path,
            {
                **preflight,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "invalid_protocol_changed_during_run",
                "protocol_sha256_end": protocol_sha256_end,
                "backend_call_count": backend_call_count,
            },
        )
        print("error: protocol changed during frozen test", file=sys.stderr)
        return 3

    aggregate = summarize_frozen_test(results)
    verification = _verify_saved_outputs(
        results,
        annotations=annotations,
        manifest_root=args.manifest.parent,
        manifest_path=args.manifest,
        manifest_sha256=preflight["manifest_sha256"],
        protocol_path=args.protocol,
        protocol_sha256=preflight["protocol_sha256_start"],
        prompt_hashes=preflight["prompt_template_hashes"],
    )
    comparison_key: str
    if args.experiment == "phase5b":
        comparison_key = "zero_shot_comparison"
        comparison = compare_with_saved_zero_shot(aggregate, args.zero_shot_summary)
    else:
        comparison_key = "validation_comparison"
        comparison = _validation_comparison(aggregate, args.validation_summary)
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
        "backend_call_count": backend_call_count,
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
        "verification": verification,
        comparison_key: comparison,
    }
    summary = {**common, **aggregate}
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    _write_jsonl(results_path, results)
    _write_json(summary_path, summary)
    _write_json(
        metadata_path,
        {
            **common,
            "status": "complete" if verification["passed"] else "invalid_verification",
        },
    )

    if verification["passed"]:
        args.repository_results_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            "phase5b_finetuned_test"
            if args.experiment == "phase5b"
            else "phase3c_frozen_test"
        )
        _write_jsonl(
            args.repository_results_dir / f"{stem}_metrics.jsonl",
            results,
        )
        _write_json(args.repository_results_dir / f"{stem}_summary.json", summary)
        zero_shot_rows = (
            _read_jsonl(args.zero_shot_metrics)
            if args.experiment == "phase5b"
            else None
        )
        _render_gallery(
            samples,
            results,
            annotations,
            args.gallery_output,
            zero_shot_rows=zero_shot_rows,
        )
        if args.experiment == "phase5b":
            _write_phase5b_report(summary, args.report_output)
        else:
            _write_report(summary, args.report_output)

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
    print(f"backend_calls={backend_call_count}")
    print(f"model_load_attempts={load_attempts}")
    print(f"protocol_sha256={protocol_sha256_end}")
    print(f"verification_passed={verification['passed']}")
    print(f"summary={summary_path}")
    print(f"gallery={args.gallery_output}")
    return 0 if verification["passed"] else 1


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    validate_frozen_test_split(args.split)
    validate_frozen_prompt_variant(args.prompt_variant)
    if args.expected_samples != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Phase 3C requires --expected-samples {EXPECTED_TEST_SAMPLES}"
        )
    if args.device != "cuda:0":
        raise ValueError("Phase 3C requires logical device cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one physical GPU index"
        )
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("TRANSFORMERS_OFFLINE=1 is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "output directory must be absent or empty for the one-shot formal run: "
            f"{args.output_dir}"
        )
    if not args.protocol.is_file():
        raise ValueError(f"protocol file does not exist: {args.protocol}")
    if args.experiment == "phase3c" and not args.validation_summary.is_file():
        raise ValueError(
            f"Phase 3B validation summary does not exist: {args.validation_summary}"
        )
    projection_audit: dict[str, Any] | None = None
    if args.experiment == "phase5b":
        if args.projection_checkpoint is None:
            raise ValueError("Phase 5B requires --projection-checkpoint")
        if not args.zero_shot_summary.is_file() or not args.zero_shot_metrics.is_file():
            raise ValueError("saved Phase 3C summary and metrics are required")
        if not args.protocol_sha256:
            raise ValueError("Phase 5B requires --protocol-sha256")
        projection_audit = validate_selected_projection(args.projection_checkpoint)
    elif args.projection_checkpoint is not None:
        raise ValueError("Phase 3C does not accept a projection checkpoint")
    manifest_sha256 = sha256_file(args.manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest SHA-256 differs from frozen identity")
    protocol_sha256 = sha256_file(args.protocol)
    expected_protocol_sha256 = (
        args.protocol_sha256
        if args.experiment == "phase5b"
        else EXPECTED_PROTOCOL_SHA256
    )
    if protocol_sha256 != expected_protocol_sha256:
        raise ValueError(
            f"{args.experiment} protocol SHA-256 differs from frozen identity"
        )
    prompt_hashes = prompt_template_hashes()
    if prompt_hashes["registry_sha256"] != EXPECTED_PROMPT_REGISTRY_SHA256:
        raise ValueError("Prompt registry SHA-256 differs from frozen identity")
    if prompt_hashes["variants"][TARGET_ONLY_ZH] != EXPECTED_P2_TEMPLATE_SHA256:
        raise ValueError("P2 template SHA-256 differs from frozen identity")
    annotations = validate_jsonl_v1(
        args.manifest, check_files=True, expected_count=320
    )
    selected = validate_balanced_test_annotations(
        annotations,
        split=args.split,
        expected_samples=args.expected_samples,
    )
    sample_list_sha256 = test_sample_id_list_sha256(annotations)
    if sample_list_sha256 != EXPECTED_TEST_SAMPLE_ID_LIST_SHA256:
        raise ValueError("test sample-ID list SHA-256 differs from preregistration")
    plan = build_frozen_test_plan(annotations)
    if len(plan) != EXPECTED_TEST_ATTEMPTS:
        raise ValueError("frozen test plan must contain exactly 64 attempts")
    revision = _detect_local_revision(args.checkpoint)
    if revision != CHECKPOINT_REVISION:
        raise ValueError(
            f"checkpoint revision mismatch: {revision!r} != {CHECKPOINT_REVISION!r}"
        )
    return {
        "experiment": args.experiment,
        "model_name": MODEL_NAME,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_revision": revision,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "test_sample_id_list_sha256": sample_list_sha256,
        "protocol_path": str(args.protocol),
        "protocol_sha256_start": protocol_sha256,
        "prompt_variant": TARGET_ONLY_ZH,
        "prompt_template_hashes": prompt_hashes,
        "expected_samples": len(selected),
        "expected_attempts": EXPECTED_TEST_ATTEMPTS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "code_commit": _git_commit(),
        "code_worktree_dirty_at_start": _git_dirty(),
        "gpu_name": _gpu_name(args.device),
        "projection_checkpoint": projection_audit,
    }


def _save_attempt(
    *,
    sample: Any,
    prediction: Any,
    exact_prompt: str,
    attempt_index: int,
    failure_reason: str | None,
    output_dir: Path,
    checkpoint: Path,
    preflight: dict[str, Any],
    dtype: str,
    device: str,
    projection_checkpoint: Path | None,
) -> dict[str, Any]:
    annotation = sample.annotation
    sample_dir = output_dir / "samples" / annotation["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    original_path = sample_dir / "original.png"
    gt_path = sample_dir / "gt_mask.png"
    predicted_path = sample_dir / "predicted_mask.png"
    overlay_path = sample_dir / "prediction_overlay.png"
    edited_path = sample_dir / "edited.png"
    prompt_path = sample_dir / "prompt.txt"
    text_path = sample_dir / "generated_text.txt"
    result_path = sample_dir / "result.json"
    sample.image.convert("RGB").save(original_path, format="PNG")
    sample.mask.convert("L").save(gt_path, format="PNG")

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
        mask_source = "model_first_prediction_mask"
    else:
        predicted = np.zeros((sample.image.height, sample.image.width), dtype=bool)
        mask_source = "empty_failure_placeholder"
    predicted_image = Image.fromarray(predicted.astype(np.uint8) * 255)
    predicted_image.save(predicted_path, format="PNG")
    mask_overlay(sample.image, predicted_image).save(overlay_path, format="PNG")
    prompt_path.write_text(exact_prompt, encoding="utf-8")
    generated_text = prediction.text_output if prediction is not None else None
    text_path.write_text(generated_text or "", encoding="utf-8")

    counts = binary_pixel_counts(predicted, gt)
    metadata = prediction.metadata if prediction is not None else {}
    semantics = result_metric_semantics(
        prediction_returned=prediction is not None,
        failure_reason=failure_reason,
        error_type=metadata.get("error_type"),
        mask_contract_valid=mask_contract_valid,
        predicted_foreground_pixels=counts["predicted_foreground_pixels"],
        intersection_pixels=counts["intersection_pixels"],
    )
    segmentation_token_present = bool(
        isinstance(generated_text, str) and "[SEG]" in generated_text
    )
    nonempty_disjoint = bool(
        semantics["nonempty_prediction"]
        and counts["gt_foreground_pixels"] > 0
        and counts["intersection_pixels"] == 0
    )

    edit_skipped_empty = not semantics["nonempty_prediction"]
    edit_execution_success: bool | None = None
    edit_error_type: str | None = None
    edit_error_message: str | None = None
    edit_input_mask_sha256: str | None = None
    if not edit_skipped_empty:
        edit_input_mask_sha256 = sha256_file(predicted_path)
        try:
            edited = apply_predicted_mask_edit(
                sample.image,
                predicted,
                edit_action=annotation["edit_action"],
                edit_parameters=annotation["edit_parameters"],
            )
            edited.save(edited_path, format="PNG")
            edit_execution_success = True
        except Exception as exc:
            edit_execution_success = False
            edit_error_type = type(exc).__name__
            edit_error_message = str(exc)

    row = {
        "attempt_index": attempt_index,
        "sample_id": annotation["sample_id"],
        "split": "test",
        "chart_type": annotation["chart_type"],
        "referring_type": annotation["referring_type"],
        "edit_action": annotation["edit_action"],
        "edit_parameters": annotation["edit_parameters"],
        "difficulty": annotation["difficulty"],
        "distractor_count": annotation["distractor_count"],
        "referring_expression": annotation["referring_expression"],
        "prompt_variant": TARGET_ONLY_ZH,
        "exact_prompt": exact_prompt,
        **semantics,
        "segmentation_token_present": segmentation_token_present,
        "generated_text": generated_text,
        "num_masks": prediction.num_masks if prediction is not None else 0,
        "raw_mask_shapes": prediction.raw_mask_shapes if prediction is not None else [],
        "raw_mask_metadata": metadata.get("raw_mask_metadata", []),
        "prediction_masks_python_type": metadata.get("prediction_masks_type"),
        **counts,
        "iou": intersection_over_union(predicted, gt),
        "dice": dice_score(predicted, gt),
        "nonempty_disjoint": nonempty_disjoint,
        "resized": bool(metadata.get("mask_resized_nearest", False)),
        "latency_ms": prediction.inference_time_ms if prediction is not None else None,
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
        "edit_skipped_empty": edit_skipped_empty,
        "edit_execution_success": edit_execution_success,
        "edit_error_type": edit_error_type,
        "edit_error_message": edit_error_message,
        "edit_input_source": (
            "predicted_mask" if edit_input_mask_sha256 is not None else None
        ),
        "edit_input_mask_sha256": edit_input_mask_sha256,
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint),
        "checkpoint_revision": preflight["checkpoint_revision"],
        "projection_checkpoint_path": (
            str(projection_checkpoint) if projection_checkpoint is not None else None
        ),
        "projection_checkpoint_sha256": (
            SELECTED_PROJECTION_SHA256 if projection_checkpoint is not None else None
        ),
        "manifest_sha256": preflight["manifest_sha256"],
        "test_sample_id_list_sha256": preflight["test_sample_id_list_sha256"],
        "protocol_sha256": preflight["protocol_sha256_start"],
        "prompt_template_sha256": preflight["prompt_template_hashes"]["variants"]
        [TARGET_ONLY_ZH],
        "prompt_registry_sha256": preflight["prompt_template_hashes"][
            "registry_sha256"
        ],
        "code_commit": preflight["code_commit"],
        "dtype": dtype,
        "device": device,
        "output_files": {
            "original": str(original_path),
            "gt_mask": str(gt_path),
            "predicted_mask": str(predicted_path),
            "prediction_overlay": str(overlay_path),
            "edited": str(edited_path) if edit_execution_success else None,
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
    manifest_path: Path,
    manifest_sha256: str,
    protocol_path: Path,
    protocol_sha256: str,
    prompt_hashes: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        validate_frozen_test_results(results)
    except ValueError as exc:
        errors.append(str(exc))
    if sha256_file(protocol_path) != protocol_sha256:
        errors.append("protocol hash changed")
    if sha256_file(manifest_path) != manifest_sha256:
        errors.append("manifest hash changed")
    if prompt_template_hashes() != prompt_hashes:
        errors.append("Prompt registry changed")
    annotations_by_id = {row["sample_id"]: row for row in annotations}
    checked_masks = 0
    checked_edits = 0
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
            errors.append(f"{row['sample_id']}: {exc}")
            continue
        if predicted_size != gt_size or predicted.shape != gt.shape:
            errors.append(f"{row['sample_id']}: size mismatch")
            continue
        if not set(np.unique(predicted).tolist()).issubset({0, 255}):
            errors.append(f"{row['sample_id']}: non-binary saved mask")
            continue
        counts = binary_pixel_counts(predicted, gt)
        for key, value in counts.items():
            if int(row[key]) != value:
                errors.append(f"{row['sample_id']}: {key} mismatch")
        if abs(float(row["iou"]) - intersection_over_union(predicted, gt)) > 1e-12:
            errors.append(f"{row['sample_id']}: IoU mismatch")
        if abs(float(row["dice"]) - dice_score(predicted, gt)) > 1e-12:
            errors.append(f"{row['sample_id']}: Dice mismatch")
        expected_nonempty = counts["predicted_foreground_pixels"] > 0
        if bool(row["nonempty_prediction"]) != (
            bool(row["mask_contract_valid"]) and expected_nonempty
        ):
            errors.append(f"{row['sample_id']}: nonempty semantics mismatch")
        if bool(row["empty_prediction"]) == bool(row["nonempty_prediction"]):
            errors.append(f"{row['sample_id']}: empty/nonempty are not complementary")
        if sha256_file(predicted_path) != row["predicted_mask_sha256"]:
            errors.append(f"{row['sample_id']}: mask hash mismatch")
        if row["nonempty_prediction"]:
            checked_edits += 1
            if row["edit_input_source"] != "predicted_mask":
                errors.append(f"{row['sample_id']}: edit input is not predicted mask")
            if row["edit_input_mask_sha256"] != row["predicted_mask_sha256"]:
                errors.append(f"{row['sample_id']}: edit mask hash mismatch")
            if row["edit_execution_success"] is True:
                edited_path = Path(row["output_files"]["edited"])
                if not edited_path.is_file():
                    errors.append(f"{row['sample_id']}: edited output missing")
        elif not row["edit_skipped_empty"]:
            errors.append(f"{row['sample_id']}: empty prediction was not skipped")
        expected_prompt = build_prompt_variant(
            TARGET_ONLY_ZH,
            instruction=annotation["referring_expression"],
            referring_expression=annotation["referring_expression"],
        )
        if row["exact_prompt"] != expected_prompt:
            errors.append(f"{row['sample_id']}: Prompt mismatch")
        checked_masks += 1
    return {
        "passed": not errors,
        "result_record_count": len(results),
        "unique_sample_count": len({row["sample_id"] for row in results}),
        "checked_mask_count": checked_masks,
        "independent_metric_recalculation_count": checked_masks,
        "checked_predicted_mask_edit_count": checked_edits,
        "errors": errors,
    }


def _validation_comparison(
    test_summary: dict[str, Any], validation_summary_path: Path
) -> dict[str, Any]:
    validation = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    val = validation["variant_summaries"][TARGET_ONLY_ZH]
    test = test_summary["metrics"]
    groups: dict[str, Any] = {}
    val_groups = val["groups"]["chart_referring"]
    test_groups = test["groups"]["chart_referring"]
    for name in sorted(test_groups):
        groups[name] = {
            "val_mean_iou": val_groups[name]["mean_iou"],
            "test_mean_iou": test_groups[name]["mean_iou"],
            "test_minus_val_mean_iou": (
                test_groups[name]["mean_iou"] - val_groups[name]["mean_iou"]
            ),
            "val_mean_dice": val_groups[name]["mean_dice"],
            "test_mean_dice": test_groups[name]["mean_dice"],
            "test_minus_val_mean_dice": (
                test_groups[name]["mean_dice"] - val_groups[name]["mean_dice"]
            ),
            "val_empty_rate": val_groups[name]["empty_rate"],
            "test_empty_rate": test_groups[name]["empty_rate"],
            "val_nonempty_disjoint_rate": val_groups[name][
                "nonempty_disjoint_rate"
            ],
            "test_nonempty_disjoint_rate": test_groups[name][
                "nonempty_disjoint_rate"
            ],
        }
    return {
        "prompt_variant": TARGET_ONLY_ZH,
        "used_for_prompt_selection": False,
        "val": {
            "group_macro_iou": val["group_macro_iou"],
            "group_macro_dice": val["group_macro_dice"],
            "empty_prediction_rate": val["empty_prediction_rate"],
            "nonempty_disjoint_rate": val["nonempty_disjoint_rate"],
            "worst_group": val["worst_group"],
            "worst_group_mean_iou": val["worst_group_mean_iou"],
        },
        "test": {
            "group_macro_iou": test["group_macro_iou"],
            "group_macro_dice": test["group_macro_dice"],
            "empty_prediction_rate": test["empty_prediction_rate"],
            "nonempty_disjoint_rate": test["nonempty_disjoint_rate"],
            "worst_group": test["worst_group"],
            "worst_group_mean_iou": test["worst_group_mean_iou"],
        },
        "test_minus_val": {
            "group_macro_iou": test["group_macro_iou"] - val["group_macro_iou"],
            "group_macro_dice": test["group_macro_dice"]
            - val["group_macro_dice"],
            "empty_prediction_rate": test["empty_prediction_rate"]
            - val["empty_prediction_rate"],
            "nonempty_disjoint_rate": test["nonempty_disjoint_rate"]
            - val["nonempty_disjoint_rate"],
        },
        "groups": groups,
    }


def _render_gallery(
    samples: list[Any],
    results: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    output_path: Path,
    *,
    zero_shot_rows: list[dict[str, Any]] | None = None,
) -> None:
    selected_ids = select_frozen_test_gallery_sample_ids(annotations)
    sample_lookup = {sample.annotation["sample_id"]: sample for sample in samples}
    result_lookup = {row["sample_id"]: row for row in results}
    zero_shot_lookup = (
        {row["sample_id"]: row for row in zero_shot_rows}
        if zero_shot_rows is not None
        else {}
    )
    if zero_shot_rows is not None and set(zero_shot_lookup) != set(result_lookup):
        raise ValueError("saved zero-shot metrics sample IDs differ from Phase 5B")
    panel_size = (240, 160)
    header_height = 48
    footer_height = 34
    gutter = 5
    columns = 4
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
        result = result_lookup[sample_id]
        y = gutter + row_index * (row_height + gutter)
        canvas = Image.new("RGB", (columns * panel_size[0], row_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (7, 5),
            f"{sample_id} | {result['chart_type']}/{result['referring_type']} | "
            f"{result['edit_action']} | {result['difficulty']}",
            fill="black",
            font=font,
        )
        if zero_shot_rows is None:
            metric_text = (
                f"P2 IoU={result['iou']:.4f} Dice={result['dice']:.4f} | "
                f"empty={result['empty_prediction']} disjoint={result['nonempty_disjoint']}"
            )
            prediction_label = "P2 prediction overlay"
        else:
            metric_text = (
                f"zero-shot IoU={zero_shot_lookup[sample_id]['iou']:.4f} | "
                f"fine-tuned IoU={result['iou']:.4f} Dice={result['dice']:.4f}"
            )
            prediction_label = "fine-tuned prediction"
        draw.text((7, 22), metric_text, fill="black", font=font)
        with Image.open(result["output_files"]["predicted_mask"]) as source:
            predicted = source.convert("L").copy()
        if result["edit_execution_success"] is True:
            with Image.open(result["output_files"]["edited"]) as source:
                edited = _rgb_for_gallery(source.copy())
            edit_label = "edited result"
        else:
            edited = sample.image.convert("RGB")
            edit_label = (
                "EMPTY / EDIT SKIPPED"
                if result["edit_skipped_empty"]
                else "EDIT FAILED"
            )
        panels = [
            ("original", sample.image.convert("RGB")),
            ("GT overlay", mask_overlay(sample.image, sample.mask)),
            (prediction_label, mask_overlay(sample.image, predicted)),
            (edit_label, edited),
        ]
        for column, (label, image) in enumerate(panels):
            thumbnail = image.copy()
            thumbnail.thumbnail(panel_size, Image.Resampling.LANCZOS)
            panel = Image.new("RGB", panel_size, "white")
            panel.paste(
                thumbnail,
                (
                    (panel_size[0] - thumbnail.width) // 2,
                    (panel_size[1] - thumbnail.height) // 2,
                ),
            )
            x = column * panel_size[0]
            canvas.paste(panel, (x, header_height))
            draw.text(
                (x + 4, header_height + panel_size[1] + 5),
                label,
                fill="black",
                font=font,
            )
        sheet.paste(canvas, (gutter, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def _rgb_for_gallery(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def _write_report(summary: dict[str, Any], output_path: Path) -> None:
    metric = summary["metrics"]
    comparison = summary["validation_comparison"]
    bootstrap = metric["bootstrap_16_group_macro"]
    lines = [
        "# Phase 3C frozen synthetic_v1 test baseline",
        "",
        f"运行日期：{summary['finished_at'][:10]}  ",
        "数据范围：仅 `synthetic_v1` test，64 条  ",
        "固定 Prompt：P2 `target_only_zh`；未比较或重新选择 Prompt",
        "",
        "## 实验身份与完整性",
        "",
        f"- checkpoint revision：`{summary['checkpoint_revision']}`",
        f"- manifest SHA-256：`{summary['manifest_sha256']}`",
        f"- test sample-ID list SHA-256：`{summary['test_sample_id_list_sha256']}`",
        f"- Prompt registry SHA-256：`{summary['prompt_template_hashes']['registry_sha256']}`",
        f"- protocol SHA-256（前/后）：`{summary['protocol_sha256_start']}` / "
        f"`{summary['protocol_sha256_end']}`",
        f"- 模型加载 {summary['model_load_attempts']} 次；backend 调用 "
        f"{summary['backend_call_count']}/64；验证通过：{summary['verification']['passed']}",
        "",
        "## 整体指标",
        "",
        "| metric | result |",
        "|---|---:|",
        f"| execution success | {metric['execution_success_count']}/64 "
        f"({metric['execution_success_rate']:.4%}) |",
        f"| mask contract valid | {metric['mask_contract_valid_count']}/64 "
        f"({metric['mask_contract_valid_rate']:.4%}) |",
        f"| `[SEG]` | {metric['segmentation_token_count']}/64 "
        f"({metric['segmentation_token_rate']:.4%}) |",
        f"| nonempty / empty | {metric['nonempty_prediction_count']} / "
        f"{metric['empty_prediction_count']} |",
        f"| overlap / nonempty-disjoint | {metric['overlapping_prediction_count']} / "
        f"{metric['nonempty_disjoint_count']} |",
        f"| 16-group Macro IoU / Dice | {metric['group_macro_iou']:.6f} / "
        f"{metric['group_macro_dice']:.6f} |",
        f"| sample Macro IoU / Dice | {metric['sample_macro_iou']:.6f} / "
        f"{metric['sample_macro_dice']:.6f} |",
        f"| Micro IoU / Dice | {metric['micro_iou']:.6f} / "
        f"{metric['micro_dice']:.6f} |",
        f"| median IoU / Dice | {metric['median_iou']:.6f} / "
        f"{metric['median_dice']:.6f} |",
        f"| mean / median latency (ms) | {metric['mean_latency_ms']:.3f} / "
        f"{metric['median_latency_ms']:.3f} |",
        "",
        "## 16 个 chart/referring 组合",
        "",
        "| group | n | mean IoU | mean Dice | empty | disjoint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in metric["groups"]["chart_referring"].items():
        lines.append(
            f"| {name} | {value['count']} | {value['mean_iou']:.6f} | "
            f"{value['mean_dice']:.6f} | {value['empty_rate']:.4f} | "
            f"{value['nonempty_disjoint_rate']:.4f} |"
        )
    for dimension, title in (
        ("chart_type", "Chart type 分组"),
        ("referring_type", "Referring type 分组"),
        ("edit_action", "Edit action 分组"),
        ("difficulty", "Difficulty 分组"),
        ("distractor_count", "Distractor count 分组"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| value | n | mean IoU | mean Dice | empty | disjoint |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, value in metric["groups"][dimension].items():
            lines.append(
                f"| {name} | {value['count']} | {value['mean_iou']:.6f} | "
                f"{value['mean_dice']:.6f} | {value['empty_rate']:.4f} | "
                f"{value['nonempty_disjoint_rate']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Bootstrap 95% CI",
            "",
            f"- 16-group Macro IoU：{bootstrap['iou']['estimate']:.6f}，95% CI "
            f"[{bootstrap['iou']['ci95_lower']:.6f}, "
            f"{bootstrap['iou']['ci95_upper']:.6f}]。",
            f"- 16-group Macro Dice：{bootstrap['dice']['estimate']:.6f}，95% CI "
            f"[{bootstrap['dice']['ci95_lower']:.6f}, "
            f"{bootstrap['dice']['ci95_upper']:.6f}]。",
            "- 该区间只描述 test 估计不确定性，未用于修改 Prompt。",
            "",
            "## P2 validation / test 对比",
            "",
            "| metric | val | test | test - val |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("group_macro_iou", "Macro IoU"),
        ("group_macro_dice", "Macro Dice"),
        ("empty_prediction_rate", "empty rate"),
        ("nonempty_disjoint_rate", "disjoint rate"),
    ):
        lines.append(
            f"| {label} | {comparison['val'][key]:.6f} | "
            f"{comparison['test'][key]:.6f} | "
            f"{comparison['test_minus_val'][key]:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Validation worst group：`{comparison['val']['worst_group']}` "
            f"({comparison['val']['worst_group_mean_iou']:.6f})；test worst group："
            f"`{comparison['test']['worst_group']}` "
            f"({comparison['test']['worst_group_mean_iou']:.6f})。",
            "",
            "### 逐组合 validation / test 差异",
            "",
            "| group | val IoU | test IoU | Δ IoU | val Dice | test Dice | Δ Dice |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in comparison["groups"].items():
        lines.append(
            f"| {name} | {value['val_mean_iou']:.6f} | "
            f"{value['test_mean_iou']:.6f} | "
            f"{value['test_minus_val_mean_iou']:+.6f} | "
            f"{value['val_mean_dice']:.6f} | {value['test_mean_dice']:.6f} | "
            f"{value['test_minus_val_mean_dice']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "这些变化只用于描述分布与泛化落差，不触发 Prompt 修改或 validation 重跑。",
            "",
            "## 编辑闭环",
            "",
            f"- 非空预测编辑尝试：{metric['edit_attempt_count']}；成功："
            f"{metric['edit_execution_success_count']}。",
            f"- 空预测/非法预测跳过编辑：{metric['edit_skipped_empty_count']}。",
            "- 每个编辑输入均记录为 `predicted_mask` 并以保存 mask 的 SHA-256 独立核对；"
            "编辑成功不代表分割正确。",
            "",
            "## 失败模式与产物",
            "",
            f"最弱组为 `{metric['worst_group']}`，mean IoU="
            f"{metric['worst_group_mean_iou']:.6f}。worst 10 样本如下；gallery 固定按每组"
            "字典序最小 sample 展示，未按效果挑选。",
            "",
            "| sample | group | IoU | Dice | execution | mask valid | empty | disjoint |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric["worst_10_samples"]:
        lines.append(
            f"| {row['sample_id']} | {row['chart_type']}/{row['referring_type']} | "
            f"{row['iou']:.6f} | {row['dice']:.6f} | "
            f"{row['execution_success']} | {row['mask_contract_valid']} | "
            f"{row['empty_prediction']} | {row['nonempty_disjoint']} |"
        )
    lines.extend(
        [
            "",
            "- 完整临时输出：由运行时 `--output-dir` 指定",
            "- 标量记录：`results/phase3c_frozen_test_metrics.jsonl`",
            "- 聚合结果：`results/phase3c_frozen_test_summary.json`",
            "- gallery：`assets/phase3c_frozen_test_gallery.png`",
            "",
            "测试、compileall、Git 检查和退出后 GPU 状态在最终验收与 `EXPERIMENTS.md` 中记录。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_phase5b_report(summary: dict[str, Any], output_path: Path) -> None:
    metric = summary["metrics"]
    comparison = summary["zero_shot_comparison"]
    bootstrap = metric["bootstrap_16_group_macro"]
    lines = [
        "# Phase 5B fine-tuned test results",
        "",
        "唯一一次正式推理覆盖 `synthetic_v1` test 64 条；使用冻结的 P2 和 "
        "Phase 5A `step960` projection。Zero-shot 数值直接读取 Phase 3C 保存结果，"
        "未重新运行 baseline。",
        "",
        "| metric | fine-tuned | zero-shot | delta |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("group_macro_iou", "16-group Macro IoU"),
        ("group_macro_dice", "16-group Macro Dice"),
        ("micro_iou", "Micro IoU"),
        ("micro_dice", "Micro Dice"),
        ("empty_prediction_rate", "Empty rate"),
        ("overlapping_prediction_rate", "Overlap rate"),
        ("nonempty_disjoint_rate", "Nonempty-disjoint rate"),
    ):
        row = comparison["overall"][name]
        lines.append(
            f"| {label} | {row['fine_tuned']:.6f} | {row['zero_shot']:.6f} | "
            f"{row['delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Sample Macro IoU/Dice: `{metric['sample_macro_iou']:.6f}` / "
            f"`{metric['sample_macro_dice']:.6f}`; median IoU/Dice: "
            f"`{metric['median_iou']:.6f}` / `{metric['median_dice']:.6f}`.",
            "",
            f"Bootstrap 16-group Macro IoU 95% CI: "
            f"`[{bootstrap['iou']['ci95_lower']:.6f}, "
            f"{bootstrap['iou']['ci95_upper']:.6f}]`; Dice 95% CI: "
            f"`[{bootstrap['dice']['ci95_lower']:.6f}, "
            f"{bootstrap['dice']['ci95_upper']:.6f}]`.",
            "",
            "## 16 chart/referring groups",
            "",
            "| group | zero-shot IoU | fine-tuned IoU | delta | fine-tuned Dice |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in comparison["groups"].items():
        lines.append(
            f"| {name} | {row['zero_shot_iou']:.6f} | "
            f"{row['fine_tuned_iou']:.6f} | {row['delta_iou']:+.6f} | "
            f"{row['fine_tuned_dice']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"提升/下降/持平组数：{comparison['improved_group_count']} / "
            f"{comparison['declined_group_count']} / {comparison['tied_group_count']}。",
            f"最大提升：`{comparison['max_gain_group']}` "
            f"({comparison['max_gain_iou']:+.6f})；最大下降："
            f"`{comparison['max_loss_group']}` ({comparison['max_loss_iou']:+.6f})。",
            f"最强组：`{comparison['strongest_group']}` "
            f"({comparison['strongest_group_iou']:.6f})；最弱组："
            f"`{comparison['weakest_group']}` ({comparison['weakest_group_iou']:.6f})。",
            "",
            "## Execution and editing",
            "",
            f"- Execution success: {metric['execution_success_count']}/64; mask contract: "
            f"{metric['mask_contract_valid_count']}/64; `[SEG]`: "
            f"{metric['segmentation_token_count']}/64.",
            f"- Nonempty/empty: {metric['nonempty_prediction_count']} / "
            f"{metric['empty_prediction_count']}; overlap/nonempty-disjoint: "
            f"{metric['overlapping_prediction_count']} / "
            f"{metric['nonempty_disjoint_count']}.",
            f"- Predicted-mask edits attempted/succeeded: {metric['edit_attempt_count']} / "
            f"{metric['edit_execution_success_count']}; empty predictions skipped: "
            f"{metric['edit_skipped_empty_count']}.",
            f"- Mean/median latency: {metric['mean_latency_ms']:.3f} / "
            f"{metric['median_latency_ms']:.3f} ms.",
            "- 编辑只使用预测 mask；编辑成功不代表分割正确。",
            "",
            "完整标量与分组结果见对应 JSON/JSONL；gallery 每组按 sample ID "
            "确定性选择一条，未按效果筛选。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
