#!/usr/bin/env python3
"""Independently verify saved Phase 3B scalar records against PNG masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets.schema_v1 import validate_jsonl_v1
from chartground_edit.inference.prompt_benchmark_v1 import (
    EXPECTED_ATTEMPTS,
    EXPECTED_VAL_SAMPLES,
    prompt_template_hashes,
    sha256_file,
    validate_balanced_val_annotations,
    validate_result_matrix,
)
from chartground_edit.inference.prompt_variants import build_prompt_variant


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_VAL_SAMPLES)
    parser.add_argument("--expected-attempts", type=int, default=EXPECTED_ATTEMPTS)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    errors: list[str] = []
    try:
        annotations = validate_jsonl_v1(
            args.manifest, check_files=True, expected_count=320
        )
        val = validate_balanced_val_annotations(
            annotations, expected_samples=args.expected_samples
        )
        rows = [
            json.loads(line)
            for line in args.results.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != args.expected_attempts:
            errors.append(
                f"expected {args.expected_attempts} result rows, found {len(rows)}"
            )
        validate_result_matrix(rows, expected_samples=args.expected_samples)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"verification_error: {exc}", file=sys.stderr)
        return 2

    manifest_hash = sha256_file(args.manifest)
    protocol_hash = sha256_file(args.protocol)
    prompt_hashes = prompt_template_hashes()
    annotations_by_id = {row["sample_id"]: row for row in val}
    checked = 0
    for row in rows:
        annotation = annotations_by_id.get(row["sample_id"])
        if annotation is None:
            errors.append(f"unknown val sample_id: {row['sample_id']}")
            continue
        expected_prompt = build_prompt_variant(
            row["prompt_variant"],
            instruction=annotation["full_instruction"],
            referring_expression=annotation["referring_expression"],
        )
        if row.get("exact_prompt") != expected_prompt:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: Prompt mismatch"
            )
        if row.get("manifest_sha256") != manifest_hash:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: manifest hash mismatch"
            )
        if row.get("protocol_sha256") != protocol_hash:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: protocol hash mismatch"
            )
        if row.get("prompt_registry_sha256") != prompt_hashes["registry_sha256"]:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: registry hash mismatch"
            )
        if row.get("prompt_template_sha256") != prompt_hashes["variants"][
            row["prompt_variant"]
        ]:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: template hash mismatch"
            )
        predicted_path = Path(row["output_files"]["predicted_mask"])
        gt_path = args.manifest.parent / annotation["mask_path"]
        try:
            with Image.open(predicted_path) as source:
                predicted = np.asarray(source)
                predicted_format, predicted_mode = source.format, source.mode
            with Image.open(gt_path) as source:
                gt = np.asarray(source)
        except OSError as exc:
            errors.append(f"{row['sample_id']}/{row['prompt_variant']}: {exc}")
            continue
        if predicted_format != "PNG" or predicted_mode != "L":
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: invalid PNG/mode"
            )
            continue
        if predicted.shape != gt.shape:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: shape mismatch"
            )
            continue
        if not set(np.unique(predicted).tolist()).issubset({0, 255}):
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: mask is not binary"
            )
            continue
        predicted_bool = predicted == 255
        gt_bool = gt == 255
        predicted_pixels = int(predicted_bool.sum())
        gt_pixels = int(gt_bool.sum())
        intersection = int(np.logical_and(predicted_bool, gt_bool).sum())
        union = predicted_pixels + gt_pixels - intersection
        iou = intersection / union if union else 1.0
        denominator = predicted_pixels + gt_pixels
        dice = 2.0 * intersection / denominator if denominator else 1.0
        expected_scalars: dict[str, int | float] = {
            "predicted_foreground_pixels": predicted_pixels,
            "gt_foreground_pixels": gt_pixels,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": iou,
            "dice": dice,
        }
        for key, expected in expected_scalars.items():
            actual = row[key]
            if isinstance(expected, float):
                matches = abs(float(actual) - expected) <= 1e-12
            else:
                matches = int(actual) == expected
            if not matches:
                errors.append(
                    f"{row['sample_id']}/{row['prompt_variant']}: {key} mismatch"
                )
        reason = str(row.get("failure_reason") or "")
        expected_execution = not (
            reason in {"model_load_failed", "model_inference_failed"}
            or reason.startswith("backend_exception:")
            or reason.startswith("not_run_after_")
        )
        expected_semantics = {
            "execution_success": expected_execution,
            "mask_contract_valid": True,
            "segmentation_token_present": "[SEG]" in str(row.get("generated_text") or ""),
            "nonempty_prediction": predicted_pixels > 0,
            "empty_prediction": predicted_pixels == 0,
            "overlapping_prediction": intersection > 0,
        }
        for key, expected in expected_semantics.items():
            if row.get(key) is not expected:
                errors.append(
                    f"{row['sample_id']}/{row['prompt_variant']}: {key} mismatch"
                )
        if row.get("inference_success_legacy_deprecated") is not True:
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: legacy marker missing"
            )
        if bool(row.get("inference_success")) != (predicted_pixels > 0):
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: legacy value changed"
            )
        if sha256_file(predicted_path) != row.get("predicted_mask_sha256"):
            errors.append(
                f"{row['sample_id']}/{row['prompt_variant']}: mask hash mismatch"
            )
        checked += 1

    report: dict[str, Any] = {
        "passed": not errors,
        "result_record_count": len(rows),
        "checked_mask_count": checked,
        "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol_hash,
        "prompt_template_hashes": prompt_hashes,
        "errors": errors,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"passed={str(report['passed']).lower()}")
    print(f"result_record_count={len(rows)}")
    print(f"checked_mask_count={checked}")
    print(f"error_count={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
