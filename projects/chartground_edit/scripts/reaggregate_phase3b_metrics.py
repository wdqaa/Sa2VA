#!/usr/bin/env python3
"""Correct Phase 3B metric semantics from saved records and masks only."""

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

from chartground_edit.inference.prompt_benchmark_v1 import (
    ensure_result_metric_semantics,
    sha256_file,
    summarize_benchmark,
)


EXPECTED_PROTOCOL_SHA256 = (
    "aab7038ec674e53360ef81b7310d8dcfdc4a9eba04107cb11d200da2cd13c9b9"
)
UNCHANGED_METRICS = (
    "group_macro_iou",
    "group_macro_dice",
    "sample_macro_iou",
    "sample_macro_dice",
    "micro_iou",
    "micro_dice",
    "median_iou",
    "median_dice",
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    protocol_before = sha256_file(args.protocol)
    if protocol_before != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(f"unexpected frozen protocol hash: {protocol_before}")
    rows = _read_jsonl(args.metrics)
    old_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    corrected = [_correct_row(row) for row in rows]
    benchmark = summarize_benchmark(corrected)
    _assert_metrics_unchanged(old_summary, benchmark)
    summary = {**old_summary, **benchmark}
    summary["metric_semantics_correction"] = {
        "source": "existing 192 scalar records and saved predicted masks",
        "model_loaded": False,
        "inference_run": False,
        "legacy_inference_success_preserved": True,
        "protocol_sha256": protocol_before,
    }
    _write_jsonl(args.metrics, corrected)
    _write_json(args.summary, summary)
    protocol_after = sha256_file(args.protocol)
    if protocol_after != protocol_before:
        raise RuntimeError("frozen protocol changed during semantics correction")
    print(f"records={len(corrected)}")
    print(f"selected_prompt={benchmark['selection']['selected_prompt']}")
    print(f"protocol_sha256={protocol_after}")
    for variant, values in benchmark["variant_summaries"].items():
        print(
            f"{variant}: execution={values['execution_success_count']}/64 "
            f"contract={values['mask_contract_valid_count']}/64 "
            f"nonempty={values['nonempty_prediction_count']}/64 "
            f"empty={values['empty_prediction_count']}/64"
        )
    return 0


def _correct_row(row: dict[str, Any]) -> dict[str, Any]:
    predicted_path = Path(row["output_files"]["predicted_mask"])
    with Image.open(predicted_path) as source:
        predicted = np.asarray(source)
        image_format = source.format
        image_mode = source.mode
    expected_shape = tuple(int(value) for value in row["gt_mask_shape"])
    contract_valid = bool(
        image_format == "PNG"
        and image_mode == "L"
        and predicted.shape == expected_shape
        and set(np.unique(predicted).tolist()).issubset({0, 255})
    )
    if not contract_valid:
        raise ValueError(
            f"saved mask violates contract: {row['sample_id']}/"
            f"{row['prompt_variant']}"
        )
    predicted_pixels = int(np.count_nonzero(predicted == 255))
    if predicted_pixels != int(row["predicted_foreground_pixels"]):
        raise ValueError(
            f"saved mask foreground mismatch: {row['sample_id']}/"
            f"{row['prompt_variant']}"
        )
    output = ensure_result_metric_semantics(
        {
            **row,
            "mask_contract_valid": True,
            "predicted_foreground_pixels": predicted_pixels,
        }
    )
    output["segmentation_token_present"] = "[SEG]" in str(
        row.get("generated_text") or ""
    )
    return output


def _assert_metrics_unchanged(
    old_summary: dict[str, Any], new_benchmark: dict[str, Any]
) -> None:
    old_variants = old_summary["variant_summaries"]
    new_variants = new_benchmark["variant_summaries"]
    for variant in old_variants:
        for metric in UNCHANGED_METRICS:
            if old_variants[variant][metric] != new_variants[variant][metric]:
                raise RuntimeError(f"metric changed unexpectedly: {variant}/{metric}")
    if old_summary["selection"] != new_benchmark["selection"]:
        raise RuntimeError("Prompt selection or bootstrap changed unexpectedly")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
