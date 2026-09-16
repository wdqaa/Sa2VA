#!/usr/bin/env python3
"""Run the three pre-registered Phase 2C Prompts on synthetic-v0 val only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets import ChartGroundDataset
from chartground_edit.inference import (
    PROMPT_VARIANT_ORDER,
    Sa2VAInternVL3Backend,
    SplitEvaluationError,
    binary_pixel_counts,
    build_paired_results,
    build_prompt_variant,
    dice_score,
    empty_prediction,
    extract_synthetic_v0_instruction,
    intersection_over_union,
    is_global_failure,
    organize_gallery_rows,
    parse_prompt_variants,
    select_split_samples,
    summarize_prompt_variants,
    validate_diagnostic_split,
)


MODEL_NAME = "ByteDance/Sa2VA-InternVL3-2B"


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


def cuda_device(value: str) -> str:
    if not value.startswith("cuda:") or not value[5:].isdigit():
        raise argparse.ArgumentTypeError(
            "device must use logical CUDA syntax such as cuda:0"
        )
    return value


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected-count must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("expected-count must be a positive integer")
    return result


def val_only_split(value: str) -> str:
    try:
        return validate_diagnostic_split(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def registered_prompt_variants(value: str) -> tuple[str, ...]:
    try:
        return parse_prompt_variants(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=existing_directory)
    parser.add_argument("--manifest", required=True, type=existing_file)
    parser.add_argument("--split", default="val", type=val_only_split)
    parser.add_argument(
        "--prompt-variants",
        required=True,
        type=registered_prompt_variants,
        help="must contain exactly full_instruction,target_only_en,target_only_zh",
    )
    parser.add_argument("--device", default="cuda:0", type=cuda_device)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=positive_integer)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = ChartGroundDataset(args.manifest)
        samples = select_split_samples(
            dataset, args.split, expected_count=args.expected_count
        )
        decompositions = {
            sample.annotation["sample_id"]: extract_synthetic_v0_instruction(
                sample.annotation
            )
            for sample in samples
        }
    except (OSError, ValueError, SplitEvaluationError) as exc:
        _write_json(
            args.output_dir / "run_error.json",
            {
                "success": False,
                "failure_reason": "input_validation_failed",
                "error": str(exc),
                "split": args.split,
                "expected_count": args.expected_count,
            },
        )
        print(f"error: {exc}", file=sys.stderr)
        return 2

    revision = detect_local_revision(args.checkpoint)
    backend = Sa2VAInternVL3Backend(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    results: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for sample in samples:
        parts = decompositions[sample.annotation["sample_id"]]
        sample_dir = args.output_dir / "samples" / sample.annotation["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample.mask.convert("L").save(sample_dir / "gt_mask.png", format="PNG")
        sample.image.convert("RGB").save(sample_dir / "original.png", format="PNG")
        for variant in args.prompt_variants:
            exact_prompt = build_prompt_variant(
                variant,
                instruction=parts.original_instruction,
                referring_expression=parts.referring_expression,
            )
            prediction = None
            failure_reason = stop_reason
            if stop_reason is None:
                try:
                    prediction = backend.predict_prompt(
                        sample.image,
                        exact_prompt,
                        instruction=parts.original_instruction,
                    )
                    failure_reason = prediction.failure_reason
                    if is_global_failure(prediction):
                        stop_reason = "not_run_after_global_error"
                except Exception as exc:
                    failure_reason = (
                        f"backend_exception:{type(exc).__name__}:{exc}"
                    )
                    stop_reason = "not_run_after_backend_exception"
            row = _save_result(
                sample,
                parts,
                variant,
                exact_prompt,
                prediction,
                failure_reason=failure_reason,
                output_root=args.output_dir,
                checkpoint=args.checkpoint,
                checkpoint_revision=revision,
                dtype=args.dtype,
                device=args.device,
            )
            results.append(row)

    variant_summaries = summarize_prompt_variants(results, args.prompt_variants)
    paired_results = build_paired_results(results, args.prompt_variants)
    summary = {
        "split": args.split,
        "sample_count": len(samples),
        "inference_count": len(results),
        "sample_ids": [sample.annotation["sample_id"] for sample in samples],
        "prompt_variants": list(args.prompt_variants),
        "model_name": MODEL_NAME,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_revision": revision,
        "dtype": args.dtype,
        "device": args.device,
        "backend_instance_count": 1,
        "model_load_attempts": backend.model_load_attempts,
        "model_load_time_ms": backend.model_load_time_ms,
        "peak_gpu_memory_mb": max(
            (
                float(row["peak_gpu_memory_mb"])
                for row in results
                if row["peak_gpu_memory_mb"] is not None
            ),
            default=None,
        ),
        "variant_summaries": variant_summaries,
    }
    _write_jsonl(args.output_dir / "results.jsonl", results)
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "paired_results.json", paired_results)
    _write_summary_csv(args.output_dir / "summary.csv", variant_summaries)
    _write_paired_csv(
        args.output_dir / "paired_results.csv", paired_results, args.prompt_variants
    )
    gallery_path = args.output_dir / "gallery.png"
    _render_gallery(samples, results, gallery_path, args.prompt_variants)
    print(f"split={args.split}")
    print(f"samples={len(samples)}")
    print(f"inferences={len(results)}")
    print(f"model_load_attempts={backend.model_load_attempts}")
    print(f"summary={args.output_dir / 'summary.json'}")
    print(f"gallery={gallery_path}")
    return 0 if all(row["success"] for row in results) else 1


def _save_result(
    sample: Any,
    parts: Any,
    variant: str,
    exact_prompt: str,
    prediction: Any,
    *,
    failure_reason: str | None,
    output_root: Path,
    checkpoint: Path,
    checkpoint_revision: str | None,
    dtype: str,
    device: str,
) -> dict[str, Any]:
    annotation = sample.annotation
    result_dir = output_root / "samples" / annotation["sample_id"] / variant
    result_dir.mkdir(parents=True, exist_ok=True)
    gt_array = np.asarray(sample.mask.convert("L"))
    if prediction is not None and prediction.mask is not None:
        evaluation_mask = np.asarray(prediction.mask, dtype=bool)
        mask_source = "model"
    else:
        evaluation_mask = np.zeros((sample.image.height, sample.image.width), dtype=bool)
        mask_source = "empty_failure_placeholder"
    predicted_image = Image.fromarray(evaluation_mask.astype(np.uint8) * 255)
    predicted_path = result_dir / "predicted_mask.png"
    prompt_path = result_dir / "prompt.txt"
    text_path = result_dir / "text_output.txt"
    result_path = result_dir / "result.json"
    predicted_image.save(predicted_path, format="PNG")
    prompt_path.write_text(exact_prompt, encoding="utf-8")
    text_path.write_text(
        (prediction.text_output or "") if prediction is not None else "",
        encoding="utf-8",
    )
    counts = binary_pixel_counts(evaluation_mask, gt_array)
    is_empty = empty_prediction(evaluation_mask)
    row: dict[str, Any] = {
        "sample_id": annotation["sample_id"],
        "split": annotation["split"],
        "chart_type": annotation["chart_type"],
        "referring_type": annotation["referring_type"],
        "instruction": parts.original_instruction,
        "referring_expression": parts.referring_expression,
        "instruction_extraction_rule": parts.extraction_rule,
        "prompt_variant": variant,
        "exact_prompt": exact_prompt,
        "text_output": prediction.text_output if prediction is not None else None,
        "seg_output": bool(
            prediction is not None
            and isinstance(prediction.text_output, str)
            and "[SEG]" in prediction.text_output
        ),
        "num_masks": prediction.num_masks if prediction is not None else 0,
        "raw_mask_shapes": prediction.raw_mask_shapes if prediction is not None else [],
        **counts,
        "iou": intersection_over_union(evaluation_mask, gt_array),
        "dice": dice_score(evaluation_mask, gt_array),
        "empty_prediction": is_empty,
        "nonempty_disjoint": bool(
            not is_empty
            and counts["gt_foreground_pixels"] > 0
            and counts["intersection_pixels"] == 0
        ),
        "inference_time_ms": (
            prediction.inference_time_ms if prediction is not None else None
        ),
        "peak_gpu_memory_mb": (
            prediction.peak_gpu_memory_mb if prediction is not None else None
        ),
        "success": bool(prediction is not None and prediction.success),
        "failure_reason": failure_reason,
        "edit_action": parts.edit_action,
        "edit_parameters": parts.edit_parameters,
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint),
        "checkpoint_revision": checkpoint_revision,
        "dtype": dtype,
        "device": device,
        "predicted_mask_source": mask_source,
        "output_files": {
            "predicted_mask": str(predicted_path),
            "prompt": str(prompt_path),
            "text_output": str(text_path),
            "result": str(result_path),
        },
    }
    _write_json(result_path, row)
    return row


def _render_gallery(
    samples: list[Any],
    results: list[dict[str, Any]],
    output_path: Path,
    variants: tuple[str, ...],
) -> None:
    gallery_rows = organize_gallery_rows(results, variants)
    sample_lookup = {sample.annotation["sample_id"]: sample for sample in samples}
    panel_size = (240, 160)
    header = 44
    footer = 38
    gutter = 6
    columns = 2 + len(variants)
    row_height = header + panel_size[1] + footer
    sheet = Image.new(
        "RGB",
        (
            columns * panel_size[0] + (columns + 1) * gutter,
            len(gallery_rows) * row_height + (len(gallery_rows) + 1) * gutter,
        ),
        (224, 224, 224),
    )
    font = ImageFont.load_default()
    for row_index, (sample_id, variant_rows) in enumerate(gallery_rows):
        sample = sample_lookup[sample_id]
        y = gutter + row_index * (row_height + gutter)
        row_canvas = Image.new(
            "RGB", (columns * panel_size[0], row_height), "white"
        )
        draw = ImageDraw.Draw(row_canvas)
        draw.text(
            (8, 7),
            f"{sample_id} | {sample.annotation['chart_type']} / "
            f"{sample.annotation['referring_type']}",
            fill="black",
            font=font,
        )
        draw.text(
            (8, 24),
            "P0 full instruction | P1 target-only EN | P2 target-only ZH",
            fill="black",
            font=font,
        )
        panels: list[tuple[str, Image.Image]] = [
            ("original", sample.image.convert("RGB")),
            ("GT", sample.mask.convert("RGB")),
        ]
        short_names = {
            "full_instruction": "P0 full",
            "target_only_en": "P1 target-en",
            "target_only_zh": "P2 target-zh",
        }
        for row in variant_rows:
            with Image.open(row["output_files"]["predicted_mask"]) as handle:
                predicted = handle.convert("RGB").copy()
            status = (
                "EMPTY"
                if row["empty_prediction"]
                else "FAILED"
                if not row["success"]
                else "DISJOINT"
                if row["nonempty_disjoint"]
                else "OVERLAP"
            )
            panels.append(
                (
                    f"{short_names[row['prompt_variant']]}\n"
                    f"IoU={row['iou']:.4f} Dice={row['dice']:.4f} {status}",
                    predicted,
                )
            )
        for column, (label, panel) in enumerate(panels):
            thumbnail = panel.copy()
            thumbnail.thumbnail(panel_size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", panel_size, "white")
            offset = (
                (panel_size[0] - thumbnail.width) // 2,
                (panel_size[1] - thumbnail.height) // 2,
            )
            canvas.paste(thumbnail, offset)
            x = column * panel_size[0]
            row_canvas.paste(canvas, (x, header))
            label_y = header + panel_size[1] + 3
            for line_index, text in enumerate(label.splitlines()):
                draw.text(
                    (x + 4, label_y + line_index * 14),
                    text,
                    fill="black",
                    font=font,
                )
        sheet.paste(row_canvas, (gutter, y))
    sheet.save(output_path, format="PNG")


def _write_summary_csv(
    path: Path, summaries: dict[str, dict[str, Any]]
) -> None:
    metric_names = list(next(iter(summaries.values())).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("prompt_variant", *metric_names))
        for variant in PROMPT_VARIANT_ORDER:
            writer.writerow((variant, *(summaries[variant][key] for key in metric_names)))


def _write_paired_csv(
    path: Path,
    pairs: list[dict[str, Any]],
    variants: tuple[str, ...],
) -> None:
    fields = ["sample_id", "chart_type", "referring_type"]
    for variant in variants:
        fields.extend(
            (
                f"{variant}_iou",
                f"{variant}_dice",
                f"{variant}_foreground_pixels",
                f"{variant}_empty",
                f"{variant}_disjoint",
                f"{variant}_seg_output",
            )
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = {key: pair[key] for key in fields[:3]}
            for variant in variants:
                metric = pair["variants"][variant]
                row.update(
                    {
                        f"{variant}_iou": metric["iou"],
                        f"{variant}_dice": metric["dice"],
                        f"{variant}_foreground_pixels": metric[
                            "predicted_foreground_pixels"
                        ],
                        f"{variant}_empty": metric["empty_prediction"],
                        f"{variant}_disjoint": metric["nonempty_disjoint"],
                        f"{variant}_seg_output": metric["seg_output"],
                    }
                )
            writer.writerow(row)


def detect_local_revision(checkpoint: Path) -> str | None:
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


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
