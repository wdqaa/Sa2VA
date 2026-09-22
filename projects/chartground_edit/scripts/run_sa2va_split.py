#!/usr/bin/env python3
"""Run one frozen Sa2VA configuration over a ChartGround-Edit manifest split."""

from __future__ import annotations

import argparse
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
from chartground_edit.editing import EditingError, edit
from chartground_edit.inference import (
    PROMPT_TEMPLATE,
    PredictionAttempt,
    Sa2VAInternVL3Backend,
    SplitEvaluationError,
    binary_pixel_counts,
    build_prompt,
    dice_score,
    empty_prediction,
    gallery_status_label,
    intersection_over_union,
    run_prediction_sequence,
    select_split_samples,
    summarize_results,
    write_summary_files,
)
from chartground_edit.visualization.render import mask_overlay


MODEL_NAME = "ByteDance/Sa2VA-InternVL3-2B"
DEFAULT_EDIT_COLOR = "#E63946"


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
        integer = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected-count must be a positive integer") from exc
    if integer <= 0:
        raise argparse.ArgumentTypeError("expected-count must be a positive integer")
    return integer


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=existing_directory)
    parser.add_argument("--manifest", required=True, type=existing_file)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", type=cuda_device, default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=positive_integer, default=None)
    parser.add_argument(
        "--continue-on-sample-error",
        action="store_true",
        help="record a sample-local failure and continue without retrying it",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = ChartGroundDataset(args.manifest)
        samples = select_split_samples(
            dataset, args.split, expected_count=args.expected_count
        )
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
    attempts = run_prediction_sequence(
        samples,
        backend,
        continue_on_sample_error=args.continue_on_sample_error,
    )

    rows: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    load_time_written = False
    for attempt in attempts:
        row, comparison = _save_sample_result(
            attempt,
            output_root=args.output_dir,
            checkpoint=args.checkpoint,
            checkpoint_revision=revision,
            dtype=args.dtype,
            device=args.device,
            include_load_time=not load_time_written,
        )
        if row["model_load_time_ms"] is not None:
            load_time_written = True
        rows.append(row)
        comparison_paths.append(comparison)

    summary = summarize_results(
        rows,
        model_load_time_ms=backend.model_load_time_ms,
        model_load_attempts=backend.model_load_attempts,
    )
    summary.update(
        {
            "split": args.split,
            "sample_ids": [row["sample_id"] for row in rows],
            "model_name": MODEL_NAME,
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_revision": revision,
            "dtype": args.dtype,
            "device": args.device,
            "prompt_template": PROMPT_TEMPLATE,
            "execution_order": "manifest",
            "backend_instance_count": 1,
            "sequential_inference": True,
            "continue_on_sample_error": args.continue_on_sample_error,
        }
    )
    write_summary_files(args.output_dir, rows, summary)
    gallery_path = args.output_dir / "gallery.png"
    _combine_comparisons(comparison_paths, gallery_path)
    print(f"split={args.split}")
    print(f"sample_count={len(rows)}")
    print(f"successful_inference_count={summary['successful_inference_count']}")
    print(f"model_load_attempts={summary['model_load_attempts']}")
    print(f"results={args.output_dir / 'results.jsonl'}")
    print(f"summary={args.output_dir / 'summary.json'}")
    print(f"gallery={gallery_path}")
    return 0 if summary["successful_inference_count"] == len(rows) else 1


def _save_sample_result(
    attempt: PredictionAttempt,
    *,
    output_root: Path,
    checkpoint: Path,
    checkpoint_revision: str | None,
    dtype: str,
    device: str,
    include_load_time: bool,
) -> tuple[dict[str, Any], Path]:
    sample = attempt.sample
    annotation = sample.annotation
    sample_id = annotation["sample_id"]
    sample_dir = output_root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(annotation["instruction"])
    prediction = attempt.prediction
    image = sample.image.convert("RGB")
    gt_image = sample.mask.convert("L")
    gt_array = np.asarray(gt_image)
    if prediction is not None and prediction.mask is not None:
        evaluation_mask = np.asarray(prediction.mask, dtype=bool)
        predicted_mask_shape: list[int] | None = [
            int(value) for value in evaluation_mask.shape
        ]
        predicted_mask_source = "model"
    else:
        evaluation_mask = np.zeros((image.height, image.width), dtype=bool)
        predicted_mask_shape = None
        predicted_mask_source = "empty_failure_placeholder"
    predicted_image = Image.fromarray(evaluation_mask.astype(np.uint8) * 255)

    prompt_path = sample_dir / "prompt.txt"
    text_path = sample_dir / "text_output.txt"
    original_path = sample_dir / "original.png"
    gt_path = sample_dir / "gt_mask.png"
    predicted_path = sample_dir / "predicted_mask.png"
    gt_overlay_path = sample_dir / "gt_overlay.png"
    prediction_overlay_path = sample_dir / "prediction_overlay.png"
    raw_metadata_path = sample_dir / "raw_mask_metadata.json"
    comparison_path = sample_dir / "comparison.png"
    result_path = sample_dir / "result.json"

    prompt_path.write_text(prompt, encoding="utf-8")
    text_path.write_text(
        prediction.text_output or "" if prediction is not None else "",
        encoding="utf-8",
    )
    image.save(original_path, format="PNG")
    gt_image.save(gt_path, format="PNG")
    predicted_image.save(predicted_path, format="PNG")
    gt_overlay = mask_overlay(image, gt_image)
    prediction_overlay = mask_overlay(image, predicted_image)
    gt_overlay.save(gt_overlay_path, format="PNG")
    prediction_overlay.save(prediction_overlay_path, format="PNG")

    raw_metadata = _raw_mask_metadata(prediction, attempt.not_run_reason)
    _write_json(raw_metadata_path, raw_metadata)
    counts = binary_pixel_counts(evaluation_mask, gt_array)
    is_empty = empty_prediction(evaluation_mask)
    failure_reason = (
        prediction.failure_reason if prediction is not None else attempt.not_run_reason
    )
    success = bool(prediction is not None and prediction.success)
    nonempty_disjoint = bool(
        not is_empty
        and counts["gt_foreground_pixels"] > 0
        and counts["intersection_pixels"] == 0
    )

    edited_image: Image.Image | None = None
    edit_error: str | None = None
    skipped_edit_reason: str | None = None
    if success and not is_empty and prediction is not None and prediction.mask is not None:
        try:
            edited_image = edit(
                image,
                prediction.mask,
                annotation["edit_action"],
                _edit_parameters(annotation["edit_action"]),
            )
            edited_image.save(sample_dir / "edited_result.png", format="PNG")
        except (EditingError, OSError) as exc:
            edit_error = str(exc)
    else:
        skipped_edit_reason = failure_reason or "empty_prediction"

    output_files: dict[str, str | None] = {
        "result": str(result_path),
        "prompt": str(prompt_path),
        "text_output": str(text_path),
        "original": str(original_path),
        "gt_mask": str(gt_path),
        "predicted_mask": str(predicted_path),
        "gt_overlay": str(gt_overlay_path),
        "prediction_overlay": str(prediction_overlay_path),
        "raw_mask_metadata": str(raw_metadata_path),
        "comparison": str(comparison_path),
        "edited_result": str(sample_dir / "edited_result.png")
        if edited_image is not None
        else None,
    }
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "split": annotation["split"],
        "chart_type": annotation["chart_type"],
        "referring_type": annotation["referring_type"],
        "instruction": annotation["instruction"],
        "prompt": prompt,
        "target_type": annotation["target_type"],
        "edit_action": annotation["edit_action"],
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint),
        "checkpoint_revision": checkpoint_revision,
        "dtype": dtype,
        "device": device,
        "text_output": prediction.text_output if prediction is not None else None,
        "success": success,
        "failure_reason": failure_reason,
        "num_masks": prediction.num_masks if prediction is not None else 0,
        "raw_mask_shapes": prediction.raw_mask_shapes if prediction is not None else [],
        "predicted_mask_shape": predicted_mask_shape,
        "gt_mask_shape": [gt_image.height, gt_image.width],
        **counts,
        "iou": intersection_over_union(evaluation_mask, gt_array),
        "dice": dice_score(evaluation_mask, gt_array),
        "empty_prediction": is_empty,
        "nonempty_disjoint": nonempty_disjoint,
        "model_load_time_ms": (
            prediction.model_load_time_ms
            if prediction is not None and include_load_time
            else None
        ),
        "inference_time_ms": (
            prediction.inference_time_ms if prediction is not None else None
        ),
        "peak_gpu_memory_mb": (
            prediction.peak_gpu_memory_mb if prediction is not None else None
        ),
        "edit_success": edited_image is not None,
        "skipped_edit_reason": skipped_edit_reason,
        "edit_error": edit_error,
        "predicted_mask_source": predicted_mask_source,
        "backend_metadata": {
            key: value
            for key, value in (prediction.metadata.items() if prediction is not None else [])
            if key != "raw_mask_metadata"
        },
        "output_files": output_files,
    }
    _save_comparison(
        image,
        gt_image,
        predicted_image,
        gt_overlay,
        prediction_overlay,
        edited_image,
        comparison_path,
        row,
    )
    _write_json(result_path, row)
    return row, comparison_path


def _raw_mask_metadata(prediction: Any, not_run_reason: str | None) -> dict[str, Any]:
    if prediction is None:
        return {
            "num_masks": 0,
            "raw_mask_shapes": [],
            "raw_masks": [],
            "not_run_reason": not_run_reason,
        }
    return {
        "num_masks": prediction.num_masks,
        "raw_mask_shapes": prediction.raw_mask_shapes,
        "raw_masks": prediction.metadata.get("raw_mask_metadata", []),
        "prediction_type": prediction.metadata.get("prediction_type"),
        "prediction_masks_type": prediction.metadata.get("prediction_masks_type"),
        "prediction_masks_present": prediction.metadata.get("prediction_masks_present"),
        "selected_mask_index": prediction.metadata.get("selected_mask_index"),
        "binarization": prediction.metadata.get("binarization"),
        "mask_resized_nearest": prediction.metadata.get("mask_resized_nearest"),
    }


def _edit_parameters(action: str) -> dict[str, object]:
    if action == "recolor":
        return {"color": DEFAULT_EDIT_COLOR}
    if action == "remove":
        return {"fill_mode": "color", "color": DEFAULT_EDIT_COLOR}
    return {}


def _save_comparison(
    image: Image.Image,
    ground_truth: Image.Image,
    predicted: Image.Image,
    gt_overlay: Image.Image,
    prediction_overlay: Image.Image,
    edited: Image.Image | None,
    output_path: Path,
    result: dict[str, Any],
) -> None:
    panel_size = (240, 160)
    labels = (
        "original",
        "GT mask",
        "predicted mask",
        "GT overlay",
        "prediction overlay",
        "predicted-mask edit",
    )
    panels = [
        image.convert("RGB"),
        ground_truth.convert("RGB"),
        predicted.convert("RGB"),
        gt_overlay.convert("RGB"),
        prediction_overlay.convert("RGB"),
        edited.convert("RGB") if edited is not None else None,
    ]
    header = 66
    footer = 22
    width = panel_size[0] * len(panels)
    sheet = Image.new("RGB", (width, header + panel_size[1] + footer), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    status = gallery_status_label(result)
    draw.text(
        (8, 7),
        f"{result['sample_id']} | {result['chart_type']} / {result['referring_type']} | {status}",
        fill="black",
        font=font,
    )
    draw.text(
        (8, 25),
        f"IoU={result['iou']:.6f} Dice={result['dice']:.6f} | "
        f"masks={result['num_masks']} | edit={result['edit_success']}",
        fill="black",
        font=font,
    )
    text_output = (result.get("text_output") or "<none>").replace("\n", " ")
    draw.text((8, 43), f"text: {text_output[:180]}", fill="black", font=font)
    for index, (label, panel) in enumerate(zip(labels, panels)):
        canvas = Image.new("RGB", panel_size, "white")
        if panel is not None:
            thumbnail = panel.copy()
            thumbnail.thumbnail(panel_size, Image.Resampling.LANCZOS)
            offset = (
                (panel_size[0] - thumbnail.width) // 2,
                (panel_size[1] - thumbnail.height) // 2,
            )
            canvas.paste(thumbnail, offset)
        else:
            placeholder = ImageDraw.Draw(canvas)
            placeholder.text((68, 72), "NOT GENERATED", fill=(170, 0, 0), font=font)
        x = index * panel_size[0]
        sheet.paste(canvas, (x, header))
        draw.text((x + 5, header + panel_size[1] + 4), label, fill="black", font=font)
    sheet.save(output_path, format="PNG")


def _combine_comparisons(paths: list[Path], output_path: Path) -> None:
    if not paths:
        raise ValueError("cannot create a gallery without comparisons")
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as handle:
            images.append(handle.convert("RGB").copy())
    gutter = 8
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gutter * (len(images) + 1)
    gallery = Image.new("RGB", (width + 2 * gutter, height), (224, 224, 224))
    y = gutter
    for image in images:
        gallery.paste(image, (gutter, y))
        y += image.height + gutter
    gallery.save(output_path, format="PNG")


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
