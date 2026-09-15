#!/usr/bin/env python3
"""Run one ChartGround-Edit sample through the local Sa2VA InternVL3 backend."""

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
    Sa2VAInternVL3Backend,
    build_prompt,
    dice_score,
    empty_prediction,
    intersection_over_union,
)
from chartground_edit.visualization.render import mask_overlay


MODEL_NAME = "ByteDance/Sa2VA-InternVL3-2B"
FIXED_SMOKE_SAMPLE_ID = "cge_bar_category_01"
FALLBACK_CHART_TYPE = "bar"
FALLBACK_REFERRING_TYPE = "category"
EDIT_ACTIONS = ("highlight", "recolor", "extract", "remove")


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


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=existing_directory)
    parser.add_argument("--manifest", required=True, type=existing_file)
    parser.add_argument("--sample-id", default=FIXED_SMOKE_SAMPLE_ID)
    parser.add_argument("--device", type=cuda_device, default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--edit-action", choices=EDIT_ACTIONS, default=None)
    parser.add_argument("--edit-color", default="#E63946")
    parser.add_argument(
        "--skip-model-run",
        action="store_true",
        help="validate inputs and prompt construction without loading Sa2VA",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"

    try:
        dataset = ChartGroundDataset(args.manifest)
        sample, used_fallback = select_sample(dataset, args.sample_id)
    except Exception as exc:
        _write_json(
            result_path,
            {
                "sample_id": args.sample_id,
                "model_name": MODEL_NAME,
                "checkpoint_path": str(args.checkpoint),
                "success": False,
                "failure_reason": "input_validation_failed",
                "error": str(exc),
                "output_files": {"result": str(result_path)},
            },
        )
        print(f"error: {exc}", file=sys.stderr)
        return 2

    annotation = sample.annotation
    instruction = annotation["instruction"]
    prompt = build_prompt(instruction)
    revision = detect_local_revision(args.checkpoint)
    action = args.edit_action or annotation["edit_action"]
    action_overridden = args.edit_action is not None and args.edit_action != annotation["edit_action"]

    prompt_path = args.output_dir / "prompt.txt"
    text_output_path = args.output_dir / "text_output.txt"
    gt_path = args.output_dir / "gt_mask.png"
    raw_metadata_path = args.output_dir / "raw_mask_metadata.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    sample.mask.convert("L").save(gt_path, format="PNG")

    output_files: dict[str, str | None] = {
        "result": str(result_path),
        "prompt": str(prompt_path),
        "text_output": str(text_output_path),
        "raw_mask_metadata": str(raw_metadata_path),
        "gt_mask": str(gt_path),
        "predicted_mask": None,
        "prediction_overlay": None,
        "comparison": None,
        "edited_result": None,
    }
    base_result: dict[str, Any] = {
        "sample_id": annotation["sample_id"],
        "requested_sample_id": args.sample_id,
        "sample_fallback_used": used_fallback,
        "chart_type": annotation["chart_type"],
        "referring_type": annotation["referring_type"],
        "instruction": instruction,
        "prompt": prompt,
        "model_name": MODEL_NAME,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_revision": revision,
        "dtype": args.dtype,
        "device": args.device,
        "text_output": None,
        "success": False,
        "failure_reason": None,
        "num_masks": 0,
        "raw_mask_shapes": [],
        "predicted_mask_shape": None,
        "gt_mask_shape": [sample.mask.height, sample.mask.width],
        "iou": None,
        "dice": None,
        "empty_prediction": True,
        "inference_success": False,
        "model_load_time_ms": None,
        "inference_time_ms": None,
        "peak_gpu_memory_mb": None,
        "edit_action": action,
        "annotation_edit_action": annotation["edit_action"],
        "edit_action_overridden": action_overridden,
        "edit_succeeded": False,
        "skipped_edit_reason": None,
        "output_files": output_files,
    }

    if args.skip_model_run:
        text_output_path.write_text("", encoding="utf-8")
        _write_json(
            raw_metadata_path,
            {"model_run_skipped": True, "raw_masks": []},
        )
        base_result["failure_reason"] = "model_run_skipped"
        base_result["skipped_edit_reason"] = "model_run_skipped"
        base_result["run_status"] = "skip_model_run_validated"
        _write_json(result_path, base_result)
        print(f"sample_id={annotation['sample_id']}")
        print("model_run=skipped")
        print(f"prompt={prompt_path}")
        print(f"result={result_path}")
        return 0

    backend = Sa2VAInternVL3Backend(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    prediction = backend.predict_mask(sample.image, instruction)
    text_output_path.write_text(prediction.text_output or "", encoding="utf-8")
    raw_metadata = {
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
    _write_json(raw_metadata_path, raw_metadata)

    gt_array = np.asarray(sample.mask.convert("L"))
    evaluation_mask = (
        prediction.mask
        if prediction.mask is not None
        else np.zeros((sample.image.height, sample.image.width), dtype=bool)
    )
    predicted_path = args.output_dir / "predicted_mask.png"
    predicted_image = Image.fromarray(evaluation_mask.astype(np.uint8) * 255)
    predicted_image.save(predicted_path, format="PNG")
    output_files["predicted_mask"] = str(predicted_path)

    overlay_path = args.output_dir / "prediction_overlay.png"
    overlay = mask_overlay(sample.image, predicted_image)
    overlay.save(overlay_path, format="PNG")
    output_files["prediction_overlay"] = str(overlay_path)

    edited_image: Image.Image | None = None
    skipped_edit_reason: str | None = None
    edit_error: str | None = None
    if prediction.success and not empty_prediction(prediction.mask):
        try:
            edited_image = edit(
                sample.image,
                prediction.mask,
                action,
                _edit_parameters(action, args.edit_color),
            )
            edited_path = args.output_dir / "edited_result.png"
            edited_image.save(edited_path, format="PNG")
            output_files["edited_result"] = str(edited_path)
        except (EditingError, OSError) as exc:
            edit_error = str(exc)
    else:
        skipped_edit_reason = prediction.failure_reason or "empty_prediction"

    comparison_path = args.output_dir / "comparison.png"
    _save_comparison(
        sample.image,
        sample.mask,
        predicted_image,
        overlay,
        edited_image,
        comparison_path,
        status=prediction.failure_reason or "success",
    )
    output_files["comparison"] = str(comparison_path)

    base_result.update(
        {
            "text_output": prediction.text_output,
            "success": prediction.success,
            "failure_reason": prediction.failure_reason,
            "num_masks": prediction.num_masks,
            "raw_mask_shapes": prediction.raw_mask_shapes,
            "predicted_mask_shape": [int(item) for item in evaluation_mask.shape],
            "iou": intersection_over_union(evaluation_mask, gt_array),
            "dice": dice_score(evaluation_mask, gt_array),
            "empty_prediction": empty_prediction(evaluation_mask),
            "inference_success": prediction.success,
            "model_load_time_ms": prediction.model_load_time_ms,
            "inference_time_ms": prediction.inference_time_ms,
            "peak_gpu_memory_mb": prediction.peak_gpu_memory_mb,
            "edit_succeeded": edited_image is not None,
            "skipped_edit_reason": skipped_edit_reason,
            "edit_error": edit_error,
            "predicted_mask_source": (
                "model" if prediction.mask is not None else "empty_failure_placeholder"
            ),
            "backend_metadata": {
                key: value
                for key, value in prediction.metadata.items()
                if key != "raw_mask_metadata"
            },
        }
    )
    _write_json(result_path, base_result)
    print(f"sample_id={annotation['sample_id']}")
    print(f"success={prediction.success}")
    print(f"failure_reason={prediction.failure_reason}")
    print(f"num_masks={prediction.num_masks}")
    print(f"result={result_path}")
    return 0 if prediction.success else 1


def select_sample(dataset: ChartGroundDataset, requested_id: str):
    for sample in dataset:
        if sample.annotation["sample_id"] == requested_id:
            return sample, False
    if requested_id == FIXED_SMOKE_SAMPLE_ID:
        for sample in dataset:
            if (
                sample.annotation["chart_type"] == FALLBACK_CHART_TYPE
                and sample.annotation["referring_type"] == FALLBACK_REFERRING_TYPE
            ):
                return sample, True
    raise ValueError(f"sample_id not found in manifest: {requested_id}")


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


def _edit_parameters(action: str, color: str) -> dict[str, object]:
    if action == "recolor":
        return {"color": color}
    if action == "remove":
        return {"fill_mode": "color", "color": color}
    return {}


def _save_comparison(
    image: Image.Image,
    ground_truth: Image.Image,
    predicted: Image.Image,
    overlay: Image.Image,
    edited: Image.Image | None,
    output_path: Path,
    *,
    status: str,
) -> None:
    panel_size = (320, 220)
    labels = ("original", "GT mask", "predicted mask", "prediction overlay", "edited")
    panels = [
        image.convert("RGB"),
        ground_truth.convert("RGB"),
        predicted.convert("RGB"),
        overlay.convert("RGB"),
        edited.convert("RGB") if edited is not None else Image.new("RGB", image.size, "white"),
    ]
    header = 34
    footer = 24
    sheet = Image.new("RGB", (panel_size[0] * len(panels), header + panel_size[1] + footer), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 10), f"Sa2VA-InternVL3-2B | {status}", fill="black", font=font)
    for index, (label, panel) in enumerate(zip(labels, panels)):
        panel.thumbnail(panel_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", panel_size, "white")
        offset = ((panel_size[0] - panel.width) // 2, (panel_size[1] - panel.height) // 2)
        canvas.paste(panel, offset)
        x = index * panel_size[0]
        sheet.paste(canvas, (x, header))
        draw.text((x + 6, header + panel_size[1] + 5), label, fill="black", font=font)
    sheet.save(output_path, format="PNG")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
