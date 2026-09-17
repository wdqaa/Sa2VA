#!/usr/bin/env python3
"""Run fine-tuned ChartGround-Edit inference and editing without ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.editing import EDIT_ACTIONS, edit
from chartground_edit.inference import Sa2VAInternVL3Backend, build_prompt_variant
from chartground_edit.inference.phase5b import (
    projection_identity,
    validate_selected_projection,
)
from chartground_edit.inference.prompt_variants import TARGET_ONLY_ZH
from chartground_edit.visualization.render import mask_overlay


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
        raise argparse.ArgumentTypeError("device must use CUDA syntax such as cuda:0")
    return value


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=existing_directory)
    parser.add_argument("--projection-checkpoint", required=True, type=existing_file)
    parser.add_argument("--image", required=True, type=existing_file)
    parser.add_argument("--referring-expression", required=True)
    parser.add_argument("--action", required=True, choices=sorted(EDIT_ACTIONS))
    parser.add_argument("--color", default="#E63946")
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--fill-mode", choices=("color", "neighbor"), default="color")
    parser.add_argument("--neighbor-radius", type=int, default=5)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", type=cuda_device)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    return parser.parse_args(arguments)


def edit_parameters(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "highlight":
        return {"strength": args.strength}
    if args.action == "recolor":
        return {"color": args.color}
    if args.action == "remove":
        return {
            "fill_mode": args.fill_mode,
            "color": args.color,
            "neighbor_radius": args.neighbor_radius,
        }
    return {}


def run(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] = Sa2VAInternVL3Backend,
    projection_validator: Callable[[Path], dict[str, Any]] = validate_selected_projection,
) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {args.output_dir}")
    projection_audit = projection_validator(args.projection_checkpoint)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    prompt = build_prompt_variant(
        TARGET_ONLY_ZH,
        instruction=args.referring_expression,
        referring_expression=args.referring_expression,
    )
    backend = backend_factory(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        projection_checkpoint=args.projection_checkpoint,
        projection_identity=projection_identity(),
    )
    prediction = backend.predict_prompt(
        image, prompt, instruction=args.referring_expression
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predicted = (
        np.asarray(prediction.mask, dtype=bool)
        if isinstance(prediction.mask, np.ndarray)
        and prediction.mask.dtype == np.bool_
        and prediction.mask.shape == (image.height, image.width)
        else np.zeros((image.height, image.width), dtype=bool)
    )
    predicted_mask = Image.fromarray(predicted.astype(np.uint8) * 255, mode="L")
    mask_path = args.output_dir / "predicted_mask.png"
    overlay_path = args.output_dir / "overlay.png"
    edited_path = args.output_dir / "edited.png"
    result_path = args.output_dir / "result.json"
    predicted_mask.save(mask_path)
    mask_overlay(image, predicted_mask).save(overlay_path)
    edit_skipped_empty = not bool(predicted.any())
    edit_execution_success = False
    edit_error: str | None = None
    if not edit_skipped_empty:
        try:
            edited = edit(image, predicted, args.action, edit_parameters(args))
            edited.save(edited_path)
            edit_execution_success = True
        except Exception as exc:
            edit_error = f"{type(exc).__name__}: {exc}"
    result = {
        "checkpoint": str(args.checkpoint),
        "projection_checkpoint": str(args.projection_checkpoint),
        "projection_checkpoint_sha256": projection_audit[
            "projection_checkpoint_sha256"
        ],
        "image": str(args.image),
        "referring_expression": args.referring_expression,
        "prompt_variant": TARGET_ONLY_ZH,
        "exact_prompt": prompt,
        "action": args.action,
        "edit_parameters": edit_parameters(args),
        "failure_reason": prediction.failure_reason,
        "segmentation_token_present": bool(
            prediction.text_output and "[SEG]" in prediction.text_output
        ),
        "mask_contract_valid": bool(
            isinstance(prediction.mask, np.ndarray)
            and prediction.mask.dtype == np.bool_
            and prediction.mask.shape == (image.height, image.width)
        ),
        "predicted_foreground_pixels": int(predicted.sum()),
        "empty_prediction": edit_skipped_empty,
        "edit_skipped_empty": edit_skipped_empty,
        "edit_execution_success": edit_execution_success,
        "edit_error": edit_error,
        "output_files": {
            "predicted_mask": str(mask_path),
            "overlay": str(overlay_path),
            "edited": str(edited_path) if edit_execution_success else None,
            "result": str(result_path),
        },
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(arguments: list[str] | None = None) -> int:
    try:
        result = run(parse_args(arguments))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["mask_contract_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
