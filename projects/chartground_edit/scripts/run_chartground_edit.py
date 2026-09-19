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
    SA2VA_REVISION,
    projection_identity,
    validate_selected_projection,
)
from chartground_edit.inference.prompt_variants import TARGET_ONLY_ZH
from chartground_edit.visualization.render import mask_overlay


FINAL_ADAPTER_SHA256 = (
    "c47ce4e38a9b1679c766d6360d66b6a3b69286d36b9d7cde50c70991ae475b97"
)
V2_MANIFEST_SHA256 = (
    "1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be"
)
PROMPT_REGISTRY_SHA256 = (
    "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0"
)
P2_SHA256 = "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806"


def adapter_identity() -> dict[str, str]:
    return {
        **projection_identity(),
        "manifest_sha256": V2_MANIFEST_SHA256,
    }


def validate_sa2va_revision(checkpoint: Path) -> str:
    """Require unambiguous Hugging Face download metadata for the frozen base."""
    metadata_dir = checkpoint / ".cache/huggingface/download"
    revisions = set()
    if metadata_dir.is_dir():
        for path in metadata_dir.glob("*.metadata"):
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines:
                revisions.add(lines[0])
    if revisions != {SA2VA_REVISION}:
        raise ValueError(
            "Sa2VA base revision mismatch or missing Hugging Face metadata: "
            f"expected {SA2VA_REVISION}, found {sorted(revisions)}"
        )
    return SA2VA_REVISION


def validate_selected_adapter(path: Path) -> dict[str, Any]:
    """Audit the one released 68-tensor Strategy B checkpoint before loading."""
    from chartground_edit.inference.phase5b import sha256_file

    actual_sha256 = sha256_file(path)
    if actual_sha256 != FINAL_ADAPTER_SHA256:
        raise ValueError(
            "ChartGround-Edit only accepts the frozen v2 Strategy B step4800 "
            f"adapter; expected SHA-256 {FINAL_ADAPTER_SHA256}, got {actual_sha256}"
        )
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"meta", "state_dict"}:
        raise ValueError("adapter checkpoint must contain only meta and state_dict")
    metadata = payload.get("meta", {}).get("chartground_phase7b")
    state = payload.get("state_dict")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("invalid Strategy B adapter payload")
    projection = {
        "text_hidden_fcs.0.weight",
        "text_hidden_fcs.0.bias",
        "text_hidden_fcs.2.weight",
        "text_hidden_fcs.2.bias",
    }
    lora = {
        f"language_model.base_model.model.model.layers.{layer}.self_attn."
        f"{name}.lora_{branch}.default.weight"
        for layer in range(20, 28)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        for branch in ("A", "B")
    }
    if set(state) != projection | lora:
        raise ValueError("adapter must contain exactly 4 projection and 64 LoRA tensors")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise TypeError("adapter state_dict contains a non-tensor value")
    actual_identity = {
        "source_hf_revision": metadata.get("source_hf_revision"),
        "full_pth_sha256": metadata.get("full_pth", {}).get("sha256"),
        "base_repo_id": metadata.get("base_checkpoint", {}).get("repo_id"),
        "base_revision": metadata.get("base_checkpoint", {}).get("revision"),
        "manifest_sha256": metadata.get("manifest_sha256"),
    }
    if actual_identity != adapter_identity():
        raise ValueError("adapter base/full-PTH/manifest identity mismatch")
    expected_lora = {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "bias": "none",
        "layers": list(range(20, 28)),
        "target_modules": [
            f"model.layers.{layer}.self_attn.{name}"
            for layer in range(20, 28)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ],
        "modules_to_save": None,
    }
    if metadata.get("lora") != expected_lora:
        raise ValueError("adapter LoRA configuration mismatch")
    expected_metadata = {
        "optimizer_step": 4800,
        "trainable_parameter_count": 3_999_488,
        "prompt_variant": TARGET_ONLY_ZH,
        "prompt_registry_sha256": PROMPT_REGISTRY_SHA256,
        "prompt_template_sha256": P2_SHA256,
        "checkpoint_state": "text_hidden_fcs_plus_llm_lora",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"adapter metadata mismatch for {key}")
    return {
        "adapter_checkpoint_sha256": actual_sha256,
        "adapter_tensor_count": len(state),
        "projection_tensor_count": len(projection),
        "lora_tensor_count": len(lora),
        "trainable_parameter_count": metadata["trainable_parameter_count"],
    }


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
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--projection-checkpoint", type=existing_file)
    checkpoint_group.add_argument("--adapter-checkpoint", type=existing_file)
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
    adapter_validator: Callable[[Path], dict[str, Any]] = validate_selected_adapter,
) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {args.output_dir}")
    projection_checkpoint = getattr(args, "projection_checkpoint", None)
    adapter_checkpoint = getattr(args, "adapter_checkpoint", None)
    if (projection_checkpoint is None) == (adapter_checkpoint is None):
        raise ValueError(
            "exactly one of projection_checkpoint or adapter_checkpoint is required"
        )
    if adapter_checkpoint is not None:
        checkpoint_audit = adapter_validator(adapter_checkpoint)
        validate_sa2va_revision(args.checkpoint)
        backend_checkpoint_kwargs = {
            "adapter_checkpoint": adapter_checkpoint,
            "adapter_identity": adapter_identity(),
        }
        checkpoint_kind = "projection_plus_lora"
    else:
        checkpoint_audit = projection_validator(projection_checkpoint)
        backend_checkpoint_kwargs = {
            "projection_checkpoint": projection_checkpoint,
            "projection_identity": projection_identity(),
        }
        checkpoint_kind = "projection_only"
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
        **backend_checkpoint_kwargs,
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
        "checkpoint_kind": checkpoint_kind,
        "projection_checkpoint": (
            str(projection_checkpoint) if projection_checkpoint is not None else None
        ),
        "projection_checkpoint_sha256": checkpoint_audit.get(
            "projection_checkpoint_sha256"
        ),
        "adapter_checkpoint": (
            str(adapter_checkpoint) if adapter_checkpoint is not None else None
        ),
        "adapter_checkpoint_sha256": checkpoint_audit.get(
            "adapter_checkpoint_sha256"
        ),
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
