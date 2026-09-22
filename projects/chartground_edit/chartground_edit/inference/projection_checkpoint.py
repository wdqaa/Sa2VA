"""Strict projection-only checkpoint loading for the HF inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch


EXPECTED_PROJECTION_KEYS = (
    "text_hidden_fcs.0.weight",
    "text_hidden_fcs.0.bias",
    "text_hidden_fcs.2.weight",
    "text_hidden_fcs.2.bias",
)


def projection_state(model) -> dict[str, torch.Tensor]:
    state = model.text_hidden_fcs.state_dict()
    return {
        f"text_hidden_fcs.{name}": tensor.detach().clone()
        for name, tensor in state.items()
    }


def restore_projection_state(model, state: Mapping[str, torch.Tensor]) -> None:
    _validate_and_copy(model, state)


def load_projection_checkpoint_into_model(
    model,
    checkpoint_path: str | Path,
    *,
    expected_identity: Mapping[str, str],
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError("projection checkpoint must contain a state_dict mapping")
    metadata = checkpoint.get("meta", {}).get("chartground_phase4b")
    if not isinstance(metadata, dict):
        raise ValueError("projection checkpoint is missing chartground_phase4b metadata")
    _validate_identity(metadata, expected_identity)
    _validate_and_copy(model, checkpoint["state_dict"])
    return dict(metadata)


def _validate_and_copy(model, state: Mapping[str, torch.Tensor]) -> None:
    import torch

    if set(state) != set(EXPECTED_PROJECTION_KEYS):
        raise ValueError(
            "projection checkpoint keys mismatch: "
            f"actual={sorted(state)}, expected={sorted(EXPECTED_PROJECTION_KEYS)}"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in EXPECTED_PROJECTION_KEYS:
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"projection value is not a tensor: {name}")
            if name not in parameters:
                raise ValueError(f"model is missing projection parameter: {name}")
            parameter = parameters[name]
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"projection shape mismatch for {name}: "
                    f"checkpoint={tuple(tensor.shape)}, model={tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def _validate_identity(
    metadata: Mapping[str, Any], expected: Mapping[str, str]
) -> None:
    actual = {
        "source_hf_revision": metadata.get("source_hf_revision"),
        "full_pth_sha256": metadata.get("full_pth", {}).get("sha256"),
        "base_repo_id": metadata.get("base_checkpoint", {}).get("repo_id"),
        "base_revision": metadata.get("base_checkpoint", {}).get("revision"),
    }
    required = set(actual)
    if set(expected) != required:
        raise ValueError(
            f"expected_identity must contain exactly {sorted(required)}"
        )
    mismatches = {
        name: {"checkpoint": actual[name], "expected": expected[name]}
        for name in sorted(required)
        if actual[name] != expected[name]
    }
    if mismatches:
        raise ValueError(f"projection checkpoint identity mismatch: {mismatches}")
