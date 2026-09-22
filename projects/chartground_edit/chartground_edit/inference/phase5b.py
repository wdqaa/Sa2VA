"""Frozen identities and fail-closed checks for the one-shot Phase 5B test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .projection_checkpoint import EXPECTED_PROJECTION_KEYS


SELECTED_CHECKPOINT = "step960"
SELECTED_PROJECTION_SHA256 = (
    "64c0d109d2985893ba1f2ba4c4fe7acc4265dc758e5d56ecb6ac8e2aa791f41e"
)
EXPECTED_TRAINABLE_PARAMETER_COUNT = 2_754_304
BASE_REPO_ID = "OpenGVLab/InternVL3-2B"
BASE_REVISION = "899155015275a9b7338c7f4677e19c784e0e5a21"
SA2VA_REVISION = "15837dcaecc304714a1f0f069e74f47e47521c7f"
FULL_PTH_SHA256 = (
    "5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_identity() -> dict[str, str]:
    return {
        "source_hf_revision": SA2VA_REVISION,
        "full_pth_sha256": FULL_PTH_SHA256,
        "base_repo_id": BASE_REPO_ID,
        "base_revision": BASE_REVISION,
    }


def validate_selected_projection(path: str | Path) -> dict[str, Any]:
    """Require the exact Phase 5A step960 projection and its four tensors."""
    return _validate_projection(path, expected_sha256=SELECTED_PROJECTION_SHA256)


def _validate_projection(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(path)
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Phase 5B only accepts the frozen step960 projection: "
            f"sha256={actual_sha256}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"meta", "state_dict"}:
        raise ValueError("projection checkpoint must contain only meta and state_dict")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or set(state) != set(EXPECTED_PROJECTION_KEYS):
        raise ValueError("projection checkpoint must contain exactly four expected tensors")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("projection checkpoint state_dict contains a non-tensor value")
    metadata = payload.get("meta", {}).get("chartground_phase4b")
    if not isinstance(metadata, dict):
        raise ValueError("projection checkpoint is missing chartground_phase4b metadata")
    expected = {
        "optimizer_step": 960,
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
        "source_hf_revision": SA2VA_REVISION,
        "prompt_variant": "target_only_zh",
        "prompt_template_sha256": (
            "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806"
        ),
        "prompt_registry_sha256": (
            "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0"
        ),
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    nested = {
        "base_repo_id": metadata.get("base_checkpoint", {}).get("repo_id"),
        "base_revision": metadata.get("base_checkpoint", {}).get("revision"),
        "full_pth_sha256": metadata.get("full_pth", {}).get("sha256"),
    }
    for key, value in {
        "base_repo_id": BASE_REPO_ID,
        "base_revision": BASE_REVISION,
        "full_pth_sha256": FULL_PTH_SHA256,
    }.items():
        if nested[key] != value:
            mismatches[key] = {"actual": nested[key], "expected": value}
    if mismatches:
        raise ValueError(f"step960 projection metadata mismatch: {mismatches}")
    if sum(tensor.numel() for tensor in state.values()) != EXPECTED_TRAINABLE_PARAMETER_COUNT:
        raise ValueError("projection checkpoint tensor count does not equal 2,754,304")
    return {
        "selected_checkpoint": SELECTED_CHECKPOINT,
        "projection_checkpoint_path": str(checkpoint_path),
        "projection_checkpoint_sha256": actual_sha256,
        "projection_tensor_count": len(state),
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
        "metadata": dict(metadata),
    }


def compare_with_saved_zero_shot(
    fine_tuned: Mapping[str, Any], zero_shot_summary_path: str | Path
) -> dict[str, Any]:
    """Compare aggregates with saved Phase 3C results without invoking a model."""
    saved = json.loads(Path(zero_shot_summary_path).read_text(encoding="utf-8"))
    zero = saved["metrics"]
    current = fine_tuned["metrics"]
    zero_groups = zero["groups"]["chart_referring"]
    current_groups = current["groups"]["chart_referring"]
    if set(zero_groups) != set(current_groups) or len(current_groups) != 16:
        raise ValueError("fine-tuned and saved zero-shot group identities differ")
    groups: dict[str, Any] = {}
    for name in sorted(current_groups):
        groups[name] = {
            "zero_shot_iou": zero_groups[name]["mean_iou"],
            "fine_tuned_iou": current_groups[name]["mean_iou"],
            "delta_iou": current_groups[name]["mean_iou"]
            - zero_groups[name]["mean_iou"],
            "zero_shot_dice": zero_groups[name]["mean_dice"],
            "fine_tuned_dice": current_groups[name]["mean_dice"],
            "delta_dice": current_groups[name]["mean_dice"]
            - zero_groups[name]["mean_dice"],
        }
    improved = [name for name, row in groups.items() if row["delta_iou"] > 0]
    declined = [name for name, row in groups.items() if row["delta_iou"] < 0]
    tied = [name for name, row in groups.items() if row["delta_iou"] == 0]
    max_gain = max(groups, key=lambda name: (groups[name]["delta_iou"], name))
    max_loss = min(groups, key=lambda name: (groups[name]["delta_iou"], name))
    strongest = max(
        groups, key=lambda name: (groups[name]["fine_tuned_iou"], name)
    )
    weakest = min(groups, key=lambda name: (groups[name]["fine_tuned_iou"], name))
    metric_names = {
        "group_macro_iou": "group_macro_iou",
        "group_macro_dice": "group_macro_dice",
        "micro_iou": "micro_iou",
        "micro_dice": "micro_dice",
        "empty_prediction_rate": "empty_prediction_rate",
        "overlapping_prediction_rate": "overlapping_prediction_rate",
        "nonempty_disjoint_rate": "nonempty_disjoint_rate",
    }
    overall = {
        name: {
            "zero_shot": zero[source],
            "fine_tuned": current[source],
            "delta": current[source] - zero[source],
        }
        for name, source in metric_names.items()
    }
    return {
        "source": str(zero_shot_summary_path),
        "baseline_rerun": False,
        "overall": overall,
        "groups": groups,
        "improved_group_count": len(improved),
        "declined_group_count": len(declined),
        "tied_group_count": len(tied),
        "improved_groups": improved,
        "declined_groups": declined,
        "max_gain_group": max_gain,
        "max_gain_iou": groups[max_gain]["delta_iou"],
        "max_loss_group": max_loss,
        "max_loss_iou": groups[max_loss]["delta_iou"],
        "strongest_group": strongest,
        "strongest_group_iou": groups[strongest]["fine_tuned_iou"],
        "weakest_group": weakest,
        "weakest_group_iou": groups[weakest]["fine_tuned_iou"],
    }


def validate_one_shot_records(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 64 or len({row.get("sample_id") for row in rows}) != 64:
        raise ValueError("Phase 5B requires exactly one record for each of 64 samples")
    if any(row.get("projection_checkpoint_sha256") != SELECTED_PROJECTION_SHA256 for row in rows):
        raise ValueError("Phase 5B result contains a non-step960 projection identity")
