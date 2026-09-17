from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from chartground_edit.inference.phase5b import (
    SELECTED_PROJECTION_SHA256,
    compare_with_saved_zero_shot,
    validate_one_shot_records,
    validate_selected_projection,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/chartground_edit"
DEMO = PROJECT / "scripts/run_chartground_edit.py"
FROZEN_RUNNER = PROJECT / "scripts/run_frozen_test_v1.py"
PROJECTION = Path(
    "/home/dqwang/Model/ChartGround-Edit/chartground_projection_step960.pth"
)
ZERO_SHOT_SUMMARY = PROJECT / "results/phase3c_frozen_test_summary.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_checkpoint_hash_identity_and_four_tensor_contract() -> None:
    audit = validate_selected_projection(PROJECTION)
    assert audit["selected_checkpoint"] == "step960"
    assert audit["projection_checkpoint_sha256"] == SELECTED_PROJECTION_SHA256
    assert audit["projection_tensor_count"] == 4
    assert audit["trainable_parameter_count"] == 2_754_304


def test_phase5b_rejects_any_other_projection(tmp_path: Path) -> None:
    altered = tmp_path / "step_192.pth"
    payload = bytearray(PROJECTION.read_bytes())
    payload[-1] ^= 1
    altered.write_bytes(payload)
    with pytest.raises(ValueError, match="only accepts the frozen step960"):
        validate_selected_projection(altered)


def test_one_shot_contract_requires_64_unique_step960_records() -> None:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "projection_checkpoint_sha256": SELECTED_PROJECTION_SHA256,
        }
        for index in range(64)
    ]
    validate_one_shot_records(rows)
    with pytest.raises(ValueError, match="exactly one record"):
        validate_one_shot_records(rows[:-1])
    wrong = [dict(row) for row in rows]
    wrong[-1]["projection_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="non-step960"):
        validate_one_shot_records(wrong)


def test_comparison_reads_saved_phase3c_summary_without_backend_call() -> None:
    saved = json.loads(ZERO_SHOT_SUMMARY.read_text(encoding="utf-8"))
    fine_tuned = {"metrics": json.loads(json.dumps(saved["metrics"]))}
    comparison = compare_with_saved_zero_shot(fine_tuned, ZERO_SHOT_SUMMARY)
    assert comparison["baseline_rerun"] is False
    assert comparison["improved_group_count"] == 0
    assert comparison["declined_group_count"] == 0
    assert comparison["tied_group_count"] == 16
    assert comparison["overall"]["group_macro_iou"]["delta"] == 0


def test_demo_cli_has_no_ground_truth_argument_and_uses_predicted_mask(
    tmp_path: Path,
) -> None:
    module = _load(DEMO, "chartground_demo")
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (8, 6), "white").save(image_path)
    output_dir = tmp_path / "output"
    predicted = np.zeros((6, 8), dtype=bool)
    predicted[2:4, 3:6] = True
    captured: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            captured["backend_kwargs"] = kwargs

        def predict_prompt(self, image, prompt, *, instruction):
            captured["image"] = image
            captured["prompt"] = prompt
            captured["instruction"] = instruction
            return SimpleNamespace(
                mask=predicted,
                text_output="Sure, [SEG].",
                failure_reason=None,
            )

    args = argparse.Namespace(
        checkpoint=tmp_path,
        projection_checkpoint=tmp_path / "projection.pth",
        image=image_path,
        referring_expression="图中上升最快的折线",
        action="recolor",
        color="#E63946",
        strength=0.65,
        fill_mode="color",
        neighbor_radius=5,
        output_dir=output_dir,
        device="cuda:0",
        dtype="bfloat16",
    )
    result = module.run(
        args,
        backend_factory=FakeBackend,
        projection_validator=lambda path: {
            "projection_checkpoint_sha256": SELECTED_PROJECTION_SHA256
        },
    )
    assert "ground_truth" not in vars(args)
    assert "gt" not in vars(args)
    assert result["edit_execution_success"] is True
    assert result["predicted_foreground_pixels"] == int(predicted.sum())
    assert (output_dir / "predicted_mask.png").is_file()
    assert (output_dir / "overlay.png").is_file()
    assert (output_dir / "edited.png").is_file()
    assert (output_dir / "result.json").is_file()
    source = DEMO.read_text(encoding="utf-8")
    assert 'add_argument("--gt' not in source
    assert 'add_argument("--mask' not in source


def test_demo_empty_prediction_skips_edit_without_ground_truth(tmp_path: Path) -> None:
    module = _load(DEMO, "chartground_demo_empty")
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (8, 6), "white").save(image_path)

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

        def predict_prompt(self, image, prompt, *, instruction):
            return SimpleNamespace(
                mask=np.zeros((6, 8), dtype=bool),
                text_output="Sure, [SEG].",
                failure_reason="empty_prediction_mask",
            )

    args = argparse.Namespace(
        checkpoint=tmp_path,
        projection_checkpoint=tmp_path / "projection.pth",
        image=image_path,
        referring_expression="目标",
        action="remove",
        color="#FFFFFF",
        strength=0.65,
        fill_mode="neighbor",
        neighbor_radius=5,
        output_dir=tmp_path / "empty-output",
        device="cuda:0",
        dtype="bfloat16",
    )
    result = module.run(
        args,
        backend_factory=FakeBackend,
        projection_validator=lambda path: {
            "projection_checkpoint_sha256": SELECTED_PROJECTION_SHA256
        },
    )
    assert result["edit_skipped_empty"] is True
    assert result["edit_execution_success"] is False
    assert result["output_files"]["edited"] is None


def test_phase5b_runner_requires_projection_and_saved_baseline_arguments() -> None:
    source = FROZEN_RUNNER.read_text(encoding="utf-8")
    assert 'args.experiment == "phase5b"' in source
    assert "validate_selected_projection(args.projection_checkpoint)" in source
    assert "compare_with_saved_zero_shot" in source
