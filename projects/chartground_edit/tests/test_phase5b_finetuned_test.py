from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from chartground_edit.inference.phase5b import (
    BASE_REPO_ID,
    BASE_REVISION,
    FULL_PTH_SHA256,
    SA2VA_REVISION,
    SELECTED_PROJECTION_SHA256,
    _validate_projection,
    compare_with_saved_zero_shot,
    sha256_file,
    validate_one_shot_records,
    validate_selected_projection,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/chartground_edit"
DEMO = PROJECT / "scripts/run_chartground_edit.py"
FROZEN_RUNNER = PROJECT / "scripts/run_frozen_test_v1.py"
ZERO_SHOT_SUMMARY = PROJECT / "results/phase3c_frozen_test_summary.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_projection(path: Path) -> Path:
    metadata = {
        "optimizer_step": 960,
        "trainable_parameter_count": 2_754_304,
        "source_hf_revision": SA2VA_REVISION,
        "prompt_variant": "target_only_zh",
        "prompt_template_sha256": (
            "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806"
        ),
        "prompt_registry_sha256": (
            "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0"
        ),
        "base_checkpoint": {"repo_id": BASE_REPO_ID, "revision": BASE_REVISION},
        "full_pth": {"sha256": FULL_PTH_SHA256},
    }
    torch.save(
        {
            "meta": {"chartground_phase4b": metadata},
            "state_dict": {
                "text_hidden_fcs.0.weight": torch.zeros(1536, 1536),
                "text_hidden_fcs.0.bias": torch.zeros(1536),
                "text_hidden_fcs.2.weight": torch.zeros(256, 1536),
                "text_hidden_fcs.2.bias": torch.zeros(256),
            },
        },
        path,
    )
    return path


def _write_strategy_b_adapter(module, path: Path) -> Path:
    targets = [
        f"model.layers.{layer}.self_attn.{name}"
        for layer in range(20, 28)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    state = {
        "text_hidden_fcs.0.weight": torch.zeros(1),
        "text_hidden_fcs.0.bias": torch.zeros(1),
        "text_hidden_fcs.2.weight": torch.zeros(1),
        "text_hidden_fcs.2.bias": torch.zeros(1),
    }
    for target in targets:
        for branch in ("A", "B"):
            state[
                f"language_model.base_model.model.{target}.lora_{branch}.default.weight"
            ] = torch.zeros(1)
    metadata = {
        "optimizer_step": 4800,
        "trainable_parameter_count": 3_999_488,
        "source_hf_revision": SA2VA_REVISION,
        "prompt_variant": "target_only_zh",
        "prompt_template_sha256": module.P2_SHA256,
        "prompt_registry_sha256": module.PROMPT_REGISTRY_SHA256,
        "base_checkpoint": {"repo_id": BASE_REPO_ID, "revision": BASE_REVISION},
        "full_pth": {"sha256": FULL_PTH_SHA256},
        "manifest_sha256": module.V2_MANIFEST_SHA256,
        "checkpoint_state": "text_hidden_fcs_plus_llm_lora",
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "layers": list(range(20, 28)),
            "target_modules": targets,
            "modules_to_save": None,
        },
    }
    torch.save(
        {"meta": {"chartground_phase7b": metadata}, "state_dict": state}, path
    )
    return path


def test_selected_checkpoint_hash_identity_and_four_tensor_contract(
    tmp_path: Path,
) -> None:
    projection = _write_projection(tmp_path / "step_960.pth")
    audit = _validate_projection(
        projection, expected_sha256=sha256_file(projection)
    )
    assert audit["selected_checkpoint"] == "step960"
    assert len(SELECTED_PROJECTION_SHA256) == 64
    assert audit["projection_tensor_count"] == 4
    assert audit["trainable_parameter_count"] == 2_754_304


def test_phase5b_rejects_any_other_projection(tmp_path: Path) -> None:
    altered = _write_projection(tmp_path / "step_192.pth")
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


def test_demo_cli_requires_projection_and_supports_all_four_actions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load(DEMO, "chartground_demo_parser")
    with pytest.raises(SystemExit) as exc_info:
        module.parse_args([])
    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--projection-checkpoint" in error and "--adapter-checkpoint" in error
    source = DEMO.read_text(encoding="utf-8")
    assert 'choices=sorted(EDIT_ACTIONS)' in source
    for action in ("highlight", "recolor", "extract", "remove"):
        assert action in module.EDIT_ACTIONS


def test_demo_accepts_strict_strategy_b_adapter_without_ground_truth(
    tmp_path: Path,
) -> None:
    module = _load(DEMO, "chartground_demo_strategy_b")
    adapter = _write_strategy_b_adapter(module, tmp_path / "adapter.pth")
    module.FINAL_ADAPTER_SHA256 = sha256_file(adapter)
    audit = module.validate_selected_adapter(adapter)
    assert audit == {
        "adapter_checkpoint_sha256": sha256_file(adapter),
        "adapter_tensor_count": 68,
        "projection_tensor_count": 4,
        "lora_tensor_count": 64,
        "trainable_parameter_count": 3_999_488,
    }

    image_path = tmp_path / "chart.png"
    Image.new("RGB", (8, 6), "white").save(image_path)
    metadata_dir = tmp_path / ".cache/huggingface/download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "config.json.metadata").write_text(
        SA2VA_REVISION + "\n", encoding="utf-8"
    )
    captured = {}

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def predict_prompt(self, image, prompt, *, instruction):
            return SimpleNamespace(
                mask=np.ones((6, 8), dtype=bool),
                text_output="Sure, [SEG].",
                failure_reason=None,
            )

    args = argparse.Namespace(
        checkpoint=tmp_path,
        projection_checkpoint=None,
        adapter_checkpoint=adapter,
        image=image_path,
        referring_expression="图中上升最快的折线",
        action="highlight",
        color="#E63946",
        strength=0.65,
        fill_mode="color",
        neighbor_radius=5,
        output_dir=tmp_path / "adapter-output",
        device="cuda:0",
        dtype="bfloat16",
    )
    result = module.run(
        args,
        backend_factory=FakeBackend,
        adapter_validator=lambda path: audit,
    )
    assert captured["adapter_checkpoint"] == adapter
    assert captured["adapter_identity"] == module.adapter_identity()
    assert "projection_checkpoint" not in captured
    assert result["checkpoint_kind"] == "projection_plus_lora"
    assert result["adapter_checkpoint_sha256"] == audit["adapter_checkpoint_sha256"]
    assert result["projection_checkpoint"] is None
    assert result["edit_execution_success"] is True
    assert "ground_truth" not in vars(args) and "gt" not in vars(args)


def test_demo_rejects_unknown_sa2va_base_revision(tmp_path: Path) -> None:
    module = _load(DEMO, "chartground_demo_base_revision")
    with pytest.raises(ValueError, match="revision mismatch"):
        module.validate_sa2va_revision(tmp_path)
    metadata_dir = tmp_path / ".cache/huggingface/download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "config.json.metadata").write_text("wrong-revision\n")
    with pytest.raises(ValueError, match="revision mismatch"):
        module.validate_sa2va_revision(tmp_path)


def test_demo_rejects_combined_projection_and_adapter(tmp_path: Path) -> None:
    module = _load(DEMO, "chartground_demo_mutual_exclusion")
    args = argparse.Namespace(
        checkpoint=tmp_path,
        projection_checkpoint=tmp_path / "projection.pth",
        adapter_checkpoint=tmp_path / "adapter.pth",
        output_dir=tmp_path / "output",
    )
    with pytest.raises(ValueError, match="exactly one"):
        module.run(args)


def test_phase5b_runner_requires_projection_and_saved_baseline_arguments() -> None:
    source = FROZEN_RUNNER.read_text(encoding="utf-8")
    assert 'args.experiment == "phase5b"' in source
    assert "validate_selected_projection(args.projection_checkpoint)" in source
    assert "compare_with_saved_zero_shot" in source
