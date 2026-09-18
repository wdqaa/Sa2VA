from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from mmengine.config import Config

from chartground_edit.training.strategy_b import (
    LORA_PROJECTIONS,
    PROJECTION_KEYS,
    canonical_trainable_name,
    exact_lora_targets,
    load_strategy_b_checkpoint_into_hf_model,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/chartground_edit"
CONFIG = PROJECT / "configs/phase7b_v2_lora.py"
SCRIPT = PROJECT / "scripts/run_phase7b_v2_lora.py"


def _runner():
    spec = importlib.util.spec_from_file_location("phase7b_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(scores: dict[str, float]) -> list[dict]:
    runner = _runner()
    rows = []
    for checkpoint in runner.B_NAMES:
        score = scores[checkpoint]
        for chart in range(4):
            for referring in range(4):
                for index in range(20):
                    rows.append({
                        "checkpoint": checkpoint,
                        "sample_id": f"{chart}-{referring}-{index}",
                        "split": "val",
                        "chart_type": f"chart{chart}",
                        "referring_type": f"ref{referring}",
                        "edit_action": ("highlight", "recolor", "extract", "remove")[index % 4],
                        "difficulty": ("easy", "medium", "hard")[index % 3],
                        "degradation_type": "clean",
                        "theme": ("light", "dark")[index % 2],
                        "iou": score,
                        "dice": score,
                        "empty_prediction": False,
                        "overlapping_prediction": True,
                        "nonempty_disjoint": False,
                        "intersection_pixels": int(score * 100),
                        "union_pixels": 100,
                        "predicted_foreground_pixels": 100,
                        "gt_foreground_pixels": 100,
                        "inference_time_ms": 10.0,
                    })
    return rows


def test_exact_last_eight_attention_targets() -> None:
    targets = exact_lora_targets(28, 8)
    assert len(targets) == 32 and len(set(targets)) == 32
    assert {int(name.split(".")[2]) for name in targets} == set(range(20, 28))
    assert {name.rsplit(".", 1)[-1] for name in targets} == set(LORA_PROJECTIONS)
    assert all("mlp" not in name and "vision" not in name for name in targets)


def test_phase7b_config_is_frozen_and_test_disabled() -> None:
    cfg = Config.fromfile(CONFIG)
    assert cfg.max_iters == 4800 and cfg.training_epochs == 5
    assert cfg.checkpoint_steps == [960, 1920, 2880, 3840, 4800]
    assert cfg.train_dataloader.dataset.split == "train"
    assert cfg.train_dataloader.dataset.dataset_protocol == "synthetic_v2"
    assert cfg.model.mllm.llm_lora.r == 16
    assert cfg.model.mllm.llm_lora.lora_alpha == 32
    assert cfg.model.mllm.llm_lora.lora_dropout == pytest.approx(0.05)
    assert cfg.model.mllm.llm_lora.bias == "none"
    assert cfg.model.mllm.llm_lora.modules_to_save is None
    assert cfg.model.object_count_policy == "strict_one_to_one"
    assert cfg.val_cfg is None and cfg.test_cfg is None
    assert cfg.expected_projection_parameter_count == 2_754_304
    assert cfg.expected_lora_parameter_count == 1_245_184
    assert cfg.expected_total_trainable_parameter_count == 3_999_488


def test_canonical_name_only_removes_training_wrapper_prefix() -> None:
    name = "mllm.model.language_model.base_model.model.model.layers.20.self_attn.q_proj.lora_A.default.weight"
    assert canonical_trainable_name(name) == "language_model.base_model.model.model.layers.20.self_attn.q_proj.lora_A.default.weight"
    assert canonical_trainable_name("text_hidden_fcs.0.weight") == "text_hidden_fcs.0.weight"


def test_phase7b_aggregation_uses_preregistered_tie_break() -> None:
    runner = _runner()
    scores = {name: 0.2 for name in runner.B_NAMES}
    scores["step1920"] = scores["step2880"] = 0.3
    summary = runner.summarize_b(_rows(scores))
    assert summary["best_checkpoint"] == "step1920"
    assert set(summary["checkpoints"]["step1920"]["groups"]) == {
        "chart_referring", "chart_type", "referring_type", "action",
        "difficulty", "degradation_type", "theme",
    }


def test_phase7b_aggregation_rejects_test() -> None:
    runner = _runner()
    rows = _rows({name: 0.2 for name in runner.B_NAMES})
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="val-only"):
        runner.summarize_b(rows)


class _FakeAdapterProjection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in LORA_PROJECTIONS:
            setattr(self, name, _FakeAdapterProjection())


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeHFStrategyB(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.text_hidden_fcs = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Identity(), torch.nn.Linear(4, 2)
        )
        self.language_model = torch.nn.Module()
        self.language_model.base_model = torch.nn.Module()
        self.language_model.base_model.model = torch.nn.Module()
        self.language_model.base_model.model.model = torch.nn.Module()
        self.language_model.base_model.model.model.layers = torch.nn.ModuleList(
            [_FakeLayer() for _ in range(8)]
        )
        self.to(dtype=dtype)


def _fake_strategy_b_payload(model: torch.nn.Module) -> dict:
    state = {
        name: torch.full_like(parameter, 0.25, dtype=torch.float32)
        for name, parameter in model.named_parameters()
        if name in PROJECTION_KEYS
        or ".lora_A.default.weight" in name
        or ".lora_B.default.weight" in name
    }
    assert len(state) == 68
    return {
        "meta": {"chartground_phase7b": {
            "source_hf_revision": "sa2va-rev",
            "full_pth": {"sha256": "full-pth"},
            "base_checkpoint": {"repo_id": "base", "revision": "base-rev"},
            "manifest_sha256": "manifest",
        }},
        "state_dict": state,
    }


def test_strategy_b_hf_loader_is_strict_and_casts_dtype(tmp_path: Path) -> None:
    model = _FakeHFStrategyB()
    checkpoint = tmp_path / "strategy_b.pth"
    torch.save(_fake_strategy_b_payload(model), checkpoint)
    identity = {
        "source_hf_revision": "sa2va-rev",
        "full_pth_sha256": "full-pth",
        "base_repo_id": "base",
        "base_revision": "base-rev",
        "manifest_sha256": "manifest",
    }
    load_strategy_b_checkpoint_into_hf_model(
        model, checkpoint, expected_identity=identity
    )
    selected = [
        parameter for name, parameter in model.named_parameters()
        if name in PROJECTION_KEYS
        or ".lora_A.default.weight" in name
        or ".lora_B.default.weight" in name
    ]
    assert len(selected) == 68
    assert all(parameter.dtype == torch.float16 for parameter in selected)
    assert all(torch.all(parameter == 0.25) for parameter in selected)

    payload = _fake_strategy_b_payload(model)
    payload["state_dict"]["unknown.weight"] = torch.ones(1)
    invalid = tmp_path / "unknown-key.pth"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match="key mismatch"):
        load_strategy_b_checkpoint_into_hf_model(
            model, invalid, expected_identity=identity
        )

    payload = _fake_strategy_b_payload(model)
    first = next(iter(payload["state_dict"]))
    payload["state_dict"][first] = torch.ones(1, dtype=torch.float32)
    wrong_shape = tmp_path / "wrong-shape.pth"
    torch.save(payload, wrong_shape)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_strategy_b_checkpoint_into_hf_model(
            model, wrong_shape, expected_identity=identity
        )

    with pytest.raises(ValueError, match="identity mismatch"):
        load_strategy_b_checkpoint_into_hf_model(
            model, checkpoint,
            expected_identity={**identity, "manifest_sha256": "other"},
        )
