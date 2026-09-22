from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

from chartground_edit.inference.projection_checkpoint import (
    EXPECTED_PROJECTION_KEYS,
    load_projection_checkpoint_into_model,
)
from chartground_edit.inference.sa2va_backend import Sa2VAInternVL3Backend
from chartground_edit.training.data_adapter import Phase4TrainDataset
from chartground_edit.training.overfit32 import (
    build_epoch_schedule,
    summarize_overfit_rows,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/chartground_edit"
MANIFEST = PROJECT / "data/synthetic_v1/annotations.jsonl"
SELECTION = PROJECT / "configs/phase4_overfit32_ids.json"
CONFIG = PROJECT / "configs/phase4c_overfit32.py"
IDENTITY = {
    "source_hf_revision": "sa2va-revision",
    "full_pth_sha256": "full-pth-hash",
    "base_repo_id": "OpenGVLab/InternVL3-2B",
    "base_revision": "base-revision",
}


class TinyProjectionModel(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)
        ).to(dtype=dtype)


def _checkpoint(path: Path, model: nn.Module, *, mutate=None) -> Path:
    state = {
        name: parameter.detach().float().clone()
        for name, parameter in model.named_parameters()
    }
    if mutate is not None:
        mutate(state)
    payload = {
        "state_dict": state,
        "meta": {
            "chartground_phase4b": {
                "source_hf_revision": IDENTITY["source_hf_revision"],
                "full_pth": {"sha256": IDENTITY["full_pth_sha256"]},
                "base_checkpoint": {
                    "repo_id": IDENTITY["base_repo_id"],
                    "revision": IDENTITY["base_revision"],
                },
            }
        },
    }
    torch.save(payload, path)
    return path


def test_projection_loader_is_strict_and_converts_dtype(tmp_path) -> None:
    source = TinyProjectionModel(dtype=torch.float32)
    target = TinyProjectionModel(dtype=torch.float64)
    path = _checkpoint(tmp_path / "projection.pth", source)
    load_projection_checkpoint_into_model(
        target, path, expected_identity=IDENTITY
    )
    assert set(dict(target.named_parameters())) == set(EXPECTED_PROJECTION_KEYS)
    for name, parameter in target.named_parameters():
        assert parameter.dtype == torch.float64
        assert parameter.device.type == "cpu"
        assert torch.equal(parameter, dict(source.named_parameters())[name].double())


@pytest.mark.parametrize("failure", ["unknown_key", "wrong_shape"])
def test_projection_loader_rejects_unknown_keys_and_shapes(
    tmp_path, failure: str
) -> None:
    model = TinyProjectionModel()

    def mutate(state):
        if failure == "unknown_key":
            state["mllm.model.illegal"] = torch.zeros(1)
        else:
            state["text_hidden_fcs.0.weight"] = torch.zeros(3, 3)

    path = _checkpoint(tmp_path / f"{failure}.pth", model, mutate=mutate)
    with pytest.raises(ValueError, match="keys mismatch|shape mismatch"):
        load_projection_checkpoint_into_model(
            model, path, expected_identity=IDENTITY
        )


def test_projection_loader_rejects_identity_mismatch(tmp_path) -> None:
    model = TinyProjectionModel()
    path = _checkpoint(tmp_path / "identity.pth", model)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_projection_checkpoint_into_model(
            model,
            path,
            expected_identity={**IDENTITY, "full_pth_sha256": "wrong"},
        )


def test_backend_without_projection_checkpoint_keeps_default_contract(tmp_path) -> None:
    backend = Sa2VAInternVL3Backend(tmp_path)
    assert backend.projection_checkpoint is None
    assert backend.projection_identity is None
    assert backend.active_projection_checkpoint is None
    assert not backend.is_loaded


def test_overfit_schedule_has_ten_deterministic_visits_per_sample() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    ids = selection["sample_ids"]
    first = build_epoch_schedule(ids, epochs=10, seed=20260916)
    second = build_epoch_schedule(ids, epochs=10, seed=20260916)
    assert first == second
    assert len(first) == 320
    assert {sample_id: sum(row["sample_id"] == sample_id for row in first) for sample_id in ids} == {
        sample_id: 10 for sample_id in ids
    }
    assert {row["epoch"] for row in first} == set(range(1, 11))


@pytest.mark.parametrize("split", ["val", "test"])
def test_phase4c_dataset_rejects_val_and_test_selection(tmp_path, split: str) -> None:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    payload["split"] = split
    path = tmp_path / f"{split}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        Phase4TrainDataset(MANIFEST, path)


def test_phase4c_config_is_fixed_320_step_train_only() -> None:
    cfg = Config.fromfile(CONFIG)
    assert cfg.max_iters == cfg.train_cfg.max_iters == 320
    assert cfg.phase4c_epochs == 10
    assert cfg.checkpoint_steps == [32, 128, 320]
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.optim_wrapper.accumulative_counts == 1
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(4e-5)
    assert cfg.optim_wrapper.optimizer.weight_decay == pytest.approx(0.05)
    assert cfg.warmup_steps == 16
    assert cfg.train_dataloader.dataset.selection_path.endswith(
        "phase4_overfit32_ids.json"
    )
    assert cfg.val_cfg is cfg.test_cfg is None
    assert cfg.resume is False


def test_overfit_metric_aggregation() -> None:
    rows = []
    for checkpoint_index, checkpoint in enumerate(
        ("baseline", "step32", "step128", "step320")
    ):
        for group_index in range(16):
            score = min(1.0, checkpoint_index * 0.2 + group_index * 0.001)
            for sample_index in range(2):
                intersection = int(score * 100)
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "sample_id": f"{group_index}-{sample_index}",
                        "split": "train",
                        "chart_type": f"chart{group_index // 4}",
                        "referring_type": f"ref{group_index % 4}",
                        "iou": score,
                        "dice": score,
                        "empty_prediction": score == 0,
                        "nonempty_disjoint": False,
                        "overlapping_prediction": score > 0,
                        "intersection_pixels": intersection,
                        "union_pixels": 100,
                        "predicted_foreground_pixels": 100,
                        "gt_foreground_pixels": 100,
                    }
                )
    summary = summarize_overfit_rows(rows)
    assert summary["best_checkpoint"] == "step320"
    assert summary["checkpoints"]["step320"]["group_macro_iou"] > summary[
        "checkpoints"
    ]["baseline"]["group_macro_iou"]
