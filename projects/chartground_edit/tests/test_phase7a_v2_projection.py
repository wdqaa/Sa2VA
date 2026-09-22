from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest
from mmengine.config import Config

from chartground_edit.training.data_adapter import (
    EXPECTED_V2_MANIFEST_SHA256,
    Phase7V2SplitDataset,
    file_sha256,
)
from chartground_edit.training.overfit32 import build_epoch_schedule


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/chartground_edit"
MANIFEST = PROJECT / "data/synthetic_v2/annotations.jsonl"
CONFIG = PROJECT / "configs/phase7a_v2_projection.py"
SCRIPT = PROJECT / "scripts/run_phase7a_v2_projection.py"


def _runner():
    spec = importlib.util.spec_from_file_location("phase7a_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_rows() -> list[dict]:
    runner = _runner()
    rows = []
    for checkpoint_index, checkpoint in enumerate(runner.CHECKPOINT_NAMES):
        score = checkpoint_index / 10
        for chart_index in range(4):
            for referring_index in range(4):
                for sample_index in range(20):
                    rows.append(
                        {
                            "checkpoint": checkpoint,
                            "sample_id": f"{chart_index}-{referring_index}-{sample_index}",
                            "split": "val",
                            "chart_type": f"chart{chart_index}",
                            "referring_type": f"ref{referring_index}",
                            "edit_action": ("highlight", "recolor", "extract", "remove")[sample_index % 4],
                            "difficulty": ("easy", "medium", "hard")[sample_index % 3],
                            "theme": ("light", "dark")[sample_index % 2],
                            "degradation_type": "clean" if sample_index % 2 else "jpeg",
                            "iou": score,
                            "dice": score,
                            "empty_prediction": False,
                            "overlapping_prediction": True,
                            "nonempty_disjoint": False,
                            "intersection_pixels": int(score * 100),
                            "union_pixels": 100,
                            "predicted_foreground_pixels": 100,
                            "gt_foreground_pixels": 100,
                            "edit_execution_success": True,
                        }
                    )
    return rows


def test_phase7a_manifest_and_split_contract() -> None:
    assert file_sha256(MANIFEST) == EXPECTED_V2_MANIFEST_SHA256
    train = Phase7V2SplitDataset(MANIFEST, split="train")
    val = Phase7V2SplitDataset(MANIFEST, split="val")
    assert len(train) == 960 and {row["split"] for row in train.records} == {"train"}
    assert len(val) == 320 and {row["split"] for row in val.records} == {"val"}
    with pytest.raises(ValueError, match="test is frozen and forbidden"):
        Phase7V2SplitDataset(MANIFEST, split="test")


def test_phase7a_fixed_schedule_visits_each_train_sample_five_times() -> None:
    dataset = Phase7V2SplitDataset(MANIFEST, split="train")
    sample_ids = [row["sample_id"] for row in dataset.records]
    schedule = build_epoch_schedule(
        sample_ids, epochs=5, seed=20260916, expected_sample_count=960
    )
    assert len(schedule) == 4800
    assert Counter(row["sample_id"] for row in schedule) == Counter(
        {sample_id: 5 for sample_id in sample_ids}
    )
    for epoch in range(1, 6):
        epoch_ids = [row["sample_id"] for row in schedule if row["epoch"] == epoch]
        assert len(epoch_ids) == len(set(epoch_ids)) == 960


def test_phase7a_config_is_frozen_projection_only() -> None:
    cfg = Config.fromfile(CONFIG)
    assert cfg.max_iters == cfg.train_cfg.max_iters == 4800
    assert cfg.expected_train_samples == 960
    assert cfg.training_epochs == 5
    assert cfg.checkpoint_steps == [960, 1920, 2880, 3840, 4800]
    assert cfg.warmup_steps == 240
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.optim_wrapper.accumulative_counts == 1
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(4e-5)
    assert cfg.optim_wrapper.optimizer.weight_decay == pytest.approx(0.05)
    assert cfg.train_dataloader.dataset.dataset_protocol == "synthetic_v2"
    assert cfg.train_dataloader.dataset.split == "train"
    assert cfg.model.seg_token_selection == "supervised_labels"
    assert cfg.model.object_count_policy == "strict_one_to_one"
    assert cfg.model.mllm.freeze_llm is True
    assert cfg.model.mllm.freeze_visual_encoder is True
    assert cfg.model.mllm.llm_lora is None
    assert cfg.model.mllm.visual_encoder_lora is None
    assert cfg.model.frozen_sam2_decoder is True
    assert cfg.val_cfg is cfg.test_cfg is None
    assert cfg.resume is False


def test_phase7a_aggregation_and_preregistered_selection() -> None:
    runner = _runner()
    summary = runner.summarize_phase7a(_metric_rows())
    assert summary["best_checkpoint"] == "step4800"
    assert list(summary["checkpoints"]) == list(runner.CHECKPOINT_NAMES)
    assert set(summary["checkpoints"]["step4800"]["groups"]) == {
        "chart_referring",
        "chart_type",
        "referring_type",
        "action",
        "difficulty",
        "degradation_type",
        "theme",
    }


def test_phase7a_aggregation_rejects_non_val_rows() -> None:
    runner = _runner()
    rows = _metric_rows()
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="val-only"):
        runner.summarize_phase7a(rows)
