from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from mmengine.config import Config

from chartground_edit.training.data_adapter import Phase5SplitDataset
from chartground_edit.training.overfit32 import build_epoch_schedule
from chartground_edit.training.phase5a import (
    CHECKPOINT_NAMES,
    summarize_phase5a_rows,
    validate_phase3b_baseline,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/chartground_edit"
MANIFEST = PROJECT / "data/synthetic_v1/annotations.jsonl"
CONFIG = PROJECT / "configs/phase5a_full_train.py"
PHASE3B_METRICS = PROJECT / "results/phase3b_balanced_val_metrics.jsonl"


def test_phase5a_complete_train_schedule_visits_every_sample_ten_times() -> None:
    dataset = Phase5SplitDataset(MANIFEST, split="train")
    ids = [record["sample_id"] for record in dataset.records]
    schedule = build_epoch_schedule(
        ids, epochs=10, seed=20260916, expected_sample_count=192
    )
    assert len(schedule) == 1920
    assert Counter(row["sample_id"] for row in schedule) == Counter(
        {sample_id: 10 for sample_id in ids}
    )
    for epoch in range(1, 11):
        epoch_ids = [row["sample_id"] for row in schedule if row["epoch"] == epoch]
        assert len(epoch_ids) == len(set(epoch_ids)) == 192


def test_phase5a_val_is_evaluation_only_and_test_is_rejected() -> None:
    val = Phase5SplitDataset(MANIFEST, split="val")
    assert len(val) == 64
    assert {record["split"] for record in val.records} == {"val"}
    with pytest.raises(ValueError, match="test is frozen and forbidden"):
        Phase5SplitDataset(MANIFEST, split="test")


def test_phase5a_config_is_frozen() -> None:
    cfg = Config.fromfile(CONFIG)
    assert cfg.max_iters == cfg.train_cfg.max_iters == 1920
    assert cfg.expected_train_samples == 192
    assert cfg.training_epochs == 10
    assert cfg.checkpoint_steps == [192, 576, 960, 1344, 1920]
    assert cfg.warmup_steps == 96
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.optim_wrapper.accumulative_counts == 1
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(4e-5)
    assert cfg.optim_wrapper.optimizer.weight_decay == pytest.approx(0.05)
    assert cfg.train_dataloader.dataset.selection_path is None
    assert cfg.train_dataloader.dataset.split == "train"
    assert cfg.val_cfg is cfg.test_cfg is None
    assert cfg.resume is False


def test_phase3b_p2_baseline_reproduction_gate_accepts_frozen_rows() -> None:
    rows = [
        json.loads(line)
        for line in PHASE3B_METRICS.read_text(encoding="utf-8").splitlines()
    ]
    p2 = [row for row in rows if row["prompt_variant"] == "target_only_zh"]
    summary = validate_phase3b_baseline(p2)
    assert summary["nonempty_count"] == 36
    altered = [dict(row) for row in p2]
    altered[0]["iou"] += 0.01
    with pytest.raises(ValueError, match="baseline reproduction failed"):
        validate_phase3b_baseline(altered)


def test_phase5a_checkpoint_selection_uses_registered_tie_breaks() -> None:
    rows = []
    scores = {
        "baseline": (0.1, 0.2, False),
        "step192": (0.3, 0.4, False),
        "step576": (0.4, 0.5, False),
        "step960": (0.4, 0.6, False),
        "step1344": (0.4, 0.6, True),
        "step1920": (0.4, 0.6, False),
    }
    for checkpoint in CHECKPOINT_NAMES:
        iou, dice, empty = scores[checkpoint]
        for chart_index in range(4):
            for referring_index in range(4):
                for sample_index in range(4):
                    rows.append(
                        {
                            "checkpoint": checkpoint,
                            "sample_id": (
                                f"{checkpoint}-{chart_index}-{referring_index}-"
                                f"{sample_index}"
                            ),
                            "split": "val",
                            "chart_type": f"chart{chart_index}",
                            "referring_type": f"ref{referring_index}",
                            "difficulty": ("easy", "medium", "hard")[sample_index % 3],
                            "iou": iou,
                            "dice": dice,
                            "empty_prediction": empty,
                            "nonempty_disjoint": False,
                            "overlapping_prediction": not empty,
                            "intersection_pixels": int(iou * 100),
                            "union_pixels": 100,
                            "predicted_foreground_pixels": 100,
                            "gt_foreground_pixels": 100,
                        }
                    )
    summary = summarize_phase5a_rows(rows)
    assert summary["best_checkpoint"] == "step960"
    assert list(summary["checkpoints"]) == list(CHECKPOINT_NAMES)
    assert set(summary["checkpoints"]["step960"]["groups"]) == {
        "chart_referring",
        "chart_type",
        "referring_type",
        "difficulty",
    }
