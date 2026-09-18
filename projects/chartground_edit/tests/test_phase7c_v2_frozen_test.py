from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from chartground_edit.training.data_adapter import Phase7V2SplitDataset


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/chartground_edit"
SCRIPT = PROJECT / "scripts/run_phase7c_v2_frozen_test.py"
PROTOCOL = PROJECT / "docs/phase7c_v2_frozen_test_protocol.md"


def _runner():
    spec = importlib.util.spec_from_file_location("phase7c_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> list[dict]:
    runner = _runner()
    rows = []
    for state_index, state in enumerate(runner.STATE_ORDER):
        score = 0.1 + state_index * 0.1
        for chart in range(4):
            for referring in range(4):
                for index in range(20):
                    rows.append({
                        "state": state,
                        "sample_id": f"{chart}-{referring}-{index}",
                        "split": "test",
                        "chart_type": f"chart{chart}",
                        "referring_type": f"ref{referring}",
                        "edit_action": ("highlight", "recolor", "extract", "remove")[index % 4],
                        "difficulty": ("easy", "medium", "hard")[index % 3],
                        "theme": ("light", "dark")[index % 2],
                        "degradation_type": "clean",
                        "iou": score,
                        "dice": score,
                        "empty_prediction": False,
                        "nonempty_prediction": True,
                        "overlapping_prediction": True,
                        "nonempty_disjoint": False,
                        "intersection_pixels": int(score * 100),
                        "union_pixels": 100,
                        "predicted_foreground_pixels": 100,
                        "gt_foreground_pixels": 100,
                        "execution_success": True,
                        "mask_contract_valid": True,
                        "segmentation_token_present": True,
                        "inference_time_ms": 10.0,
                        "edit_attempted": state == "v2_strategy_b",
                        "edit_execution_success": state == "v2_strategy_b",
                        "edit_skipped_empty": False if state == "v2_strategy_b" else None,
                    })
    return rows


def test_protocol_identity_and_test_manifest_are_frozen() -> None:
    runner = _runner()
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == runner.PROTOCOL_SHA256
    records = [
        json.loads(line)
        for line in (PROJECT / "data/synthetic_v2/annotations.jsonl").read_text().splitlines()
    ]
    test = [row for row in records if row["split"] == "test"]
    assert len(test) == 320
    assert runner.test_id_hash(test) == runner.TEST_ID_SHA256
    assert runner.EXPECTED_CALLS == 4 * 320
    assert runner.STATE_ORDER == (
        "zero_shot", "v1_step960", "v2_strategy_a", "v2_strategy_b"
    )
    dataset = Phase7V2SplitDataset(
        PROJECT / "data/synthetic_v2/annotations.jsonl",
        split="test",
        allow_test=True,
    )
    assert len(dataset) == 320 and {row["split"] for row in dataset.records} == {"test"}


def test_phase7c_aggregation_is_test_only_and_uses_all_four_states() -> None:
    runner = _runner()
    summary = runner.summarize(_rows())
    assert list(summary["states"]) == list(runner.STATE_ORDER)
    assert summary["comparisons"]["b_minus_zero_shot"]["group_macro_iou"] == pytest.approx(0.3)
    assert summary["states"]["v2_strategy_b"]["edit_attempt_count"] == 320
    invalid = _rows()
    invalid[0]["split"] = "val"
    with pytest.raises(ValueError, match="test-only"):
        runner.summarize(invalid)


def test_paired_group_bootstrap_is_deterministic() -> None:
    runner = _runner()
    rows = _rows()
    a = [row for row in rows if row["state"] == "v2_strategy_a"]
    b = [row for row in rows if row["state"] == "v2_strategy_b"]
    first = runner.paired_group_bootstrap(b, a)
    second = runner.paired_group_bootstrap(b, a)
    assert first == second
    assert first["iou"]["difference"] == pytest.approx(0.1)
    assert first["dice"]["difference"] == pytest.approx(0.1)


def test_only_strategy_b_runs_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    calls = []
    monkeypatch.setattr(
        runner,
        "apply_predicted_mask_edit",
        lambda image, mask, **kwargs: calls.append(kwargs) or image.copy(),
    )
    sample = SimpleNamespace(
        sample_id="sentinel",
        image=Image.new("RGB", (4, 4), "white"),
        mask=np.ones((1, 4, 4), dtype=np.uint8),
    )
    annotation = {
        "chart_type": "line",
        "referring_type": "trend",
        "edit_action": "highlight",
        "edit_parameters": {"color": "#ff0000", "strength": 0.5},
        "difficulty": "hard",
        "distractor_count": 3,
        "diversity_metadata": {
            "theme": "light",
            "degradation": {
                "jpeg_quality": 100,
                "blur_radius": 0.0,
                "screenshot_scale": 1.0,
                "antialias_factor": 1,
            },
        },
    }
    result = SimpleNamespace(
        mask=np.ones((4, 4), dtype=bool),
        failure_reason=None,
        text_output="[SEG]",
        metadata={},
        raw_mask_shapes=[[4, 4]],
        num_masks=1,
        inference_time_ms=1.0,
    )
    row, _, edited = runner._prediction_row(
        "zero_shot", sample, annotation, result, run_edit=False
    )
    assert calls == [] and edited is None
    assert row["edit_attempted"] is False
    row, _, edited = runner._prediction_row(
        "v2_strategy_b", sample, annotation, result, run_edit=True
    )
    assert len(calls) == 1 and edited is not None
    assert row["edit_attempted"] is True and row["edit_execution_success"] is True


def test_runner_has_no_training_entrypoint() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "optimizer.step" not in text
    assert "backward(" not in text
    assert 'split="test"' in text
    assert 'split="train"' not in text and 'split="val"' not in text
