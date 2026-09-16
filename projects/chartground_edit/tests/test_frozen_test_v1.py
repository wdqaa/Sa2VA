from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from chartground_edit.inference.frozen_test_v1 import (
    BOOTSTRAP_ITERATIONS,
    EXPECTED_TEST_SAMPLES,
    apply_predicted_mask_edit,
    bootstrap_group_macro_ci,
    build_frozen_test_plan,
    select_frozen_test_gallery_sample_ids,
    summarize_frozen_test,
    validate_balanced_test_annotations,
    validate_frozen_prompt_variant,
    validate_frozen_test_results,
    validate_frozen_test_split,
)
from chartground_edit.inference.metrics import dice_score, intersection_over_union
from chartground_edit.inference.prompt_benchmark_v1 import result_metric_semantics
from chartground_edit.inference.prompt_variants import TARGET_ONLY_ZH
from chartground_edit.inference.split_evaluation import SplitEvaluationError


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "projects/chartground_edit/data/synthetic_v1/annotations.jsonl"
RUNNER = ROOT / "projects/chartground_edit/scripts/run_frozen_test_v1.py"
CHECKPOINT = Path("/home/dqwang/Model/Sa2VA-InternVL3-2B")


def _annotations() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]


def _test_annotations() -> list[dict[str, Any]]:
    return validate_balanced_test_annotations(_annotations())


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted(
        {(row["chart_type"], row["referring_type"]) for row in _test_annotations()}
    )
    group_score = {group: index / 15.0 for index, group in enumerate(groups)}
    for annotation in _test_annotations():
        score = group_score[(annotation["chart_type"], annotation["referring_type"])]
        intersection = int(round(score * 1000))
        rows.append(
            {
                "sample_id": annotation["sample_id"],
                "split": "test",
                "prompt_variant": TARGET_ONLY_ZH,
                "chart_type": annotation["chart_type"],
                "referring_type": annotation["referring_type"],
                "edit_action": annotation["edit_action"],
                "difficulty": annotation["difficulty"],
                "distractor_count": annotation["distractor_count"],
                "execution_success": True,
                "mask_contract_valid": True,
                "segmentation_token_present": True,
                "nonempty_prediction": score > 0,
                "empty_prediction": score == 0,
                "overlapping_prediction": score > 0,
                "nonempty_disjoint": False,
                "predicted_foreground_pixels": 1000 if score > 0 else 0,
                "gt_foreground_pixels": 1000,
                "intersection_pixels": intersection,
                "union_pixels": 2000 - intersection if score > 0 else 1000,
                "iou": score,
                "dice": score,
                "latency_ms": 10.0,
                "edit_execution_success": True if score > 0 else None,
                "edit_skipped_empty": score == 0,
            }
        )
    return rows


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("phase3c_test_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("split", ["train", "val"])
def test_runner_rejects_train_and_val(split: str) -> None:
    with pytest.raises(ValueError):
        validate_frozen_test_split(split)
    runner = _load_runner()
    arguments = [
        "--checkpoint",
        str(CHECKPOINT),
        "--manifest",
        str(MANIFEST),
        "--split",
        split,
        "--prompt-variant",
        TARGET_ONLY_ZH,
        "--expected-samples",
        "64",
        "--output-dir",
        "/tmp/phase3c_argparse_test",
    ]
    with pytest.raises(SystemExit):
        runner.parse_args(arguments)


@pytest.mark.parametrize("variant", ["full_instruction", "target_only_en"])
def test_runner_rejects_p0_and_p1(variant: str) -> None:
    with pytest.raises(ValueError):
        validate_frozen_prompt_variant(variant)
    runner = _load_runner()
    arguments = [
        "--checkpoint",
        str(CHECKPOINT),
        "--manifest",
        str(MANIFEST),
        "--split",
        "test",
        "--prompt-variant",
        variant,
        "--expected-samples",
        "64",
        "--output-dir",
        "/tmp/phase3c_argparse_test",
    ]
    with pytest.raises(SystemExit):
        runner.parse_args(arguments)


def test_test_split_must_have_exactly_64_samples() -> None:
    annotations = _annotations()
    removed_id = next(row["sample_id"] for row in annotations if row["split"] == "test")
    incomplete = [row for row in annotations if row["sample_id"] != removed_id]
    with pytest.raises(SplitEvaluationError, match="expected 64"):
        validate_balanced_test_annotations(incomplete)
    with pytest.raises(SplitEvaluationError, match="expected_samples=64"):
        validate_balanced_test_annotations(annotations, expected_samples=63)


def test_plan_and_results_contain_each_sample_once_and_exactly_64() -> None:
    plan = build_frozen_test_plan(_annotations())
    assert len(plan) == EXPECTED_TEST_SAMPLES
    assert len({row["sample_id"] for row in plan}) == EXPECTED_TEST_SAMPLES
    assert {row["prompt_variant"] for row in plan} == {TARGET_ONLY_ZH}

    rows = _rows()
    validate_frozen_test_results(rows)
    with pytest.raises(SplitEvaluationError, match="64 result records"):
        validate_frozen_test_results(rows[:-1])
    with pytest.raises(SplitEvaluationError, match="64 result records"):
        validate_frozen_test_results(rows + [rows[-1]])
    duplicated = [dict(row) for row in rows]
    duplicated[-1]["sample_id"] = duplicated[0]["sample_id"]
    with pytest.raises(SplitEvaluationError, match="exactly once"):
        validate_frozen_test_results(duplicated)


def test_empty_mask_metric_semantics_are_execution_successful() -> None:
    semantics = result_metric_semantics(
        prediction_returned=True,
        failure_reason="empty_prediction_mask",
        error_type=None,
        mask_contract_valid=True,
        predicted_foreground_pixels=0,
        intersection_pixels=0,
    )
    empty = np.zeros((4, 5), dtype=bool)
    gt = np.zeros((4, 5), dtype=bool)
    gt[1, 2] = True
    assert semantics == {
        "execution_success": True,
        "mask_contract_valid": True,
        "nonempty_prediction": False,
        "empty_prediction": True,
        "overlapping_prediction": False,
    }
    assert intersection_over_union(empty, gt) == 0.0
    assert dice_score(empty, gt) == 0.0


def test_exception_sample_remains_in_formal_aggregate() -> None:
    rows = _rows()
    failed = rows[0]
    failed.update(
        {
            "execution_success": False,
            "mask_contract_valid": False,
            "segmentation_token_present": False,
            "nonempty_prediction": False,
            "empty_prediction": True,
            "overlapping_prediction": False,
            "nonempty_disjoint": False,
            "predicted_foreground_pixels": 0,
            "intersection_pixels": 0,
            "union_pixels": failed["gt_foreground_pixels"],
            "iou": 0.0,
            "dice": 0.0,
            "latency_ms": None,
            "edit_execution_success": None,
            "edit_skipped_empty": True,
        }
    )
    summary = summarize_frozen_test(rows)
    assert summary["sample_count"] == 64
    assert summary["metrics"]["execution_success_count"] == 63
    assert summary["metrics"]["timed_attempt_count"] == 63


def test_16_group_macro_uses_equal_group_weight() -> None:
    summary = summarize_frozen_test(_rows())
    expected = float(np.mean([index / 15.0 for index in range(16)]))
    assert summary["metrics"]["group_macro_iou"] == pytest.approx(expected)
    assert summary["metrics"]["group_macro_dice"] == pytest.approx(expected)
    assert len(summary["metrics"]["groups"]["chart_referring"]) == 16


def test_bootstrap_is_deterministic_for_iou_and_dice() -> None:
    group_summaries = summarize_frozen_test(_rows())["metrics"]["groups"][
        "chart_referring"
    ]
    first = bootstrap_group_macro_ci(
        group_summaries, seed=20260916, iterations=BOOTSTRAP_ITERATIONS
    )
    second = bootstrap_group_macro_ci(
        group_summaries, seed=20260916, iterations=BOOTSTRAP_ITERATIONS
    )
    assert first == second
    assert set(first) == {"seed", "iterations", "unit", "iou", "dice"}


def test_gallery_selection_is_independent_of_effect_metrics() -> None:
    annotations = _test_annotations()
    selected = select_frozen_test_gallery_sample_ids(annotations)
    modified = [dict(row, iou=1.0 if index % 2 else 0.0) for index, row in enumerate(annotations)]
    assert select_frozen_test_gallery_sample_ids(modified) == selected
    assert len(selected) == 16


def test_predicted_mask_edit_never_receives_ground_truth() -> None:
    image = Image.new("RGB", (5, 4), "white")
    predicted = np.zeros((4, 5), dtype=bool)
    predicted[1, 2] = True
    ground_truth = np.zeros((4, 5), dtype=bool)
    ground_truth[3, 4] = True
    captured: dict[str, Any] = {}

    def fake_editor(
        actual_image: Image.Image,
        actual_mask: np.ndarray,
        action: str,
        parameters: dict[str, Any],
    ) -> Image.Image:
        captured["mask"] = actual_mask.copy()
        captured["action"] = action
        captured["parameters"] = parameters
        return actual_image.copy()

    apply_predicted_mask_edit(
        image,
        predicted,
        edit_action="recolor",
        edit_parameters={"color": "#F4A261"},
        editor=fake_editor,
    )
    assert np.array_equal(captured["mask"], predicted)
    assert not np.array_equal(captured["mask"], ground_truth)
    assert captured["action"] == "recolor"
    assert captured["parameters"] == {"color": "#F4A261"}


def test_runner_contains_no_prompt_search_or_selection_logic() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "select_global_prompt" not in source
    assert "PROMPT_VARIANT_ORDER" not in source
    assert "rotated_prompt_order" not in source
    assert "build_attempt_plan" not in source
    assert "prompt_order_position" not in source
