from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from chartground_edit.inference.prompt_benchmark_v1 import (
    BOOTSTRAP_ITERATIONS,
    EXPECTED_ATTEMPTS,
    build_attempt_plan,
    paired_bootstrap_group_differences,
    prompt_template_hashes,
    result_metric_semantics,
    rotated_prompt_order,
    select_gallery_sample_ids,
    select_global_prompt,
    summarize_benchmark,
    validate_balanced_val_annotations,
    validate_benchmark_split,
    validate_prompt_set,
)
from chartground_edit.inference.prompt_variants import (
    PROMPT_TEMPLATES,
    PROMPT_VARIANT_ORDER,
    build_prompt_variant,
)
from chartground_edit.inference.split_evaluation import SplitEvaluationError


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "projects/chartground_edit/scripts/run_prompt_benchmark_v1.py"
RESULTS_PATH = REPO_ROOT / (
    "projects/chartground_edit/results/phase3b_balanced_val_metrics.jsonl"
)
CHARTS = ("line", "bar", "scatter", "confidence_band")
REFERENCES = ("category", "appearance", "legend", "trend")
ACTIONS = ("highlight", "recolor", "extract", "remove")
VARIANTS = PROMPT_VARIANT_ORDER


def _annotations() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chart in CHARTS:
        for referring in REFERENCES:
            for action_index, action in enumerate(ACTIONS):
                rows.append(
                    {
                        "sample_id": f"{chart}_{referring}_{action}",
                        "split": "val",
                        "chart_type": chart,
                        "referring_type": referring,
                        "edit_action": action,
                        "difficulty": ("easy", "medium", "hard", "easy")[
                            action_index
                        ],
                        "distractor_count": (2, 3, 4, 2)[action_index],
                    }
                )
    return rows


def _results(
    *,
    iou_by_variant: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    ious = iou_by_variant or {
        "full_instruction": 0.3,
        "target_only_en": 0.2,
        "target_only_zh": 0.1,
    }
    rows: list[dict[str, object]] = []
    for annotation in _annotations():
        for variant in VARIANTS:
            iou = ious[variant]
            intersection = round(iou * 100)
            predicted = 100
            gt = 100
            union = predicted + gt - intersection
            actual_iou = intersection / union
            dice = 2 * intersection / (predicted + gt)
            rows.append(
                {
                    **annotation,
                    "prompt_variant": variant,
                    "inference_success": True,
                    "inference_success_legacy_deprecated": True,
                    "execution_success": True,
                    "mask_contract_valid": True,
                    "segmentation_token_present": True,
                    "nonempty_prediction": True,
                    "empty_prediction": False,
                    "overlapping_prediction": intersection > 0,
                    "nonempty_disjoint": intersection == 0,
                    "predicted_foreground_pixels": predicted,
                    "gt_foreground_pixels": gt,
                    "intersection_pixels": intersection,
                    "union_pixels": union,
                    "iou": actual_iou,
                    "dice": dice,
                    "inference_time_ms": 10.0,
                    "predicted_mask_sha256": f"{annotation['sample_id']}-{variant}",
                }
            )
    return rows


def test_cli_rejects_train_and_test_before_model_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    for split, phrase in (("train", "train is forbidden"), ("test", "test is frozen")):
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--split",
                split,
                "--expected-samples",
                "64",
                "--output-dir",
                str(tmp_path / "out"),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert phrase in result.stderr


def test_split_boundary_rejects_train_test_and_unknown() -> None:
    assert validate_benchmark_split("val") == "val"
    with pytest.raises(ValueError, match="train is forbidden"):
        validate_benchmark_split("train")
    with pytest.raises(ValueError, match="test is frozen"):
        validate_benchmark_split("test")
    with pytest.raises(ValueError, match="only permits"):
        validate_benchmark_split("validation")


def test_manifest_must_have_exactly_64_val_samples() -> None:
    with pytest.raises(SplitEvaluationError, match="expected 64"):
        validate_balanced_val_annotations(_annotations()[:-1])


def test_manifest_must_cover_all_combinations() -> None:
    rows = _annotations()
    removed = [
        row
        for row in rows
        if not (row["chart_type"] == "line" and row["referring_type"] == "category")
    ]
    replacement = [
        copy.deepcopy(row)
        for row in rows
        if row["chart_type"] == "bar" and row["referring_type"] == "category"
    ]
    for index, row in enumerate(replacement):
        row["sample_id"] = f"replacement_{index}"
    with pytest.raises(SplitEvaluationError, match="coverage mismatch"):
        validate_balanced_val_annotations(removed + replacement)


def test_each_combination_requires_all_four_actions() -> None:
    rows = _annotations()
    target = next(
        row
        for row in rows
        if row["chart_type"] == "line"
        and row["referring_type"] == "category"
        and row["edit_action"] == "remove"
    )
    target["edit_action"] = "highlight"
    with pytest.raises(SplitEvaluationError, match="each edit action once"):
        validate_balanced_val_annotations(rows)


def test_prompt_set_is_only_existing_p0_p1_p2() -> None:
    assert validate_prompt_set(VARIANTS) == VARIANTS
    with pytest.raises(ValueError, match="requires exactly"):
        validate_prompt_set(VARIANTS[:2])
    with pytest.raises(ValueError, match="requires exactly"):
        validate_prompt_set((*VARIANTS[:2], "P3"))


def test_prompt_text_and_hashes_match_phase2c_registry() -> None:
    instruction = "完整编辑指令"
    expression = "目标短语"
    assert build_prompt_variant(
        "full_instruction",
        instruction=instruction,
        referring_expression=expression,
    ) == PROMPT_TEMPLATES["full_instruction"].format(
        instruction=instruction, referring_expression=expression
    )
    hashes = prompt_template_hashes()
    assert hashes == {
        "variants": {
            "full_instruction": "175bf7d46db1ee62eb73a2111850c05d1a53659f142a3409dd711f3c2e14031a",
            "target_only_en": "5260f3469f0da8708e6df4c6461ae2fbb851dc668a9210946d863da7059008fc",
            "target_only_zh": "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806",
        },
        "registry_sha256": "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0",
    }


def test_rotating_order_and_positions_are_balanced() -> None:
    assert rotated_prompt_order(0) == VARIANTS
    assert rotated_prompt_order(1) == (VARIANTS[1], VARIANTS[2], VARIANTS[0])
    assert rotated_prompt_order(2) == (VARIANTS[2], VARIANTS[0], VARIANTS[1])
    assert rotated_prompt_order(3) == VARIANTS
    plan = build_attempt_plan(_annotations())
    assert len(plan) == EXPECTED_ATTEMPTS == 192
    for sample_index in range(64):
        attempts = plan[sample_index * 3 : sample_index * 3 + 3]
        assert [row["prompt_order_position"] for row in attempts] == [0, 1, 2]
        assert [row["prompt_variant"] for row in attempts] == list(
            rotated_prompt_order(sample_index)
        )
    for variant in VARIANTS:
        positions = Counter(
            row["prompt_order_position"]
            for row in plan
            if row["prompt_variant"] == variant
        )
        assert max(positions.values()) - min(positions.values()) <= 1


def test_aggregator_keeps_failures_as_zero() -> None:
    rows = _results()
    failed = rows[0]
    failed.update(
        {
            "inference_success": False,
            "execution_success": False,
            "mask_contract_valid": False,
            "segmentation_token_present": False,
            "nonempty_prediction": False,
            "empty_prediction": True,
            "overlapping_prediction": False,
            "nonempty_disjoint": False,
            "predicted_foreground_pixels": 0,
            "intersection_pixels": 0,
            "union_pixels": 100,
            "iou": 0.0,
            "dice": 0.0,
            "inference_time_ms": None,
        }
    )
    summary = summarize_benchmark(rows, bootstrap_iterations=50)
    p0 = summary["variant_summaries"]["full_instruction"]
    assert p0["attempt_count"] == 64
    assert p0["successful_inference_count"] == 63
    assert p0["execution_success_count"] == 63
    assert p0["empty_prediction_count"] == 1
    assert p0["sample_macro_iou"] == pytest.approx(
        sum(float(row["iou"]) for row in rows if row["prompt_variant"] == "full_instruction")
        / 64
    )


def test_empty_contract_valid_mask_is_not_an_execution_failure() -> None:
    semantics = result_metric_semantics(
        prediction_returned=True,
        failure_reason="empty_prediction_mask",
        error_type=None,
        mask_contract_valid=True,
        predicted_foreground_pixels=0,
        intersection_pixels=0,
    )
    assert semantics == {
        "execution_success": True,
        "mask_contract_valid": True,
        "nonempty_prediction": False,
        "empty_prediction": True,
        "overlapping_prediction": False,
    }


def test_runtime_exception_is_execution_failure() -> None:
    semantics = result_metric_semantics(
        prediction_returned=False,
        failure_reason="backend_exception:RuntimeError:boom",
        error_type="RuntimeError",
        mask_contract_valid=False,
        predicted_foreground_pixels=0,
        intersection_pixels=0,
    )
    assert semantics["execution_success"] is False
    assert semantics["mask_contract_valid"] is False


def test_invalid_mask_is_contract_failure_after_successful_execution() -> None:
    semantics = result_metric_semantics(
        prediction_returned=True,
        failure_reason="invalid_prediction_mask",
        error_type=None,
        mask_contract_valid=False,
        predicted_foreground_pixels=0,
        intersection_pixels=0,
    )
    assert semantics["execution_success"] is True
    assert semantics["mask_contract_valid"] is False


def test_nonempty_and_empty_rates_are_complementary() -> None:
    rows = _results()
    for index, row in enumerate(rows):
        if index % 4 == 0:
            row.update(
                {
                    "inference_success": False,
                    "nonempty_prediction": False,
                    "empty_prediction": True,
                    "overlapping_prediction": False,
                    "predicted_foreground_pixels": 0,
                    "intersection_pixels": 0,
                    "union_pixels": row["gt_foreground_pixels"],
                    "iou": 0.0,
                    "dice": 0.0,
                }
            )
    summary = summarize_benchmark(rows, bootstrap_iterations=50)
    for values in summary["variant_summaries"].values():
        assert values["execution_success_count"] == 64
        assert values["mask_contract_valid_count"] == 64
        assert values["nonempty_prediction_rate"] + values["empty_prediction_rate"] == 1.0


def test_frozen_metrics_and_p2_selection_are_unchanged() -> None:
    rows = [
        json.loads(line)
        for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_benchmark(rows, bootstrap_iterations=50)
    expected = {
        "full_instruction": (0.1645025837986922, 0.1927812516296302),
        "target_only_en": (0.17909270212571599, 0.21022560615742003),
        "target_only_zh": (0.18219856304270052, 0.21622086057617812),
    }
    for variant, (iou, dice) in expected.items():
        values = summary["variant_summaries"][variant]
        assert values["group_macro_iou"] == pytest.approx(iou)
        assert values["group_macro_dice"] == pytest.approx(dice)
    assert summary["selection"]["selected_prompt"] == "target_only_zh"


def test_group_macro_and_micro_iou_are_calculated_separately() -> None:
    rows = _results()
    summary = summarize_benchmark(rows, bootstrap_iterations=50)
    for variant in VARIANTS:
        metrics = summary["variant_summaries"][variant]
        assert len(metrics["groups"]["chart_referring"]) == 16
        assert metrics["group_macro_iou"] == pytest.approx(
            metrics["sample_macro_iou"]
        )
        expected_micro = metrics["total_intersection_pixels"] / metrics[
            "total_union_pixels"
        ]
        assert metrics["micro_iou"] == pytest.approx(expected_micro)


def test_paired_bootstrap_is_deterministic() -> None:
    summary = summarize_benchmark(_results(), bootstrap_iterations=200)
    variants = summary["variant_summaries"]
    first = paired_bootstrap_group_differences(
        variants,
        selected_variant="full_instruction",
        seed=123,
        iterations=200,
    )
    second = paired_bootstrap_group_differences(
        variants,
        selected_variant="full_instruction",
        seed=123,
        iterations=200,
    )
    assert first == second
    assert all(item["iterations"] == 200 for item in first.values())


def test_preregistered_selection_and_exact_tie_fallback() -> None:
    def metrics(iou: float, dice: float, disjoint: float) -> dict[str, float]:
        return {
            "group_macro_iou": iou,
            "group_macro_dice": dice,
            "nonempty_disjoint_rate": disjoint,
        }

    selected = select_global_prompt(
        {
            "full_instruction": metrics(0.2, 0.4, 0.2),
            "target_only_en": metrics(0.3, 0.3, 0.1),
            "target_only_zh": metrics(0.1, 0.8, 0.0),
        }
    )
    assert selected["selected_prompt"] == "target_only_en"
    tied = select_global_prompt(
        {variant: metrics(0.25, 0.4, 0.1) for variant in VARIANTS}
    )
    assert tied["selected_prompt"] == "full_instruction"
    assert tied["selection_rule_applied"] == "p0_or_canonical_final_tiebreak"


def test_gallery_selection_depends_only_on_manifest() -> None:
    annotations = _annotations()
    first = select_gallery_sample_ids(annotations)
    altered = copy.deepcopy(annotations)
    for index, row in enumerate(altered):
        row["model_iou"] = 1.0 if index % 2 else 0.0
    assert select_gallery_sample_ids(altered) == first
    assert len(first) == 16
    assert first[0] == min(
        row["sample_id"]
        for row in annotations
        if row["chart_type"] == "line" and row["referring_type"] == "category"
    )


def test_pure_benchmark_module_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import chartground_edit.inference.prompt_benchmark_v1; "
                "print('torch' in sys.modules); print('transformers' in sys.modules)"
            ),
        ],
        cwd=REPO_ROOT / "projects/chartground_edit",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["False", "False"]
