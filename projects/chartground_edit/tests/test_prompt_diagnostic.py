from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from chartground_edit.datasets.synthetic import _instruction, _split
from chartground_edit.inference import (
    PROMPT_VARIANT_ORDER,
    InstructionProcessingError,
    audit_split_distribution,
    build_paired_results,
    build_prompt_variant,
    extract_synthetic_v0_instruction,
    organize_gallery_rows,
    parse_prompt_variants,
    summarize_prompt_variants,
    validate_diagnostic_split,
    validate_synthetic_v1_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "projects/chartground_edit/scripts/run_prompt_diagnostic.py"
CHARTS = ("line", "bar", "scatter", "confidence_band")
REFERENCES = ("category", "appearance", "legend", "trend")
ACTIONS = ("highlight", "recolor", "extract", "remove")


def _synthetic_annotations() -> list[dict[str, object]]:
    annotations: list[dict[str, object]] = []
    index = 0
    for chart in CHARTS:
        for reference in REFERENCES:
            for repetition in range(2):
                action = ACTIONS[index % 4]
                attributes = {
                    "category": f"category_{repetition}",
                    "color": "#2878b5",
                    "series_name": f"series_{repetition}",
                    "trend": "increasing",
                }
                annotations.append(
                    {
                        "sample_id": f"{chart}_{reference}_{repetition}",
                        "chart_type": chart,
                        "referring_type": reference,
                        "edit_action": action,
                        "instruction": _instruction(
                            chart, reference, action, attributes
                        ),
                        "target_attributes": {"hidden": "must-not-be-read"},
                        "split": _split(index),
                    }
                )
                index += 1
    return annotations


def _result(
    sample_id: str,
    variant: str,
    *,
    iou: float,
    dice: float,
    predicted: int,
    intersection: int,
    success: bool = True,
    empty: bool = False,
    text: str | None = "Sure [SEG]",
) -> dict[str, object]:
    target = 10
    return {
        "sample_id": sample_id,
        "chart_type": "line",
        "referring_type": "category",
        "prompt_variant": variant,
        "success": success,
        "text_output": text,
        "empty_prediction": empty,
        "nonempty_disjoint": bool(not empty and intersection == 0),
        "predicted_foreground_pixels": predicted,
        "gt_foreground_pixels": target,
        "intersection_pixels": intersection,
        "union_pixels": predicted + target - intersection,
        "iou": iou,
        "dice": dice,
        "inference_time_ms": 2.0,
        "peak_gpu_memory_mb": 4.0,
        "num_masks": 1 if predicted else 0,
    }


def test_split_cross_statistics_expose_generation_order_bias() -> None:
    audit = audit_split_distribution(_synthetic_annotations())
    assert audit["sample_count"] == 32
    assert audit["split_counts"] == {"train": 24, "val": 4, "test": 4}
    assert audit["split_chart_type"]["train"] == {
        "line": 6,
        "bar": 6,
        "scatter": 6,
        "confidence_band": 6,
    }
    assert audit["split_referring_type"]["train"] == {
        "category": 0,
        "appearance": 8,
        "legend": 8,
        "trend": 8,
    }
    assert audit["split_referring_type"]["val"] == {
        "category": 4,
        "appearance": 0,
        "legend": 0,
        "trend": 0,
    }
    assert audit["split_referring_type"]["test"] == audit[
        "split_referring_type"
    ]["val"]
    assert audit["split_edit_action"]["val"]["highlight"] == 4
    assert audit["split_edit_action"]["test"]["recolor"] == 4
    assert len(audit["missing_chart_referring"]["train"]) == 4
    assert len(audit["missing_chart_referring"]["val"]) == 12
    assert len(audit["missing_chart_referring"]["test"]) == 12


def test_test_split_is_explicitly_frozen_for_prompt_diagnosis() -> None:
    assert validate_diagnostic_split("val") == "val"
    with pytest.raises(ValueError, match="test split is frozen"):
        validate_diagnostic_split("test")
    with pytest.raises(ValueError, match="only permits"):
        validate_diagnostic_split("train")


def test_referring_expression_extraction_covers_registered_templates() -> None:
    for annotation in _synthetic_annotations():
        parts = extract_synthetic_v0_instruction(annotation)
        assert parts.referring_expression in parts.original_instruction
        assert parts.original_instruction.startswith("请将" + parts.referring_expression)
        assert parts.edit_action == annotation["edit_action"]
        assert "synthetic_v0" in parts.extraction_rule


def test_extraction_does_not_read_hidden_target_attributes() -> None:
    annotation = _synthetic_annotations()[0]
    first = extract_synthetic_v0_instruction(annotation)
    annotation["target_attributes"] = {
        "category": "SECRET",
        "color": "#ffffff",
        "series_name": "SECRET",
    }
    second = extract_synthetic_v0_instruction(annotation)
    assert first == second
    assert "SECRET" not in second.referring_expression


def test_unregistered_or_mismatched_instruction_fails_instead_of_guessing() -> None:
    annotation = _synthetic_annotations()[0]
    annotation["instruction"] = "把这个东西处理一下"
    with pytest.raises(InstructionProcessingError, match="does not match"):
        extract_synthetic_v0_instruction(annotation)


def test_three_prompt_variants_are_exact_strings() -> None:
    instruction = "请将类别 B 对应的柱子改成红色"
    expression = "类别 B 对应的柱子"
    assert build_prompt_variant(
        "full_instruction", instruction=instruction, referring_expression=expression
    ) == (
        "<image>Please segment the chart element targeted by this instruction: "
        f"{instruction}\nPlease respond with a segmentation mask."
    )
    assert build_prompt_variant(
        "target_only_en", instruction=instruction, referring_expression=expression
    ) == (
        "<image>Please segment the chart element described by this referring expression: "
        f"{expression}\nPlease respond with a segmentation mask."
    )
    assert build_prompt_variant(
        "target_only_zh", instruction=instruction, referring_expression=expression
    ) == (
        f"<image>请分割图中由以下指代表达式指定的图表元素：{expression}\n"
        "请使用 [SEG] 标记返回分割掩码。"
    )


def test_prompt_variant_order_is_canonical_and_exactly_three() -> None:
    parsed = parse_prompt_variants(
        "target_only_zh,full_instruction,target_only_en"
    )
    assert parsed == PROMPT_VARIANT_ORDER
    with pytest.raises(ValueError, match="requires exactly"):
        parse_prompt_variants("full_instruction,target_only_en")
    with pytest.raises(ValueError, match="unknown"):
        parse_prompt_variants("full_instruction,target_only_en,other")


def test_prompt_aggregate_metrics_include_empty_and_seg_output() -> None:
    rows: list[dict[str, object]] = []
    for variant in PROMPT_VARIANT_ORDER:
        rows.extend(
            (
                _result("a", variant, iou=1.0, dice=1.0, predicted=10, intersection=10),
                _result(
                    "b",
                    variant,
                    iou=0.0,
                    dice=0.0,
                    predicted=0,
                    intersection=0,
                    success=False,
                    empty=True,
                    text=None,
                ),
            )
        )
    summaries = summarize_prompt_variants(rows)
    for summary in summaries.values():
        assert summary["sample_count"] == 2
        assert summary["inference_success_rate"] == 0.5
        assert summary["seg_output_rate"] == 0.5
        assert summary["empty_prediction_rate"] == 0.5
        assert summary["mean_iou"] == 0.5
        assert summary["micro_iou"] == 0.5


def test_paired_results_capture_foreground_and_state_transitions() -> None:
    rows = [
        _result("a", "full_instruction", iou=0.0, dice=0.0, predicted=5, intersection=0),
        _result("a", "target_only_en", iou=0.2, dice=0.3, predicted=7, intersection=2),
        _result(
            "a",
            "target_only_zh",
            iou=0.0,
            dice=0.0,
            predicted=0,
            intersection=0,
            success=False,
            empty=True,
            text=None,
        ),
    ]
    pair = build_paired_results(rows)[0]
    assert pair["variants"]["target_only_en"]["iou"] == 0.2
    transitions = pair["transitions_from_full_instruction"]
    assert transitions["target_only_en"]["disjoint_to_overlap"]
    assert transitions["target_only_en"]["foreground_pixel_delta"] == 2
    assert transitions["target_only_zh"]["mask_to_empty"]
    assert transitions["target_only_zh"]["seg_to_no_seg"]


def test_gallery_rows_are_sample_major_and_variant_complete() -> None:
    rows = [
        _result(sample, variant, iou=0.0, dice=0.0, predicted=1, intersection=0)
        for sample in ("b", "a")
        for variant in PROMPT_VARIANT_ORDER
    ]
    organized = organize_gallery_rows(rows)
    assert [sample_id for sample_id, _ in organized] == ["b", "a"]
    assert [row["prompt_variant"] for row in organized[0][1]] == list(
        PROMPT_VARIANT_ORDER
    )


def test_synthetic_v1_balanced_plan_is_320_with_all_split_combinations() -> None:
    plan = validate_synthetic_v1_plan()
    assert plan["combination_count"] == 16
    assert plan["per_combination"] == 20
    assert plan["split_per_combination"] == {"train": 12, "val": 4, "test": 4}
    assert plan["split_totals"] == {"train": 192, "val": 64, "test": 64}
    assert plan["total"] == 320
    with pytest.raises(ValueError, match="at least 20"):
        validate_synthetic_v1_plan(
            split_per_combination={"train": 10, "val": 2, "test": 2}
        )


def test_prompt_diagnostic_cli_help_and_test_rejection(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    for option in (
        "--checkpoint",
        "--manifest",
        "--split",
        "--prompt-variants",
        "--device",
        "--dtype",
        "--output-dir",
        "--expected-count",
    ):
        assert option in help_result.stdout

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--split",
            "test",
            "--prompt-variants",
            ",".join(PROMPT_VARIANT_ORDER),
            "--output-dir",
            str(tmp_path / "output"),
            "--expected-count",
            "4",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "test split is frozen" in rejected.stderr
