from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from chartground_edit.inference import (
    PredictionResult,
    SplitEvaluationError,
    binary_pixel_counts,
    gallery_status_label,
    run_prediction_sequence,
    select_split_samples,
    summarize_results,
    write_summary_files,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "projects/chartground_edit/scripts/run_sa2va_split.py"


def _sample(sample_id: str, split: str = "test") -> SimpleNamespace:
    return SimpleNamespace(
        annotation={
            "sample_id": sample_id,
            "split": split,
            "instruction": f"instruction for {sample_id}",
        },
        image=Image.new("RGB", (2, 2), "white"),
    )


def _prediction(
    instruction: str,
    *,
    mask: np.ndarray | None = None,
    success: bool = True,
    failure_reason: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PredictionResult:
    if mask is None and success:
        mask = np.asarray([[True, False], [False, False]])
    return PredictionResult(
        mask=mask,
        raw_masks=[mask[None]] if mask is not None else None,
        text_output="Sure, [SEG]" if success else None,
        model_name="fake",
        checkpoint_path="/fake",
        instruction=instruction,
        prompt=f"prompt: {instruction}",
        num_masks=1 if mask is not None else 0,
        inference_time_ms=2.0,
        model_load_time_ms=3.0,
        peak_gpu_memory_mb=4.0,
        success=success,
        failure_reason=failure_reason,
        raw_mask_shapes=[[1, 2, 2]] if mask is not None else [],
        metadata=metadata or {},
    )


class _FakeBackend:
    def __init__(self, predictions: list[PredictionResult]) -> None:
        self.predictions = list(predictions)
        self.calls: list[str] = []

    def predict_mask(self, image: Image.Image, instruction: str) -> PredictionResult:
        assert image.mode == "RGB"
        self.calls.append(instruction)
        return self.predictions.pop(0)


def test_split_filter_preserves_manifest_order() -> None:
    dataset = [_sample("train", "train"), _sample("b"), _sample("a")]
    selected = select_split_samples(dataset, "test", expected_count=2)
    assert [item.annotation["sample_id"] for item in selected] == ["b", "a"]


def test_split_filter_rejects_missing_and_expected_count_mismatch() -> None:
    dataset = [_sample("only")]
    with pytest.raises(SplitEvaluationError, match="contains no samples"):
        select_split_samples(dataset, "val")
    with pytest.raises(SplitEvaluationError, match="expected 4"):
        select_split_samples(dataset, "test", expected_count=4)


def test_one_backend_is_reused_once_per_sample_without_retries() -> None:
    samples = [_sample(str(index)) for index in range(4)]
    backend = _FakeBackend(
        [_prediction(item.annotation["instruction"]) for item in samples]
    )
    attempts = run_prediction_sequence(
        samples, backend, continue_on_sample_error=True
    )
    assert len(attempts) == 4
    assert backend.calls == [item.annotation["instruction"] for item in samples]
    assert all(attempt.prediction is not None for attempt in attempts)


def test_sample_failure_continues_when_requested() -> None:
    samples = [_sample(str(index)) for index in range(3)]
    predictions = [
        _prediction(
            samples[0].annotation["instruction"],
            mask=np.zeros((2, 2), dtype=bool),
            success=False,
            failure_reason="empty_prediction_mask",
        ),
        *[_prediction(item.annotation["instruction"]) for item in samples[1:]],
    ]
    backend = _FakeBackend(predictions)
    attempts = run_prediction_sequence(
        samples, backend, continue_on_sample_error=True
    )
    assert len(backend.calls) == 3
    assert attempts[0].prediction.failure_reason == "empty_prediction_mask"
    assert attempts[1].prediction.success


def test_global_error_stops_later_model_calls_and_records_remaining() -> None:
    samples = [_sample(str(index)) for index in range(4)]
    failed = _prediction(
        samples[0].annotation["instruction"],
        mask=None,
        success=False,
        failure_reason="model_load_failed",
    )
    backend = _FakeBackend([failed])
    attempts = run_prediction_sequence(
        samples, backend, continue_on_sample_error=True
    )
    assert len(backend.calls) == 1
    assert len(attempts) == 4
    assert attempts[0].prediction is failed
    assert all(
        item.not_run_reason == "not_run_after_global_error"
        for item in attempts[1:]
    )


def test_sample_error_stops_without_continue_flag() -> None:
    samples = [_sample(str(index)) for index in range(2)]
    failed = _prediction(
        samples[0].annotation["instruction"],
        mask=None,
        success=False,
        failure_reason="missing_prediction_masks",
    )
    backend = _FakeBackend([failed])
    attempts = run_prediction_sequence(
        samples, backend, continue_on_sample_error=False
    )
    assert len(backend.calls) == 1
    assert attempts[1].not_run_reason == "not_run_after_sample_error"


def test_binary_counts_and_macro_micro_include_empty_prediction() -> None:
    target = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    overlap = np.asarray([[255, 0], [0, 0]], dtype=np.uint8)
    empty = np.zeros((2, 2), dtype=bool)
    first = binary_pixel_counts(overlap, target)
    second = binary_pixel_counts(empty, target)
    rows = [
        {
            **first,
            "success": True,
            "empty_prediction": False,
            "nonempty_disjoint": False,
            "iou": 1.0,
            "dice": 1.0,
            "inference_time_ms": 2.0,
            "peak_gpu_memory_mb": 10.0,
            "num_masks": 1,
        },
        {
            **second,
            "success": False,
            "empty_prediction": True,
            "nonempty_disjoint": False,
            "iou": 0.0,
            "dice": 0.0,
            "inference_time_ms": 4.0,
            "peak_gpu_memory_mb": 11.0,
            "num_masks": 0,
        },
    ]
    summary = summarize_results(
        rows, model_load_time_ms=20.0, model_load_attempts=1
    )
    assert summary["sample_count"] == 2
    assert summary["mean_iou"] == 0.5
    assert summary["median_dice"] == 0.5
    assert summary["micro_iou"] == 0.5
    assert summary["micro_dice"] == pytest.approx(2 / 3)
    assert summary["empty_prediction_count"] == 1
    assert summary["inference_success_rate"] == 0.5
    assert summary["mean_inference_time_ms"] == 3.0


def test_nonempty_disjoint_and_multiple_mask_statistics() -> None:
    row = {
        "predicted_foreground_pixels": 1,
        "gt_foreground_pixels": 1,
        "intersection_pixels": 0,
        "union_pixels": 2,
        "success": True,
        "empty_prediction": False,
        "nonempty_disjoint": True,
        "iou": 0.0,
        "dice": 0.0,
        "inference_time_ms": 1.0,
        "peak_gpu_memory_mb": 5.0,
        "num_masks": 2,
    }
    summary = summarize_results(
        [row], model_load_time_ms=3.0, model_load_attempts=1
    )
    assert summary["nonempty_disjoint_count"] == 1
    assert summary["nonempty_disjoint_rate"] == 1.0
    assert summary["overlap_count"] == 0
    assert summary["multiple_mask_count"] == 1


def test_summary_serialization_writes_jsonl_json_and_csv(tmp_path: Path) -> None:
    rows = [{"sample_id": "a", "success": False}]
    summary = {"sample_count": 1, "null_metric_reasons": {}}
    jsonl_path, json_path, csv_path = write_summary_files(tmp_path, rows, summary)
    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["sample_id"] == "a"
    assert json.loads(json_path.read_text(encoding="utf-8")) == summary
    assert csv_path.read_text(encoding="utf-8").startswith("metric,value")


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {"empty_prediction": True, "failure_reason": "empty_prediction_mask"},
            "EMPTY (empty_prediction_mask)",
        ),
        (
            {"empty_prediction": False, "success": False, "failure_reason": "bad"},
            "FAILED (bad)",
        ),
        (
            {
                "empty_prediction": False,
                "success": True,
                "nonempty_disjoint": True,
            },
            "SUCCESS / NONEMPTY_DISJOINT",
        ),
    ],
)
def test_gallery_failure_status_data(row: dict[str, object], expected: str) -> None:
    assert gallery_status_label(row) == expected


def test_split_cli_help_lists_frozen_batch_arguments() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for option in (
        "--checkpoint",
        "--manifest",
        "--split",
        "--device",
        "--dtype",
        "--output-dir",
        "--expected-count",
        "--continue-on-sample-error",
    ):
        assert option in result.stdout
