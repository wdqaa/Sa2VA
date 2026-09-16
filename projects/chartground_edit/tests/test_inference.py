from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chartground_edit.datasets.synthetic import generate_dataset
from chartground_edit.inference import (
    MaskProcessingError,
    PredictionResult,
    Sa2VAInternVL3Backend,
    build_prompt,
    dice_score,
    empty_prediction,
    inference_success,
    intersection_over_union,
    process_prediction_masks,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "projects/chartground_edit/scripts/run_sa2va_baseline.py"


def test_prediction_result_keeps_model_and_failure_fields() -> None:
    mask = np.asarray([[False, True]], dtype=bool)
    result = PredictionResult(
        mask=mask,
        raw_masks=[mask[None]],
        text_output="target [SEG]",
        model_name="example",
        checkpoint_path="/checkpoint",
        instruction="instruction",
        prompt="prompt",
        num_masks=1,
        inference_time_ms=10.0,
        model_load_time_ms=20.0,
        peak_gpu_memory_mb=30.0,
        success=True,
        failure_reason=None,
        raw_mask_shapes=[[1, 1, 2]],
        metadata={"source": "test"},
    )
    assert result.mask is mask
    assert result.num_masks == 1
    assert result.failure_reason is None
    assert result.metadata == {"source": "test"}


def test_prompt_template_preserves_instruction_exactly() -> None:
    instruction = "请将类别 B 对应的柱子改成红色  "
    assert build_prompt(instruction) == (
        "<image>Please segment the chart element targeted by this instruction: "
        f"{instruction}\nPlease respond with a segmentation mask."
    )
    with pytest.raises(ValueError, match="non-empty"):
        build_prompt("   ")


@pytest.mark.parametrize(
    "array",
    [
        np.asarray([[False, True]], dtype=bool),
        np.asarray([[0, 1]], dtype=np.uint8),
        np.asarray([[0, 255]], dtype=np.uint8),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    ],
)
def test_mask_binarization_accepts_documented_domains(array: np.ndarray) -> None:
    result = process_prediction_masks([array], image_size=(2, 1))
    assert result.failure_reason is None
    assert result.mask is not None
    assert result.mask.dtype == np.bool_
    assert result.mask.tolist() == [[False, True]]


@pytest.mark.parametrize(
    "array",
    [
        np.asarray([[0, 2]], dtype=np.uint8),
        np.asarray([[0.0, 0.5]], dtype=np.float32),
        np.asarray([[0.0, np.nan]], dtype=np.float32),
        np.asarray([["0", "1"]]),
    ],
)
def test_mask_binarization_rejects_invalid_values(array: np.ndarray) -> None:
    with pytest.raises(MaskProcessingError):
        process_prediction_masks([array], image_size=(2, 1))


def test_missing_and_empty_prediction_masks_have_distinct_failures() -> None:
    missing = process_prediction_masks(None, image_size=(4, 3))
    empty = process_prediction_masks([], image_size=(4, 3))
    assert missing.mask is None and missing.failure_reason == "missing_prediction_masks"
    assert empty.mask is None and empty.failure_reason == "empty_prediction_masks"


def test_multiple_masks_use_first_without_gt_selection() -> None:
    first = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
    second = np.asarray([[1, 1], [1, 1]], dtype=np.uint8)
    result = process_prediction_masks([first, second], image_size=(2, 2))
    assert result.num_masks == 2
    assert result.raw_mask_shapes == [[2, 2], [2, 2]]
    assert result.mask is not None
    assert np.array_equal(result.mask, first.astype(bool))


def test_leading_singletons_are_removed_but_ambiguous_shape_fails() -> None:
    source = np.asarray([[[[0, 1], [1, 0]]]], dtype=np.uint8)
    result = process_prediction_masks([source], image_size=(2, 2))
    assert result.mask is not None and result.mask.shape == (2, 2)
    with pytest.raises(MaskProcessingError, match="leading singleton"):
        process_prediction_masks(
            [np.zeros((2, 3, 4), dtype=np.uint8)], image_size=(4, 3)
        )


def test_resize_uses_nearest_and_preserves_binary_values() -> None:
    source = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    result = process_prediction_masks([source], image_size=(6, 4))
    assert result.resized
    assert result.mask is not None and result.mask.shape == (4, 6)
    assert set(np.unique(result.mask).tolist()) == {False, True}


def test_empty_foreground_is_preserved_as_failure() -> None:
    result = process_prediction_masks(
        [np.zeros((1, 3, 4), dtype=bool)], image_size=(4, 3)
    )
    assert result.mask is not None
    assert not result.mask.any()
    assert result.failure_reason == "empty_prediction_mask"


def test_iou_and_dice_boundary_cases_and_encodings() -> None:
    empty = np.zeros((2, 2), dtype=bool)
    one = np.asarray([[255, 0], [0, 0]], dtype=np.uint8)
    same_01 = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    disjoint = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
    assert intersection_over_union(empty, empty) == 1.0
    assert dice_score(empty, empty) == 1.0
    assert intersection_over_union(one, same_01) == 1.0
    assert dice_score(one, same_01) == 1.0
    assert intersection_over_union(empty, one) == 0.0
    assert dice_score(one, empty) == 0.0
    assert intersection_over_union(one, disjoint) == 0.0
    assert dice_score(one, disjoint) == 0.0


def test_metric_shape_and_value_validation() -> None:
    with pytest.raises(ValueError, match="size mismatch"):
        intersection_over_union(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(MaskProcessingError):
        dice_score(np.asarray([[0, 7]]), np.asarray([[0, 1]]))


def test_empty_prediction_and_inference_success() -> None:
    empty = np.zeros((2, 2), dtype=np.uint8)
    nonempty = np.asarray([[0, 255], [0, 0]], dtype=np.uint8)
    assert empty_prediction(None)
    assert empty_prediction(empty)
    assert not empty_prediction(nonempty)
    assert not inference_success(None)
    assert not inference_success(nonempty, failure_reason="model_failed")
    assert inference_success(nonempty)


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict_forward(self, **kwargs):
        self.calls += 1
        assert set(kwargs) == {"image", "text", "tokenizer"}
        return {
            "prediction": "Here is the target [SEG]",
            "prediction_masks": [
                np.asarray([[[0, 1], [0, 0]]], dtype=np.uint8),
                np.ones((1, 2, 2), dtype=bool),
            ],
        }


class _FakeBackend(Sa2VAInternVL3Backend):
    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__(checkpoint_path)
        self.load_calls = 0
        self.fake_model = _FakeModel()

    def _prepare_cuda_for_load(self) -> None:
        return None

    def _perform_load(self) -> None:
        self.load_calls += 1
        self._model = self.fake_model
        self._tokenizer = object()

    def _synchronize(self) -> None:
        return None

    def _peak_gpu_memory_mb(self) -> float:
        return 12.5


def test_backend_lazy_loads_once_and_records_upstream_contract(tmp_path: Path) -> None:
    backend = _FakeBackend(tmp_path)
    assert not backend.is_loaded
    image = Image.new("RGB", (2, 2), "white")
    first = backend.predict_mask(image, "target")
    second = backend.predict_mask(image, "target")
    assert backend.is_loaded
    assert backend.load_calls == 1
    assert backend.model_load_attempts == 1
    assert backend.fake_model.calls == 2
    assert first.success and second.success
    assert first.num_masks == 2
    assert first.raw_mask_shapes == [[1, 2, 2], [1, 2, 2]]
    assert first.metadata["selected_mask_index"] == 0
    assert first.mask is not None
    assert first.mask.tolist() == [[False, True], [False, False]]


def test_backend_structures_missing_seg_and_masks(tmp_path: Path) -> None:
    backend = _FakeBackend(tmp_path)
    backend.fake_model.predict_forward = lambda **kwargs: {
        "prediction": "No target found",
        "prediction_masks": [],
    }
    result = backend.predict_mask(Image.new("RGB", (2, 2)), "target")
    assert not result.success
    assert result.failure_reason == "missing_seg_token"
    assert result.num_masks == 0


def test_backend_structures_missing_prediction_masks_key(tmp_path: Path) -> None:
    backend = _FakeBackend(tmp_path)
    backend.fake_model.predict_forward = lambda **kwargs: {"prediction": "target [SEG]"}
    result = backend.predict_mask(Image.new("RGB", (2, 2)), "target")
    assert not result.success
    assert result.failure_reason == "missing_prediction_masks"


def test_backend_returns_structured_load_failure(tmp_path: Path) -> None:
    class FailingBackend(_FakeBackend):
        def _perform_load(self) -> None:
            self.load_calls += 1
            raise RuntimeError("synthetic load error")

        def _release_after_failure(self) -> None:
            self._model = None
            self._tokenizer = None
            self._loaded = False

    backend = FailingBackend(tmp_path)
    result = backend.predict_mask(Image.new("RGB", (2, 2)), "target")
    assert not result.success
    assert not backend.is_loaded
    assert result.failure_reason == "model_load_failed"
    assert result.metadata["stage"] == "model_load"
    assert "synthetic load error" in result.metadata["error"]


def test_backend_load_uses_frozen_huggingface_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    calls: dict[str, dict[str, object]] = {}

    class FakeModel:
        def eval(self):
            calls["eval"] = {}
            return self

        def to(self, device):
            calls["to"] = {"device": device}
            return self

    def fake_tokenizer(path, **kwargs):
        calls["tokenizer"] = {"path": path, **kwargs}
        return object()

    def fake_model(path, **kwargs):
        calls["model"] = {"path": path, **kwargs}
        return FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model)
    backend = Sa2VAInternVL3Backend(tmp_path)

    class FakeTorch:
        bfloat16 = object()

    monkeypatch.setattr(backend, "_import_torch", lambda: FakeTorch)
    backend._perform_load()
    assert calls["tokenizer"]["path"] == str(tmp_path)
    assert calls["tokenizer"]["local_files_only"] is True
    assert calls["tokenizer"]["trust_remote_code"] is True
    assert calls["model"]["path"] == str(tmp_path)
    assert calls["model"]["local_files_only"] is True
    assert calls["model"]["trust_remote_code"] is True
    assert calls["model"]["low_cpu_mem_usage"] is True
    assert calls["model"]["use_flash_attn"] is True
    assert calls["model"]["torch_dtype"] is FakeTorch.bfloat16
    assert "device_map" not in calls["model"]
    assert calls["to"] == {"device": "cuda:0"}


def test_cli_help_and_argument_validation() -> None:
    help_result = _run_cli("--help")
    assert help_result.returncode == 0
    for option in (
        "--checkpoint",
        "--manifest",
        "--sample-id",
        "--device",
        "--dtype",
        "--output-dir",
        "--edit-action",
        "--edit-color",
        "--skip-model-run",
    ):
        assert option in help_result.stdout


def test_cli_skip_model_run_validates_sample_without_loading(
    tmp_path: Path,
) -> None:
    manifest = generate_dataset(tmp_path / "synthetic", seed=20260915, clean=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "output"
    result = _run_cli(
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(manifest),
        "--sample-id",
        "cge_bar_category_01",
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--output-dir",
        str(output),
        "--edit-action",
        "recolor",
        "--edit-color",
        "#E63946",
        "--skip-model-run",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["sample_id"] == "cge_bar_category_01"
    assert payload["failure_reason"] == "model_run_skipped"
    assert payload["run_status"] == "skip_model_run_validated"
    assert payload["prompt"].startswith("<image>Please segment")
    assert (output / "prompt.txt").is_file()
    assert (output / "gt_mask.png").is_file()
    assert (output / "raw_mask_metadata.json").is_file()
    assert not (output / "predicted_mask.png").exists()


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
