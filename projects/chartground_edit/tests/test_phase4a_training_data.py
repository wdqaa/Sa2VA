from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from chartground_edit.inference.prompt_variants import TARGET_ONLY_ZH
from chartground_edit.training.data_adapter import (
    ASSISTANT_TARGET,
    AUDIT_ONLY_FIELDS,
    EXPECTED_MANIFEST_SHA256,
    Phase4TrainDataset,
    validate_training_output_dir,
)
from chartground_edit.training.subsets import audit_phase4_subsets, prepare_phase4_subsets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
MANIFEST = PROJECT_ROOT / "data" / "synthetic_v1" / "annotations.jsonl"
SMOKE = PROJECT_ROOT / "configs" / "phase4_smoke1_ids.json"
OVERFIT = PROJECT_ROOT / "configs" / "phase4_overfit32_ids.json"
PLAN = PROJECT_ROOT / "configs" / "phase4_strategy_a.json"


def _load_manifest() -> list[dict[str, object]]:
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]


def test_frozen_manifest_hash_and_selection_determinism(tmp_path: Path) -> None:
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    smoke_a = tmp_path / "a-smoke.json"
    overfit_a = tmp_path / "a-overfit.json"
    smoke_b = tmp_path / "b-smoke.json"
    overfit_b = tmp_path / "b-overfit.json"
    prepare_phase4_subsets(MANIFEST, smoke_a, overfit_a)
    prepare_phase4_subsets(MANIFEST, smoke_b, overfit_b)
    assert smoke_a.read_bytes() == smoke_b.read_bytes() == SMOKE.read_bytes()
    assert overfit_a.read_bytes() == overfit_b.read_bytes() == OVERFIT.read_bytes()


def test_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "annotations.jsonl"
    broken.write_bytes(MANIFEST.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_phase4_subsets(broken, tmp_path / "smoke.json", tmp_path / "overfit.json")


def test_smoke1_is_exactly_one_train_bar_category_sample() -> None:
    payload = json.loads(SMOKE.read_text(encoding="utf-8"))
    records = {record["sample_id"]: record for record in _load_manifest()}
    assert payload["sample_count"] == len(payload["sample_ids"]) == 1
    selected = records[payload["sample_ids"][0]]
    assert selected["split"] == "train"
    assert (selected["chart_type"], selected["referring_type"]) == ("bar", "category")


def test_overfit32_has_two_per_group_and_balanced_audit_fields() -> None:
    payload = json.loads(OVERFIT.read_text(encoding="utf-8"))
    records = {record["sample_id"]: record for record in _load_manifest()}
    ids = payload["sample_ids"]
    assert payload["sample_count"] == len(ids) == len(set(ids)) == 32
    selected = [records[sample_id] for sample_id in ids]
    assert {record["split"] for record in selected} == {"train"}
    group_counts = Counter(
        (record["chart_type"], record["referring_type"]) for record in selected
    )
    assert set(group_counts.values()) == {2}
    assert Counter(record["edit_action"] for record in selected) == {
        "highlight": 8,
        "recolor": 8,
        "extract": 8,
        "remove": 8,
    }
    assert Counter(record["difficulty"] for record in selected) == {
        "easy": 11,
        "medium": 10,
        "hard": 11,
    }


def test_selected_ids_and_scene_content_are_disjoint_from_val_test() -> None:
    records = _load_manifest()
    ids = set(json.loads(OVERFIT.read_text(encoding="utf-8"))["sample_ids"])
    train = [record for record in records if record["sample_id"] in ids]
    held_out = [record for record in records if record["split"] in {"val", "test"}]
    assert ids.isdisjoint(record["sample_id"] for record in held_out)
    for field in ("scene_id", "content_id"):
        assert {record[field] for record in train}.isdisjoint(
            record[field] for record in held_out
        )
    payload_text = OVERFIT.read_text(encoding="utf-8")
    assert '"scene_id"' not in payload_text
    assert '"content_id"' not in payload_text


def test_independent_subset_audit_passes() -> None:
    report = audit_phase4_subsets(MANIFEST, SMOKE, OVERFIT)
    assert report["passed"] is True
    assert report["failures"] == []
    assert report["smoke1_count"] == 1
    assert report["overfit32_count"] == 32
    assert set(report["chart_referring_counts"].values()) == {2}


@pytest.mark.parametrize(("selection", "count"), [(SMOKE, 1), (OVERFIT, 32)])
def test_adapter_contract_masks_prompts_and_seg_count(selection: Path, count: int) -> None:
    dataset = Phase4TrainDataset(MANIFEST, selection)
    assert len(dataset) == count
    for sample in dataset:
        assert sample.image.mode == "RGB"
        assert sample.image.size == (sample.image_width, sample.image_height)
        assert sample.mask.dtype == np.uint8
        assert sample.mask.shape == (1, sample.image_height, sample.image_width)
        assert set(np.unique(sample.mask).tolist()) == {0, 1}
        assert int(sample.mask.sum()) > 0
        assert sample.assistant_target == ASSISTANT_TARGET
        assert sample.assistant_target.count("[SEG]") == 1
        assert sample.prompt.startswith("<image>请分割图中")
        assert sample.prompt.count("[SEG]") == 1
        record = next(
            record
            for record in dataset.records
            if record["sample_id"] == sample.sample_id
        )
        assert record["referring_expression"] in sample.prompt
        for field in AUDIT_ONLY_FIELDS:
            value = record[field]
            if isinstance(value, str):
                assert (
                    value not in sample.prompt
                    or value in record["referring_expression"]
                )
        serialized_conversation = json.dumps(sample.conversation, ensure_ascii=False)
        assert record["full_instruction"] not in serialized_conversation
        assert json.dumps(record["edit_parameters"], ensure_ascii=False) not in serialized_conversation
        assert json.dumps(record["target_attributes"], ensure_ascii=False) not in serialized_conversation


def test_selection_with_nontrain_id_is_rejected(tmp_path: Path) -> None:
    held_out = next(record for record in _load_manifest() if record["split"] == "val")
    payload = json.loads(SMOKE.read_text(encoding="utf-8"))
    payload["sample_ids"] = [held_out["sample_id"]]
    selection = tmp_path / "bad.json"
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-train"):
        Phase4TrainDataset(MANIFEST, selection)


def test_strategy_config_is_explicit_and_safe() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["status"] == "phase4b0_static_validated_not_executed"
    assert plan["prompt_variant"] == TARGET_ONLY_ZH
    assert plan["trainable_modules"] == ["text_hidden_fcs"]
    assert set(plan["frozen_modules"]) == {
        "mllm.model.language_model",
        "mllm.model.vision_model",
        "mllm.model.mlp1",
        "grounding_encoder",
    }
    assert plan["phase4b_optimizer_steps"] == 1
    assert set(plan["prohibited_splits"]) == {"val", "test"}
    validate_training_output_dir(
        plan["output_dir"],
        repo_root=REPO_ROOT,
        data_root=PROJECT_ROOT / "data",
    )


def test_training_output_guard_rejects_repo_and_data() -> None:
    data_root = PROJECT_ROOT / "data"
    for unsafe in (REPO_ROOT, data_root, data_root / "synthetic_v1" / "runs"):
        with pytest.raises(ValueError, match="must not"):
            validate_training_output_dir(unsafe, repo_root=REPO_ROOT, data_root=data_root)


def test_subset_script_source_has_no_model_or_heldout_logic() -> None:
    script = PROJECT_ROOT / "scripts" / "prepare_phase4_overfit_subsets.py"
    module = PROJECT_ROOT / "chartground_edit" / "training" / "subsets.py"
    source = script.read_text(encoding="utf-8") + module.read_text(encoding="utf-8")
    assert "from_pretrained" not in source
    assert "predict_forward" not in source
    assert "Phase 3C" not in source
    assert "phase3c" not in source.lower()


def test_subset_cli_rejects_wrong_hash(tmp_path: Path) -> None:
    bad_manifest = tmp_path / "bad.jsonl"
    bad_manifest.write_text("{}\n", encoding="utf-8")
    script = PROJECT_ROOT / "scripts" / "prepare_phase4_overfit_subsets.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(bad_manifest),
            "--smoke-output",
            str(tmp_path / "smoke.json"),
            "--overfit-output",
            str(tmp_path / "overfit.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "SHA-256 mismatch" in result.stderr
