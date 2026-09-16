from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chartground_edit.datasets.audit_v1 import audit_synthetic_v1
from chartground_edit.datasets.gallery_v1 import (
    create_synthetic_v1_gallery,
    select_gallery_records,
)
from chartground_edit.datasets.reader_v1 import ChartGroundV1Dataset
from chartground_edit.datasets.schema import validate_annotation
from chartground_edit.datasets.schema_v1 import (
    SchemaValidationError,
    validate_annotation_v1,
    validate_jsonl_v1,
)
from chartground_edit.datasets.synthetic_v1 import (
    ACTION_ORDER_V1,
    CHART_ORDER_V1,
    REFERRING_ORDER_V1,
    SPLIT_COUNTS_V1,
    dataset_fingerprint_v1,
    generate_synthetic_v1,
)


@pytest.fixture(scope="module")
def v1_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    first = tmp_path_factory.mktemp("synthetic_v1_first")
    second = tmp_path_factory.mktemp("synthetic_v1_second")
    manifest = generate_synthetic_v1(first, seed=20260916, clean=False)
    repeat_manifest = generate_synthetic_v1(second, seed=20260916, clean=False)
    return {
        "root": first,
        "manifest": manifest,
        "repeat_root": second,
        "repeat_manifest": repeat_manifest,
        "records": validate_jsonl_v1(manifest, expected_count=320),
    }


def test_v1_valid_records_and_reader(v1_dataset: dict[str, object]) -> None:
    manifest = Path(v1_dataset["manifest"])
    records = validate_jsonl_v1(manifest, check_files=True, expected_count=320)
    dataset = ChartGroundV1Dataset(manifest)
    assert len(dataset) == 320
    sample = dataset[0]
    assert sample.full_instruction == records[0]["full_instruction"]
    assert sample.referring_expression == records[0]["referring_expression"]
    assert sample.referring_expression in sample.full_instruction
    assert sample.image.size == sample.mask.size == (480, 320)


def test_v1_schema_required_enum_and_unknown_fields(
    v1_dataset: dict[str, object]
) -> None:
    record = dict(v1_dataset["records"][0])
    missing = dict(record)
    missing.pop("scene_id")
    with pytest.raises(SchemaValidationError, match="scene_id"):
        validate_annotation_v1(missing)
    invalid_enum = dict(record, difficulty="extreme")
    with pytest.raises(SchemaValidationError, match="difficulty"):
        validate_annotation_v1(invalid_enum)
    unknown = dict(record, inferred_prompt="forbidden")
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        validate_annotation_v1(unknown)


@pytest.mark.parametrize(
    ("action", "parameters"),
    [
        ("highlight", {"color": "#FFFFFF"}),
        ("recolor", {"color": "red"}),
        ("extract", {}),
        ("remove", {"fill_mode": "neighbor", "color": "#FFFFFF"}),
    ],
)
def test_v1_edit_action_parameters_joint_validation(
    v1_dataset: dict[str, object], action: str, parameters: dict[str, object]
) -> None:
    record = dict(v1_dataset["records"][0])
    record["edit_action"] = action
    record["edit_parameters"] = parameters
    with pytest.raises(SchemaValidationError, match="edit_parameters"):
        validate_annotation_v1(record)


def test_v0_schema_remains_backward_compatible() -> None:
    record = {
        "sample_id": "old",
        "image_path": "images/old.png",
        "mask_path": "masks/old.png",
        "width": 10,
        "height": 10,
        "chart_type": "line",
        "referring_type": "category",
        "instruction": "old instruction",
        "target_type": "curve",
        "target_attributes": {},
        "edit_action": "highlight",
        "split": "train",
        "generator_seed": 1,
        "metadata": {
            "schema_version": "chartground-edit-v0",
            "generator_version": "synthetic-v0",
            "source": "synthetic",
        },
    }
    validate_annotation(record, check_files=False)


def test_exact_split_combination_and_action_distribution(
    v1_dataset: dict[str, object]
) -> None:
    records = v1_dataset["records"]
    assert len(records) == 320
    assert Counter(item["split"] for item in records) == {
        "train": 192,
        "val": 64,
        "test": 64,
    }
    for chart in CHART_ORDER_V1:
        for referring in REFERRING_ORDER_V1:
            for split, expected_count in SPLIT_COUNTS_V1.items():
                subset = [
                    item
                    for item in records
                    if item["chart_type"] == chart
                    and item["referring_type"] == referring
                    and item["split"] == split
                ]
                assert len(subset) == expected_count
                assert Counter(item["edit_action"] for item in subset) == {
                    action: 3 if split == "train" else 1
                    for action in ACTION_ORDER_V1
                }
                assert {item["difficulty"] for item in subset} == {
                    "easy",
                    "medium",
                    "hard",
                }
    assert Counter(item["edit_action"] for item in records) == {
        action: 80 for action in ACTION_ORDER_V1
    }
    difficulty_counts = Counter(item["difficulty"] for item in records)
    assert max(difficulty_counts.values()) - min(difficulty_counts.values()) <= 1


def test_all_identity_fields_and_paths_are_unique(
    v1_dataset: dict[str, object]
) -> None:
    records = v1_dataset["records"]
    for field in (
        "sample_id",
        "image_path",
        "mask_path",
        "seed",
        "scene_id",
        "content_id",
    ):
        values = [item[field] for item in records]
        assert len(values) == len(set(values)), field


def test_style_and_template_families_are_split_disjoint(
    v1_dataset: dict[str, object]
) -> None:
    records = v1_dataset["records"]
    for field in ("style_family", "instruction_template_family"):
        families = {
            split: {item[field] for item in records if item["split"] == split}
            for split in ("train", "val", "test")
        }
        assert families["train"].isdisjoint(families["val"])
        assert families["train"].isdisjoint(families["test"])
        assert families["val"].isdisjoint(families["test"])


def test_masks_are_binary_nonempty_aligned_and_exclude_legend(
    v1_dataset: dict[str, object]
) -> None:
    root = Path(v1_dataset["root"])
    for record in v1_dataset["records"]:
        with Image.open(root / record["image_path"]) as image:
            image_size = image.size
        with Image.open(root / record["mask_path"]) as source:
            mask = np.asarray(source.convert("L"))
        assert image_size == (record["image_width"], record["image_height"])
        assert mask.shape == (image_size[1], image_size[0])
        assert set(np.unique(mask).tolist()).issubset({0, 255})
        assert 0 < np.count_nonzero(mask) < mask.size
        left, top, right, bottom = record["generation_metadata"]["legend_box"]
        assert not np.any(mask[top : bottom + 1, left : right + 1])


def test_chart_specific_mask_semantics(v1_dataset: dict[str, object]) -> None:
    root = Path(v1_dataset["root"])
    for record in v1_dataset["records"]:
        with Image.open(root / record["mask_path"]) as source:
            mask = np.asarray(source.convert("L")) == 255
        geometry = record["generation_metadata"]["target_geometry"]
        if record["chart_type"] in {"line", "scatter"}:
            assert all(mask[y, x] for x, y in geometry["marker_centers"])
        elif record["chart_type"] == "bar":
            x0, y0, x1, y1 = geometry["bar_box"]
            assert mask[y0 : y1 + 1, x0 : x1 + 1].all()
            assert int(mask.sum()) == (x1 - x0 + 1) * (y1 - y0 + 1)
        else:
            assert mask.mean() >= 0.01
            rows, columns = np.nonzero(mask)
            assert int(rows.max() - rows.min()) > 10


def test_referring_expression_uniquely_identifies_actual_entity(
    v1_dataset: dict[str, object]
) -> None:
    for record in v1_dataset["records"]:
        metadata = record["generation_metadata"]
        key = metadata["reference_key"]
        value = metadata["reference_value"]
        assert sum(
            entity[key] == value for entity in metadata["entity_descriptors"]
        ) == 1
        assert metadata["reference_match_count"] == 1
        assert record["distractor_count"] == metadata["entity_count"] - 1
        assert record["referring_expression"] in record["full_instruction"]


def test_repeated_generation_is_byte_identical(v1_dataset: dict[str, object]) -> None:
    assert dataset_fingerprint_v1(v1_dataset["root"]) == dataset_fingerprint_v1(
        v1_dataset["repeat_root"]
    )


def test_unsafe_clean_targets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe|does not contain"):
        generate_synthetic_v1(tmp_path / "ordinary_output", clean=True)
    with pytest.raises(ValueError, match="unsafe|does not contain"):
        generate_synthetic_v1(Path.cwd(), clean=True)


def test_independent_audit_passes_and_reports_expected_fields(
    v1_dataset: dict[str, object]
) -> None:
    report = audit_synthetic_v1(v1_dataset["manifest"])
    assert report["passed"] is True
    assert report["hard_failure_count"] == 0
    assert report["mask_audit"]["empty_count"] == 0
    assert report["mask_audit"]["full_count"] == 0
    assert report["mask_audit"]["dimension_error_count"] == 0
    assert report["mask_audit"]["non_binary_count"] == 0
    assert report["mask_audit"]["format_or_mode_error_count"] == 0
    assert report["mask_audit"]["semantic_error_count"] == 0
    assert report["content_hash_duplicate_groups"] == []
    assert report["exact_file_duplicates"] == {"images": [], "masks": []}
    assert report["family_leakage"] == {
        "style_family": {},
        "instruction_template_family": {},
    }
    assert report["near_duplicate_check"]["threshold"] == 0.01
    assert report["near_duplicate_check"]["is_hard_failure"] is False


def test_audit_cli_returns_nonzero_for_hard_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "broken.jsonl"
    manifest.write_text(json.dumps({"schema_version": "broken"}) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_synthetic_v1.py"
    result = subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "passed=false" in result.stdout


def test_gallery_selection_covers_all_dimensions(
    v1_dataset: dict[str, object], tmp_path: Path
) -> None:
    selected = select_gallery_records(v1_dataset["records"])
    assert len(selected) == 16
    assert {
        (item["chart_type"], item["referring_type"]) for item in selected
    } == {
        (chart, referring)
        for chart in CHART_ORDER_V1
        for referring in REFERRING_ORDER_V1
    }
    assert {item["edit_action"] for item in selected} == set(ACTION_ORDER_V1)
    assert {item["difficulty"] for item in selected} == {"easy", "medium", "hard"}
    gallery = create_synthetic_v1_gallery(
        v1_dataset["manifest"], tmp_path / "gallery.png"
    )
    with Image.open(gallery) as image:
        assert image.size == (1720, 820)
