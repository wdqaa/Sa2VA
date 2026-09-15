from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from chartground_edit.datasets.reader import ChartGroundDataset
from chartground_edit.datasets.schema import (
    SchemaValidationError,
    validate_annotation,
    validate_jsonl,
)
from chartground_edit.datasets.synthetic import dataset_fingerprint, generate_dataset
from chartground_edit.datasets.transforms import (
    apply_transform_params,
    sample_transform_params,
)


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("synthetic_v0")
    return generate_dataset(output_dir, seed=20260915, clean=True)


def test_all_32_annotations_pass_schema_and_files(generated_dataset: Path) -> None:
    annotations = validate_jsonl(
        generated_dataset, check_files=True, expected_count=32
    )
    assert len(annotations) == 32


def test_balanced_chart_and_referring_coverage(generated_dataset: Path) -> None:
    annotations = validate_jsonl(generated_dataset)
    combinations = Counter(
        (item["chart_type"], item["referring_type"]) for item in annotations
    )
    assert len(combinations) == 16
    assert set(combinations.values()) == {2}
    assert set(Counter(item["chart_type"] for item in annotations).values()) == {8}
    assert set(Counter(item["referring_type"] for item in annotations).values()) == {8}
    assert set(Counter(item["edit_action"] for item in annotations).values()) == {8}


def test_referenced_files_sizes_and_binary_nonempty_masks(
    generated_dataset: Path,
) -> None:
    for annotation in validate_jsonl(generated_dataset):
        image_path = generated_dataset.parent / annotation["image_path"]
        mask_path = generated_dataset.parent / annotation["mask_path"]
        assert image_path.is_file()
        assert mask_path.is_file()
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            assert image.size == mask.size == (
                annotation["width"],
                annotation["height"],
            )
            assert mask.mode == "L"
            values = set(np.unique(np.asarray(mask)).tolist())
            assert values.issubset({0, 255})
            assert 255 in values


def test_dataset_reader_loads_all_32_samples(generated_dataset: Path) -> None:
    dataset = ChartGroundDataset(generated_dataset)
    assert len(dataset) == 32
    loaded = list(dataset)
    assert len(loaded) == 32
    assert all(sample.image.mode == "RGB" for sample in loaded)
    assert all(sample.mask.mode == "L" for sample in loaded)
    assert all(sample.image.size == sample.mask.size for sample in loaded)


def test_same_seed_reproduces_annotations_and_masks(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, seed=7341)
    generate_dataset(second, seed=7341)
    assert dataset_fingerprint(first) == dataset_fingerprint(second)


def test_synchronized_resize_crop_flip_stays_aligned_and_binary() -> None:
    image = Image.new("RGB", (120, 90), "white")
    image_draw = ImageDraw.Draw(image)
    image_draw.rectangle((25, 20, 85, 65), fill="black")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((25, 20, 85, 65), fill=255)

    params = sample_transform_params(
        image.size,
        resize_size=(180, 135),
        crop_size=(140, 100),
        horizontal_flip_probability=1.0,
        seed=99,
    )
    transformed_image, transformed_mask = apply_transform_params(image, mask, params)

    assert transformed_image.size == transformed_mask.size == (140, 100)
    assert set(np.unique(np.asarray(transformed_mask)).tolist()).issubset({0, 255})
    image_foreground = np.asarray(transformed_image.convert("L")) < 128
    mask_foreground = np.asarray(transformed_mask) == 255
    assert image_foreground.any() and mask_foreground.any()
    image_bbox = _array_bbox(image_foreground)
    mask_bbox = _array_bbox(mask_foreground)
    assert max(abs(a - b) for a, b in zip(image_bbox, mask_bbox)) <= 1
    intersection = np.logical_and(image_foreground, mask_foreground).sum()
    union = np.logical_or(image_foreground, mask_foreground).sum()
    assert intersection / union > 0.97


def test_random_transform_parameters_are_reproducible() -> None:
    kwargs = dict(
        original_size=(100, 80),
        resize_size=(160, 128),
        crop_size=(120, 96),
        horizontal_flip_probability=0.5,
        seed=12345,
    )
    assert sample_transform_params(**kwargs) == sample_transform_params(**kwargs)


def test_schema_rejects_boolean_integer_and_unknown_field(
    generated_dataset: Path,
) -> None:
    annotation = validate_jsonl(generated_dataset, check_files=False)[0]
    invalid_boolean = dict(annotation, width=True)
    with pytest.raises(SchemaValidationError, match="width"):
        validate_annotation(invalid_boolean)
    invalid_unknown = dict(annotation, misspelled_field="value")
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        validate_annotation(invalid_unknown)


def test_visualization_artifacts_exist(generated_dataset: Path) -> None:
    output_dir = generated_dataset.parent
    assert (output_dir / "gallery.png").is_file()
    assert len(list((output_dir / "visualizations").glob("*.png"))) == 32


def _array_bbox(array: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(array)
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )
