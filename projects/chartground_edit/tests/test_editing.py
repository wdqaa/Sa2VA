from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from chartground_edit.datasets.reader import ChartGroundDataset
from chartground_edit.datasets.synthetic import generate_dataset
from chartground_edit.editing import (
    EmptyMaskError,
    MaskValidationError,
    ParameterValidationError,
    edit,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "projects/chartground_edit/scripts/edit_with_mask.py"
ACTIONS = ("highlight", "recolor", "extract", "remove")
CHART_TYPES = ("line", "bar", "scatter", "confidence_band")


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> ChartGroundDataset:
    output_dir = tmp_path_factory.mktemp("editing_synthetic_v0")
    manifest = generate_dataset(output_dir, seed=20260915, clean=True)
    return ChartGroundDataset(manifest)


def test_size_mismatch_raises_clear_error() -> None:
    image, _ = _simple_pair()
    mask = Image.new("L", (image.width - 1, image.height), 255)
    with pytest.raises(MaskValidationError, match="size mismatch"):
        edit(image, mask, "recolor", {"color": "#E63946"})


def test_nonbinary_mask_is_rejected() -> None:
    image, mask = _simple_pair()
    array = np.asarray(mask).copy()
    array[0, 0] = 128
    with pytest.raises(MaskValidationError, match="only 0/1 or 0/255"):
        edit(image, array, "remove")


def test_recolor_keeps_every_outside_pixel_unchanged() -> None:
    image, mask = _simple_pair()
    output = edit(image, mask, "recolor", {"color": "#E63946"})
    _assert_outside_equal(image, output, mask)
    assert np.any(np.asarray(output)[np.asarray(mask) == 255] != np.asarray(image)[np.asarray(mask) == 255])


def test_extract_uses_mask_as_alpha_and_preserves_rgb() -> None:
    image, mask = _simple_pair()
    output = edit(image, mask, "extract")
    output_array = np.asarray(output)
    active = np.asarray(mask) == 255
    assert output.mode == "RGBA"
    assert output.size == image.size
    assert np.array_equal(output_array[..., :3], np.asarray(image))
    assert np.all(output_array[~active, 3] == 0)
    assert np.all(output_array[active, 3] == 255)


@pytest.mark.parametrize("fill_mode", ["color", "neighbor"])
def test_remove_keeps_every_outside_pixel_unchanged(fill_mode: str) -> None:
    image, mask = _simple_pair()
    output = edit(
        image,
        mask,
        "remove",
        {"fill_mode": fill_mode, "color": "#FAFAFA", "neighbor_radius": 3},
    )
    _assert_outside_equal(image, output, mask)


@pytest.mark.parametrize("action", ACTIONS)
def test_same_input_and_parameters_are_deterministic(action: str) -> None:
    image, mask = _simple_pair()
    parameters = _parameters(action)
    first = edit(image, mask, action, parameters)
    second = edit(image, mask, action, parameters)
    assert first.mode == second.mode
    assert np.array_equal(np.asarray(first), np.asarray(second))


@pytest.mark.parametrize("chart_type", CHART_TYPES)
@pytest.mark.parametrize("action", ACTIONS)
def test_all_actions_handle_all_chart_types(
    synthetic_dataset: ChartGroundDataset, chart_type: str, action: str
) -> None:
    sample = next(
        item for item in synthetic_dataset if item.annotation["chart_type"] == chart_type
    )
    output = edit(sample.image, sample.mask, action, _parameters(action))
    assert output.size == sample.image.size
    assert output.mode == ("RGBA" if action == "extract" else "RGB")


def test_cli_creates_output_file(synthetic_dataset: ChartGroundDataset, tmp_path: Path) -> None:
    annotation = synthetic_dataset.annotations[0]
    image_path = synthetic_dataset.root / annotation["image_path"]
    mask_path = synthetic_dataset.root / annotation["mask_path"]
    output_path = tmp_path / "edited.png"
    result = _run_cli(
        "--image",
        str(image_path),
        "--mask",
        str(mask_path),
        "--action",
        "recolor",
        "--output",
        str(output_path),
        "--color",
        "#E63946",
    )
    assert result.returncode == 0, result.stderr
    assert output_path.is_file()
    with Image.open(output_path) as output:
        assert output.mode == "RGB"
        assert output.size == (annotation["width"], annotation["height"])


def test_empty_mask_rejected_and_full_mask_supported() -> None:
    image, _ = _simple_pair()
    empty = Image.new("L", image.size, 0)
    with pytest.raises(EmptyMaskError, match="empty masks"):
        edit(image, empty, "highlight")

    full = Image.new("L", image.size, 255)
    highlighted = edit(image, full, "highlight", {"strength": 1.0})
    extracted = edit(image, full, "extract")
    removed = edit(image, full, "remove", {"fill_mode": "neighbor", "color": "#123456"})
    recolored = edit(image, full, "recolor", {"color": "#E63946"})
    assert np.array_equal(np.asarray(highlighted), np.asarray(image))
    assert np.all(np.asarray(extracted)[..., 3] == 255)
    assert np.all(np.asarray(removed) == np.asarray((18, 52, 86), dtype=np.uint8))
    assert recolored.size == image.size


def test_highlight_strength_and_output_range() -> None:
    image, mask = _simple_pair()
    no_op = edit(image, mask, "highlight", {"strength": 0.0})
    strong = edit(image, mask, "highlight", {"strength": 1.0})
    active = np.asarray(mask) == 255
    assert np.array_equal(np.asarray(no_op), np.asarray(image))
    assert np.array_equal(np.asarray(strong)[active], np.asarray(image)[active])
    assert np.any(np.asarray(strong)[~active] != np.asarray(image)[~active])
    assert np.asarray(strong).min() >= 0
    assert np.asarray(strong).max() <= 255
    with pytest.raises(ParameterValidationError, match=r"\[0, 1\]"):
        edit(image, mask, "highlight", {"strength": 1.01})


def test_cli_missing_input_has_understandable_error(tmp_path: Path) -> None:
    result = _run_cli(
        "--image",
        str(tmp_path / "missing.png"),
        "--mask",
        str(tmp_path / "also-missing.png"),
        "--action",
        "remove",
        "--output",
        str(tmp_path / "output.png"),
    )
    assert result.returncode != 0
    assert "file does not exist" in result.stderr


def test_cli_help_lists_required_controls() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    for option in (
        "--image",
        "--mask",
        "--action",
        "--output",
        "--color",
        "--highlight-strength",
        "--remove-fill-mode",
    ):
        assert option in result.stdout


def _simple_pair() -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (48, 36), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 8, 34, 27), fill=(40, 120, 180), outline=(20, 40, 60), width=2)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((12, 8, 34, 27), fill=255)
    return image, mask


def _parameters(action: str) -> dict[str, object]:
    return {
        "highlight": {"strength": 0.7},
        "recolor": {"color": "#E63946"},
        "extract": {},
        "remove": {"fill_mode": "neighbor", "color": "#FFFFFF", "neighbor_radius": 3},
    }[action]


def _assert_outside_equal(
    source: Image.Image, output: Image.Image, mask: Image.Image
) -> None:
    outside = np.asarray(mask) == 0
    assert np.array_equal(np.asarray(source)[outside], np.asarray(output)[outside])


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
