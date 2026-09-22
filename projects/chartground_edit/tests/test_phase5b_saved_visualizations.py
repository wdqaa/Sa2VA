from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chartground_edit.visualization.phase5b_saved import (
    BODY_FONT_SIZE,
    ERROR_COLORS,
    EVAL_SIZE,
    FIGURE_DPI,
    TITLE_FONT_SIZE,
    error_map,
    load_frozen_phase5b_records,
    render_all_phase5b_figures,
    select_evaluation_records,
    select_product_records,
)


FROZEN_OUTPUT = Path("/tmp/chartground_edit_phase5b_finetuned_test")


def test_error_map_color_contract() -> None:
    gt = Image.fromarray(np.asarray([[255, 0], [255, 0]], dtype=np.uint8))
    prediction = Image.fromarray(np.asarray([[255, 255], [0, 0]], dtype=np.uint8))
    pixels = np.asarray(error_map(gt, prediction))
    assert tuple(pixels[0, 0]) == ERROR_COLORS["true_positive"]
    assert tuple(pixels[0, 1]) == ERROR_COLORS["false_positive"]
    assert tuple(pixels[1, 0]) == ERROR_COLORS["false_negative"]
    assert tuple(pixels[1, 1]) == (246, 246, 246)


@pytest.mark.skipif(not FROZEN_OUTPUT.is_dir(), reason="frozen Phase 5B output unavailable")
def test_saved_selection_is_deterministic_and_score_independent() -> None:
    records = load_frozen_phase5b_records(FROZEN_OUTPUT)
    for chart in ("line", "bar", "scatter", "confidence_band"):
        selected = select_evaluation_records(records, chart)
        assert len(selected) == 4
        assert [row["referring_type"] for row in selected] == [
            "category", "appearance", "legend", "trend"
        ]
        for row in selected:
            group = [
                item["sample_id"]
                for item in records
                if item["chart_type"] == row["chart_type"]
                and item["referring_type"] == row["referring_type"]
            ]
            assert row["sample_id"] == min(group)
    assert [row["edit_action"] for row in select_product_records(records)] == [
        "highlight", "recolor", "extract", "remove"
    ]


@pytest.mark.skipif(not FROZEN_OUTPUT.is_dir(), reason="frozen Phase 5B output unavailable")
def test_saved_figures_are_high_resolution_and_readable(tmp_path: Path) -> None:
    report = render_all_phase5b_figures(FROZEN_OUTPUT, tmp_path)
    assert TITLE_FONT_SIZE >= 28
    assert BODY_FONT_SIZE >= 20
    assert FIGURE_DPI >= 180
    for key in ("eval_line", "eval_bar", "eval_scatter", "eval_confidence_band"):
        with Image.open(report["outputs"][key]) as image:
            assert image.size == EVAL_SIZE
            assert image.width >= 1800
            assert image.info["dpi"][0] >= 179
    for key in ("product", "failures"):
        with Image.open(report["outputs"][key]) as image:
            assert image.width >= 1800
