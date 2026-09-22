from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/render_phase8b_saved_visualizations.py"
FROZEN = Path("/tmp/chartground_edit_phase7c_v2_frozen_test")


def _module():
    spec = importlib.util.spec_from_file_location("phase8b_saved_visualizations", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tp_fp_fn_colors_are_exact() -> None:
    visual = _module()
    gt = np.array([[True, False], [True, False]])
    prediction = np.array([[True, True], [False, False]])
    pixels = np.asarray(visual.error_map(gt, prediction))
    assert tuple(pixels[0, 0]) == visual.ERROR_COLORS["TP"]
    assert tuple(pixels[0, 1]) == visual.ERROR_COLORS["FP"]
    assert tuple(pixels[1, 0]) == visual.ERROR_COLORS["FN"]
    assert tuple(pixels[1, 1]) == (247, 247, 247)


def test_selection_rules_are_fixed_and_tie_broken() -> None:
    visual = _module()
    product_rows = [
        {"chart_type": chart, "edit_action": action, "edit_execution_success": True,
         "sample_id": f"{chart}-low", "iou": 0.2}
        for chart, action in visual.PRODUCT_PAIRS
    ]
    product_rows += [
        {"chart_type": chart, "edit_action": action, "edit_execution_success": True,
         "sample_id": f"{chart}-high", "iou": 0.9}
        for chart, action in visual.PRODUCT_PAIRS
    ]
    assert all(row["iou"] == 0.9 for row in visual.select_product(product_rows))
    median_rows = [
        {"chart_type": "line", "referring_type": referring, "sample_id": f"{referring}-{index:02d}",
         "iou": float(index)}
        for referring in visual.REFERRING for index in range(20)
    ]
    assert [row["sample_id"] for row in visual.select_median(median_rows, "line")] == [
        f"{referring}-09" for referring in visual.REFERRING
    ]
    failures = [
        {"sample_id": str(index), "iou": float(index), "dice": float(index)}
        for index in reversed(range(10))
    ]
    assert [row["sample_id"] for row in visual.select_failures(failures)] == [str(i) for i in range(6)]


@pytest.mark.skipif(not FROZEN.is_dir(), reason="saved frozen-test masks are unavailable")
def test_frozen_inputs_and_displayed_samples_match_metrics() -> None:
    visual = _module()
    rows, annotations = visual.load_frozen_inputs(FROZEN)
    assert len(rows) == len(annotations) == 320
    selected = visual.select_product(rows)
    assert [(row["chart_type"], row["edit_action"]) for row in selected] == list(visual.PRODUCT_PAIRS)
    for row in selected:
        candidates = [
            item for item in rows
            if item["chart_type"] == row["chart_type"]
            and item["edit_action"] == row["edit_action"]
            and item["edit_execution_success"] is True
        ]
        assert row["iou"] == max(item["iou"] for item in candidates)
        assert visual._paths(annotations[row["sample_id"]], FROZEN)["edited"].is_file()
    for chart in visual.CHARTS:
        for row in visual.select_median(rows, chart):
            group = [
                item["iou"] for item in rows
                if item["chart_type"] == chart and item["referring_type"] == row["referring_type"]
            ]
            assert len(group) == 20
            assert abs(row["iou"] - statistics.median(group)) == min(
                abs(score - statistics.median(group)) for score in group
            )
    assert [row["sample_id"] for row in visual.select_failures(rows)] == [
        "cgev2_bar_appearance_9a7e3eff7828", "cgev2_bar_appearance_aa73797fde12",
        "cgev2_bar_category_2d20d46bdf7d", "cgev2_bar_category_f801c5a14d50",
        "cgev2_bar_trend_01e9d17f055d", "cgev2_confidence_band_appearance_f67d1cd324ab",
    ]


def test_saved_figures_and_readme_order() -> None:
    visual = _module()
    assets = PROJECT / "assets"
    names = ["chartground_edit_v2_demo.png", "phase7c_v2_test_ablation.png"]
    names += [f"phase7c_v2_eval_{chart}.png" for chart in visual.CHARTS]
    names += ["phase7c_v2_failure_cases.png", "phase7c_v2_final_gallery.png"]
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    positions = [readme.index(name) for name in names]
    assert positions == sorted(positions)
    assert "<details>" in readme and "Selected qualitative examples" in readme
    for name in names:
        with Image.open(assets / name) as figure:
            if name != "phase7c_v2_final_gallery.png" and name != "phase7c_v2_test_ablation.png":
                assert figure.width >= 2400
                assert figure.height >= 1800
                assert figure.info["dpi"][0] >= 179
