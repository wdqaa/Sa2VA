from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from chartground_edit.datasets.gallery_v2 import GALLERY_DPI_V2, GALLERY_SIZE_V2
from chartground_edit.datasets.render_v2 import render_scene_v2
from chartground_edit.datasets.schema_v1 import SCHEMA_VERSION_V1
from chartground_edit.datasets.schema_v2 import SCHEMA_VERSION_V2
from chartground_edit.datasets.synthetic_v2 import (
    ACTION_COUNTS_V2,
    ACTION_ORDER_V2,
    CHART_ORDER_V2,
    REFERRING_ORDER_V2,
    SPLIT_COUNTS_V2,
    SPLIT_ORDER_V2,
    build_generation_plan_v2,
)


def test_v2_exact_size_split_combination_and_action_balance() -> None:
    plan = build_generation_plan_v2()
    assert len(plan) == 1600
    assert Counter(row["split"] for row in plan) == {"train": 960, "val": 320, "test": 320}
    for chart in CHART_ORDER_V2:
        for referring in REFERRING_ORDER_V2:
            for split in SPLIT_ORDER_V2:
                subset = [row for row in plan if row["chart_type"] == chart and row["referring_type"] == referring and row["split"] == split]
                assert len(subset) == SPLIT_COUNTS_V2[split]
                assert Counter(row["edit_action"] for row in subset) == {
                    action: ACTION_COUNTS_V2[split] for action in ACTION_ORDER_V2
                }


def test_v2_difficulty_is_independent_of_distractor_count() -> None:
    by_distractor: dict[int, set[str]] = defaultdict(set)
    by_difficulty: dict[str, set[int]] = defaultdict(set)
    for row in build_generation_plan_v2():
        by_distractor[row["distractor_count"]].add(row["difficulty"])
        by_difficulty[row["difficulty"]].add(row["distractor_count"])
    assert set(by_distractor) == {1, 2, 3, 4, 5}
    assert all(values == {"easy", "medium", "hard"} for values in by_distractor.values())
    assert all(values == {1, 2, 3, 4, 5} for values in by_difficulty.values())


def test_v2_split_identity_style_and_templates_are_disjoint() -> None:
    plan = build_generation_plan_v2()
    assert len({row["seed"] for row in plan}) == 1600
    assert len({row["scene_id"] for row in plan}) == 1600
    for field in ("style_family", "instruction_template_family"):
        values = {split: {row[field] for row in plan if row["split"] == split} for split in SPLIT_ORDER_V2}
        assert values["train"].isdisjoint(values["val"])
        assert values["train"].isdisjoint(values["test"])
        assert values["val"].isdisjoint(values["test"])


def test_v2_render_is_deterministic_and_masks_have_chart_semantics() -> None:
    plan = build_generation_plan_v2()
    probes = [plan[index] for index in (0, 62, 197, 401, 536, 731, 803, 998, 1135, 1207, 1398, 1599)]
    for item in probes:
        first = render_scene_v2(item)
        second = render_scene_v2(item)
        first_image, first_mask, attributes, metadata, diversity, content_id = first
        second_image, second_mask, _, _, _, second_content_id = second
        assert first_image.tobytes() == second_image.tobytes()
        assert first_mask.tobytes() == second_mask.tobytes()
        assert content_id == second_content_id
        mask = np.asarray(first_mask)
        foreground = mask == 255
        assert first_image.size == first_mask.size == tuple(item["canvas_size"])
        assert set(np.unique(mask).tolist()).issubset({0, 255})
        assert 0 < foreground.sum() < foreground.size
        assert diversity["series_count"] == item["distractor_count"] + 1
        key, value = metadata["reference_key"], metadata["reference_value"]
        assert sum(entity[key] == value for entity in metadata["entity_descriptors"]) == 1
        for left, top, right, bottom in metadata["legend_glyph_boxes"]:
            assert not foreground[top : bottom + 1, left : right + 1].any()
        for x, y in metadata["target_geometry"].get("marker_centers", []):
            assert foreground[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3].any()
        if item["chart_type"] == "bar":
            assert len(metadata["target_geometry"]["bar_boxes"]) >= 4
        if item["chart_type"] == "confidence_band":
            assert foreground.mean() >= 0.004
        assert attributes["includes_legend_proxy"] is False


def test_v1_and_v2_protocols_remain_separate() -> None:
    assert SCHEMA_VERSION_V1 == "chartground-edit-v1"
    assert SCHEMA_VERSION_V2 == "chartground-edit-v2"
    assert SCHEMA_VERSION_V1 != SCHEMA_VERSION_V2


def test_versioned_v2_gallery_is_high_resolution_when_present() -> None:
    gallery = Path(__file__).resolve().parents[1] / "assets" / "synthetic_v2_gallery.png"
    assert gallery.is_file()
    with Image.open(gallery) as image:
        assert image.size == GALLERY_SIZE_V2
        assert image.width >= 1800 and image.height >= 1200
        assert image.info["dpi"][0] >= GALLERY_DPI_V2 - 1
