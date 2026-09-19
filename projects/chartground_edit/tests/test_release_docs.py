from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/chartground_edit"


def test_release_metrics_match_saved_summaries() -> None:
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    model_card = (PROJECT / "MODEL_CARD.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    zero = json.loads(
        (PROJECT / "results/phase3c_frozen_test_summary.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]
    tuned = json.loads(
        (PROJECT / "results/phase5b_finetuned_test_summary.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]
    expected = {
        f"{zero['group_macro_iou']:.6f}",
        f"{zero['group_macro_dice']:.6f}",
        f"{zero['empty_prediction_rate']:.4%}",
        f"{tuned['group_macro_iou']:.6f}",
        f"{tuned['group_macro_dice']:.6f}",
        f"{tuned['micro_iou']:.6f}",
        f"{tuned['micro_dice']:.6f}",
    }
    for value in expected:
        assert value in readme
    phase7c = json.loads(
        (PROJECT / "results/phase7c_v2_frozen_test_summary.json").read_text(
            encoding="utf-8"
        )
    )["states"]
    trainables = {
        "zero_shot": "0",
        "v1_step960": "2,754,304",
        "v2_strategy_a": "2,754,304",
        "v2_strategy_b": "3,999,488",
    }
    for state, trainable in trainables.items():
        metrics = phase7c[state]
        row_values = (
            trainable,
            f"{metrics['group_macro_iou']:.6f}",
            f"{metrics['group_macro_dice']:.6f}",
            f"{metrics['micro_iou']:.6f}",
            f"{metrics['empty_rate']:.4%}",
        )
        for document in (readme, model_card):
            assert all(value in document for value in row_values)
    assert f"{phase7c['zero_shot']['group_macro_iou']:.4f}" in root_readme
    assert f"{phase7c['v2_strategy_b']['group_macro_iou']:.4f}" in root_readme


def test_phase8_release_identity_limitations_and_gallery_order() -> None:
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    model_card = (PROJECT / "MODEL_CARD.md").read_text(encoding="utf-8")
    for value in (
        "c47ce4e38a9b1679c766d6360d66b6a3b69286d36b9d7cde50c70991ae475b97",
        "5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6",
        "1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be",
        "0.1726%",
        "256/320",
        "under-segmentation",
        "line/trend",
    ):
        assert value in model_card
    for value in (
        "+0.212465",
        "+0.062052",
        "[0.038416, 0.088712]",
        "315",
        "Selected qualitative examples",
        "不是生产级精度",
    ):
        assert value in readme
    gallery_order = (
        "assets/chartground_edit_demo_gallery.png",
        "assets/phase7c_v2_test_ablation.png",
        "assets/phase7c_v2_final_gallery.png",
        "assets/phase7c_v2_failure_cases.png",
    )
    positions = [readme.index(path) for path in gallery_order]
    assert positions == sorted(positions)
    assert "--adapter-checkpoint <CHARTGROUND_ADAPTER>" in readme
    assert "--image <INPUT_IMAGE>" in readme


def test_release_markdown_relative_links_exist() -> None:
    for document in (ROOT / "README.md", PROJECT / "README.md", PROJECT / "MODEL_CARD.md"):
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = (document.parent / target.split("#", 1)[0]).resolve()
            assert path.exists(), f"broken link in {document}: {target}"


def test_public_release_docs_do_not_contain_private_paths() -> None:
    for document in (ROOT / "README.md", PROJECT / "README.md", PROJECT / "MODEL_CARD.md"):
        text = document.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "/tmp/chartground" not in text
        assert "dqwang" not in text
