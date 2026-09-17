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
        assert value in model_card
    assert f"{zero['group_macro_iou']:.4f}" in root_readme
    assert f"{tuned['group_macro_iou']:.4f}" in root_readme


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
