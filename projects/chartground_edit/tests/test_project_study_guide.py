"""Static checks for the data-flow code study guide; no model imports."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
GUIDE = PROJECT / "docs/PROJECT_STUDY_GUIDE_ZH.md"
SUMMARY = PROJECT / "results/phase7c_v2_frozen_test_summary.json"


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_guide_has_all_chapters_six_mapped_diagrams_and_readme_entry() -> None:
    guide = _guide()
    for index in range(22):
        assert re.search(rf"^## {index}\. ", guide, flags=re.MULTILINE)
    for heading in (
        "### 16.1 训练样本",
        "### 16.2 推理样本",
        "## 附录 A：逐站排错速查表",
        "## 附录 B：术语表",
    ):
        assert heading in guide
    assert guide.count("```mermaid\n") == 6
    for suffix in guide.split("```mermaid\n")[1:]:
        assert re.search(r"\n```\n\n\| 节点", suffix)
        diagram = suffix.split("\n```", 1)[0]
        lines = diagram.splitlines()
        assert lines[0] in {"flowchart TB", "flowchart LR"}
        assert len(lines) >= 5
        for line in lines[1:]:
            assert "-->" in line and line.count("[") == line.count("]")
    assert "docs/PROJECT_STUDY_GUIDE_ZH.md" in (PROJECT / "README.md").read_text(encoding="utf-8")


def test_every_guide_relative_link_resolves_and_no_private_path() -> None:
    guide = _guide()
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", guide)
    assert len(links) >= 100
    for target in links:
        assert not target.startswith(("/", "http:", "https:"))
        path = (GUIDE.parent / target.split("#", 1)[0]).resolve()
        assert path.is_file(), f"broken study-guide link: {target}"
        assert ROOT == path or ROOT in path.parents
    assert not re.search(r"/home/|/tmp/|hf_[A-Za-z0-9]{20,}|Bearer\s+\S+", guide)


@pytest.mark.parametrize(
    ("relative_path", "symbols"),
    [
        ("projects/chartground_edit/chartground_edit/datasets/schema_v2.py", ("validate_annotation_v2", "validate_jsonl_v2")),
        ("projects/chartground_edit/chartground_edit/datasets/synthetic_v2.py", ("build_generation_plan_v2", "generate_synthetic_v2")),
        ("projects/chartground_edit/chartground_edit/training/data_adapter.py", ("Phase7V2SplitDataset", "_load_sample")),
        ("projects/chartground_edit/chartground_edit/training/sa2va_adapter.py", ("ChartGroundPhase4Dataset", "chartground_sa2va_collect_fn")),
        ("projects/chartground_edit/chartground_edit/training/alignment.py", ("select_supervised_seg_tokens", "validate_strict_one_to_one")),
        ("projects/sa2va/datasets/base.py", ("Sa2VADatasetMixin", "_process_single_image", "get_inputid_labels")),
        ("projects/sa2va/datasets/data_utils.py", ("dynamic_preprocess", "tokenize_conversation", "sa2va_collect_fn")),
        ("projects/sa2va/models/sa2va.py", ("Sa2VAModel", "select_seg_token_mask", "validate_strict_alignment", "check_obj_number", "forward", "sample_points")),
        ("projects/sa2va/models/mllm/internvl.py", ("InternVLMLLM", "_llm_forward", "_embed_visual_features", "_compute_loss")),
        ("projects/sa2va/models/sam2_train.py", ("SAM2TrainRunner", "get_sam2_embeddings", "inject_language_embd")),
        ("projects/sa2va/hf/models/modeling_sa2va_chat.py", ("Sa2VAChatModel", "extract_feature", "predict_forward", "get_seg_hidden_states")),
        ("projects/sa2va/hf/models/sam2.py", ("SAM2", "language_embd_inference")),
        ("projects/chartground_edit/chartground_edit/inference/sa2va_backend.py", ("Sa2VAInternVL3Backend", "predict_prompt", "_perform_load")),
        ("projects/chartground_edit/chartground_edit/inference/mask_processing.py", ("process_prediction_masks", "normalize_binary_values")),
        ("projects/chartground_edit/chartground_edit/editing/editor.py", ("edit", "_highlight", "_recolor", "_extract", "_remove")),
        ("projects/chartground_edit/chartground_edit/training/strategy_b.py", ("ChartGroundStrategyBModel", "exact_lora_targets", "save_strategy_b_checkpoint", "load_strategy_b_checkpoint_into_hf_model")),
        ("projects/chartground_edit/scripts/run_phase7c_v2_frozen_test.py", ("_prediction_row", "summarize", "paired_group_bootstrap")),
    ],
)
def test_core_symbols_exist_in_the_linked_source(relative_path: str, symbols: tuple[str, ...]) -> None:
    source = ROOT / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    defined = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert set(symbols) <= defined, f"missing {set(symbols) - defined} in {relative_path}"


def test_frozen_metrics_quoted_in_guide_are_exact() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    guide = _guide()
    states = summary["states"]
    for name in ("zero_shot", "v2_strategy_a", "v2_strategy_b"):
        assert f'{states[name]["group_macro_iou"]:.6f}' in guide
    difference = summary["comparisons"]["b_minus_a"]["group_macro_iou"]
    ci = summary["paired_group_bootstrap"]["b_minus_a"]["iou"]["ci95"]
    assert f'+{difference:.6f}' in guide
    assert all(f"{value:.6f}" in guide for value in ci)
    assert "256/320" in guide
