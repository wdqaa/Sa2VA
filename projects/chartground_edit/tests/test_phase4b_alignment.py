from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config
from transformers import AutoTokenizer
from xtuner.utils import PROMPT_TEMPLATE

from chartground_edit.inference.prompt_variants import (
    PROMPT_TEMPLATES,
    TARGET_ONLY_ZH,
)
from chartground_edit.training.alignment import (
    select_supervised_seg_tokens,
    validate_strict_one_to_one,
)
from chartground_edit.training.data_adapter import ASSISTANT_TARGET, Phase4TrainDataset
from chartground_edit.training.sa2va_adapter import (
    ChartGroundPhase4Dataset,
    chartground_sa2va_collect_fn,
)
from chartground_edit.training.runtime import (
    ChartGroundStrategyAModel,
    Phase4BContractHook,
)
from projects.sa2va.models import DirectResize
from projects.sa2va.models.sa2va import Sa2VAModel, get_seg_hidden_states


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
MANIFEST = PROJECT_ROOT / "data" / "synthetic_v1" / "annotations.jsonl"
SMOKE = PROJECT_ROOT / "configs" / "phase4_smoke1_ids.json"
CONFIG = PROJECT_ROOT / "configs" / "phase4b_smoke1.py"
EXPECTED_P2_TEMPLATE = (
    "<image>请分割图中由以下指代表达式指定的图表元素：{referring_expression}\n"
    "请使用 [SEG] 标记返回分割掩码。"
)


def _model_contract(selection: str = "supervised_labels", policy: str = "strict_one_to_one"):
    model = object.__new__(Sa2VAModel)
    object.__setattr__(model, "seg_token_idx", 99)
    object.__setattr__(model, "seg_token_selection", selection)
    object.__setattr__(model, "object_count_policy", policy)
    object.__setattr__(model, "expected_masks_per_sample", 1 if policy == "strict_one_to_one" else None)
    object.__setattr__(model, "ignore_index", -100)
    return model


def _select(input_ids: list[list[int]], labels: list[list[int]]) -> torch.Tensor:
    return select_supervised_seg_tokens(
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        seg_token_idx=99,
    )


def test_two_seg_tokens_select_only_supervised_assistant() -> None:
    selected = _select([[1, 99, 2, 99, 3]], [[-100, -100, -100, 99, 3]])
    assert torch.nonzero(selected[0]).flatten().tolist() == [3]


@pytest.mark.parametrize(
    ("input_ids", "labels", "expected"),
    [
        ([99, 1, 99, 2], [-100, -100, 99, 2], [2]),
        ([1, 99, 2, 99], [1, 99, -100, -100], [1]),
        ([99, 99, 1, 99], [-100, -100, -100, 99], [3]),
    ],
)
def test_user_seg_order_and_multiplicity_do_not_affect_selection(
    input_ids: list[int], labels: list[int], expected: list[int]
) -> None:
    selected = _select([input_ids], [labels])
    assert torch.nonzero(selected[0]).flatten().tolist() == expected


def test_upstream_selection_is_labels_aware_and_validates_tensor_contract() -> None:
    model = _model_contract()
    input_ids = torch.tensor([[99, 1, 99]])
    labels = torch.tensor([[-100, -100, 99]])
    assert torch.nonzero(model.select_seg_token_mask(input_ids, labels)[0]).flatten().tolist() == [2]
    with pytest.raises(ValueError, match="shape must match"):
        model.select_seg_token_mask(input_ids, labels[:, :2])
    with pytest.raises(TypeError, match="both use torch.long"):
        model.select_seg_token_mask(input_ids, labels.to(torch.int32))


def test_invalid_alignment_configurations_fail_before_model_build() -> None:
    model = _model_contract(selection="all", policy="strict_one_to_one")
    with pytest.raises(ValueError, match="requires.*supervised_labels"):
        model._validate_alignment_config()
    object.__setattr__(model, "seg_token_selection", "supervised_labels")
    object.__setattr__(model, "expected_masks_per_sample", 2)
    with pytest.raises(ValueError, match="expected_masks_per_sample=1"):
        model._validate_alignment_config()


@pytest.mark.parametrize(
    ("input_ids", "labels", "mask_count", "pattern"),
    [
        ([1, 2], [-100, -100], 1, "supervised_seg_count=0"),
        ([99, 99], [99, 99], 1, "supervised_seg_count=2"),
        ([99], [99], 0, "gt_mask_count=0"),
        ([99], [99], 2, "gt_mask_count=2"),
    ],
)
def test_strict_alignment_fails_closed_without_count_repair(
    input_ids: list[int], labels: list[int], mask_count: int, pattern: str
) -> None:
    selected = _select([input_ids], [labels])
    masks = [torch.zeros((mask_count, 4, 4), dtype=torch.uint8)]
    with pytest.raises(ValueError, match=pattern) as exc_info:
        validate_strict_one_to_one(selected, masks, ["sample-x"])
    message = str(exc_info.value)
    assert "sample_id=sample-x" in message
    assert "token_positions=" in message
    assert "object_count_policy=strict_one_to_one" in message
    assert masks[0].shape[0] == mask_count


def test_batch_is_checked_per_sample() -> None:
    selected = _select(
        [[99, 1, 2], [1, 99, 99]],
        [[99, -100, -100], [-100, 99, 99]],
    )
    masks = [torch.ones((1, 2, 2), dtype=torch.uint8)] * 2
    with pytest.raises(ValueError, match="sample_id=second") as exc_info:
        validate_strict_one_to_one(selected, masks, ["first", "second"])
    assert "sample_id=first" not in str(exc_info.value)


def test_upstream_strict_error_contains_complete_alignment_context() -> None:
    model = _model_contract()
    selected = _select([[99, 99]], [[99, 99]])
    with pytest.raises(ValueError) as exc_info:
        model.validate_strict_alignment(
            selected,
            [torch.ones((1, 2, 2), dtype=torch.uint8)],
            ["upstream-sample"],
        )
    message = str(exc_info.value)
    assert "sample_id=upstream-sample" in message
    assert "supervised_seg_count=2" in message
    assert "gt_mask_count=1" in message
    assert "token_positions=[0, 1]" in message
    assert "object_count_policy=strict_one_to_one" in message


def test_strict_forward_branch_does_not_call_legacy_fix_number() -> None:
    source = inspect.getsource(Sa2VAModel.forward)
    assert "if self.object_count_policy == 'legacy_fix_number'" in source
    assert "self.check_obj_number" in source
    assert source.index("if self.object_count_policy == 'legacy_fix_number'") < source.index("self.check_obj_number")


def test_default_legacy_selection_and_fix_number_behavior_are_unchanged() -> None:
    model = _model_contract(selection="all", policy="legacy_fix_number")
    ids = torch.tensor([[99, 1, 99]])
    labels = torch.tensor([[-100, -100, 99]])
    assert torch.nonzero(model.select_seg_token_mask(ids, labels)[0]).flatten().tolist() == [0, 2]
    embeddings = [torch.tensor([[1.0], [2.0]])]
    masks = [torch.ones((1, 2, 2), dtype=torch.uint8)]
    fixed_embeddings, fixed_masks = model.check_obj_number(embeddings, masks)
    assert fixed_embeddings[0].shape[0] == fixed_masks[0].shape[0] == 5
    assert fixed_embeddings[0].flatten().tolist() == [1.0] * 5


def test_inference_seg_hidden_state_helper_is_unchanged() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    output_ids = torch.tensor([1, 99, 2, 99])
    result = get_seg_hidden_states(hidden, output_ids, seg_id=99)
    assert torch.equal(result, hidden[[1, 3]])


def test_strategy_a_checkpoint_contains_only_projection_and_alignment_metadata() -> None:
    model = object.__new__(ChartGroundStrategyAModel)
    nn.Module.__init__(model)
    model.frozen_backbone = nn.Linear(2, 2)
    model.text_hidden_fcs = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    model.requires_grad_(False)
    model.text_hidden_fcs.requires_grad_(True)
    model.expected_trainable_parameter_count = sum(
        parameter.numel() for parameter in model.text_hidden_fcs.parameters()
    )
    model.alignment_metadata = {
        "seg_token_selection": "supervised_labels",
        "object_count_policy": "strict_one_to_one",
    }
    assert set(model.state_dict()) == {
        "text_hidden_fcs.0.weight",
        "text_hidden_fcs.0.bias",
        "text_hidden_fcs.1.weight",
        "text_hidden_fcs.1.bias",
    }

    class Runner:
        pass

    runner = Runner()
    runner.model = model
    checkpoint: dict[str, object] = {}
    Phase4BContractHook().before_save_checkpoint(runner, checkpoint)
    metadata = checkpoint["meta"]["chartground_phase4b"]
    assert metadata["seg_token_selection"] == "supervised_labels"
    assert metadata["object_count_policy"] == "strict_one_to_one"
    assert metadata["checkpoint_state"] == "text_hidden_fcs_only"


@pytest.fixture(scope="module")
def real_smoke_batch():
    checkpoint = Path(
        os.environ.get(
            "CHARTGROUND_TOKENIZER_PATH",
            str(Path.home() / "Model" / "Sa2VA-InternVL3-2B"),
        )
    )
    if not (checkpoint / "tokenizer.json").is_file():
        pytest.skip("local Sa2VA tokenizer is unavailable")
    tokenizer_cfg = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=str(checkpoint),
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    dataset = ChartGroundPhase4Dataset(
        manifest_path=MANIFEST,
        selection_path=SMOKE,
        tokenizer=tokenizer_cfg,
        prompt_template=PROMPT_TEMPLATE.qwen_chat,
        special_tokens=["[SEG]", "<p>", "</p>", "<vp>", "</vp>"],
        extra_image_processor=dict(type=DirectResize, target_length=1024),
        max_length=8192,
        arch_type="intern_vl",
    )
    instance = dataset.prepare_data(0)
    return dataset, instance, chartground_sa2va_collect_fn([instance])


def test_real_smoke_collator_is_strict_one_to_one(real_smoke_batch) -> None:
    dataset, instance, batch = real_smoke_batch
    data = batch["data"]
    positions = torch.nonzero(
        data["input_ids"][0] == instance["seg_token_idx"], as_tuple=False
    ).flatten().tolist()
    assert len(positions) == 2
    assert data["labels"][0, positions[0]].item() == -100
    assert data["labels"][0, positions[1]].item() == instance["seg_token_idx"]
    assert data["alignment_records"] == [
        {
            "sample_id": "cgev1_bar_category_6d51bac154",
            "supervised_seg_count": 1,
            "gt_mask_count": 1,
            "token_positions": [positions[1]],
            "object_count_policy": "strict_one_to_one",
        }
    ]
    assert data["masks"][0].shape == (1, 320, 480)
    assert data["masks"][0].dtype == torch.uint8
    assert set(torch.unique(data["masks"][0]).tolist()) == {0, 1}
    assert torch.count_nonzero(data["masks"][0]) > 0
    assert instance["g_pixel_values"].shape == (3, 1024, 1024)
    assert dataset.source.records[0]["split"] == "train"


def test_real_adapter_model_boundary_excludes_audit_fields(real_smoke_batch) -> None:
    dataset, _, _ = real_smoke_batch
    sample = dataset.source[0]
    assert set(sample.__dataclass_fields__) == {
        "sample_id",
        "image",
        "mask",
        "image_width",
        "image_height",
        "prompt",
        "assistant_target",
    }
    assert sample.assistant_target == ASSISTANT_TARGET
    record = dataset.source.records[0]
    conversation = str(sample.conversation)
    assert record["full_instruction"] not in conversation
    assert str(record["edit_parameters"]) not in conversation
    assert str(record["target_attributes"]) not in conversation


def test_p2_template_and_hash_are_unchanged() -> None:
    assert PROMPT_TEMPLATES[TARGET_ONLY_ZH] == EXPECTED_P2_TEMPLATE
    assert hashlib.sha256(EXPECTED_P2_TEMPLATE.encode()).hexdigest() == (
        "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806"
    )
    sample = Phase4TrainDataset(MANIFEST, SMOKE)[0]
    assert sample.prompt == EXPECTED_P2_TEMPLATE.format(
        referring_expression=Phase4TrainDataset(MANIFEST, SMOKE).records[0][
            "referring_expression"
        ]
    )


def test_phase4b_config_is_one_step_train_only_strategy_a() -> None:
    cfg = Config.fromfile(CONFIG)
    assert cfg.train_cfg.max_iters == 1
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.optim_wrapper.accumulative_counts == 1
    assert cfg.optim_wrapper.dtype == "bfloat16"
    assert cfg.val_cfg is cfg.test_cfg is None
    assert cfg.val_dataloader is cfg.test_dataloader is None
    assert cfg.resume is False and cfg.load_from is None
    assert cfg.work_dir.startswith("/tmp/")
    assert cfg.randomness.seed == 20260916
    assert cfg.model.seg_token_selection == "supervised_labels"
    assert cfg.model.object_count_policy == "strict_one_to_one"
    assert cfg.model.expected_masks_per_sample == 1
    assert cfg.model.frozen_sam2_decoder is True
    assert cfg.model.mllm.freeze_llm is True
    assert cfg.model.mllm.freeze_visual_encoder is True
    assert cfg.model.mllm.llm_lora is None
    assert cfg.model.expected_trainable_parameter_count == 2754304
    assert cfg.trainable_modules == ["text_hidden_fcs"]
    assert set(cfg.frozen_modules) == {
        "mllm.model.language_model",
        "mllm.model.vision_model",
        "mllm.model.mlp1",
        "grounding_encoder",
    }
    assert cfg.train_dataset.selection_path.endswith("phase4_smoke1_ids.json")
    assert "val" not in cfg.train_dataset.selection_path
    assert "test" not in cfg.train_dataset.selection_path
