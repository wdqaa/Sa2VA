"""Phase 4B smoke1: parse-only in Phase 4B-0; do not build or run here."""

_base_ = []

from mmengine.dataset import DefaultSampler
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
)
from torch.optim import AdamW
from transformers import AutoTokenizer
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import CrossEntropyLoss, DiceLoss
from projects.sa2va.models import DirectResize, InternVLMLLM, SAM2TrainRunner

from projects.chartground_edit.chartground_edit.training.runtime import (
    ChartGroundStrategyAModel,
    FailFastAmpOptimWrapper,
    Phase4BContractHook,
)
from projects.chartground_edit.chartground_edit.training.sa2va_adapter import (
    ChartGroundPhase4Dataset,
    chartground_sa2va_collect_fn,
)


seed = 20260916
path = "{{$CHARTGROUND_BASE_MODEL_PATH:__REQUIRED_CHARTGROUND_BASE_MODEL_PATH__}}"
pretrained_pth = "{{$CHARTGROUND_SA2VA_PTH:__REQUIRED_CHARTGROUND_SA2VA_PTH__}}"
source_hf_revision = "{{$CHARTGROUND_SA2VA_HF_REVISION:15837dcaecc304714a1f0f069e74f47e47521c7f}}"

work_dir = "/tmp/chartground_edit_phase4b_smoke1"
batch_size = 1
accumulative_counts = 1
max_iters = 1
max_length = 8192
expected_trainable_parameter_count = 2754304
trainable_modules = ["text_hidden_fcs"]
frozen_modules = [
    "mllm.model.language_model",
    "mllm.model.vision_model",
    "mllm.model.mlp1",
    "grounding_encoder",
]
special_tokens = ["[SEG]", "<p>", "</p>", "<vp>", "</vp>"]
prompt_template = PROMPT_TEMPLATE.qwen_chat

alignment_metadata = dict(
    protocol_version="phase4b-alignment-v1",
    prompt_variant="target_only_zh",
    prompt_template_sha256=(
        "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806"
    ),
    seg_token_selection="supervised_labels",
    object_count_policy="strict_one_to_one",
    expected_masks_per_sample=1,
    ignore_index=-100,
    source_hf_revision=source_hf_revision,
    training_strategy="text_hidden_fcs_only",
)

checkpoint_contract = dict(
    input_format="Sa2VA training-format full state_dict .pth",
    input_env="CHARTGROUND_SA2VA_PTH",
    base_model_env="CHARTGROUND_BASE_MODEL_PATH",
    source_hf_revision=source_hf_revision,
    output_format="MMEngine .pth with text_hidden_fcs-only state_dict",
    output_directory=work_dir,
    reload_requires=(
        "base InternVL3-2B directory + converted full Sa2VA PTH + this config"
    ),
)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    local_files_only=True,
    padding_side="right",
)
extra_image_processor = dict(type=DirectResize, target_length=1024)

model = dict(
    type=ChartGroundStrategyAModel,
    expected_trainable_parameter_count=expected_trainable_parameter_count,
    alignment_metadata=alignment_metadata,
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    frozen_sam2_decoder=True,
    seg_token_selection="supervised_labels",
    object_count_policy="strict_one_to_one",
    expected_masks_per_sample=1,
    ignore_index=-100,
    mllm=dict(
        type=InternVLMLLM,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=None,
        visual_encoder_lora=None,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(type=SAM2TrainRunner),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction="mean",
        loss_weight=2.0,
    ),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction="mean",
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5,
    ),
)

train_dataset = dict(
    type=ChartGroundPhase4Dataset,
    manifest_path="projects/chartground_edit/data/synthetic_v1/annotations.jsonl",
    selection_path="projects/chartground_edit/configs/phase4_smoke1_ids.json",
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=max_length,
    arch_type="intern_vl",
    single_image_mode=False,
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(
        type=chartground_sa2va_collect_fn,
        ignore_index=-100,
        object_count_policy="strict_one_to_one",
        expected_masks_per_sample=1,
    ),
)

optim_wrapper = dict(
    type=FailFastAmpOptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=4e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=1.0, error_if_nonfinite=True),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)
param_scheduler = []
train_cfg = dict(type=TrainLoop, max_iters=max_iters)
val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

custom_hooks = [dict(type=Phase4BContractHook, priority="VERY_HIGH")]
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=1),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=1,
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
visualizer = None
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=seed, deterministic=True)
log_processor = dict(by_epoch=False)
