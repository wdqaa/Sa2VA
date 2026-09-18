"""Phase 7B: frozen small-LLM-LoRA Strategy-B ablation on synthetic_v2."""

_base_ = ["./phase7a_v2_projection.py"]

import os
import tempfile
from pathlib import Path

from peft import LoraConfig

from projects.chartground_edit.chartground_edit.training.strategy_b import (
    ChartGroundStrategyBModel,
    exact_lora_targets,
)


work_dir = os.environ.get(
    "CHARTGROUND_WORK_DIR",
    str(Path(tempfile.gettempdir()) / "chartground_edit_phase7b_v2_lora"),
)
experiment_name = "phase7b_v2_lora"
llm_total_decoder_layers = 28
llm_lora_layer_indices = list(range(20, 28))
llm_lora_target_modules = list(exact_lora_targets(llm_total_decoder_layers, 8))
llm_lora_rank = 16
llm_lora_alpha = 32
llm_lora_dropout = 0.05
expected_projection_parameter_count = 2754304
expected_lora_parameter_count = 1245184
expected_total_trainable_parameter_count = 3999488

alignment_metadata = dict(
    protocol_version="phase7b-v2-small-llm-lora-v1",
    dataset="synthetic_v2",
    manifest_sha256=(
        "1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be"
    ),
    training_strategy="text_hidden_fcs_plus_last8_attention_lora",
)
checkpoint_contract = dict(
    output_directory=work_dir,
    checkpoint_state="text_hidden_fcs_plus_llm_lora",
    tensor_count=68,
)

model = dict(
    type=ChartGroundStrategyBModel,
    expected_trainable_parameter_count=None,
    expected_projection_parameter_count=expected_projection_parameter_count,
    expected_lora_parameter_count=expected_lora_parameter_count,
    expected_total_trainable_parameter_count=expected_total_trainable_parameter_count,
    lora_target_modules=llm_lora_target_modules,
    alignment_metadata=alignment_metadata,
    mllm=dict(
        freeze_llm=True,
        freeze_visual_encoder=True,
        visual_encoder_lora=None,
        llm_lora=dict(
            _delete_=True,
            type=LoraConfig,
            r=llm_lora_rank,
            lora_alpha=llm_lora_alpha,
            lora_dropout=llm_lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=llm_lora_target_modules,
            modules_to_save=None,
        ),
    ),
)

projection_optimizer = dict(lr=4e-5, weight_decay=0.05)
lora_optimizer = dict(lr=1e-4, weight_decay=0.01)
optimizer_betas = (0.9, 0.999)

val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
load_from = None
resume = False
