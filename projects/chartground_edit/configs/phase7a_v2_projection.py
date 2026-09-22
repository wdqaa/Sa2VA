"""Phase 7A: fixed synthetic_v2 projection-only Strategy-A training."""

_base_ = ["./phase5a_full_train.py"]

import os
import tempfile
from pathlib import Path


work_dir = os.environ.get(
    "CHARTGROUND_WORK_DIR",
    str(Path(tempfile.gettempdir()) / "chartground_edit_phase7a_v2_projection"),
)
experiment_name = "phase7a_v2_projection"
expected_train_samples = 960
training_epochs = 5
max_iters = 4800
checkpoint_steps = [960, 1920, 2880, 3840, 4800]
warmup_steps = 240
smoke_sample_id = "cgev2_line_category_1609c9161fd4"
smoke_expected_token_positions = None
smoke_optimizer_steps = 1

alignment_metadata = dict(
    protocol_version="phase7a-v2-projection-v1",
    dataset="synthetic_v2",
    manifest_sha256=(
        "1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be"
    ),
)
checkpoint_contract = dict(output_directory=work_dir)
model = dict(alignment_metadata=alignment_metadata)

train_dataset = dict(
    manifest_path="projects/chartground_edit/data/synthetic_v2/annotations.jsonl",
    selection_path=None,
    split="train",
    dataset_protocol="synthetic_v2",
)
train_dataloader = dict(
    batch_size=1,
    sampler=dict(shuffle=True),
    dataset=train_dataset,
)
optim_wrapper = dict(accumulative_counts=1)
train_cfg = dict(max_iters=max_iters)
val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
load_from = None
resume = False
