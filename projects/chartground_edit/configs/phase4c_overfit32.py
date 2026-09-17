"""Phase 4C: fixed 32-sample projection-only overfit run."""

_base_ = ["./phase4b_smoke1.py"]

work_dir = "/tmp/chartground_edit_phase4c_overfit32"
experiment_name = "phase4c_overfit32"
expected_train_samples = 32
training_epochs = 10
phase4c_epochs = 10
max_iters = 320
checkpoint_steps = [32, 128, 320]
warmup_steps = 16
scheduler_name = "linear_warmup_cosine"
alignment_metadata = dict(protocol_version="phase4c-overfit32-v1")
checkpoint_contract = dict(output_directory=work_dir)
model = dict(alignment_metadata=alignment_metadata)

train_dataset = dict(
    selection_path="projects/chartground_edit/configs/phase4_overfit32_ids.json"
)

train_dataloader = dict(
    batch_size=1,
    sampler=dict(shuffle=True),
    dataset=train_dataset,
)
optim_wrapper = dict(
    accumulative_counts=1,
    optimizer=dict(
        lr=4e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)
train_cfg = dict(max_iters=max_iters)
load_from = None
resume = False
