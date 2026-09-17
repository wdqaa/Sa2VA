"""Phase 5A: strategy-A training on the complete synthetic_v1 train split."""

_base_ = ["./phase4c_overfit32.py"]

work_dir = "/tmp/chartground_edit_phase5a_full_train"
experiment_name = "phase5a_full_train"
expected_train_samples = 192
training_epochs = 10
max_iters = 1920
checkpoint_steps = [192, 576, 960, 1344, 1920]
warmup_steps = 96
alignment_metadata = dict(protocol_version="phase5a-full-train-v1")
checkpoint_contract = dict(output_directory=work_dir)
model = dict(alignment_metadata=alignment_metadata)

train_dataset = dict(selection_path=None, split="train")
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
val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
load_from = None
resume = False
