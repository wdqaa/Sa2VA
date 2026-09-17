#!/usr/bin/env python3
"""Run the fixed 320-step Phase 4C projection-only training experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config
from xtuner.registry import BUILDER

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "projects" / "chartground_edit"
for root in (REPO_ROOT, PACKAGE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from chartground_edit.training.overfit32 import build_epoch_schedule
from chartground_edit.training.runtime import save_projection_checkpoint
from chartground_edit.training.sa2va_adapter import chartground_sa2va_collect_fn


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--full-pth", required=True, type=Path)
    parser.add_argument("--full-pth-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-repo-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--sa2va-hf-revision", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync() -> None:
    torch.cuda.synchronize()


def _initialize(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29548")
        dist.init_process_group("nccl", rank=0, world_size=1)


def _gradient_stats(model) -> tuple[float, int, list[str]]:
    squared = torch.zeros((), device="cuda", dtype=torch.float64)
    nonzero = 0
    frozen_with_grad = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if not parameter.requires_grad:
            if gradient is not None:
                frozen_with_grad.append(name)
            continue
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"non-finite gradient: {name}")
        squared += gradient.detach().double().pow(2).sum()
        nonzero += int(torch.count_nonzero(gradient).item() > 0)
    return float(squared.sqrt().item()), nonzero, frozen_with_grad


def _lr_factor(step_index: int, warmup: int, total: int) -> float:
    if step_index < warmup:
        return float(step_index + 1) / warmup
    progress = (step_index - warmup) / (total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if _sha256(args.full_pth) != args.full_pth_sha256:
        raise ValueError("full PTH SHA-256 mismatch")
    os.environ["CHARTGROUND_BASE_MODEL_PATH"] = str(args.base_model)
    os.environ["CHARTGROUND_SA2VA_PTH"] = str(args.full_pth)
    os.environ["CHARTGROUND_SA2VA_HF_REVISION"] = args.sa2va_hf_revision
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = Config.fromfile(args.config)
    if (
        cfg.max_iters != 320
        or cfg.phase4c_epochs != 10
        or cfg.train_dataloader.batch_size != 1
        or cfg.optim_wrapper.accumulative_counts != 1
    ):
        raise ValueError("invalid fixed Phase 4C step/batch/epoch contract")
    if cfg.val_cfg is not None or cfg.test_cfg is not None or cfg.resume:
        raise ValueError("Phase 4C must disable val/test/resume")

    dataset = BUILDER.build(cfg.train_dataloader.dataset)
    sample_ids = [dataset.source[index].sample_id for index in range(len(dataset))]
    schedule = build_epoch_schedule(
        sample_ids, epochs=cfg.phase4c_epochs, seed=cfg.randomness.seed
    )
    instances = [dataset.prepare_data(index) for index in range(len(dataset))]
    for instance in instances:
        batch = chartground_sa2va_collect_fn([instance])
        record = batch["data"]["alignment_records"][0]
        if record["supervised_seg_count"] != 1 or record["gt_mask_count"] != 1:
            raise ValueError(f"alignment preflight failed: {record}")

    args.output_dir.mkdir(parents=True)
    log_path = args.output_dir / "train_steps.jsonl"
    _initialize(cfg.randomness.seed)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    build_started = time.perf_counter()
    model = BUILDER.build(cfg.model)
    model.cuda().train()
    _sync()
    build_seconds = time.perf_counter() - build_started
    trainable_names, trainable_count = model.assert_strategy_a_trainables()
    if trainable_count != 2_754_304:
        raise RuntimeError(f"unexpected trainable count: {trainable_count}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_cfg = dict(cfg.optim_wrapper.optimizer)
    optimizer_type = optimizer_cfg.pop("type")
    optimizer = optimizer_type(trainable, **optimizer_cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, cfg.warmup_steps, cfg.max_iters),
    )
    if {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer contains parameters outside text_hidden_fcs")

    common_metadata = {
        **dict(cfg.alignment_metadata),
        "protocol_version": "phase4c-overfit32-v1",
        "base_checkpoint": {
            "repo_id": args.base_repo_id,
            "revision": args.base_revision,
            "local_path": str(args.base_model),
        },
        "full_pth": {
            "path": str(args.full_pth),
            "sha256": args.full_pth_sha256,
            "bytes": args.full_pth.stat().st_size,
            "source_hf_revision": args.sa2va_hf_revision,
        },
        "prompt_registry_sha256": (
            "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0"
        ),
        "selection_path": cfg.train_dataloader.dataset.selection_path,
        "epochs": cfg.phase4c_epochs,
        "max_iters": cfg.max_iters,
        "seed": cfg.randomness.seed,
        "optimizer": {
            "type": "AdamW",
            "lr": 4e-5,
            "betas": [0.9, 0.999],
            "weight_decay": 0.05,
            "scheduler": "linear_warmup_5pct_cosine",
        },
    }
    checkpoint_paths = {}
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log_stream:
        for item in schedule:
            step_started = time.perf_counter()
            instance = instances[item["sample_index"]]
            batch = chartground_sa2va_collect_fn([instance])
            optimizer.zero_grad(set_to_none=True)
            prepared = model.data_preprocessor(batch, True)
            _sync()
            forward_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = model._run_forward(prepared, mode="loss")
            _sync()
            forward_seconds = time.perf_counter() - forward_started
            total_loss, log_vars = model.parse_losses(losses)
            if any(
                not torch.isfinite(value.detach()).all()
                for value in log_vars.values()
            ):
                raise FloatingPointError(f"non-finite loss at step {item['step']}")
            backward_started = time.perf_counter()
            total_loss.backward()
            _sync()
            backward_seconds = time.perf_counter() - backward_started
            grad_norm, nonzero_grads, frozen_with_grad = _gradient_stats(model)
            if not np.isfinite(grad_norm) or grad_norm <= 0 or nonzero_grads == 0:
                raise FloatingPointError(f"invalid gradient at step {item['step']}")
            if frozen_with_grad:
                raise RuntimeError(
                    f"frozen gradients at step {item['step']}: {frozen_with_grad}"
                )
            torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=float(cfg.optim_wrapper.clip_grad.max_norm),
                error_if_nonfinite=True,
            )
            lr = float(optimizer.param_groups[0]["lr"])
            step_only_started = time.perf_counter()
            optimizer.step()
            scheduler.step()
            _sync()
            optimizer_seconds = time.perf_counter() - step_only_started
            record = {
                **item,
                "language_loss": float(log_vars["llm_loss"].detach().float().cpu()),
                "mask_ce_loss": float(log_vars["loss_mask"].detach().float().cpu()),
                "dice_loss": float(log_vars["loss_dice"].detach().float().cpu()),
                "total_loss": float(total_loss.detach().float().cpu()),
                "gradient_norm_before_clip": grad_norm,
                "nonzero_gradient_tensor_count": nonzero_grads,
                "frozen_parameter_gradient_count": 0,
                "learning_rate": lr,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "step_seconds": time.perf_counter() - step_started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
            log_stream.write(json.dumps(record, sort_keys=True) + "\n")
            log_stream.flush()
            print("PHASE4C_STEP=" + json.dumps(record, sort_keys=True), flush=True)
            if item["step"] in cfg.checkpoint_steps:
                checkpoint = args.output_dir / f"step_{item['step']}.pth"
                save_projection_checkpoint(
                    model,
                    checkpoint,
                    {**common_metadata, "optimizer_step": item["step"]},
                )
                checkpoint_paths[str(item["step"])] = {
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": _sha256(checkpoint),
                }
    total_seconds = time.perf_counter() - started
    counts = Counter(item["sample_id"] for item in schedule)
    if set(counts.values()) != {10} or len(counts) != 32:
        raise RuntimeError(f"final sample counts are invalid: {counts}")
    summary = {
        "completed_steps": len(schedule),
        "epochs": 10,
        "sample_counts": dict(sorted(counts.items())),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "checkpoint_paths": checkpoint_paths,
        "model_build_seconds": build_seconds,
        "training_seconds": total_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "oom": False,
        "retry_count": 0,
    }
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PHASE4C_TRAIN_RESULT=" + json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    args = _args()
    try:
        run(args)
    except torch.cuda.OutOfMemoryError:
        print(
            "PHASE4C_OOM="
            + json.dumps(
                {
                    "peak_allocated_mib": torch.cuda.max_memory_allocated()
                    / (1024**2),
                    "peak_reserved_mib": torch.cuda.max_memory_reserved()
                    / (1024**2),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
