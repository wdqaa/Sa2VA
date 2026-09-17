#!/usr/bin/env python3
"""Run the single authorized Phase 4B smoke1 optimizer step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config
from xtuner.registry import BUILDER

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "projects" / "chartground_edit"
for import_root in (REPO_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from chartground_edit.inference.prompt_benchmark_v1 import prompt_template_hashes
from chartground_edit.training.runtime import (
    load_projection_checkpoint,
    save_projection_checkpoint,
)
from chartground_edit.training.sa2va_adapter import chartground_sa2va_collect_fn


SMOKE_ID = "cgev1_bar_category_6d51bac154"
PROMPT_REGISTRY_SHA256 = (
    "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0"
)
P2_TEMPLATE_SHA256 = (
    "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync() -> None:
    torch.cuda.synchronize()


def _seconds_since(start: float) -> float:
    _sync()
    return time.perf_counter() - start


def _parameter_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _gradient_stats(model) -> tuple[float, int, list[str]]:
    squared_norm = torch.zeros((), device="cuda", dtype=torch.float64)
    nonzero_tensors = 0
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
        if torch.count_nonzero(gradient).item() > 0:
            nonzero_tensors += 1
        squared_norm += gradient.detach().double().pow(2).sum()
    return float(squared_norm.sqrt().item()), nonzero_tensors, frozen_with_grad


def _change_stats(
    model, before: dict[str, torch.Tensor]
) -> tuple[int, float, float]:
    changed = 0
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = (parameter.detach() - before[name]).abs().float()
        changed += int(torch.count_nonzero(delta).item())
        maximum = max(maximum, float(delta.max().item()))
        absolute_sum += float(delta.double().sum().item())
        element_count += delta.numel()
    return changed, maximum, absolute_sum / element_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--full-pth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-repo-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--sa2va-hf-revision", required=True)
    parser.add_argument("--full-pth-sha256", required=True)
    return parser.parse_args()


def _initialize_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29547")
        dist.init_process_group("nccl", rank=0, world_size=1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    if not args.full_pth.is_file():
        raise FileNotFoundError(args.full_pth)
    actual_pth_hash = _sha256(args.full_pth)
    if actual_pth_hash != args.full_pth_sha256:
        raise ValueError(
            f"full PTH hash mismatch: {actual_pth_hash} != {args.full_pth_sha256}"
        )

    os.environ["CHARTGROUND_BASE_MODEL_PATH"] = str(args.base_model)
    os.environ["CHARTGROUND_SA2VA_PTH"] = str(args.full_pth)
    os.environ["CHARTGROUND_SA2VA_HF_REVISION"] = args.sa2va_hf_revision
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/chartground_mpl")
    cfg = Config.fromfile(args.config)
    if cfg.train_cfg.max_iters != 1 or cfg.train_dataloader.batch_size != 1:
        raise ValueError("Phase 4B smoke config must be exactly one sample/iteration")
    if cfg.val_cfg is not None or cfg.test_cfg is not None or cfg.resume:
        raise ValueError("Phase 4B smoke config must disable val/test/resume")

    hashes = prompt_template_hashes()
    if hashes["registry_sha256"] != PROMPT_REGISTRY_SHA256:
        raise ValueError("P2 prompt registry hash changed")
    if hashes["variants"]["target_only_zh"] != P2_TEMPLATE_SHA256:
        raise ValueError("P2 template hash changed")

    dataset = BUILDER.build(cfg.train_dataset)
    if len(dataset) != 1 or dataset.source[0].sample_id != SMOKE_ID:
        raise ValueError("smoke1 sample identity changed")
    instance = dataset.prepare_data(0)
    batch = chartground_sa2va_collect_fn(
        [instance],
        ignore_index=cfg.model.ignore_index,
        object_count_policy=cfg.model.object_count_policy,
        expected_masks_per_sample=cfg.model.expected_masks_per_sample,
    )
    alignment_records = batch["data"]["alignment_records"]
    if alignment_records != [
        {
            "sample_id": SMOKE_ID,
            "supervised_seg_count": 1,
            "gt_mask_count": 1,
            "token_positions": [1840],
            "object_count_policy": "strict_one_to_one",
        }
    ]:
        raise ValueError(f"unexpected smoke1 alignment record: {alignment_records}")

    _initialize_runtime(cfg.randomness.seed)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    build_started = time.perf_counter()
    model = BUILDER.build(cfg.model)
    model.cuda()
    _sync()
    build_seconds = time.perf_counter() - build_started
    model.train()

    trainable_names, trainable_count = model.assert_strategy_a_trainables()
    if trainable_count != 2_754_304:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable_count}")
    frozen_prefixes = tuple(cfg.frozen_modules)
    for name, parameter in model.named_parameters():
        if name.startswith(frozen_prefixes) and parameter.requires_grad:
            raise RuntimeError(f"frozen module contains trainable parameter: {name}")
    if model.seg_token_selection != "supervised_labels":
        raise RuntimeError("labels-aware [SEG] selection is disabled")
    if model.object_count_policy != "strict_one_to_one":
        raise RuntimeError("strict one-to-one alignment is disabled")

    optimizer_cfg = dict(cfg.optim_wrapper.optimizer)
    optimizer_type = optimizer_cfg.pop("type")
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = optimizer_type(trainable_parameters, **optimizer_cfg)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in trainable_parameters}:
        raise RuntimeError("optimizer parameter set differs from text_hidden_fcs")

    optimizer.zero_grad(set_to_none=True)
    prepared = model.data_preprocessor(batch, True)
    forward_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses = model._run_forward(prepared, mode="loss")
    forward_seconds = _seconds_since(forward_started)
    total_loss, log_vars = model.parse_losses(losses)
    for name, value in log_vars.items():
        if not torch.isfinite(value.detach()).all():
            raise FloatingPointError(f"non-finite loss {name}: {value.detach()}")

    backward_started = time.perf_counter()
    total_loss.backward()
    backward_seconds = _seconds_since(backward_started)
    grad_norm, nonzero_grad_tensors, frozen_with_grad = _gradient_stats(model)
    if not np.isfinite(grad_norm) or grad_norm <= 0 or nonzero_grad_tensors == 0:
        raise FloatingPointError(
            f"projection gradient is not finite/nonzero: norm={grad_norm}"
        )
    if frozen_with_grad:
        raise RuntimeError(f"frozen parameters received gradients: {frozen_with_grad}")

    before_step = _parameter_snapshot(model)
    torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        max_norm=float(cfg.optim_wrapper.clip_grad.max_norm),
        error_if_nonfinite=True,
    )
    step_started = time.perf_counter()
    optimizer.step()
    step_seconds = _seconds_since(step_started)
    changed, maximum_change, mean_change = _change_stats(model, before_step)
    if changed == 0:
        raise RuntimeError("optimizer.step did not change text_hidden_fcs")

    saved_projection = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_metadata = {
        **dict(cfg.alignment_metadata),
        "base_checkpoint": {
            "repo_id": args.base_repo_id,
            "revision": args.base_revision,
            "local_path": str(args.base_model),
        },
        "full_pth": {
            "path": str(args.full_pth),
            "sha256": actual_pth_hash,
            "bytes": args.full_pth.stat().st_size,
            "source_hf_revision": args.sa2va_hf_revision,
        },
        "prompt_registry_sha256": PROMPT_REGISTRY_SHA256,
        "smoke_sample_id": SMOKE_ID,
        "optimizer_step": 1,
    }
    save_projection_checkpoint(model, args.output, checkpoint_metadata)
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    if any(not name.startswith("text_hidden_fcs.") for name in state_dict):
        raise RuntimeError("projection checkpoint contains non-projection tensors")
    if any(
        not torch.equal(state_dict[name], saved_projection[name])
        for name in saved_projection
    ):
        raise RuntimeError("saved projection differs from in-memory step result")

    with torch.no_grad():
        for parameter in model.text_hidden_fcs.parameters():
            parameter.zero_()
    reload_metadata = load_projection_checkpoint(model, args.output)
    reloaded_projection = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    reload_exact = all(
        torch.equal(reloaded_projection[name], saved_projection[name])
        for name in saved_projection
    )
    if not reload_exact or reload_metadata["optimizer_step"] != 1:
        raise RuntimeError("projection checkpoint reload verification failed")

    report = {
        "sample_id": SMOKE_ID,
        "base_repo_id": args.base_repo_id,
        "base_revision": args.base_revision,
        "base_model_path": str(args.base_model),
        "sa2va_hf_revision": args.sa2va_hf_revision,
        "full_pth_path": str(args.full_pth),
        "full_pth_sha256": actual_pth_hash,
        "full_pth_bytes": args.full_pth.stat().st_size,
        "alignment_records": alignment_records,
        "legacy_fix_number_called": False,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "optimizer_parameter_count": sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ),
        "losses": {
            "language": float(log_vars["llm_loss"].detach().float().cpu()),
            "mask_ce": float(log_vars["loss_mask"].detach().float().cpu()),
            "dice": float(log_vars["loss_dice"].detach().float().cpu()),
            "total": float(total_loss.detach().float().cpu()),
            "all_finite": True,
        },
        "gradient_norm_before_clip": grad_norm,
        "nonzero_gradient_tensor_count": nonzero_grad_tensors,
        "frozen_parameter_gradient_count": len(frozen_with_grad),
        "parameter_change": {
            "changed_element_count": changed,
            "max_abs": maximum_change,
            "mean_abs_all_trainable_elements": mean_change,
        },
        "timing_seconds": {
            "model_build_and_cuda": build_seconds,
            "forward": forward_seconds,
            "backward": backward_seconds,
            "optimizer_step": step_seconds,
        },
        "cuda_peak_mib": {
            "allocated": torch.cuda.max_memory_allocated() / (1024**2),
            "reserved": torch.cuda.max_memory_reserved() / (1024**2),
        },
        "projection_checkpoint": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
            "tensor_count": len(state_dict),
            "tensor_names": list(state_dict),
            "reload_exact": reload_exact,
        },
        "optimizer_steps": 1,
        "forward_calls": 1,
        "backward_calls": 1,
        "oom": False,
    }
    report_path = args.output.parent / "phase4b_smoke1_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE4B_RESULT=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def main() -> int:
    args = _parse_args()
    try:
        run(args)
    except torch.cuda.OutOfMemoryError:
        summary = {
            "oom": True,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }
        print("PHASE4B_OOM=" + json.dumps(summary, sort_keys=True), file=sys.stderr)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
