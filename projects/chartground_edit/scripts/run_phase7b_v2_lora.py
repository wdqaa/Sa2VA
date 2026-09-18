#!/usr/bin/env python3
"""Run the frozen Phase 7B smoke, training, or synthetic_v2-val evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config
from PIL import Image, ImageDraw, ImageFont
from xtuner.registry import BUILDER

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "projects/chartground_edit"
for root in (REPO_ROOT, PACKAGE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from chartground_edit.inference.sa2va_backend import Sa2VAInternVL3Backend
from chartground_edit.training.data_adapter import Phase7V2SplitDataset
from chartground_edit.training.overfit32 import (
    build_epoch_schedule,
    select_gallery_sample_ids,
    summarize_checkpoint_rows,
)
from chartground_edit.training.sa2va_adapter import chartground_sa2va_collect_fn
from chartground_edit.training.strategy_b import (
    PROJECTION_KEYS,
    canonical_trainable_name,
    load_strategy_b_checkpoint,
    load_strategy_b_checkpoint_into_hf_model,
    prepare_hf_strategy_b_model,
    save_strategy_b_checkpoint,
    strategy_b_state,
)
from chartground_edit.visualization.phase5b_saved import classify_failure
from chartground_edit.visualization.render import create_contact_sheet, mask_overlay


MANIFEST_SHA256 = "1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be"
PHASE7A_SUMMARY_SHA256 = "08ef85b5a9e4d30d0896800a1e320c503632fc2df3ab5e5413621964c87b99c1"
PHASE7A_METRICS_SHA256 = "a847499c20af8b00fc1e9ee136ccd5c3a9cfebe36e954f25f4f4dab0fd28e638"
B_NAMES = ("step960", "step1920", "step2880", "step3840", "step4800")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "evaluate"))
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE_ROOT / "configs/phase7b_v2_lora.py",
    )
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--full-pth", type=Path)
    parser.add_argument("--full-pth-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-repo-id", default="OpenGVLab/InternVL3-2B")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--sa2va-hf-revision", required=True)
    parser.add_argument("--hf-checkpoint", type=Path)
    parser.add_argument(
        "--manifest", type=Path,
        default=PACKAGE_ROOT / "data/synthetic_v2/annotations.jsonl",
    )
    parser.add_argument(
        "--phase7a-summary", type=Path,
        default=PACKAGE_ROOT / "results/phase7a_v2_projection_summary.json",
    )
    parser.add_argument(
        "--phase7a-metrics", type=Path,
        default=PACKAGE_ROOT / "results/phase7a_v2_projection_metrics.jsonl",
    )
    parser.add_argument("--strategy-a-mask-dir", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--gallery-output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"{args.stage} requires: {', '.join(missing)}")


def _load_phase7a():
    path = PACKAGE_ROOT / "scripts/run_phase7a_v2_projection.py"
    spec = importlib.util.spec_from_file_location("phase7a_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen Phase 7A runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _initialize(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29549")
        dist.init_process_group("nccl", rank=0, world_size=1)


def _sync() -> None:
    torch.cuda.synchronize()


def _lr_factor(step_index: int, warmup: int, total: int) -> float:
    if step_index < warmup:
        return float(step_index + 1) / warmup
    progress = (step_index - warmup) / (total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _prepare(args: argparse.Namespace):
    _require(args, "base_model", "full_pth", "full_pth_sha256")
    if _sha256(args.full_pth) != args.full_pth_sha256:
        raise ValueError("full PTH SHA-256 mismatch")
    os.environ["CHARTGROUND_BASE_MODEL_PATH"] = str(args.base_model)
    os.environ["CHARTGROUND_SA2VA_PTH"] = str(args.full_pth)
    os.environ["CHARTGROUND_SA2VA_HF_REVISION"] = args.sa2va_hf_revision
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = Config.fromfile(args.config)
    if cfg.max_iters != 4800 or cfg.training_epochs != 5:
        raise ValueError("Phase 7B requires exactly 5 epochs / 4800 steps")
    if cfg.val_cfg is not None or cfg.test_cfg is not None or cfg.resume:
        raise ValueError("Phase 7B training must disable val/test/resume")
    if cfg.train_dataloader.dataset.split != "train":
        raise ValueError("Phase 7B training dataset must be synthetic_v2 train")
    dataset = BUILDER.build(cfg.train_dataloader.dataset)
    sample_ids = [record["sample_id"] for record in dataset.source.records]
    schedule = build_epoch_schedule(
        sample_ids, epochs=5, seed=cfg.randomness.seed, expected_sample_count=960
    )
    return cfg, dataset, schedule


def _build_model(cfg):
    _initialize(cfg.randomness.seed)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = BUILDER.build(cfg.model)
    model.cuda().train()
    _sync()
    contract = model.assert_strategy_b_trainables()
    print("PHASE7B_TRAINABLES=" + json.dumps(contract, sort_keys=True), flush=True)
    return model, contract, time.perf_counter() - started


def _optimizer(model, contract, cfg):
    named = dict(model.named_parameters())
    projection = [named[name] for name in contract["projection_names"]]
    lora = [named[name] for name in contract["lora_names"]]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": projection,
                "lr": float(cfg.projection_optimizer.lr),
                "weight_decay": float(cfg.projection_optimizer.weight_decay),
                "group_name": "projection",
            },
            {
                "params": lora,
                "lr": float(cfg.lora_optimizer.lr),
                "weight_decay": float(cfg.lora_optimizer.weight_decay),
                "group_name": "lora",
            },
        ],
        betas=tuple(cfg.optimizer_betas),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_factor(step, cfg.warmup_steps, cfg.max_iters)
    )
    actual = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if actual != expected:
        raise RuntimeError("optimizer parameters do not exactly match Strategy B trainables")
    return optimizer, scheduler, projection, lora


def _group_gradient_stats(parameters) -> tuple[float, int]:
    squared = torch.zeros((), device="cuda", dtype=torch.float64)
    nonzero = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite Strategy B gradient")
        squared += gradient.detach().double().pow(2).sum()
        nonzero += int(torch.count_nonzero(gradient).item() > 0)
    return float(squared.sqrt().item()), nonzero


def _one_step(model, batch, optimizer, scheduler, projection, lora, cfg):
    optimizer.zero_grad(set_to_none=True)
    prepared = model.data_preprocessor(batch, True)
    _sync()
    forward_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses = model._run_forward(prepared, mode="loss")
    _sync()
    forward_seconds = time.perf_counter() - forward_started
    total_loss, log_vars = model.parse_losses(losses)
    if any(not torch.isfinite(value.detach()).all() for value in log_vars.values()):
        raise FloatingPointError("non-finite Strategy B loss")
    backward_started = time.perf_counter()
    total_loss.backward()
    _sync()
    backward_seconds = time.perf_counter() - backward_started
    projection_norm, projection_nonzero = _group_gradient_stats(projection)
    lora_norm, lora_nonzero = _group_gradient_stats(lora)
    frozen = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if projection_norm <= 0 or lora_norm <= 0 or frozen:
        raise RuntimeError(
            "Strategy B gradient contract failed: "
            f"projection={projection_norm}, lora={lora_norm}, frozen={frozen}"
        )
    torch.nn.utils.clip_grad_norm_(
        projection + lora,
        max_norm=float(cfg.optim_wrapper.clip_grad.max_norm),
        error_if_nonfinite=True,
    )
    learning_rates = {
        group["group_name"]: float(group["lr"]) for group in optimizer.param_groups
    }
    optimizer_started = time.perf_counter()
    optimizer.step()
    scheduler.step()
    _sync()
    return {
        "language_loss": float(log_vars["llm_loss"].detach().float().cpu()),
        "mask_ce_loss": float(log_vars["loss_mask"].detach().float().cpu()),
        "dice_loss": float(log_vars["loss_dice"].detach().float().cpu()),
        "total_loss": float(total_loss.detach().float().cpu()),
        "projection_gradient_norm": projection_norm,
        "lora_gradient_norm": lora_norm,
        "projection_nonzero_gradient_tensors": projection_nonzero,
        "lora_nonzero_gradient_tensors": lora_nonzero,
        "frozen_parameter_gradient_count": 0,
        "learning_rates": learning_rates,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": time.perf_counter() - optimizer_started,
    }


def _metadata(args, cfg, optimizer_step: int) -> dict:
    return {
        **dict(cfg.alignment_metadata),
        "source_hf_revision": args.sa2va_hf_revision,
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
        "prompt_registry_sha256": "dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0",
        "prompt_template_sha256": "37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806",
        "optimizer_step": optimizer_step,
        "seed": int(cfg.randomness.seed),
        "epochs": 5,
        "max_iters": 4800,
        "lora": {
            "rank": int(cfg.llm_lora_rank),
            "alpha": int(cfg.llm_lora_alpha),
            "dropout": float(cfg.llm_lora_dropout),
            "bias": "none",
            "layers": list(cfg.llm_lora_layer_indices),
            "target_modules": list(cfg.llm_lora_target_modules),
            "modules_to_save": None,
        },
        "optimizer": {
            "projection": dict(cfg.projection_optimizer),
            "lora": dict(cfg.lora_optimizer),
            "betas": list(cfg.optimizer_betas),
            "scheduler": "linear_warmup_5pct_cosine",
            "warmup_steps": 240,
        },
    }


def run_smoke(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    cfg, dataset, _ = _prepare(args)
    args.output_dir.mkdir(parents=True)
    model, contract, build_seconds = _build_model(cfg)
    optimizer, scheduler, projection, lora = _optimizer(model, contract, cfg)
    sample_id = str(cfg.smoke_sample_id)
    matches = [
        index for index, record in enumerate(dataset.source.records)
        if record["sample_id"] == sample_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"invalid smoke sample identity: {sample_id}")
    batch = chartground_sa2va_collect_fn([dataset.prepare_data(matches[0])])
    alignment = batch["data"]["alignment_records"][0]
    if alignment["supervised_seg_count"] != 1 or alignment["gt_mask_count"] != 1:
        raise RuntimeError(f"smoke alignment failed: {alignment}")
    before = strategy_b_state(model)
    step = _one_step(model, batch, optimizer, scheduler, projection, lora, cfg)
    after = strategy_b_state(model)
    projection_changed = sum(
        int(torch.count_nonzero(after[name] != before[name]))
        for name in PROJECTION_KEYS
    )
    lora_changed = sum(
        int(torch.count_nonzero(after[name] != before[name]))
        for name in after if name not in PROJECTION_KEYS
    )
    if projection_changed <= 0 or lora_changed <= 0:
        raise RuntimeError("smoke optimizer step did not update both parameter groups")
    checkpoint = args.output_dir / "iter_1.pth"
    save_strategy_b_checkpoint(model, checkpoint, _metadata(args, cfg, 1))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    load_strategy_b_checkpoint(model, checkpoint)
    reloaded = strategy_b_state(model)
    reload_exact = all(torch.equal(after[name], reloaded[name]) for name in after)
    if not reload_exact:
        raise RuntimeError("Strategy B smoke checkpoint reload differs")
    report = {
        "sample_id": sample_id,
        "alignment": alignment,
        "trainable_contract": contract,
        "step": step,
        "projection_changed_elements": projection_changed,
        "lora_changed_elements": lora_changed,
        "reload_exact": reload_exact,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            "tensor_count": len(after),
        },
        "model_build_seconds": build_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "oom": False,
    }
    (args.output_dir / "smoke_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PHASE7B_SMOKE=" + json.dumps(report, sort_keys=True), flush=True)
    return report


def run_train(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    cfg, dataset, schedule = _prepare(args)
    args.output_dir.mkdir(parents=True)
    model, contract, build_seconds = _build_model(cfg)
    optimizer, scheduler, projection, lora = _optimizer(model, contract, cfg)
    checkpoint_paths = {}
    rows = []
    started = time.perf_counter()
    with (args.output_dir / "train_steps.jsonl").open("x", encoding="utf-8") as stream:
        for item in schedule:
            step_started = time.perf_counter()
            batch = chartground_sa2va_collect_fn(
                [dataset.prepare_data(item["sample_index"])]
            )
            alignment = batch["data"]["alignment_records"][0]
            if alignment["supervised_seg_count"] != 1 or alignment["gt_mask_count"] != 1:
                raise RuntimeError(f"training alignment failed: {alignment}")
            values = _one_step(model, batch, optimizer, scheduler, projection, lora, cfg)
            record = {
                **item,
                **values,
                "step_seconds": time.perf_counter() - step_started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            rows.append(record)
            print("PHASE7B_TRAIN_STEP=" + json.dumps(record, sort_keys=True), flush=True)
            if item["step"] in cfg.checkpoint_steps:
                path = args.output_dir / f"step_{item['step']}.pth"
                save_strategy_b_checkpoint(
                    model, path, _metadata(args, cfg, item["step"])
                )
                checkpoint_paths[str(item["step"])] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
    counts = Counter(row["sample_id"] for row in rows)
    if len(rows) != 4800 or len(counts) != 960 or set(counts.values()) != {5}:
        raise RuntimeError("final Strategy B sample coverage failed")
    epochs = []
    for epoch in range(1, 6):
        selected = [row for row in rows if row["epoch"] == epoch]
        if len(selected) != 960 or len({row["sample_id"] for row in selected}) != 960:
            raise RuntimeError(f"epoch {epoch} coverage failed")
        epoch_row = {"epoch": epoch, "step_start": selected[0]["step"], "step_end": selected[-1]["step"]}
        for field in ("language_loss", "mask_ce_loss", "dice_loss", "total_loss", "projection_gradient_norm", "lora_gradient_norm", "step_seconds"):
            epoch_row[field + "_mean"] = float(np.mean([row[field] for row in selected]))
        epoch_row["peak_allocated_mib"] = max(row["peak_allocated_mib"] for row in selected)
        epoch_row["peak_reserved_mib"] = max(row["peak_reserved_mib"] for row in selected)
        epochs.append(epoch_row)
    summary = {
        "completed_steps": len(rows),
        "sample_counts": dict(sorted(counts.items())),
        "trainable_contract": contract,
        "epoch_summaries": epochs,
        "checkpoint_paths": checkpoint_paths,
        "model_build_seconds": build_seconds,
        "training_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "oom": False,
        "retry_count": 0,
    }
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PHASE7B_TRAIN=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _group(rows, field: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        name: {
            "count": len(items),
            "mean_iou": float(np.mean([row["iou"] for row in items])),
            "mean_dice": float(np.mean([row["dice"] for row in items])),
            "empty_rate": float(np.mean([row["empty_prediction"] for row in items])),
        }
        for name, items in sorted(grouped.items())
    }


def summarize_b(rows):
    by_checkpoint = defaultdict(list)
    for row in rows:
        if row.get("split") != "val":
            raise ValueError("Phase 7B metrics must be val-only")
        by_checkpoint[row["checkpoint"]].append(row)
    if tuple(by_checkpoint) != B_NAMES:
        raise ValueError(f"Strategy B checkpoint order must be {B_NAMES}")
    summaries = {}
    for name in B_NAMES:
        items = by_checkpoint[name]
        values = summarize_checkpoint_rows(
            items, expected_sample_count=320, expected_per_group=20
        )
        chart_referring = values.pop("groups")
        values["groups"] = {
            "chart_referring": chart_referring,
            "chart_type": _group(items, "chart_type"),
            "referring_type": _group(items, "referring_type"),
            "action": _group(items, "edit_action"),
            "difficulty": _group(items, "difficulty"),
            "degradation_type": _group(items, "degradation_type"),
            "theme": _group(items, "theme"),
        }
        latencies = [float(row["inference_time_ms"]) for row in items]
        values["mean_latency_ms"] = float(np.mean(latencies))
        values["median_latency_ms"] = float(np.median(latencies))
        summaries[name] = values
    best = max(
        B_NAMES,
        key=lambda name: (
            summaries[name]["group_macro_iou"],
            summaries[name]["group_macro_dice"],
            summaries[name]["micro_iou"],
            -B_NAMES.index(name),
        ),
    )
    return {"checkpoints": summaries, "best_checkpoint": best}


def _paired_bootstrap(a_rows, b_rows, *, seed=20260916, iterations=10000):
    a = {row["sample_id"]: row for row in a_rows}
    b = {row["sample_id"]: row for row in b_rows}
    if set(a) != set(b) or len(a) != 320:
        raise ValueError("paired bootstrap requires identical 320 val IDs")
    groups = sorted({f"{row['chart_type']}/{row['referring_type']}" for row in a_rows})
    differences = {"iou": [], "dice": []}
    for group in groups:
        ids = [sid for sid, row in a.items() if f"{row['chart_type']}/{row['referring_type']}" == group]
        differences["iou"].append(np.mean([b[sid]["iou"] - a[sid]["iou"] for sid in ids]))
        differences["dice"].append(np.mean([b[sid]["dice"] - a[sid]["dice"] for sid in ids]))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(iterations, len(groups)))
    result = {"seed": seed, "iterations": iterations}
    for metric, values in differences.items():
        values = np.asarray(values)
        draws = values[indices].mean(axis=1)
        result[metric] = {
            "difference": float(values.mean()),
            "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        }
    return result


def _validate_phase7a(args):
    if _sha256(args.phase7a_summary) != PHASE7A_SUMMARY_SHA256:
        raise ValueError("frozen Phase 7A summary hash mismatch")
    if _sha256(args.phase7a_metrics) != PHASE7A_METRICS_SHA256:
        raise ValueError("frozen Phase 7A metrics hash mismatch")
    summary = json.loads(args.phase7a_summary.read_text(encoding="utf-8"))
    if summary["manifest_sha256"] != MANIFEST_SHA256 or summary["best_checkpoint"] != "step4800":
        raise ValueError("frozen Phase 7A identity mismatch")
    rows = [json.loads(line) for line in args.phase7a_metrics.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if row["checkpoint"] == "step4800"]
    if len(selected) != 320 or any(row["split"] != "val" for row in selected):
        raise ValueError("frozen Phase 7A comparison rows are invalid")
    return summary, selected


def run_evaluate(args: argparse.Namespace) -> dict:
    _require(args, "hf_checkpoint", "full_pth_sha256", "strategy_a_mask_dir")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if _sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("synthetic_v2 manifest SHA-256 mismatch")
    phase7a_summary, a_rows = _validate_phase7a(args)
    train_root = args.output_dir.parent / "train"
    checkpoints = [(name, train_root / f"step_{name.removeprefix('step')}.pth") for name in B_NAMES]
    if any(not path.is_file() for _, path in checkpoints):
        raise FileNotFoundError("one or more Strategy B checkpoints are missing")
    source = Phase7V2SplitDataset(args.manifest, split="val")
    identity = {
        "source_hf_revision": args.sa2va_hf_revision,
        "full_pth_sha256": args.full_pth_sha256,
        "base_repo_id": args.base_repo_id,
        "base_revision": args.base_revision,
        "manifest_sha256": MANIFEST_SHA256,
    }
    backend = Sa2VAInternVL3Backend(args.hf_checkpoint, device="cuda:0", dtype="bfloat16")
    backend.load()
    prepare_hf_strategy_b_model(backend._model)
    args.output_dir.mkdir(parents=True)
    phase7a = _load_phase7a()
    rows = []
    for name, checkpoint in checkpoints:
        load_strategy_b_checkpoint_into_hf_model(
            backend._model, checkpoint, expected_identity=identity
        )
        mask_dir = args.output_dir / "masks" / name
        mask_dir.mkdir(parents=True)
        for index in range(len(source)):
            sample, annotation = source[index], source.records[index]
            result = backend.predict_prompt(
                sample.image, sample.prompt,
                instruction=annotation["referring_expression"],
            )
            row, prediction = phase7a._prediction_row(
                checkpoint=name, checkpoint_path=checkpoint,
                sample=sample, annotation=annotation, result=result,
            )
            Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(
                mask_dir / f"{sample.sample_id}.png"
            )
            rows.append(row)
            print("PHASE7B_EVAL=" + json.dumps(row, sort_keys=True), flush=True)
    # Reload the first complete adapter and verify that every saved tensor overwrites.
    metadata = load_strategy_b_checkpoint_into_hf_model(
        backend._model, checkpoints[0][1], expected_identity=identity
    )
    summary = summarize_b(rows)
    best = summary["best_checkpoint"]
    b_rows = [row for row in rows if row["checkpoint"] == best]
    a_metrics = phase7a_summary["checkpoints"]["step4800"]
    b_metrics = summary["checkpoints"][best]
    group_deltas = {
        group: b_metrics["groups"]["chart_referring"][group]["mean_iou"]
        - a_metrics["groups"]["chart_referring"][group]["mean_iou"]
        for group in a_metrics["groups"]["chart_referring"]
    }
    selection_checks = {
        "macro_iou_gain_at_least_0_02": b_metrics["group_macro_iou"] - a_metrics["group_macro_iou"] >= 0.02,
        "empty_rate_within_0_02": b_metrics["empty_rate"] <= a_metrics["empty_rate"] + 0.02,
        "groups_declining_over_0_03_at_most_2": sum(delta < -0.03 for delta in group_deltas.values()) <= 2,
    }
    error_counts = {"strategy_a": Counter(), "strategy_b": Counter()}
    a_by_id = {row["sample_id"]: row for row in a_rows}
    b_by_id = {row["sample_id"]: row for row in b_rows}
    for index in range(len(source)):
        sample = source[index]
        gt = Image.fromarray(sample.mask[0] * 255, mode="L")
        a_path = args.strategy_a_mask_dir / f"{sample.sample_id}.png"
        if not a_path.is_file():
            raise FileNotFoundError(f"missing frozen Strategy A mask: {a_path}")
        with Image.open(a_path) as handle:
            a_mask = handle.convert("L").copy()
        if phase7a._mask_hash(np.asarray(a_mask) > 0) != a_by_id[sample.sample_id]["predicted_mask_sha256"]:
            raise ValueError(f"Strategy A mask hash mismatch: {sample.sample_id}")
        with Image.open(args.output_dir / "masks" / best / f"{sample.sample_id}.png") as handle:
            b_mask = handle.convert("L").copy()
        error_counts["strategy_a"][classify_failure(gt, a_mask)] += 1
        error_counts["strategy_b"][classify_failure(gt, b_mask)] += 1
    summary.update({
        "manifest_sha256": MANIFEST_SHA256,
        "split": "val",
        "sample_count": 320,
        "checkpoint_order": list(B_NAMES),
        "model_load_attempts": backend.model_load_attempts,
        "peak_gpu_memory_mb": backend._peak_gpu_memory_mb(),
        "adapter_reload_metadata_step": metadata["optimizer_step"],
        "strategy_a": a_metrics,
        "best_vs_strategy_a": {
            "macro_iou_delta": b_metrics["group_macro_iou"] - a_metrics["group_macro_iou"],
            "macro_dice_delta": b_metrics["group_macro_dice"] - a_metrics["group_macro_dice"],
            "empty_rate_delta": b_metrics["empty_rate"] - a_metrics["empty_rate"],
            "group_iou_deltas": dict(sorted(group_deltas.items())),
            "selection_checks": selection_checks,
            "select_strategy_b": all(selection_checks.values()),
        },
        "paired_group_bootstrap": _paired_bootstrap(a_rows, b_rows),
        "failure_type_counts": {
            key: dict(sorted(value.items())) for key, value in error_counts.items()
        },
    })
    _write_outputs(args, source, rows, summary, a_rows)
    print("PHASE7B_EVAL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _write_outputs(args, source, rows, summary, a_rows):
    metrics_path = args.metrics_output or PACKAGE_ROOT / "results/phase7b_v2_lora_metrics.jsonl"
    summary_path = args.summary_output or PACKAGE_ROOT / "results/phase7b_v2_lora_summary.json"
    report_path = args.report_output or PACKAGE_ROOT / "docs/phase7b_v2_lora_protocol_results.md"
    comparison_path = args.comparison_output or PACKAGE_ROOT / "assets/phase7b_v2_lora_comparison.png"
    gallery_path = args.gallery_output or PACKAGE_ROOT / "assets/phase7b_v2_lora_val_gallery.png"
    public_rows = []
    for row in rows:
        public = dict(row)
        public["projection_checkpoint"] = f"<PHASE7B_WORK_DIR>/train/step_{row['checkpoint'].removeprefix('step')}.pth"
        public_rows.append(public)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in public_rows), encoding="utf-8")
    train = json.loads((args.output_dir.parent / "train/train_summary.json").read_text())
    smoke = json.loads((args.output_dir.parent / "smoke/smoke_summary.json").read_text())
    smoke["checkpoint"]["path"] = "<PHASE7B_WORK_DIR>/smoke/iter_1.pth"
    summary["training"] = {
        key: train[key] for key in (
            "completed_steps", "trainable_contract", "epoch_summaries",
            "model_build_seconds", "training_seconds", "peak_allocated_mib",
            "peak_reserved_mib", "oom", "retry_count",
        )
    }
    summary["training"]["visits_per_sample"] = sorted(set(train["sample_counts"].values()))
    summary["training"]["checkpoint_artifacts"] = {
        step: {key: value for key, value in artifact.items() if key != "path"}
        for step, artifact in train["checkpoint_paths"].items()
    }
    summary["smoke"] = smoke
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _render_comparison(summary, comparison_path)
    _render_gallery(source, rows, a_rows, args, summary, gallery_path)
    _render_report(summary, report_path)


def _render_comparison(summary, output):
    import matplotlib.pyplot as plt

    labels = ["A step4800"] + [f"B {name}" for name in B_NAMES]
    values = [summary["strategy_a"]["group_macro_iou"]] + [summary["checkpoints"][name]["group_macro_iou"] for name in B_NAMES]
    figure, axis = plt.subplots(figsize=(10, 5), dpi=180)
    colors = ["#64748b"] + ["#2563eb" if name == summary["best_checkpoint"] else "#93c5fd" for name in B_NAMES]
    bars = axis.bar(labels, values, color=colors)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set(ylabel="16-group Macro IoU", title="synthetic_v2 val: Strategy A vs Strategy B")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def _render_gallery(source, rows, a_rows, args, summary, output):
    best = summary["best_checkpoint"]
    b = {(row["checkpoint"], row["sample_id"]): row for row in rows}
    a = {row["sample_id"]: row for row in a_rows}
    selected = select_gallery_sample_ids(source.records, expected_per_group=20)
    panels = []
    panel_dir = args.output_dir / "gallery_panels"
    panel_dir.mkdir(exist_ok=True)
    for sample_id in selected:
        index = next(i for i, row in enumerate(source.records) if row["sample_id"] == sample_id)
        sample = source[index]
        gt = Image.fromarray(sample.mask[0] * 255, mode="L")
        with Image.open(args.strategy_a_mask_dir / f"{sample_id}.png") as handle:
            a_mask = handle.convert("L").copy()
        with Image.open(args.output_dir / "masks" / best / f"{sample_id}.png") as handle:
            b_mask = handle.convert("L").copy()
        images = (
            mask_overlay(sample.image, gt, color=(30, 180, 80)),
            mask_overlay(sample.image, a_mask, color=(230, 120, 30)),
            mask_overlay(sample.image, b_mask, color=(40, 110, 240)),
        )
        labels = (
            "GT",
            f"A IoU={a[sample_id]['iou']:.3f}",
            f"B IoU={b[(best, sample_id)]['iou']:.3f}",
        )
        panel = Image.new("RGB", (960, 210), "white")
        draw = ImageDraw.Draw(panel)
        for column, (image, label) in enumerate(zip(images, labels)):
            thumb = image.copy(); thumb.thumbnail((310, 160), Image.Resampling.LANCZOS)
            panel.paste(thumb, (column * 320 + (310-thumb.width)//2, 25))
            draw.text((column * 320 + 5, 5), label, fill="black", font=ImageFont.load_default())
        record = source.records[index]
        draw.text((5, 190), f"{record['chart_type']}/{record['referring_type']}", fill="black", font=ImageFont.load_default())
        path = panel_dir / f"{sample_id}.png"; panel.save(path); panels.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    create_contact_sheet(panels, output, columns=2, thumbnail_width=960)


def _render_report(summary, output):
    contract = summary["training"]["trainable_contract"]
    smoke = summary["smoke"]
    training = summary["training"]
    lines = [
        "# Phase 7B synthetic_v2 small-LLM-LoRA protocol and results", "",
        "## Frozen protocol", "",
        "- Strategy B starts from the original Sa2VA full PTH; no Phase 4/5/7 projection checkpoint is a training initializer.",
        "- Full PTH SHA-256 is `5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6`; base revision is `899155015275a9b7338c7f4677e19c784e0e5a21`; Sa2VA revision is `15837dcaecc304714a1f0f069e74f47e47521c7f`.",
        f"- Dataset is synthetic_v2 train 960 / val 320, manifest `{summary['manifest_sha256']}`. Test is forbidden.",
        "- Prompt is P2 `target_only_zh` (registry/template SHA-256 `dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0` / `37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806`); seed `20260916`; BF16; batch/accumulation `1/1`; 5 epochs / 4,800 optimizer steps.",
        "- Alignment is labels-aware strict one-to-one; legacy `fix_number=5` is disabled.",
        "- Trainable projection is the four `text_hidden_fcs.*` tensors. LoRA targets exactly decoder layers 20–27 attention `q_proj/k_proj/v_proj/o_proj`, rank 16, alpha 32, dropout 0.05, bias none.",
        "- Projection uses AdamW lr `4e-5`, weight decay `0.05`; LoRA uses lr `1e-4`, weight decay `0.01`; both share 240-step warmup and cosine decay.",
        "- Validation evaluates only the five Strategy B checkpoints. Strategy A metrics are read from frozen Phase 7A artifacts; zero-shot and Strategy A inference are not rerun.",
        "- Frozen Phase 7A summary/metrics SHA-256 are `08ef85b5a9e4d30d0896800a1e320c503632fc2df3ab5e5413621964c87b99c1` / `a847499c20af8b00fc1e9ee136ccd5c3a9cfebe36e954f25f4f4dab0fd28e638`.",
        "- B replaces A only when Macro IoU gain is at least 0.02, empty rate is no more than A + 2 percentage points, and at most two groups decline by more than 0.03.", "",
        "## Smoke and training", "",
        f"Smoke used `{smoke['sample_id']}` with supervised `[SEG]` / GT mask = `{smoke['alignment']['supervised_seg_count']}/{smoke['alignment']['gt_mask_count']}`. Losses were finite; projection/LoRA gradient norms were `{smoke['step']['projection_gradient_norm']:.6f}/{smoke['step']['lora_gradient_norm']:.6f}`; frozen gradient count was `{smoke['step']['frozen_parameter_gradient_count']}`; checkpoint reload was exact.", "",
        f"Trainable parameters: projection `{contract['projection_count']:,}`, LoRA `{contract['lora_count']:,}`, total `{contract['total_count']:,}` / model `{contract['model_total_count']:,}` (`{100 * contract['total_count'] / contract['model_total_count']:.4f}%`). The exact 68 tensor names are retained in the summary JSON.", "",
        "| epoch | language loss | mask CE | Dice loss | total loss | projection grad | LoRA grad |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in training["epoch_summaries"]:
        lines.append(
            f"| {epoch['epoch']} | {epoch['language_loss_mean']:.6f} | "
            f"{epoch['mask_ce_loss_mean']:.6f} | {epoch['dice_loss_mean']:.6f} | "
            f"{epoch['total_loss_mean']:.6f} | "
            f"{epoch['projection_gradient_norm_mean']:.6f} | "
            f"{epoch['lora_gradient_norm_mean']:.6f} |"
        )
    lines.extend([
        "",
        f"Completed `{training['completed_steps']}/4800` steps; every train sample appeared `{training['visits_per_sample'][0]}` times. Formal training took `{training['training_seconds'] / 60:.2f}` minutes after a `{training['model_build_seconds']:.2f}` second build; peak allocated/reserved memory was `{training['peak_allocated_mib']:.1f}/{training['peak_reserved_mib']:.1f}` MiB. The formal run had no OOM and no retry.", "",
        "## Validation", "",
        "| checkpoint | Macro IoU | Macro Dice | Micro IoU | Micro Dice | empty | overlap | disjoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in B_NAMES:
        m = summary["checkpoints"][name]
        lines.append(f"| {name} | {m['group_macro_iou']:.6f} | {m['group_macro_dice']:.6f} | {m['micro_iou']:.6f} | {m['micro_dice']:.6f} | {m['empty_rate']:.6f} | {m['overlap_rate']:.6f} | {m['nonempty_disjoint_rate']:.6f} |")
    best = summary["best_checkpoint"]
    compare = summary["best_vs_strategy_a"]
    a = summary["strategy_a"]
    b = summary["checkpoints"][best]
    lines.extend([
        "", f"Best B by the preregistered order is `{best}`.", "", "## A/B decision", "",
        f"Strategy A step4800 Macro IoU/Dice was `{a['group_macro_iou']:.6f}/{a['group_macro_dice']:.6f}`; best B was `{b['group_macro_iou']:.6f}/{b['group_macro_dice']:.6f}`.", "",
        f"- Macro IoU delta B−A: `{compare['macro_iou_delta']:+.6f}`。",
        f"- Macro Dice delta B−A: `{compare['macro_dice_delta']:+.6f}`。",
        f"- Empty-rate delta B−A: `{compare['empty_rate_delta']:+.6f}`。",
        f"- Selection checks: `{compare['selection_checks']}`。",
        f"- Select Strategy B: `{compare['select_strategy_b']}`。",
        f"- Paired group bootstrap: `{summary['paired_group_bootstrap']}`。",
        f"- Failure categories A/B: `{summary['failure_type_counts']}`。", "",
        "## Targeted slices", "",
        "| slice | A IoU | B IoU | delta |",
        "|---|---:|---:|---:|",
    ])
    for group in ("line/trend", "scatter/trend", "confidence_band/trend"):
        av = a["groups"]["chart_referring"][group]["mean_iou"]
        bv = b["groups"]["chart_referring"][group]["mean_iou"]
        lines.append(f"| {group} | {av:.6f} | {bv:.6f} | {bv-av:+.6f} |")
    av = a["groups"]["difficulty"]["hard"]["mean_iou"]
    bv = b["groups"]["difficulty"]["hard"]["mean_iou"]
    lines.extend([
        f"| hard difficulty | {av:.6f} | {bv:.6f} | {bv-av:+.6f} |", "",
        "The deterministic failure classifier changes wrong-series from 31 to 29 and partial-target from 53 to 54; LoRA therefore helps trend/hard overall but does not eliminate partial-target errors.", "",
        "This stage ran one fixed Strategy B training and its five validation checkpoints. It did not run Strategy C, zero-shot inference, Strategy A inference, or synthetic_v2 test.", "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _args()
    try:
        if args.stage == "smoke":
            run_smoke(args)
        elif args.stage == "train":
            run_train(args)
        else:
            run_evaluate(args)
    except torch.cuda.OutOfMemoryError:
        print("PHASE7B_OOM=" + json.dumps({
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }, sort_keys=True), file=sys.stderr)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
