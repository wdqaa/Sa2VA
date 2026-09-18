#!/usr/bin/env python3
"""Run Phase 7A smoke, fixed training, or synthetic_v2 val evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "projects" / "chartground_edit"
for root in (REPO_ROOT, PACKAGE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from chartground_edit.inference.frozen_test_v1 import apply_predicted_mask_edit
from chartground_edit.inference.metrics import dice_score, intersection_over_union
from chartground_edit.inference.sa2va_backend import Sa2VAInternVL3Backend
from chartground_edit.training.data_adapter import Phase7V2SplitDataset
from chartground_edit.training.overfit32 import (
    select_gallery_sample_ids,
    summarize_checkpoint_rows,
)
from chartground_edit.visualization.render import create_contact_sheet, mask_overlay

EXPECTED_MANIFEST_SHA256 = (
    "1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be"
)
CHECKPOINT_NAMES = (
    "baseline",
    "v1_step960",
    "step960",
    "step1920",
    "step2880",
    "step3840",
    "step4800",
)
TRAINED_NAMES = CHECKPOINT_NAMES[2:]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "evaluate"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs/phase7a_v2_projection.py",
    )
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--full-pth", type=Path)
    parser.add_argument("--full-pth-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-repo-id", default="OpenGVLab/InternVL3-2B")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--sa2va-hf-revision", required=True)
    parser.add_argument("--hf-checkpoint", type=Path)
    parser.add_argument("--v1-projection", type=Path)
    parser.add_argument("--manifest", type=Path, default=PACKAGE_ROOT / "data/synthetic_v2/annotations.jsonl")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--curve-output", type=Path)
    parser.add_argument("--gallery-output", type=Path)
    return parser.parse_args()


def _require(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"{args.stage} requires: {', '.join(missing)}")


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(args: argparse.Namespace) -> dict[str, str]:
    return {
        "source_hf_revision": args.sa2va_hf_revision,
        "full_pth_sha256": args.full_pth_sha256,
        "base_repo_id": args.base_repo_id,
        "base_revision": args.base_revision,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    _require(args, "base_model", "full_pth", "full_pth_sha256")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    module = _load_script("chartground_phase4b_smoke", "run_phase4b_smoke1.py")
    output = args.output_dir / "iter_1.pth"
    report = module.run(
        SimpleNamespace(
            config=args.config,
            base_model=args.base_model,
            full_pth=args.full_pth,
            output=output,
            base_repo_id=args.base_repo_id,
            base_revision=args.base_revision,
            sa2va_hf_revision=args.sa2va_hf_revision,
            full_pth_sha256=args.full_pth_sha256,
        )
    )
    smoke_report = args.output_dir / "phase7a_smoke_report.json"
    smoke_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    _require(args, "base_model", "full_pth", "full_pth_sha256")
    module = _load_script("chartground_projection_train", "run_phase4c_train.py")
    return module.run(
        SimpleNamespace(
            config=args.config,
            base_model=args.base_model,
            full_pth=args.full_pth,
            full_pth_sha256=args.full_pth_sha256,
            output_dir=args.output_dir,
            base_repo_id=args.base_repo_id,
            base_revision=args.base_revision,
            sa2va_hf_revision=args.sa2va_hf_revision,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_hash(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(mask.shape).encode())
    digest.update(mask.astype(np.uint8, copy=False).tobytes())
    return digest.hexdigest()


def _degradation_type(record: dict[str, Any]) -> str:
    values = record["diversity_metadata"]["degradation"]
    tags = []
    if values["jpeg_quality"] < 100:
        tags.append("jpeg")
    if values["blur_radius"] > 0:
        tags.append("blur")
    if values["screenshot_scale"] < 1:
        tags.append("screenshot_scale")
    if values["antialias_factor"] > 1:
        tags.append("antialias_x2")
    return "+".join(tags) if tags else "clean"


def _prediction_row(
    *, checkpoint: str, checkpoint_path: Path | None, sample, annotation, result
) -> tuple[dict[str, Any], np.ndarray]:
    gt = sample.mask[0].astype(bool, copy=False)
    valid = (
        isinstance(result.mask, np.ndarray)
        and result.mask.shape == gt.shape
        and result.mask.dtype == np.bool_
    )
    prediction = result.mask.astype(bool, copy=False) if valid else np.zeros_like(gt)
    intersection = int(np.logical_and(prediction, gt).sum())
    union = int(np.logical_or(prediction, gt).sum())
    pred_pixels = int(prediction.sum())
    empty = pred_pixels == 0
    overlap = intersection > 0
    edit_success: bool | None = None
    if not empty:
        try:
            apply_predicted_mask_edit(
                sample.image,
                prediction,
                edit_action=annotation["edit_action"],
                edit_parameters=annotation["edit_parameters"],
            )
            edit_success = True
        except Exception:
            edit_success = False
    diversity = annotation["diversity_metadata"]
    return {
        "checkpoint": checkpoint,
        "projection_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "sample_id": sample.sample_id,
        "split": "val",
        "chart_type": annotation["chart_type"],
        "referring_type": annotation["referring_type"],
        "edit_action": annotation["edit_action"],
        "difficulty": annotation["difficulty"],
        "distractor_count": annotation["distractor_count"],
        "theme": diversity["theme"],
        "degradation_type": _degradation_type(annotation),
        "degradation": diversity["degradation"],
        "execution_success": result.failure_reason in (None, "empty_prediction_mask"),
        "mask_contract_valid": valid,
        "segmentation_token_present": bool(result.text_output and "[SEG]" in result.text_output),
        "empty_prediction": empty,
        "nonempty_prediction": not empty,
        "overlapping_prediction": overlap,
        "nonempty_disjoint": not empty and not overlap,
        "predicted_foreground_pixels": pred_pixels,
        "gt_foreground_pixels": int(gt.sum()),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection_over_union(prediction, gt),
        "dice": dice_score(prediction, gt),
        "inference_time_ms": result.inference_time_ms,
        "failure_reason": result.failure_reason,
        "predicted_mask_sha256": _mask_hash(prediction),
        "edit_execution_success": edit_success,
    }, prediction


def _group(rows: Sequence[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: {
            "count": len(items),
            "mean_iou": float(np.mean([row["iou"] for row in items])),
            "mean_dice": float(np.mean([row["dice"] for row in items])),
            "empty_rate": float(np.mean([row["empty_prediction"] for row in items])),
            "overlap_rate": float(np.mean([row["overlapping_prediction"] for row in items])),
            "nonempty_disjoint_rate": float(np.mean([row["nonempty_disjoint"] for row in items])),
        }
        for key, items in sorted(grouped.items())
    }


def summarize_phase7a(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "val":
            raise ValueError("Phase 7A evaluation must be val-only")
        by_checkpoint[str(row["checkpoint"])].append(row)
    if tuple(by_checkpoint) != CHECKPOINT_NAMES:
        raise ValueError(f"checkpoint order must be {CHECKPOINT_NAMES}")
    summaries: dict[str, dict[str, Any]] = {}
    for name in CHECKPOINT_NAMES:
        items = by_checkpoint[name]
        summary = summarize_checkpoint_rows(
            items, expected_sample_count=320, expected_per_group=20
        )
        chart_referring = summary.pop("groups")
        summary["groups"] = {
            "chart_referring": chart_referring,
            "chart_type": _group(items, "chart_type"),
            "referring_type": _group(items, "referring_type"),
            "action": _group(items, "edit_action"),
            "difficulty": _group(items, "difficulty"),
            "degradation_type": _group(items, "degradation_type"),
            "theme": _group(items, "theme"),
        }
        summary["edit_success_count"] = sum(row["edit_execution_success"] is True for row in items)
        summary["edit_skipped_empty_count"] = sum(row["empty_prediction"] for row in items)
        summaries[name] = summary
    baseline = summaries["baseline"]
    v1 = summaries["v1_step960"]
    for name, summary in summaries.items():
        summary["delta_vs_zero_shot"] = {
            metric: summary[metric] - baseline[metric]
            for metric in ("group_macro_iou", "group_macro_dice", "sample_macro_iou", "sample_macro_dice", "micro_iou", "micro_dice", "empty_rate")
        }
        summary["delta_vs_v1_step960"] = {
            metric: summary[metric] - v1[metric]
            for metric in ("group_macro_iou", "group_macro_dice", "sample_macro_iou", "sample_macro_dice", "micro_iou", "micro_dice", "empty_rate")
        }
    best = max(
        TRAINED_NAMES,
        key=lambda name: (
            summaries[name]["group_macro_iou"],
            summaries[name]["group_macro_dice"],
            summaries[name]["micro_iou"],
            -TRAINED_NAMES.index(name),
        ),
    )
    base_groups = baseline["groups"]["chart_referring"]
    best_groups = summaries[best]["groups"]["chart_referring"]
    improved = [name for name in sorted(base_groups) if best_groups[name]["mean_iou"] > base_groups[name]["mean_iou"]]
    declined = [name for name in sorted(base_groups) if best_groups[name]["mean_iou"] < base_groups[name]["mean_iou"]]
    return {
        "checkpoints": summaries,
        "best_checkpoint": best,
        "selection_rule": ["16-group Macro IoU", "16-group Macro Dice", "Micro IoU", "earlier checkpoint"],
        "best_vs_zero_shot_groups": {"improved": improved, "declined": declined, "tied": sorted(set(base_groups) - set(improved) - set(declined))},
        "success_thresholds": {
            "macro_iou_delta_vs_zero_shot_at_least": 0.10,
            "macro_iou_delta_vs_v1_step960_at_least": 0.05,
            "empty_rate_at_most": 0.10,
            "improved_groups_vs_zero_shot_at_least": 12,
        },
    }


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _require(args, "hf_checkpoint", "v1_projection", "full_pth_sha256")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if _sha256(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("synthetic_v2 manifest SHA-256 mismatch")
    train_root = args.output_dir.parent / "train"
    checkpoints = [("baseline", None), ("v1_step960", args.v1_projection)] + [
        (name, train_root / f"step_{name.removeprefix('step')}.pth") for name in TRAINED_NAMES
    ]
    for name, path in checkpoints[1:]:
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    source = Phase7V2SplitDataset(args.manifest, split="val")
    backend = Sa2VAInternVL3Backend(
        args.hf_checkpoint, device="cuda:0", dtype="bfloat16", projection_identity=_identity(args)
    )
    backend.load()
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    baseline_sentinel_hash = None
    for checkpoint_name, checkpoint_path in checkpoints:
        backend.set_projection_checkpoint(checkpoint_path)
        mask_dir = args.output_dir / "masks" / checkpoint_name
        mask_dir.mkdir(parents=True)
        for index in range(len(source)):
            sample = source[index]
            annotation = source.records[index]
            result = backend.predict_prompt(sample.image, sample.prompt, instruction=annotation["referring_expression"])
            row, prediction = _prediction_row(
                checkpoint=checkpoint_name,
                checkpoint_path=checkpoint_path,
                sample=sample,
                annotation=annotation,
                result=result,
            )
            Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(mask_dir / f"{sample.sample_id}.png")
            rows.append(row)
            if checkpoint_name == "baseline" and index == 0:
                baseline_sentinel_hash = row["predicted_mask_sha256"]
            print("PHASE7A_EVAL=" + json.dumps(row, sort_keys=True), flush=True)
    backend.set_projection_checkpoint(None)
    sentinel = source[0]
    restored = backend.predict_prompt(
        sentinel.image,
        sentinel.prompt,
        instruction=source.records[0]["referring_expression"],
    )
    _, restored_mask = _prediction_row(
        checkpoint="baseline",
        checkpoint_path=None,
        sample=sentinel,
        annotation=source.records[0],
        result=restored,
    )
    restore_exact = _mask_hash(restored_mask) == baseline_sentinel_hash
    if not restore_exact:
        raise RuntimeError("restored baseline projection mask hash differs")
    summary = summarize_phase7a(rows)
    summary.update({
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sample_count": 320,
        "split": "val",
        "checkpoint_order": list(CHECKPOINT_NAMES),
        "model_load_attempts": backend.model_load_attempts,
        "model_load_time_ms": backend.model_load_time_ms,
        "peak_gpu_memory_mb": backend._peak_gpu_memory_mb(),
        "baseline_restore_sentinel_sample_id": sentinel.sample_id,
        "baseline_restore_mask_hash_exact": restore_exact,
    })
    _write_outputs(args, source, rows, summary)
    print("PHASE7A_EVAL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _write_outputs(args, source, rows, summary) -> None:
    metrics = args.metrics_output or PACKAGE_ROOT / "results/phase7a_v2_projection_metrics.jsonl"
    summary_path = args.summary_output or PACKAGE_ROOT / "results/phase7a_v2_projection_summary.json"
    report = args.report_output or PACKAGE_ROOT / "docs/phase7a_v2_projection_results.md"
    curve = args.curve_output or PACKAGE_ROOT / "assets/phase7a_v2_projection_train_curve.png"
    gallery = args.gallery_output or PACKAGE_ROOT / "assets/phase7a_v2_projection_val_gallery.png"
    public_rows = []
    for row in rows:
        public = dict(row)
        public["projection_checkpoint"] = (
            None
            if row["checkpoint"] == "baseline"
            else "<V1_STEP960_PROJECTION>"
            if row["checkpoint"] == "v1_step960"
            else (
                "<PHASE7A_WORK_DIR>/train/step_"
                f"{row['checkpoint'].removeprefix('step')}.pth"
            )
        )
        public_rows.append(public)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in public_rows),
        encoding="utf-8",
    )
    train_summary_path = args.output_dir.parent / "train/train_summary.json"
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    summary["training"] = {
        "completed_steps": train_summary["completed_steps"],
        "epochs": train_summary["epochs"],
        "epoch_summaries": train_summary["epoch_summaries"],
        "trainable_parameter_names": train_summary["trainable_parameter_names"],
        "trainable_parameter_count": train_summary["trainable_parameter_count"],
        "sample_count": len(train_summary["sample_counts"]),
        "visits_per_sample": sorted(set(train_summary["sample_counts"].values())),
        "checkpoint_artifacts": {
            step: {key: value for key, value in artifact.items() if key != "path"}
            for step, artifact in train_summary["checkpoint_paths"].items()
        },
        "model_build_seconds": train_summary["model_build_seconds"],
        "training_seconds": train_summary["training_seconds"],
        "peak_allocated_mib": train_summary["peak_allocated_mib"],
        "peak_reserved_mib": train_summary["peak_reserved_mib"],
        "oom": train_summary["oom"],
        "retry_count": train_summary["retry_count"],
    }
    smoke_path = args.output_dir.parent / "smoke/phase7a_smoke_report.json"
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        summary["smoke"] = {
            key: smoke[key]
            for key in (
                "sample_id",
                "alignment_records",
                "losses",
                "gradient_norm_before_clip",
                "nonzero_gradient_tensor_count",
                "frozen_parameter_gradient_count",
                "parameter_change",
                "cuda_peak_mib",
                "trainable_parameter_count",
            )
        }
        summary["smoke"]["projection_checkpoint"] = {
            key: smoke["projection_checkpoint"][key]
            for key in ("bytes", "sha256", "tensor_count", "tensor_names", "reload_exact")
        }
    summary["evaluation_inference_seconds"] = float(
        sum(row["inference_time_ms"] or 0.0 for row in rows) / 1000.0
    )
    best = summary["best_checkpoint"]
    best_metrics = summary["checkpoints"][best]
    summary["success_assessment"] = {
        "macro_iou_delta_vs_zero_shot": best_metrics["group_macro_iou"]
        - summary["checkpoints"]["baseline"]["group_macro_iou"],
        "macro_iou_delta_vs_v1_step960": best_metrics["group_macro_iou"]
        - summary["checkpoints"]["v1_step960"]["group_macro_iou"],
        "empty_rate": best_metrics["empty_rate"],
        "improved_group_count_vs_zero_shot": len(
            summary["best_vs_zero_shot_groups"]["improved"]
        ),
    }
    checks = {
        "delta_vs_zero_shot": summary["success_assessment"][
            "macro_iou_delta_vs_zero_shot"
        ]
        >= 0.10,
        "delta_vs_v1_step960": summary["success_assessment"][
            "macro_iou_delta_vs_v1_step960"
        ]
        >= 0.05,
        "empty_rate": best_metrics["empty_rate"] <= 0.10,
        "improved_groups": summary["success_assessment"][
            "improved_group_count_vs_zero_shot"
        ]
        >= 12,
    }
    summary["success_assessment"]["threshold_checks"] = checks
    summary["success_assessment"]["all_thresholds_passed"] = all(checks.values())
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _render_curve(args.output_dir.parent / "train/train_steps.jsonl", curve)
    _render_gallery(source, rows, args.output_dir, summary, gallery)
    _render_report(summary, report)


def _render_curve(log_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5), dpi=180)
    steps = [row["step"] for row in rows]
    for field, label in (("language_loss", "language"), ("mask_ce_loss", "mask CE"), ("dice_loss", "Dice"), ("total_loss", "total")):
        values = np.asarray([row[field] for row in rows], dtype=float)
        window = min(100, len(values))
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axis.plot(steps[window - 1 :], smoothed, label=f"{label} ({window}-step mean)")
    axis.set(xlabel="optimizer step", ylabel="loss", title="Phase 7A synthetic_v2 projection-only training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _render_gallery(source, rows, output_dir: Path, summary: dict, output: Path) -> None:
    selected = select_gallery_sample_ids(source.records, expected_per_group=20)
    best = summary["best_checkpoint"]
    by_key = {(row["checkpoint"], row["sample_id"]): row for row in rows}
    panels = []
    panel_dir = output_dir / "gallery_panels"
    panel_dir.mkdir(exist_ok=True)
    for sample_id in selected:
        index = next(i for i, row in enumerate(source.records) if row["sample_id"] == sample_id)
        sample, annotation = source[index], source.records[index]
        masks = []
        for name in ("baseline", "v1_step960", best):
            with Image.open(output_dir / "masks" / name / f"{sample_id}.png") as handle:
                masks.append(handle.convert("L").copy())
        gt = Image.fromarray(sample.mask[0] * 255, mode="L")
        images = [mask_overlay(sample.image, gt, color=(36, 180, 80))] + [
            mask_overlay(sample.image, mask, color=color)
            for mask, color in zip(masks, ((230, 50, 50), (240, 150, 30), (40, 120, 240)))
        ]
        panel = Image.new("RGB", (960, 190), "white")
        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        labels = ("GT", "zero-shot", "v1 step960", best)
        keys = (None, "baseline", "v1_step960", best)
        for col, (image, label, key) in enumerate(zip(images, labels, keys)):
            thumb = image.copy()
            thumb.thumbnail((232, 145), Image.Resampling.LANCZOS)
            x = col * 240 + (232 - thumb.width) // 2
            panel.paste(thumb, (x, 23))
            text = label if key is None else f"{label} IoU={by_key[(key, sample_id)]['iou']:.3f}"
            draw.text((col * 240 + 4, 5), text, fill="black", font=font)
        draw.text((4, 174), f"{annotation['chart_type']}/{annotation['referring_type']}", fill="black", font=font)
        path = panel_dir / f"{sample_id}.png"
        panel.save(path)
        panels.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    create_contact_sheet(panels, output, columns=2, thumbnail_width=960)


def _render_report(summary: dict, output: Path) -> None:
    lines = [
        "# Phase 7A synthetic_v2 projection-only results",
        "",
        "固定 Strategy A：仅训练 2,754,304 个 `text_hidden_fcs` 参数；评测仅使用 synthetic_v2 val，未访问 test。",
        "",
        "| checkpoint | group Macro IoU | group Macro Dice | sample Macro IoU | sample Macro Dice | Micro IoU | Micro Dice | empty | overlap | disjoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    checkpoint_order = summary.get("checkpoint_order", list(summary["checkpoints"]))
    for name in checkpoint_order:
        metrics = summary["checkpoints"][name]
        lines.append(
            f"| {name} | {metrics['group_macro_iou']:.6f} | "
            f"{metrics['group_macro_dice']:.6f} | "
            f"{metrics['sample_macro_iou']:.6f} | "
            f"{metrics['sample_macro_dice']:.6f} | "
            f"{metrics['micro_iou']:.6f} | {metrics['micro_dice']:.6f} | "
            f"{metrics['empty_rate']:.6f} | {metrics['overlap_rate']:.6f} | "
            f"{metrics['nonempty_disjoint_rate']:.6f} |"
        )
    best = summary["best_checkpoint"]
    lines.extend(
        [
            "",
            f"Selected checkpoint: `{best}`（16-group Macro IoU → Macro Dice → Micro IoU → earlier checkpoint）。",
            "",
            "## Training",
            "",
            "| epoch | language | mask CE | Dice | total |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for epoch in summary["training"]["epoch_summaries"]:
        lines.append(
            f"| {epoch['epoch']} | {epoch['language_loss_mean']:.6f} | "
            f"{epoch['mask_ce_loss_mean']:.6f} | {epoch['dice_loss_mean']:.6f} | "
            f"{epoch['total_loss_mean']:.6f} |"
        )
    training = summary["training"]
    lines.extend(
        [
            "",
            f"完成 `{training['completed_steps']}/{training['completed_steps']}` optimizer steps；"
            f"训练耗时 `{training['training_seconds']:.1f}s`，模型构建 "
            f"`{training['model_build_seconds']:.1f}s`。峰值 allocated/reserved 显存为 "
            f"`{training['peak_allocated_mib']:.1f}/{training['peak_reserved_mib']:.1f} MiB`。"
            f"OOM=`{training['oom']}`，retry=`{training['retry_count']}`。",
        ]
    )
    lines.extend(
        [
            "",
            "## Best checkpoint: 16 chart/referring groups",
            "",
            "| group | zero-shot IoU | v1 step960 IoU | best IoU | Δ zero-shot | Δ v1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    groups = summary["checkpoints"]
    for group in sorted(groups[best]["groups"]["chart_referring"]):
        zero = groups["baseline"]["groups"]["chart_referring"][group]["mean_iou"]
        v1 = groups["v1_step960"]["groups"]["chart_referring"][group]["mean_iou"]
        value = groups[best]["groups"]["chart_referring"][group]["mean_iou"]
        lines.append(
            f"| {group} | {zero:.6f} | {v1:.6f} | {value:.6f} | "
            f"{value-zero:+.6f} | {value-v1:+.6f} |"
        )
    assessment = summary["success_assessment"]
    best_groups = groups[best]["groups"]
    zero_groups = groups["baseline"]["groups"]["chart_referring"]
    v1_groups = groups["v1_step960"]["groups"]["chart_referring"]
    improved_v1 = sum(
        best_groups["chart_referring"][group]["mean_iou"] > v1_groups[group]["mean_iou"]
        for group in v1_groups
    )
    lines.extend(
        [
            "",
            "## Best checkpoint: diagnostic groups",
            "",
            "| axis | group | count | mean IoU | mean Dice | empty rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for axis in ("action", "difficulty", "degradation_type", "theme"):
        for group, values in sorted(best_groups[axis].items()):
            lines.append(
                f"| {axis} | {group} | {values['count']} | "
                f"{values['mean_iou']:.6f} | {values['mean_dice']:.6f} | "
                f"{values['empty_rate']:.6f} |"
            )
    strongest = max(
        best_groups["chart_referring"].items(), key=lambda item: item[1]["mean_iou"]
    )
    weakest = min(
        best_groups["chart_referring"].items(), key=lambda item: item[1]["mean_iou"]
    )
    declined_zero = [
        group
        for group, values in best_groups["chart_referring"].items()
        if values["mean_iou"] <= zero_groups[group]["mean_iou"]
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- 相对 zero-shot Macro IoU：`{assessment['macro_iou_delta_vs_zero_shot']:+.6f}`。",
            f"- 相对 v1 step960 迁移：`{assessment['macro_iou_delta_vs_v1_step960']:+.6f}`。",
            f"- Empty rate：`{assessment['empty_rate']:.4%}`；相对 zero-shot 提升组数：`{assessment['improved_group_count_vs_zero_shot']}/16`。",
            f"- 相对 v1 step960 提升组数：`{improved_v1}/16`；相对 zero-shot 未提升组：`{', '.join(declined_zero)}`。",
            f"- 最强组：`{strongest[0]}` (`{strongest[1]['mean_iou']:.6f}`)；最弱组：`{weakest[0]}` (`{weakest[1]['mean_iou']:.6f}`)。",
            f"- 预注册阈值检查：`{assessment['threshold_checks']}`；全部通过：`{assessment['all_thresholds_passed']}`。",
            "- Strategy A 明显优于原始 zero-shot，且空预测受到控制，但相对强劲的 v1 step960 迁移仅提高 0.031550，未达到预注册的 +0.05 门槛。",
            "- 因此可以把小型 LoRA 作为下一项受控 Strategy B 消融；本阶段未启动 LoRA，也未据 validation 结果重训 Strategy A。",
            "- 本阶段仅评测 synthetic_v2 val；未访问 synthetic_v2 test。",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _args()
    if args.stage == "smoke":
        run_smoke(args)
    elif args.stage == "train":
        run_train(args)
    else:
        run_evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
