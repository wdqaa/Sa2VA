#!/usr/bin/env python3
"""Evaluate baseline/projection checkpoints on a frozen synthetic_v1 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "projects" / "chartground_edit"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.inference.metrics import dice_score, intersection_over_union
from chartground_edit.inference.sa2va_backend import Sa2VAInternVL3Backend
from chartground_edit.training.data_adapter import Phase4TrainDataset, Phase5SplitDataset
from chartground_edit.training.overfit32 import (
    CHECKPOINT_NAMES as PHASE4C_CHECKPOINT_NAMES,
    select_gallery_sample_ids,
    summarize_checkpoint_rows,
    summarize_overfit_rows,
)
from chartground_edit.training.phase5a import (
    CHECKPOINT_NAMES as PHASE5A_CHECKPOINT_NAMES,
    summarize_phase5a_rows,
    validate_phase3b_baseline,
)
from chartground_edit.visualization.render import create_contact_sheet, mask_overlay


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--experiment", choices=("phase4c", "phase5a"), default="phase4c"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--expected-samples", type=int, default=32)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--projection", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--reference-baseline", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--gallery-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--sa2va-hf-revision", required=True)
    parser.add_argument("--full-pth-sha256", required=True)
    parser.add_argument("--base-repo-id", required=True)
    parser.add_argument("--base-revision", required=True)
    return parser.parse_args()


def _projection_specs(
    values: list[str], checkpoint_names: tuple[str, ...]
) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid projection spec: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if name not in checkpoint_names[1:] or not path.is_file():
            raise ValueError(f"invalid projection checkpoint: {value}")
        specs.append((name, path))
    if len({name for name, _ in specs}) != len(specs):
        raise ValueError("duplicate projection checkpoint name")
    return specs


def _mask_hash(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(mask.shape).encode())
    digest.update(mask.astype(np.uint8, copy=False).tobytes())
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    checkpoint_names = (
        PHASE4C_CHECKPOINT_NAMES
        if args.experiment == "phase4c"
        else PHASE5A_CHECKPOINT_NAMES
    )
    specs = _projection_specs(args.projection, checkpoint_names)
    if args.experiment == "phase4c":
        if args.selection is None or args.split != "train" or args.expected_samples != 32:
            raise ValueError("Phase 4C inference requires overfit32 train selection")
        source = Phase4TrainDataset(args.manifest, args.selection)
    else:
        if args.selection is not None or args.split != "val" or args.expected_samples != 64:
            raise ValueError("Phase 5A inference requires the complete val split")
        source = Phase5SplitDataset(args.manifest, split=args.split)
    if len(source) != args.expected_samples or any(
        row["split"] != args.split for row in source.records
    ):
        raise ValueError("evaluation split/sample contract failed")
    identity = {
        "source_hf_revision": args.sa2va_hf_revision,
        "full_pth_sha256": args.full_pth_sha256,
        "base_repo_id": args.base_repo_id,
        "base_revision": args.base_revision,
    }
    backend = Sa2VAInternVL3Backend(
        args.checkpoint,
        device="cuda:0",
        dtype="bfloat16",
        projection_identity=identity,
    )
    backend.load()
    args.output_dir.mkdir(parents=True)
    rows = []
    checkpoint_specs: list[tuple[str, Path | None]] = [("baseline", None), *specs]
    for checkpoint_name, checkpoint_path in checkpoint_specs:
        backend.set_projection_checkpoint(checkpoint_path)
        mask_dir = args.output_dir / "masks" / checkpoint_name
        mask_dir.mkdir(parents=True)
        for index in range(len(source)):
            sample = source[index]
            annotation = source.records[index]
            result = backend.predict_prompt(
                sample.image,
                sample.prompt,
                instruction=annotation["referring_expression"],
            )
            gt = sample.mask[0].astype(bool, copy=False)
            mask_contract_valid = (
                isinstance(result.mask, np.ndarray)
                and result.mask.shape == gt.shape
                and result.mask.dtype == np.bool_
            )
            prediction = (
                result.mask.astype(bool, copy=False)
                if mask_contract_valid
                else np.zeros_like(gt)
            )
            intersection = int(np.logical_and(prediction, gt).sum())
            union = int(np.logical_or(prediction, gt).sum())
            pred_pixels = int(prediction.sum())
            gt_pixels = int(gt.sum())
            empty = pred_pixels == 0
            overlap = intersection > 0
            Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(
                mask_dir / f"{sample.sample_id}.png"
            )
            row = {
                "checkpoint": checkpoint_name,
                "projection_checkpoint": (
                    str(checkpoint_path) if checkpoint_path is not None else None
                ),
                "sample_id": sample.sample_id,
                "split": annotation["split"],
                "chart_type": annotation["chart_type"],
                "referring_type": annotation["referring_type"],
                "edit_action": annotation["edit_action"],
                "difficulty": annotation["difficulty"],
                "distractor_count": annotation["distractor_count"],
                "execution_success": result.failure_reason
                in (None, "empty_prediction_mask"),
                "mask_contract_valid": mask_contract_valid,
                "segmentation_token_present": bool(
                    result.text_output and "[SEG]" in result.text_output
                ),
                "empty_prediction": empty,
                "nonempty_prediction": not empty,
                "overlapping_prediction": overlap,
                "nonempty_disjoint": not empty and not overlap,
                "predicted_foreground_pixels": pred_pixels,
                "gt_foreground_pixels": gt_pixels,
                "intersection_pixels": intersection,
                "union_pixels": union,
                "iou": intersection_over_union(prediction, gt),
                "dice": dice_score(prediction, gt),
                "inference_time_ms": result.inference_time_ms,
                "failure_reason": result.failure_reason,
                "predicted_mask_sha256": _mask_hash(prediction),
            }
            rows.append(row)
            print("PROJECTION_EVAL=" + json.dumps(row, sort_keys=True), flush=True)
        if args.experiment == "phase5a" and checkpoint_name == "baseline":
            validate_phase3b_baseline(rows)
            print("PHASE5A_BASELINE_REPRODUCTION=pass", flush=True)

    metrics_path = args.output_dir / "metrics.jsonl"
    _write_jsonl(metrics_path, rows)
    if specs:
        if tuple(name for name, _ in checkpoint_specs) != checkpoint_names:
            raise ValueError(f"post-training order must be {checkpoint_names}")
        summary = (
            summarize_overfit_rows(rows)
            if args.experiment == "phase4c"
            else summarize_phase5a_rows(rows)
        )
    else:
        summary = {
            "checkpoints": {
                "baseline": summarize_checkpoint_rows(
                    rows,
                    expected_sample_count=args.expected_samples,
                    expected_per_group=args.expected_samples // 16,
                )
            },
            "best_checkpoint": None,
        }
    summary.update(
        {
            "model_load_attempts": backend.model_load_attempts,
            "model_load_time_ms": backend.model_load_time_ms,
            "peak_gpu_memory_mb": backend._peak_gpu_memory_mb(),
            "experiment": args.experiment,
            "sample_count": args.expected_samples,
            "split": args.split,
            "checkpoint_order": [name for name, _ in checkpoint_specs],
        }
    )
    if args.reference_baseline is not None:
        reference = [
            json.loads(line)
            for line in args.reference_baseline.read_text(encoding="utf-8").splitlines()
        ]
        current = [row for row in rows if row["checkpoint"] == "baseline"]
        reference_by_id = {row["sample_id"]: row for row in reference}
        if set(reference_by_id) != {row["sample_id"] for row in current}:
            raise ValueError("reference baseline sample IDs differ")
        if any(
            reference_by_id[row["sample_id"]]["predicted_mask_sha256"]
            != row["predicted_mask_sha256"]
            for row in current
        ):
            raise ValueError("pre/post-training baseline predictions differ")
        summary["reference_baseline_exact"] = True
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.metrics_output is not None:
        _write_jsonl(args.metrics_output, rows)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if specs and args.gallery_output is not None:
        _render_gallery(
            source,
            rows,
            args.output_dir,
            summary,
            args.gallery_output,
            expected_per_group=args.expected_samples // 16,
        )
    if specs and args.report_output is not None:
        _write_report(summary, args.report_output, experiment=args.experiment)
    print("PROJECTION_EVAL_RESULT=" + json.dumps(summary, sort_keys=True))
    return summary


def _render_gallery(
    source,
    rows,
    output_dir: Path,
    summary: dict,
    output: Path,
    *,
    expected_per_group: int,
) -> None:
    selected = select_gallery_sample_ids(
        source.records, expected_per_group=expected_per_group
    )
    best = summary["best_checkpoint"]
    by_key = {(row["checkpoint"], row["sample_id"]): row for row in rows}
    panels = []
    panel_dir = output_dir / "gallery_panels"
    panel_dir.mkdir()
    for sample_id in selected:
        index = next(i for i, row in enumerate(source.records) if row["sample_id"] == sample_id)
        sample = source[index]
        annotation = source.records[index]
        masks = []
        for name in ("baseline", best):
            with Image.open(output_dir / "masks" / name / f"{sample_id}.png") as file:
                masks.append(file.convert("L").copy())
        gt = Image.fromarray(sample.mask[0] * 255, mode="L")
        images = [
            mask_overlay(sample.image, gt, color=(36, 180, 80)),
            mask_overlay(sample.image, masks[0], color=(230, 50, 50)),
            mask_overlay(sample.image, masks[1], color=(40, 120, 240)),
        ]
        cell = (200, 133)
        panel = Image.new("RGB", (cell[0] * 3, 166), "white")
        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        labels = (
            "original + GT",
            f"baseline IoU={by_key[('baseline', sample_id)]['iou']:.3f}",
            f"{best} IoU={by_key[(best, sample_id)]['iou']:.3f}",
        )
        for column, (image, label) in enumerate(zip(images, labels)):
            thumb = image.resize(cell, Image.Resampling.LANCZOS)
            panel.paste(thumb, (column * cell[0], 20))
            draw.text((column * cell[0] + 4, 4), label, fill="black", font=font)
        draw.text(
            (4, 154),
            f"{annotation['chart_type']}/{annotation['referring_type']} | {sample_id}",
            fill="black",
            font=font,
        )
        path = panel_dir / f"{sample_id}.png"
        panel.save(path)
        panels.append(path)
    create_contact_sheet(panels, output, columns=2, thumbnail_width=600)


def _write_report(summary: dict, output: Path, *, experiment: str) -> None:
    phase5a = experiment == "phase5a"
    lines = [
        "# Phase 5A full-train validation results"
        if phase5a
        else "# Phase 4C overfit32 results",
        "",
        "仅使用冻结的 64 条 val 样本进行 checkpoint 选择；未访问 test。"
        if phase5a
        else "仅使用冻结的 32 条 train 样本；这些指标只表示训练集 learnability。",
        "",
        "| checkpoint | group Macro IoU | group Macro Dice | sample Macro IoU | Micro IoU | empty rate | disjoint rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["checkpoints"].items():
        lines.append(
            f"| {name} | {metrics['group_macro_iou']:.6f} | "
            f"{metrics['group_macro_dice']:.6f} | {metrics['sample_macro_iou']:.6f} | "
            f"{metrics['micro_iou']:.6f} | {metrics['empty_rate']:.6f} | "
            f"{metrics['nonempty_disjoint_rate']:.6f} |"
        )
    lines.extend(["", f"Selected checkpoint: `{summary['best_checkpoint']}`.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(0 if run(_args()) else 1)
