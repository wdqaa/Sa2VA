#!/usr/bin/env python3
"""Render Phase 8B figures from frozen Phase 7C files; never load a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN_OUTPUT = Path("/tmp/chartground_edit_phase7c_v2_frozen_test")
MANIFEST_SHA256 = "1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be"
CHARTS = ("line", "bar", "scatter", "confidence_band")
REFERRING = ("category", "appearance", "legend", "trend")
PRODUCT_PAIRS = tuple(zip(CHARTS, ("highlight", "recolor", "extract", "remove")))
ERROR_COLORS = {"TP": (39, 174, 96), "FP": (231, 76, 60), "FN": (52, 152, 219)}
FIGURE_DPI = (200, 200)
FONT_LATIN = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_CJK = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_hash(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(mask.shape).encode())
    digest.update(mask.astype(np.uint8, copy=False).tobytes())
    return digest.hexdigest()


def _font(size: int, *, cjk: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_CJK if cjk else FONT_LATIN
    if not path.is_file():
        raise FileNotFoundError(f"required readable font is missing: {path}")
    return ImageFont.truetype(str(path), size)


def _short(text: str, length: int = 34) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"


def _draw_instruction(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, *, size: int, fill: tuple[int, int, int]) -> None:
    """Draw Chinese and ASCII runs with fonts that actually cover each script."""
    value = value.replace("“", '"').replace("”", '"')
    x, y = xy
    runs: list[tuple[bool, str]] = []
    for character in value:
        is_ascii = ord(character) < 128
        if runs and runs[-1][0] == is_ascii:
            runs[-1] = (is_ascii, runs[-1][1] + character)
        else:
            runs.append((is_ascii, character))
    for is_ascii, run in runs:
        font = _font(size, cjk=not is_ascii)
        draw.text((x, y), run, font=font, fill=fill)
        x += draw.textlength(run, font=font)


def select_product(rows: list[dict]) -> list[dict]:
    """Fixed chart/action pairs; maximum IoU, then smallest sample ID."""
    selected = []
    for chart, action in PRODUCT_PAIRS:
        candidates = [
            row for row in rows
            if row["chart_type"] == chart
            and row["edit_action"] == action
            and row["edit_execution_success"] is True
        ]
        if not candidates:
            raise ValueError(f"no successful saved edit for {chart}/{action}")
        selected.append(min(candidates, key=lambda row: (-row["iou"], row["sample_id"])))
    return selected


def select_median(rows: list[dict], chart: str) -> list[dict]:
    """Nearest to the group's 20-sample median IoU; ID breaks ties."""
    selected = []
    for referring in REFERRING:
        group = [
            row for row in rows
            if row["chart_type"] == chart and row["referring_type"] == referring
        ]
        if len(group) != 20:
            raise ValueError(f"expected 20 frozen results for {chart}/{referring}")
        median = statistics.median(row["iou"] for row in group)
        selected.append(min(group, key=lambda row: (abs(row["iou"] - median), row["sample_id"])))
    return selected


def select_failures(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["iou"], row["dice"], row["sample_id"]))[:6]


def load_frozen_inputs(frozen_output: Path) -> tuple[list[dict], dict[str, dict]]:
    """Fail before drawing if any frozen image/mask/edit or metric is missing."""
    if not frozen_output.is_dir():
        raise FileNotFoundError(f"Phase 7C frozen output is missing: {frozen_output}")
    metrics = PROJECT / "results/phase7c_v2_frozen_test_metrics.jsonl"
    manifest = PROJECT / "data/synthetic_v2/annotations.jsonl"
    if not metrics.is_file() or not manifest.is_file():
        raise FileNotFoundError("Phase 7C metrics or synthetic_v2 manifest is missing")
    if _sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("synthetic_v2 manifest identity changed")
    all_rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in all_rows if row.get("state") == "v2_strategy_b"]
    annotations = {
        item["sample_id"]: item
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if (item := json.loads(line))["split"] == "test"
    }
    if len(all_rows) != 1280 or len(rows) != 320 or len(annotations) != 320:
        raise ValueError("frozen Phase 7C 4×320/test identity is incomplete")
    if len({row["sample_id"] for row in rows}) != 320 or set(annotations) != {row["sample_id"] for row in rows}:
        raise ValueError("frozen Strategy B test sample IDs are incomplete")
    for row in rows:
        sid = row["sample_id"]
        annotation = annotations[sid]
        paths = _paths(annotation, frozen_output)
        required = ("original", "gt", "prediction") + (("edited",) if row["edit_execution_success"] is True else ())
        for name in required:
            if not paths[name].is_file():
                raise FileNotFoundError(f"missing frozen {name} for {sid}: {paths[name]}; do not rerun inference")
        with Image.open(paths["original"]) as original, Image.open(paths["gt"]) as gt_image, Image.open(paths["prediction"]) as prediction_image:
            gt_raw = np.asarray(gt_image.convert("L"))
            pred_raw = np.asarray(prediction_image.convert("L"))
            if original.size != gt_image.size or original.size != prediction_image.size:
                raise ValueError(f"image/mask geometry mismatch: {sid}")
            if not set(np.unique(gt_raw)).issubset({0, 255}) or not set(np.unique(pred_raw)).issubset({0, 255}):
                raise ValueError(f"non-binary saved mask: {sid}")
            gt = gt_raw > 0
            pred = pred_raw > 0
        if _mask_hash(pred) != row["predicted_mask_sha256"]:
            raise ValueError(f"saved prediction hash mismatch: {sid}")
        intersection = int(np.logical_and(gt, pred).sum())
        union = int(np.logical_or(gt, pred).sum())
        if (
            intersection != row["intersection_pixels"]
            or union != row["union_pixels"]
            or abs(intersection / union - row["iou"]) > 1e-12
            or abs(2 * intersection / (int(gt.sum()) + int(pred.sum())) - row["dice"]) > 1e-12
        ):
            raise ValueError(f"saved prediction metric mismatch: {sid}")
    if Counter(row["edit_execution_success"] is True for row in rows)[True] != 315:
        raise ValueError("frozen Strategy B edited-success count changed")
    return rows, annotations


def _paths(annotation: dict, frozen_output: Path) -> dict[str, Path]:
    sid = annotation["sample_id"]
    data = PROJECT / "data/synthetic_v2"
    return {
        "original": data / annotation["image_path"],
        "gt": data / annotation["mask_path"],
        "prediction": frozen_output / "masks/v2_strategy_b" / f"{sid}.png",
        "edited": frozen_output / "edited/v2_strategy_b" / f"{sid}.png",
    }


def error_map(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    if gt.shape != pred.shape:
        raise ValueError("GT/prediction geometry differs")
    output = np.full((*gt.shape, 3), 247, dtype=np.uint8)
    output[gt & pred] = ERROR_COLORS["TP"]
    output[~gt & pred] = ERROR_COLORS["FP"]
    output[gt & ~pred] = ERROR_COLORS["FN"]
    return Image.fromarray(output)


def _checkerboard(size: tuple[int, int], block: int = 20) -> Image.Image:
    y, x = np.indices((size[1], size[0]))
    shade = np.where(((x // block) + (y // block)) % 2 == 0, 238, 204).astype(np.uint8)
    return Image.fromarray(np.stack((shade, shade, shade), axis=-1))


def _edited_rgb(edited: Image.Image) -> Image.Image:
    if edited.mode != "RGBA":
        return edited.convert("RGB")
    background = _checkerboard(edited.size).convert("RGBA")
    return Image.alpha_composite(background, edited.convert("RGBA")).convert("RGB")


def _overlay(original: Image.Image, mask: np.ndarray, *, gt: bool = False) -> Image.Image:
    image = np.asarray(original.convert("RGB")).copy()
    color = np.asarray((0, 210, 245) if gt else (255, 49, 174), dtype=np.uint8)
    image[mask] = ((0.28 * image[mask]) + (0.72 * color)).astype(np.uint8)
    dilation = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))) > 0
    image[dilation & ~mask] = (255, 225, 0)
    return Image.fromarray(image)


def _fit(image: Image.Image, size: tuple[int, int], *, nearest: bool = False) -> Image.Image:
    canvas = Image.new("RGB", size, (255, 255, 255))
    thumb = image.convert("RGB").copy()
    thumb.thumbnail(size, Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    return canvas


def _load_images(annotation: dict, frozen_output: Path) -> tuple[Image.Image, np.ndarray, np.ndarray, Image.Image | None]:
    paths = _paths(annotation, frozen_output)
    with Image.open(paths["original"]) as source, Image.open(paths["gt"]) as gt_file, Image.open(paths["prediction"]) as pred_file:
        original = source.convert("RGB").copy()
        gt = np.asarray(gt_file.convert("L")) > 0
        pred = np.asarray(pred_file.convert("L")) > 0
    edited = None
    if paths["edited"].is_file():
        with Image.open(paths["edited"]) as edited_file:
            edited = _edited_rgb(edited_file)
    return original, gt, pred, edited


def _draw_error_legend(draw: ImageDraw.ImageDraw, x: int, y: int, *, size: int = 27) -> None:
    font = _font(size)
    for label in ("TP", "FP", "FN"):
        draw.rectangle((x, y + 4, x + 24, y + 28), fill=ERROR_COLORS[label])
        draw.text((x + 35, y), f"{label}  ", font=font, fill=(34, 45, 58))
        x += 110


def render_product(rows: list[dict], annotations: dict[str, dict], frozen_output: Path, output: Path) -> list[dict]:
    selected = select_product(rows)
    sheet = Image.new("RGB", (2400, 2000), (242, 245, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((45, 32), "Selected qualitative examples", font=_font(53), fill=(20, 31, 45))
    draw.text((47, 99), "Highest-IoU successful edit within each fixed chart/action pair; synthetic_v2 test, not typical performance.", font=_font(28), fill=(65, 76, 91))
    columns = ("Input + instruction", "Predicted mask overlay", "Edited result")
    for index, label in enumerate(columns):
        draw.text((56 + index * 785, 159), label, font=_font(32), fill=(21, 48, 77))
    for index, row in enumerate(selected):
        annotation = annotations[row["sample_id"]]
        original, _, pred, edited = _load_images(annotation, frozen_output)
        if edited is None:
            raise FileNotFoundError(f"selected edit disappeared: {row['sample_id']}")
        top = 211 + index * 440
        draw.rounded_rectangle((34, top, 2366, top + 428), radius=20, fill=(255, 255, 255))
        chart = row["chart_type"].replace("_", " ")
        action = row["edit_action"]
        suffix = "  |  deterministic fill" if action == "remove" else ""
        draw.text((55, top + 15), f"{chart} / {row['referring_type']}  |  {action}  |  IoU {row['iou']:.3f}{suffix}", font=_font(31), fill=(24, 41, 59))
        _draw_instruction(draw, (55, top + 57), _short(annotation["referring_expression"], 63), size=28, fill=(65, 73, 87))
        panels = (original, _overlay(original, pred), edited)
        for column, panel in enumerate(panels):
            sheet.paste(_fit(panel, (728, 325)), (55 + column * 785, top + 99))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, dpi=FIGURE_DPI)
    return selected


def render_evaluation(chart: str, rows: list[dict], annotations: dict[str, dict], frozen_output: Path, output: Path) -> list[dict]:
    selected = select_median(rows, chart)
    sheet = Image.new("RGB", (2400, 2000), (242, 245, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((38, 26), f"Strategy B frozen test | {chart.replace('_', ' ')}", font=_font(49), fill=(20, 31, 45))
    draw.text((40, 91), "Representative sample nearest each 20-sample group's median IoU; sample ID breaks ties.", font=_font(27), fill=(65, 76, 91))
    labels = ("Original", "GT contour", "Strategy B prediction", "TP / FP / FN error map")
    for column, label in enumerate(labels):
        draw.text((37 + column * 595, 146), label, font=_font(29), fill=(21, 48, 77))
    _draw_error_legend(draw, 1920, 92, size=24)
    for index, row in enumerate(selected):
        original, gt, pred, _ = _load_images(annotations[row["sample_id"]], frozen_output)
        top = 199 + index * 444
        draw.rounded_rectangle((24, top, 2376, top + 433), radius=16, fill=(255, 255, 255))
        draw.text((42, top + 13), f"{row['referring_type']}   |   IoU {row['iou']:.3f}   Dice {row['dice']:.3f}   |   {row['difficulty']}   |   {row['edit_action']}", font=_font(29), fill=(26, 39, 55))
        panels = (original, _overlay(original, gt, gt=True), _overlay(original, pred), error_map(gt, pred))
        for column, panel in enumerate(panels):
            sheet.paste(_fit(panel, (558, 365), nearest=column == 3), (39 + column * 595, top + 62))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, dpi=FIGURE_DPI)
    return selected


def render_failures(rows: list[dict], annotations: dict[str, dict], frozen_output: Path, output: Path) -> list[dict]:
    selected = select_failures(rows)
    sheet = Image.new("RGB", (2400, 1900), (242, 245, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((40, 27), "Deterministic worst six | Strategy B frozen test", font=_font(48), fill=(20, 31, 45))
    draw.text((42, 91), "Lowest IoU, then Dice, then sample ID. Empty predictions and zero-IoU cases remain visible.", font=_font(27), fill=(65, 76, 91))
    _draw_error_legend(draw, 1890, 91, size=24)
    for index, row in enumerate(selected):
        annotation = annotations[row["sample_id"]]
        original, gt, pred, _ = _load_images(annotation, frozen_output)
        column, line = index % 2, index // 2
        left, top = 28 + column * 1190, 164 + line * 565
        draw.rounded_rectangle((left, top, left + 1160, top + 546), radius=19, fill=(255, 255, 255))
        chart = row["chart_type"].replace("_", " ")
        draw.text((left + 20, top + 15), f"{chart} / {row['referring_type']}  |  IoU {row['iou']:.3f}  |  {row['failure_category']}", font=_font(29), fill=(26, 39, 55))
        _draw_instruction(draw, (left + 20, top + 58), _short(annotation["referring_expression"], 45), size=26, fill=(65, 73, 87))
        draw.text((left + 23, top + 102), "Original", font=_font(26), fill=(21, 48, 77))
        draw.text((left + 593, top + 102), "GT / prediction error", font=_font(26), fill=(21, 48, 77))
        sheet.paste(_fit(original, (543, 399)), (left + 16, top + 140))
        sheet.paste(_fit(error_map(gt, pred), (543, 399), nearest=True), (left + 585, top + 140))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, dpi=FIGURE_DPI)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN_OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=PROJECT / "assets")
    args = parser.parse_args()
    rows, annotations = load_frozen_inputs(args.frozen_output)
    assets = args.assets_dir
    product = render_product(rows, annotations, args.frozen_output, assets / "chartground_edit_v2_demo.png")
    evaluation = {
        chart: render_evaluation(chart, rows, annotations, args.frozen_output, assets / f"phase7c_v2_eval_{chart}.png")
        for chart in CHARTS
    }
    failures = render_failures(rows, annotations, args.frozen_output, assets / "phase7c_v2_failure_cases.png")
    print(json.dumps({
        "product": [{"sample_id": row["sample_id"], "chart_type": row["chart_type"], "action": row["edit_action"], "iou": row["iou"]} for row in product],
        "evaluation": {chart: [row["sample_id"] for row in selected] for chart, selected in evaluation.items()},
        "failures": [row["sample_id"] for row in failures],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
