"""Readable Phase 5B figures rendered only from frozen saved artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


CHART_ORDER = ("line", "bar", "scatter", "confidence_band")
REFERRING_ORDER = ("category", "appearance", "legend", "trend")
ACTION_ORDER = ("highlight", "recolor", "extract", "remove")
ERROR_COLORS = {
    "true_positive": (39, 174, 96),
    "false_positive": (231, 76, 60),
    "false_negative": (52, 152, 219),
}
TITLE_FONT_SIZE = 30
BODY_FONT_SIZE = 22
SMALL_FONT_SIZE = 18
FIGURE_DPI = 200
EVAL_SIZE = (2240, 1760)
PRODUCT_SIZE = (2160, 1770)
FAILURE_SIZE = (2040, 1380)

_LATIN_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_CJK_FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")


def load_frozen_phase5b_records(output_dir: str | Path) -> list[dict[str, Any]]:
    """Load and validate the 64 already-saved records without invoking a model."""
    root = Path(output_dir)
    results = root / "results.jsonl"
    if not results.is_file():
        raise FileNotFoundError(
            f"frozen Phase 5B results are unavailable: {results}; do not rerun test"
        )
    records = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 64 or len({record["sample_id"] for record in records}) != 64:
        raise ValueError("frozen Phase 5B output must contain 64 unique samples")
    for record in records:
        for key in ("original", "gt_mask", "predicted_mask"):
            path = Path(record["output_files"][key])
            if not path.is_file():
                raise FileNotFoundError(f"missing saved Phase 5B artifact: {path}")
    return records


def error_map(gt_mask: Image.Image, prediction_mask: Image.Image) -> Image.Image:
    """Return the canonical TP-green, FP-red, FN-blue error map."""
    gt = np.asarray(gt_mask.convert("L")) > 0
    prediction = np.asarray(prediction_mask.convert("L")) > 0
    if gt.shape != prediction.shape:
        raise ValueError(f"mask shape mismatch: gt={gt.shape}, prediction={prediction.shape}")
    output = np.full((*gt.shape, 3), 246, dtype=np.uint8)
    output[np.logical_and(gt, prediction)] = ERROR_COLORS["true_positive"]
    output[np.logical_and(~gt, prediction)] = ERROR_COLORS["false_positive"]
    output[np.logical_and(gt, ~prediction)] = ERROR_COLORS["false_negative"]
    return Image.fromarray(output)


def select_evaluation_records(
    records: Iterable[dict[str, Any]], chart_type: str
) -> list[dict[str, Any]]:
    """Select the lexicographically first ID in each referring group."""
    rows = list(records)
    selected = []
    for referring_type in REFERRING_ORDER:
        candidates = [
            row
            for row in rows
            if row["chart_type"] == chart_type
            and row["referring_type"] == referring_type
        ]
        if not candidates:
            raise ValueError(f"missing Phase 5B group: {chart_type}/{referring_type}")
        selected.append(min(candidates, key=lambda row: row["sample_id"]))
    return selected


def select_product_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use a fixed action/chart pairing, independent of saved evaluation scores."""
    rows = list(records)
    requested = (
        ("highlight", "line", "category"),
        ("recolor", "bar", "legend"),
        ("extract", "scatter", "appearance"),
        ("remove", "confidence_band", "trend"),
    )
    selected = []
    for action, chart_type, referring_type in requested:
        candidates = [
            row
            for row in rows
            if row["edit_action"] == action
            and row["chart_type"] == chart_type
            and row["referring_type"] == referring_type
            and Path(row["output_files"].get("edited", "")).is_file()
        ]
        if not candidates:
            raise ValueError(f"missing product case: {action}/{chart_type}/{referring_type}")
        selected.append(min(candidates, key=lambda row: row["sample_id"]))
    return selected


def select_failure_records(
    records: Iterable[dict[str, Any]], count: int = 6
) -> list[dict[str, Any]]:
    """Select the objectively worst saved results with a stable tie-break."""
    if count not in range(4, 7):
        raise ValueError("failure figure must contain 4 to 6 samples")
    return sorted(
        records,
        key=lambda row: (float(row["iou"]), float(row["dice"]), row["sample_id"]),
    )[:count]


def classify_failure(gt_mask: Image.Image, prediction_mask: Image.Image) -> str:
    """Assign one of the predeclared error categories using mask geometry only."""
    gt = np.asarray(gt_mask.convert("L")) > 0
    prediction = np.asarray(prediction_mask.convert("L")) > 0
    tp = int(np.logical_and(gt, prediction).sum())
    fp = int(np.logical_and(~gt, prediction).sum())
    fn = int(np.logical_and(gt, ~prediction).sum())
    if tp == 0 and prediction.any():
        return "wrong series"
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if 0 < recall < 0.6 and precision >= 0.55:
        return "partial target"
    if fp > max(1.35 * fn, 8):
        return "over-segmentation"
    if fn > max(1.35 * fp, 8):
        return "under-segmentation"
    return "boundary error"


def render_all_phase5b_figures(
    frozen_output_dir: str | Path, assets_dir: str | Path
) -> dict[str, Any]:
    records = load_frozen_phase5b_records(frozen_output_dir)
    assets = Path(assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for chart_type in CHART_ORDER:
        path = assets / f"phase5b_eval_{chart_type}.png"
        _render_evaluation_figure(select_evaluation_records(records, chart_type), path)
        outputs[f"eval_{chart_type}"] = str(path)
    product = assets / "chartground_edit_demo_gallery.png"
    _render_product_figure(select_product_records(records), product)
    outputs["product"] = str(product)
    failure_records = select_failure_records(records)
    failure = assets / "phase5b_failure_cases.png"
    failure_types = _render_failure_figure(failure_records, failure)
    all_failure_types: list[str] = []
    low_iou_failure_types: list[str] = []
    for record in records:
        _, gt, prediction = _load_record_images(record)
        category = classify_failure(gt, prediction)
        all_failure_types.append(category)
        if float(record["iou"]) < 0.5:
            low_iou_failure_types.append(category)
    outputs["failures"] = str(failure)
    return {
        "outputs": outputs,
        "selection_policy": {
            "evaluation": "lexicographically smallest sample_id per chart/referring group",
            "product": "fixed action/chart/referring tuple, then smallest sample_id",
            "failures": "lowest IoU, then Dice, then sample_id",
        },
        "failure_sample_ids": [record["sample_id"] for record in failure_records],
        "failure_type_counts": dict(sorted(Counter(failure_types).items())),
        "all_64_error_type_counts": dict(sorted(Counter(all_failure_types).items())),
        "iou_below_0_5_error_type_counts": dict(
            sorted(Counter(low_iou_failure_types).items())
        ),
    }


def _render_evaluation_figure(records: list[dict[str, Any]], output: Path) -> None:
    sheet = Image.new("RGB", EVAL_SIZE, (240, 242, 245))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(TITLE_FONT_SIZE)
    body_font = _font(BODY_FONT_SIZE)
    small_font = _font(SMALL_FONT_SIZE)
    chart_label = records[0]["chart_type"].replace("_", " ").title()
    _draw_text(draw, (40, 22), f"Phase 5B frozen test — {chart_label}", (20, 28, 38), TITLE_FONT_SIZE)
    columns = ("Original", "GT mask / contour", "Prediction mask / contour", "Error map")
    x_positions = (28, 580, 1132, 1684)
    for x, label in zip(x_positions, columns):
        _draw_text(draw, (x + 12, 72), label, (32, 43, 56), BODY_FONT_SIZE)
    for row_index, record in enumerate(records):
        row_top = 116 + row_index * 405
        original, gt, prediction = _load_record_images(record)
        panels = (
            original,
            _mask_contour_panel(original, gt, (0, 210, 255)),
            _mask_contour_panel(original, prediction, (255, 48, 190)),
            error_map(gt, prediction),
        )
        for x, panel in zip(x_positions, panels):
            sheet.paste(_fit(panel, (528, 300), background=(255, 255, 255)), (x, row_top + 70))
        expression = _shorten(record["referring_expression"], 46)
        heading = (
            f"{record['chart_type']} / {record['referring_type']}    "
            f"IoU {record['iou']:.3f}    Dice {record['dice']:.3f}"
        )
        _draw_text(draw, (40, row_top), heading, (17, 31, 48), BODY_FONT_SIZE)
        _draw_text(draw, (40, row_top + 35), expression, (65, 74, 86), SMALL_FONT_SIZE)
    _save_png(sheet, output)


def _render_product_figure(records: list[dict[str, Any]], output: Path) -> None:
    sheet = Image.new("RGB", PRODUCT_SIZE, (245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    _draw_text(draw, (42, 22), "ChartGround-Edit — Selected qualitative examples", (20, 28, 38), TITLE_FONT_SIZE)
    headers = ("Input + instruction", "Predicted mask", "Edited result")
    x_positions = (35, 745, 1455)
    for x, label in zip(x_positions, headers):
        _draw_text(draw, (x + 10, 72), label, (38, 49, 61), BODY_FONT_SIZE)
    for index, record in enumerate(records):
        row_top = 112 + index * 405
        original, _, prediction = _load_record_images(record)
        with Image.open(record["output_files"]["edited"]) as source:
            edited = source.copy()
        if edited.mode == "RGBA":
            edited_panel = _checkerboard_composite(edited)
        else:
            edited_panel = edited.convert("RGB")
        panels = (
            original,
            _mask_contour_panel(original, prediction, (255, 48, 190)),
            edited_panel,
        )
        for x, panel in zip(x_positions, panels):
            sheet.paste(_fit(panel, (670, 292), background=(255, 255, 255)), (x, row_top + 76))
        action_note = record["edit_action"]
        if action_note == "remove":
            action_note += " (deterministic fill)"
        title = f"{action_note} · {record['chart_type']} / {record['referring_type']}"
        _draw_text(draw, (45, row_top), title, (18, 32, 48), BODY_FONT_SIZE)
        _draw_text(draw, (45, row_top + 37), _shorten(record["referring_expression"], 66), (66, 73, 84), SMALL_FONT_SIZE)
    _save_png(sheet, output)


def _render_failure_figure(
    records: list[dict[str, Any]], output: Path
) -> list[str]:
    sheet = Image.new("RGB", FAILURE_SIZE, (244, 245, 247))
    draw = ImageDraw.Draw(sheet)
    _draw_text(draw, (40, 22), "Phase 5B — deterministic worst cases", (20, 28, 38), TITLE_FONT_SIZE)
    types: list[str] = []
    for index, record in enumerate(records):
        row, column = divmod(index, 3)
        x0, y0 = 25 + column * 675, 92 + row * 625
        original, gt, prediction = _load_record_images(record)
        error = error_map(gt, prediction)
        failure_type = classify_failure(gt, prediction)
        types.append(failure_type)
        sheet.paste(_fit(original, (315, 330)), (x0, y0 + 105))
        sheet.paste(_fit(error, (315, 330)), (x0 + 325, y0 + 105))
        gt_array = np.asarray(gt.convert("L")) > 0
        pred_array = np.asarray(prediction.convert("L")) > 0
        fp = int(np.logical_and(~gt_array, pred_array).sum())
        fn = int(np.logical_and(gt_array, ~pred_array).sum())
        _draw_text(draw, (x0, y0), f"{record['chart_type']} / {record['referring_type']}", (24, 36, 51), BODY_FONT_SIZE)
        _draw_text(draw, (x0, y0 + 36), f"IoU {record['iou']:.3f} · FP {fp} · FN {fn}", (57, 65, 76), SMALL_FONT_SIZE)
        _draw_text(draw, (x0, y0 + 68), failure_type, (174, 45, 45), SMALL_FONT_SIZE)
        _draw_text(draw, (x0, y0 + 447), "Original", (45, 53, 64), SMALL_FONT_SIZE)
        _draw_text(draw, (x0 + 325, y0 + 447), "TP green · FP red · FN blue", (45, 53, 64), SMALL_FONT_SIZE)
        _draw_text(draw, (x0, y0 + 485), _shorten(record["referring_expression"], 45), (70, 77, 87), SMALL_FONT_SIZE)
    _save_png(sheet, output)
    return types


def _load_record_images(record: dict[str, Any]) -> tuple[Image.Image, Image.Image, Image.Image]:
    with Image.open(record["output_files"]["original"]) as source:
        original = source.convert("RGB").copy()
    with Image.open(record["output_files"]["gt_mask"]) as source:
        gt = source.convert("L").copy()
    with Image.open(record["output_files"]["predicted_mask"]) as source:
        prediction = source.convert("L").copy()
    return original, gt, prediction


def _mask_contour_panel(
    original: Image.Image, mask: Image.Image, color: tuple[int, int, int]
) -> Image.Image:
    base = original.convert("RGB")
    array = np.asarray(base, dtype=np.float32)
    binary = np.asarray(mask.convert("L")) > 0
    array *= 0.58
    array[binary] = 0.55 * array[binary] + 0.45 * np.asarray(color, dtype=np.float32)
    panel = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    mask_image = Image.fromarray(binary.astype(np.uint8) * 255)
    outer = np.asarray(mask_image.filter(ImageFilter.MaxFilter(9))) > 0
    inner = np.asarray(mask_image.filter(ImageFilter.MinFilter(9))) > 0
    contour = np.logical_xor(outer, inner)
    white = np.asarray(mask_image.filter(ImageFilter.MaxFilter(13))) > 0
    white_inner = np.asarray(mask_image.filter(ImageFilter.MinFilter(13))) > 0
    white_contour = np.logical_xor(white, white_inner)
    panel_array = np.asarray(panel).copy()
    panel_array[white_contour] = (255, 255, 255)
    panel_array[contour] = color
    return Image.fromarray(panel_array)


def _checkerboard_composite(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    tile = 24
    background = Image.new("RGB", rgba.size, (238, 238, 238))
    draw = ImageDraw.Draw(background)
    for y in range(0, rgba.height, tile):
        for x in range(0, rgba.width, tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(210, 210, 210))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _fit(
    image: Image.Image,
    size: tuple[int, int],
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _font(size: int, *, cjk: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _CJK_FONT_PATH if cjk else _LATIN_FONT_PATH
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    fill: tuple[int, int, int],
    size: int,
) -> None:
    """Draw mixed CJK/Latin runs with fonts that cover each script."""
    x, y = position
    current_script: bool | None = None
    run = ""
    for character in value + "\0":
        is_cjk = character != "\0" and ord(character) > 127 and character not in "—·…"
        if run and (character == "\0" or is_cjk != current_script):
            font = _font(size, cjk=bool(current_script))
            draw.text((x, y), run, fill=fill, font=font)
            box = draw.textbbox((x, y), run, font=font)
            x = box[2]
            run = ""
        if character != "\0":
            current_script = is_cjk
            run += character


def _save_png(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", dpi=(FIGURE_DPI, FIGURE_DPI), optimize=True)
