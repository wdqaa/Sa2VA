#!/usr/bin/env python3
"""Apply a deterministic ChartGround-Edit operation using an existing mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.editing import EditingError, edit


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=existing_file, help="RGB source image")
    parser.add_argument("--mask", required=True, type=existing_file, help="binary mask")
    parser.add_argument(
        "--action",
        required=True,
        choices=("highlight", "recolor", "extract", "remove"),
    )
    parser.add_argument("--output", required=True, type=Path, help="output PNG path")
    parser.add_argument(
        "--color",
        default="#E63946",
        help="recolor target or remove fill/fallback color (#RGB or #RRGGBB)",
    )
    parser.add_argument(
        "--highlight-strength",
        type=float,
        default=0.65,
        help="highlight background suppression in [0, 1] (default: 0.65)",
    )
    parser.add_argument(
        "--remove-fill-mode",
        choices=("color", "neighbor"),
        default="color",
        help="deterministic remove strategy (default: color)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".png":
        raise SystemExit("error: --output must end with .png")
    try:
        with Image.open(args.image) as image_file:
            image = image_file.convert("RGB").copy()
        with Image.open(args.mask) as mask_file:
            mask = mask_file.copy()
        parameters = _parameters_for(args)
        edited = edit(image, mask, args.action, parameters)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        edited.save(args.output, format="PNG")
    except (EditingError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"action={args.action}")
    print(f"mode={edited.mode}")
    print(f"output={args.output}")


def _parameters_for(args: argparse.Namespace) -> dict[str, object]:
    if args.action == "highlight":
        return {"strength": args.highlight_strength}
    if args.action == "recolor":
        return {"color": args.color}
    if args.action == "remove":
        return {"fill_mode": args.remove_fill_mode, "color": args.color}
    return {}


if __name__ == "__main__":
    main()
