#!/usr/bin/env python3
"""Generate and validate the local ChartGround-Edit Phase 1A dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from chartground_edit.datasets.schema import validate_jsonl
from chartground_edit.datasets.synthetic import DEFAULT_SEED, generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("projects/chartground_edit/data/synthetic_v0"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove only generator-owned artifacts inside --output-dir first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_dataset(args.output_dir, seed=args.seed, clean=args.clean)
    annotations = validate_jsonl(manifest, check_files=True, expected_count=32)
    print(f"generated_and_validated={len(annotations)}")
    print(f"manifest={manifest}")
    print(f"gallery={args.output_dir / 'gallery.png'}")


if __name__ == "__main__":
    main()

