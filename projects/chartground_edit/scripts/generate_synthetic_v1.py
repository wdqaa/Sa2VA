#!/usr/bin/env python3
"""Generate and strictly validate the balanced synthetic_v1 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets.gallery_v1 import create_synthetic_v1_gallery
from chartground_edit.datasets.schema_v1 import validate_jsonl_v1
from chartground_edit.datasets.synthetic_v1 import (
    DEFAULT_SEED_V1,
    generate_synthetic_v1,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("projects/chartground_edit/data/synthetic_v1"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_V1)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove only generator-owned files in a safely named synthetic_v1 directory.",
    )
    parser.add_argument(
        "--gallery-output",
        type=Path,
        help="Optionally write a compact 16-combination gallery.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = generate_synthetic_v1(
            args.output_dir, seed=args.seed, clean=args.clean
        )
        records = validate_jsonl_v1(manifest, check_files=True, expected_count=320)
        gallery = None
        if args.gallery_output is not None:
            gallery = create_synthetic_v1_gallery(manifest, args.gallery_output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"generated_and_validated={len(records)}")
    print(f"manifest={manifest}")
    if gallery is not None:
        print(f"gallery={gallery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
