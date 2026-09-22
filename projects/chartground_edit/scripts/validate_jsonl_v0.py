#!/usr/bin/env python3
"""Validate a ChartGround-Edit JSONL v0 manifest and referenced files."""

from __future__ import annotations

import argparse
from pathlib import Path

from chartground_edit.datasets.schema import validate_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Validate JSON fields only, without opening image and mask paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = validate_jsonl(
        args.manifest,
        check_files=not args.skip_files,
        expected_count=args.expected_count,
    )
    print(f"valid={len(annotations)}")
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()

