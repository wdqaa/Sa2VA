#!/usr/bin/env python3
"""Strictly validate a synthetic_v2 JSONL manifest and referenced files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets.schema_v2 import validate_jsonl_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1600)
    args = parser.parse_args(argv)
    try:
        rows = validate_jsonl_v2(args.manifest, check_files=True, expected_count=args.expected_count)
    except (OSError, ValueError) as exc:
        print(f"validation_error: {exc}", file=sys.stderr)
        return 1
    print(f"validated={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
