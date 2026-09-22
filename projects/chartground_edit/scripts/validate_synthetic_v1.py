#!/usr/bin/env python3
"""Independently validate a synthetic_v1 JSONL manifest and referenced files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets.schema import SchemaValidationError
from chartground_edit.datasets.schema_v1 import validate_jsonl_v1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=320)
    args = parser.parse_args(argv)
    try:
        records = validate_jsonl_v1(
            args.manifest, check_files=True, expected_count=args.expected_count
        )
    except (OSError, SchemaValidationError) as exc:
        print(f"validation_failed: {exc}", file=sys.stderr)
        return 1
    print(f"validation_passed={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
