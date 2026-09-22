#!/usr/bin/env python3
"""Audit the frozen Phase 4 train-only subset manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.training.subsets import audit_phase4_subsets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--overfit", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_phase4_subsets(args.manifest, args.smoke, args.overfit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"phase4_subset_audit_error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
