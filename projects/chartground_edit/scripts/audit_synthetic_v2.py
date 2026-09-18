#!/usr/bin/env python3
"""Run the independent synthetic_v2 quality and diversity audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets.audit_v2 import audit_synthetic_v2, write_audit_json_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v1-manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_synthetic_v2(args.manifest, v1_manifest_path=args.v1_manifest)
        if args.output_json:
            write_audit_json_v2(report, args.output_json)
    except (OSError, ValueError) as exc:
        print(f"audit_error: {exc}", file=sys.stderr)
        return 2
    print(f"passed={str(report['passed']).lower()}")
    print(f"sample_count={report['sample_count']}")
    print(f"hard_failure_count={report['hard_failure_count']}")
    print(f"near_duplicate_candidate_count={report['near_duplicate_check']['candidate_count']}")
    if not report["passed"]:
        for failure in report["hard_failures"][:30]:
            print(f"failure: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
