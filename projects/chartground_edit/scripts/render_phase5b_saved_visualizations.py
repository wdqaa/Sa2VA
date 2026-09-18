#!/usr/bin/env python3
"""Render Phase 5B figures exclusively from frozen saved files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.visualization.phase5b_saved import render_all_phase5b_figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-output-dir",
        type=Path,
        default=Path("/tmp/chartground_edit_phase5b_finetuned_test"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("projects/chartground_edit/assets"),
    )
    args = parser.parse_args(argv)
    try:
        report = render_all_phase5b_figures(args.frozen_output_dir, args.assets_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
