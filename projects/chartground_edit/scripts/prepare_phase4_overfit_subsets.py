#!/usr/bin/env python3
"""Freeze deterministic synthetic_v1 train-only Phase 4 subsets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.training.subsets import prepare_phase4_subsets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--smoke-output", required=True, type=Path)
    parser.add_argument("--overfit-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        smoke, overfit = prepare_phase4_subsets(
            args.manifest, args.smoke_output, args.overfit_output
        )
    except (OSError, ValueError) as exc:
        print(f"phase4_subset_error: {exc}", file=sys.stderr)
        return 1
    print(f"smoke1={smoke['sample_ids'][0]}")
    print(f"overfit32_count={overfit['sample_count']}")
    print(f"action_distribution={overfit['distribution']['edit_action']}")
    print(f"difficulty_distribution={overfit['distribution']['difficulty']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
