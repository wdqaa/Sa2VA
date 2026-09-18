#!/usr/bin/env python3
"""Generate and strictly validate the balanced synthetic_v2 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from chartground_edit.datasets.gallery_v2 import create_synthetic_v2_gallery
from chartground_edit.datasets.schema_v2 import validate_jsonl_v2
from chartground_edit.datasets.synthetic_v2 import DEFAULT_SEED_V2, generate_synthetic_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("projects/chartground_edit/data/synthetic_v2"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_V2)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--gallery-output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = generate_synthetic_v2(args.output_dir, seed=args.seed, clean=args.clean)
        records = validate_jsonl_v2(manifest, check_files=True, expected_count=1600)
        gallery = create_synthetic_v2_gallery(manifest, args.gallery_output) if args.gallery_output else None
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"generated_and_validated={len(records)}")
    print(f"manifest={manifest}")
    if gallery:
        print(f"gallery={gallery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
