"""Mine run logs → data/eval/gold_set_YYYYMMDD_HHMMSS.jsonl.

Run once after your baseline experiments. Then commit the file — it's
your fixed evaluation contract for all remaining stages.

Usage:
    python scripts/build_gold_set.py
    python scripts/build_gold_set.py --runs data/runs/ --out data/eval/gold_set.jsonl
"""
from __future__ import annotations
import argparse
from pathlib import Path

from polimibot.config import CATEGORIES, PATHS, ts
from polimibot.eval.gold_set import harvest_gold_set, save_gold_set


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=PATHS.runs_dir)
    p.add_argument("--out",  type=Path, default=None,
                   help="Output path (default: data/eval/gold_set_YYYYMMDD_HHMMSS.jsonl)")
    args = p.parse_args()

    # Compute timestamped default at runtime (not at argparse-definition time)
    out_path: Path = args.out if args.out is not None else (
        PATHS.eval_dir / f"gold_set_{ts()}.jsonl"
    )

    items = harvest_gold_set(args.runs)

    # Print breakdown by category
    print(f"\nHarvested {len(items)} gold items from {args.runs}")
    for cid, info in CATEGORIES.items():
        n = sum(1 for it in items if it.competition_id == cid)
        print(f"  {info.display_name:<35} {n:>4} items")
    print(f"\n  (elimination recovery not counted separately in this version)")

    if not items:
        print("\nNo gold items found. Run some games first:\n"
              "  python scripts/play_baseline.py --mock")
        return

    save_gold_set(items, out_path)


if __name__ == "__main__":
    main()
