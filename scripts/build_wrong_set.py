"""Mine run logs → data/eval/wrong_set.jsonl.

Symmetrical companion to build_gold_set.py.  Where the gold set harvests
questions the bot answered correctly, this script harvests questions that
were answered *incorrectly*.  Run it after accumulating game sessions to
build a persistent wrong-answer set for error analysis.

Usage:
    python scripts/build_wrong_set.py
    python scripts/build_wrong_set.py --runs data/runs/ --out data/eval/wrong_set.jsonl
"""
from __future__ import annotations
import argparse
from pathlib import Path

from polimibot.config import CATEGORIES, PATHS
from polimibot.eval.wrong_set import harvest_wrong_set, save_wrong_set


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=PATHS.runs_dir)
    p.add_argument("--out",  type=Path, default=PATHS.eval_dir / "wrong_set.jsonl")
    args = p.parse_args()

    items = harvest_wrong_set(args.runs)

    # Print breakdown by category
    print(f"\nHarvested {len(items)} wrong items from {args.runs}")
    for cid, info in CATEGORIES.items():
        n = sum(1 for it in items if it.competition_id == cid)
        print(f"  {info.display_name:<35} {n:>4} items")

    n_known = sum(1 for it in items if it.correct_index >= 0)
    if items:
        print(f"\nCorrect answer recovered: {n_known}/{len(items)} "
              f"({n_known / len(items):.1%})")
    else:
        print("\nNo wrong items found — play more live games first.")

    if not items:
        return

    save_wrong_set(items, args.out)
    print(f"Wrong set saved → {args.out}")


if __name__ == "__main__":
    main()
