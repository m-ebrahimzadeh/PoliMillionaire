"""CLI: plot reliability diagram from a gold-set JSONL or eval report.

Usage:
    python scripts/plot_calibration.py data/eval/gold_set.jsonl
    python scripts/plot_calibration.py data/eval/gold_set.jsonl --bins 8 --out data/eval/calibration.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from polimibot.eval.calibration import calibration_from_gold_set, plot_calibration


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("gold_set", type=Path)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", default="Reliability Diagram — PoliMillionaire")
    args = p.parse_args()

    result = calibration_from_gold_set(args.gold_set, n_bins=args.bins)
    print(f"ECE = {result.ece:.4f}")
    for i, (c, a, n) in enumerate(
        zip(result.bin_confidences, result.bin_accuracies, result.bin_counts)
    ):
        print(f"  bin {i:02d}  conf={c:.2f}  acc={a:.2f}  n={n}")
    plot_calibration(result, title=args.title, output_path=args.out)


if __name__ == "__main__":
    main()