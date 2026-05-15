"""CLI: plot reliability diagram from a run-log JSONL.

The input must be a file produced by ``RunLogger`` — gold-set JSONL has
no confidence/correct fields and is not a valid input.

Usage:
    python scripts/plot_calibration.py data/runs/run_xxx.jsonl
    python scripts/plot_calibration.py data/runs/run_xxx.jsonl --bins 8
    python scripts/plot_calibration.py data/runs/run_xxx.jsonl --out data/eval/calibration.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from polimibot.config import PATHS, ts
from polimibot.eval.calibration import calibration_from_runs, plot_calibration


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_log", type=Path,
                   help="Path to a RunLogger JSONL (data/runs/run_*.jsonl)")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--out", type=Path, default=None,
                   help="Save PNG to this path (default: data/eval/calibration_YYYYMMDD_HHMMSS.png)")
    p.add_argument("--title", default="Reliability Diagram — PoliMillionaire")
    args = p.parse_args()

    # Compute timestamped default at runtime (not at argparse-definition time)
    out_path: Path | None = args.out if args.out is not None else (
        PATHS.eval_dir / f"calibration_{ts()}.png"
    )

    result = calibration_from_runs(args.run_log, n_bins=args.bins)
    print(f"ECE = {result.ece:.4f}")
    for i, (c, a, n) in enumerate(
        zip(result.bin_confidences, result.bin_accuracies, result.bin_counts)
    ):
        print(f"  bin {i:02d}  conf={c:.2f}  acc={a:.2f}  n={n}")
    plot_calibration(result, title=args.title, output_path=out_path)


if __name__ == "__main__":
    main()
