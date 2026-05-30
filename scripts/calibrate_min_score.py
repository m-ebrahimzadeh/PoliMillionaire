"""Calibrate the path-aware ``min_score`` thresholds for the RAG gate.

For a given retrieval path (dense / hybrid-RRF / cross-encoder), find the
threshold τ that maximises the *gated-policy* accuracy:

    expected_acc(τ) =
        acc(score ≥ τ) · P(score ≥ τ)
      + bare_baseline_acc · P(score < τ)

Reads one or more eval JSONL files produced by ``scripts/eval_rag.py``
(or any runner that writes ``extras.top_score`` and ``correct``) and
prints the top candidate thresholds sorted by expected accuracy.

Usage
-----
    python scripts/calibrate_min_score.py \\
        --runs 'data/runs/run_rag_*.jsonl' \\
        --bare-baseline-acc 0.45 \\
        --path dense

The matching ``--path`` value depends on which RAG variant produced the
runs: ``dense`` for dense-only, ``rrf`` for hybrid (RRF) fusion, or
``rerank`` for the cross-encoder reranker. The bare-baseline accuracy
should be measured separately on the same eval set with RAG disabled.

The algorithm lives in :mod:`polimibot.eval.threshold_calibration` so the
in-notebook tuning section can call it directly (no separate process).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from polimibot.eval.threshold_calibration import (
    calibrate_threshold,
    load_pairs_from_runs,
)


def main() -> int:
    args = _parse_args()

    paths: list[Path] = []
    for pattern in args.runs:
        paths.extend(Path().glob(pattern))
    pairs = load_pairs_from_runs(*paths)
    if not pairs:
        print("No (score, correct) pairs found — check --runs glob.")
        return 1

    rows = calibrate_threshold(
        [s for s, _ in pairs],
        [c for _, c in pairs],
        bare_baseline_acc=args.bare_baseline_acc,
    )

    best = rows[0]
    print(f"\nCalibration for path={args.path!r}, "
          f"baseline_acc={args.bare_baseline_acc:.2%}, n={len(pairs)}\n")
    print("  τ        P(ungated)   acc(ungated)   expected   ")
    print("  ──────  ───────────  ─────────────  ──────────")
    for row in rows[:args.top_n]:
        flag = " ★" if row["tau"] == best["tau"] else ""
        print(
            f"  {row['tau']:>5.3f}   {row['p_ungated']:>10.2%}   "
            f"{row['acc_ungated']:>11.2%}   {row['expected']:>8.2%}{flag}"
        )

    print(f"\nBest τ for {args.path} path: {best['tau']:.3f}  "
          f"(expected acc {best['expected']:.2%})")
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Glob(s) of run JSONL files containing RAG outcomes.")
    ap.add_argument("--bare-baseline-acc", type=float, default=0.45,
                    dest="bare_baseline_acc",
                    help="Accuracy of the bare-LLM baseline on the same set "
                         "(measured separately). Default 0.45.")
    ap.add_argument("--path", choices=["dense", "rrf", "rerank"], default="dense",
                    help="Which score scale these runs use. The threshold is "
                         "calibrated for that path only.")
    ap.add_argument("--top-n", type=int, default=20, dest="top_n",
                    help="How many candidate rows to print (default 20).")
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
