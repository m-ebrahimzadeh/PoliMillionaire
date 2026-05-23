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
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    args = _parse_args()

    pairs = _load_pairs(args.runs)
    if not pairs:
        print("No (score, correct) pairs found — check --runs glob.")
        return 1

    scores  = np.array([s for s, _ in pairs], dtype=float)
    correct = np.array([c for _, c in pairs], dtype=bool)

    # Candidate thresholds: every observed score, plus 0 and 1. Rounded
    # to 3 dp so identical scores collapse into one candidate.
    candidates = sorted(set(np.round(scores, 3).tolist()) | {0.0, 1.0})

    bare = args.bare_baseline_acc
    rows = []
    best: tuple[float | None, float] = (None, -1.0)

    for tau in candidates:
        ungated   = scores >= tau
        n_ungated = int(ungated.sum())
        n_total   = len(scores)
        acc_ungated = float(correct[ungated].mean()) if n_ungated > 0 else 0.0
        p_ungated   = n_ungated / n_total
        expected    = acc_ungated * p_ungated + bare * (1.0 - p_ungated)
        rows.append((tau, p_ungated, acc_ungated, expected))
        if expected > best[1]:
            best = (tau, expected)

    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"\nCalibration for path={args.path!r}, "
          f"baseline_acc={bare:.2%}, n={len(scores)}\n")
    print("  τ        P(ungated)   acc(ungated)   expected   ")
    print("  ──────  ───────────  ─────────────  ──────────")
    for tau, p, acc, exp in rows[:args.top_n]:
        flag = " ★" if tau == best[0] else ""
        print(f"  {tau:>5.3f}   {p:>10.2%}   {acc:>11.2%}   {exp:>8.2%}{flag}")

    if best[0] is None:
        print("\nNo candidate found (empty score distribution).")
        return 1

    print(f"\nBest τ for {args.path} path: {best[0]:.3f}  "
          f"(expected acc {best[1]:.2%})")
    return 0


def _load_pairs(globs: list[str]) -> list[tuple[float, bool]]:
    """Pull (top_score, correct) tuples from one or more eval JSONLs.

    Each line must be a dict shaped ``{..., "correct": bool, "extras":
    {"top_score": float}}``. Lines missing either field are skipped.
    """
    pairs: list[tuple[float, bool]] = []
    for pattern in globs:
        for path in Path().glob(pattern):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    extras = rec.get("extras") or {}
                    score  = extras.get("top_score")
                    if score is None:
                        continue
                    pairs.append((float(score), bool(rec.get("correct", False))))
    return pairs


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
