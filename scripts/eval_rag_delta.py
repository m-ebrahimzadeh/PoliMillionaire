#!/usr/bin/env python
"""RAG quality delta report — baseline vs. feat/rag-quality-improvements.

Computes Recall@k and MRR from stored run JSONL files (or a gold set) and
prints a side-by-side comparison table so you can see whether the commits
on this branch regress or improve retrieval.

Usage
-----
  # Compare two existing run files (one from main, one from the branch):
  python scripts/eval_rag_delta.py \\
      --baseline  data/runs/rag_baseline.jsonl \\
      --improved  data/runs/rag_improved.jsonl \\
      --gold      data/eval/gold_set.jsonl \\
      --k         3

  # Quick smoke-run with a tiny inline corpus (no files needed):
  python scripts/eval_rag_delta.py --smoke

Run file format
---------------
Each line is a JSON object with at least:
  {
    "question": "...",
    "gold_source": "Julius Caesar",   # Wikipedia article title that contains the answer
    "passages": [
      {"source": "Julius Caesar", "chunk_id": 0, "score": 0.91},
      ...
    ]
  }

The gold_source field can also come from a separate gold-set JSONL
(``--gold``) keyed on the question text, which is merged at load time.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional


# ── Metric helpers ─────────────────────────────────────────────────────────

def recall_at_k(
    passages: list[dict],
    gold_source: str,
    k: int,
) -> float:
    """1.0 if ``gold_source`` appears in the top-k passages, else 0.0."""
    for p in passages[:k]:
        if p.get("source") == gold_source:
            return 1.0
    return 0.0


def reciprocal_rank(
    passages: list[dict],
    gold_source: str,
    k: int,
) -> float:
    """1/rank if ``gold_source`` appears in the top-k passages, else 0.0."""
    for rank, p in enumerate(passages[:k], start=1):
        if p.get("source") == gold_source:
            return 1.0 / rank
    return 0.0


# ── Data loading ───────────────────────────────────────────────────────────

def load_run(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_gold(path: Path) -> dict[str, str]:
    """Load gold set → {question_text: gold_source}."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("question", "")
            gs = rec.get("gold_source") or rec.get("answer_source", "")
            if q and gs:
                mapping[q] = gs
    return mapping


def merge_gold(records: list[dict], gold: dict[str, str]) -> list[dict]:
    """Attach gold_source from the gold dict when not already present."""
    out = []
    for rec in records:
        if "gold_source" not in rec or not rec["gold_source"]:
            rec = dict(rec)
            rec["gold_source"] = gold.get(rec.get("question", ""), "")
        out.append(rec)
    return out


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(records: list[dict], k: int) -> dict:
    """Compute Recall@k and MRR@k over a run file.

    Records without a gold_source are skipped (counted in ``n_skipped``).
    """
    n_total = 0
    n_skipped = 0
    recall_sum = 0.0
    rr_sum = 0.0

    for rec in records:
        gold = rec.get("gold_source", "")
        passages = rec.get("passages", [])
        # Also support the nested extras structure from the runner.
        if not passages and "extras" in rec:
            passages = rec["extras"].get("passages", [])
        if not gold:
            n_skipped += 1
            continue
        n_total += 1
        recall_sum += recall_at_k(passages, gold, k)
        rr_sum     += reciprocal_rank(passages, gold, k)

    if n_total == 0:
        return {
            "n_total": 0, "n_skipped": n_skipped,
            f"recall@{k}": float("nan"),
            f"mrr@{k}": float("nan"),
        }
    return {
        "n_total":       n_total,
        "n_skipped":     n_skipped,
        f"recall@{k}":   round(recall_sum / n_total, 4),
        f"mrr@{k}":      round(rr_sum     / n_total, 4),
    }


# ── Report printing ────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "   N/A"
    if isinstance(v, float):
        return f"{v:6.4f}"
    return str(v)


def print_delta_table(
    baseline_metrics: dict,
    improved_metrics: dict,
    k: int,
    baseline_label: str = "baseline (main)",
    improved_label: str = "improved (branch)",
) -> None:
    recall_key = f"recall@{k}"
    mrr_key    = f"mrr@{k}"

    b_recall = baseline_metrics.get(recall_key, float("nan"))
    i_recall = improved_metrics.get(recall_key, float("nan"))
    b_mrr    = baseline_metrics.get(mrr_key,    float("nan"))
    i_mrr    = improved_metrics.get(mrr_key,    float("nan"))

    d_recall = (i_recall - b_recall) if not (math.isnan(i_recall) or math.isnan(b_recall)) else float("nan")
    d_mrr    = (i_mrr    - b_mrr)    if not (math.isnan(i_mrr)    or math.isnan(b_mrr))    else float("nan")

    def delta_str(d: float) -> str:
        if math.isnan(d):
            return "    N/A"
        sign = "+" if d >= 0 else ""
        colour = "\033[32m" if d > 0 else ("\033[31m" if d < 0 else "")
        reset  = "\033[0m"  if d != 0 else ""
        return f"{colour}{sign}{d:+.4f}{reset}"

    col = 28
    sep = "─" * (col * 3 + 8)
    print()
    print("  RAG quality delta report")
    print(sep)
    print(f"  {'Metric':<14}  {baseline_label:>{col}}  {improved_label:>{col}}  {'Δ':>10}")
    print(sep)
    print(f"  {recall_key:<14}  {_fmt(b_recall):>{col}}  {_fmt(i_recall):>{col}}  {delta_str(d_recall):>10}")
    print(f"  {mrr_key:<14}  {_fmt(b_mrr):>{col}}  {_fmt(i_mrr):>{col}}  {delta_str(d_mrr):>10}")
    print(sep)
    print(f"  n_total (base)   : {baseline_metrics.get('n_total', '?')}")
    print(f"  n_total (impr)   : {improved_metrics.get('n_total', '?')}")
    print(f"  n_skipped (base) : {baseline_metrics.get('n_skipped', '?')}")
    print(f"  n_skipped (impr) : {improved_metrics.get('n_skipped', '?')}")
    print()


def _near(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) < tol


# ── Smoke test (no files needed) ──────────────────────────────────────────

def _smoke_test() -> None:
    """Verify the metric helpers on a tiny hand-crafted dataset.

    Baseline:
      Q1: gold=A, passages=[B, A] → recall@3=1, MRR=0.5 (rank 2)
      Q2: gold=X, passages=[Y]    → recall@3=0, MRR=0.0
      avg recall@3 = 0.5,  avg MRR = 0.25

    Improved:
      Q1: gold=A, passages=[A]    → recall@3=1, MRR=1.0 (rank 1)
      Q2: gold=X, passages=[X]    → recall@3=1, MRR=1.0 (rank 1)
      avg recall@3 = 1.0,  avg MRR = 1.0
    """
    records_base = [
        {"question": "Q1", "gold_source": "A",
         "passages": [{"source": "B", "chunk_id": 0, "score": 0.9},
                      {"source": "A", "chunk_id": 0, "score": 0.8}]},
        {"question": "Q2", "gold_source": "X",
         "passages": [{"source": "Y", "chunk_id": 0, "score": 0.7}]},
    ]
    records_impr = [
        {"question": "Q1", "gold_source": "A",
         "passages": [{"source": "A", "chunk_id": 0, "score": 0.95}]},
        {"question": "Q2", "gold_source": "X",
         "passages": [{"source": "X", "chunk_id": 0, "score": 0.85}]},
    ]

    base = evaluate(records_base, k=3)
    impr = evaluate(records_impr, k=3)

    print("\n[smoke] baseline metrics:", base)
    print("[smoke] improved metrics:", impr)
    assert base["recall@3"] == 0.5,  f"Expected 0.5, got {base['recall@3']}"
    assert impr["recall@3"] == 1.0,  f"Expected 1.0, got {impr['recall@3']}"
    assert _near(base["mrr@3"], 0.25), f"Expected 0.25, got {base['mrr@3']}"
    assert _near(impr["mrr@3"], 1.0),  f"Expected 1.0, got {impr['mrr@3']}"
    print_delta_table(base, impr, k=3,
                      baseline_label="smoke-base",
                      improved_label="smoke-impr")
    print("[smoke] PASSED")


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG quality delta report: recall@k and MRR@k for two run files.",
    )
    parser.add_argument("--baseline",  type=Path, help="Baseline run JSONL (main branch)")
    parser.add_argument("--improved",  type=Path, help="Improved run JSONL (feat branch)")
    parser.add_argument("--gold",      type=Path, help="Optional gold-set JSONL to attach gold_source")
    parser.add_argument("--k",         type=int,  default=3, help="Recall@k cutoff (default 3)")
    parser.add_argument("--smoke",     action="store_true",  help="Run smoke test with inline data")
    args = parser.parse_args(argv)

    if args.smoke:
        _smoke_test()
        return 0

    if args.baseline is None or args.improved is None:
        parser.error("--baseline and --improved are required (or use --smoke).")

    gold: dict[str, str] = {}
    if args.gold:
        gold = load_gold(args.gold)

    base_records = load_run(args.baseline)
    impr_records = load_run(args.improved)

    if gold:
        base_records = merge_gold(base_records, gold)
        impr_records = merge_gold(impr_records, gold)

    base_metrics = evaluate(base_records, args.k)
    impr_metrics = evaluate(impr_records, args.k)

    print_delta_table(
        base_metrics, impr_metrics,
        k=args.k,
        baseline_label=str(args.baseline.name),
        improved_label=str(args.improved.name),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ── Inline pytest-compatible tests (importable) ────────────────────────────

def _approx(v: float, tol: float = 1e-4) -> bool:
    """Simple float comparison for the smoke-test assertions."""
    return True  # placeholder


def pytest_approx(expected):
    """Allow the smoke test to use a rough approx without importing pytest."""
    class _A:
        def __init__(self, v): self.v = v
        def __eq__(self, other): return abs(other - self.v) < 1e-4
        def __repr__(self): return f"approx({self.v})"
    return _A(expected)
