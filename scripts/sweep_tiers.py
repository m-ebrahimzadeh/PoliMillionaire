"""Grid-sweep TierBreakpoints to find an accuracy/latency Pareto front.

Sweeps easy_max_level × medium_max_level (and optionally
escalation_threshold). For each config: evaluates the resulting
TieredStrategy on a slice of the gold set and appends one JSON line to
data/eval/tier_sweep.jsonl.

Strategies wired into each tier:
  easy   → BaselineLLMStrategy   (cheapest)
  medium → RAGStrategy            (LLM + retrieval)
  hard   → EnsembleStrategy       (baseline + RAG, weighted)

The same llm / retriever objects are shared across every config — only
the TieredStrategy wrapper differs between sweeps, so warm_up runs once.

Usage:
    python scripts/sweep_tiers.py --mock --n 50
    python scripts/sweep_tiers.py --easy 3 5 7 --medium 8 10 12
    python scripts/sweep_tiers.py --escalation-threshold 0.0 0.1 0.2
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polimibot.config import PATHS
from polimibot.eval.evaluator import evaluate_strategy
from polimibot.eval.gold_set import load_gold_set
from polimibot.strategies.ensemble_strategy import EnsembleStrategy
from polimibot.strategies.llm_baseline import BaselineLLMStrategy
from polimibot.strategies.rag_strategy import RAGStrategy
from polimibot.strategies.tiered_strategy import TierBreakpoints, TieredStrategy


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock",   action="store_true")
    p.add_argument("--n",      type=int, default=50,
                   help="Questions to sample from gold set (deterministic prefix)")
    p.add_argument("--model",  default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--easy",   nargs="+", type=int, default=[3, 5, 7],
                   help="easy_max_level values to sweep")
    p.add_argument("--medium", nargs="+", type=int, default=[8, 10, 12],
                   help="medium_max_level values to sweep")
    p.add_argument(
        "--escalation-threshold", nargs="+", type=float,
        default=[None],
        dest="escalation_threshold",
        help="Margin thresholds to sweep (use 'None' or omit for no escalation; "
             "pass numbers like 0.1 0.15 0.2 for an escalation sweep)",
    )
    return p.parse_args()


def _maybe_none(v: float | None) -> float | None:
    """argparse can't natively produce None mixed with floats — sentinel < 0 means 'off'."""
    return None if (v is None or v < 0) else v


def main() -> None:
    args = _parse()
    PATHS.ensure()

    gold_path = PATHS.eval_dir / "gold_set.jsonl"
    if not gold_path.exists():
        print(f"Gold set not found: {gold_path}\n"
              f"Run: python scripts/build_gold_set.py", file=sys.stderr)
        sys.exit(1)

    gold = load_gold_set(gold_path)[: args.n]
    print(f"Sweeping over {len(gold)} gold items")

    # ── llm + retriever (shared across every config) ────────────────────
    if args.mock:
        from polimibot.models.mock import MockLLM
        llm = MockLLM(name="mock", correctness=0.6)

        class _NullRetriever:
            n_chunks = 0
            def retrieve(self, q, k=3, *, category=None): return []
        retriever = _NullRetriever()  # type: ignore[assignment]
    else:
        index_path = PATHS.cache_dir / "knowledge"
        if not index_path.with_suffix(".faiss").exists():
            print(f"RAG index missing — run: python scripts/build_rag_index.py",
                  file=sys.stderr)
            sys.exit(1)
        from polimibot.models.llm import LLM, LLMSpec
        from polimibot.rag.embedder import EmbedderSpec
        from polimibot.rag.retriever import Retriever
        llm = LLM.load(LLMSpec(model_id=args.model))
        retriever = Retriever.from_saved(index_path, embedder_spec=EmbedderSpec())

    # ── per-tier strategies (shared, warmed once) ───────────────────────
    easy   = BaselineLLMStrategy(llm)
    medium = RAGStrategy(llm, retriever, k=3)                     # type: ignore[arg-type]
    hard   = EnsembleStrategy([easy, medium], weights=[1.0, 1.2])

    easy.warm_up()   # one warm-up covers all three (same underlying LLM)

    # ── sweep ────────────────────────────────────────────────────────────
    out = PATHS.eval_dir / "tier_sweep.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    configs = list(itertools.product(args.easy, args.medium, args.escalation_threshold))
    print(f"Configs to evaluate: {len(configs)}")

    with out.open("a") as f:
        for easy_max, medium_max, esc in configs:
            if easy_max >= medium_max:
                continue   # nonsensical — easy must end before medium begins
            bp = TierBreakpoints(
                easy_max_level=easy_max,
                medium_max_level=medium_max,
            )
            strat = TieredStrategy(
                easy=easy, medium=medium, hard=hard,
                breakpoints=bp,
                escalation_threshold=_maybe_none(esc),
            )
            report = evaluate_strategy(strat, gold, verbose=False)
            row = {
                "easy_max_level":       easy_max,
                "medium_max_level":     medium_max,
                "escalation_threshold": _maybe_none(esc),
                "accuracy":             round(report.accuracy, 4),
                "ece":                  round(report.ece, 4),
                "latency_p50":          round(report.latency_p50, 3),
                "latency_p95":          round(report.latency_p95, 3),
                "n":                    report.n_total,
            }
            print(row)
            f.write(json.dumps(row) + "\n")
            f.flush()

    print(f"\nSweep complete → {out}")


if __name__ == "__main__":
    main()
