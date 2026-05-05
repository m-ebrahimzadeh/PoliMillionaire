"""Profile phase-level latency for all strategies.

Usage:
    python scripts/profile_strategies.py --mock --n 20

Output:
    data/eval/profile_report.json
    Printed table: strategy | tokenise_ms | forward_ms | decode_ms | retrieval_ms | total_ms
"""
from __future__ import annotations

import argparse, json, os, sys, textwrap, time
from pathlib import Path

# ── allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from polimibot.eval.gold_set import load_gold_set
from polimibot.eval.evaluator import evaluate_strategy


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock",  action="store_true")
    p.add_argument("--n",     type=int, default=50,
                   help="Questions to sample from gold set")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    return p.parse_args()


def main() -> None:
    args = _parse()
    gold = load_gold_set()
    sample = gold[:args.n]   # deterministic slice — first N items

    if args.mock:
        from polimibot.models.mock import MockLLM
        llm = MockLLM(name="mock", correctness=0.6)
    else:
        from polimibot.models.llm import LLM, LLMSpec
        llm = LLM.load(LLMSpec(model_id=args.model))

    from polimibot.strategies.llm_baseline import BaselineLLMStrategy
    from polimibot.strategies.rag_strategy import RAGStrategy
    from polimibot.strategies.agent_strategy import AgentStrategy

    strategies = {
        "baseline": BaselineLLMStrategy(llm),
        "agent":    AgentStrategy(llm),
    }

    # RAG needs an index — skip gracefully if not built
    try:
        from polimibot.rag.retriever import Retriever
        from polimibot.rag.index import FAISSIndex
        from polimibot import PATHS
        idx = FAISSIndex.load(PATHS.faiss_index, PATHS.chunk_store)
        strategies["rag"] = RAGStrategy(llm, Retriever(idx))
    except FileNotFoundError:
        print("[warn] RAG index not found — skipping RAG profile")

    results = {}
    for name, strat in strategies.items():
        print(f"\nProfiling {name!r} on {len(sample)} questions …")
        t0 = time.perf_counter()
        report = evaluate_strategy(strat, sample, progress=True)
        elapsed = time.perf_counter() - t0
        results[name] = {
            "accuracy":    round(report.accuracy, 4),
            "latency_p50": round(report.latency_p50, 3),
            "latency_p95": round(report.latency_p95, 3),
            "wall_clock_s": round(elapsed, 1),
        }

    # Pretty table
    header = f"{'strategy':<12} {'acc':>5} {'p50':>6} {'p95':>6} {'wall':>7}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<12} {r['accuracy']:>5.3f} {r['latency_p50']:>6.2f}s "
              f"{r['latency_p95']:>6.2f}s {r['wall_clock_s']:>6.1f}s")

    out = Path("data/eval/profile_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()