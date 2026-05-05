"""Ablation: Baseline vs RAG vs Ensemble on hard-tier gold questions.

Filters to level >= min_level (default 8) where model uncertainty is real
and probability fusion's advantage materialises.

Usage:
    python scripts/eval_ensemble.py --mock
    python scripts/eval_ensemble.py
    python scripts/eval_ensemble.py --min-level 5   # if few hard items in gold set
    python scripts/eval_ensemble.py --rag-weight 1.5
"""
from __future__ import annotations

import argparse
import sys

from polimibot.config import PATHS
from polimibot.eval.evaluator import EvalReport, evaluate_strategy
from polimibot.eval.gold_set import load_gold_set
from polimibot.models.mock import MockLLM
from polimibot.prompts.templates import PromptStyle
from polimibot.strategies.ensemble_strategy import EnsembleStrategy
from polimibot.strategies.llm_baseline import BaselineLLMStrategy
from polimibot.strategies.rag_strategy import RAGStrategy


def main() -> int:
    args = _parse_args()
    PATHS.ensure()

    gold_path  = PATHS.eval_dir  / "gold_set.jsonl"
    index_path = PATHS.cache_dir / "knowledge"

    if not gold_path.exists():
        print(f"Gold set not found: {gold_path}\nRun: python scripts/build_gold_set.py",
              file=sys.stderr)
        return 1

    gold = load_gold_set(gold_path)
    hard = [g for g in gold if g.level >= args.min_level]
    print(f"Gold set total: {len(gold)}  |  level >= {args.min_level}: {len(hard)}")

    if not hard:
        print(f"No items at level >= {args.min_level}. Lower --min-level or collect more games.",
              file=sys.stderr)
        return 1

    # ── Build LLM once ────────────────────────────────────────────────────
    if args.mock:
        llm = MockLLM(name="mock", correctness=0.6)
        print("Using MockLLM\n")
    else:
        if not index_path.with_suffix(".faiss").exists():
            print(f"RAG index not found. Run: python scripts/build_rag_index.py",
                  file=sys.stderr)
            return 1
        from polimibot.models.llm import LLM, LLMSpec
        llm = LLM.load(LLMSpec(model_id=args.model, load_in_4bit=not args.no_4bit))

    # ── Build retriever ───────────────────────────────────────────────────
    if args.mock:
        class _NullRetriever:
            n_chunks = 0
            def retrieve(self, query: str, k: int = 3): return []
        retriever = _NullRetriever()   # type: ignore[assignment]
    else:
        from polimibot.rag.embedder import EmbedderSpec
        from polimibot.rag.retriever import Retriever
        retriever = Retriever.from_saved(index_path, embedder_spec=EmbedderSpec())

    # ── Three strategies, one LLM ─────────────────────────────────────────
    style    = PromptStyle.ZERO_SHOT
    baseline = BaselineLLMStrategy(llm, style=style)
    rag      = RAGStrategy(llm, retriever, k=3, style=style)  # type: ignore[arg-type]
    ensemble = EnsembleStrategy(
        [baseline, rag],
        weights=[1.0, args.rag_weight],
        mode="weighted",
    )

    # Warm up once — ensemble.warm_up() deduplicates on object identity
    ensemble.warm_up()

    reports: list[EvalReport] = []
    for strat in (baseline, rag, ensemble):
        print(f"\n{'='*55}\nEvaluating: {strat.name}")
        r = evaluate_strategy(strat, hard, verbose=True)
        r.print_summary()
        r.save(PATHS.eval_dir / f"report_{strat.name}_hard.json")
        reports.append(r)

    ensemble.shutdown()
    _print_table(reports, min_level=args.min_level)
    return 0


def _print_table(reports: list[EvalReport], min_level: int) -> None:
    print(f"\n{'═'*70}")
    print(f"  Hard-tier comparison (level >= {min_level})")
    print(f"  {'Strategy':<38} {'Acc':>7}  {'ECE':>7}  {'p50':>6}  {'p95':>6}")
    print(f"{'─'*70}")
    for r in reports:
        print(
            f"  {r.strategy_name:<38} {r.accuracy:>7.1%}  "
            f"{r.ece:>7.4f}  {r.latency_p50:>5.2f}s  {r.latency_p95:>5.2f}s"
        )
    if len(reports) >= 3:
        delta = reports[2].accuracy - reports[0].accuracy
        print(f"{'─'*70}")
        print(f"  Ensemble vs Baseline delta: {delta:+.1%}")
    print(f"{'═'*70}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock",       action="store_true")
    p.add_argument("--model",      default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--no-4bit",    action="store_true", dest="no_4bit")
    p.add_argument("--min-level",  type=int, default=8, dest="min_level")
    p.add_argument("--rag-weight", type=float, default=1.2, dest="rag_weight",
                   help="Trust weight for RAG relative to baseline (default: 1.2)")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())