"""Full leaderboard: compare all strategies on the complete gold set.

Builds the production tiered strategy and measures it against each
component in isolation. This is the Stage 8 capstone evaluation.

Usage:
    python scripts/eval_tiered.py --mock
    python scripts/eval_tiered.py
    python scripts/eval_tiered.py --escalation-threshold 0.15
"""
from __future__ import annotations

import argparse
import sys

from polimibot.config import PATHS, Category
from polimibot.eval.evaluator import EvalReport, evaluate_strategy
from polimibot.eval.gold_set import load_gold_set
from polimibot.eval.report_io import model_slug, save_report
from polimibot.models.mock import MockLLM
from polimibot.prompts.templates import PromptStyle
from polimibot.strategies.agent_strategy import AgentStrategy
from polimibot.strategies.ensemble_strategy import EnsembleStrategy
from polimibot.strategies.llm_baseline import BaselineLLMStrategy
from polimibot.strategies.rag_strategy import RAGStrategy
from polimibot.strategies.tiered_strategy import TierBreakpoints, TieredStrategy
from polimibot.tools.maths_tool import MathsTool
from polimibot.strategies.tool_strategy import ToolStrategy


def main() -> int:
    args = _parse_args()
    PATHS.ensure()

    gold_path  = PATHS.eval_dir  / "gold_set.jsonl"
    index_path = PATHS.cache_dir / "knowledge"

    if not gold_path.exists():
        print(f"Gold set not found: {gold_path}", file=sys.stderr)
        return 1

    gold = load_gold_set(gold_path)
    print(f"Gold set: {len(gold)} items")
    _print_gold_breakdown(gold)

    # ── LLM and retriever (built once, shared by all) ────────────────────
    if args.mock:
        llm = MockLLM(name="mock", correctness=0.65)
        class _NullRetriever:
            n_chunks = 0
            def retrieve(self, q, k=3, *, category=None): return []
        retriever = _NullRetriever()  # type: ignore[assignment]
        print("\nUsing MockLLM + NullRetriever\n")
    else:
        if not index_path.with_suffix(".faiss").exists():
            print("RAG index missing — run: python scripts/build_rag_index.py",
                  file=sys.stderr)
            return 1
        from polimibot.models.llm import LLM, LLMSpec
        from polimibot.rag.embedder import EmbedderSpec
        from polimibot.rag.retriever import Retriever
        llm = LLM.load(LLMSpec(model_id=args.model, load_in_4bit=not args.no_4bit))
        retriever = Retriever.from_saved(index_path, embedder_spec=EmbedderSpec())

    # ── Assemble all strategies ───────────────────────────────────────────
    style    = PromptStyle.ZERO_SHOT
    baseline = BaselineLLMStrategy(llm, style=style)
    rag      = RAGStrategy(llm, retriever, k=3, style=style)          # type: ignore[arg-type]
    ensemble = EnsembleStrategy([baseline, rag], weights=[1.0, 1.2])
    agent    = AgentStrategy(llm, max_iterations=3)

    tiered = TieredStrategy(
        easy=baseline,
        medium=rag,
        hard=ensemble,
        breakpoints=TierBreakpoints(
            easy_max_level=args.easy_max,
            medium_max_level=args.medium_max,
        ),
        maths_override=agent,
        escalation_threshold=args.escalation_threshold,
    )

    # Warm once via tiered — it deduplicates internally
    tiered.warm_up()

    mslug = model_slug(args.model, mock=args.mock)
    slug_for = {
        baseline: f"baseline_zs__{mslug}",
        rag:      f"rag__{mslug}",
        ensemble: f"ensemble__{mslug}",
        tiered:   f"tiered_final__{mslug}",
    }

    # ── Evaluate ─────────────────────────────────────────────────────────
    strategies_to_eval = [baseline, rag, ensemble, tiered]
    reports: list[EvalReport] = []
    for strat in strategies_to_eval:
        print(f"\n{'='*55}\nEvaluating: {strat.name}")
        r = evaluate_strategy(strat, gold, verbose=True)
        r.print_summary()
        # Persist immediately — leaderboard builder reads these files.
        save_report(r, name=slug_for[strat], eval_dir=PATHS.eval_dir)
        reports.append(r)

    tiered.shutdown()
    _print_leaderboard(reports)
    return 0


def _print_gold_breakdown(gold) -> None:
    from collections import Counter
    by_cat   = Counter(g.category.value if g.category else "unknown" for g in gold)
    by_level = Counter(g.level for g in gold)
    print("  By category:", dict(sorted(by_cat.items())))
    level_dist = {
        "easy(1-5)":   sum(v for k, v in by_level.items() if k <= 5),
        "medium(6-10)":sum(v for k, v in by_level.items() if 6 <= k <= 10),
        "hard(11+)":   sum(v for k, v in by_level.items() if k > 10),
    }
    print("  By tier:    ", level_dist)


def _print_leaderboard(reports: list[EvalReport]) -> None:
    print(f"\n{'═'*72}")
    print("  STRATEGY LEADERBOARD (full gold set)")
    print(f"  {'Strategy':<40} {'Acc':>7}  {'ECE':>7}  {'p50':>6}  {'p95':>6}")
    print(f"{'─'*72}")
    ranked = sorted(reports, key=lambda r: r.accuracy, reverse=True)
    for r in ranked:
        print(
            f"  {r.strategy_name:<40} {r.accuracy:>7.1%}  "
            f"{r.ece:>7.4f}  {r.latency_p50:>5.2f}s  {r.latency_p95:>5.2f}s"
        )
    print(f"{'═'*72}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock",     action="store_true")
    p.add_argument("--model",    default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--no-4bit",  action="store_true", dest="no_4bit")
    p.add_argument("--easy-max", type=int, default=5, dest="easy_max")
    p.add_argument("--medium-max", type=int, default=10, dest="medium_max")
    p.add_argument(
        "--escalation-threshold", type=float, default=None,
        dest="escalation_threshold",
        help="Margin below which to escalate to the next tier (e.g. 0.15)",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())