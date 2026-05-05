"""Three-way ablation on maths gold: Baseline vs ToolStrategy vs AgentStrategy.

Usage:
    python scripts/eval_agent.py --mock        # CPU smoke test
    python scripts/eval_agent.py               # real model on Colab GPU
"""
from __future__ import annotations

import argparse
import sys

from polimibot.config import PATHS, Category
from polimibot.eval.evaluator import EvalReport, evaluate_strategy
from polimibot.eval.gold_set import load_gold_set
from polimibot.models.mock import MockLLM
from polimibot.prompts.templates import PromptStyle
from polimibot.strategies.agent_strategy import AgentStrategy
from polimibot.strategies.llm_baseline import BaselineLLMStrategy
from polimibot.strategies.tool_strategy import ToolStrategy
from polimibot.tools.maths_tool import MathsTool


def main() -> int:
    args = _parse_args()
    PATHS.ensure()

    gold_path = PATHS.eval_dir / "gold_set.jsonl"
    if not gold_path.exists():
        print(f"Gold set not found: {gold_path}\nRun: python scripts/build_gold_set.py",
              file=sys.stderr)
        return 1

    gold = load_gold_set(gold_path)
    maths = [g for g in gold if g.category == Category.MATHS]
    print(f"Maths gold items: {len(maths)}")
    if not maths:
        print("No maths items — play some maths games first.", file=sys.stderr)
        return 1

    # ── Load LLM once (controlled comparison) ────────────────────────────
    if args.mock:
        llm = MockLLM(name="mock", correctness=0.6)
        print("Using MockLLM\n")
    else:
        from polimibot.models.llm import LLM, LLMSpec
        llm = LLM.load(LLMSpec(model_id=args.model, load_in_4bit=not args.no_4bit))

    # ── Three strategies, one LLM ─────────────────────────────────────────
    baseline  = BaselineLLMStrategy(llm, style=PromptStyle.ZERO_SHOT)
    tool_strat = ToolStrategy(tools=[MathsTool()], fallback=baseline)
    agent     = AgentStrategy(llm, max_iterations=args.max_iter)

    baseline.warm_up()   # warms the LLM once; all three share it

    reports: list[EvalReport] = []
    for strat in (baseline, tool_strat, agent):
        print(f"\n{'='*55}\nEvaluating: {strat.name}")
        r = evaluate_strategy(strat, maths, verbose=True)
        r.print_summary()
        r.save(PATHS.eval_dir / f"report_{strat.name}_maths.json")
        reports.append(r)

    baseline.shutdown()

    # ── Comparison table ──────────────────────────────────────────────────
    _print_table(reports)
    return 0


def _print_table(reports: list[EvalReport]) -> None:
    print(f"\n{'═'*65}")
    print(f"  {'Strategy':<35} {'Acc':>7}  {'ECE':>7}  {'p50':>6}  {'p95':>6}")
    print(f"{'─'*65}")
    for r in reports:
        print(
            f"  {r.strategy_name:<35} {r.accuracy:>7.1%}  "
            f"{r.ece:>7.4f}  {r.latency_p50:>5.2f}s  {r.latency_p95:>5.2f}s"
        )
    print(f"{'═'*65}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock",     action="store_true")
    p.add_argument("--model",    default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--no-4bit",  action="store_true", dest="no_4bit")
    p.add_argument("--max-iter", type=int, default=3, dest="max_iter")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())