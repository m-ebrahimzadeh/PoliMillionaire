"""Ablation: ToolStrategy vs Baseline on the maths subset of the gold set.

Usage:
    python scripts/eval_tools.py --mock     # CPU smoke test
    python scripts/eval_tools.py            # real model on Colab GPU
"""
from __future__ import annotations

import argparse
import sys

from polimibot.config import PATHS, Category
from polimibot.eval.evaluator import evaluate_strategy
from polimibot.eval.gold_set import load_gold_set
from polimibot.models.mock import MockLLM
from polimibot.prompts.templates import PromptStyle
from polimibot.strategies.llm_baseline import BaselineLLMStrategy
from polimibot.strategies.tool_strategy import ToolStrategy
from polimibot.tools.maths_tool import MathsTool


def main() -> int:
    args = _parse_args()
    PATHS.ensure()

    gold_path = PATHS.eval_dir / "gold_set.jsonl"
    if not gold_path.exists():
        print(f"Gold set not found at {gold_path}.\nRun: python scripts/build_gold_set.py",
              file=sys.stderr)
        return 1

    gold = load_gold_set(gold_path)
    maths_gold = [g for g in gold if g.category == Category.MATHS]

    print(f"Total gold: {len(gold)} items")
    print(f"Maths gold: {len(maths_gold)} items")

    if not maths_gold:
        print("No maths items in gold set — play some maths games first.", file=sys.stderr)
        return 1

    # Build LLM once — shared by both strategies (controlled comparison)
    if args.mock:
        llm = MockLLM(name="mock", correctness=0.6)
        print("Using MockLLM\n")
    else:
        from polimibot.models.llm import LLM, LLMSpec
        llm = LLM.load(LLMSpec(model_id=args.model, load_in_4bit=not args.no_4bit))

    baseline = BaselineLLMStrategy(llm, style=PromptStyle.ZERO_SHOT, use_score_options=True)
    tool_strat = ToolStrategy(tools=[MathsTool()], fallback=baseline)

    # ── Evaluate both on the maths subset ──
    print("=" * 55)
    print("Evaluating: BASELINE (maths only)")
    baseline.warm_up()
    report_base = evaluate_strategy(baseline, maths_gold, verbose=True)
    report_base.print_summary()
    report_base.save(PATHS.eval_dir / f"report_{baseline.name}_maths.json")

    print("=" * 55)
    print("Evaluating: TOOL STRATEGY (maths only)")
    # No warm_up needed: ToolStrategy.warm_up() would re-warm the already-warm LLM
    report_tool = evaluate_strategy(tool_strat, maths_gold, verbose=True)
    report_tool.print_summary()
    report_tool.save(PATHS.eval_dir / f"report_{tool_strat.name}_maths.json")

    baseline.shutdown()

    # ── How many did the tool actually answer? ──
    tool_answered = sum(
        1 for s in report_tool.samples
        if s.extras.get("tool") == "maths_tool"  # note: need extras on EvalSample
    )
    # Simpler proxy: high-confidence answers are tool answers
    tool_answered = sum(1 for s in report_tool.samples if s.confidence > 0.95)
    print(f"\nTool coverage: {tool_answered}/{len(maths_gold)} "
          f"({tool_answered/max(len(maths_gold),1):.1%}) questions answered by tool")
    print(f"Accuracy delta: {report_tool.accuracy - report_base.accuracy:+.1%}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock",   action="store_true")
    p.add_argument("--model",  default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--no-4bit", action="store_true", dest="no_4bit")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())