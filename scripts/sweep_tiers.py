"""Grid-sweep TierBreakpoints to find accuracy/latency Pareto front.

Sweeps easy_max ∈ {3,5,7} × medium_max ∈ {8,10,12}.
For each config: runs evaluate_strategy, appends one JSON line to
data/eval/tier_sweep.jsonl.

Usage:
    python scripts/sweep_tiers.py --mock --n 50
"""
from __future__ import annotations

import argparse, json, itertools, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polimibot.eval.gold_set import load_gold_set
from polimibot.eval.evaluator import evaluate_strategy
from polimibot.strategies.tiered_strategy import TieredStrategy, TierBreakpoints


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock",  action="store_true")
    p.add_argument("--n",     type=int, default=50)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--easy",  nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--medium",nargs="+", type=int, default=[8, 10, 12])
    return p.parse_args()


def main() -> None:
    args = _parse()
    gold  = load_gold_set()[:args.n]

    if args.mock:
        from polimibot.models.mock import MockLLM
        llm = MockLLM(name="mock", correctness=0.6)
    else:
        from polimibot.models.llm import LLM, LLMSpec
        llm = LLM.load(LLMSpec(model_id=args.model))

    out = Path("data/eval/tier_sweep.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    for easy_max, medium_max in itertools.product(args.easy, args.medium):
        if easy_max >= medium_max:
            continue   # nonsensical config
        bp = TierBreakpoints(easy_max=easy_max, medium_max=medium_max)
        strat = TieredStrategy(llm, breakpoints=bp)
        report = evaluate_strategy(strat, gold, progress=False)
        row = {
            "easy_max":   easy_max,
            "medium_max": medium_max,
            "accuracy":   round(report.accuracy, 4),
            "p50":        round(report.latency_p50, 3),
            "p95":        round(report.latency_p95, 3),
        }
        print(row)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"\nSweep complete → {out}")


if __name__ == "__main__":
    main()