"""Run one game per competition with the baseline LLM strategy.

Usage (Colab):
    %run scripts/play_baseline.py

Usage (local, CPU/no-GPU):
    POLIMI_USER=x POLIMI_PASS=y python scripts/play_baseline.py --mock

Reads credentials from env: POLIMI_USER, POLIMI_PASS
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from millionaire_client import MillionaireClient

from polimibot import CATEGORIES, RUNTIME
from polimibot.models.mock import MockLLM
from polimibot.strategies.llm_baseline import BaselineLLMStrategy

# scripts/ is not a package; add it to sys.path so we can import _session
# as a sibling helper without forcing a package layout.
sys.path.insert(0, str(Path(__file__).parent))
from _session import play_session  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true",
                   help="Use MockLLM instead of a real model (no GPU needed)")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model ID (ignored when --mock)")
    p.add_argument("--comp", type=int, default=None,
                   help="Single competition ID to run (default: all 4)")
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization (for CPU testing)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    user = os.environ.get("POLIMI_USER")
    pw   = os.environ.get("POLIMI_PASS")
    if not user or not pw:
        print("Set POLIMI_USER and POLIMI_PASS env vars.", file=sys.stderr)
        return 2

    # ── Build the strategy ─────────────────────────────────────────
    if args.mock:
        llm = MockLLM(name="mock", correctness=0.6)   # 60% accuracy: beats random
        print("Using MockLLM (no GPU required)")
    else:
        from polimibot.models.llm import LLM, LLMSpec
        spec = LLMSpec(
            model_id=args.model,
            load_in_4bit=not args.no_4bit,
        )
        llm = LLM.load(spec)

    strategy = BaselineLLMStrategy(llm)

    # ── Connect and play ───────────────────────────────────────────
    client = MillionaireClient(RUNTIME.api_url)
    client.login(user, pw)

    comp_ids = [args.comp] if args.comp is not None else list(CATEGORIES.keys())
    summaries = play_session(
        client,
        competition_ids=comp_ids,
        strategy=strategy,
        run_id=f"baseline_{llm.name}",
        verbose=True,
    )

    # ── Print summary table ────────────────────────────────────────
    print("\n\n=== Final Results ===")
    print(f"{'Competition':<30} {'Level':>5} {'Earned':>10} {'Acc':>6} {'Time':>6}")
    print("-" * 62)
    for s in summaries:
        print(
            f"{s.competition_name:<30} {s.final_level:>5} "
            f"${s.earned_amount:>9,.0f} {s.accuracy:>5.0%} {s.elapsed_seconds:>5.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())