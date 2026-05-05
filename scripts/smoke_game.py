"""Smoke test: RandomStrategy plays one full game.

Run:  POLIMI_USER=... POLIMI_PASS=... python scripts/smoke_game.py
"""
from __future__ import annotations
import os, sys

from millionaire_client import MillionaireClient
from polimibot import (
    PATHS, RUNTIME, RandomStrategy, RunLogger, play_game,
)

COMPETITION_ID = 0


def main() -> int:
    user, pw = os.environ.get("POLIMI_USER"), os.environ.get("POLIMI_PASS")
    if not user or not pw:
        print("Set POLIMI_USER and POLIMI_PASS.", file=sys.stderr)
        return 2

    PATHS.ensure()
    client = MillionaireClient(RUNTIME.api_url)
    client.login(user, pw)

    strategy = RandomStrategy(seed=0)
    with RunLogger(PATHS.runs_dir, run_id="smoke_random",
                   extra={"strategy": strategy.name}) as log:
        result = play_game(client, COMPETITION_ID, strategy, logger=log)

    print(f"\n=== final ===  L{result.final_level}  "
          f"€{result.earned_amount:.0f}  acc={result.accuracy:.2f}  "
          f"({result.elapsed_seconds:.1f}s total)")
    print(f"log: {log.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())