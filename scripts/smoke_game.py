"""Smoke test: connect, start a game, read the first question, pick A.

Run with:  python scripts/smoke_game.py
Reads username/password from env: POLIMI_USER, POLIMI_PASS
"""
from __future__ import annotations
import os
import sys

from millionaire_client import MillionaireClient
from polimibot import GameAdapter

from polimibot import RUNTIME


COMPETITION_ID = 0  # any valid one; we'll inventory them in a later commit


def main() -> int:
    user = os.environ.get("POLIMI_USER")
    pw = os.environ.get("POLIMI_PASS")
    if not user or not pw:
        print("Set POLIMI_USER and POLIMI_PASS env vars.", file=sys.stderr)
        return 2

    client = MillionaireClient(RUNTIME.api_url)
    client.login(user, pw)

    game = GameAdapter(client, competition_id=COMPETITION_ID)
    print(f"Started session {game.session_id}, level {game.current_level}")
    q = game.current_question
    assert q is not None
    print(f"\nQ (L{q.level}): {q.text}")
    for letter, opt in zip("ABCD", q.options):
        print(f"  {letter}. {opt}")
    print(f"\nTime remaining: {game.time_remaining_seconds:.1f}s")

    # Pick option A and submit, just to exercise the round-trip.
    outcome = game.submit_answer(0)
    print(f"\ncorrect={outcome.correct}  earned={outcome.earned_amount}  "
          f"game_over={outcome.game_over}  reached={outcome.reached_level}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())