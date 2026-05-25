"""Multi-game session helper.

A session is a sequence of games — usually one per competition, sometimes
several. The runner stays focused on a single game; this helper composes
games into a session, manages the strategy lifecycle (warm_up / shutdown),
and writes one JSONL run log per session.

Folded out of polimibot.runner so the runner remains a thin spine. Imported
by scripts/play_baseline.py via sibling-path manipulation.
"""
from __future__ import annotations

import time
from typing import Any

from polimibot import PATHS
from polimibot import config as _config
from polimibot.logging_utils import RunLogger
from polimibot.runner import GameResult, play_game
from polimibot.strategies import Strategy


def play_session(
    client: Any,
    competition_ids: list[int],
    strategy: Strategy,
    *,
    games_per_competition: int = 1,
    run_id: str = "run",
    verbose: bool = True,
    mode: str | None = None,
    transcriber: Any = None,
) -> list[GameResult]:
    """Play multiple games, log everything to a single JSONL file.

    Returns a list of :class:`GameResult` (one per game). The strategy is
    warmed up once before the first game and shut down once after the last
    — even if a game raises mid-session.

    Args:
        mode: game mode — ``"text"`` or ``"speech"``. When ``None`` the value
            from ``RUNTIME.game_mode`` is used.
        transcriber: a :class:`~polimibot.models.speech.SpeechTranscriber`
            instance. Required when ``mode="speech"``.
    """
    PATHS.ensure()
    results: list[GameResult] = []

    with RunLogger(PATHS.runs_dir, run_id=run_id, extra={"strategy": strategy.name}) as logger:
        strategy.warm_up()                  # compiles CUDA kernels once, here
        try:
            for cid in competition_ids:
                for _ in range(games_per_competition):
                    result = play_game(
                        client, cid, strategy,
                        logger=logger, verbose=verbose,
                        mode=mode, transcriber=transcriber,
                    )
                    results.append(result)
                    time.sleep(_config.RUNTIME.api_min_delay_seconds)  # inter-game pause
        finally:
            strategy.shutdown()             # release GPU memory, always runs
    return results
