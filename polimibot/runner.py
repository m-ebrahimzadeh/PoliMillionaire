"""Game-loop runner. The spine that wires Adapter + Strategy + Logger together.

One public entry: :func:`play_game`. Watchdog enforces per-question deadline,
throttle keeps us polite toward the proof-of-concept server.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

from .config import CATEGORIES, PATHS, RUNTIME, Category
from .game import GameAdapter, GameQuestion
from .logging_utils import GameSummaryRecord, NullLogger, QuestionRecord, RunLogger
from .strategies import Strategy, StrategyInput, StrategyOutput


# ──────────────────────────── Result type ────────────────────────────

@dataclass(frozen=True)
class GameResult:
    """End-of-game snapshot, returned to the caller."""
    competition_id: int
    competition_name: str
    session_id: int
    final_level: int
    earned_amount: float
    n_questions: int
    n_correct: int
    finished_normally: bool
    strategy_name: str
    elapsed_seconds: float

    @property
    def accuracy(self) -> float:
        return (self.n_correct / self.n_questions) if self.n_questions else 0.0


# ──────────────────────────── Watchdog ────────────────────────────

class _DeadlineExceeded(Exception):
    """Internal sentinel — strategy ran past its budget."""


def _call_with_deadline(
    strategy: Strategy, inp: StrategyInput, deadline_seconds: float
) -> StrategyOutput:
    """Run strategy in a daemon thread; raise _DeadlineExceeded on overrun.

    The runaway thread is *not* killed — Python can't safely interrupt one
    in the middle of a torch/CUDA call. It finishes in the background and
    its result is discarded.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["out"] = strategy.answer(inp)
        except BaseException as exc:  # noqa: BLE001
            box["err"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(deadline_seconds)
    if th.is_alive():
        raise _DeadlineExceeded(f"strategy exceeded {deadline_seconds:.1f}s")
    if "err" in box:
        raise box["err"]
    return box["out"]


# ──────────────────────────── Throttle ────────────────────────────

class _Pacer:
    """Ensures at least `min_delay` seconds between waits()."""
    def __init__(self, min_delay: float) -> None:
        self._min_delay = min_delay
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if 0 < elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last = time.monotonic()


# ──────────────────────────── Runner ────────────────────────────

def _category_for(competition_id: int) -> Optional[Category]:
    info = CATEGORIES.get(competition_id)
    return info.category if info else None


# Type alias: anything with the right shape passes. NullLogger and RunLogger both qualify.
LoggerLike = Union[RunLogger, NullLogger]


def play_game(
    client: Any,
    competition_id: int,
    strategy: Strategy,
    *,
    logger: Optional[LoggerLike] = None,
    fallback_index: int = 0,
    verbose: bool = True,
) -> GameResult:
    """Play one full game with the given strategy.

    Args:
        client: a logged-in ``MillionaireClient``.
        competition_id: which competition to play.
        strategy: must already be warmed up (or call .warm_up() inside this fn —
            we choose the former so the runner stays thin).
        logger: optional :class:`RunLogger`. Defaults to NullLogger.
        fallback_index: option submitted on timeout / exception.
        verbose: print per-question status to stdout.

    Returns:
        :class:`GameResult` with totals.
    """
    log: LoggerLike = logger if logger is not None else NullLogger()
    info = CATEGORIES.get(competition_id)
    competition_name = info.display_name if info else f"comp_{competition_id}"
    category = _category_for(competition_id)

    pacer = _Pacer(RUNTIME.api_min_delay_seconds)
    t_game_start = time.monotonic()

    if verbose:
        print(f"\n=== {competition_name} | strategy={strategy.name} ===")

    game = GameAdapter(client, competition_id=competition_id)
    n_q = n_correct = 0
    final_level = game.current_level

    while not game.is_over:
        q: Optional[GameQuestion] = game.current_question
        if q is None:
            break

        # --- build strategy input ---
        time_left = game.time_remaining_seconds  # may be None
        # Hard cutoff = min(server-time-left minus margin, our config cap).
        if time_left is not None:
            budget = max(1.0, min(time_left - 2.0, RUNTIME.hard_cutoff_seconds))
        else:
            budget = RUNTIME.hard_cutoff_seconds

        inp = StrategyInput(
            question=q.text,
            options=q.options,
            level=q.level,
            max_level=15,
            category=category,
            competition_id=competition_id,
            time_budget_seconds=budget,
        )

        if verbose:
            print(f"\n--- L{q.level} (budget {budget:.1f}s) ---\n{q.text}")
            for letter, opt in zip("ABCD", q.options):
                print(f"  {letter}. {opt}")

        # --- run strategy under watchdog ---
        t0 = time.monotonic()
        out: Optional[StrategyOutput]
        timed_out = False
        try:
            out = _call_with_deadline(strategy, inp, budget)
        except _DeadlineExceeded:
            timed_out = True
            out = None
            if verbose:
                print(f"  ! strategy timed out; submitting fallback index={fallback_index}")
        except Exception as exc:  # noqa: BLE001
            out = None
            if verbose:
                print(f"  ! strategy raised {type(exc).__name__}: {exc}; submitting fallback")

        elapsed = time.monotonic() - t0
        # Strategy can be None (timeout / exception) or non-None with
        # is_abstain=True (e.g. parse failure, all-abstain ensemble). Both
        # routes use fallback_index — abstention is logged via the strategy
        # output, but the runner still has to submit *something*.
        if out is None or out.is_abstain:
            chosen_idx = fallback_index
        else:
            chosen_idx = out.chosen_index
        chosen_idx = max(0, min(chosen_idx, len(q.options) - 1))  # safety clamp

        # --- submit, throttled ---
        pacer.wait()
        outcome = game.submit_answer(chosen_idx)

        # --- log ---
        n_q += 1
        if outcome.correct is True:
            n_correct += 1
        final_level = outcome.reached_level or final_level

        log.log_question(QuestionRecord(
            session_id=game.session_id,
            competition_id=competition_id,
            competition_name=competition_name,
            level=q.level,
            question_text=q.text,
            options=list(q.options),
            chosen_index=chosen_idx,
            chosen_text=q.options[chosen_idx],
            correct=outcome.correct,
            timed_out=timed_out or outcome.timed_out,
            latency_seconds=round(elapsed, 4),
            strategy=strategy.name,
            confidence=(out.confidence if out is not None else None),
            probs=(out.extras.get("probs") if out is not None else None),
            extras={"rationale": out.rationale} if (out and out.rationale) else {},
        ))

        if verbose:
            mark = "✓" if outcome.correct else ("·" if outcome.correct is None else "✗")
            print(f"  → {'ABCD'[chosen_idx]}  {mark}  ({elapsed:.2f}s)")

        if outcome.game_over:
            break

    summary = game.summary()
    log.log_summary(GameSummaryRecord(
        session_id=summary.session_id,
        competition_id=competition_id,
        competition_name=competition_name,
        final_level=summary.final_level,
        earned_amount=summary.earned_amount,
        n_questions=n_q,
        n_correct=n_correct,
        finished_normally=summary.finished_normally,
    ))

    return GameResult(
        competition_id=competition_id,
        competition_name=competition_name,
        session_id=summary.session_id,
        final_level=summary.final_level,
        earned_amount=summary.earned_amount,
        n_questions=n_q,
        n_correct=n_correct,
        finished_normally=summary.finished_normally,
        strategy_name=strategy.name,
        elapsed_seconds=round(time.monotonic() - t_game_start, 3),
    )







def play_session(
    client: Any,
    competition_ids: list[int],
    strategy: Strategy,
    *,
    games_per_competition: int = 1,
    run_id: str = "run",
    verbose: bool = True,
) -> list[GameSummary]:
    """Play multiple games, log everything to a single JSONL file."""
    PATHS.ensure()
    summaries: list[GameSummary] = []

    with RunLogger(PATHS.runs_dir, run_id=run_id, extra={"strategy": strategy.name}) as logger:
        strategy.warm_up()                  # ← compiles CUDA kernels once, here
        try:
            for cid in competition_ids:
                for _ in range(games_per_competition):
                    summary = play_game(
                        client, cid, strategy,
                        logger=logger, verbose=verbose,
                    )
                    summaries.append(summary)
                    time.sleep(RUNTIME.api_min_delay_seconds)  # inter-game pause
        finally:
            strategy.shutdown()             # ← release GPU memory, always runs
    return summaries