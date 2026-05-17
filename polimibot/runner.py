"""Game-loop runner. The spine that wires Adapter + Strategy + Logger together.

One public entry: :func:`play_game`. Watchdog enforces per-question deadline,
throttle keeps us polite toward the proof-of-concept server.
"""
from __future__ import annotations

import contextlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from . import config as _config
from .config import CATEGORIES, PATHS, Category
from .game import GameAdapter, GameQuestion
from .logging_utils import GameSummaryRecord, NullLogger, QuestionRecord, RunLogger
from .observability import print_retrieval_summary
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


def _slug(text: str, max_len: int = 40) -> str:
    """Filesystem-safe short tag derived from a strategy name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return cleaned[:max_len] or "play_game"


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
        logger: a :class:`RunLogger` or :class:`NullLogger`. When ``None``
            (default) a fresh ``RunLogger`` is opened in ``PATHS.runs_dir``
            for the lifetime of this call. **Pass an explicit
            ``NullLogger()`` if you genuinely want runs to not be persisted —
            silent data loss has bitten us before.**
        fallback_index: option submitted on timeout / exception.
        verbose: print per-question status to stdout.

    Returns:
        :class:`GameResult` with totals.
    """
    info = CATEGORIES.get(competition_id)
    competition_name = info.display_name if info else f"comp_{competition_id}"
    category = _category_for(competition_id)

    # Auto-open a RunLogger if none provided. Wrapped in an ExitStack so we
    # close (and fsync) the file on every exit path — including exceptions.
    with contextlib.ExitStack() as stack:
        if logger is None:
            PATHS.ensure()
            log: LoggerLike = stack.enter_context(RunLogger(
                PATHS.runs_dir,
                run_id=f"play_game_c{competition_id}_{_slug(strategy.name)}",
                extra={"strategy": strategy.name, "competition_id": competition_id},
            ))
        else:
            log = logger

        pacer = _Pacer(_config.RUNTIME.api_min_delay_seconds)
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
                budget = max(1.0, min(time_left - 2.0, _config.RUNTIME.hard_cutoff_seconds))
            else:
                budget = _config.RUNTIME.hard_cutoff_seconds

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

            # --- confirm learning (before log, so n_learned is up-to-date) ---
            # If the strategy has an IndexGrower attached and the game server
            # confirmed the answer was correct, promote the buffered live-search
            # article to the permanent in-memory index.  The question_id key
            # matches what RAGStrategy.answer() used when calling buffer().
            _grower = getattr(strategy, "index_grower", None)
            if _grower is not None:
                _question_id = f"lvl_{q.level}"
                if outcome.correct is True:
                    _grower.confirm(_question_id)
                else:
                    _grower.discard(_question_id)

            # --- log ---
            n_q += 1
            if outcome.correct is True:
                n_correct += 1
            final_level = outcome.reached_level or final_level

            # Carry forward everything the strategy emitted under extras
            # (passages, margin, query, n_tool_calls, …) so post-hoc analysis
            # — recall@k harness, error inspection, ablation traces — can be
            # done from the run JSONL without re-running. probs is already
            # promoted to its own QuestionRecord field, so drop it here to
            # avoid duplicating a large dict per row.
            record_extras: dict[str, Any] = {}
            if out is not None and isinstance(out.extras, dict):
                record_extras.update(
                    {k: v for k, v in out.extras.items() if k != "probs"}
                )
            if out is not None and out.rationale:
                record_extras["rationale"] = out.rationale

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
                extras=record_extras,
            ))

            if verbose:
                # Print compact retrieval summary (offline/live gate, articles,
                # passages, parse status) before the final answer line so you can
                # see what the strategy used at a glance during a live session.
                # No-op for strategies that don't emit extras (e.g. baseline).
                if out is not None and out.extras:
                    print_retrieval_summary(out.extras)
                mark = "✓" if outcome.correct else ("·" if outcome.correct is None else "✗")
                print(f"  → {'ABCD'[chosen_idx]}  {mark}  ({elapsed:.2f}s)")

            if outcome.game_over:
                break

        # --- flush learned index to disk (session end) ---
        # Persist any live-search articles that were confirmed correct
        # during this game.  No-op if nothing was learned.
        _grower = getattr(strategy, "index_grower", None)
        if _grower is not None:
            try:
                _grower.flush()
            except Exception as _flush_exc:  # noqa: BLE001
                if verbose:
                    print(f"  ! IndexGrower.flush() failed: {_flush_exc}")

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
