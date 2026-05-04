"""Adapter, between millionaire_client and the rest of polimibot it sits.

Anywhere else in polimibot importing `millionaire_client` directly is a
code smell. Add a method here instead.
"""
from __future__ import annotations
from typing import Optional

from millionaire_client import MillionaireClient
from millionaire_client.models import Question as ApiQuestion

from .types import AnswerOutcome, GameQuestion, GameSummary


def _to_game_question(q: Optional[ApiQuestion]) -> Optional[GameQuestion]:
    """API question → our frozen DTO. None passes through (game over)."""
    if q is None:
        return None
    return GameQuestion(
        text=q.text,
        options=tuple(opt.text for opt in q.options),
        level=q.level,
    )


class GameAdapter:
    """Thin wrapper over a MillionaireClient game session.

    One adapter == one game. Construct via `GameAdapter.start(...)`.
    """

    def __init__(self, client: MillionaireClient, competition_id: int) -> None:
        self._client = client
        self._competition_id = competition_id
        self._session = client.game.start(competition_id=competition_id)
        self._competition_name = self._session.state.competition.name

    # --- read-only state ---
    @property
    def session_id(self) -> int:
        return self._session.session_id

    @property
    def current_level(self) -> int:
        return self._session.current_level

    @property
    def time_remaining_seconds(self) -> Optional[float]:
        return self._session.time_remaining

    @property
    def is_over(self) -> bool:
        return self._session.is_game_over

    @property
    def current_question(self) -> Optional[GameQuestion]:
        return _to_game_question(self._session.current_question)

    # --- the one mutating verb ---
    def submit_answer(self, option_index: int) -> AnswerOutcome:
        """Submit by 0-indexed option. Returns a typed outcome."""
        if self._session.current_question is None:
            raise RuntimeError("No active question to answer.")
        options = self._session.current_question.options
        if not 0 <= option_index < len(options):
            raise ValueError(f"option_index {option_index} out of range 0..{len(options)-1}")

        result = self._session.answer(option_id=options[option_index].id)
        return AnswerOutcome(
            correct=result.correct,
            timed_out=result.timed_out,
            game_over=result.game_over,
            earned_amount=result.earned_amount,
            next_question=_to_game_question(result.question),
            reached_level=result.reached_level,
        )

    def summary(self) -> GameSummary:
        """Snapshot of how the game ended."""
        return GameSummary(
            competition_id=self._competition_id,
            competition_name=self._competition_name,
            session_id=self.session_id,
            final_level=self._session.current_level,
            earned_amount=self._session.earned_amount,
            finished_normally=self._session.is_game_over,
        )