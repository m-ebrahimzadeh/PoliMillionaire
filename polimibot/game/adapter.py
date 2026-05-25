"""Adapter, between millionaire_client and the rest of polimibot it sits.

Anywhere else in polimibot importing `millionaire_client` directly is a
code smell. Add a method here instead.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from millionaire_client import MillionaireClient
from millionaire_client.models import Question as ApiQuestion

from .types import AnswerOutcome, GameQuestion, SessionRecord

if TYPE_CHECKING:
    from ..models.speech import SpeechTranscriber


def _to_game_question(q: Optional[ApiQuestion]) -> Optional[GameQuestion]:
    """API question → our frozen DTO (text mode). None passes through (game over)."""
    if q is None:
        return None
    return GameQuestion(
        text=q.text,
        options=tuple(opt.text for opt in q.options),
        level=q.level,
    )


class GameAdapter:
    """Thin wrapper over a MillionaireClient game session.

    One adapter == one game. Construct via GameAdapter(...).

    In speech mode, pass a SpeechTranscriber. The adapter will fetch WAV
    audio for the question and each option, transcribe them, and expose the
    same GameQuestion DTO to the rest of polimibot. The strategy layer sees
    only text regardless of the underlying game mode.
    """

    def __init__(
        self,
        client: MillionaireClient,
        competition_id: int,
        *,
        mode: str = "text",
        transcriber: Optional["SpeechTranscriber"] = None,
    ) -> None:
        if mode == "speech" and transcriber is None:
            raise ValueError(
                "mode='speech' requires a SpeechTranscriber. "
                "Pass transcriber=SpeechTranscriber.load(...)."
            )
        self._competition_id = competition_id
        self._mode = mode
        self._transcriber = transcriber
        self._session = client.game.start(competition_id=competition_id, mode=mode)
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
        # In speech mode we ignore the server's text payload and fetch+transcribe fresh audio for every question.
        if self._mode == "speech":
            return self._fetch_and_transcribe_question()
        
        q = _to_game_question(self._session.current_question)
        if q is None:
            return None
        if q.level == 0:
            # Server may omit level in the per-question payload — Question.from_dict
            # then defaults it to 0, which would silently route every question to
            # the easy tier and log level=0 for everything. Inject the session's
            # current_level (1..15) as the source of truth.
            q = GameQuestion(text=q.text, options=q.options, level=self._session.current_level)
        return q

    # --- the one mutating verb ---
    def submit_answer(self, option_index: int) -> AnswerOutcome:
        """Submit by 0-indexed option. Returns a typed outcome."""
        if self._session.current_question is None:
            raise RuntimeError("No active question to answer.")
        options = self._session.current_question.options
        if not 0 <= option_index < len(options):
            raise ValueError(f"option_index {option_index} out of range 0..{len(options)-1}")

        result = self._session.answer(option_id=options[option_index].id)

        # In speech mode the server returns a Question with text=None
        api_q = None if self._mode == "speech" else result.question
        
        return AnswerOutcome(
            correct=result.correct,
            timed_out=result.timed_out,
            game_over=result.game_over,
            earned_amount=result.earned_amount,
            next_question=_to_game_question(api_q),
            reached_level=result.reached_level,
        )

    # --- speech helpers ---

    def _fetch_and_transcribe_question(self) -> Optional[GameQuestion]:
        """Fetch audio for the current question + all 4 options and transcribe.

        We collect all four options before returning so the caller always 
        receives a complete GameQuestion.
        """
        if self._session.is_game_over:
            return None

        question_wav = self._session.fetch_audio_question()
        question_text = self._transcriber.transcribe(question_wav).text

        option_texts = []
        for _ in range(4):
            option_wav = self._session.fetch_audio_option_next()
            option_texts.append(self._transcriber.transcribe(option_wav).text)

        level = self._session.current_level or 1
        return GameQuestion(
            text=question_text,
            options=tuple(option_texts),
            level=level,
        )

    def summary(self) -> SessionRecord:
        """Snapshot of how the game ended."""
        return SessionRecord(
            competition_id=self._competition_id,
            competition_name=self._competition_name,
            session_id=self.session_id,
            final_level=self._session.current_level,
            earned_amount=self._session.earned_amount,
            finished_normally=self._session.is_game_over,
        )