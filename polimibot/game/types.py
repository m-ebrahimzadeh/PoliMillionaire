"""Game-domain types. The boundary, here begins.

These dataclasses are what the rest of polimibot sees. The adapter in
`adapter.py` is the only place that knows about millionaire_client.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GameQuestion:
    """A question, frozen and minimal. Strategies receive these."""
    text: str
    options: tuple[str, ...]   # always 4; index 0..3 maps to 'A'..'D'
    level: int                 # 1..15 (or whatever max_levels says)


@dataclass(frozen=True)
class AnswerOutcome:
    """Result of submitting an answer."""
    correct: Optional[bool]    # None when the server doesn't reveal it
    timed_out: bool
    game_over: bool
    earned_amount: float
    next_question: Optional[GameQuestion]
    reached_level: Optional[int]


@dataclass(frozen=True)
class SessionRecord:
    """End-of-session snapshot for one game. Logged to JSONL later.

    "Session" follows the millionaire_client terminology: one full play of
    one competition. Renamed from GameSummary to disambiguate from the
    JSONL ``GameSummaryRecord`` row in logging_utils.
    """
    competition_id: int
    competition_name: str
    session_id: int
    final_level: int
    earned_amount: float
    finished_normally: bool    # vs crashed / timed out mid-game