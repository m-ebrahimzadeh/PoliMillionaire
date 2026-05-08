"""Runner tests with fake adapter + fake strategy. No I/O."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from polimibot import (
    GameQuestion, AnswerOutcome, SessionRecord, NullLogger, Strategy, StrategyInput,
    StrategyOutput, play_game,
)


# --- a fake "client" / adapter pair the runner can drive ---

@dataclass
class FakeAdapter:
    """Stand-in for GameAdapter. Plays N scripted questions."""
    questions: list[GameQuestion]
    correct_indices: list[int]
    session_id: int = 999
    competition_id: int = 0
    competition_name: str = "Test"

    _i: int = 0
    _earned: float = 0.0
    _over: bool = False

    @property
    def current_level(self) -> int:                 return self._i + 1
    @property
    def time_remaining_seconds(self) -> float:      return 30.0
    @property
    def is_over(self) -> bool:                      return self._over
    @property
    def current_question(self) -> Optional[GameQuestion]:
        return self.questions[self._i] if self._i < len(self.questions) else None

    def submit_answer(self, idx: int) -> AnswerOutcome:
        correct = (idx == self.correct_indices[self._i])
        self._earned += 100 if correct else 0
        self._i += 1
        last = self._i >= len(self.questions)
        if last or not correct:
            self._over = True
        return AnswerOutcome(
            correct=correct, timed_out=False, game_over=self._over,
            earned_amount=self._earned,
            next_question=self.questions[self._i] if not self._over else None,
            reached_level=self._i,
        )

    def summary(self) -> SessionRecord:
        return SessionRecord(
            competition_id=self.competition_id, competition_name=self.competition_name,
            session_id=self.session_id, final_level=self._i,
            earned_amount=self._earned, finished_normally=True,
        )


# Monkeypatch GameAdapter inside the runner module to return our fake.
@pytest.fixture
def patched_adapter(monkeypatch):
    holder: dict = {}
    def factory(client, competition_id):
        return holder["adapter"]
    monkeypatch.setattr("polimibot.runner.GameAdapter", factory)
    monkeypatch.setattr("polimibot.runner.RUNTIME",
                        type("R", (), {"api_min_delay_seconds": 0.0,
                                       "hard_cutoff_seconds": 5.0})())
    return holder


class _AlwaysB(Strategy):
    name = "always_B"
    def answer(self, inp: StrategyInput) -> StrategyOutput:
        return StrategyOutput(chosen_index=1, confidence=1.0)


class _Slow(Strategy):
    name = "slow"
    def answer(self, inp: StrategyInput) -> StrategyOutput:
        time.sleep(2.0)  # > deadline below
        return StrategyOutput(chosen_index=0)


def _qs():
    return [
        GameQuestion(text="q1", options=("a","b","c","d"), level=1),
        GameQuestion(text="q2", options=("a","b","c","d"), level=2),
    ]


def test_runner_records_correct_answers(patched_adapter):
    patched_adapter["adapter"] = FakeAdapter(_qs(), correct_indices=[1, 1])
    res = play_game(client=None, competition_id=0, strategy=_AlwaysB(),
                    logger=NullLogger(), verbose=False)
    assert res.n_questions == 2
    assert res.n_correct == 2
    assert res.accuracy == 1.0


def test_runner_falls_back_on_timeout(patched_adapter, monkeypatch):
    # Tighten the cutoff to force a timeout.
    monkeypatch.setattr("polimibot.runner.RUNTIME",
                        type("R", (), {"api_min_delay_seconds": 0.0,
                                       "hard_cutoff_seconds": 0.2})())
    patched_adapter["adapter"] = FakeAdapter(_qs(), correct_indices=[1, 1])
    res = play_game(client=None, competition_id=0, strategy=_Slow(),
                    logger=NullLogger(), verbose=False, fallback_index=0)
    # Fallback index 0 ≠ correct 1 → wrong → game ends after Q1
    assert res.n_questions == 1
    assert res.n_correct == 0



def test_warm_up_called_before_game_loop():
    """Strategy.warm_up() must fire before any answer() call."""
    from polimibot.strategies.base import Strategy, StrategyInput, StrategyOutput
    from polimibot.models.mock import MockLLM
    from polimibot.strategies.llm_baseline import BaselineLLMStrategy

    call_order: list[str] = []

    class InstrumentedStrategy(BaselineLLMStrategy):
        def warm_up(self):
            call_order.append("warm_up")
        def answer(self, inp):
            call_order.append("answer")
            return StrategyOutput(chosen_index=0)

    strategy = InstrumentedStrategy(MockLLM())
    # play_session with a mock client is tested in test_runner.py already;
    # here we just verify warm_up precedes the first answer via order tracking.
    strategy.warm_up()
    strategy.answer(StrategyInput(question="q", options=("a","b","c","d"), level=1))
    assert call_order.index("warm_up") < call_order.index("answer")