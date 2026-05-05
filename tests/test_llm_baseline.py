import pytest
from polimibot.models.mock import MockLLM
from polimibot.strategies.llm_baseline import (
    BaselineLLMStrategy, _build_messages, _parse_letter
)
from polimibot.strategies.base import StrategyInput


def _inp(gold_letter: str = "B") -> StrategyInput:
    # Inject gold marker so MockLLM knows the correct answer
    return StrategyInput(
        question=f"What is 2+2? <gold>{gold_letter}</gold>",
        options=("3", "4", "5", "6"),
        level=1,
    )


def test_score_options_path_picks_gold():
    llm = MockLLM(correctness=1.0)
    strategy = BaselineLLMStrategy(llm, use_score_options=True)
    out = strategy.answer(_inp("B"))
    assert out.chosen_index == 1          # "B" = index 1
    assert out.confidence > 0.8
    assert "probs" in out.extras


def test_generation_path_picks_gold():
    llm = MockLLM(correctness=1.0)
    strategy = BaselineLLMStrategy(llm, use_score_options=False)
    out = strategy.answer(_inp("C"))
    assert out.chosen_index == 2          # "C" = index 2
    assert out.extras["parse_ok"] is True


def test_parse_failure_falls_back_to_index_0():
    """If the model outputs garbage, we return index 0 with low confidence."""
    idx = _parse_letter("I'm not sure, probably the third option maybe.")
    assert idx is None
    # Now via the full strategy with a mock that returns unparseable text
    # We patch generate to return bad text
    class BadMock(MockLLM):
        def generate(self, messages, **_):
            from polimibot.models.llm import LLMResponse
            return LLMResponse(text="I cannot determine the answer.", elapsed_seconds=0.001)
    strategy = BaselineLLMStrategy(BadMock(), use_score_options=False)
    out = strategy.answer(_inp())
    assert out.chosen_index == 0
    assert out.confidence == pytest.approx(0.25)


def test_mock_call_counter():
    llm = MockLLM()
    strategy = BaselineLLMStrategy(llm)
    strategy.answer(_inp())
    strategy.answer(_inp())
    assert llm.calls == 2


def test_message_format_contains_all_options():
    inp = _inp()
    msgs = _build_messages(inp)
    user_msg = msgs[-1]["content"]
    for letter in "ABCD":
        assert f"{letter})" in user_msg