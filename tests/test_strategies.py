import pytest

from polimibot import RandomStrategy, Strategy, StrategyInput


def _make_input() -> StrategyInput:
    return StrategyInput(
        question="What is 2+2?",
        options=("3", "4", "5", "6"),
        level=1,
    )


def test_random_strategy_returns_valid_index():
    s = RandomStrategy(seed=0)
    out = s.answer(_make_input())
    assert 0 <= out.chosen_index < 4
    assert out.confidence == pytest.approx(0.25)
    assert out.is_abstain is False


def test_random_strategy_is_reproducible_with_seed():
    a = [RandomStrategy(seed=42).answer(_make_input()).chosen_index for _ in range(5)]
    b = [RandomStrategy(seed=42).answer(_make_input()).chosen_index for _ in range(5)]
    assert a == b


def test_strategy_input_is_immutable():
    import dataclasses
    inp = _make_input()
    with pytest.raises(dataclasses.FrozenInstanceError):
        inp.question = "different"  # type: ignore[misc]


def test_strategy_cannot_be_instantiated_directly():
    # ABC enforcement: forgetting answer() must fail loudly.
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]