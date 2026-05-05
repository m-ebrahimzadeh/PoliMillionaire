"""Tool unit tests. No GPU, no network, no FAISS required."""
from __future__ import annotations
import pytest
from polimibot.tools.calculator import safe_eval
from polimibot.tools.maths_tool import (
    MathsTool, _extract_expression, _parse_option_value, _match_options,
)
from polimibot.strategies.base import StrategyInput
from polimibot.strategies.tool_strategy import ToolStrategy
from polimibot.config import Category


# ── safe_eval ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr,expected", [
    ("2 + 2",          4),
    ("15 / 100 * 200", 30.0),
    ("2 ** 8",         256),
    ("sqrt(144)",      12.0),
    ("factorial(5)",   120),
    ("gcd(12, 8)",     4),
])
def test_safe_eval_arithmetic(expr, expected):
    assert abs(safe_eval(expr) - expected) < 1e-9


def test_safe_eval_rejects_import():
    with pytest.raises((ValueError, AttributeError)):
        safe_eval("__import__('os')")


def test_safe_eval_rejects_unknown_function():
    with pytest.raises(ValueError, match="not in whitelist"):
        safe_eval("open('secret.txt')")


# ── _extract_expression ───────────────────────────────────────────────────────

@pytest.mark.parametrize("question,contains", [
    ("What is 15% of 200?",         "15/100*200"),
    ("What is 2 to the power of 8?","2**8"),
    ("Calculate 17 * 23",           "17"),  # expression contains 17
    ("What is the square root of 144?", "sqrt(144)"),
    ("What is 50 plus 30?",         "+"),
])
def test_extract_expression_captures_math(question, contains):
    expr = _extract_expression(question)
    assert expr is not None
    assert contains in expr


def test_extract_expression_returns_none_for_factual():
    # Factual question — no compute prefix
    assert _extract_expression("Who invented calculus?") is None
    assert _extract_expression("In what year was Rome founded?") is None


# ── MathsTool ────────────────────────────────────────────────────────────────

def _maths_inp(q: str, opts: tuple[str,...]) -> StrategyInput:
    return StrategyInput(question=q, options=opts, level=3, category=Category.MATHS)


def test_maths_tool_solves_percentage():
    tool = MathsTool()
    inp = _maths_inp("What is 15% of 200?", ("25", "30", "35", "40"))
    assert tool.can_handle(inp)
    out = tool.use(inp)
    assert out is not None
    assert out.chosen_index == 1   # "30"
    assert out.confidence > 0.95


def test_maths_tool_solves_power():
    tool = MathsTool()
    inp = _maths_inp("What is 2 to the power of 8?", ("128", "256", "512", "1024"))
    out = tool.use(inp)
    assert out is not None
    assert out.chosen_index == 1   # "256"


def test_maths_tool_abstains_on_factual():
    tool = MathsTool()
    inp = _maths_inp("Who proved the Pythagorean theorem?",
                     ("Euclid", "Pythagoras", "Archimedes", "Plato"))
    out = tool.use(inp)
    assert out is None   # options aren't numeric → abstain


def test_maths_tool_abstains_on_wrong_category():
    tool = MathsTool()
    inp = StrategyInput(
        question="What is 2 + 2?", options=("3","4","5","6"),
        level=1, category=Category.SCIENCE,  # wrong category
    )
    assert not tool.can_handle(inp)


# ── ToolStrategy ─────────────────────────────────────────────────────────────

from polimibot.models.mock import MockLLM
from polimibot.strategies.llm_baseline import BaselineLLMStrategy


def test_tool_strategy_uses_tool_when_available():
    """Tool answers maths → LLM never called."""
    llm = MockLLM(correctness=1.0)
    strategy = ToolStrategy(tools=[MathsTool()], fallback=BaselineLLMStrategy(llm))
    inp = _maths_inp("What is 15% of 200?", ("25", "30", "35", "40"))
    out = strategy.answer(inp)
    assert out.chosen_index == 1
    assert llm.calls == 0   # tool answered — LLM not called


def test_tool_strategy_falls_back_on_abstain():
    """Tool abstains on factual question → LLM gets called."""
    llm = MockLLM(correctness=1.0)
    strategy = ToolStrategy(tools=[MathsTool()], fallback=BaselineLLMStrategy(llm))
    inp = _maths_inp("Who proved the Pythagorean theorem? <gold>B</gold>",
                     ("Euclid", "Pythagoras", "Archimedes", "Plato"))
    out = strategy.answer(inp)
    assert llm.calls > 0   # fallback was used


def test_tool_strategy_name_reflects_composition():
    llm = MockLLM()
    baseline = BaselineLLMStrategy(llm)
    strategy = ToolStrategy(tools=[MathsTool()], fallback=baseline)
    assert "maths_tool" in strategy.name
    assert "fallback=" in strategy.name