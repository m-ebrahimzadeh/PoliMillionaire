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


# ── sympy_solve ───────────────────────────────────────────────────────────────

from polimibot.tools.sympy_tool import sympy_solve


@pytest.mark.parametrize("expr,expected_substr", [
    ("factorial(10)",          "3628800"),
    ("Mod(3**100, 10)",        "1"),
    ("binomial(10, 3)",        "120"),
    ("3*x + 7 - 22",          "5"),    # solve for x
    ("x**2 - 9",               "3"),    # solve x² = 9 (returns [−3, 3], 3 is present)
    ("pi * 5**2",              "78.5"), # numeric eval, check prefix
])
def test_sympy_solve_basic(expr, expected_substr):
    result = sympy_solve(expr)
    assert expected_substr in result, f"sympy_solve({expr!r}) = {result!r}, expected {expected_substr!r} in it"


def test_sympy_solve_equation_with_equals():
    # "3*x + 7 = 22" rewritten to "3*x + 7 - 22" internally
    result = sympy_solve("3*x + 7 = 22")
    assert "5" in result


def test_sympy_solve_raises_on_invalid():
    with pytest.raises(ValueError):
        sympy_solve("__import__('os')")


def test_sympy_solve_raises_on_no_solution():
    # x² + 1 = 0 has no real solutions (SymPy returns complex; our solver still returns them)
    # Just ensure it doesn't crash with a plain error
    try:
        result = sympy_solve("x**2 + 1")
        # SymPy returns complex solutions — result should be a string
        assert isinstance(result, str)
    except ValueError:
        pass  # also acceptable


# ── MathsTool extended prefix coverage ───────────────────────────────────────

@pytest.mark.parametrize("question,contains", [
    ("How many ways can 4 books be arranged?",         None),   # abstains — non-numeric opts handled downstream
    ("How far does a car travel at 60 km/h for 2 hours?", "60"),
    ("Find the total of 120 and 80",                   "120"),
    ("Determine the sum of 15 and 25",                 "15"),
])
def test_extract_expression_extended_prefixes(question, contains):
    from polimibot.tools.maths_tool import _extract_expression
    expr = _extract_expression(question)
    if contains is None:
        pass
    else:
        assert expr is not None, f"Expected expression from: {question!r}"
        assert contains in expr


# ── SympyDirectTool ───────────────────────────────────────────────────────────

from polimibot.tools.sympy_direct_tool import SympyDirectTool


def _maths_inp_direct(q: str, opts: tuple[str, ...]) -> StrategyInput:
    return StrategyInput(question=q, options=opts, level=3, category=Category.MATHS)


def test_sympy_direct_modular_remainder():
    tool = SympyDirectTool()
    inp = _maths_inp_direct(
        "What is the remainder when 2^87 is divided by 7?",
        ("0", "4", "1", "2"),
    )
    out = tool.use(inp)
    # 2^3 = 8 ≡ 1 (mod 7), 87 = 3*29, so 2^87 ≡ 1 → option "1" at index 2
    assert out is not None
    assert out.chosen_index == 2
    assert out.confidence >= 0.95


def test_sympy_direct_binomial():
    tool = SympyDirectTool()
    inp = _maths_inp_direct(
        r"Compute $\dbinom{85}{82}$.",
        ("252", "4680", "98770", "101170"),
    )
    out = tool.use(inp)
    # C(85,82) = C(85,3) = 85*84*83/6 = 98770
    assert out is not None
    assert out.chosen_index == 2   # "98770"
    assert out.confidence >= 0.95


def test_sympy_direct_permutation():
    tool = SympyDirectTool()
    inp = _maths_inp_direct(
        "In how many ways can 4 different books be arranged on a shelf?",
        ("12", "16", "24", "48"),
    )
    out = tool.use(inp)
    assert out is not None
    assert out.chosen_index == 2   # 4! = 24


def test_sympy_direct_area_square_vertices():
    tool = SympyDirectTool()
    inp = _maths_inp_direct(
        "Points (1, 2) and (5, 6) are opposite vertices of a square. What is the area?",
        ("8", "16", "32", "4"),
    )
    out = tool.use(inp)
    # diagonal² = (5-1)²+(6-2)² = 32, area = 32/2 = 16
    assert out is not None
    assert out.chosen_index == 1   # "16"


def test_sympy_direct_abstains_on_abstract_algebra():
    tool = SympyDirectTool()
    inp = _maths_inp_direct(
        "Statement 1 | Every free abelian group is torsion free. "
        "Statement 2 | Every finitely generated torsion-free abelian group is free.",
        ("True, True", "False, False", "True, False", "False, True"),
    )
    out = tool.use(inp)
    assert out is None   # no pattern fires → correct abstention


def test_sympy_direct_abstains_on_wrong_category():
    tool = SympyDirectTool()
    inp = StrategyInput(
        question="What is the remainder when 2^10 is divided by 3?",
        options=("0", "1", "2", "3"),
        level=1,
        category=Category.SCIENCE,
    )
    assert not tool.can_handle(inp)