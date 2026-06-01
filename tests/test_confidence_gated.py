"""Tests for ConfidenceGatedStrategy — routing on the primary's logit margin.

Uses tiny in-process test-doubles for the primary/fallback arms; no MockLLM
or torch stubbing needed since the strategy logic only inspects extras and
the chosen index — pure orchestration testing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from polimibot.config import Category
from polimibot.strategies.base import Strategy, StrategyInput, StrategyOutput
from polimibot.strategies.confidence_gated_strategy import (
    ConfidenceGatedStrategy,
    DEFAULT_MARGIN_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────
# Test doubles — minimal Strategy implementations that emit a configurable
# StrategyOutput. Avoids dragging MockLLM + chat-template machinery into a
# pure routing test.
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class _FakeStrategy(Strategy):
    """A Strategy whose answer() returns a pre-configured StrategyOutput."""
    chosen_index: int = 0
    confidence: float = 0.5
    margin: Optional[float] = 0.5
    name_str: str = "fake"
    answer_calls: int = field(default=0, init=False)
    warm_up_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.name = self.name_str

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        self.answer_calls += 1
        extras: dict = {}
        if self.margin is not None:
            extras["margin"] = self.margin
        return StrategyOutput(
            chosen_index=self.chosen_index,
            confidence=self.confidence,
            extras=extras,
        )


def _input(category: Optional[Category] = None) -> StrategyInput:
    return StrategyInput(
        question="What is 2+2?",
        options=("3", "4", "5", "6"),
        level=1,
        category=category,
    )


# ──────────────────────────────────────────────────────────────────────────
# Routing tests — the core contract.
# ──────────────────────────────────────────────────────────────────────────
def test_high_margin_commits_primary_no_fallback_call():
    """When primary's margin is at-or-above threshold, fallback must NOT fire."""
    primary  = _FakeStrategy(chosen_index=1, margin=0.40, name_str="prim")
    fallback = _FakeStrategy(chosen_index=2, margin=0.99, name_str="back")
    s = ConfidenceGatedStrategy(primary, fallback, margin_threshold=0.20)

    out = s.answer(_input())

    assert out.chosen_index == 1, "Primary's answer should win on high margin"
    assert primary.answer_calls == 1
    assert fallback.answer_calls == 0, "Fallback must not be invoked"
    assert out.extras["confgated_used_primary"] is True
    assert out.extras["confgated_margin"] == 0.40
    assert out.extras["confgated_threshold"] == 0.20
    # Fallback-only fields must NOT be set when fallback didn't fire.
    assert "confgated_primary_choice" not in out.extras
    assert "confgated_disagrees" not in out.extras


def test_low_margin_escalates_to_fallback():
    """When primary's margin is below threshold, fallback fires and its
    answer is returned with full routing annotations."""
    primary  = _FakeStrategy(chosen_index=1, confidence=0.35, margin=0.05, name_str="prim")
    fallback = _FakeStrategy(chosen_index=3, confidence=0.80, margin=0.60, name_str="back")
    s = ConfidenceGatedStrategy(primary, fallback, margin_threshold=0.20)

    out = s.answer(_input())

    assert out.chosen_index == 3, "Fallback's answer should win on low margin"
    assert primary.answer_calls == 1
    assert fallback.answer_calls == 1
    assert out.extras["confgated_used_primary"] is False
    assert out.extras["confgated_margin"] == 0.05
    assert out.extras["confgated_primary_choice"] == 1
    assert out.extras["confgated_primary_confidence"] == 0.35
    assert out.extras["confgated_disagrees"] is True


def test_low_margin_fallback_agrees_with_primary():
    """When fallback agrees with primary, confgated_disagrees must be False."""
    primary  = _FakeStrategy(chosen_index=2, margin=0.05)
    fallback = _FakeStrategy(chosen_index=2, margin=0.70)
    s = ConfidenceGatedStrategy(primary, fallback, margin_threshold=0.20)

    out = s.answer(_input())

    assert out.chosen_index == 2
    assert out.extras["confgated_used_primary"] is False
    assert out.extras["confgated_disagrees"] is False


def test_missing_margin_commits_primary():
    """If primary doesn't emit a margin (e.g. free-generation mode), the gate
    has no uncertainty signal to act on and must commit primary's answer."""
    primary  = _FakeStrategy(chosen_index=1, margin=None)
    fallback = _FakeStrategy(chosen_index=3, margin=0.99)
    s = ConfidenceGatedStrategy(primary, fallback, margin_threshold=0.20)

    out = s.answer(_input())

    assert out.chosen_index == 1
    assert fallback.answer_calls == 0, "Fallback must NOT fire without margin signal"
    assert out.extras["confgated_used_primary"] is True
    assert out.extras["confgated_margin"] is None


def test_margin_exactly_at_threshold_commits_primary():
    """Boundary: margin == threshold means primary is *just* confident enough.
    Spec says "below threshold fires fallback" — so equality goes to primary."""
    primary  = _FakeStrategy(chosen_index=1, margin=0.20)
    fallback = _FakeStrategy(chosen_index=3, margin=0.99)
    s = ConfidenceGatedStrategy(primary, fallback, margin_threshold=0.20)

    out = s.answer(_input())

    assert out.chosen_index == 1
    assert fallback.answer_calls == 0


# ──────────────────────────────────────────────────────────────────────────
# Forced-category escalation — the margin gate is bypassed for configured
# categories (e.g. NEWS) the primary cannot answer from parametric memory.
# ──────────────────────────────────────────────────────────────────────────
def test_forced_category_escalates_despite_high_margin():
    """A category in always_fallback_categories escalates even at high margin."""
    primary  = _FakeStrategy(chosen_index=1, margin=0.99, name_str="prim")
    fallback = _FakeStrategy(chosen_index=2, margin=0.10, name_str="back")
    s = ConfidenceGatedStrategy(
        primary, fallback, margin_threshold=0.20,
        always_fallback_categories=frozenset({Category.NEWS}),
    )

    out = s.answer(_input(category=Category.NEWS))

    assert out.chosen_index == 2, "Fallback must fire despite a confident primary"
    assert fallback.answer_calls == 1
    assert out.extras["confgated_used_primary"] is False
    assert out.extras["confgated_forced_category"] is True
    # Primary still ran once so the disagreement analysis is preserved.
    assert primary.answer_calls == 1
    assert out.extras["confgated_primary_choice"] == 1


def test_non_forced_category_still_uses_margin_gate():
    """A category NOT in the forced set keeps the normal margin behaviour."""
    primary  = _FakeStrategy(chosen_index=1, margin=0.99, name_str="prim")
    fallback = _FakeStrategy(chosen_index=2, margin=0.10, name_str="back")
    s = ConfidenceGatedStrategy(
        primary, fallback, margin_threshold=0.20,
        always_fallback_categories=frozenset({Category.NEWS}),
    )

    out = s.answer(_input(category=Category.HISTORY))

    assert out.chosen_index == 1, "High margin should commit primary off-category"
    assert fallback.answer_calls == 0


def test_name_includes_always_fallback_tag():
    s = ConfidenceGatedStrategy(
        _FakeStrategy(name_str="baseline[phi-4|few_shot]"),
        _FakeStrategy(name_str="rag[phi-4|k=3|hybrid]"),
        margin_threshold=0.25,
        always_fallback_categories=frozenset({Category.NEWS}),
    )
    assert s.name == "gated[baseline→rag|m≥0.25+always:news]", (
        f"Unexpected name: {s.name!r}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Lifecycle tests — warm_up and shutdown cascade to both arms.
# ──────────────────────────────────────────────────────────────────────────
def test_warm_up_cascades_to_both_arms():
    primary  = _FakeStrategy()
    fallback = _FakeStrategy()
    s = ConfidenceGatedStrategy(primary, fallback)

    s.warm_up()

    assert primary.warm_up_calls == 1
    assert fallback.warm_up_calls == 1


# ──────────────────────────────────────────────────────────────────────────
# Construction validation.
# ──────────────────────────────────────────────────────────────────────────
def test_invalid_threshold_below_zero_rejected():
    with pytest.raises(ValueError):
        ConfidenceGatedStrategy(_FakeStrategy(), _FakeStrategy(), margin_threshold=-0.1)


def test_invalid_threshold_above_one_rejected():
    with pytest.raises(ValueError):
        ConfidenceGatedStrategy(_FakeStrategy(), _FakeStrategy(), margin_threshold=1.5)


def test_default_threshold_is_documented_constant():
    """The exported DEFAULT_MARGIN_THRESHOLD must match the constructor default."""
    s = ConfidenceGatedStrategy(_FakeStrategy(), _FakeStrategy())
    assert s.margin_threshold == DEFAULT_MARGIN_THRESHOLD


def test_strategy_name_uses_short_arm_tags_and_threshold():
    """The strategy name surfaces the arm *type tags* (baseline / rag / …)
    and the threshold, but NOT the arm's full configuration — that detail
    is already captured in the surrounding ``report_id`` (model + prompt
    style) and would be redundant duplication if repeated here."""
    primary  = _FakeStrategy(name_str="baseline[phi-4|few_shot]")
    fallback = _FakeStrategy(name_str="rag[phi-4|k=3|hybrid|mq|rerank|live]")
    s = ConfidenceGatedStrategy(primary, fallback, margin_threshold=0.25)

    # Short, leaderboard-friendly format
    assert s.name == "gated[baseline→rag|m≥0.25]", (
        f"Unexpected name: {s.name!r}"
    )
    # And it must NOT include the verbose inner configurations.
    assert "phi-4" not in s.name
    assert "hybrid" not in s.name
