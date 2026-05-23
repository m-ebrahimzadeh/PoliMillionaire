"""TieredStrategy — dispatch questions by difficulty tier, with escalation.

Routing logic (evaluated in order):
  1. category in category_overrides → that strategy (e.g. Wikidata for ENTERTAINMENT,
     ToolStrategy/AgentStrategy for MATHS). ``maths_override`` is folded into
     this map for back-compat.
  2. level <= easy_max  → easy strategy
  3. level <= medium_max → medium strategy
  4. else               → hard strategy

Confidence-based escalation (optional):
  After running the routed strategy, if extras["margin"] < escalation_threshold,
  re-run the next-tier strategy and return its output instead.
  Disabled when escalation_threshold is None (default).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..config import Category
from .base import Strategy, StrategyInput, StrategyOutput


@dataclass(frozen=True)
class TierBreakpoints:
    """Level boundaries between tiers. Tune these in Stage 9."""
    easy_max_level: int = 5     # levels 1..easy_max → easy strategy
    medium_max_level: int = 10  # levels easy_max+1..medium_max → medium strategy
                                 # levels above medium_max → hard strategy


class TieredStrategy(Strategy):
    """Route questions to the cheapest strategy that can handle them.

    Args:
        easy:   strategy for low-difficulty questions (e.g. BaselineLLMStrategy)
        medium: strategy for mid-difficulty questions  (e.g. RAGStrategy)
        hard:   strategy for high-difficulty questions (e.g. EnsembleStrategy)
        breakpoints: level thresholds separating tiers.
        maths_override: legacy shortcut for ``category_overrides[Category.MATHS]``.
            Kept for back-compat; ``category_overrides`` wins on conflict.
        category_overrides: per-category strategy override map. A category present
            in this map bypasses the level-tier dispatch entirely. Empty by
            default — categories not in the map fall through to easy/medium/hard.
        escalation_threshold: if extras["margin"] < this after the routed answer,
            escalate to the next tier. None disables escalation.
    """

    def __init__(
        self,
        easy: Strategy,
        medium: Strategy,
        hard: Strategy,
        *,
        breakpoints: TierBreakpoints = TierBreakpoints(),
        maths_override: Optional[Strategy] = None,
        category_overrides: Optional[dict[Category, Strategy]] = None,
        escalation_threshold: Optional[float] = None,
    ) -> None:
        self.easy = easy
        self.medium = medium
        self.hard = hard
        self.breakpoints = breakpoints
        self.escalation_threshold = escalation_threshold

        # Merge the legacy maths_override into the new map. The new map
        # wins on conflict so callers migrating to the new API can replace
        # the legacy behaviour without removing the old kwarg from their
        # config.
        overrides: dict[Category, Strategy] = {}
        if maths_override is not None:
            overrides[Category.MATHS] = maths_override
        if category_overrides:
            overrides.update(category_overrides)
        self.category_overrides = overrides
        # Preserve the legacy attribute so external code that still reads
        # ``.maths_override`` (e.g. logging) keeps working.
        self.maths_override = self.category_overrides.get(Category.MATHS)

        override_tag = ""
        if self.category_overrides:
            parts = [f"{cat.value}={s.name}" for cat, s in self.category_overrides.items()]
            override_tag = "|" + ",".join(parts)
        self.name = (
            f"tiered["
            f"easy={easy.name}|"
            f"med={medium.name}|"
            f"hard={hard.name}"
            + override_tag
            + (f"|esc={escalation_threshold}" if escalation_threshold else "")
            + "]"
        )

    def _all_substrategies(self) -> list[Strategy]:
        out: list[Strategy] = [self.easy, self.medium, self.hard]
        out.extend(self.category_overrides.values())
        return out

    def warm_up(self) -> None:
        """Warm up all unique sub-strategies."""
        seen: set[int] = set()
        for s in self._all_substrategies():
            if id(s) not in seen:
                seen.add(id(s))
                s.warm_up()

    def shutdown(self) -> None:
        seen: set[int] = set()
        for s in self._all_substrategies():
            if id(s) not in seen:
                seen.add(id(s))
                s.shutdown()

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        primary, next_tier = self._route(inp)

        # Determine a human-readable tier label for observability.
        tier_label = self._tier_label(primary)

        out = primary.answer(inp)

        # Confidence-based escalation: if the cheap path is uncertain and
        # there's a more expensive tier above it, try that instead.
        escalated_flag = False
        if (
            next_tier is not None
            and self.escalation_threshold is not None
            and not out.is_abstain
        ):
            margin = out.extras.get("margin") if isinstance(out.extras, dict) else None
            if isinstance(margin, float) and margin < self.escalation_threshold:
                escalated_out = next_tier.answer(inp)
                escalated_flag = True
                # Merge tiered routing info into the escalated output's extras.
                escalated_extras = dict(escalated_out.extras)
                escalated_extras["escalated_from"]       = primary.name
                escalated_extras["tier_selected"]        = self._tier_label(next_tier)
                escalated_extras["tier_originally"]      = tier_label
                escalated_extras["escalated"]            = True
                escalated_extras["escalation_threshold"] = self.escalation_threshold
                return StrategyOutput(
                    chosen_index=escalated_out.chosen_index,
                    confidence=escalated_out.confidence,
                    rationale=escalated_out.rationale,
                    is_abstain=escalated_out.is_abstain,
                    extras=escalated_extras,
                )

        # No escalation — annotate the primary output with tier routing info.
        tagged_extras = dict(out.extras) if isinstance(out.extras, dict) else {}
        tagged_extras["tier_selected"]        = tier_label
        tagged_extras["escalated"]            = False
        tagged_extras["escalation_threshold"] = self.escalation_threshold
        return StrategyOutput(
            chosen_index=out.chosen_index,
            confidence=out.confidence,
            rationale=out.rationale,
            is_abstain=out.is_abstain,
            extras=tagged_extras,
        )

    def _tier_label(self, strategy: Strategy) -> str:
        """Return a short label for which tier slot this strategy occupies."""
        for cat, s in self.category_overrides.items():
            if strategy is s:
                return f"override:{cat.value}"
        if strategy is self.easy:
            return "easy"
        if strategy is self.medium:
            return "medium"
        if strategy is self.hard:
            return "hard"
        return strategy.name

    def _route(self, inp: StrategyInput) -> tuple[Strategy, Optional[Strategy]]:
        """Return (primary_strategy, next_tier_or_None)."""
        # Category override wins over the level-tier dispatch. Escalation
        # is intentionally disabled when an override fires — overrides are
        # category-specialist paths whose answers shouldn't be second-
        # guessed by a generic ensemble.
        if inp.category is not None:
            override = self.category_overrides.get(inp.category)
            if override is not None:
                return override, None

        bp = self.breakpoints
        if inp.level <= bp.easy_max_level:
            return self.easy, self.medium
        if inp.level <= bp.medium_max_level:
            return self.medium, self.hard
        return self.hard, None