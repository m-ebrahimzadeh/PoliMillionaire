"""Random baseline. The statistical floor. Beat this, every other strategy must."""
from __future__ import annotations

import random

from .base import Strategy, StrategyInput, StrategyOutput


class RandomStrategy(Strategy):
    """Uniform random over the four options.

    Args:
        seed: when set, makes runs reproducible (useful for tests + ablations).
    """

    def __init__(self, seed: int | None = None) -> None:
        self.name = "random"
        # Local Random instance — does not touch the global random state.
        self._rng = random.Random(seed)

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        n = len(inp.options)
        idx = self._rng.randrange(n)
        return StrategyOutput(
            chosen_index=idx,
            confidence=1.0 / n,           # honest about uncertainty
            rationale=None,
        )