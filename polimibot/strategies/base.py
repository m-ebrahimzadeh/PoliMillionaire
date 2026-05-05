"""Strategy interface, the contract every answerer signs.

Every approach — baseline LLM, RAG, tool-using agent, ensemble — a
:class:`Strategy` it must be. The runner one verb only invokes:
``strategy.answer(StrategyInput) -> StrategyOutput``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Category


@dataclass(frozen=True)
class StrategyInput:
    """Everything a strategy is allowed to see when answering one question.

    Frozen by design — strategies must not mutate their input.
    """
    question: str
    options: tuple[str, ...]                 # always 4; index 0..3 = letters A..D
    level: int                               # 1..max_level (typically 15)
    max_level: int = 15
    category: Optional[Category] = None      # routing hint; None when unknown
    competition_id: Optional[int] = None
    time_budget_seconds: float = 25.0        # soft hint; runner enforces hard cutoff


@dataclass(frozen=True)
class StrategyOutput:
    """What the strategy returns. ``elapsed_seconds`` is filled by the runner."""
    chosen_index: int                        # 0..len(options)-1
    confidence: float = 0.25                 # in [0, 1]; default = uniform random
    rationale: Optional[str] = None          # free-text, for logging/debugging only
    is_abstain: bool = False                 # True if strategy refused; runner falls back
    extras: dict[str, Any] = field(default_factory=dict)  # probs, token counts, etc.

    @property
    def chosen_letter(self) -> str:
        """'A', 'B', 'C', or 'D' — derived from chosen_index."""
        return "ABCD"[self.chosen_index]

class Strategy(ABC):
    """Abstract base. Subclass and implement :meth:`answer`."""

    #: Human-readable name; used in logs and reports. Subclasses should set.
    name: str = "strategy"

    def warm_up(self) -> None:
        """Optional: load models, build indices. Called once before any answer()."""
        return None

    def shutdown(self) -> None:
        """Optional: release resources. Called once when the runner is done."""
        return None

    @abstractmethod
    def answer(self, inp: StrategyInput) -> StrategyOutput:
        """Pick one option. Must return within ``inp.time_budget_seconds``."""