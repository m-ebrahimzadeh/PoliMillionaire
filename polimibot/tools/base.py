"""Tool interface. Implement and register in ToolStrategy — nothing else to change."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from ..strategies.base import StrategyInput, StrategyOutput


class Tool(ABC):
    """A tool either answers confidently or abstains (returns None).
    
    Abstaining is correct behavior — it lets the next handler in the chain try.
    Never return a guess; only return when confidence is high.
    """
    name: str = "tool"

    @abstractmethod
    def can_handle(self, inp: StrategyInput) -> bool:
        """Fast guard based on metadata (category, level, etc.). No computation here."""

    @abstractmethod
    def use(self, inp: StrategyInput) -> Optional[StrategyOutput]:
        """Attempt to answer. Return None to abstain and pass control downstream."""