"""ToolStrategy: Chain of Responsibility pattern over a list of tools.

Each tool is tried in order. First non-None answer wins.
If all tools abstain, the fallback strategy answers.

This separates two orthogonal concerns:
  - *What* to try (the tool list)
  - *How* to answer if tools don't help (the fallback)
"""
from __future__ import annotations

from ..tools.base import Tool
from .base import Strategy, StrategyInput, StrategyOutput


class ToolStrategy(Strategy):
    """Try registered tools in order; fall back to an LLM strategy.

    Args:
        tools: ordered list of Tool instances. First match wins.
        fallback: strategy to invoke if all tools abstain.
    """

    def __init__(self, tools: list[Tool], fallback: Strategy) -> None:
        self.tools = tools
        self.fallback = fallback
        tool_names = "+".join(t.name for t in tools)
        self.name = f"tool[{tool_names}|fallback={fallback.name}]"

    def warm_up(self) -> None:
        """Warm up the fallback LLM (tools are instant-start)."""
        self.fallback.warm_up()

    def shutdown(self) -> None:
        self.fallback.shutdown()

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        for tool in self.tools:
            if tool.can_handle(inp):
                out = tool.use(inp)
                if out is not None:
                    return out   # tool answered — done, no LLM call
        # All tools abstained (or none could handle this input).
        return self.fallback.answer(inp)