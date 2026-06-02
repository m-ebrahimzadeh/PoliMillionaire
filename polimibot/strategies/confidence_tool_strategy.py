"""ConfidenceToolStrategy — pin-then-improve with deterministic tools.

Answers the question in two stages:

  Stage 1 (always): fast baseline LLM call (logit-scored, ~0.8s).
                    Result is kept as the "pinned" fallback answer.

  Stage 2 (conditional): if the LLM's margin is below `confidence_threshold`,
                         run the tool chain against the same question.
                         If a tool fires within the remaining time budget,
                         its answer replaces the pinned one.
                         If the tools abstain or time runs out, the pinned
                         answer is submitted as-is.

Rationale:
  - The LLM is always fast and always produces an answer — zero timeout risk.
  - The tool chain is deterministic and near-instant when it fires.
  - On questions where the LLM is uncertain (low margin), tools can often
    provide a confident, correct answer (e.g. algebra, geometry).
  - On questions where the LLM is already confident, the tool overhead is
    skipped entirely — no wasted time.

Time budget:
  Stage 1 uses ~0.8s. Stage 2 (tools) uses <1ms when warm.
  Total budget consumption is well within the 25s hard cutoff.

Confidence threshold:
  `confidence_threshold=0.70` means: run tools whenever the LLM's top-option
  probability is below 70%.  At 70% the LLM is uncertain; at 100% it is sure.
  Tune this knob: lower → more tool attempts; higher → fewer.
"""
from __future__ import annotations

import time
from typing import Optional

from .base import Strategy, StrategyInput, StrategyOutput
from ..tools.base import Tool


class ConfidenceToolStrategy(Strategy):
    """Pin-then-improve: LLM first, tools only when LLM is uncertain.

    Args:
        primary:              Fast baseline strategy (e.g. BaselineLLMStrategy).
        tools:                Ordered list of Tool instances to attempt when
                              primary confidence is low.
        confidence_threshold: Attempt tools when primary confidence < this.
                              Default 0.70 — covers the ~50-75% confidence range
                              seen on difficult questions in live games.
        tool_time_budget:     Max seconds to spend on tools. Defaults to 5s —
                              enough for SymPy warm calls, well within 25s cutoff.
    """

    def __init__(
        self,
        primary: Strategy,
        tools: list[Tool],
        *,
        confidence_threshold: float = 0.70,
        tool_time_budget: float = 5.0,
    ) -> None:
        self.primary              = primary
        self.tools                = tools
        self.confidence_threshold = confidence_threshold
        self.tool_time_budget     = tool_time_budget
        tool_names = "+".join(t.name for t in tools)
        self.name = (
            f"conf_tool[{primary.name}"
            f"|tools={tool_names}"
            f"|thresh={confidence_threshold:.0%}]"
        )

    def warm_up(self) -> None:
        self.primary.warm_up()

    def shutdown(self) -> None:
        self.primary.shutdown()

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        # ── Stage 1: LLM baseline (always) ──────────────────────────────
        pinned: StrategyOutput = self.primary.answer(inp)

        # ── Stage 2: tools only when LLM is uncertain ───────────────────
        if pinned.confidence >= self.confidence_threshold:
            # LLM is confident — trust it, skip tools
            return pinned

        t_tool_start = time.monotonic()
        deadline     = t_tool_start + self.tool_time_budget

        for tool in self.tools:
            if time.monotonic() >= deadline:
                break
            if not tool.can_handle(inp):
                continue
            tool_out: Optional[StrategyOutput] = tool.use(inp)
            if tool_out is not None:
                # Tool fired — return its answer, annotating that we upgraded
                return StrategyOutput(
                    chosen_index=tool_out.chosen_index,
                    confidence=tool_out.confidence,
                    rationale=(
                        f"[upgraded from LLM conf={pinned.confidence:.0%}] "
                        + (tool_out.rationale or "")
                    ),
                    extras={
                        **tool_out.extras,
                        "pinned_index":      pinned.chosen_index,
                        "pinned_confidence": pinned.confidence,
                        "tool_upgraded":     True,
                    },
                )

        # Tools abstained or time ran out — return the pinned LLM answer
        return StrategyOutput(
            chosen_index=pinned.chosen_index,
            confidence=pinned.confidence,
            rationale=pinned.rationale,
            extras={
                **(pinned.extras or {}),
                "tool_upgraded": False,
                "tools_attempted": True,
            },
        )
