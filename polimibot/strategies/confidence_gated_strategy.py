"""Confidence-gated fallback strategy: primary first, fallback only when uncertain.

Motivation
──────────
When a calibrated primary model (e.g. Phi-4 few-shot logit-scored, measured
ECE ≈ 0.04 on the project's History eval) is highly confident, additional
retrieval adds latency without benefit — and often hurts via context-distraction.
Gating fallback on the primary's logit-margin preserves any lift on questions
where retrieval actually matters while dramatically reducing external-API load.

This is the deployment-architecture answer to two empirical findings from the
project's ablation work:

  1. RAG provides no net lift on Phi-4 + History (~92% saturation): high-confidence
     primary answers don't improve, low-confidence ones occasionally do.
  2. Wikipedia rate-limits ~10 unauthenticated requests per burst window (HTTP
     429 with ``Retry-After: 55``, measured directly via the action API).

By routing the high-confidence majority through the bare LLM (zero API calls)
and reserving live retrieval for the ~10-20% low-margin tail, the deployed
system keeps API volume well under the burst threshold without changing the
model's accuracy ceiling.

Routing logic
─────────────
1. Run ``primary.answer(inp)`` — typically a logit-scored BaselineLLMStrategy.
2. Inspect ``primary.extras["margin"]`` — the gap between top1 and top2
   answer-letter probabilities. For a well-calibrated model this is a
   meaningful uncertainty signal.
3. If ``margin >= margin_threshold`` (or if margin is unavailable, e.g.
   under free-generation mode), commit the primary's answer.
4. Otherwise, run ``fallback.answer(inp)`` (typically a RAGStrategy with
   live Wikipedia fallback enabled) and return that. Annotations on
   ``extras`` make the routing decision visible to post-hoc analysis.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..config import Category
from .base import Strategy, StrategyInput, StrategyOutput


# Default margin threshold for the gate. Calibrated empirically for Phi-4 on
# the History gold set (ECE ≈ 0.04 → margin is well-calibrated, 0.20 is the
# point where roughly 80-90% of primary answers commit and the remaining
# ~10-20% escalate). Tune per-model if calibration differs.
DEFAULT_MARGIN_THRESHOLD = 0.20


class ConfidenceGatedStrategy(Strategy):
    """Run ``primary`` first; on low-margin output, fall back to ``fallback``.

    Args:
        primary: cheap, fast strategy that runs first. Must populate
            ``extras["margin"]`` for the gate to engage. The expected
            shape is the value returned by :meth:`LLM.score_options` —
            top probability minus second-top, in [0, 1]. When ``margin``
            is missing (e.g. the primary used free generation rather than
            logit-scoring), the gate degrades to "always commit primary".
        fallback: heavier strategy invoked only when the primary's margin
            is below ``margin_threshold``. Typically a ``RAGStrategy``
            with live fallback enabled. Must implement the same
            :class:`Strategy` interface.
        margin_threshold: minimum top1 - top2 probability gap to commit
            to the primary's answer. Below this, fallback fires.
            Reasonable range ``[0.05, 0.30]``. Default 0.20 is calibrated
            for Phi-4 on History; values closer to 0 commit more often
            (less fallback, more reliance on parametric knowledge), values
            closer to 1 escalate more often (more retrieval, more API
            calls, potentially more context-distraction).
        always_fallback_categories: categories whose questions ALWAYS escalate
            to ``fallback``, bypassing the margin gate entirely. The margin is
            a valid uncertainty signal only when the primary *could* know the
            answer; for a category the model has no parametric knowledge of —
            e.g. NEWS, whose questions reference specific dated articles past
            the model's knowledge cutoff — a confident margin is meaningless,
            so retrieval is mandatory rather than optional lift. Defaults to
            the empty set (pure margin gate, unchanged behaviour).

    Side-effects on ``extras``:
        The returned ``StrategyOutput.extras`` always includes:
          - ``confgated_used_primary``: True iff primary's answer was kept
          - ``confgated_margin``: the observed primary margin (or None)
          - ``confgated_threshold``: the threshold that was applied
        When fallback fires, additionally:
          - ``confgated_primary_choice``: what primary would have picked
          - ``confgated_primary_confidence``: primary's reported confidence
          - ``confgated_disagrees``: True iff fallback flipped the answer
    """

    def __init__(
        self,
        primary: Strategy,
        fallback: Strategy,
        *,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
        always_fallback_categories: Iterable[Category] = frozenset(),
    ) -> None:
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError(
                f"margin_threshold must be in [0.0, 1.0]; got {margin_threshold}"
            )
        self.primary = primary
        self.fallback = fallback
        self.margin_threshold = float(margin_threshold)
        self.always_fallback_categories = frozenset(always_fallback_categories)
        # Short, leaderboard-friendly name. We use only each arm's *type tag*
        # (the prefix before its first ``[``) rather than the arm's full name —
        # the surrounding ``report_id`` already encodes model + prompt style,
        # so repeating them here would be redundant. The optional ``+always:…``
        # suffix surfaces any forced-fallback categories.
        primary_tag  = getattr(primary,  "name", "primary").split("[", 1)[0]
        fallback_tag = getattr(fallback, "name", "fallback").split("[", 1)[0]
        always_tag = ""
        if self.always_fallback_categories:
            cats = ",".join(sorted(c.value for c in self.always_fallback_categories))
            always_tag = f"+always:{cats}"
        self.name = (
            f"gated[{primary_tag}→{fallback_tag}|m≥{self.margin_threshold}{always_tag}]"
        )

    def warm_up(self) -> None:
        """Warm up BOTH arms. The fallback only fires on some questions, but we
        don't want its first invocation mid-game to pay the JIT-compilation
        cost — that latency hit could exceed the per-question budget on a
        question we badly needed the fallback for. Pay it up-front, we do."""
        if hasattr(self.primary, "warm_up"):
            self.primary.warm_up()
        if hasattr(self.fallback, "warm_up"):
            self.fallback.warm_up()

    def shutdown(self) -> None:
        """Cascade shutdown to both arms — symmetric to warm_up."""
        for arm in (self.primary, self.fallback):
            if hasattr(arm, "shutdown"):
                arm.shutdown()

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        # ── Phase 1: cheap primary pass ─────────────────────────────────────
        # Single LLM forward (or whatever the primary does); no retrieval.
        primary_out: StrategyOutput = self.primary.answer(inp)
        margin: Optional[float] = primary_out.extras.get("margin")

        # ── Forced escalation ───────────────────────────────────────────────
        # Some categories (e.g. NEWS) ask about facts the primary cannot know
        # from parametric memory, so its margin is not a trustworthy signal —
        # always escalate to retrieval regardless of how "confident" it looks.
        force_fallback = (
            inp.category is not None
            and inp.category in self.always_fallback_categories
        )

        # ── Gate decision ───────────────────────────────────────────────────
        # Commit primary when NOT force-escalating and either:
        #   (a) margin signal is absent (e.g. free-generation primary) — we
        #       have no uncertainty estimate to act on, so defer to primary; OR
        #   (b) margin is at-or-above threshold — primary is confident enough
        #       that retrieval is unlikely to help and might hurt.
        if not force_fallback and (margin is None or margin >= self.margin_threshold):
            # Annotate routing decision on the existing extras dict. We must
            # not mutate frozen StrategyOutput itself, but extras is a dict
            # (mutable by design of dataclass field default_factory).
            primary_out.extras.update({
                "confgated_used_primary":  True,
                "confgated_margin":         margin,
                "confgated_threshold":      self.margin_threshold,
            })
            return primary_out

        # ── Phase 2: uncertain — escalate to fallback ───────────────────────
        # Annotated with both choices so post-hoc analysis can answer
        # "when fallback fires, does it actually change the answer?"
        fallback_out: StrategyOutput = self.fallback.answer(inp)
        fallback_out.extras.update({
            "confgated_used_primary":      False,
            "confgated_margin":             margin,
            "confgated_threshold":          self.margin_threshold,
            "confgated_forced_category":    force_fallback,
            "confgated_primary_choice":     primary_out.chosen_index,
            "confgated_primary_confidence": primary_out.confidence,
            "confgated_disagrees":          (
                fallback_out.chosen_index != primary_out.chosen_index
            ),
        })
        return fallback_out
