"""Zero-shot LLM baseline. The performance floor every later strategy must beat."""
from __future__ import annotations

from typing import Dict

from ..models.llm import LLM, AnswerProbabilities
from ..models.mock import MockLLM
from .base import Strategy, StrategyInput, StrategyOutput

from ..prompts.templates import PromptStyle, build_messages, parse_answer

_AnyLLM = LLM | MockLLM

_COT_STYLES = {PromptStyle.ZERO_SHOT_COT, PromptStyle.FEW_SHOT_COT}


class BaselineLLMStrategy(Strategy):
    """One model, one prompt, one forward pass.

    Args:
        llm: A loaded LLM (or MockLLM for tests).
        style: Which prompt variant to use.
        use_score_options: If True (default), read logits directly — faster
            and more reliable. Must be False for CoT styles.
    """

    def __init__(
        self,
        llm: _AnyLLM,
        *,
        style: PromptStyle = PromptStyle.ZERO_SHOT,
        use_score_options: bool = True,
    ) -> None:
        if style in _COT_STYLES and use_score_options:
            raise ValueError(
                f"style={style.value} requires free generation (use_score_options=False). "
                "score_options reads the first predicted token — which is the start of "
                "the reasoning trace, not the answer letter."
            )
        self.llm = llm
        self.style = style
        self.use_score_options = use_score_options
        self.name = f"baseline[{llm.name}|{style.value}]"

    def warm_up(self) -> None:
        """Dummy forward pass to absorb CUDA JIT compilation before the first real question."""
        dummy = StrategyInput(
            question="Warm-up",
            options=("A", "B", "C", "D"),
            level=1,
        )
        self.answer(dummy)

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        messages = build_messages(
            inp.question,
            inp.options,
            category=inp.category,
            style=self.style,
        )
        if self.use_score_options:
            return self._answer_via_logits(messages)
        return self._answer_via_generation(messages)

    # ── private ──────────────────────────────────────────────────────────────

    def _answer_via_logits(self, messages: list[Dict]) -> StrategyOutput:
        result: AnswerProbabilities = self.llm.score_options(messages)
        chosen_index = ord(result.top_letter) - ord("A")
        return StrategyOutput(
            chosen_index=chosen_index,
            confidence=result.top_prob,
            rationale=None,
            extras={
                "probs": result.probs,
                "margin": result.margin,
                "elapsed_seconds": result.elapsed_seconds,
            },
        )

    def _answer_via_generation(self, messages: list[Dict]) -> StrategyOutput:
        max_tok = 16 if self.style not in _COT_STYLES else 256
        response = self.llm.generate(messages, max_new_tokens=max_tok, temperature=0.0)
        idx = parse_answer(response.text)
        parse_ok = idx is not None
        # On parse failure, abstain explicitly. The runner will substitute
        # its fallback_index; the ensemble will redistribute weight to the
        # surviving strategies. Defaulting to 0 here would silently bias
        # accuracy toward whichever competition has the most A-correct
        # gold answers.
        return StrategyOutput(
            chosen_index=idx if parse_ok else 0,
            confidence=0.5 if parse_ok else 0.25,
            rationale=response.text,
            is_abstain=not parse_ok,
            extras={"parse_ok": parse_ok},
        )