"""Zero-shot LLM baseline. The performance floor every later strategy must beat."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from ..models.llm import LLM, AnswerProbabilities
from ..models.mock import MockLLM
from .base import Strategy, StrategyInput, StrategyOutput

from ..prompts.templates import PromptStyle, build_messages, parse_answer

_AnyLLM = LLM | MockLLM

# Styles that require free generation (not single-token logit scoring).
# Despite the name kept for git-blame stability, this set is broader than
# strict CoT — ELIMINATION (per-option scaffolding) also generates and
# uses the same generation budget + boxed early-stop defaults.
_COT_STYLES = {
    PromptStyle.ZERO_SHOT_COT,
    PromptStyle.FEW_SHOT_COT,
    PromptStyle.ELIMINATION,
}

# Generation budgets. Direct (non-CoT) only needs space for "Answer: X" plus
# a few stray tokens; CoT needs room for the reasoning trace AND the answer.
DEFAULT_DIRECT_MAX_NEW_TOKENS = 16
DEFAULT_COT_MAX_NEW_TOKENS    = 256

# Threshold below which a non-CoT generation cap risks truncating before the
# letter is emitted. 16 is fine on Qwen-Instruct; 8 is not. Tuned empirically.
_MIN_DIRECT_CAP = 8

# Default early-stop strings for CoT generation. \boxed{X} is the
# Qwen-Math / DeepSeek-Math native answer format; matching it cuts off
# the "Hope this helps!" chatter that follows the answer otherwise.
DEFAULT_COT_STOP_STRINGS: tuple[str, ...] = (
    r"\boxed{A}", r"\boxed{B}", r"\boxed{C}", r"\boxed{D}",
)


class BaselineLLMStrategy(Strategy):
    """One model, one prompt, one forward pass.

    Args:
        llm: A loaded LLM (or MockLLM for tests).
        style: Which prompt variant to use.
        use_score_options: If True (default), read logits directly — faster
            and more reliable. Must be False for CoT styles.
        direct_max_new_tokens: generation cap for non-CoT styles
            (ZERO_SHOT, FEW_SHOT). Default 16.
        cot_max_new_tokens: generation cap for CoT styles. Default 256.
        stop_strings: passed to ``LLM.generate``; defaults to the
            \\boxed{X} family for CoT styles, ``None`` for non-CoT.
    """

    def __init__(
        self,
        llm: _AnyLLM,
        *,
        style: PromptStyle = PromptStyle.ZERO_SHOT,
        use_score_options: bool = True,
        direct_max_new_tokens: int = DEFAULT_DIRECT_MAX_NEW_TOKENS,
        cot_max_new_tokens:    int = DEFAULT_COT_MAX_NEW_TOKENS,
        stop_strings: Optional[Sequence[str]] = None,
    ) -> None:
        if style in _COT_STYLES and use_score_options:
            raise ValueError(
                f"style={style.value} requires free generation (use_score_options=False). "
                "score_options reads the first predicted token — which is the start of "
                "the reasoning trace, not the answer letter."
            )
        # Non-CoT free generation with a tight cap silently truncates before
        # any letter is emitted, parse_answer returns None, and the strategy
        # abstains on every question — looks like an "always A" run when
        # the runner falls back to fallback_index=0. Catch it loudly.
        if (
            style not in _COT_STYLES
            and not use_score_options
            and direct_max_new_tokens < _MIN_DIRECT_CAP
        ):
            raise ValueError(
                f"direct_max_new_tokens={direct_max_new_tokens} is too small for "
                f"style={style.value} with use_score_options=False. The model needs "
                f"at least ~{_MIN_DIRECT_CAP} tokens to emit 'Answer: X'. Increase "
                "the cap, or set use_score_options=True (logit-scoring needs no "
                "generated tokens)."
            )

        self.llm = llm
        self.style = style
        self.use_score_options = use_score_options
        self.direct_max_new_tokens = direct_max_new_tokens
        self.cot_max_new_tokens    = cot_max_new_tokens
        # Default to boxed-stop-strings for CoT; off for direct (no boxed
        # output expected at 16 tokens).
        if stop_strings is None and style in _COT_STYLES:
            self.stop_strings: Optional[tuple[str, ...]] = DEFAULT_COT_STOP_STRINGS
        else:
            self.stop_strings = tuple(stop_strings) if stop_strings else None
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
        max_tok = (
            self.cot_max_new_tokens if self.style in _COT_STYLES
            else self.direct_max_new_tokens
        )
        response = self.llm.generate(
            messages,
            max_new_tokens=max_tok,
            temperature=0.0,
            stop_strings=self.stop_strings,
        )
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