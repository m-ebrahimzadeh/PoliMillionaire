"""Zero-shot LLM baseline. The performance floor every later strategy must beat."""
from __future__ import annotations

import re
from typing import Dict, Sequence

from ..models.llm import LLM, AnswerProbabilities
from ..models.mock import MockLLM
from .base import Strategy, StrategyInput, StrategyOutput

# Either real or mock — type union avoids coupling to a concrete class
_AnyLLM = LLM | MockLLM

_LETTER_RE = re.compile(r"\b([A-D])\b")


def _build_messages(inp: StrategyInput) -> list[Dict[str, str]]:
    """Format a StrategyInput as chat messages.

    This function is intentionally separate from the class — stage 4
    will replace it with versioned prompt templates without touching the
    strategy itself.
    """
    options_block = "\n".join(
        f"{letter}) {text}"
        for letter, text in zip("ABCD", inp.options)
    )
    user_content = (
        f"{inp.question}\n\n"
        f"{options_block}\n\n"
        "Reply with only the letter of the correct answer (A, B, C, or D)."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable quiz contestant. "
                "Answer multiple-choice questions accurately and concisely."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _parse_letter(text: str) -> int | None:
    """Extract the first A/B/C/D from generated text. Returns 0-based index or None."""
    match = _LETTER_RE.search(text.strip().upper())
    if match:
        return ord(match.group(1)) - ord("A")
    return None


class BaselineLLMStrategy(Strategy):
    """One model, one zero-shot prompt, one forward pass.

    Args:
        llm: A loaded LLM (or MockLLM for tests).
        use_score_options: If True (default), read logits directly — faster
            and more reliable. If False, use free generation + regex parse
            (useful for CoT prompts introduced in stage 4).
    """

    def __init__(self, llm: _AnyLLM, *, use_score_options: bool = True) -> None:
        self.llm = llm
        self.use_score_options = use_score_options
        self.name = f"baseline[{llm.name}|{'score' if use_score_options else 'gen'}]"

    def warm_up(self) -> None:
        """Run a dummy forward pass to JIT-compile CUDA kernels.

        Without this, the first real question pays a ~2s compilation penalty.
        """
        dummy = StrategyInput(
            question="Warm-up",
            options=("A", "B", "C", "D"),
            level=1,
        )
        self.answer(dummy)

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        messages = _build_messages(inp)

        if self.use_score_options:
            return self._answer_via_logits(messages)
        return self._answer_via_generation(messages)

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
        response = self.llm.generate(messages, max_new_tokens=16, temperature=0.0)
        idx = _parse_letter(response.text)
        parse_ok = idx is not None
        return StrategyOutput(
            chosen_index=idx if parse_ok else 0,
            confidence=0.5 if parse_ok else 0.25,  # 0.25 = "I just guessed"
            rationale=response.text,
            extras={"parse_ok": parse_ok},
        )