"""GPU-free stand-in for LLM. Tests the orchestration layer, not model accuracy."""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

from .llm import AnswerProbabilities, LLMResponse

_GOLD_RE = re.compile(r"<gold>([A-D])</gold>")


class MockLLM:
    """Reads '<gold>X</gold>' injected into the prompt; returns that letter with high confidence.

    If no gold marker is present, returns 'A' with uniform confidence.
    Set correctness < 1.0 for stochastic mocks (integration tests, ablations).
    """
    def __init__(self, name: str = "mock", correctness: float = 1.0) -> None:
        self._name = name
        self.correctness = correctness
        self.calls: int = 0

    @property
    def name(self) -> str:
        return self._name

    def score_options(
        self,
        messages: Sequence[Dict[str, str]],
        letters: Sequence[str] = ("A", "B", "C", "D"),
    ) -> AnswerProbabilities:
        self.calls += 1
        gold = self._find_gold(messages)
        probs = self._make_probs(gold, list(letters))
        return AnswerProbabilities.from_probs(probs, elapsed=0.001)

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls += 1
        gold = self._find_gold(messages)
        text = f"Answer: {gold}" if gold else "Answer: A"
        return LLMResponse(text=text, elapsed_seconds=0.001, input_tokens=10, output_tokens=4)

    def _find_gold(self, messages: Sequence[Dict[str, str]]) -> Optional[str]:
        for m in messages:
            match = _GOLD_RE.search(m.get("content", ""))
            if match:
                return match.group(1)
        return None

    def _make_probs(self, gold: Optional[str], letters: list[str]) -> Dict[str, float]:
        if gold and gold in letters:
            base = (1.0 - self.correctness) / max(len(letters) - 1, 1)
            return {l: (self.correctness if l == gold else base) for l in letters}
        return {l: 1.0 / len(letters) for l in letters}