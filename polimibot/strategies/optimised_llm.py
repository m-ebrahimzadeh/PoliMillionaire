"""BaselineLLMStrategy with a hard max_new_tokens cap.

Key insight: score_options() never generates tokens (reads logits only).
The cap only fires for fallback free-generation paths. We set it tight
because answer letters are always ≤ 3 tokens ("A", "B", "C", "D").
"""
from __future__ import annotations

from dataclasses import dataclass

from polimibot.strategies.llm_baseline import BaselineLLMStrategy
from polimibot.models.llm import LLM


@dataclass(frozen=True)
class OptConfig:
    max_new_tokens: int = 16     # A/B/C/D + punctuation; never needs more
    use_compile: bool   = False  # torch.compile; set True after first warm-up


class OptimisedLLMStrategy(BaselineLLMStrategy):
    """Drop-in replacement — same interface, configurable generation cap."""

    def __init__(self, llm: LLM, cfg: OptConfig | None = None) -> None:
        super().__init__(llm)
        self.cfg = cfg or OptConfig()
        # Patch the LLM's generation kwargs — only for this strategy instance
        self._max_new_tokens = self.cfg.max_new_tokens

    # score_options() is inherited unchanged — no generation happens there.
    # Only _generate() (free-text fallback) is affected by the cap.