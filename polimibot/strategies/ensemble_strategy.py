"""EnsembleStrategy — fuse probability distributions from multiple strategies.

Two fusion modes:
  "weighted": sum per-letter probs scaled by strategy weight (preferred).
              Falls back to soft one-hot if a strategy omits probs.
  "vote":     weighted majority vote. Works with any strategy type.

Sequential execution is intentional — strategies share one GPU.
"""
from __future__ import annotations

import time
from typing import Optional

from .base import Strategy, StrategyInput, StrategyOutput

_LETTERS = ("A", "B", "C", "D")


def _extract_probs(out: StrategyOutput) -> dict[str, float]:
    """Get a normalised letter→probability dict from a StrategyOutput.

    Preference order:
      1. extras["probs"] — full distribution from score_options (best)
      2. Soft one-hot    — chosen letter gets out.confidence,
                           remaining letters share (1 - confidence) / 3

    The soft one-hot is the honest thing to do when probs are unavailable:
    it encodes "I picked this letter with this confidence" rather than
    inventing a fabricated distribution.
    """
    raw = out.extras.get("probs") if isinstance(out.extras, dict) else None
    if isinstance(raw, dict) and len(raw) == 4:
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    # Soft one-hot fallback
    n = len(_LETTERS)
    residual = max(0.0, 1.0 - out.confidence) / max(n - 1, 1)
    chosen = _LETTERS[out.chosen_index] if 0 <= out.chosen_index < n else "A"
    return {
        letter: (out.confidence if letter == chosen else residual)
        for letter in _LETTERS
    }


class EnsembleStrategy(Strategy):
    """Run sub-strategies sequentially, combine their verdicts.

    Args:
        strategies: list of Strategy instances. May share an LLM.
        weights:    per-strategy trust weights (default: uniform).
                    Need not sum to 1 — normalised internally.
        mode:       "weighted" (prob fusion) or "vote" (majority vote).
    """

    def __init__(
        self,
        strategies: list[Strategy],
        *,
        weights: Optional[list[float]] = None,
        mode: str = "weighted",
    ) -> None:
        if not strategies:
            raise ValueError("EnsembleStrategy requires at least one sub-strategy.")
        if mode not in ("weighted", "vote"):
            raise ValueError(f"mode must be 'weighted' or 'vote', got {mode!r}")

        n = len(strategies)
        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError(f"weights has {len(weights)} elements; expected {n}")

        self.strategies = list(strategies)
        self.weights = list(weights)
        self.mode = mode
        names = "+".join(s.name for s in self.strategies)
        self.name = f"ensemble[{names}|{mode}]"

    # ── lifecycle ────────────────────────────────────────────────────────────

    def warm_up(self) -> None:
        """Warm up each unique sub-strategy once.

        Deduplication is by object identity — safe even when two strategy
        objects share the same underlying LLM.
        """
        seen: set[int] = set()
        for s in self.strategies:
            if id(s) not in seen:
                seen.add(id(s))
                s.warm_up()

    def shutdown(self) -> None:
        seen: set[int] = set()
        for s in self.strategies:
            if id(s) not in seen:
                seen.add(id(s))
                s.shutdown()

    # ── core logic ───────────────────────────────────────────────────────────

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        t0 = time.monotonic()
        outputs = [s.answer(inp) for s in self.strategies]

        # Drop strategies that explicitly abstained (e.g. MathsTool on a
        # non-computable question). Redistribute their weight automatically
        # via normalisation inside the fusion methods.
        active = [
            (out, w)
            for out, w in zip(outputs, self.weights)
            if not out.is_abstain
        ]

        if not active:
            # Every sub-strategy abstained — nothing better to do.
            return outputs[0]

        elapsed = time.monotonic() - t0
        if self.mode == "weighted":
            return self._weighted_fusion(active, elapsed)
        return self._vote_fusion(active, elapsed)

    # ── fusion modes ─────────────────────────────────────────────────────────

    def _weighted_fusion(
        self,
        active: list[tuple[StrategyOutput, float]],
        elapsed: float,
    ) -> StrategyOutput:
        """Sum per-letter probs, each weighted by strategy trust weight."""
        combined: dict[str, float] = {l: 0.0 for l in _LETTERS}
        total_w = sum(w for _, w in active)

        for out, w in active:
            probs = _extract_probs(out)
            norm_w = w / total_w           # normalise so weights sum to 1
            for letter in _LETTERS:
                combined[letter] += norm_w * probs.get(letter, 0.0)

        best_letter = max(combined, key=lambda l: combined[l])
        best_idx = ord(best_letter) - ord("A")

        sorted_vals = sorted(combined.values(), reverse=True)
        margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 1.0

        return StrategyOutput(
            chosen_index=best_idx,
            confidence=combined[best_letter],
            extras={
                "probs":        {k: round(v, 4) for k, v in combined.items()},
                "margin":       round(margin, 4),
                "n_active":     len(active),
                "elapsed_seconds": round(elapsed, 3),
            },
        )

    def _vote_fusion(
        self,
        active: list[tuple[StrategyOutput, float]],
        elapsed: float,
    ) -> StrategyOutput:
        """Weighted majority vote over chosen_index."""
        votes: dict[int, float] = {i: 0.0 for i in range(4)}
        total_w = sum(w for _, w in active)

        for out, w in active:
            votes[out.chosen_index] += w

        best_idx = max(votes, key=lambda i: votes[i])

        return StrategyOutput(
            chosen_index=best_idx,
            confidence=votes[best_idx] / total_w,
            extras={
                "votes":        {k: round(v, 3) for k, v in votes.items()},
                "n_active":     len(active),
                "elapsed_seconds": round(elapsed, 3),
            },
        )