"""Per-phase latency profiler wrapping any Strategy.

Phases timed
------------
tokenise  : text → token ids (CPU only)
forward   : one logit-scoring pass (GPU)
decode    : autoregressive token generation (GPU, 0 if score_options only)
retrieval : wall-clock outside forward+decode (captures RAG / tool I/O)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polimibot.eval.evaluator import EvalReport
    from polimibot.strategies.base import Strategy
    from polimibot.game.types import Question


@dataclass(frozen=True)
class PhaseBreakdown:
    tokenise_ms: float   = 0.0
    forward_ms: float    = 0.0
    decode_ms: float     = 0.0
    retrieval_ms: float  = 0.0   # wall-clock residual

    @property
    def total_ms(self) -> float:
        return self.tokenise_ms + self.forward_ms + self.decode_ms + self.retrieval_ms

    def to_dict(self) -> dict:
        return {
            "tokenise_ms":  round(self.tokenise_ms,  1),
            "forward_ms":   round(self.forward_ms,   1),
            "decode_ms":    round(self.decode_ms,    1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "total_ms":     round(self.total_ms,     1),
        }