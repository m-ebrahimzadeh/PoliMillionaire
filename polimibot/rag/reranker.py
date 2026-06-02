"""Cross-encoder reranking.

Bi-encoder retrieval (the FAISS dense path) is fast but coarse: it
encodes query and chunk independently and compares their vectors. A
cross-encoder sees the query and chunk JOINTLY and emits a single
relevance score — more accurate, slower. The standard pattern is
"retrieve broadly, rerank precisely":

    1. Dense retrieval returns top-N (N ~ 5×k) cheap candidates.
    2. Cross-encoder scores all N (query, chunk) pairs.
    3. Top-k of N by cross-encoder score is the final result.

This module wraps a sentence-transformers ``CrossEncoder`` model
behind a small interface so that tests can swap in a pure-Python
scoring function (no torch/sentence_transformers required at test time).

Default model: ``BAAI/bge-reranker-base`` — trivia-friendly,
~100 MB on disk, ~30 ms for 20 pairs on a T4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .chunker import Chunk
from .embedder import _resolve_fp16

# A scoring function takes a list of (query, doc) string pairs and
# returns one float per pair (higher = more relevant). The CrossEncoder
# wrapper conforms; tests inject lambdas of the same shape.
ScorePairsFn = Callable[[List[Tuple[str, str]]], List[float]]


@dataclass(frozen=True)
class RerankerSpec:
    """Config for the cross-encoder. Frozen → safe to share."""
    model_name: str = "BAAI/bge-reranker-base"
    batch_size: int = 32
    fp16: Optional[bool] = None  # None → auto (fp16 on CUDA, fp32 on CPU)


class CrossEncoderReranker:
    """Rescore retrieval candidates by joint query-chunk relevance.

    Construction:
        - ``CrossEncoderReranker.load(spec)``: load a sentence-transformers
          model from the HuggingFace hub. Heavy — call once per session.
        - ``CrossEncoderReranker(score_pairs_fn)``: inject a callable for
          testing or for swapping in an LLM-as-judge later.

    Usage:
        candidates = retriever.retrieve(query, k=15)
        top_k = reranker.rerank(query, candidates, top_k=3)
        # → list[(Chunk, cross_encoder_score)], sorted descending.

    Important: the returned score is the CROSS-ENCODER score, not the
    original dense score. Downstream code that thresholds on score
    (e.g. RAGStrategy.min_score) sees the new scale, which is
    model-dependent.
    """

    def __init__(
        self,
        score_pairs: ScorePairsFn,
        *,
        name: str = "reranker",
        batch_size: int = 32,
    ) -> None:
        self._score_pairs = score_pairs
        self.name = name
        self.batch_size = batch_size

    @classmethod
    def load(cls, spec: Optional[RerankerSpec] = None) -> "CrossEncoderReranker":
        """Load a sentence-transformers CrossEncoder. Slow (~5–15 s)."""
        spec = spec or RerankerSpec()
        # Lazy import — keeps the rest of the module importable without
        # sentence-transformers installed (tests use the injected path).
        from sentence_transformers import CrossEncoder
        import torch
        model = CrossEncoder(spec.model_name)
        # Half-precision weights on GPU — ~2x smaller VRAM, negligible quality
        # loss. The HF transformer is exposed as CrossEncoder.model.
        if _resolve_fp16(spec.fp16):
            model.model = model.model.half()

        # Newer sentence-transformers (v3+) passes BatchEncoding with list-backed
        # tensors to the HF model, which crashes transformers' padding check with
        # "list indices must be integers or slices, not tuple". Bypass predict()
        # and tokenize + forward directly so tensors are always proper torch.Tensors.
        hf_model = model.model
        tokenizer = model.tokenizer
        device = next(hf_model.parameters()).device

        def _score(pairs: List[Tuple[str, str]]) -> List[float]:
            all_scores: List[float] = []
            for i in range(0, len(pairs), spec.batch_size):
                batch = pairs[i : i + spec.batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = hf_model(**enc).logits.squeeze(-1)
                all_scores.extend(logits.float().cpu().tolist())
            return all_scores

        return cls(_score, name=spec.model_name, batch_size=spec.batch_size)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[Chunk, float]],
        *,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Rescore ``candidates`` by cross-encoder relevance to ``query``.

        Args:
            query: the retrieval query.
            candidates: ``(Chunk, dense_score)`` pairs from a base
                retriever. The dense score is discarded — only the
                Chunk's text is fed to the cross-encoder.
            top_k: when set, truncate to the top-k reranked items.

        Returns:
            List of ``(Chunk, cross_encoder_score)`` pairs in descending
            score order. Length is ``min(top_k or ∞, len(candidates))``.
            Empty input → empty output.
        """
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self._score_pairs(pairs)
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"Reranker scored {len(scores)} pairs but got "
                f"{len(candidates)} candidates."
            )
        scored: List[Tuple[Chunk, float]] = [
            (chunk, float(s)) for (chunk, _), s in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored if top_k is None else scored[:top_k]
