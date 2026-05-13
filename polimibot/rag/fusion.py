"""Reciprocal Rank Fusion — combine ranked lists from different retrievers
or different queries into a single ranking, robust to score-scale drift.

The problem with score-based fusion: dense cosine scores live in [-1, 1],
BM25 scores live in [0, 30+], cross-encoder scores live wherever the
model was trained. Summing them weights whichever has the biggest
magnitude. Z-score normalisation helps but the distributions differ
per query, so it's brittle.

RRF (Cormack, Clarke, Buettcher 2009) ignores scores entirely and uses
ranks:

    RRF(d) = Σ_{list_i} 1 / (k + rank_i(d))

where ``rank_i(d)`` is d's 1-indexed position in list i (or ∞ if not
present). The constant ``k=60`` is the original paper's value and works
empirically across IR benchmarks — small enough that rank-1 dominates,
large enough that ranks differ smoothly through the top-100.

Usage:

    dense_hits = retriever_dense.search(...)     # list[(Chunk, score)]
    bm25_hits  = bm25_index.search(...)           # list[(Chunk, score)]
    fused = reciprocal_rank_fusion([dense_hits, bm25_hits], k=10)
    # → list[(Chunk, rrf_score)] — top-10 by combined rank.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

from .chunker import Chunk


# Standard RRF damping constant. From Cormack et al. 2009. Don't change
# without a measurement — empirically robust at 60 across IR benchmarks.
RRF_K = 60


def _chunk_key(chunk: Chunk) -> Tuple[str, int]:
    """Identity key for fusion deduplication.

    A chunk is uniquely identified by (source article, chunk_id) — two
    Chunk objects with the same source + chunk_id are the same
    underlying passage even if they came from different retrievers.
    Using ``id()`` would treat them as distinct (different Python
    instances loaded by different indices).
    """
    return (chunk.source, chunk.chunk_id)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Tuple[Chunk, float]]],
    *,
    k: int = 10,
    rrf_k: int = RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[Chunk, float]]:
    """Fuse multiple ranked lists into one by Reciprocal Rank Fusion.

    Args:
        ranked_lists: each inner sequence is ``(Chunk, score)`` pairs in
            descending source-relevance order. The ``score`` values are
            IGNORED — RRF only uses positions.
        k: top-k to return after fusion.
        rrf_k: the damping constant. 60 is the published default.
        weights: optional per-list multiplier on the RRF contribution.
            When None (default) every list weighs 1.0 — the classic
            symmetric fusion. Pass e.g. ``[1.0, 1.5]`` to weight BM25
            more than dense in a hybrid setting where entity-style
            queries dominate. Length must equal ``len(ranked_lists)``.

    Returns:
        ``list[(Chunk, rrf_score)]`` sorted by RRF score descending,
        truncated to ``k``. The Chunk instance returned for any given
        (source, chunk_id) is the first occurrence across the input lists
        (so its ``.text`` reflects whichever index emitted it first).

    Edge cases:
        - empty input list → empty output
        - empty inner lists are tolerated (contribute nothing)
        - ``k`` larger than the fused pool → all results returned
    """
    if not ranked_lists:
        return []
    if weights is None:
        ws: Sequence[float] = [1.0] * len(ranked_lists)
    else:
        if len(weights) != len(ranked_lists):
            raise ValueError(
                f"weights has {len(weights)} entries but {len(ranked_lists)} "
                f"ranked lists were given — they must match 1:1."
            )
        ws = weights

    rrf_scores: dict[Tuple[str, int], float] = defaultdict(float)
    # First-seen Chunk per identity key — used for stable output Chunks.
    first_seen: dict[Tuple[str, int], Chunk] = {}

    for ranked, w in zip(ranked_lists, ws):
        if not ranked or w == 0:
            continue
        for rank, (chunk, _score) in enumerate(ranked, start=1):
            key = _chunk_key(chunk)
            rrf_scores[key] += w * (1.0 / (rrf_k + rank))
            first_seen.setdefault(key, chunk)

    # Sort by RRF score descending, take top-k.
    ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(first_seen[key], score) for key, score in ordered[:k]]
