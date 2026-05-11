"""Reciprocal Rank Fusion — pure-Python, no external deps."""
from __future__ import annotations

import pytest

from polimibot.rag.chunker import Chunk
from polimibot.rag.fusion import RRF_K, reciprocal_rank_fusion


def _c(idx: int, source: str = None) -> Chunk:
    return Chunk(text=f"t{idx}", source=source or f"S{idx}", chunk_id=idx)


def _rank(items: list[Chunk]) -> list[tuple[Chunk, float]]:
    """Helper: build a ranked list from chunks (scores are placeholder)."""
    return [(c, 1.0 - 0.01 * i) for i, c in enumerate(items)]


# ── Basics ──────────────────────────────────────────────────────────────────


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_single_list_preserves_order():
    """One input list → fusion is a no-op (RRF score is just 1/(k+rank))."""
    chunks = [_c(0), _c(1), _c(2)]
    out = reciprocal_rank_fusion([_rank(chunks)], k=10)
    assert [c.chunk_id for c, _ in out] == [0, 1, 2]


def test_rrf_chunk_in_both_lists_outranks_chunk_in_one():
    """Appearing high in TWO lists beats appearing high in one."""
    a, b, c = _c(0), _c(1), _c(2)
    list1 = _rank([a, b])      # a@1, b@2
    list2 = _rank([c, a])      # c@1, a@2

    out = reciprocal_rank_fusion([list1, list2], k=3)
    # a: 1/(60+1) + 1/(60+2) ≈ 0.0327
    # b: 1/(60+2)             ≈ 0.0161
    # c: 1/(60+1)             ≈ 0.0164
    # → a > c > b
    assert [c.chunk_id for c, _ in out] == [0, 2, 1]


def test_rrf_truncates_to_k():
    chunks = [_c(i) for i in range(10)]
    out = reciprocal_rank_fusion([_rank(chunks)], k=3)
    assert len(out) == 3


def test_rrf_k_larger_than_pool_returns_all():
    chunks = [_c(0), _c(1)]
    out = reciprocal_rank_fusion([_rank(chunks)], k=99)
    assert len(out) == 2


def test_rrf_score_decreases_monotonically_in_output():
    chunks = [_c(0), _c(1), _c(2), _c(3)]
    out = reciprocal_rank_fusion([_rank(chunks)], k=4)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


# ── Score-scale independence ────────────────────────────────────────────────


def test_rrf_ignores_input_scores():
    """RRF uses ranks, not scores. Wildly different score scales must
    produce the same fusion as long as the ranks are the same."""
    a, b = _c(0), _c(1)
    cheap_scores  = [(a, 0.01),  (b, 0.005)]
    huge_scores   = [(a, 9999.0), (b, 9000.0)]
    out_cheap = reciprocal_rank_fusion([cheap_scores], k=2)
    out_huge  = reciprocal_rank_fusion([huge_scores], k=2)
    assert [c.chunk_id for c, _ in out_cheap] == [c.chunk_id for c, _ in out_huge]
    assert pytest.approx(out_cheap[0][1]) == out_huge[0][1]


# ── Identity / deduplication ────────────────────────────────────────────────


def test_rrf_dedupes_by_source_and_chunk_id_not_object_identity():
    """Two Chunk objects with the same (source, chunk_id) must be treated
    as the SAME passage — important when dense and BM25 load chunks from
    separate sidecar files and emit different Python objects for the
    same underlying chunk."""
    dense_a  = Chunk(text="from dense",  source="Caesar", chunk_id=0)
    bm25_a   = Chunk(text="from bm25",   source="Caesar", chunk_id=0)
    other    = Chunk(text="other",       source="Other",  chunk_id=0)

    list1 = [(dense_a, 0.9), (other, 0.5)]
    list2 = [(bm25_a, 0.8)]
    out = reciprocal_rank_fusion([list1, list2], k=3)
    # Caesar/0 should appear once, not twice.
    keys = [(c.source, c.chunk_id) for c, _ in out]
    assert keys.count(("Caesar", 0)) == 1
    # Caesar/0 ranks above 'Other' (in two lists vs one).
    assert keys[0] == ("Caesar", 0)


def test_rrf_returns_first_seen_chunk_for_duplicates():
    """When the same key appears across lists with different Chunk
    instances, the first one seen wins for output identity (its .text
    is what downstream sees)."""
    dense_a  = Chunk(text="from dense", source="A", chunk_id=0)
    bm25_a   = Chunk(text="from bm25",  source="A", chunk_id=0)
    out = reciprocal_rank_fusion([[(dense_a, 0.9)], [(bm25_a, 0.8)]], k=1)
    assert out[0][0].text == "from dense"   # dense was first


# ── Constant ────────────────────────────────────────────────────────────────


def test_rrf_default_k_is_60():
    """Standard from Cormack et al. 2009. Changing this is a research call —
    pin it down in tests so an accidental edit shows up."""
    assert RRF_K == 60


# ── Tolerance for partial overlap ───────────────────────────────────────────


def test_rrf_handles_list_with_unique_chunk():
    """A chunk that appears in only one list still contributes its single
    rank reciprocal — it just can't compete with multi-list winners."""
    only_in_list1 = _c(99)
    a = _c(0)
    out = reciprocal_rank_fusion(
        [[(a, 1.0), (only_in_list1, 0.9)], [(a, 1.0)]],
        k=5,
    )
    ids = [c.chunk_id for c, _ in out]
    assert ids == [0, 99]
