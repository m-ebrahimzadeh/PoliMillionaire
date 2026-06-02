"""CrossEncoderReranker — pure-Python, no torch / sentence-transformers needed."""
from __future__ import annotations

import pytest

from polimibot.rag.chunker import Chunk
from polimibot.rag.reranker import CrossEncoderReranker, RerankerSpec


def _chunk(idx: int, text: str = "doc") -> Chunk:
    return Chunk(text=text, source=f"S{idx}", chunk_id=idx, category=None)


def _candidates(*texts: str) -> list[tuple[Chunk, float]]:
    return [(_chunk(i, t), 0.5 + 0.01 * i) for i, t in enumerate(texts)]


# ── Basic behaviour ──────────────────────────────────────────────────────────


def test_reranker_sorts_by_cross_encoder_score_descending():
    """The reranker should reorder by the scoring function, not preserve the
    dense order."""
    # Mock scorer: longer doc → higher score.
    reranker = CrossEncoderReranker(
        lambda pairs: [float(len(doc)) for _, doc in pairs]
    )
    # Lengths: "much longer document text" = 25, "short" = 5, "mid" = 3.
    # Dense order (chunk_ids 0,1,2) should be reordered descending → 1,0,2.
    cands = _candidates("short", "much longer document text", "mid")
    out = reranker.rerank("q", cands)
    assert [c.chunk_id for c, _ in out] == [1, 0, 2]


def test_reranker_score_replaces_dense_score():
    """Returned scores are cross-encoder outputs, NOT the dense scores
    from the input. Downstream code that thresholds on score sees a
    different scale."""
    reranker = CrossEncoderReranker(lambda pairs: [9.99 for _ in pairs])
    cands = _candidates("a", "b")
    out = reranker.rerank("q", cands)
    assert all(score == 9.99 for _, score in out)
    # Dense scores were 0.50, 0.51 — both replaced.


def test_reranker_top_k_truncates():
    reranker = CrossEncoderReranker(
        lambda pairs: [float(i) for i, _ in enumerate(pairs)]
    )
    cands = _candidates(*[f"doc{i}" for i in range(10)])
    out = reranker.rerank("q", cands, top_k=3)
    assert len(out) == 3
    # Highest-scoring (last in score order) come first.
    assert [c.chunk_id for c, _ in out] == [9, 8, 7]


def test_reranker_top_k_none_returns_all():
    reranker = CrossEncoderReranker(lambda pairs: [1.0] * len(pairs))
    cands = _candidates("a", "b", "c", "d", "e")
    out = reranker.rerank("q", cands, top_k=None)
    assert len(out) == 5


def test_reranker_empty_input_returns_empty():
    reranker = CrossEncoderReranker(lambda pairs: [])
    assert reranker.rerank("q", []) == []
    assert reranker.rerank("q", [], top_k=3) == []


def test_reranker_pair_count_mismatch_raises():
    """Scorer must return one float per input pair. Silent length drift
    would silently mis-pair scores with chunks."""
    bad = CrossEncoderReranker(lambda pairs: [1.0])  # always returns 1 score
    cands = _candidates("a", "b", "c")
    with pytest.raises(RuntimeError, match="scored 1 pairs but got 3"):
        bad.rerank("q", cands)


# ── Scoring contract ─────────────────────────────────────────────────────────


def test_reranker_passes_query_and_chunk_text_to_scorer():
    """The scorer receives (query, chunk.text) tuples — NOT chunk.source,
    NOT the dense score."""
    captured: list[tuple[str, str]] = []
    def scorer(pairs):
        captured.extend(pairs)
        return [0.0] * len(pairs)

    reranker = CrossEncoderReranker(scorer)
    cands = [
        (Chunk(text="body text 1", source="Title A", chunk_id=0), 0.9),
        (Chunk(text="body text 2", source="Title B", chunk_id=0), 0.1),
    ]
    reranker.rerank("the query", cands)
    assert captured == [
        ("the query", "body text 1"),
        ("the query", "body text 2"),
    ]


def test_reranker_preserves_chunk_identity():
    """Reranking changes order, not the Chunk objects."""
    reranker = CrossEncoderReranker(lambda pairs: [float(len(d)) for _, d in pairs])
    c1 = Chunk(text="short", source="A", chunk_id=42, category="maths")
    c2 = Chunk(text="much longer text", source="B", chunk_id=99, category="history")
    out = reranker.rerank("q", [(c1, 0.5), (c2, 0.3)])
    # c2 should win — and it should be the SAME object, with .category intact.
    assert out[0][0] is c2
    assert out[0][0].category == "history"


# ── Stable tie-breaking ──────────────────────────────────────────────────────


def test_reranker_stable_order_on_ties():
    """When scores tie, input order is preserved (Python sort is stable)."""
    reranker = CrossEncoderReranker(lambda pairs: [1.0] * len(pairs))
    cands = _candidates("a", "b", "c", "d")
    out = reranker.rerank("q", cands)
    assert [c.chunk_id for c, _ in out] == [0, 1, 2, 3]


# ── Spec defaults ────────────────────────────────────────────────────────────


def test_default_spec_targets_bge_reranker():
    """Default model is the trivia-friendly bge-reranker-base. Document drift
    here intentional — a change to the default should be a conscious commit."""
    spec = RerankerSpec()
    assert spec.model_name == "BAAI/bge-reranker-base"
    assert spec.batch_size == 32


def test_reranker_spec_fp16_defaults_to_auto():
    """fp16 defaults to None (auto: fp16 on CUDA, fp32 on CPU) for back-compat."""
    assert RerankerSpec().fp16 is None
    assert RerankerSpec(fp16=True).fp16 is True
    assert RerankerSpec(fp16=False).fp16 is False
