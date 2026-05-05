"""RAG unit tests. Chunker is pure Python (no deps). Index/Retriever tests
skip cleanly if faiss is not installed in the dev environment."""
from __future__ import annotations

import numpy as np
import pytest

from polimibot.rag.chunker import Chunk, chunk_text

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed — skipping index tests")


# ── Chunker (no external deps) ──────────────────────────────────────────────

def test_chunk_splits_into_windows():
    text = " ".join(str(i) for i in range(100))  # 100 words: "0 1 2 ... 99"
    chunks = chunk_text(text, source="test", chunk_size=30, overlap=10)
    assert len(chunks) > 1
    # Each chunk should have at most 30 words
    for c in chunks:
        assert len(c.text.split()) <= 30


def test_chunk_overlap_shares_words():
    text = " ".join(str(i) for i in range(50))
    chunks = chunk_text(text, source="test", chunk_size=20, overlap=5)
    # Last word of chunk[0] should appear in chunk[1]
    last_word_of_first = chunks[0].text.split()[-1]
    assert last_word_of_first in chunks[1].text


def test_chunk_source_and_ids_preserved():
    chunks = chunk_text("word " * 50, source="wiki:Rome", chunk_size=20, overlap=0)
    assert all(c.source == "wiki:Rome" for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_chunk_empty_text_returns_empty():
    assert chunk_text("", source="x") == []


def test_chunk_short_text_is_single_chunk():
    chunks = chunk_text("hello world", source="x", chunk_size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


# ── FAISSIndex (requires faiss-cpu) ─────────────────────────────────────────

from polimibot.rag.index import FAISSIndex


def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(text=f"chunk {i}", source="test", chunk_id=i) for i in range(n)]


def _random_vecs(n: int, dim: int = 8) -> np.ndarray:
    """Normalized random vectors — simulates embedder output."""
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_index_add_and_search():
    dim = 8
    idx = FAISSIndex(dim=dim)
    chunks = _make_chunks(10)
    vecs = _random_vecs(10, dim)
    idx.add(chunks, vecs)
    assert idx.n_chunks == 10

    # Query with the exact vector of chunk 0 — must be its own top result
    results = idx.search(vecs[0:1], k=3)
    assert results[0][0].chunk_id == 0
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)  # cosine of identical vecs = 1


def test_index_shape_mismatch_raises():
    idx = FAISSIndex(dim=8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        idx.add(_make_chunks(3), _random_vecs(2, dim=8))  # 3 chunks, 2 vectors


def test_index_save_load_roundtrip(tmp_path):
    dim = 8
    idx = FAISSIndex(dim=dim)
    chunks = _make_chunks(5)
    vecs = _random_vecs(5, dim)
    idx.add(chunks, vecs)
    idx.save(tmp_path / "test_idx")

    loaded = FAISSIndex.load(tmp_path / "test_idx")
    assert loaded.n_chunks == 5
    # Same query → same top result
    r1 = idx.search(vecs[2:3], k=1)
    r2 = loaded.search(vecs[2:3], k=1)
    assert r1[0][0].chunk_id == r2[0][0].chunk_id


def test_retriever_dim_mismatch_raises():
    """Catch the easy mistake of building with one embedder and querying with another."""
    from polimibot.rag.embedder import Embedder, EmbedderSpec
    from polimibot.rag.retriever import Retriever

    class FakeEmbedder:
        dim = 16  # wrong dimension

    idx = FAISSIndex(dim=8)
    with pytest.raises(ValueError, match="dim=8 != embedder dim=16"):
        Retriever(idx, FakeEmbedder())  # type: ignore[arg-type]