"""Unit tests for IndexGrower — no GPU, no FAISS, no network required.

Uses lightweight mock objects for Retriever, FAISSIndex, BM25Index, and
Embedder so the tests run on CPU in any environment.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from polimibot.config import Category
from polimibot.rag.chunker import Chunk
from polimibot.rag.corpus import Article
from polimibot.rag.index_grower import IndexGrower


# ── Shared fixtures ───────────────────────────────────────────────────────────

DIM = 8  # tiny embedding dimension for fast tests


def _make_embedder(dim: int = DIM) -> MagicMock:
    """Mock Embedder that returns random unit vectors."""
    emb = MagicMock()
    emb.dim = dim

    def _encode_passage(texts):
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)

    emb.encode_passage.side_effect = _encode_passage
    return emb


def _make_faiss_index(dim: int = DIM) -> MagicMock:
    """Mock FAISSIndex that tracks append() calls."""
    idx = MagicMock()
    idx.dim = dim
    idx._chunks = []
    idx.manifest = {"embedder_model_name": "test-model", "n_chunks": 0}

    def _append(chunks, embeddings):
        idx._chunks.extend(chunks)

    idx.append.side_effect = _append

    @property
    def n_chunks(self):
        return len(self._chunks)

    idx.n_chunks = 0  # will be updated via property below

    def _save(path, *, manifest=None):
        pass  # no-op in tests

    idx.save.side_effect = _save
    return idx


def _make_bm25_index() -> MagicMock:
    """Mock BM25Index that tracks append() calls."""
    bm = MagicMock()
    bm._chunks = []

    def _append(chunks):
        bm._chunks.extend(chunks)

    bm.append.side_effect = _append

    def _save(path):
        pass

    bm.save.side_effect = _save
    return bm


def _make_retriever(dim: int = DIM) -> MagicMock:
    """Mock Retriever with a mock FAISSIndex and BM25Index."""
    retriever = MagicMock()
    retriever._index = _make_faiss_index(dim)
    retriever._bm25 = _make_bm25_index()

    appended_chunks = []
    appended_embeddings = []

    def _append_chunks(chunks, embeddings):
        appended_chunks.extend(chunks)
        appended_embeddings.append(embeddings)
        retriever._index._chunks.extend(chunks)
        retriever._bm25._chunks.extend(chunks)

    retriever.append_chunks.side_effect = _append_chunks
    retriever._appended_chunks = appended_chunks
    return retriever


def _article(title: str = "Julius Caesar",
             text: str = "Caesar crossed the Rubicon in 49 BC.",
             category: Category = Category.HISTORY) -> Article:
    return Article(title=title, text=text, category=category, url="")


def _make_grower(tmp_path: Path, retriever=None, embedder=None) -> IndexGrower:
    retriever = retriever or _make_retriever()
    embedder  = embedder  or _make_embedder()
    index_path = tmp_path / "knowledge"
    return IndexGrower(
        retriever, embedder, index_path,
        corpus_path=tmp_path / "corpus.jsonl",
    )


# ── Construction ──────────────────────────────────────────────────────────────

def test_grower_initial_state(tmp_path):
    grower = _make_grower(tmp_path)
    assert grower.n_pending == 0
    assert grower.n_learned == 0


# ── buffer() ──────────────────────────────────────────────────────────────────

def test_buffer_returns_chunks(tmp_path):
    grower = _make_grower(tmp_path)
    chunks = grower.buffer(_article(), "q1")
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_buffer_increments_pending(tmp_path):
    grower = _make_grower(tmp_path)
    grower.buffer(_article(), "q1")
    assert grower.n_pending == 1


def test_buffer_dedup_same_title(tmp_path):
    """Buffering the same article title twice returns existing chunks."""
    grower = _make_grower(tmp_path)
    chunks1 = grower.buffer(_article(title="Julius Caesar"), "q1")
    chunks2 = grower.buffer(_article(title="Julius Caesar"), "q2")
    # Second buffer returns the same chunk list (not an empty list or new list).
    assert chunks1 == chunks2
    # Still only one pending entry.
    assert grower.n_pending == 1


def test_buffer_skips_already_indexed_title(tmp_path):
    """If a title is already in the offline index, buffer() returns []."""
    retriever = _make_retriever()
    # Pre-populate the FAISS mock with a chunk from "Julius Caesar".
    retriever._index._chunks = [Chunk(text="...", source="Julius Caesar", chunk_id=0)]

    grower = _make_grower(tmp_path, retriever=retriever)
    chunks = grower.buffer(_article(title="Julius Caesar"), "q1")
    assert chunks == []
    assert grower.n_pending == 0


def test_buffer_preserves_category(tmp_path):
    grower = _make_grower(tmp_path)
    chunks = grower.buffer(_article(category=Category.SCIENCE), "q1")
    assert all(c.category == "science" for c in chunks)


# ── confirm() ─────────────────────────────────────────────────────────────────

def test_confirm_appends_to_retriever(tmp_path):
    retriever = _make_retriever()
    grower = _make_grower(tmp_path, retriever=retriever)

    grower.buffer(_article(), "q1")
    assert grower.n_pending == 1

    grower.confirm("q1")

    # Confirm should have called retriever.append_chunks once.
    assert retriever.append_chunks.called
    assert grower.n_learned == 1
    assert grower.n_pending == 0


def test_confirm_noop_on_unknown_id(tmp_path):
    retriever = _make_retriever()
    grower = _make_grower(tmp_path, retriever=retriever)
    # confirm() for a key that was never buffered — must be a silent no-op.
    grower.confirm("not_buffered")
    assert not retriever.append_chunks.called
    assert grower.n_learned == 0


def test_confirm_removes_from_pending(tmp_path):
    grower = _make_grower(tmp_path)
    grower.buffer(_article(), "q1")
    grower.confirm("q1")
    assert grower.n_pending == 0


def test_confirm_prevents_double_learning(tmp_path):
    retriever = _make_retriever()
    grower = _make_grower(tmp_path, retriever=retriever)
    grower.buffer(_article(), "q1")
    grower.confirm("q1")
    # Confirm the same question_id again — no double append.
    grower.confirm("q1")
    assert retriever.append_chunks.call_count == 1
    assert grower.n_learned == 1


# ── discard() ─────────────────────────────────────────────────────────────────

def test_discard_removes_pending(tmp_path):
    retriever = _make_retriever()
    grower = _make_grower(tmp_path, retriever=retriever)
    grower.buffer(_article(), "q1")
    grower.discard("q1")
    assert grower.n_pending == 0
    assert not retriever.append_chunks.called
    assert grower.n_learned == 0


def test_discard_noop_on_unknown_id(tmp_path):
    grower = _make_grower(tmp_path)
    grower.discard("never_buffered")  # must not raise


# ── flush() ───────────────────────────────────────────────────────────────────

def test_flush_noop_when_nothing_learned(tmp_path):
    """flush() with no confirmed articles must not touch disk at all."""
    grower = _make_grower(tmp_path)
    grower.flush()
    # No corpus.jsonl created.
    assert not (tmp_path / "corpus.jsonl").exists()


def test_flush_saves_faiss_after_confirm(tmp_path):
    retriever = _make_retriever()
    grower = _make_grower(tmp_path, retriever=retriever)
    grower.buffer(_article(), "q1")
    grower.confirm("q1")
    grower.flush()
    # FAISSIndex.save() should have been called exactly once.
    retriever._index.save.assert_called_once()


def test_flush_saves_bm25_after_confirm(tmp_path):
    retriever = _make_retriever()
    grower = _make_grower(tmp_path, retriever=retriever)
    grower.buffer(_article(), "q1")
    grower.confirm("q1")
    grower.flush()
    retriever._bm25.save.assert_called_once()


def test_flush_appends_to_corpus_jsonl(tmp_path):
    """Confirmed articles are appended to corpus.jsonl."""
    retriever = _make_retriever()
    grower = _make_grower(tmp_path, retriever=retriever)
    grower.buffer(_article(title="Julius Caesar", text="Caesar crossed the Rubicon."), "q1")
    grower.confirm("q1")
    grower.flush()

    corpus_path = tmp_path / "corpus.jsonl"
    assert corpus_path.exists()
    lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["title"] == "Julius Caesar"
    assert record["category"] == "history"


def test_flush_dedup_corpus_existing_title(tmp_path):
    """If the article title is already in corpus.jsonl, don't append again."""
    corpus_path = tmp_path / "corpus.jsonl"
    # Pre-populate corpus with Julius Caesar.
    corpus_path.write_text(
        json.dumps({"title": "Julius Caesar", "text": "old text",
                    "category": "history", "url": ""}) + "\n",
        encoding="utf-8",
    )

    retriever = _make_retriever()
    grower = IndexGrower(
        retriever, _make_embedder(), tmp_path / "knowledge",
        corpus_path=corpus_path,
    )
    grower.buffer(_article(title="Julius Caesar"), "q1")
    grower.confirm("q1")
    grower.flush()

    lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
    # Still only the original line — no duplicate.
    assert len(lines) == 1


def test_flush_multiple_confirms(tmp_path):
    """Two different articles confirmed in one session → both in corpus."""
    retriever = _make_retriever()
    grower = _make_grower(tmp_path, retriever=retriever)
    grower.buffer(_article(title="Julius Caesar"), "q1")
    grower.buffer(_article(title="Roman Republic", text="The Republic started in 509 BC."), "q2")
    grower.confirm("q1")
    grower.confirm("q2")
    grower.flush()

    corpus_path = tmp_path / "corpus.jsonl"
    lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
    titles = {json.loads(l)["title"] for l in lines}
    assert "Julius Caesar" in titles
    assert "Roman Republic" in titles


# ── FAISSIndex.append() ───────────────────────────────────────────────────────

def test_faiss_append_raises_on_dim_mismatch():
    """FAISSIndex.append() must raise on shape mismatch."""
    pytest.importorskip("faiss")
    from polimibot.rag.index import FAISSIndex

    idx = FAISSIndex(dim=4)
    chunks = [Chunk(text="hi", source="S", chunk_id=0)]
    bad_embeddings = np.ones((1, 8), dtype=np.float32)  # dim=8 ≠ 4
    with pytest.raises(ValueError, match="Append shape mismatch"):
        idx.append(chunks, bad_embeddings)


def test_faiss_append_empty_is_noop():
    pytest.importorskip("faiss")
    from polimibot.rag.index import FAISSIndex

    idx = FAISSIndex(dim=4)
    before = idx.n_chunks
    idx.append([], np.empty((0, 4), dtype=np.float32))
    assert idx.n_chunks == before


def test_faiss_append_grows_n_chunks():
    pytest.importorskip("faiss")
    from polimibot.rag.index import FAISSIndex

    idx = FAISSIndex(dim=4)
    chunks = [Chunk(text="hello world", source="S", chunk_id=0)]
    embeddings = np.random.randn(1, 4).astype(np.float32)
    idx.append(chunks, embeddings)
    assert idx.n_chunks == 1


# ── BM25Index.append() ────────────────────────────────────────────────────────

def test_bm25_append_grows_n_chunks():
    from polimibot.rag.bm25 import BM25Index

    initial_chunks = [Chunk(text="Caesar crossed the Rubicon.", source="A", chunk_id=0)]
    idx = BM25Index(initial_chunks)
    assert idx.n_chunks == 1

    new_chunks = [Chunk(text="The Roman Republic was founded.", source="B", chunk_id=0)]
    idx.append(new_chunks)
    assert idx.n_chunks == 2


def test_bm25_append_searchable():
    """Chunks appended via append() should be findable via search()."""
    from polimibot.rag.bm25 import BM25Index

    idx = BM25Index([Chunk(text="unrelated text", source="A", chunk_id=0)])
    idx.append([Chunk(text="Caesar crossed the Rubicon in 49 BC.", source="B", chunk_id=0)])

    results = idx.search("Caesar Rubicon", k=1)
    assert len(results) == 1
    assert results[0][0].source == "B"


def test_bm25_append_updates_idf():
    """After append(), IDF for new tokens must exist in the table."""
    from polimibot.rag.bm25 import BM25Index

    idx = BM25Index([Chunk(text="apple orange", source="A", chunk_id=0)])
    assert "rubicon" not in idx._idf

    idx.append([Chunk(text="Caesar Rubicon Rubicon", source="B", chunk_id=0)])
    assert "rubicon" in idx._idf


def test_bm25_append_empty_is_noop():
    from polimibot.rag.bm25 import BM25Index

    idx = BM25Index([Chunk(text="hello", source="A", chunk_id=0)])
    before = idx.n_chunks
    idx.append([])
    assert idx.n_chunks == before


# ── Retriever.append_chunks() ─────────────────────────────────────────────────

def test_retriever_append_chunks_delegates_to_faiss_and_bm25():
    pytest.importorskip("faiss")
    from polimibot.rag.index import FAISSIndex
    from polimibot.rag.bm25 import BM25Index
    from polimibot.rag.embedder import Embedder, EmbedderSpec
    from polimibot.rag.retriever import Retriever

    dim = 4
    # Build a minimal in-memory index with one chunk.
    idx = FAISSIndex(dim=dim)
    existing_chunk = Chunk(text="original chunk", source="Orig", chunk_id=0)
    idx.add([existing_chunk], np.random.randn(1, dim).astype(np.float32))

    bm25 = BM25Index([existing_chunk])

    # Use a mock embedder (no sentence-transformers needed).
    embedder = MagicMock()
    embedder.dim = dim
    retriever = Retriever(idx, embedder, bm25=bm25)

    assert retriever.n_chunks == 1
    new_chunk = Chunk(text="new chunk about Caesar", source="Caesar", chunk_id=0)
    new_embeddings = np.random.randn(1, dim).astype(np.float32)

    retriever.append_chunks([new_chunk], new_embeddings)

    assert retriever.n_chunks == 2
    assert bm25.n_chunks == 2
