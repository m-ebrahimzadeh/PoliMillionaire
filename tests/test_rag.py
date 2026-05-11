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


# ── Manifest support ─────────────────────────────────────────────────────────


def test_index_save_writes_manifest_alongside(tmp_path):
    idx = FAISSIndex(dim=8)
    chunks = _make_chunks(3)
    idx.add(chunks, _random_vecs(3, dim=8))
    manifest = {
        "embedder_model_name": "all-MiniLM-L6-v2",
        "embedder_dim": 8,
        "normalize": True,
        "chunk_size": 300,
        "chunk_overlap": 50,
        "n_articles": 5,
        "text_cleanup_version": 1,
    }
    idx.save(tmp_path / "idx", manifest=manifest)
    mpath = (tmp_path / "idx").with_suffix(".manifest.json")
    assert mpath.is_file()
    import json
    written = json.loads(mpath.read_text())
    # Required fields preserved
    assert written["embedder_model_name"] == "all-MiniLM-L6-v2"
    assert written["chunk_size"] == 300
    # Auto-filled fields
    assert "build_timestamp" in written
    assert written["n_chunks"] == 3


def test_index_load_without_manifest_warns_but_works(tmp_path):
    idx = FAISSIndex(dim=8)
    idx.add(_make_chunks(2), _random_vecs(2, dim=8))
    idx.save(tmp_path / "idx")   # no manifest

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = FAISSIndex.load(tmp_path / "idx")
    assert loaded.n_chunks == 2
    assert loaded.manifest is None
    assert any("manifest" in str(w.message).lower() for w in caught)


def test_index_load_reads_manifest(tmp_path):
    idx = FAISSIndex(dim=8)
    idx.add(_make_chunks(1), _random_vecs(1, dim=8))
    idx.save(tmp_path / "idx", manifest={
        "embedder_model_name": "all-MiniLM-L6-v2",
        "embedder_dim": 8,
        "normalize": True,
    })
    loaded = FAISSIndex.load(tmp_path / "idx")
    assert loaded.manifest is not None
    assert loaded.manifest["embedder_model_name"] == "all-MiniLM-L6-v2"


def test_retriever_from_saved_rejects_model_mismatch(tmp_path):
    """Different embedder name → vectors live in incompatible spaces. Refuse to load."""
    from polimibot.rag.retriever import _check_manifest_compat
    from polimibot.rag.embedder import EmbedderSpec

    spec_wrong = EmbedderSpec(model_name="bge-small-en-v1.5")
    manifest = {
        "embedder_model_name": "all-MiniLM-L6-v2",
        "embedder_dim": 384,
        "normalize": True,
    }
    with pytest.raises(ValueError, match="incompatible spaces"):
        _check_manifest_compat(manifest, spec_wrong)


# ── Category filter ──────────────────────────────────────────────────────────


def test_chunk_records_category_when_provided():
    from polimibot.rag.chunker import chunk_text
    chunks = chunk_text("word " * 50, source="x", chunk_size=20, overlap=0,
                        category="maths")
    assert all(c.category == "maths" for c in chunks)


def test_chunk_category_defaults_to_none_for_back_compat():
    from polimibot.rag.chunker import chunk_text
    chunks = chunk_text("word " * 50, source="x", chunk_size=20, overlap=0)
    assert all(c.category is None for c in chunks)


def test_index_save_load_preserves_category(tmp_path):
    """Round-trip a mixed-category index through .jsonl, verify category survives."""
    idx = FAISSIndex(dim=8)
    chunks = [
        Chunk(text="m", source="A", chunk_id=0, category="maths"),
        Chunk(text="h", source="B", chunk_id=0, category="history"),
        Chunk(text="legacy", source="C", chunk_id=0, category=None),
    ]
    idx.add(chunks, _random_vecs(3, dim=8))
    idx.save(tmp_path / "idx", manifest={"embedder_model_name": "m", "embedder_dim": 8})

    loaded = FAISSIndex.load(tmp_path / "idx")
    cats = [c.category for c in loaded._chunks]
    assert cats == ["maths", "history", None]


def test_retriever_category_filter_drops_off_category_chunks():
    """When category= is passed, only matching chunks come back."""
    from polimibot.rag.retriever import Retriever
    import numpy as np

    # Build a 4-chunk index: 2 maths, 2 history. All embeddings identical
    # so cosine score doesn't bias the ranking — we're testing the filter.
    idx = FAISSIndex(dim=8)
    chunks = [
        Chunk(text="m1", source="M1", chunk_id=0, category="maths"),
        Chunk(text="m2", source="M2", chunk_id=0, category="maths"),
        Chunk(text="h1", source="H1", chunk_id=0, category="history"),
        Chunk(text="h2", source="H2", chunk_id=0, category="history"),
    ]
    same_vec = np.ones((4, 8), dtype=np.float32)
    same_vec /= np.linalg.norm(same_vec[0])
    idx.add(chunks, same_vec)

    class _FixedEmbedder:
        dim = 8
        def encode(self, texts):
            v = np.ones((1, 8), dtype=np.float32)
            return v / np.linalg.norm(v[0])

    r = Retriever(idx, _FixedEmbedder())  # type: ignore[arg-type]
    hits = r.retrieve("anything", k=3, category="maths")
    assert all(c.category == "maths" for c, _ in hits)


def test_retriever_no_category_returns_all_categories():
    """Without category=, the filter is off — chunks from any category appear."""
    from polimibot.rag.retriever import Retriever
    import numpy as np

    idx = FAISSIndex(dim=8)
    chunks = [
        Chunk(text="m", source="M", chunk_id=0, category="maths"),
        Chunk(text="h", source="H", chunk_id=0, category="history"),
    ]
    idx.add(chunks, np.eye(2, 8, dtype=np.float32))

    class _Embed:
        dim = 8
        def encode(self, texts):
            v = np.ones((1, 8), dtype=np.float32)
            return v / np.linalg.norm(v[0])

    r = Retriever(idx, _Embed())  # type: ignore[arg-type]
    hits = r.retrieve("anything", k=2)
    cats = {c.category for c, _ in hits}
    assert cats == {"maths", "history"}


def test_retriever_from_saved_warns_on_normalize_drift():
    """normalize mismatch is less catastrophic — warn, don't raise."""
    from polimibot.rag.retriever import _check_manifest_compat
    from polimibot.rag.embedder import EmbedderSpec
    import warnings

    spec = EmbedderSpec(model_name="all-MiniLM-L6-v2", normalize=False)
    manifest = {
        "embedder_model_name": "all-MiniLM-L6-v2",
        "embedder_dim": 384,
        "normalize": True,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_manifest_compat(manifest, spec)
    assert any("normalize" in str(w.message).lower() for w in caught)


# NB: clean_wikipedia_text tests live in tests/test_corpus.py — they don't
# need FAISS, so keeping them there means they run even when faiss-cpu
# isn't installed (CI fast-path).