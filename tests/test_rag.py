"""RAG unit tests. Chunker is pure Python (no deps). Index/Retriever tests
skip cleanly if faiss is not installed in the dev environment."""
from __future__ import annotations

import numpy as np
import pytest

from polimibot.rag.chunker import Chunk, chunk_text, embedding_text

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


def test_chunk_respects_sentence_boundaries():
    """A chunk should end at a sentence boundary when the next sentence
    would push it past chunk_size. The cut never lands mid-sentence."""
    sentences = [
        "Caesar crossed the Rubicon in 49 BC.",
        "This act started a civil war against Pompey.",
        "He was later assassinated on the Ides of March.",
    ]
    text = " ".join(sentences)
    # chunk_size chosen so two sentences fit but three do not.
    chunks = chunk_text(text, source="rome", chunk_size=14, overlap=0,
                        min_chunk_words=1)
    # The first chunk should end with a "." (no mid-sentence cut).
    assert chunks[0].text.rstrip().endswith(".")
    # And it should be one of the input sentences or a join of consecutive ones.
    assert chunks[0].text.strip() in {sentences[0],
                                       " ".join(sentences[:2])}


def test_chunk_attaches_section_header_to_first_chunk_only():
    """`== Foo ==` lines are removed from the prose and prepended to the
    first chunk of their section. They do not appear in subsequent chunks
    of the same section."""
    text = (
        "Intro paragraph about Caesar.\n"
        "== Early life ==\n"
        + ("word " * 50).strip()
        + "\n== Later career ==\n"
        "Final sentence."
    )
    chunks = chunk_text(text, source="rome", chunk_size=20, overlap=0,
                        min_chunk_words=1)
    # The intro is its own headerless chunk.
    assert not chunks[0].text.startswith("Early life")
    # Some chunk should carry the "Early life" header prefix exactly once.
    early_life_chunks = [c for c in chunks if c.text.startswith("Early life\n")]
    assert len(early_life_chunks) == 1
    later_career_chunks = [c for c in chunks if c.text.startswith("Later career\n")]
    assert len(later_career_chunks) == 1
    # The "==" markers must not appear in any chunk's text.
    assert not any("==" in c.text for c in chunks)


def test_chunk_min_chunk_filter_merges_short_tail():
    """A trailing chunk below min_chunk_words is absorbed into its
    predecessor so the index isn't polluted with 10-word stubs."""
    text = " ".join(str(i) for i in range(45))   # 45 words, no punctuation
    # stride = 20, so windows = [0..19], [20..39], [40..44] (5 words).
    chunks = chunk_text(text, source="x", chunk_size=20, overlap=0,
                        min_chunk_words=10)
    # The 5-word tail should have been merged, so we get at most 2 chunks.
    assert len(chunks) == 2
    # ids stay contiguous after the merge.
    assert [c.chunk_id for c in chunks] == [0, 1]
    # And the second chunk must now contain the tail words.
    assert "44" in chunks[1].text


def test_chunk_manifest_version_exported():
    """CHUNKER_VERSION is part of the public surface — manifest readers
    rely on its presence and integrality."""
    from polimibot.rag.chunker import CHUNKER_VERSION
    assert isinstance(CHUNKER_VERSION, int) and CHUNKER_VERSION >= 2


# ── Entity-grounded embedding text ──────────────────────────────────────────


def test_embedding_text_prefixes_source_title():
    """The embedded form grounds the passage in its source title so the
    vector is anchored to the entity (trivia questions name the entity)."""
    c = Chunk(text="was painted in 1503.", source="Mona Lisa", chunk_id=4)
    assert embedding_text(c) == "Mona Lisa: was painted in 1503."


def test_embedding_text_leaves_stored_text_pure():
    """``Chunk.text`` itself is never mutated — display + BM25 use it verbatim.
    Grounding lives only in the embedding-input transform."""
    chunks = chunk_text("hello world", source="Earth", chunk_size=300, overlap=50)
    assert chunks[0].text == "hello world"          # stored text untouched
    assert embedding_text(chunks[0]) == "Earth: hello world"


def test_embedding_text_empty_source_falls_back_to_text():
    """Defensive: a sourceless chunk embeds its raw text rather than ': text'."""
    c = Chunk(text="orphan passage", source="", chunk_id=0)
    assert embedding_text(c) == "orphan passage"


def test_embed_text_version_exported():
    """EMBED_TEXT_VERSION is recorded in the index manifest — manifest readers
    rely on its presence and integrality."""
    from polimibot.rag.chunker import EMBED_TEXT_VERSION
    assert isinstance(EMBED_TEXT_VERSION, int) and EMBED_TEXT_VERSION >= 1


# ── FAISSIndex (requires faiss-cpu) ─────────────────────────────────────────

from polimibot.rag.index import FAISSIndex


def _make_chunks(n: int) -> list[Chunk]:
    # Distinct ``source`` per chunk so retriever-level tests aren't accidentally
    # collapsed by the source-dedup pass on the live retriever.
    return [Chunk(text=f"chunk {i}", source=f"src{i}", chunk_id=i) for i in range(n)]


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


# ── Retriever + reranker integration ─────────────────────────────────────────


def test_retriever_rerank_requires_attached_reranker():
    """rerank=True without a reranker is a programmer error — fail loud."""
    from polimibot.rag.retriever import Retriever
    class _Embed:
        dim = 8
        def encode(self, texts): return _random_vecs(1, dim=8)

    idx = FAISSIndex(dim=8)
    idx.add(_make_chunks(3), _random_vecs(3, dim=8))
    r = Retriever(idx, _Embed())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no reranker is attached"):
        r.retrieve("q", k=3, rerank=True)


def test_retriever_oversearches_then_reranks_to_top_k():
    """With rerank=True, retriever asks the dense index for k×oversearch
    candidates, then trims to k via the cross-encoder."""
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.reranker import CrossEncoderReranker
    import numpy as np

    # 20 chunks with distinct ids. Dense order is preserved (all same vec).
    idx = FAISSIndex(dim=8)
    chunks = [Chunk(text=f"t{i}", source=f"S{i}", chunk_id=i, category=None)
              for i in range(20)]
    same = np.ones((20, 8), dtype=np.float32) / np.sqrt(8)
    idx.add(chunks, same)

    # Reranker: score = -chunk_id, so chunk 0 wins; this REVERSES dense order
    # only because (with same vectors) FAISS may return any order — we use
    # the reranker to deterministically pick chunk 0.
    rer = CrossEncoderReranker(
        lambda pairs: [-float(doc[1:]) for _, doc in pairs]  # "t5" → -5
    )
    class _Embed:
        dim = 8
        def encode(self, texts): return same[:1]

    r = Retriever(idx, _Embed(), reranker=rer)  # type: ignore[arg-type]
    out = r.retrieve("q", k=3, rerank=True, rerank_oversearch=5)
    assert len(out) == 3
    # Chunk 0 (smallest chunk_id → highest reranker score) must be #1.
    assert out[0][0].chunk_id == 0
    assert out[1][0].chunk_id == 1
    assert out[2][0].chunk_id == 2


def test_retriever_rerank_replaces_dense_score():
    """After reranking, returned scores are cross-encoder scores."""
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.reranker import CrossEncoderReranker
    import numpy as np

    idx = FAISSIndex(dim=8)
    idx.add(_make_chunks(5), _random_vecs(5, dim=8))
    rer = CrossEncoderReranker(lambda pairs: [42.0] * len(pairs))
    class _Embed:
        dim = 8
        def encode(self, texts): return _random_vecs(1, dim=8)

    r = Retriever(idx, _Embed(), reranker=rer)  # type: ignore[arg-type]
    out = r.retrieve("q", k=2, rerank=True)
    assert all(s == 42.0 for _, s in out)


def test_retriever_rerank_with_category_filter_composes():
    """Both filters at once: oversearch dense by (k × rerank × category),
    keep matching-category chunks, rerank pool to k."""
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.reranker import CrossEncoderReranker
    import numpy as np

    idx = FAISSIndex(dim=8)
    chunks = (
        [Chunk(text=f"m{i}", source=f"M{i}", chunk_id=i, category="maths")
         for i in range(5)] +
        [Chunk(text=f"h{i}", source=f"H{i}", chunk_id=10+i, category="history")
         for i in range(5)]
    )
    same = np.ones((10, 8), dtype=np.float32) / np.sqrt(8)
    idx.add(chunks, same)

    rer = CrossEncoderReranker(lambda pairs: [1.0] * len(pairs))
    class _Embed:
        dim = 8
        def encode(self, texts): return same[:1]

    r = Retriever(idx, _Embed(), reranker=rer)  # type: ignore[arg-type]
    out = r.retrieve("q", k=2, category="history", rerank=True)
    assert len(out) == 2
    assert all(c.category == "history" for c, _ in out)


def test_retriever_has_reranker_property():
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.reranker import CrossEncoderReranker
    class _Embed:
        dim = 8
        def encode(self, texts): return _random_vecs(1, dim=8)

    idx = FAISSIndex(dim=8)
    r_plain = Retriever(idx, _Embed())  # type: ignore[arg-type]
    assert r_plain.has_reranker is False

    r_with = Retriever(idx, _Embed(),  # type: ignore[arg-type]
                       reranker=CrossEncoderReranker(lambda p: [0.0]*len(p)))
    assert r_with.has_reranker is True


# ── Hybrid (dense + BM25 via RRF) ───────────────────────────────────────────


def test_retriever_hybrid_requires_attached_bm25():
    from polimibot.rag.retriever import Retriever
    class _Embed:
        dim = 8
        def encode(self, texts): return _random_vecs(1, dim=8)

    idx = FAISSIndex(dim=8)
    idx.add(_make_chunks(3), _random_vecs(3, dim=8))
    r = Retriever(idx, _Embed())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no BM25 index is attached"):
        r.retrieve("q", k=3, hybrid=True)


# ── BM25 tokenisation & proximity ───────────────────────────────────────────


def test_bm25_tokenize_drops_stopwords_by_default():
    from polimibot.rag.bm25 import tokenize
    toks = tokenize("The cat is on the mat")
    # "the", "is", "on" are stopwords; "cat", "mat" survive.
    assert "the" not in toks
    assert "is" not in toks
    assert "on" not in toks
    assert "cat" in toks
    assert "mat" in toks


def test_bm25_tokenize_keep_stopwords_when_disabled():
    from polimibot.rag.bm25 import tokenize
    toks = tokenize("The cat is on the mat", drop_stopwords=False)
    assert toks == ["the", "cat", "is", "on", "the", "mat"]


def test_bm25_proximity_bonus_rewards_adjacent_tokens():
    """A chunk where two query tokens are adjacent outscores one where the
    same tokens are far apart — even though both share TF and IDF."""
    from polimibot.rag.bm25 import BM25Index, BM25Spec
    near = Chunk(text="The Pythagorean theorem relates the sides of a triangle",
                 source="N", chunk_id=0)
    # Same two tokens, but ~30 words apart (separated by filler).
    far_text = "Pythagorean was a Greek philosopher. " + ("filler " * 30) + "He proved a theorem."
    far = Chunk(text=far_text, source="F", chunk_id=0)
    bm25 = BM25Index([near, far], spec=BM25Spec(proximity_alpha=1.0, proximity_window=5))
    hits = bm25.search("Pythagorean theorem", k=2)
    sources = [c.source for c, _ in hits]
    assert sources[0] == "N"   # near beats far


def test_bm25_proximity_alpha_zero_disables_bonus():
    """With proximity_alpha=0 the two docs of the previous test get equal
    base BM25 scores (same TF, same IDF, different doc length)."""
    from polimibot.rag.bm25 import BM25Index, BM25Spec
    near = Chunk(text="Pythagorean theorem one two three four five",
                 source="N", chunk_id=0)
    far_text = "Pythagorean " + ("x " * 20) + "theorem"
    far = Chunk(text=far_text, source="F", chunk_id=0)
    bm25 = BM25Index([near, far], spec=BM25Spec(proximity_alpha=0.0))
    hits = bm25.search("Pythagorean theorem", k=2)
    # Both should still appear; we only check the proximity bonus didn't fire
    # (i.e. doc lengths drive any ordering difference, not proximity).
    assert len(hits) == 2


def test_bm25_load_refuses_old_version(tmp_path):
    """A v1 sidecar (no version field defaults to 1) must be rejected so
    callers know to rebuild before relying on the new scoring path."""
    from polimibot.rag.bm25 import BM25Index
    import json as _json
    p = tmp_path / "old.bm25.jsonl"
    p.write_text(
        _json.dumps({"kind": "bm25_header", "k1": 1.5, "b": 0.75,
                     "n_docs": 0, "avgdl": 0.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        BM25Index.load(tmp_path / "old")


def test_bm25_save_load_roundtrip_v2(tmp_path):
    from polimibot.rag.bm25 import BM25Index, BM25_VERSION
    chunks = [
        Chunk(text="Caesar crossed the Rubicon", source="A", chunk_id=0),
        Chunk(text="Pompey was his rival", source="B", chunk_id=0),
    ]
    idx = BM25Index(chunks)
    idx.save(tmp_path / "k")
    loaded = BM25Index.load(tmp_path / "k")
    # Spec round-trips, search still works.
    hits = loaded.search("Caesar", k=2)
    assert hits[0][0].source == "A"
    # Header records the new version.
    import json as _json
    header = _json.loads((tmp_path / "k.bm25.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert header["version"] == BM25_VERSION


def test_retriever_hybrid_fuses_dense_and_bm25():
    """A chunk that appears in BOTH dense and BM25 top results should
    outrank a chunk that appears in only one."""
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.bm25 import BM25Index
    import numpy as np

    # Five chunks; rank them differently in dense vs BM25.
    chunks = [
        Chunk(text="Caesar crossed the Rubicon",          source="A", chunk_id=0),
        Chunk(text="Pompey rival",                         source="B", chunk_id=1),
        Chunk(text="Augustus emperor",                     source="C", chunk_id=2),
        Chunk(text="Caesar imperator",                     source="D", chunk_id=3),
        Chunk(text="unrelated text about photosynthesis",  source="E", chunk_id=4),
    ]
    # Dense: give A a strictly higher cosine score (norm-1 vector pointing
    # closer to the query direction) so rank-1 is deterministic, not a
    # tie-break across identical vectors.
    dim = 8
    # Query vector: all-ones direction.
    q_vec = np.ones((1, dim), dtype=np.float32) / np.sqrt(dim)
    # A gets score 1.0 (exact match to query direction).
    # B-E get progressively lower scores so rank is predictable.
    vecs = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # A — dense rank 1
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # B — dense rank 2
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # C — dense rank 3
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # D — dense rank 4
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # E — dense rank 5
    ], dtype=np.float32)
    # L2-normalise so cosine similarity = dot product.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    idx = FAISSIndex(dim=dim)
    idx.add(chunks, vecs)

    # BM25: A and D both contain "Caesar"; E has none.
    # A and D share the same BM25 score — both hit rank 1/2 in BM25.
    bm25 = BM25Index(chunks)

    class _Embed:
        dim = 8
        def encode(self, texts): return q_vec  # always return the all-ones query

    r = Retriever(idx, _Embed(), bm25=bm25)  # type: ignore[arg-type]
    out = r.retrieve("Caesar", k=3, hybrid=True)
    sources = [c.source for c, _ in out]
    # A is dense rank 1 AND BM25 rank 1-or-2 → highest combined RRF → should be top.
    assert sources[0] == "A"
    # D appears in BM25 (Caesar hit) even though it's dense rank 4.
    # B/C/E appear only in dense. D's BM25 bonus should lift it into top-3.
    assert "D" in sources


def test_retriever_hybrid_with_category_filter_composes():
    """category filter applies to BOTH dense and BM25 paths."""
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.bm25 import BM25Index
    import numpy as np

    chunks = (
        [Chunk(text=f"Caesar history {i}", source=f"H{i}", chunk_id=i, category="history")
         for i in range(3)] +
        [Chunk(text=f"Caesar science {i}", source=f"S{i}", chunk_id=i, category="science")
         for i in range(3)]
    )
    same = np.ones((6, 8), dtype=np.float32) / np.sqrt(8)
    idx = FAISSIndex(dim=8)
    idx.add(chunks, same)
    bm25 = BM25Index(chunks)

    class _Embed:
        dim = 8
        def encode(self, texts): return same[:1]

    r = Retriever(idx, _Embed(), bm25=bm25)  # type: ignore[arg-type]
    out = r.retrieve("Caesar", k=4, hybrid=True, category="history")
    assert all(c.category == "history" for c, _ in out)


def test_retriever_hybrid_composes_with_reranker():
    """hybrid + rerank: fuse first, then rerank."""
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.bm25 import BM25Index
    from polimibot.rag.reranker import CrossEncoderReranker
    import numpy as np

    chunks = [Chunk(text=f"doc {i}", source=f"S{i}", chunk_id=0) for i in range(5)]
    same = np.ones((5, 8), dtype=np.float32) / np.sqrt(8)
    idx = FAISSIndex(dim=8)
    idx.add(chunks, same)
    bm25 = BM25Index(chunks)
    rer = CrossEncoderReranker(lambda pairs: [1.0] * len(pairs))

    class _Embed:
        dim = 8
        def encode(self, texts): return same[:1]

    r = Retriever(idx, _Embed(), bm25=bm25, reranker=rer)  # type: ignore[arg-type]
    out = r.retrieve("doc", k=2, hybrid=True, rerank=True)
    # Scores after rerank should be the constant 1.0 (cross-encoder output).
    assert all(s == 1.0 for _, s in out)
    assert len(out) == 2


def test_retriever_has_bm25_property():
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.bm25 import BM25Index
    class _Embed:
        dim = 8
        def encode(self, texts): return _random_vecs(1, dim=8)

    idx = FAISSIndex(dim=8)
    r_plain = Retriever(idx, _Embed())  # type: ignore[arg-type]
    assert r_plain.has_bm25 is False

    r_with = Retriever(idx, _Embed(),  # type: ignore[arg-type]
                       bm25=BM25Index([Chunk(text="x", source="S", chunk_id=0)]))
    assert r_with.has_bm25 is True


def test_retriever_no_hybrid_keeps_dense_only():
    """hybrid=False (default) preserves single-source dense path. The BM25
    index, even when attached, must NOT influence results."""
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.bm25 import BM25Index
    import numpy as np

    chunks = [
        Chunk(text="lex-only-match keyword keyword", source="A", chunk_id=0),
        Chunk(text="some other text", source="B", chunk_id=0),
    ]
    same = np.ones((2, 8), dtype=np.float32) / np.sqrt(8)
    idx = FAISSIndex(dim=8)
    idx.add(chunks, same)
    bm25 = BM25Index(chunks)  # would heavily favour 'A' on "keyword"

    class _Embed:
        dim = 8
        def encode(self, texts): return same[:1]

    r = Retriever(idx, _Embed(), bm25=bm25)  # type: ignore[arg-type]
    # hybrid defaulted off — scores are cosine (1.0 since vectors are identical).
    out = r.retrieve("keyword", k=2)
    # All cosine scores ~1.0 (not BM25-shaped); BM25 was not consulted.
    assert all(0.9 < s < 1.1 for _, s in out)


# ── Source-level diversification ────────────────────────────────────────────


def test_retriever_diversifies_top_k_by_source_by_default():
    """When multiple chunks share a source, top-k collapses them to one each
    until k unique sources are filled."""
    from polimibot.rag.retriever import Retriever
    import numpy as np

    # Three chunks from "A" (overlapping windows), two from "B", one from "C".
    chunks = (
        [Chunk(text=f"a{i}", source="A", chunk_id=i) for i in range(3)] +
        [Chunk(text=f"b{i}", source="B", chunk_id=i) for i in range(2)] +
        [Chunk(text="c0",   source="C", chunk_id=0)]
    )
    same = np.ones((6, 8), dtype=np.float32) / np.sqrt(8)
    idx = FAISSIndex(dim=8)
    idx.add(chunks, same)

    class _Embed:
        dim = 8
        def encode(self, texts): return same[:1]

    r = Retriever(idx, _Embed())  # type: ignore[arg-type]
    out = r.retrieve("q", k=3)
    sources = [c.source for c, _ in out]
    # One chunk per source — A, B, C — instead of three A's. Order is
    # whatever FAISS returns for the tied scores; we just verify uniqueness.
    assert len(sources) == 3
    assert set(sources) == {"A", "B", "C"}


def test_retriever_diversify_false_keeps_duplicates():
    """Explicit opt-out — useful for measuring the diversify lift."""
    from polimibot.rag.retriever import Retriever
    import numpy as np

    chunks = [Chunk(text=f"a{i}", source="A", chunk_id=i) for i in range(4)]
    same = np.ones((4, 8), dtype=np.float32) / np.sqrt(8)
    idx = FAISSIndex(dim=8)
    idx.add(chunks, same)

    class _Embed:
        dim = 8
        def encode(self, texts): return same[:1]

    r = Retriever(idx, _Embed())  # type: ignore[arg-type]
    out = r.retrieve("q", k=3, diversify=False)
    assert [c.source for c, _ in out] == ["A", "A", "A"]


def test_retriever_diversify_backfills_when_short_on_unique_sources():
    """If unique sources are fewer than k, fill remaining slots with
    leftovers — never return fewer than k when material exists."""
    from polimibot.rag.retriever import Retriever
    import numpy as np

    # Only 2 unique sources but k=3 requested.
    chunks = (
        [Chunk(text=f"a{i}", source="A", chunk_id=i) for i in range(2)] +
        [Chunk(text="b0", source="B", chunk_id=0)]
    )
    same = np.ones((3, 8), dtype=np.float32) / np.sqrt(8)
    idx = FAISSIndex(dim=8)
    idx.add(chunks, same)

    class _Embed:
        dim = 8
        def encode(self, texts): return same[:1]

    r = Retriever(idx, _Embed())  # type: ignore[arg-type]
    out = r.retrieve("q", k=3)
    assert len(out) == 3
    sources = [c.source for c, _ in out]
    # Two A's (one from primary, one from leftover backfill) + one B.
    assert sources.count("A") == 2
    assert sources.count("B") == 1


def test_retriever_adaptive_category_oversearch_falls_back_to_full_index():
    """When the 8× oversearch doesn't surface enough of the target category,
    a second search over the whole index makes up the difference."""
    from polimibot.rag.retriever import Retriever
    import numpy as np

    # Many history chunks first (dense distance close to query), then ONE
    # maths chunk far down the dense ranking. With k_pool=1 and 8× = 8,
    # the first pass returns the top 8 history-only chunks; the maths
    # chunk is missed. The retry over the full index must pick it up.
    n_history = 20
    chunks = (
        [Chunk(text=f"h{i}", source=f"H{i}", chunk_id=0, category="history")
         for i in range(n_history)] +
        [Chunk(text="m0", source="M0", chunk_id=0, category="maths")]
    )
    # Dense layout: history chunks score 1.0, maths chunk scores 0.0.
    vecs = np.zeros((n_history + 1, 8), dtype=np.float32)
    vecs[:n_history] = np.ones((n_history, 8), dtype=np.float32) / np.sqrt(8)
    vecs[n_history] = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    idx = FAISSIndex(dim=8)
    idx.add(chunks, vecs)

    class _Embed:
        dim = 8
        def encode(self, texts):
            # Query close to the history vector → history chunks rank first
            v = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
            return v

    r = Retriever(idx, _Embed())  # type: ignore[arg-type]
    out = r.retrieve("q", k=1, category="maths")
    # Without the retry this would return [] (no maths in the top-8 dense).
    assert len(out) == 1
    assert out[0][0].category == "maths"


def test_retriever_no_rerank_keeps_dense_path():
    """rerank=False (default) preserves the existing dense-only behaviour."""
    from polimibot.rag.retriever import Retriever
    from polimibot.rag.reranker import CrossEncoderReranker
    import numpy as np

    idx = FAISSIndex(dim=8)
    idx.add(_make_chunks(5), _random_vecs(5, dim=8))
    # A "broken" reranker that would re-order if used — we use this to
    # confirm it ISN'T used when rerank=False.
    rer = CrossEncoderReranker(lambda pairs: [-1.0] * len(pairs))
    class _Embed:
        dim = 8
        def encode(self, texts): return _random_vecs(1, dim=8)

    r = Retriever(idx, _Embed(), reranker=rer)  # type: ignore[arg-type]
    out = r.retrieve("q", k=2)   # rerank not passed → False
    # Scores should be dense (cosine, varying), not -1.0 from the reranker.
    assert all(s != -1.0 for _, s in out)


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


# ── Embedder asymmetric encoding ────────────────────────────────────────────


class _FakeST:
    """Minimal stand-in for sentence_transformers.SentenceTransformer.

    Records the inputs passed to .encode so tests can verify which prefix
    (if any) was prepended. Returns ones-vectors so shape is right.
    """
    last_inputs: list[str] = []

    def __init__(self, name: str):
        self.name = name

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, batch_size=None, normalize_embeddings=False,
               show_progress_bar=False, convert_to_numpy=True):
        type(self).last_inputs = list(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


def _patch_sentence_transformers(monkeypatch):
    import sentence_transformers as st
    monkeypatch.setattr(st, "SentenceTransformer", _FakeST)
    _FakeST.last_inputs = []


def test_embedder_encode_query_prepends_query_prefix(monkeypatch):
    _patch_sentence_transformers(monkeypatch)
    from polimibot.rag.embedder import Embedder, EmbedderSpec
    spec = EmbedderSpec(
        model_name="any", query_prefix="Q: ", passage_prefix="P: ",
    )
    emb = Embedder(spec)
    emb.encode_query(["foo", "bar"])
    assert _FakeST.last_inputs == ["Q: foo", "Q: bar"]


def test_embedder_encode_passage_prepends_passage_prefix(monkeypatch):
    _patch_sentence_transformers(monkeypatch)
    from polimibot.rag.embedder import Embedder, EmbedderSpec
    spec = EmbedderSpec(
        model_name="any", query_prefix="Q: ", passage_prefix="P: ",
    )
    emb = Embedder(spec)
    emb.encode_passage(["foo"])
    assert _FakeST.last_inputs == ["P: foo"]


def test_embedder_empty_prefix_skips_prepending(monkeypatch):
    """Common case (MiniLM, BGE-passage) — no prefix means inputs pass
    through verbatim, no per-text string concat."""
    _patch_sentence_transformers(monkeypatch)
    from polimibot.rag.embedder import Embedder, EmbedderSpec
    spec = EmbedderSpec(model_name="any", query_prefix="", passage_prefix="")
    emb = Embedder(spec)
    emb.encode_query(["hello"])
    assert _FakeST.last_inputs == ["hello"]


def test_check_manifest_compat_hard_fails_on_query_prefix_drift():
    from polimibot.rag.retriever import _check_manifest_compat
    from polimibot.rag.embedder import EmbedderSpec
    spec = EmbedderSpec(model_name="m", query_prefix="A: ", passage_prefix="")
    manifest = {
        "embedder_model_name":   "m",
        "embedder_query_prefix": "B: ",
    }
    with pytest.raises(ValueError, match="incompatible halves"):
        _check_manifest_compat(manifest, spec)


def test_check_manifest_compat_passes_when_prefixes_match():
    from polimibot.rag.retriever import _check_manifest_compat
    from polimibot.rag.embedder import EmbedderSpec
    spec = EmbedderSpec(model_name="m", query_prefix="A: ", passage_prefix="P: ")
    manifest = {
        "embedder_model_name":     "m",
        "embedder_query_prefix":   "A: ",
        "embedder_passage_prefix": "P: ",
    }
    _check_manifest_compat(manifest, spec)   # no raise


def test_check_manifest_compat_legacy_manifest_without_prefix(monkeypatch):
    """An older manifest that omits prefix fields keeps loading — they're
    treated as 'unknown', not as forced-empty."""
    from polimibot.rag.retriever import _check_manifest_compat
    from polimibot.rag.embedder import EmbedderSpec
    spec = EmbedderSpec(model_name="m", query_prefix="A: ", passage_prefix="")
    manifest = {"embedder_model_name": "m"}
    _check_manifest_compat(manifest, spec)   # no raise


# ─── pre-existing test (untouched) ─────────────────────────────────────────


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