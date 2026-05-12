"""BM25Index — pure-Python sparse retrieval. No FAISS, no torch."""
from __future__ import annotations

import json

import pytest

from polimibot.rag.bm25 import BM25Index, BM25Spec, tokenize
from polimibot.rag.chunker import Chunk


def _c(idx: int, text: str, category: str | None = None) -> Chunk:
    return Chunk(text=text, source=f"S{idx}", chunk_id=idx, category=category)


# ── Tokeniser ────────────────────────────────────────────────────────────────


def test_tokenize_lowercases():
    # "the" is a stopword → removed by default; "crossed" and "rubicon" are not.
    assert tokenize("Caesar Crossed THE Rubicon") == ["caesar", "crossed", "rubicon"]


def test_tokenize_lowercases_no_stopwords_flag():
    # With drop_stopwords=False the original tokens come through intact.
    assert tokenize("Caesar Crossed THE Rubicon", drop_stopwords=False) == [
        "caesar", "crossed", "the", "rubicon"
    ]


def test_tokenize_drops_punctuation():
    # "it" and "s" come from "It's"; "it" is a stopword → removed.
    # "hello", "world", "s", "49", "bc" survive; "it" does not.
    result = tokenize("Hello, world! It's 49 BC.")
    assert "hello" in result
    assert "world" in result
    assert "s" in result
    assert "49" in result
    assert "bc" in result
    assert "it" not in result  # stopword


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   \n  ") == []


def test_tokenize_handles_unicode_word_chars():
    # Accented characters should remain — Wikipedia has plenty.
    out = tokenize("Niño's CAFÉ")
    assert "niño" in out
    assert "café" in out


# ── Construction + basic search ──────────────────────────────────────────────


def _three_doc_corpus():
    return [
        _c(0, "Caesar crossed the Rubicon in 49 BC."),
        _c(1, "Pompey was Caesar's rival in the late Republic."),
        _c(2, "DNA contains the genetic code."),
    ]


def test_bm25_search_returns_relevant_doc_first():
    idx = BM25Index(_three_doc_corpus())
    hits = idx.search("Caesar Rubicon", k=2)
    assert len(hits) == 2
    # Doc 0 has BOTH query tokens; doc 1 has only "caesar"; doc 2 has neither.
    assert hits[0][0].chunk_id == 0
    assert hits[1][0].chunk_id == 1


def test_bm25_rare_tokens_outweigh_common_ones():
    """IDF should down-weight 'the' and up-weight 'Rubicon'."""
    idx = BM25Index(_three_doc_corpus())
    # 'the' appears in doc 0 and doc 2 (corpus DF=2/3 → low IDF).
    # 'Rubicon' appears only in doc 0 (DF=1/3 → high IDF).
    hits = idx.search("the Rubicon", k=3)
    assert hits[0][0].chunk_id == 0   # the Rubicon-only doc wins despite 'the' tie


def test_bm25_returns_empty_when_no_terms_match():
    idx = BM25Index(_three_doc_corpus())
    hits = idx.search("xyzzyxyzzy nonexistentword", k=3)
    assert hits == []


def test_bm25_returns_fewer_than_k_when_pool_too_small():
    idx = BM25Index([_c(0, "Caesar crossed the Rubicon.")])
    hits = idx.search("Caesar", k=10)
    assert len(hits) == 1


# ── Category filter ──────────────────────────────────────────────────────────


def test_bm25_category_filter_drops_off_category_chunks():
    corpus = [
        _c(0, "Caesar Rubicon 49 BC", category="history"),
        _c(1, "Caesar Rubicon science version", category="science"),
        _c(2, "Caesar Rubicon math fragment", category="maths"),
    ]
    idx = BM25Index(corpus)
    hits = idx.search("Caesar Rubicon", k=3, category="history")
    assert all(c.category == "history" for c, _ in hits)
    assert len(hits) == 1


def test_bm25_no_category_returns_all_categories():
    corpus = [
        _c(0, "Caesar history", category="history"),
        _c(1, "Caesar science", category="science"),
    ]
    idx = BM25Index(corpus)
    hits = idx.search("Caesar", k=2)
    cats = {c.category for c, _ in hits}
    assert cats == {"history", "science"}


# ── Persistence ──────────────────────────────────────────────────────────────


def test_bm25_save_load_roundtrip(tmp_path):
    corpus = _three_doc_corpus()
    idx = BM25Index(corpus, spec=BM25Spec(k1=2.0, b=0.5))
    idx.save(tmp_path / "stem")

    loaded = BM25Index.load(tmp_path / "stem")
    assert loaded.n_chunks == 3
    assert loaded.spec.k1 == 2.0
    assert loaded.spec.b == 0.5

    # Same query → same ranking.
    a = [c.chunk_id for c, _ in idx.search("Caesar Rubicon", k=3)]
    b = [c.chunk_id for c, _ in loaded.search("Caesar Rubicon", k=3)]
    assert a == b


def test_bm25_save_writes_header_first(tmp_path):
    """Header lets readers detect format version + spec without reading every line."""
    from polimibot.rag.bm25 import BM25_VERSION
    idx = BM25Index(_three_doc_corpus())
    idx.save(tmp_path / "stem")
    first = (tmp_path / "stem.bm25.jsonl").open(encoding="utf-8").readline()
    rec = json.loads(first)
    assert rec.get("kind") == "bm25_header"
    assert rec.get("n_docs") == 3
    assert rec.get("version") == BM25_VERSION  # v2: positional postings + stopwords


def test_bm25_load_preserves_category(tmp_path):
    corpus = [
        _c(0, "x", category="history"),
        _c(1, "y", category=None),
    ]
    idx = BM25Index(corpus)
    idx.save(tmp_path / "stem")
    loaded = BM25Index.load(tmp_path / "stem")
    cats = [c.category for c in loaded._chunks]
    assert cats == ["history", None]


# ── Length normalisation ─────────────────────────────────────────────────────


def test_bm25_length_normalisation_favours_concise_match():
    """A short doc that's all about Caesar should beat a long padded one."""
    short = _c(0, "Caesar")                                  # 1 token, all signal
    long_padded = _c(
        1,
        "Caesar " + ("filler word " * 100),                  # 1 signal + 200 noise
    )
    idx = BM25Index([short, long_padded])
    hits = idx.search("Caesar", k=2)
    assert hits[0][0].chunk_id == 0


# ── Spec defaults ────────────────────────────────────────────────────────────


def test_bm25_default_spec_matches_okapi_standard():
    """Default k1=1.5, b=0.75 — standard BM25Okapi. A change to defaults
    should be a conscious commit."""
    spec = BM25Spec()
    assert spec.k1 == 1.5
    assert spec.b == 0.75
