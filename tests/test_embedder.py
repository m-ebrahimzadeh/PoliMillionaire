"""EmbedderSpec prefix auto-derivation.

A model_name change without a matching prefix change silently corrupts
cosine scores on asymmetric models (BGE, E5). EmbedderSpec eliminates
that footgun by deriving prefixes from model_name when they're left
as None — these tests pin that behaviour.
"""
from polimibot.rag.embedder import EmbedderSpec, _prefixes_for_model


def test_default_spec_uses_bge_prefix():
    spec = EmbedderSpec()
    assert spec.model_name == "BAAI/bge-small-en-v1.5"
    assert "Represent this sentence" in spec.query_prefix
    assert spec.passage_prefix == ""


def test_minilm_auto_derives_empty_prefixes():
    spec = EmbedderSpec(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert spec.query_prefix == ""
    assert spec.passage_prefix == ""


def test_e5_auto_derives_query_passage_prefixes():
    spec = EmbedderSpec(model_name="intfloat/e5-small-v2")
    assert spec.query_prefix == "query: "
    assert spec.passage_prefix == "passage: "


def test_bge_m3_uses_no_query_instruction():
    """BGE-M3 dense retrieval is trained without a query instruction; deriving
    the v1.5 instruction prefix for it silently degrades cosine scores. M3 must
    win over the generic 'bge' branch and get empty prefixes on both sides."""
    spec = EmbedderSpec(model_name="BAAI/bge-m3")
    assert spec.query_prefix == ""
    assert spec.passage_prefix == ""
    # And the v1.5 line still gets its instruction prefix (no regression).
    assert "Represent this sentence" in EmbedderSpec(
        model_name="BAAI/bge-small-en-v1.5"
    ).query_prefix


def test_explicit_prefix_overrides_auto_derivation():
    spec = EmbedderSpec(
        model_name="BAAI/bge-small-en-v1.5",
        query_prefix="custom: ",
    )
    assert spec.query_prefix == "custom: "
    # Other prefix still auto-derived (passage_prefix=None → "")
    assert spec.passage_prefix == ""


def test_empty_string_prefix_is_not_overridden():
    """Explicit empty string ≠ None — caller intentionally wants no prefix
    even on an asymmetric model. Must not be replaced by auto-derivation."""
    spec = EmbedderSpec(
        model_name="BAAI/bge-small-en-v1.5",
        query_prefix="",
        passage_prefix="",
    )
    assert spec.query_prefix == ""
    assert spec.passage_prefix == ""


def test_prefixes_for_model_unknown_family_returns_empty():
    q, p = _prefixes_for_model("some-random-model-name")
    assert q == ""
    assert p == ""
