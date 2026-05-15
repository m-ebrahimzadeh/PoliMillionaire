"""RAGStrategy tests — no GPU, no FAISS, no network required."""
from __future__ import annotations

import numpy as np
import pytest

from polimibot.models.mock import MockLLM
from polimibot.prompts.templates import build_messages_with_context, PromptStyle
from polimibot.rag.chunker import Chunk
from polimibot.rag.retriever import Retriever
from polimibot.strategies.base import StrategyInput
from polimibot.strategies.rag_strategy import RAGStrategy, _build_query, _format_context


# ── Helpers ───────────────────────────────────────────────────────────────────

class MockRetriever:
    """Returns canned passages. No FAISS, no embedder.

    ``has_reranker``/``has_bm25`` default to False so the corresponding
    strategy flags raise at construction; pass the flags True for tests
    that exercise the integration.
    """
    # Sentinel to distinguish "never called" from "called with None".
    _DEFAULT_RERANK_OVERSEARCH = 5   # mirrors Retriever constant

    def __init__(
        self,
        passages: list[tuple[Chunk, float]],
        *,
        has_reranker: bool = False,
        has_bm25: bool = False,
    ) -> None:
        self._passages = passages
        self.has_reranker = has_reranker
        self.has_bm25 = has_bm25
        self.last_query: str = ""
        self.last_category = None
        self.last_rerank: bool = False
        self.last_rerank_oversearch = None
        self.last_hybrid: bool = False
        self.queries_seen: list[str] = []
        self.n_chunks = len(passages)
        # Tracks calls to rerank_pool (fuse-then-rerank path).
        self.rerank_pool_calls: list[dict] = []

    def retrieve(
        self,
        query: str,
        k: int = 3,
        *,
        category=None,
        rerank: bool = False,
        rerank_oversearch=None,
        hybrid: bool = False,
        diversify: bool = True,
    ) -> list[tuple[Chunk, float]]:
        self.last_query = query
        self.queries_seen.append(query)
        self.last_category = category
        self.last_rerank = rerank
        self.last_rerank_oversearch = rerank_oversearch
        self.last_hybrid = hybrid
        return self._passages[:k]

    def rerank_pool(
        self,
        query: str,
        pool: list[tuple[Chunk, float]],
        *,
        k: int,
    ) -> list[tuple[Chunk, float]]:
        """Mock rerank_pool: records the call and returns pool[:k]."""
        self.rerank_pool_calls.append({"query": query, "pool_size": len(pool), "k": k})
        return pool[:k]


def _inp(gold: str = "B") -> StrategyInput:
    return StrategyInput(
        question=f"Who crossed the Rubicon? <gold>{gold}</gold>",
        options=("Pompey", "Caesar", "Augustus", "Cicero"),
        level=3,
    )


def _passages() -> list[tuple[Chunk, float]]:
    return [
        (Chunk(text="Caesar crossed the Rubicon in 49 BC.", source="Julius Caesar", chunk_id=0), 0.91),
        (Chunk(text="The Roman Republic began in 509 BC.", source="Roman Republic", chunk_id=0), 0.72),
    ]


# ── RAGStrategy core ──────────────────────────────────────────────────────────

def test_rag_picks_gold_answer():
    strategy = RAGStrategy(MockLLM(correctness=1.0), MockRetriever(_passages()), k=2)
    out = strategy.answer(_inp("B"))
    assert out.chosen_index == 1   # "B" = Caesar = index 1


def test_rag_query_includes_options():
    """In multi-query mode (the new default), each option appears in one
    of the per-option queries even though no single query carries all options."""
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2)  # use_multi_query=True by default
    strategy.answer(_inp())
    all_queries = " ".join(retriever.queries_seen)
    assert "Caesar" in all_queries
    assert "Pompey" in all_queries


def test_rag_extras_populated():
    out = RAGStrategy(MockLLM(), MockRetriever(_passages()), k=2).answer(_inp())
    assert "probs" in out.extras
    assert out.extras["n_passages"] == 2
    assert out.extras["top_source"] == "Julius Caesar"


def test_rag_rationale_contains_context():
    """Context is stored in rationale — visible in run logs."""
    out = RAGStrategy(MockLLM(), MockRetriever(_passages()), k=2).answer(_inp())
    assert "Julius Caesar" in out.rationale


def test_rag_empty_retrieval_degrades_gracefully():
    """No passages → falls back to plain prompt, still returns a valid answer."""
    strategy = RAGStrategy(MockLLM(correctness=1.0), MockRetriever([]), k=3)
    out = strategy.answer(_inp("A"))
    assert 0 <= out.chosen_index < 4


# ── build_messages_with_context ───────────────────────────────────────────────

def test_context_appears_in_user_turn():
    msgs = build_messages_with_context(
        "Who was Caesar?", ("A", "B", "C", "D"),
        context="[1] Julius Caesar\nRoman general.",
    )
    user_content = msgs[-1]["content"]
    assert "[1] Julius Caesar" in user_content
    assert "Who was Caesar?" in user_content


def test_context_appears_AFTER_question():
    """Question + options must come before retrieved context — chat-tuned
    models attend most strongly to the most recent tokens, so retrieval
    shouldn't crowd the actual question out of the attention window."""
    msgs = build_messages_with_context(
        "Who was Caesar?", ("A", "B", "C", "D"),
        context="[1] Julius Caesar\nRoman general.",
    )
    user_content = msgs[-1]["content"]
    q_pos = user_content.find("Question:")
    c_pos = user_content.find("Reference material")
    assert q_pos != -1 and c_pos != -1
    assert q_pos < c_pos


def test_context_framed_as_optional_evidence():
    """Framing matters — 'Context (from Wikipedia)' implies authority;
    'Reference material (may or may not be relevant)' invites scepticism
    so off-topic retrievals don't pull the model toward fabrication."""
    msgs = build_messages_with_context(
        "Q?", ("A", "B", "C", "D"),
        context="[1] something",
    )
    user_content = msgs[-1]["content"]
    assert "may or may not be relevant" in user_content.lower()


def test_empty_context_falls_back_to_plain_messages():
    from polimibot.prompts.templates import build_messages
    plain = build_messages("Q?", ("A", "B", "C", "D"))
    rag   = build_messages_with_context("Q?", ("A", "B", "C", "D"), context="")
    assert plain == rag


def test_wrong_option_count_raises():
    with pytest.raises(ValueError):
        build_messages_with_context("Q?", ("A", "B"), context="ctx")


# ── _format_context ───────────────────────────────────────────────────────────

def test_format_context_numbers_passages():
    ctx = _format_context(_passages())
    assert "[1]" in ctx
    assert "[2]" in ctx


def test_format_context_respects_char_budget():
    big_chunk = Chunk(text="x" * 3000, source="Big Article", chunk_id=0)
    ctx = _format_context([(big_chunk, 0.9)] * 10, max_total_chars=500)
    assert len(ctx) <= 600   # small tolerance for numbering overhead


def test_format_context_passage_char_cap_per_chunk():
    """max_passage_chars trims each chunk individually, before joining."""
    long_chunk = Chunk(text="abcdefghij" * 200, source="Long", chunk_id=0)  # 2000 chars
    ctx = _format_context([(long_chunk, 0.9)], max_passage_chars=50, max_total_chars=10000)
    # Should contain ~50 chars of body + a small header overhead.
    body = ctx.split("\n", 1)[1] if "\n" in ctx else ctx
    assert len(body) <= 60


def test_format_context_includes_chunk_id_in_header():
    """Citation discipline: chunk_id must appear in the passage header so
    the LLM can reference '[Passage 1]' in reasoning."""
    chunk = Chunk(text="Caesar crossed the Rubicon.", source="Julius Caesar",
                  chunk_id=7, category=None)
    ctx = _format_context([(chunk, 0.9)])
    assert "chunk 7" in ctx


def test_format_context_includes_category_when_available():
    """When category is set, it should appear in the header for disambiguation."""
    chunk = Chunk(text="Some fact.", source="Article", chunk_id=0, category="history")
    ctx = _format_context([(chunk, 0.9)])
    assert "history" in ctx


def test_format_context_no_category_suffix_when_none():
    """When category is None, no extra comma-suffix in the header."""
    chunk = Chunk(text="Some fact.", source="Article", chunk_id=0, category=None)
    ctx = _format_context([(chunk, 0.9)])
    # Header should be "[1] Article (chunk 0)" without trailing comma.
    header_line = ctx.split("\n")[0]
    assert not header_line.endswith(",")
    assert "None" not in header_line


def test_format_context_truncates_at_sentence_boundary():
    """Hard-truncation mid-word is replaced with sentence-boundary truncation."""
    from polimibot.strategies.rag_strategy import _truncate_to_sentence
    text = "First sentence. Second sentence. Third sentence."
    # Budget fits first two sentences (32 chars) but not the third.
    result = _truncate_to_sentence(text, 35)
    assert result.endswith(".")
    assert "Third" not in result


def test_truncate_to_sentence_no_truncation_when_fits():
    from polimibot.strategies.rag_strategy import _truncate_to_sentence
    text = "Short text."
    assert _truncate_to_sentence(text, 1000) == text


def test_truncate_to_sentence_falls_back_to_word_boundary():
    """When no sentence-boundary fits, fall back to word boundary."""
    from polimibot.strategies.rag_strategy import _truncate_to_sentence
    text = "word1 word2 word3 word4"
    result = _truncate_to_sentence(text, 12)  # fits "word1 word2"
    assert not result.endswith(" ")
    assert "word3" not in result


# ── Low-score gate ────────────────────────────────────────────────────────────

def test_rag_min_score_gate_drops_context_below_threshold():
    """When the top retrieval score is below min_score, RAG degrades to
    plain (no-context) prompting. The model is not fed irrelevant evidence.
    Uses use_multi_query=False so score units are cosine (predictable)."""
    low_score_passages = [
        (Chunk(text="off-topic", source="Wrong", chunk_id=0), 0.10),
    ]
    strategy = RAGStrategy(
        MockLLM(correctness=1.0),
        MockRetriever(low_score_passages),
        k=1,
        min_score=0.30,
        use_multi_query=False,
    )
    out = strategy.answer(_inp("B"))
    assert out.extras["gated_by_min_score"] is True
    assert out.extras["top_score"] == pytest.approx(0.10)
    # Rationale (the context block) should be empty when gated.
    assert out.rationale == ""


def test_rag_min_score_gate_keeps_context_above_threshold():
    high_score_passages = [
        (Chunk(text="Caesar crossed the Rubicon in 49 BC.", source="Julius Caesar", chunk_id=0), 0.85),
    ]
    strategy = RAGStrategy(
        MockLLM(correctness=1.0),
        MockRetriever(high_score_passages),
        k=1,
        min_score=0.30,
        use_multi_query=False,
    )
    out = strategy.answer(_inp("B"))
    assert out.extras["gated_by_min_score"] is False
    assert "Julius Caesar" in out.rationale


def test_rag_min_score_default_none_never_gates():
    """Default behaviour: no gate, never drop context — backwards-compatible."""
    low_score_passages = [
        (Chunk(text="off-topic", source="Wrong", chunk_id=0), 0.05),
    ]
    strategy = RAGStrategy(MockLLM(), MockRetriever(low_score_passages), k=1)
    out = strategy.answer(_inp())
    assert out.extras["gated_by_min_score"] is False
    assert out.extras["min_score_threshold"] is None
    assert out.rationale != ""


def test_rag_name_includes_min_score_when_set():
    strategy = RAGStrategy(
        MockLLM(), MockRetriever(_passages()), k=2, min_score=0.30,
    )
    assert "min_score=0.3" in strategy.name


def test_rag_path_aware_gate_uses_rrf_threshold_on_hybrid():
    """On the hybrid path, min_score_rrf is used, not min_score.
    A low RRF score (<min_score_rrf) should gate; a cosine-range value
    passed as min_score should NOT gate the hybrid path."""
    rrf_passages = [
        (Chunk(text="some text", source="S", chunk_id=0), 0.005),  # low RRF score
    ]
    retriever = MockRetriever(rrf_passages, has_bm25=True)
    # min_score=0.30 should NOT fire on the hybrid (RRF) path.
    # min_score_rrf=0.01 > 0.005 → should gate.
    strategy = RAGStrategy(
        MockLLM(correctness=1.0), retriever, k=1,
        use_hybrid=True,
        use_multi_query=False,
        min_score=0.30,       # dense threshold — must NOT apply here
        min_score_rrf=0.01,   # RRF threshold — MUST apply here
    )
    out = strategy.answer(_inp("A"))
    assert out.extras["gated_by_min_score"] is True
    assert out.extras["min_score_threshold"] == pytest.approx(0.01)


def test_rag_path_aware_gate_dense_threshold_ignored_on_hybrid():
    """min_score alone set, no min_score_rrf: hybrid path never gates
    (even if the dense threshold would gate a cosine score)."""
    rrf_passages = [
        (Chunk(text="some text", source="S", chunk_id=0), 0.005),
    ]
    retriever = MockRetriever(rrf_passages, has_bm25=True)
    strategy = RAGStrategy(
        MockLLM(correctness=1.0), retriever, k=1,
        use_hybrid=True,
        use_multi_query=False,
        min_score=0.30,   # would gate a dense score of 0.005, but not RRF
    )
    out = strategy.answer(_inp("A"))
    # min_score_rrf=None → no gate on hybrid path
    assert out.extras["gated_by_min_score"] is False
    assert out.extras["min_score_threshold"] is None


def test_rag_path_aware_gate_rerank_threshold():
    """On the rerank path, min_score_rerank is consulted, not min_score."""
    passages = [
        (Chunk(text="some text", source="S", chunk_id=0), -1.5),  # negative CE logit
    ]
    retriever = MockRetriever(passages, has_reranker=True)
    strategy = RAGStrategy(
        MockLLM(correctness=1.0), retriever, k=1,
        use_reranker=True,
        use_multi_query=False,
        min_score=0.30,           # dense threshold — must NOT apply
        min_score_rerank=-1.0,    # CE threshold — -1.5 < -1.0 → gates
    )
    out = strategy.answer(_inp("A"))
    assert out.extras["gated_by_min_score"] is True
    assert out.extras["min_score_threshold"] == pytest.approx(-1.0)


def test_rag_extras_carries_full_passage_triples():
    """Run logs need full top-k for recall@k post-hoc analysis."""
    strategy = RAGStrategy(MockLLM(), MockRetriever(_passages()), k=2)
    out = strategy.answer(_inp())
    assert "passages" in out.extras
    triples = out.extras["passages"]
    assert len(triples) == 2
    assert triples[0]["source"] == "Julius Caesar"
    assert "chunk_id" in triples[0] and "score" in triples[0]
    assert "query" in out.extras   # the constructed retrieval query


# ── Category filter passthrough ─────────────────────────────────────────────

def _inp_with_cat(gold="B", category=None) -> StrategyInput:
    from polimibot.config import Category as Cat
    return StrategyInput(
        question=f"Who crossed the Rubicon? <gold>{gold}</gold>",
        options=("Pompey", "Caesar", "Augustus", "Cicero"),
        level=3,
        category=category if category else Cat.HISTORY,
    )


def test_rag_passes_category_to_retriever_when_filter_on():
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_category_filter=True)
    strategy.answer(_inp_with_cat())
    assert retriever.last_category == "history"


def test_rag_skips_category_when_filter_off():
    """use_category_filter=False ablates the filter — retriever sees None."""
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_category_filter=False)
    strategy.answer(_inp_with_cat())
    assert retriever.last_category is None


def test_rag_extras_records_category_filter():
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_category_filter=True)
    out = strategy.answer(_inp_with_cat())
    assert out.extras["category_filter"] == "history"


def test_rag_name_includes_no_cat_filter_when_disabled():
    strategy = RAGStrategy(MockLLM(), MockRetriever(_passages()), k=2,
                            use_category_filter=False)
    assert "no_cat_filter" in strategy.name


# ── Style/scoring contract (#8) ─────────────────────────────────────────────

def test_rag_rejects_cot_with_score_options():
    """CoT + score_options is contradictory: score_options reads the first
    predicted token (start of reasoning), not the answer letter."""
    with pytest.raises(ValueError, match="requires free generation"):
        RAGStrategy(
            MockLLM(), MockRetriever(_passages()),
            style=PromptStyle.ZERO_SHOT_COT,
            use_score_options=True,
        )


def test_rag_rejects_elimination_with_score_options():
    with pytest.raises(ValueError, match="requires free generation"):
        RAGStrategy(
            MockLLM(), MockRetriever(_passages()),
            style=PromptStyle.ELIMINATION,
            use_score_options=True,
        )


def test_rag_generation_path_picks_gold():
    """use_score_options=False routes through generate()+parse_answer.
    The MockLLM emits "Answer: <gold>" so the strategy should pick that
    letter."""
    strategy = RAGStrategy(
        MockLLM(correctness=1.0),
        MockRetriever(_passages()),
        k=2,
        style=PromptStyle.ZERO_SHOT,
        use_score_options=False,
    )
    out = strategy.answer(_inp("C"))
    assert out.chosen_index == 2
    assert out.extras["decoding_path"] == "generate"
    assert out.extras["parse_ok"] is True


def test_rag_generation_path_abstains_on_unparseable_output():
    """On parse failure the strategy abstains, same shape as BaselineLLMStrategy."""
    class BadMock(MockLLM):
        def generate(self, messages, **_):
            from polimibot.models.llm import LLMResponse
            return LLMResponse(text="I cannot determine.", elapsed_seconds=0.001)

    strategy = RAGStrategy(
        BadMock(),
        MockRetriever(_passages()),
        k=2,
        style=PromptStyle.ZERO_SHOT,
        use_score_options=False,
    )
    out = strategy.answer(_inp())
    assert out.is_abstain is True
    assert out.extras["parse_ok"] is False


def test_rag_cot_uses_generation_path_by_default():
    """When style is CoT, use_score_options must default to False."""
    strategy = RAGStrategy(
        MockLLM(correctness=1.0),
        MockRetriever(_passages()),
        k=2,
        style=PromptStyle.ZERO_SHOT_COT,
        use_score_options=False,
    )
    out = strategy.answer(_inp("D"))
    assert out.extras["decoding_path"] == "generate"


def test_rag_name_tags_generation_path():
    strategy = RAGStrategy(
        MockLLM(), MockRetriever(_passages()),
        style=PromptStyle.ZERO_SHOT,
        use_score_options=False,
    )
    assert "|gen" in strategy.name


# ── Reranker integration ─────────────────────────────────────────────────────

def test_rag_rejects_use_reranker_when_retriever_has_none():
    """use_reranker=True with a retriever that has no reranker is a config
    error — fail at construction, not on the first answer."""
    retriever = MockRetriever(_passages(), has_reranker=False)
    with pytest.raises(ValueError, match="no reranker"):
        RAGStrategy(MockLLM(), retriever, use_reranker=True)


def test_rag_passes_rerank_true_to_retriever_when_enabled():
    """Single-query path (use_multi_query=False): rerank=True goes into retrieve()."""
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_reranker=True,
                           use_multi_query=False)
    strategy.answer(_inp())
    assert retriever.last_rerank is True


def test_rag_does_not_pass_rerank_when_disabled():
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_reranker=False,
                           use_multi_query=False)
    strategy.answer(_inp())
    assert retriever.last_rerank is False


def test_rag_forwards_rerank_oversearch_when_set():
    """Single-query path: rerank_oversearch passes through to retrieve()."""
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(
        MockLLM(), retriever, k=2,
        use_reranker=True, rerank_oversearch=12,
        use_multi_query=False,
    )
    strategy.answer(_inp())
    assert retriever.last_rerank_oversearch == 12


def test_rag_omits_rerank_oversearch_when_none():
    """rerank_oversearch=None → don't pass the kwarg, let Retriever use its default."""
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_reranker=True,
                           use_multi_query=False)
    strategy.answer(_inp())
    assert retriever.last_rerank_oversearch is None


def test_rag_extras_records_reranked_flag():
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_reranker=True)
    out = strategy.answer(_inp())
    assert out.extras["reranked"] is True


def test_rag_name_includes_rerank_tag():
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_reranker=True)
    assert "rerank" in strategy.name


# ── Hybrid passthrough ───────────────────────────────────────────────────────

def test_rag_rejects_use_hybrid_when_retriever_has_no_bm25():
    retriever = MockRetriever(_passages(), has_bm25=False)
    with pytest.raises(ValueError, match="no BM25 index"):
        RAGStrategy(MockLLM(), retriever, use_hybrid=True)


def test_rag_passes_hybrid_true_to_retriever_when_enabled():
    retriever = MockRetriever(_passages(), has_bm25=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_hybrid=True)
    strategy.answer(_inp())
    assert retriever.last_hybrid is True


def test_rag_does_not_pass_hybrid_when_disabled():
    retriever = MockRetriever(_passages(), has_bm25=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_hybrid=False)
    strategy.answer(_inp())
    assert retriever.last_hybrid is False


def test_rag_extras_records_hybrid_flag():
    retriever = MockRetriever(_passages(), has_bm25=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_hybrid=True)
    out = strategy.answer(_inp())
    assert out.extras["hybrid"] is True


def test_rag_name_includes_hybrid_tag():
    retriever = MockRetriever(_passages(), has_bm25=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_hybrid=True)
    assert "hybrid" in strategy.name


# ── Multi-query ──────────────────────────────────────────────────────────────

def test_rag_multi_query_issues_one_call_per_query():
    """4 options + 1 question-only → 5 retrieval calls per answer."""
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_multi_query=True)
    strategy.answer(_inp())
    assert len(retriever.queries_seen) == 5


def test_rag_multi_query_queries_cover_question_and_each_option():
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_multi_query=True)
    strategy.answer(_inp("B"))
    # First query is the bare question.
    assert retriever.queries_seen[0].startswith("Who crossed the Rubicon?")
    # Subsequent queries each include exactly one option text.
    for opt in ("Pompey", "Caesar", "Augustus", "Cicero"):
        assert any(opt in q for q in retriever.queries_seen[1:])


def test_rag_single_query_issues_one_call():
    """use_multi_query=False → exactly one retrieval call (ablation/legacy path)."""
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_multi_query=False)
    strategy.answer(_inp())
    assert len(retriever.queries_seen) == 1


def test_rag_multi_query_is_on_by_default():
    """use_multi_query=True is the new default (audit §3/#4 fix): embedding
    question+4 distractors in one vector dilutes the right-answer signal."""
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2)  # default
    out = strategy.answer(_inp())
    # Default should issue 5 retrieval calls (question + 4 per-option).
    assert len(retriever.queries_seen) == 5
    assert out.extras["multi_query"] is True


def test_rag_extras_records_multi_query_flag():
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_multi_query=True)
    out = strategy.answer(_inp())
    assert out.extras["multi_query"] is True


def test_rag_multi_query_extras_query_preview_is_truncated():
    """The full multi-query list is reconstructible from question+options
    at analysis time; the extras 'query' field is a preview only."""
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_multi_query=True)
    out = strategy.answer(_inp())
    assert "…" in out.extras["query"]   # 5 queries > 3 → ellipsis


def test_rag_name_includes_mq_tag_when_multi_query():
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_multi_query=True)
    assert "mq" in strategy.name


def test_rag_multi_query_composes_with_hybrid_no_rerank():
    """multi_query + hybrid (no rerank): 5 retrieve() calls, each with hybrid=True."""
    retriever = MockRetriever(_passages(), has_bm25=True)
    strategy = RAGStrategy(
        MockLLM(), retriever, k=2,
        use_hybrid=True, use_multi_query=True,
    )
    strategy.answer(_inp())
    assert len(retriever.queries_seen) == 5
    assert retriever.last_hybrid is True
    # No reranker — rerank_pool must NOT have been called.
    assert retriever.rerank_pool_calls == []


def test_rag_multi_query_rerank_fuses_first_then_reranks_once():
    """Correct composition order (audit §5): per-query retrieve() calls do NOT
    use the cross-encoder (rerank=False); the cross-encoder is invoked exactly
    once via rerank_pool() on the fused pool.

    Old (broken) order: rerank per query (5× cost) then outer RRF discards
    cross-encoder scores.
    New (correct) order: retrieve-per-query → RRF-fuse → rerank_pool once.
    """
    retriever = MockRetriever(_passages(), has_bm25=True, has_reranker=True)
    strategy = RAGStrategy(
        MockLLM(), retriever, k=2,
        use_hybrid=True, use_reranker=True, use_multi_query=True,
    )
    strategy.answer(_inp())

    # Per-query calls must NOT have used the cross-encoder.
    assert retriever.last_rerank is False, (
        "Per-query retrieve() calls should not rerank — "
        "that wastes 5× cross-encoder compute and narrows each candidate pool."
    )
    # hybrid flag still reaches each retrieve() call.
    assert retriever.last_hybrid is True

    # rerank_pool() must have been called exactly once on the fused pool.
    assert len(retriever.rerank_pool_calls) == 1, (
        "Expected rerank_pool to be called once (fuse-then-rerank), "
        f"got {len(retriever.rerank_pool_calls)} calls."
    )
    call = retriever.rerank_pool_calls[0]
    # The rerank query is the bare question (most relevant signal for the
    # cross-encoder — not diluted by all four option strings).
    assert call["query"].startswith("Who crossed the Rubicon?")
    # k requested must equal the strategy's top-k.
    assert call["k"] == 2


# ── LLM-based live-search query extraction ──────────────────────────────────

class _RecordingMockLLM(MockLLM):
    """MockLLM that records every generate() call and returns a canned response."""

    def __init__(self, generate_response: str = "Julius Caesar Rubicon", **kw):
        super().__init__(**kw)
        self.generate_calls: list[list[dict]] = []
        self._generate_response = generate_response

    def generate(self, messages, **_):
        from polimibot.models.llm import LLMResponse
        self.generate_calls.append(list(messages))
        return LLMResponse(text=self._generate_response, elapsed_seconds=0.001)


class _GatedRetriever(MockRetriever):
    """Always returns a passage with a score below any reasonable threshold
    so the gate always fires and live search is triggered."""

    def retrieve(self, query, k=3, **kwargs):
        super().retrieve(query, k=k, **kwargs)
        return [(Chunk(text="irrelevant", source="X", chunk_id=0), 0.01)]


def test_live_use_llm_query_fires_generate_before_search():
    """When live_use_llm_query=True, a generate() call is made to extract
    search keywords before live search fires."""
    llm = _RecordingMockLLM(generate_response="Julius Caesar Roman general")
    # Gated retriever: top score=0.01 < min_score=0.30 → gate fires.
    retriever = _GatedRetriever(_passages())

    # Patch live search to avoid real HTTP calls.
    class _FakeLiveSearch:
        searched: list[str] = []
        def search(self, query, *, category=None):
            _FakeLiveSearch.searched.append(query)
            return []   # no articles — just checking query was passed

    strategy = RAGStrategy(
        llm, retriever, k=2,
        min_score=0.30,
        use_multi_query=False,
        use_live_fallback=True,
        live_use_llm_query=True,
    )
    strategy._live_search = _FakeLiveSearch()  # inject fake live search

    strategy.answer(_inp("B"))

    # A generate() call should have been made.
    assert len(llm.generate_calls) >= 1, (
        "Expected at least one llm.generate() call for keyword extraction."
    )
    # The generate call's prompt should mention the question.
    prompt_content = llm.generate_calls[0][-1]["content"]
    assert "Who crossed the Rubicon?" in prompt_content

    # The generated keywords should be passed to live search.
    assert len(_FakeLiveSearch.searched) == 1
    assert _FakeLiveSearch.searched[0] == "Julius Caesar Roman general"


def test_live_use_llm_query_false_uses_bare_question():
    """When live_use_llm_query=False (default), no extra generate() call
    is made — the bare question goes to live search unchanged."""
    llm = _RecordingMockLLM()
    retriever = _GatedRetriever(_passages())

    class _FakeLiveSearch:
        searched: list[str] = []
        def search(self, query, *, category=None):
            _FakeLiveSearch.searched.append(query)
            return []

    strategy = RAGStrategy(
        llm, retriever, k=2,
        min_score=0.30,
        use_multi_query=False,
        use_live_fallback=True,
        live_use_llm_query=False,   # default
    )
    strategy._live_search = _FakeLiveSearch()

    strategy.answer(_inp("B"))

    # No generate() calls should have been made for keyword extraction
    # (score_options is used for the final answer — those go through
    # score_options(), not generate()).
    for call in llm.generate_calls:
        content = call[-1]["content"]
        assert "Wikipedia search keywords" not in content, (
            "Unexpected keyword-extraction prompt when live_use_llm_query=False."
        )

    # The live search should receive the bare question.
    assert len(_FakeLiveSearch.searched) == 1
    assert _FakeLiveSearch.searched[0].startswith("Who crossed the Rubicon?")


def test_live_search_query_in_extras():
    """The query actually sent to live search appears in extras as
    live_search_query — visible in the trace viewer."""
    llm = _RecordingMockLLM(generate_response="Roman history")
    retriever = _GatedRetriever(_passages())

    class _FakeLiveSearch:
        def search(self, query, *, category=None):
            return []

    strategy = RAGStrategy(
        llm, retriever, k=2,
        min_score=0.30,
        use_multi_query=False,
        use_live_fallback=True,
        live_use_llm_query=True,
    )
    strategy._live_search = _FakeLiveSearch()

    out = strategy.answer(_inp("B"))
    assert "live_search_query" in out.extras
    assert out.extras["live_search_query"] == "Roman history"


def test_extract_search_query_strips_think_tags():
    """Qwen3 complete <think>…</think> traces should be stripped."""
    llm = _RecordingMockLLM(
        generate_response="<think>some internal reasoning</think>Julius Caesar"
    )
    retriever = _GatedRetriever(_passages())

    class _FakeLiveSearch:
        searched: list[str] = []
        def search(self, query, *, category=None):
            _FakeLiveSearch.searched.append(query)
            return []

    strategy = RAGStrategy(
        llm, retriever, k=2,
        min_score=0.30,
        use_multi_query=False,
        use_live_fallback=True,
        live_use_llm_query=True,
    )
    strategy._live_search = _FakeLiveSearch()

    strategy.answer(_inp("B"))

    # <think>…</think> should be stripped; only "Julius Caesar" remains.
    assert _FakeLiveSearch.searched[0] == "Julius Caesar"


def test_extract_search_query_strips_truncated_think_tags():
    """Qwen3 <think>… blocks truncated by max_new_tokens (no closing tag)
    should also be stripped — this was the real-world failure mode where
    the model never reached the keywords within the token budget."""
    llm = _RecordingMockLLM(
        # Simulates a think block cut off mid-stream (no </think>)
        generate_response="<think>\nOkay, let's tackle this. The user wants me to extract 2-5 Wikipedia"
    )
    retriever = _GatedRetriever(_passages())

    class _FakeLiveSearch:
        searched: list[str] = []
        def search(self, query, *, category=None):
            _FakeLiveSearch.searched.append(query)
            return []

    strategy = RAGStrategy(
        llm, retriever, k=2,
        min_score=0.30,
        use_multi_query=False,
        use_live_fallback=True,
        live_use_llm_query=True,
    )
    strategy._live_search = _FakeLiveSearch()

    strategy.answer(_inp("B"))

    # The truncated <think> block should be stripped and result is empty →
    # fallback to the bare question (not garbage Wikipedia query).
    assert _FakeLiveSearch.searched[0].startswith("Who crossed the Rubicon?"), (
        f"Expected question fallback, got: {_FakeLiveSearch.searched[0]!r}"
    )


def test_extract_search_query_falls_back_to_question_on_empty_response():
    """If the LLM response is empty after stripping, fall back to the
    bare question so live search always has a non-empty query."""
    llm = _RecordingMockLLM(generate_response="<think>all thinking</think>")
    retriever = _GatedRetriever(_passages())

    class _FakeLiveSearch:
        searched: list[str] = []
        def search(self, query, *, category=None):
            _FakeLiveSearch.searched.append(query)
            return []

    strategy = RAGStrategy(
        llm, retriever, k=2,
        min_score=0.30,
        use_multi_query=False,
        use_live_fallback=True,
        live_use_llm_query=True,
    )
    strategy._live_search = _FakeLiveSearch()

    strategy.answer(_inp("B"))

    # Should have fallen back to the bare question.
    assert _FakeLiveSearch.searched[0].startswith("Who crossed the Rubicon?")
