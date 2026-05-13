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
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2)
    strategy.answer(_inp())
    # Options ("Pompey", "Caesar", ...) should appear in the query
    assert "Caesar" in retriever.last_query
    assert "Pompey" in retriever.last_query


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


# ── Low-score gate ────────────────────────────────────────────────────────────

def test_rag_min_score_gate_drops_context_below_threshold():
    """When the top retrieval score is below min_score, RAG degrades to
    plain (no-context) prompting. The model is not fed irrelevant evidence."""
    low_score_passages = [
        (Chunk(text="off-topic", source="Wrong", chunk_id=0), 0.10),
    ]
    strategy = RAGStrategy(
        MockLLM(correctness=1.0),
        MockRetriever(low_score_passages),
        k=1,
        min_score=0.30,
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
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_reranker=True)
    strategy.answer(_inp())
    assert retriever.last_rerank is True


def test_rag_does_not_pass_rerank_when_disabled():
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_reranker=False)
    strategy.answer(_inp())
    assert retriever.last_rerank is False


def test_rag_forwards_rerank_oversearch_when_set():
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(
        MockLLM(), retriever, k=2,
        use_reranker=True, rerank_oversearch=12,
    )
    strategy.answer(_inp())
    assert retriever.last_rerank_oversearch == 12


def test_rag_omits_rerank_oversearch_when_none():
    """rerank_oversearch=None → don't pass the kwarg, let Retriever use its default."""
    retriever = MockRetriever(_passages(), has_reranker=True)
    strategy = RAGStrategy(MockLLM(), retriever, k=2, use_reranker=True)
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
    """use_multi_query=False (default) → exactly one retrieval call."""
    retriever = MockRetriever(_passages())
    strategy = RAGStrategy(MockLLM(), retriever, k=2)
    strategy.answer(_inp())
    assert len(retriever.queries_seen) == 1


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
