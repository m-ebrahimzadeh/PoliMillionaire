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
    """Returns canned passages. No FAISS, no embedder."""
    def __init__(self, passages: list[tuple[Chunk, float]]) -> None:
        self._passages = passages
        self.last_query: str = ""
        self.n_chunks = len(passages)

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        self.last_query = query
        return self._passages[:k]


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