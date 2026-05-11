"""RAG strategy: retrieve relevant Wikipedia passages, then score with the LLM.

The retrieval query includes both question text and option texts — this
improves recall on questions where key entities appear only in the options.
"""
from __future__ import annotations

from typing import Sequence

from ..models.llm import LLM, AnswerProbabilities
from ..models.mock import MockLLM
from ..prompts.templates import PromptStyle, build_messages_with_context
from ..rag.chunker import Chunk
from ..rag.retriever import Retriever
from .base import Strategy, StrategyInput, StrategyOutput

_AnyLLM = LLM | MockLLM

# Truncate each passage to this many characters before concatenating.
# Prevents the context window from overflowing on very long Wikipedia paragraphs.
_MAX_PASSAGE_CHARS = 800


def _build_query(inp: StrategyInput) -> str:
    """Include option texts in the query for better entity recall."""
    return f"{inp.question} {' '.join(inp.options)}"


def _format_context(passages: list[tuple[Chunk, float]], max_total_chars: int = 2400) -> str:
    """Number passages and truncate to fit within the context budget.

    Args:
        passages: (Chunk, score) pairs from Retriever.retrieve()
        max_total_chars: hard cap on total context length

    Returns:
        Human-readable numbered list, e.g.:
        [1] Julius Caesar
        Caesar crossed the Rubicon in 49 BC...

        [2] Roman Republic
        ...
    """
    parts: list[str] = []
    used = 0
    for i, (chunk, _score) in enumerate(passages, start=1):
        snippet = chunk.text[:_MAX_PASSAGE_CHARS]
        entry = f"[{i}] {chunk.source}\n{snippet}"
        if used + len(entry) > max_total_chars:
            break
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


class RAGStrategy(Strategy):
    """Retriever + LLM. Retrieves top-k passages, then scores options via logits.

    Args:
        llm: loaded LLM (or MockLLM for tests).
        retriever: a Retriever with a built index. Call warm_up() to load it.
        k: number of passages to retrieve per question.
        style: prompt variant (ZERO_SHOT or FEW_SHOT; CoT requires use_score_options=False).
    """

    def __init__(
        self,
        llm: _AnyLLM,
        retriever: Retriever,
        *,
        k: int = 3,
        style: PromptStyle = PromptStyle.ZERO_SHOT,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.style = style
        self.name = f"rag[{getattr(llm, 'name', 'llm')}|k={k}|{style.value}]"

    def warm_up(self) -> None:
        """Dummy forward pass to absorb CUDA JIT compilation."""
        from .base import StrategyInput
        dummy = StrategyInput(question="Warm-up", options=("A", "B", "C", "D"), level=1)
        self.answer(dummy)

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        # 1. Retrieve
        query = _build_query(inp)
        passages = self.retriever.retrieve(query, k=self.k)

        # 2. Format context
        context = _format_context(passages)

        # 3. Build RAG-augmented prompt
        messages = build_messages_with_context(
            inp.question,
            inp.options,
            context,
            category=inp.category,
            style=self.style,
        )

        # 4. Score options via one forward pass
        result: AnswerProbabilities = self.llm.score_options(messages)
        chosen_index = ord(result.top_letter) - ord("A")

        # Full retrieval triples are kept in `extras` (and propagate into
        # the run JSONL through the runner) so that recall@k / MRR can be
        # recomputed post-hoc from any historical run, without re-running
        # retrieval. The audit's recall harness reads these.
        passage_triples = [
            {"source": chunk.source, "chunk_id": chunk.chunk_id, "score": round(score, 4)}
            for chunk, score in passages
        ]

        return StrategyOutput(
            chosen_index=chosen_index,
            confidence=result.top_prob,
            rationale=context,   # stored in logs — useful for error analysis
            extras={
                "probs":   result.probs,
                "margin":  result.margin,
                "n_passages": len(passages),
                "top_source": passages[0][0].source if passages else None,
                "query":      query,
                "passages":   passage_triples,
            },
        )