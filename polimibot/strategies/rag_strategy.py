"""RAG strategy: retrieve relevant Wikipedia passages, then score with the LLM.

The retrieval query includes both question text and option texts — this
improves recall on questions where key entities appear only in the options.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..models.llm import LLM, AnswerProbabilities
from ..models.mock import MockLLM
from ..prompts.templates import PromptStyle, build_messages_with_context
from ..rag.chunker import Chunk
from ..rag.retriever import Retriever
from .base import Strategy, StrategyInput, StrategyOutput

_AnyLLM = LLM | MockLLM

# Default per-passage and total-context character budgets. Promoted to
# constructor args on RAGStrategy so ablations don't need code edits.
DEFAULT_MAX_PASSAGE_CHARS = 800
DEFAULT_MAX_TOTAL_CHARS   = 2400


def _build_query(inp: StrategyInput) -> str:
    """Include option texts in the query for better entity recall."""
    return f"{inp.question} {' '.join(inp.options)}"


def _format_context(
    passages: list[tuple[Chunk, float]],
    *,
    max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Number passages and truncate to fit within the context budget.

    Args:
        passages: (Chunk, score) pairs from Retriever.retrieve()
        max_passage_chars: hard cap on each individual passage
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
        snippet = chunk.text[:max_passage_chars]
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
        min_score: Optional[float] = None,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    ) -> None:
        """
        Args:
            llm: loaded LLM (or MockLLM for tests).
            retriever: a Retriever with a built index.
            k: number of passages to retrieve per question.
            style: prompt variant (see PromptStyle).
            min_score: if set, drop the retrieval context entirely when the
                top score is below this threshold — the prompt then degrades
                to the no-RAG baseline shape. None = never gate. Calibrate
                empirically from the score distribution (~0.30–0.45 for
                normalised cosine on Wikipedia trivia).
            max_passage_chars: cap per individual passage. Tighten when
                context-length pressure is high; loosen to retain detail.
            max_total_chars: cap on the joined context block.
        """
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.style = style
        self.min_score = min_score
        self.max_passage_chars = max_passage_chars
        self.max_total_chars = max_total_chars
        gate_tag = f"|min_score={min_score}" if min_score is not None else ""
        self.name = f"rag[{getattr(llm, 'name', 'llm')}|k={k}|{style.value}{gate_tag}]"

    def warm_up(self) -> None:
        """Dummy forward pass to absorb CUDA JIT compilation."""
        from .base import StrategyInput
        dummy = StrategyInput(question="Warm-up", options=("A", "B", "C", "D"), level=1)
        self.answer(dummy)

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        # 1. Retrieve
        query = _build_query(inp)
        passages = self.retriever.retrieve(query, k=self.k)

        # 2. Low-score gate. If the top retrieval is below the threshold,
        #    pass an empty context — build_messages_with_context then
        #    degrades to the plain (no-RAG) prompt shape, instead of
        #    feeding the model irrelevant "evidence".
        top_score = float(passages[0][1]) if passages else 0.0
        gated = (
            self.min_score is not None
            and (not passages or top_score < self.min_score)
        )
        if gated:
            context = ""
        else:
            context = _format_context(
                passages,
                max_passage_chars=self.max_passage_chars,
                max_total_chars=self.max_total_chars,
            )

        # 3. Build RAG-augmented prompt (or degraded plain prompt if gated)
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
                "probs":      result.probs,
                "margin":     result.margin,
                "n_passages": len(passages),
                "top_source": passages[0][0].source if passages else None,
                "top_score":  round(top_score, 4) if passages else None,
                "gated_by_min_score":  gated,
                "min_score_threshold": self.min_score,
                "query":      query,
                "passages":   passage_triples,
            },
        )