"""RAG strategy: retrieve relevant Wikipedia passages, then score with the LLM.

The retrieval query includes both question text and option texts — this
improves recall on questions where key entities appear only in the options.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..models.llm import LLM, AnswerProbabilities
from ..models.mock import MockLLM
from ..prompts.templates import PromptStyle, build_messages_with_context, parse_answer
from ..rag.chunker import Chunk
from ..rag.retriever import Retriever
from .base import Strategy, StrategyInput, StrategyOutput
from .llm_baseline import (
    DEFAULT_DIRECT_MAX_NEW_TOKENS,
    DEFAULT_COT_MAX_NEW_TOKENS,
    DEFAULT_COT_STOP_STRINGS,
    _COT_STYLES,
)

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
        use_score_options: bool = True,
        use_category_filter: bool = True,
        min_score: Optional[float] = None,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
        direct_max_new_tokens: int = DEFAULT_DIRECT_MAX_NEW_TOKENS,
        cot_max_new_tokens:    int = DEFAULT_COT_MAX_NEW_TOKENS,
        stop_strings: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            llm: loaded LLM (or MockLLM for tests).
            retriever: a Retriever with a built index.
            k: number of passages to retrieve per question.
            style: prompt variant (see PromptStyle).
            use_score_options: if True (default), read logits at the
                next-token position; if False, free-generate and parse the
                emitted letter. **CoT and ELIMINATION styles require
                free generation** — score_options reads the first
                predicted token, which is the start of the reasoning
                trace, not the answer letter.
            use_category_filter: when True, the retriever is asked to
                restrict to chunks tagged with ``inp.category``. Set False
                to ablate the filter (i.e. show every question all of
                the corpus).
            min_score: if set, drop the retrieval context entirely when the
                top score is below this threshold — the prompt then degrades
                to the no-RAG baseline shape. None = never gate. Calibrate
                empirically from the score distribution (~0.30–0.45 for
                normalised cosine on Wikipedia trivia).
            max_passage_chars: cap per individual passage. Tighten when
                context-length pressure is high; loosen to retain detail.
            max_total_chars: cap on the joined context block.
            direct_max_new_tokens / cot_max_new_tokens / stop_strings:
                generation budgets and early-stop strings used when
                use_score_options=False. Defaults mirror BaselineLLMStrategy.
        """
        if style in _COT_STYLES and use_score_options:
            raise ValueError(
                f"style={style.value} requires free generation "
                f"(use_score_options=False). score_options reads the first "
                f"predicted token — which is the start of the reasoning "
                f"trace, not the answer letter."
            )

        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.style = style
        self.use_score_options = use_score_options
        self.use_category_filter = use_category_filter
        self.min_score = min_score
        self.max_passage_chars = max_passage_chars
        self.max_total_chars = max_total_chars
        self.direct_max_new_tokens = direct_max_new_tokens
        self.cot_max_new_tokens    = cot_max_new_tokens
        # Default to boxed-stops on CoT/ELIMINATION; off for direct (no
        # boxed output expected in 16-token budgets).
        if stop_strings is None and style in _COT_STYLES:
            self.stop_strings: Optional[tuple[str, ...]] = DEFAULT_COT_STOP_STRINGS
        else:
            self.stop_strings = tuple(stop_strings) if stop_strings else None

        gate_tag = f"|min_score={min_score}" if min_score is not None else ""
        path_tag = "" if use_score_options else "|gen"
        cat_tag  = "" if use_category_filter else "|no_cat_filter"
        self.name = (
            f"rag[{getattr(llm, 'name', 'llm')}"
            f"|k={k}|{style.value}{path_tag}{cat_tag}{gate_tag}]"
        )

    def warm_up(self) -> None:
        """Dummy forward pass to absorb CUDA JIT compilation."""
        from .base import StrategyInput
        dummy = StrategyInput(question="Warm-up", options=("A", "B", "C", "D"), level=1)
        self.answer(dummy)

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        # 1. Retrieve. Pass the question's category to the retriever when
        #    the category filter is enabled — surfaces only on-topic
        #    chunks, prevents (say) a MATHS question from pulling
        #    HISTORY noise into the prompt.
        query = _build_query(inp)
        category_filter = (
            inp.category.value
            if (self.use_category_filter and inp.category is not None)
            else None
        )
        passages = self.retriever.retrieve(
            query, k=self.k, category=category_filter,
        )

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

        # 4. Score options via logits — OR generate freely and parse —
        #    depending on the style. CoT / ELIMINATION constructor
        #    validation guarantees we're in the right branch.
        if self.use_score_options:
            chosen_index, confidence, probs, margin, parse_ok = self._answer_via_logits(messages)
            rationale_text = context
        else:
            chosen_index, confidence, probs, margin, parse_ok, gen_text = (
                self._answer_via_generation(messages)
            )
            rationale_text = gen_text if gen_text else context

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
            confidence=confidence,
            rationale=rationale_text,
            is_abstain=(not parse_ok),
            extras={
                "probs":      probs,
                "margin":     margin,
                "n_passages": len(passages),
                "top_source": passages[0][0].source if passages else None,
                "top_score":  round(top_score, 4) if passages else None,
                "gated_by_min_score":  gated,
                "min_score_threshold": self.min_score,
                "category_filter":     category_filter,
                "query":               query,
                "passages":            passage_triples,
                "decoding_path":       "logits" if self.use_score_options else "generate",
                "parse_ok":            parse_ok,
            },
        )

    # ── private — two decoding paths ────────────────────────────────────────

    def _answer_via_logits(self, messages):
        """Single forward pass: read logits at the answer-letter position."""
        result: AnswerProbabilities = self.llm.score_options(messages)
        chosen_index = ord(result.top_letter) - ord("A")
        return (
            chosen_index,
            result.top_prob,
            result.probs,
            result.margin,
            True,   # parse_ok — logits always produce a letter
        )

    def _answer_via_generation(self, messages):
        """Free generation + parse_answer. For CoT and ELIMINATION styles."""
        max_tok = (
            self.cot_max_new_tokens if self.style in _COT_STYLES
            else self.direct_max_new_tokens
        )
        response = self.llm.generate(
            messages,
            max_new_tokens=max_tok,
            temperature=0.0,
            stop_strings=self.stop_strings,
        )
        idx = parse_answer(response.text)
        parse_ok = idx is not None
        return (
            idx if parse_ok else 0,
            0.5 if parse_ok else 0.25,
            None,    # no per-letter probs from the generation path
            None,    # no margin
            parse_ok,
            response.text,
        )