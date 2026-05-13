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
from ..rag.fusion import reciprocal_rank_fusion
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
    """Single-query dense mode: question-only for dense retrieval.

    Used when ``multi_query=False``. The audit (§3) notes that
    concatenating all four options into one dense vector dilutes the
    signal — three options are distractors, so the right-answer entity
    gets ~20% weight in the averaged embedding. Question-only is the
    correct dense query.

    For BM25 (inside a hybrid retrieve() call), option tokens still
    contribute useful IDF-weighted signal because BM25 scores each
    token independently — there is no averaging over the four options.
    The Retriever handles the dense vs. BM25 split internally; the
    strategy just supplies the text and lets the retriever decide.
    """
    return inp.question


def _build_multi_queries(inp: StrategyInput) -> list[str]:
    """Multi-query mode: one query for the question, one per option.

    The audit's #4 point — concatenating four wrong-option entities
    into a single dense vector dilutes the question signal. Encoding
    each (question, option_i) separately and RRF-fusing gives every
    option a fair shot at pulling its supporting article into top-k,
    and the question-only query anchors the topic.

    Returns:
        ``[question, "question option_A", "question option_B", ...]``
        — 5 queries total for a 4-option MCQ.
    """
    return [inp.question] + [
        f"{inp.question} {opt}" for opt in inp.options
    ]


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
        use_reranker: bool = False,
        use_hybrid: bool = False,
        use_multi_query: bool = True,
        rerank_oversearch: Optional[int] = None,
        min_score: Optional[float] = None,
        min_score_rrf: Optional[float] = None,
        min_score_rerank: Optional[float] = None,
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
            use_reranker: when True, the retriever oversearches and reranks
                the pool with its attached cross-encoder. Requires the
                retriever to have been constructed with ``reranker=``
                (raises otherwise — surfaces the misconfiguration early
                rather than per-question).
            use_hybrid: when True, the retriever queries both the dense
                FAISS index AND the attached BM25 sidecar, RRF-fusing
                the two ranked lists. Requires bm25= on the retriever.
            use_multi_query: when True, retrieve once per query in
                ``[question, question+opt_A, ..., question+opt_D]``
                and RRF-fuse the five resulting ranked lists. Better
                recall on questions where the answer entity appears
                only in one option. Off by default — costs 5× the
                retrieval calls.
            rerank_oversearch: passes through to Retriever.retrieve. None
                = use Retriever's default (5×).
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
        if use_reranker and not getattr(retriever, "has_reranker", False):
            raise ValueError(
                "use_reranker=True but the retriever has no reranker. "
                "Construct it with Retriever(index, embedder, reranker=...)."
            )
        if use_hybrid and not getattr(retriever, "has_bm25", False):
            raise ValueError(
                "use_hybrid=True but the retriever has no BM25 index. "
                "Construct it with Retriever(index, embedder, bm25=...)."
            )

        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.style = style
        self.use_score_options = use_score_options
        self.use_category_filter = use_category_filter
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid
        self.use_multi_query = use_multi_query
        self.rerank_oversearch = rerank_oversearch
        self.min_score = min_score
        self.min_score_rrf = min_score_rrf
        self.min_score_rerank = min_score_rerank
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
        rer_tag  = "|rerank" if use_reranker else ""
        hyb_tag  = "|hybrid" if use_hybrid else ""
        mq_tag   = "|mq"     if use_multi_query else ""
        self.name = (
            f"rag[{getattr(llm, 'name', 'llm')}"
            f"|k={k}|{style.value}{path_tag}{cat_tag}{hyb_tag}{mq_tag}{rer_tag}{gate_tag}]"
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
        category_filter = (
            inp.category.value
            if (self.use_category_filter and inp.category is not None)
            else None
        )
        common_kwargs: dict = {"k": self.k, "category": category_filter}
        if self.use_reranker:
            common_kwargs["rerank"] = True
            if self.rerank_oversearch is not None:
                common_kwargs["rerank_oversearch"] = self.rerank_oversearch
        if self.use_hybrid:
            common_kwargs["hybrid"] = True

        if self.use_multi_query:
            # Correct pipeline for multi-query + optional rerank (audit §5):
            #   1. Retrieve top-N per query (no cross-encoder — rerank=False).
            #   2. RRF-fuse the per-query lists (across queries).
            #   3. If rerank is enabled, run the cross-encoder ONCE over the
            #      fused pool — not once per query (5× wasteful, narrow pools).
            # The retriever's own internal fusion (hybrid) still happens
            # inside each per-query retrieve() call.
            queries = _build_multi_queries(inp)

            # Strip rerank and k from per-query kwargs; k is passed
            # explicitly as pool_k below and reranking is handled after fusion.
            per_query_kwargs = {key: v for key, v in common_kwargs.items()
                                if key not in ("rerank", "rerank_oversearch", "k")}

            rerank_x = (
                self.rerank_oversearch
                or self.retriever._DEFAULT_RERANK_OVERSEARCH  # type: ignore[attr-defined]
            )
            # Ask each retriever call for enough candidates so the fused
            # pool is wide enough for a useful rerank pass.
            pool_k = self.k * rerank_x if self.use_reranker else self.k
            ranked_lists = [
                self.retriever.retrieve(q, k=pool_k, **per_query_kwargs)
                for q in queries
            ]
            fused = reciprocal_rank_fusion(ranked_lists, k=pool_k)

            if self.use_reranker:
                # Rerank the fused pool once with the question as the query.
                passages = self.retriever.rerank_pool(
                    inp.question, fused, k=self.k,
                )
            else:
                passages = fused[:self.k]

            # ``query`` for logging is a preview of the first queries.
            query = " | ".join(queries[:3]) + (" | …" if len(queries) > 3 else "")
        else:
            query = _build_query(inp)
            passages = self.retriever.retrieve(query, **common_kwargs)

        # 2. Path-aware low-score gate (audit §4).
        #
        # Score units differ by retrieval path:
        #   dense-only  → cosine ∈ [-1, 1]         use min_score
        #   hybrid RRF  → RRF ∈ ~0–0.03            use min_score_rrf
        #   reranker    → cross-encoder logit       use min_score_rerank
        #
        # Applying a cosine-calibrated threshold on an RRF or cross-encoder
        # score is meaningless (gates every question or none). Each path
        # selects its own threshold; None = never gate on this path.
        top_score = float(passages[0][1]) if passages else 0.0
        if self.use_reranker:
            active_threshold = self.min_score_rerank
        elif self.use_hybrid:
            active_threshold = self.min_score_rrf
        else:
            active_threshold = self.min_score
        gated = (
            active_threshold is not None
            and (not passages or top_score < active_threshold)
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
                "min_score_threshold": active_threshold,
                "category_filter":     category_filter,
                "reranked":            self.use_reranker,
                "hybrid":              self.use_hybrid,
                "multi_query":         self.use_multi_query,
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