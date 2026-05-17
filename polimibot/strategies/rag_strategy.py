"""RAG strategy: retrieve relevant Wikipedia passages, then score with the LLM.

The retrieval query includes both question text and option texts — this
improves recall on questions where key entities appear only in the options.

Live-search fallback
────────────────────
When ``use_live_fallback=True`` and the offline retrieval score is below the
configured threshold (``gated_by_min_score=True``), the strategy fires a
real-time Wikipedia API query via ``LiveSearchFallback`` instead of degrading
to a bare-LLM prompt.  The fetched articles are:

  1. Formatted as context identical to the offline RAG path — the LLM sees
     the same "Reference material" block regardless of source.
  2. Buffered in the attached ``IndexGrower`` (if one is provided) so that
     if the game server later confirms the answer was correct, the articles
     are appended to the live index for future questions.

The ``IndexGrower`` is an optional dependency — ``RAGStrategy`` works without
it (no learning), and with it (self-growing index).  The runner wires both
together; see ``runner.py``.
"""
from __future__ import annotations

import time
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


def _truncate_to_sentence(text: str, max_chars: int) -> str:
    """Truncate ``text`` to at most ``max_chars``, rounding down to the
    nearest sentence boundary so the model never receives dangling fragments.

    Strategy:
      1. If text fits, return as-is.
      2. Find the last sentence-ending punctuation (.!?) followed by a
         space or end-of-string within the budget. Cut there.
      3. Fall back to the last whitespace boundary within the budget.
      4. Hard-truncate only if no whitespace exists (pathological input).
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Walk backwards looking for a sentence-ending punctuation + boundary.
    for i in range(len(window) - 1, -1, -1):
        ch = window[i]
        if ch in ".!?":
            # Accept if it's at the end of the window or followed by
            # whitespace in the original text.
            next_idx = i + 1
            if next_idx >= len(text) or text[next_idx].isspace():
                return window[:next_idx]
    # No sentence boundary — try word boundary.
    last_space = window.rfind(" ")
    if last_space > 0:
        return window[:last_space]
    return window  # hard truncate (no spaces at all)


def _format_context(
    passages: list[tuple[Chunk, float]],
    *,
    max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Number passages, add citation headers, and truncate at sentence boundaries.

    Args:
        passages: (Chunk, score) pairs from Retriever.retrieve()
        max_passage_chars: hard cap on each individual passage
        max_total_chars: hard cap on total context block

    Returns:
        Human-readable numbered list, e.g.:

        [1] Julius Caesar (chunk 3, history)
        Caesar crossed the Rubicon in 49 BC.

        [2] Roman Republic (chunk 0)
        The Roman Republic was founded in 509 BC.

    Citation discipline:
      - Each passage carries its source article, chunk_id, and (when
        available) its category. The LLM can refer to "[Passage 1]" in
        its reasoning without needing the full article name.
      - Truncation rounds to the nearest preceding sentence boundary so
        the model never reads dangling mid-sentence fragments.
    """
    parts: list[str] = []
    used = 0
    for i, (chunk, _score) in enumerate(passages, start=1):
        snippet = _truncate_to_sentence(chunk.text, max_passage_chars)
        cat_suffix = f", {chunk.category}" if chunk.category else ""
        header = f"[{i}] {chunk.source} (chunk {chunk.chunk_id}{cat_suffix})"
        entry = f"{header}\n{snippet}"
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
        # ── Live-search fallback + self-growing index ──────────────────
        use_live_fallback: bool = False,
        live_search_timeout: float = 5.0,
        live_max_articles: int = 2,
        index_grower=None,   # Optional[IndexGrower] — avoids circular import
        live_use_llm_query: bool = False,
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
            use_live_fallback: when True, fire a real-time Wikipedia API
                query whenever offline retrieval is gated by min_score.
                Off by default — zero behavior change for existing code.
                Requires the ``wikipedia`` package to be installed.
            live_search_timeout: hard wall-clock limit (seconds) for each
                live Wikipedia query.  Default: 5 s.
            live_max_articles: maximum Wikipedia articles to fetch per
                live query.  Each article's summary becomes a passage.
                Default: 2.
            index_grower: optional ``IndexGrower`` instance.  When provided,
                live-fetched articles are buffered here; the runner calls
                ``grower.confirm(question_id)`` after a correct answer so
                the article is permanently added to the offline index.
                When None, live-fetched context is used for this question
                only and nothing is learned.
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

        # ── Live-search fallback fields ────────────────────────────────
        self.use_live_fallback   = use_live_fallback
        self.live_search_timeout = live_search_timeout
        self.live_max_articles   = live_max_articles
        self.index_grower        = index_grower
        self.live_use_llm_query  = live_use_llm_query

        # Lazily constructed — only if use_live_fallback=True so the
        # wikipedia import isn't required for normal offline-only use.
        self._live_search = None
        if use_live_fallback:
            from ..rag.live_search import LiveSearchFallback
            self._live_search = LiveSearchFallback(
                timeout_seconds=live_search_timeout,
                max_articles=live_max_articles,
            )

        gate_tag = f"|min_score={min_score}" if min_score is not None else ""
        path_tag = "" if use_score_options else "|gen"
        cat_tag  = "" if use_category_filter else "|no_cat_filter"
        rer_tag  = "|rerank" if use_reranker else ""
        hyb_tag  = "|hybrid" if use_hybrid else ""
        mq_tag   = "|mq"     if use_multi_query else ""
        live_tag = "|live"   if use_live_fallback else ""
        self.name = (
            f"rag[{getattr(llm, 'name', 'llm')}"
            f"|k={k}|{style.value}{path_tag}{cat_tag}{hyb_tag}{mq_tag}{rer_tag}{gate_tag}{live_tag}]"
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

        # ── 3. Live-search fallback (fires only when offline is gated) ──────
        live_fired           = False
        live_articles_titles: list[str] = []
        live_latency_seconds: Optional[float] = None
        live_passages: list[tuple[Chunk, float]] = []
        # Pre-filter scored live passages — populated when live search fires and
        # articles are scored, before the threshold filter is applied.
        # Used to build the merged debug pool in extras when all passages are
        # filtered out. Empty list = live search did not fire / no articles.
        _live_all_scored: list[tuple[Chunk, float]] = []

        live_search_query: Optional[str] = None
        if gated and self._live_search is not None:
            _t0 = time.monotonic()
            # Optionally use the LLM to distil the question into focused
            # Wikipedia keyword terms.  When live_use_llm_query=False (default)
            # we fall back to the bare question string — identical to the
            # previous behaviour.
            if self.live_use_llm_query:
                live_search_query = self._extract_search_query(inp)
            else:
                live_search_query = inp.question
            live_articles = self._live_search.search(
                live_search_query,
                category=inp.category,
            )
            live_latency_seconds = round(time.monotonic() - _t0, 3)

            if live_articles:
                live_fired = True
                live_articles_titles = [a.title for a in live_articles]

                # Buffer articles in the IndexGrower so the runner can
                # confirm them after a correct answer.  question_id is
                # derived from the level to keep it stable across re-attempts.
                question_id = f"lvl_{inp.level}"

                # ── Collect all candidate chunks from live articles ──────────
                # We gather ALL chunks first (no per-article cap), then rank
                # the full pool and keep the globally best self.k passages.
                all_live_chunks: list[Chunk] = []
                if self.index_grower is not None:
                    for article in live_articles:
                        buffered = self.index_grower.buffer(article, question_id)
                        if buffered:
                            all_live_chunks.extend(buffered)
                else:
                    # No grower — chunk inline for context only.
                    from ..rag.chunker import chunk_text
                    for article in live_articles:
                        chunks = chunk_text(
                            article.text,
                            source=article.title,
                            category=article.category.value if article.category else None,
                        )
                        all_live_chunks.extend(chunks)

                # ── Score the candidate chunks with a real relevance signal ──
                # Priority:
                #   1. Cross-encoder reranker (highest accuracy, ~50 ms).
                #   2. Dense cosine similarity via the retriever's embedder.
                # Both paths produce a proper score in [-1, 1] / logit space
                # so the gating threshold is meaningful and the display shows
                # something other than a flat 1.0 for every live passage.
                if all_live_chunks:
                    unscored = [(c, 0.0) for c in all_live_chunks]
                    if self.use_reranker and getattr(self.retriever, "has_reranker", False):
                        # Cross-encoder rerank over the full live-chunk pool.
                        live_passages = self.retriever.rerank_pool(
                            inp.question, unscored, k=self.k
                        )
                    else:
                        # Dense cosine similarity — embed the question once,
                        # embed all candidate texts, compute dot products.
                        import numpy as np
                        embedder = getattr(self.retriever, "_embedder", None)
                        if embedder is not None:
                            q_vec = embedder.encode_query(inp.question)        # (1, D)
                            p_texts = [c.text for c in all_live_chunks]
                            p_vecs  = embedder.encode_passage(p_texts)         # (N, D)
                            # Cosine similarity: vectors are already normalised
                            # by the embedder, so dot product == cosine sim.
                            sims = (p_vecs @ q_vec.T).squeeze(-1)              # (N,)
                            scored = sorted(
                                zip(all_live_chunks, sims.tolist()),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                            live_passages = scored[:self.k]
                        else:
                            # No embedder available — fall back to 1.0 with a
                            # comment so it's clear why the score is synthetic.
                            live_passages = [(c, 1.0) for c in all_live_chunks[:self.k]]

                # ── Filter by the active threshold (same gate as offline) ─
                    # Prevents irrelevant articles (e.g. "Mary Rose" for a
                    # chemistry question) from reaching the LLM prompt.
                    # Save pre-filter list so we can build the debug pool later.
                    _live_all_scored = list(live_passages)
                    if active_threshold is not None:
                        live_passages = [
                            (c, s) for c, s in live_passages
                            if s >= active_threshold
                        ]

        # Best live score (pre-filter) — stored in extras for observability.
        live_top_score: Optional[float] = (
            round(max(s for _, s in _live_all_scored), 4)
            if _live_all_scored else None
        )

        # ── 4. Build context from offline passages OR live passages ──────────
        #
        # live_all_below: True when live search fired, retrieved articles,
        # scored them, but ALL scored below the threshold.
        # In that case the LLM still gets no context (same as fully-gated),
        # but we record the merged offline+live debug pool in extras for the
        # trace display.  _live_all_scored was set just before the threshold
        # filter above; it remains [] when live search never fired.
        live_all_below = live_fired and bool(_live_all_scored) and not live_passages

        if live_fired and live_passages:
            # Live search succeeded — use its passages as context.
            # The format is identical to offline RAG: same _format_context(),
            # same "Reference material" framing in the prompt.
            context = _format_context(
                live_passages,
                max_passage_chars=self.max_passage_chars,
                max_total_chars=self.max_total_chars,
            )
            effective_passages = live_passages
        elif gated:
            # Gated and live search either disabled or found nothing.
            context = ""
            effective_passages = passages
        else:
            # Normal offline RAG path.
            context = _format_context(
                passages,
                max_passage_chars=self.max_passage_chars,
                max_total_chars=self.max_total_chars,
            )
            effective_passages = passages

        # ── Build debug pool for the "all below threshold" case ─────────────
        # Merge offline + live passages sorted by score descending, keep top-k.
        # Tagged with source ("live" / "offline") for display in show_trace().
        # This is ONLY for observability — never sent to the LLM.
        debug_passages: list[dict] = []
        if live_all_below:
            merged_pool = (
                [(c, s, "live")    for c, s in _live_all_scored] +
                [(c, s, "offline") for c, s in passages]
            )
            merged_pool.sort(key=lambda x: x[1], reverse=True)
            debug_passages = [
                {
                    "source":       chunk.source,
                    "chunk_id":     chunk.chunk_id,
                    "score":        round(score, 4),
                    "text_preview": chunk.text[:200],
                    "pool":         pool_src,
                }
                for chunk, score, pool_src in merged_pool[:self.k * 2]
            ]

        # ── 5. Build RAG-augmented prompt (or degraded plain prompt if gated)
        messages = build_messages_with_context(
            inp.question,
            inp.options,
            context,
            category=inp.category,
            style=self.style,
        )

        # ── 6. Score options via logits — OR generate freely and parse ───────
        if self.use_score_options:
            chosen_index, confidence, probs, margin, parse_ok = self._answer_via_logits(messages)
            rationale_text = context
        else:
            chosen_index, confidence, probs, margin, parse_ok, gen_text = (
                self._answer_via_generation(messages)
            )
            rationale_text = gen_text if gen_text else context

        # Full retrieval triples — kept in extras for recall@k analysis.
        # text_preview (first 200 chars) lets the user read what the LLM
        # actually received without opening the full context string.
        passage_triples = [
            {
                "source":       chunk.source,
                "chunk_id":     chunk.chunk_id,
                "score":        round(score, 4),
                "text_preview": chunk.text[:200],
            }
            for chunk, score in effective_passages
        ]

        return StrategyOutput(
            chosen_index=chosen_index,
            confidence=confidence,
            rationale=rationale_text,
            is_abstain=(not parse_ok),
            extras={
                "probs":      probs,
                "margin":     margin,
                "n_passages": len(effective_passages),
                "top_source": effective_passages[0][0].source if effective_passages else None,
                "top_score":  round(top_score, 4) if effective_passages else None,
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
                # Live-search extras — always present for uniform log schema.
                "live_search_fired":    live_fired,
                "live_search_articles": live_articles_titles,
                "live_search_latency":  live_latency_seconds,
                "live_search_query":    live_search_query,
                "live_top_score":       live_top_score,
                # Debug pool — populated only when live fired but all passages
                # scored below threshold.  NOT sent to LLM; trace display only.
                "live_all_below_threshold": live_all_below,
                "debug_passages":           debug_passages or None,
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

    def _extract_search_query(self, inp: StrategyInput) -> str:
        """Ask the LLM to distil the MCQ into 2-5 focused Wikipedia keywords.

        Uses a minimal system-free chat turn so the model stays in
        instruction-following mode without any extra context overhead.
        The response is stripped of Qwen3 ``<think>…</think>`` blocks —
        including *truncated* blocks that were cut off by the token budget —
        before being returned; if the result is empty after stripping we
        fall back to the bare question to guarantee a non-empty query.

        Args:
            inp: the current question being answered.

        Returns:
            A short keyword string suitable for ``wikipedia.search()``,
            e.g. ``"Xenia ancient Greece hospitality"`` instead of the
            full 30-word question text.
        """
        import re
        prompt = (
            "Extract 2-5 Wikipedia search keywords from this trivia question. "
            "Do NOT use <think> tags. Output ONLY the keywords, nothing else.\n\n"
            f"Question: {inp.question}\n"
            f"Options: {', '.join(inp.options)}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            # 50 tokens: enough for the keywords even if the model emits a
            # short think preamble despite the instruction above.
            resp = self.llm.generate(messages, max_new_tokens=50, temperature=0.0)
            # Strip Qwen3 thinking traces — both complete (<think>…</think>)
            # *and* truncated ones (<think>… with no closing tag) that can
            # appear when max_new_tokens cuts off mid-block.
            text = re.sub(
                r"<think>.*?(?:</think>|$)", "", resp.text, flags=re.DOTALL
            ).strip()
            return text if text else inp.question
        except Exception:  # noqa: BLE001 — never crash the answer pipeline
            return inp.question
