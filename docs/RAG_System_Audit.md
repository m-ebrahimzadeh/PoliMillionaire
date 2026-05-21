# RAG System Audit — PoliMillionaire

**Auditor's stance:** I've read every file (`rag_strategy.py`, `retriever.py`, `corpus.py`, `chunker.py`, `embedder.py`, `reranker.py`, `bm25.py`, `fusion.py`, `live_search.py`, `index_grower.py`, `index.py`, `templates.py`) plus the orchestration notebook. The system is genuinely well-engineered — better than most student projects and many production prototypes. The audit below assumes that baseline of quality and focuses on the things that will actually move the per-category accuracy needle for *Who Wants to Be a Millionaire?*-style trivia under your stated constraint (Wikipedia + free APIs only).

The audit is organized as: (1) what's already excellent, (2) category-by-category diagnosis, (3) ranked recommendations with implementation specifics, and (4) a summary table.

---

## 1. What the system already does well

Before suggesting changes, the things that should **stay** because they're already correct (and that many production RAG systems get wrong):

1. **Asymmetric BGE encoding with prefix-checked manifest** (`embedder.py`, `retriever._check_manifest_compat`). The hard-fail on `query_prefix` / `passage_prefix` drift is the right discipline — BGE vectors silently corrupt scores when the prefix is missing on the query side, and you've correctly fenced it.
2. **Path-aware `min_score` gates** (`rag_strategy.py`). Three separate thresholds (`min_score`, `min_score_rrf`, `min_score_rerank`) because the score scales are not comparable. This is exactly right. Many systems apply a cosine-calibrated threshold to RRF scores and produce nonsense.
3. **Multi-query per-option fan-out + RRF fusion** (`_build_multi_queries`). For MCQ the answer entity often appears only in one option; question-only dense embedding under-weights it. Encoding `(question, option_i)` separately and RRF-fusing is the textbook fix, and you've implemented it correctly.
4. **Rerank-once-after-fusion** (in the multi-query branch). The audit comment in the code calls this out explicitly. Reranking inside each per-query call and then RRF-ing the cross-encoder scores would be both wasteful and a score-scale category error. You avoided it.
5. **Source-level diversification** (`_diversify_by_source`). Stops top-k from being three overlapping windows of the same article — a real failure mode on Wikipedia chunking with overlap.
6. **Sentence-boundary truncation** in `_truncate_to_sentence`. The LLM never sees a dangling clause. Small, easy-to-skip detail; matters for prompt quality.
7. **BM25 with positional postings + proximity bonus + stopword filter** (`bm25.py`). This is sophisticated. Proper-noun retrieval ("Pythagorean theorem", "49 BC") benefits enormously from the proximity bonus, and you've kept negation words in the stoplist — the right call for MCQ where "not" and "never" are semantically loaded.
8. **`IndexGrower` with confirm-only learning**. Embedding cost paid only on confirmed-correct articles. The thread-safe buffer→confirm→flush lifecycle is solid.
9. **Category filter with oversearch fallback** in `_dense_search`. The retry-with-full-index when 8× oversearch doesn't fill the quota handles thin categories gracefully.

These represent the project's strongest engineering. Don't regress them.

---

## 2. Category-by-category diagnosis

Your four categories have *fundamentally different* retrieval requirements and the current system applies a one-size-fits-all RAG to all of them. This is the single biggest weakness.

### 2.1 MATHS — RAG is mostly noise here

**The diagnosis:** Trivia maths questions (your example: t-test p-value computation) are *procedural*, not factual. The right answer to "What is the p-value of P(t > 2) with 15 df?" is not found by retrieving the Wikipedia article on Student's t-distribution — it requires computation. Yet your category filter forces every maths question into a tiny corpus of 20 maths-concept articles, then injects them into the prompt where they compete with the model's actual reasoning for attention.

Worse: your maths seed list (`TOPIC_SEEDS[Category.MATHS]`) is 20 *high-level concept articles* ("Mathematics", "Algebra", "Geometry"). These articles describe what algebra *is*; they don't contain solved problems. The retrieval is structurally incapable of helping a procedural question.

**Evidence in the prompt template:** Your `_CATEGORY_SYSTEM[Category.MATHS]` correctly says "Compute precisely — do not guess." This implies the model, not RAG, should solve it. But the RAG pipeline then injects irrelevant context.

**Recommendation:** For MATHS, **disable RAG entirely** by default, and route through `ToolStrategy(MathsTool)` or `AgentStrategy` instead. Your notebook already supports this via `USE_MATHS_TOOL` / `USE_AGENT_FOR_MATHS` and `TieredStrategy`'s `maths_override`. **Use it.** A `BaselineLLMStrategy` with `PromptStyle.ZERO_SHOT_COT` will outperform RAG on maths questions because the model has the computation skill and Wikipedia has nothing to add.

If you want to keep *some* RAG for the few definitional maths questions ("Who invented calculus?"), use a confidence-based escalation: try the bare LLM with CoT first, fall back to RAG only if the model's margin is below a threshold.

### 2.2 SCIENCE — corpus is too shallow

**The diagnosis:** Your science seed list is 25 articles, of which ~15 are high-level concept articles ("Quantum mechanics", "Thermodynamics", "Evolution"). Real WWtBaM science questions are entity-specific: "Who discovered penicillin?", "What's the half-life of Carbon-14?", "Which element has atomic number 79?". The 25-article corpus has poor coverage of named entities.

**Estimated recall@5 ceiling:** Without measurement (you have `retrieval_gold.jsonl` plumbing — use it), my prediction is ~40-50% for science. The cross-encoder reranker can't rescue a corpus that doesn't contain the answer.

**Recommendation:** Either (a) expand the science seed list by ~5× by adding category-specific subsets — list of elements, Nobel laureates in science, famous scientists, common organisms, planets and moons, common diseases — **or** (b) add a live Wikipedia fallback as the primary path for science (not just a gated fallback). Wikipedia's category graph is a natural seed source: scrape titles from `Category:Chemical_elements`, `Category:Nobel_laureates_in_Physics`, etc., via the MediaWiki API (free).

### 2.3 ENTERTAINMENT — Wikidata, not Wikipedia, is the right source

**The diagnosis:** Entertainment trivia (films, music, TV) is heavily *structured*: "Who directed X?", "In what year was Y released?", "What album was Z on?". This is exactly the data Wikidata stores as **structured triples** (P57=director, P577=publication date, etc.). Querying Wikipedia article text to extract structured facts that already exist as queryable triples is fighting your tools.

Your entertainment corpus is 25 articles. The Beatles' Wikipedia article is 15k+ words; embedding it as ~50 chunks means the chunk about "Hey Jude" recording dates may not rank top-3 for a question about that song.

**Recommendation:** Add a **Wikidata SPARQL fallback** for entertainment questions. The Wikidata Query Service (`https://query.wikidata.org/sparql`) is free, returns JSON, and supports queries like "Who directed the film titled X?". A two-line SPARQL query with rate-limiting handles this category vastly better than dense retrieval over prose. Wikipedia API remains your fallback when Wikidata has no match.

For the offline corpus, add the **MusicBrainz dump** (also free) for music questions — it's the canonical structured source.

### 2.4 HISTORY — current setup is closest to fit-for-purpose

**The diagnosis:** History questions are entity-rich and Wikipedia is dense on them. Your 25-article history seed is the most defensible of the four. The BM25 + proximity bonus correctly captures date and name signals. This is where your current RAG should perform best.

**Weak spots:** Disambiguation. "Henry V" the king vs. the Shakespeare play. "Napoleon" — Bonaparte vs. Napoleon III. Your `_seed_keywords` disambiguation walk only triggers on `wikipedia.DisambiguationError`; many entity collisions don't surface as that error.

**Recommendation:** Add entity-disambiguation prompting. Before retrieval, ask the model to identify the *type* of entity in the question (person/place/event/work) — use that as a category filter on Wikidata or as a side-prompt to BM25 (e.g., boost BM25 score when chunk text contains entity-type-indicator words).

---

## 3. Ranked recommendations (highest leverage first)

### Priority 1 — **Category-conditional retrieval strategy**

This is the single highest-leverage change. The current code routes all four categories through the same `RAGStrategy` instance. Replace this with a router:

```
HISTORY        → RAG (current pipeline, well-tuned)
ENTERTAINMENT  → Wikidata SPARQL → fall back to RAG
SCIENCE        → expanded-corpus RAG + live fallback (aggressive)
MATHS          → Tool / Agent / Baseline CoT (no RAG)
```

Your `TieredStrategy` infrastructure already supports per-category overrides — but currently only `maths_override`. **Extend it** to accept a `dict[Category, Strategy]` mapping. This is a ~30-line change to `tiered_strategy.py` and unlocks the rest.

### Priority 2 — **Replace seed-list corpus with category-graph harvest**

Your `TOPIC_SEEDS` is hand-curated and small (95 articles total). The MediaWiki API exposes the Wikipedia category graph for free:

```
https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Nobel_laureates_in_Physics&cmlimit=500
```

A one-time crawl seeded from ~20 well-chosen categories per topic will produce ~500-2000 articles per category instead of 25. **This alone should lift retrieval recall@5 by 20-30 points** because the bottleneck right now is "the right article isn't in the index". Cost: ~2 hours of one-time fetching + ~500 MB of FAISS index (still small).

Seed categories to start with:

- **History**: `Roman_emperors`, `Ancient_battles`, `World_War_II`, `Medieval_kings`, `Pharaohs`, `Founding_Fathers_of_the_United_States`
- **Science**: `Chemical_elements`, `Nobel_laureates_in_Physics`, `Nobel_laureates_in_Chemistry`, `Planets_of_the_Solar_System`, `Diseases_and_disorders`, `Famous_scientists`
- **Entertainment**: `Academy_Award_winners`, `Best_Picture_Academy_Award_winners`, `Grammy_Award_winners`, `Rock_and_Roll_Hall_of_Fame_inductees`, `American_sitcoms`

### Priority 3 — **Upgrade the embedding model**

You're using `BAAI/bge-small-en-v1.5` (384-dim, MTEB ~62). On 2026 leaderboards this is mid-tier. For a one-line swap:

- **`BAAI/bge-base-en-v1.5`** (768-dim, ~110 MB) — same prefix, drop-in. Expect +2-3 points on retrieval recall@5.
- **`BAAI/bge-m3`** (1024-dim, multilingual, supports dense+sparse+multi-vector) — heavier but the multi-vector mode partially replaces your reranker. Probably overkill for an assignment.
- **`mixedbread-ai/mxbai-embed-large-v1`** (1024-dim, MTEB ~64) — strong on factoid retrieval.

Your manifest discipline means the swap is safe — change `EmbedderSpec.model_name`, rebuild the index, and the hard-fail on prefix drift will catch any mismatch. **Critical:** `bge-m3` uses a *different* query prefix than `bge-small`; check the model card.

### Priority 4 — **HyDE (Hypothetical Document Embeddings) for hard questions**

HyDE: instead of embedding the question, ask the LLM to *hypothesize a 2-sentence answer*, then embed *that* and retrieve. The hypothesis lives in passage-space (closer to Wikipedia article prose), so retrieval is more accurate. Especially powerful for questions whose phrasing is unlike encyclopedia text ("Which famous physicist's wife also won a Nobel Prize?" → hypothesis "Marie Curie was married to Pierre Curie; both won Nobel Prizes" → retrieves the Curie articles cleanly).

**Integration into your code:** Add a `use_hyde: bool` flag to `RAGStrategy`. When True, between query construction and retrieval, call `self.llm.generate(messages=[{"role": "user", "content": f"Write one sentence that would be the answer to: {question}"}], max_new_tokens=80)` and use that hypothesis as the dense-retrieval query. Keep BM25 on the original question (lexical signal is in the question, not the hypothesis). RRF-fuse the two.

Cost: one extra LLM forward pass per question (~200 ms on Qwen 2.5 7B 4-bit). Latency hit acceptable on the medium/hard tiers.

### Priority 5 — **LLM query rewriting for live search** (refine existing code)

You already have `live_use_llm_query` and `_extract_search_query` — well done. Two refinements:

1. **Make it default-on for live search.** The bare question is a poor Wikipedia search query (too many stop words and complete-sentence syntax). 200ms of LLM time saves the live-search latency budget many times over by returning relevant articles on the first try.
2. **Cache the rewrite.** If the same question hits live search twice (re-attempt during the same game), the rewrite is identical — cache it on `question.text → rewritten_query`.

### Priority 6 — **Use Wikidata for entity normalization, not just lookup**

Even when you're using Wikipedia text, Wikidata is invaluable for disambiguation. Free API:

```
https://www.wikidata.org/w/api.php?action=wbsearchentities&search=Napoleon&language=en&format=json
```

Returns a ranked list of entities (Q517 = Napoleon Bonaparte, Q7721 = Napoleon III, …) with their types. Use this to canonicalize question entities before searching the offline index. Especially useful when your retriever returns a near-miss (e.g., Napoleon III article when the question is about Bonaparte).

### Priority 7 — **Tune chunk size per category**

You're using `chunk_size=300, overlap=50` globally. Empirically:

- **Entertainment & history** (entity facts, dates): smaller chunks (200 words, overlap 30) localize the answer better — the right chunk is denser in the right entity tokens.
- **Science** (causal/conceptual): your current 300/50 is fine.
- **Maths**: irrelevant (no RAG).

Rebuilding the index with two chunk-size profiles and category-aware retrieval (use the small-chunk index for entertainment+history, large-chunk for science) is straightforward but doubles the index footprint.

### Priority 8 — **Calibrate the `min_score` thresholds empirically**

Right now they're `None` by default and the notebook lists them as "calibrate empirically" without telling the student how. Add a script that:

1. Runs the current retriever on every gold-set question.
2. For each question, records the top-1 score and whether the *answer* was eventually correct.
3. Computes the score threshold that maximizes `gated_accuracy_when_ungated × ungated_fraction + bare_baseline_accuracy × gated_fraction`.

This is the calibration step you're missing. Without it, the gate is either always on or always off, neither of which uses the live-fallback machinery properly.

### Priority 9 — **Add a Wikipedia-categories-based negative filter**

Currently the category filter is positive ("only chunks tagged HISTORY"). Add a negative filter: when retrieving for HISTORY, **also** exclude chunks whose source article belongs to clearly off-topic Wikipedia categories. The MediaWiki API exposes each article's categories. Tag every chunk with its parent article's top-3 Wikipedia categories at build time; use them to penalize obvious off-topic hits at retrieval time.

### Priority 10 — **Minor code-quality items**

- **`_collect_existing_titles` in `IndexGrower`** reaches into `self._retriever._index._chunks` — three layers of private access. Add a `Retriever.iter_sources()` public method.
- **`live_search.py`** uses a daemon-thread timeout pattern which leaks threads on timeout (the worker keeps running in the background, then the result is discarded). For an assignment it's fine; in production, switch to `concurrent.futures.ThreadPoolExecutor` with `future.cancel()` semantics or use `asyncio` + `aiohttp` against the MediaWiki REST API directly.
- **The `_BARE_LETTER` parser** in `templates.py` takes the LAST match — this is wrong for `ELIMINATION` style where the model writes "A: …, B: …, C: …, D: …" and then `\boxed{B}`. The boxed pattern wins because it's structured-first, but if the model omits the box and just ends with "Therefore D", the LAST-match heuristic picks D correctly. Edge case worth a regression test.
- **`fusion.py` `weights` parameter** is plumbed but never used in the strategy. For category-conditional retrieval you may want to weight BM25 higher for entity-heavy categories (entertainment, history) and dense higher for conceptual ones (science).

---

## 4. What I would **not** recommend (and why)

Some things sound appealing but won't pay off on your constraints:

- **ColBERT / late-interaction retrieval.** Heavy index (~10× FAISS), heavy inference, and the gain over `cross-encoder reranker on a dense+BM25 fused pool` is small. Your reranker setup is already doing the precision-stage work ColBERT would do.
- **Larger LLM (Qwen 2.5 14B / 32B).** Your accuracy ceiling on this task is far more constrained by retrieval recall than by reasoning ability at 7B. Spend the VRAM budget on a better embedder + reranker first.
- **Fine-tuning the embedder.** You have no labeled query→passage pairs at scale, and 50 hand-labeled gold examples is below the threshold where contrastive fine-tuning is stable. Use a stronger off-the-shelf model instead.
- **Vector DB swap (Pinecone, Weaviate, Qdrant).** Your `FAISS IndexFlatIP` over ~50k chunks is fast (sub-10ms). The vector DB ecosystem solves scale problems you don't have.
- **GraphRAG / Knowledge-graph fusion.** Implementation cost for an assignment is huge; Wikidata SPARQL gives you 80% of the upside with 5% of the work.

---

## 5. Implementation roadmap (suggested order)

**Practical Ordering;**

1. Calibrate the three `min_score` thresholds on your existing gold set (Priority 8). Without this, you can't measure anything else cleanly.
2. Route MATHS through Tool/Agent (Priority 1, partial — the easy win). Measure delta.
3. Harvest category-graph corpus (Priority 2). Rebuild index. Re-run gold set. This is your big accuracy lift.
4. Swap to `bge-base-en-v1.5` (Priority 3). Drop-in change with the manifest checks.
5. Wikidata SPARQL integration for entertainment (Priorities 1 + 6). Wire as a new `WikidataStrategy` arm in the per-category router.
6. HyDE behind a flag (Priority 4). A/B against current. Keep if it wins on the medium/hard tier; reject if it doesn't.
7. Per-category chunk sizes (Priority 7) — only if time allows. Marginal vs. earlier items.

Run your `retrieval_dashboard` and `show_trace` infrastructure at every step — they're already excellent and will tell you instantly whether each change helped.

---

