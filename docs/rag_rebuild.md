# RAG index rebuild & enrichment runbook

This is the operational guide for the `feat/rag-rebuild-enrich` work: rebuild the
knowledge index from scratch on Google Colab with the enriched corpus and the
`bge-m3` embedder, recalibrate the gate, and **prove** the change improves
performance with a before/after measurement.

The index itself is built on Colab (GPU embedding + the multi-thousand-article
Wikipedia harvest); these steps assume the Colab notebook and Google Drive
mount already used by the project.

## What changed (and why)

| Area | Before | After |
|---|---|---|
| Live-search query | returned `"Thinking Process:"` for every gated question (qwen3.5-4b) → fallback dead | model-agnostic distillation, falls back to the question |
| Live-search robustness | `simplejson.JSONDecodeError` retried a throttled endpoint; `with ThreadPoolExecutor` blocked past the timeout | rate-limit detected by class name; `shutdown(wait=False)` bounds wall-clock |
| Embedder | `bge-small-en-v1.5` (384-d) | `bge-m3` (1024-d); M3 prefix bug fixed |
| Embedding grounding | `"title: text"` (lead-only section header) | `"title (also known as: …) — section: text"` on every chunk |
| Corpus seeds | entity-only | entity **+** concept seeds (depth-1) + 222 explicit `CONCEPT_TITLES` + log-mined gaps |
| Schema | flat `Article` | `+aliases` (Wikipedia redirects) `+competition` |

## 1. Mine the gap queue (local or Colab, from run logs)

Put the run JSONL logs (the ones that recorded `extras.top_score` /
`gated_by_min_score` / `correct`) in `data/runs/`, then:

```bash
python scripts/mine_corpus_gaps.py data/runs/*.jsonl \
    --out data/cache/gap_titles.json --resolve
```

`--resolve` canonicalises candidate phrases to real article titles via Wikipedia
search (online). Drop it to emit raw phrases. NEWS and Maths are excluded.

## 2. Rebuild the index from scratch (Colab, Section 0.4)

Section 0.4 is split into two cells so the GPU-free harvest and the GPU embed
can run on different runtimes. Set the shared knobs cell, then run **0.4a** then
**0.4b**:

```python
REBUILD_INDEX      = True
INDEX_REFETCH      = True
EMBEDDER_MODEL     = 'BAAI/bge-m3'          # already the default after this branch
INDEX_GAP_QUEUE    = 'data/cache/gap_titles.json'   # or None
# INDEX_HARVEST_MAX_DEPTH stays 0 for entity seeds; concept seeds always recurse 1 level.
```

- **0.4a — Harvest corpus (CPU runtime).** Fetches Wikipedia → `data/cache/corpus.jsonl`,
  checkpointing as it goes and saving *before* the gap-queue fetch. Pure
  network/CPU — no GPU hours burned.
- Switch to a **GPU runtime** (re-run 0.1–0.3 + the knobs cell).
- **0.4b — Embed & index (GPU runtime).** Loads `corpus.jsonl`, chunks, embeds
  with `bge-m3`, writes the FAISS + BM25 index.

`data/` is symlinked to Drive by cell 1 (or you work directly from Drive), so
`corpus.jsonl` and the index already persist across the runtime switch — no
explicit copy step. To stay in one runtime, just run 0.4a then 0.4b back to back.

> Why the split: a single cell forced the CPU-only harvest onto a GPU runtime, and
> a crash in the gap phase (which ran before the corpus was saved) discarded the
> entire download. 0.4a now makes the harvest durable before anything else.

Equivalent CLI (if not using the notebook cells — fetches + embeds in one run):

```bash
python scripts/build_rag_index.py --fresh --refetch \
    --model BAAI/bge-m3 --gap-queue data/cache/gap_titles.json
```

Writes `data/cache/knowledge.{faiss,jsonl,manifest.json,bm25.jsonl}`. Persist to
Drive. **Sanity check** the manifest: `embedder_model_name = BAAI/bge-m3`,
`embedder_dim = 1024`, both prefixes `""`, `corpus_version = 4`, chunk count far
above the previous build.

### Chunk-size sweep (pick the size that ships)

Cross-encoders reward focused passages; rebuild at a couple of sizes and keep the
best by measured Recall@k/MRR (step 4):

```bash
for SZ in 180 220 256; do
  python scripts/build_rag_index.py --fresh --chunk-size $SZ --overlap 50 \
      --model BAAI/bge-m3 --gap-queue data/cache/gap_titles.json
  python scripts/eval_rag_delta.py   # record Recall@k / MRR for this size
done
```

## 3. Recalibrate the gate (Colab §2.8c)

The reranker score distribution changed (m3 + bge-reranker-v2-m3 + enriched
corpus), so the old `RAG_MIN_SCORE_RERANK` is meaningless. Run the in-notebook
calibration (§2.8c) or:

```bash
python scripts/calibrate_min_score.py --path rerank   # → suggested RAG_MIN_SCORE_RERANK
```

Set `RAG_MIN_SCORE_RERANK` (and re-check `MIN_LIVE_SCORE` on the new scale) to
the value that maximises gated-policy accuracy. Leaving it `None` keeps offline
RAG always-on (fine now that the corpus has the content); set it when you want
the live fallback to fire on genuine gaps.

## 4. Prove it improves (before/after)

Build the measurement set once from the logs, then compare old vs new index:

```bash
python scripts/build_gold_set.py   --runs data/runs/ --out data/eval/gold_set.jsonl
python scripts/build_wrong_set.py  --runs data/runs/ --out data/eval/wrong_set.jsonl
python scripts/eval_rag.py         # end-to-end accuracy on the gold set
python scripts/eval_rag_delta.py   # Recall@k / MRR — does the right article surface?
```

Targets:
- The previously-gated conceptual misses now retrieve their source above the
  calibrated threshold: *Necessity and sufficiency*, *Demography of the Roman
  Empire*, *Achtung Baby*, *The One Where Dr. Ramoray Dies* (via the
  "Dr. Drake Ramoray" alias), *Chromesthesia* + absolute pitch, *Weathering*,
  *Entropy*, *Generative adversarial network*.
- End-to-end accuracy delta per competition is **non-negative**; live-search
  latency stays well under the 25 s budget (no more dead "Thinking Process:"
  fetches). Open the PR only once these hold.
