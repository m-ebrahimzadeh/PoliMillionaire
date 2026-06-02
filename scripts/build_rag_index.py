"""Build (or incrementally update) the RAG knowledge index.

Workflow
--------
1.  If index exists and no --fresh flag → load existing index for incremental update.
2.  If data/cache/corpus.jsonl exists and not --refetch → load cached corpus.
    Pass --refetch to force a fresh download.
3.  Fetch new articles using category-graph harvest (or legacy seeds).
4.  Deduplicate: skip articles already in the index.
5.  Chunk only NEW articles.
6.  Embed new chunks (sentence-transformers, batch=64).
7.  Append to existing index (or build from scratch if --fresh).
8.  Save FAISS index to data/cache/knowledge.{faiss,jsonl}.

Usage
-----
    python scripts/build_rag_index.py                 # incremental update (default)
    python scripts/build_rag_index.py --fresh         # full rebuild from scratch
    python scripts/build_rag_index.py --refetch       # re-fetch + incremental update
    python scripts/build_rag_index.py --categories history science
    python scripts/build_rag_index.py --chunk-size 250 --overlap 40

Key behavior
------------
- **Default (no flags)**: Loads existing index, fetches new articles with the
  better category-graph approach, deduplicates against existing content, and
  only adds new articles. Preserves all gameplay-learned articles.
- **--fresh**: Completely wipes and rebuilds from scratch (loses gameplay learning).
- **--skip-if-exists**: Exits early if index exists (prevents any update).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from polimibot.config import PATHS, Category
from polimibot.rag.bm25 import BM25Index
from polimibot.rag.chunker import (
    CHUNKER_VERSION, EMBED_TEXT_VERSION, chunk_text, embedding_text,
)
from polimibot.rag.corpus import (
    CLEANUP_VERSION, CORPUS_VERSION, clean_wikipedia_text,
    fetch_articles, fetch_articles_by_title, fetch_articles_from_categories,
    load_raw_corpus, save_raw_corpus,
)
from polimibot.rag.embedder import Embedder, EmbedderSpec
from polimibot.rag.index import FAISSIndex
from polimibot.rag.retriever import Retriever


def main() -> None:
    args = _parse_args()
    PATHS.ensure()

    corpus_path  = PATHS.cache_dir / "corpus.jsonl"
    index_path   = PATHS.cache_dir / "knowledge"   # .faiss + .jsonl appended by FAISSIndex

    # Determine mode: fresh rebuild vs incremental update
    do_fresh = args.fresh or not index_path.with_suffix(".faiss").exists()

    if args.skip_if_exists and not do_fresh:
        print(f"Index already exists at {index_path}.faiss — skipping (remove --skip-if-exists to rebuild/update).")
        return

    # ── Step 1: Load existing index (incremental mode) or start fresh ───────
    existing_titles: set[str] = set()
    existing_manifest: dict | None = None

    if not do_fresh:
        print(f"Loading existing index from {index_path}.faiss for incremental update…")
        retriever = Retriever.from_saved(index_path)
        existing_titles = retriever.iter_sources()
        existing_manifest = retriever._index.manifest
        print(f"  → {retriever.n_chunks} existing chunks from {len(existing_titles)} articles")
    else:
        print("Building index from scratch (--fresh or no existing index)…")

    # ── Step 2: corpus ───────────────────────────────────────────────────────
    categories = (
        [Category(c) for c in args.categories] if args.categories
        else None  # None → all four
    )

    if corpus_path.exists() and not args.refetch:
        print(f"Loading cached corpus from {corpus_path} (--refetch to overwrite)…")
        cached_articles = load_raw_corpus(corpus_path)
        if categories:
            cached_articles = [a for a in cached_articles if a.category in categories]
        # Defensive cleanup pass for corpora cached before clean_wikipedia_text
        # existed. Idempotent — already-clean articles pass through unchanged.
        from dataclasses import replace
        cached_articles = [replace(a, text=clean_wikipedia_text(a.text)) for a in cached_articles]
    else:
        cached_articles = []

    # Fetch new articles
    if args.legacy_seeds:
        print("Fetching articles from Wikipedia (legacy hand-curated TOPIC_SEEDS)…")
        fetched_articles = fetch_articles(categories=categories, verbose=True)
    else:
        print("Fetching articles from Wikipedia (category-graph harvest)…")
        fetched_articles = fetch_articles_from_categories(
            categories=categories,
            cache_path=PATHS.cache_dir / "harvested_titles.json",
            max_per_category=args.max_per_category,
            max_depth=args.max_depth,
            # Durable partial harvest: a crash mid-crawl leaves a usable corpus.
            checkpoint_path=PATHS.cache_dir / "corpus.partial.jsonl",
            verbose=True,
        )

    # Gap queue: fetch the log-mined back-fill titles directly (bypasses the
    # category crawl). Skips titles already in the index or just fetched.
    if args.gap_queue:
        import json as _json
        gap_path = Path(args.gap_queue)
        if not gap_path.is_file():
            print(f"  ! --gap-queue {gap_path} not found — skipping gap back-fill")
        else:
            raw = _json.loads(gap_path.read_text(encoding="utf-8"))
            titles_by_cat = {}
            for val, titles in raw.items():
                try:
                    titles_by_cat[Category(val)] = list(titles)
                except ValueError:
                    continue  # unknown category value in the queue file
            if categories:
                titles_by_cat = {c: t for c, t in titles_by_cat.items() if c in categories}
            already = existing_titles | {a.title for a in fetched_articles}
            gap_articles = fetch_articles_by_title(
                titles_by_cat, existing_titles=already, verbose=True,
            )
            print(f"Gap queue added {len(gap_articles)} new articles from {gap_path.name}")
            fetched_articles = fetched_articles + gap_articles

    # Deduplicate: keep only articles not already in the index
    new_articles = [a for a in fetched_articles if a.title not in existing_titles]
    print(f"\nFetched {len(fetched_articles)} articles, {len(new_articles)} are new (not in existing index)")

    # Combine: use fetched articles (new ones for incremental, all for fresh)
    # In incremental mode, we only process new articles; existing ones stay in index
    articles_to_process = new_articles if not do_fresh else fetched_articles

    if not articles_to_process:
        if do_fresh:
            print("No articles fetched — aborting.")
            return
        print("No new articles to add — index is up to date.")
        # Still save to update manifest timestamp if needed
        if existing_manifest and retriever._index.manifest:
            retriever._index.save(index_path, manifest=retriever._index.manifest)
        return

    # ── Step 3: chunk ────────────────────────────────────────────────────────
    print(f"\nChunking {len(articles_to_process)} new articles "
          f"(size={args.chunk_size}, overlap={args.overlap})…")
    new_chunks = []
    for article in articles_to_process:
        chunks = chunk_text(
            article.text,
            source=article.title,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            category=article.category.value,
            url=article.url,
            aliases=article.aliases or None,
        )
        new_chunks.extend(chunks)
    print(f"  → {len(new_chunks)} new chunks total "
          f"(avg {len(new_chunks)//max(len(articles_to_process),1)} per article)")

    # ── Step 4: embed ────────────────────────────────────────────────────────
    print("\nLoading embedding model…")
    spec = EmbedderSpec(model_name=args.model)
    embedder = Embedder(spec)
    print(f"  dim={embedder.dim}  batch={spec.batch_size}")

    print("Embedding new chunks…")
    t0 = time.monotonic()
    # Ground each passage in its source title before embedding (see
    # chunker.embedding_text). Chunk.text stays pure for display + BM25.
    texts = [embedding_text(c) for c in new_chunks]
    new_embeddings = embedder.encode_passage(texts)
    print(f"  → done in {time.monotonic()-t0:.1f}s")

    # ── Step 5: build/merge + save index ─────────────────────────────────────
    if do_fresh:
        # Fresh build: create new index from scratch
        idx = FAISSIndex(dim=embedder.dim)
        idx.add(new_chunks, new_embeddings)
        total_chunks = len(new_chunks)
    else:
        # Incremental: append to existing index
        retriever.append_chunks(new_chunks, new_embeddings)
        idx = retriever._index
        total_chunks = idx.n_chunks
        print(f"\nAppended {len(new_chunks)} chunks to existing index")

    # Build manifest
    all_article_titles = existing_titles | {a.title for a in articles_to_process}
    idx.save(index_path, manifest={
        "embedder_model_name":     spec.model_name,
        "embedder_dim":            embedder.dim,
        "embedder_query_prefix":   spec.query_prefix,
        "embedder_passage_prefix": spec.passage_prefix,
        "normalize":               spec.normalize,
        "chunk_size":              args.chunk_size,
        "chunk_overlap":           args.overlap,
        "chunker_version":         CHUNKER_VERSION,
        "embed_text_version":      EMBED_TEXT_VERSION,
        "corpus_version":          CORPUS_VERSION,
        "corpus_source":           "hand_curated" if args.legacy_seeds else "category_graph",
        "max_per_category":        args.max_per_category,
        "max_depth":               args.max_depth,
        "n_articles":              len(all_article_titles),
        "text_cleanup_version":    CLEANUP_VERSION,
        "categories":              sorted({a.category.value for a in (articles_to_process if do_fresh else list(fetched_articles))}),
        "build_mode":              "fresh" if do_fresh else "incremental",
        "previous_n_chunks":       len(existing_titles) if not do_fresh else 0,
    })

    print(f"\n✓  FAISS index ready at {index_path}.{{faiss,jsonl,manifest.json}}")
    print(f"   {total_chunks} total chunks  |  dim={embedder.dim}  |  model={args.model}")
    if not do_fresh:
        print(f"   (+{len(new_chunks)} new chunks from {len(new_articles)} new articles)")

    # ── Step 6: BM25 sidecar (skip with --no-bm25) ──────────────────────────
    if not args.no_bm25:
        if do_fresh:
            print("\nBuilding BM25 index over the same chunks…")
            t0 = time.monotonic()
            bm25 = BM25Index(new_chunks)
            bm25.save(index_path)
            print(f"   built in {time.monotonic()-t0:.1f}s")
        else:
            # Incremental: rebuild BM25 with all chunks (existing + new)
            print("\nRebuilding BM25 index with all chunks (existing + new)…")
            t0 = time.monotonic()
            # Get all chunks from the updated retriever
            all_chunks_for_bm25 = list(retriever._index._chunks)
            bm25 = BM25Index(all_chunks_for_bm25)
            bm25.save(index_path)
            print(f"   built {bm25.n_chunks} docs in {time.monotonic()-t0:.1f}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--refetch", action="store_true",
                   help="Re-download Wikipedia articles even if corpus.jsonl exists")
    p.add_argument("--fresh", action="store_true",
                   help="Force a full rebuild from scratch (ignore existing index)")
    p.add_argument("--categories", nargs="+",
                   choices=[c.value for c in Category],
                   help="Subset of categories to build (default: all)")
    p.add_argument("--chunk-size", type=int, default=300,
                   dest="chunk_size")
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--model", default=EmbedderSpec().model_name,
                   help="Sentence-transformers model name (default matches EmbedderSpec)")
    p.add_argument("--skip-if-exists", action="store_true", dest="skip_if_exists",
                   help="Exit early if the FAISS index already exists on disk")
    p.add_argument("--no-bm25", action="store_true", dest="no_bm25",
                   help="Skip building the BM25 sidecar (dense-only build)")
    p.add_argument("--legacy-seeds", action="store_true", dest="legacy_seeds",
                   help="Use the hand-curated TOPIC_SEEDS (~95 articles) instead of "
                        "the category-graph harvest. Useful for fast iteration / tests.")
    p.add_argument("--max-per-category", type=int, default=500,
                   dest="max_per_category",
                   help="Cap on titles per seed-category during harvest (default 500)")
    p.add_argument("--max-depth", type=int, default=0, dest="max_depth",
                   help="Subcategory recursion depth for the harvester. 0 = this "
                        "category only (safe default); 1 = one level of subcats. "
                        "Concept seeds always recurse one level regardless.")
    p.add_argument("--gap-queue", default=None, dest="gap_queue",
                   help="Path to a gap_titles.json (from scripts/mine_corpus_gaps.py) "
                        "whose titles are fetched directly and added to the corpus.")
    return p.parse_args()


if __name__ == "__main__":
    main()