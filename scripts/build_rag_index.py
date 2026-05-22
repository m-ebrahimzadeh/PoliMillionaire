"""Build (or rebuild) the RAG knowledge index.

Workflow
--------
1.  If data/cache/corpus.jsonl exists → load it (skip Wikipedia fetch).
    Pass --refetch to force a fresh download.
2.  Chunk every article.
3.  Embed all chunks (sentence-transformers, batch=64).
4.  Save FAISS index to data/cache/knowledge.{faiss,jsonl}.

Usage
-----
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --refetch
    python scripts/build_rag_index.py --categories history science
    python scripts/build_rag_index.py --chunk-size 250 --overlap 40
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from polimibot.config import PATHS, Category
from polimibot.rag.bm25 import BM25Index
from polimibot.rag.chunker import CHUNKER_VERSION, chunk_text
from polimibot.rag.corpus import (
    CLEANUP_VERSION, CORPUS_VERSION, clean_wikipedia_text,
    fetch_articles, fetch_articles_from_categories,
    load_raw_corpus, save_raw_corpus,
)
from polimibot.rag.embedder import Embedder, EmbedderSpec
from polimibot.rag.index import FAISSIndex


def main() -> None:
    args = _parse_args()
    PATHS.ensure()

    corpus_path  = PATHS.cache_dir / "corpus.jsonl"
    index_path   = PATHS.cache_dir / "knowledge"   # .faiss + .jsonl appended by FAISSIndex

    if args.skip_if_exists and index_path.with_suffix(".faiss").exists():
        print(f"Index already exists at {index_path}.faiss — skipping (remove --skip-if-exists to rebuild).")
        return

    # ── Step 1: corpus ───────────────────────────────────────────────────────
    categories = (
        [Category(c) for c in args.categories] if args.categories
        else None  # None → all four
    )

    if corpus_path.exists() and not args.refetch:
        print(f"Loading cached corpus from {corpus_path} (--refetch to overwrite)…")
        articles = load_raw_corpus(corpus_path)
        if categories:
            articles = [a for a in articles if a.category in categories]
        # Defensive cleanup pass for corpora cached before clean_wikipedia_text
        # existed. Idempotent — already-clean articles pass through unchanged.
        from dataclasses import replace
        articles = [replace(a, text=clean_wikipedia_text(a.text)) for a in articles]
    else:
        if args.legacy_seeds:
            print("Fetching articles from Wikipedia (legacy hand-curated TOPIC_SEEDS)…")
            articles = fetch_articles(categories=categories, verbose=True)
        else:
            print("Fetching articles from Wikipedia (category-graph harvest)…")
            articles = fetch_articles_from_categories(
                categories=categories,
                cache_path=PATHS.cache_dir / "harvested_titles.json",
                max_per_category=args.max_per_category,
                max_depth=args.max_depth,
                verbose=True,
            )
        save_raw_corpus(articles, corpus_path)

    if not articles:
        print("No articles fetched — aborting.")
        return

    # ── Step 2: chunk ────────────────────────────────────────────────────────
    print(f"\nChunking {len(articles)} articles "
          f"(size={args.chunk_size}, overlap={args.overlap})…")
    all_chunks = []
    for article in articles:
        chunks = chunk_text(
            article.text,
            source=article.title,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            category=article.category.value,
        )
        all_chunks.extend(chunks)
    print(f"  → {len(all_chunks)} chunks total "
          f"(avg {len(all_chunks)//max(len(articles),1)} per article)")

    # ── Step 3: embed ────────────────────────────────────────────────────────
    print("\nLoading embedding model…")
    spec = EmbedderSpec(model_name=args.model)
    embedder = Embedder(spec)
    print(f"  dim={embedder.dim}  batch={spec.batch_size}")

    print("Embedding…")
    t0 = time.monotonic()
    texts = [c.text for c in all_chunks]
    embeddings = embedder.encode_passage(texts)
    print(f"  → done in {time.monotonic()-t0:.1f}s")

    # ── Step 4: build + save index ───────────────────────────────────────────
    idx = FAISSIndex(dim=embedder.dim)
    idx.add(all_chunks, embeddings)
    idx.save(index_path, manifest={
        "embedder_model_name":     spec.model_name,
        "embedder_dim":            embedder.dim,
        "embedder_query_prefix":   spec.query_prefix,
        "embedder_passage_prefix": spec.passage_prefix,
        "normalize":               spec.normalize,
        "chunk_size":              args.chunk_size,
        "chunk_overlap":           args.overlap,
        "chunker_version":         CHUNKER_VERSION,
        "corpus_version":          CORPUS_VERSION,
        "corpus_source":           "hand_curated" if args.legacy_seeds else "category_graph",
        "max_per_category":        args.max_per_category,
        "max_depth":               args.max_depth,
        "n_articles":              len(articles),
        "text_cleanup_version":    CLEANUP_VERSION,
        "categories":              sorted({a.category.value for a in articles}),
    })

    print(f"\n✓  FAISS index ready at {index_path}.{{faiss,jsonl,manifest.json}}")
    print(f"   {idx.n_chunks} chunks  |  dim={embedder.dim}  |  model={args.model}")

    # ── Step 5: BM25 sidecar (skip with --no-bm25) ──────────────────────────
    if not args.no_bm25:
        print("\nBuilding BM25 index over the same chunks…")
        t0 = time.monotonic()
        bm25 = BM25Index(all_chunks)
        bm25.save(index_path)   # writes {stem}.bm25.jsonl
        print(f"   built in {time.monotonic()-t0:.1f}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--refetch", action="store_true",
                   help="Re-download Wikipedia articles even if corpus.jsonl exists")
    p.add_argument("--categories", nargs="+",
                   choices=[c.value for c in Category],
                   help="Subset of categories to build (default: all)")
    p.add_argument("--chunk-size", type=int, default=300,
                   dest="chunk_size")
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--model", default="all-MiniLM-L6-v2",
                   help="Sentence-transformers model name")
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
                        "category only (safe default); 1 = one level of subcats.")
    return p.parse_args()


if __name__ == "__main__":
    main()