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
from polimibot.rag.chunker import chunk_text
from polimibot.rag.corpus import (
    fetch_articles, load_raw_corpus, save_raw_corpus,
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
    else:
        print("Fetching articles from Wikipedia…")
        articles = fetch_articles(categories=categories, verbose=True)
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
    embeddings = embedder.encode(texts)
    print(f"  → done in {time.monotonic()-t0:.1f}s")

    # ── Step 4: build + save index ───────────────────────────────────────────
    idx = FAISSIndex(dim=embedder.dim)
    idx.add(all_chunks, embeddings)
    idx.save(index_path)

    print(f"\n✓  Index ready at {index_path}.{{faiss,jsonl}}")
    print(f"   {idx.n_chunks} chunks  |  dim={embedder.dim}  |  model={args.model}")


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
    return p.parse_args()


if __name__ == "__main__":
    main()