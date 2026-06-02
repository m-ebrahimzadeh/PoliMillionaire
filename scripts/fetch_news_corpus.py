"""Harvest recent Guardian news into the offline RAG corpus / index.

The NEWS category's offline corpus is otherwise Wikipedia-only and sparse
(``category_seeds.py`` calls News "the weakest category for Wikipedia").  This
script pulls a date range of Guardian articles — full body text — so the
offline index gains real, recent News coverage and a network-free fallback.
The online ``NewsLiveSearch`` fallback and the self-growing ``IndexGrower``
keep it fresh after that.

Usage
-----
    export GUARDIAN_API_KEY=...          # free key: open-platform.theguardian.com
    python scripts/fetch_news_corpus.py --days 30                 # fetch -> corpus.jsonl
    python scripts/fetch_news_corpus.py --days 30 --build         # fetch -> corpus.jsonl + index
    python scripts/fetch_news_corpus.py --from 2026-05-01 --to 2026-05-31 \
        --sections world,uk-news,business --build

Modes
-----
- **Fetch-only** (default): needs only ``requests``.  Appends new articles to
  ``data/cache/corpus.jsonl`` (dedup by title) and prints the build command.
- **--build**: additionally chunks / embeds / appends to the FAISS + BM25
  index by reusing ``IndexGrower``.  Requires an existing index (run
  ``build_rag_index.py`` first) and the ``rag`` extras (sentence-transformers).
"""
from __future__ import annotations

import argparse
import datetime as _dt

from polimibot.config import NEWS, PATHS
from polimibot.rag.corpus import Article, append_raw_corpus
from polimibot.rag.news_search import harvest_news_range


def main() -> None:
    args = _parse_args()
    PATHS.ensure()

    from_date, to_date = _date_window(args)
    print(f"Harvesting Guardian articles {from_date} → {to_date}"
          + (f"  sections={args.sections}" if args.sections else "")
          + (f"  q={args.query!r}" if args.query else ""))

    if not NEWS.guardian_api_key:
        print(
            "WARNING: GUARDIAN_API_KEY is not set, so the Guardian source skips "
            "the network and this harvest will fetch nothing. Register a free "
            "key at open-platform.theguardian.com and `export "
            "GUARDIAN_API_KEY=...` before running."
        )

    # Harvest day-by-day (see harvest_news_range) so every date in the window
    # gets its own pagination budget and the older end of the range is not
    # silently dropped — that is exactly where the dated News questions live.
    unique = harvest_news_range(
        from_date, to_date,
        query=args.query,
        sections=args.sections,
        page_size=args.page_size,
        max_pages=args.max_pages,
        verbose=True,
    )

    if not unique:
        print("Nothing fetched — check the key, date window, or query and retry.")
        return

    if args.build:
        _build_index(unique)
    else:
        n = append_raw_corpus(unique, PATHS.cache_dir / "corpus.jsonl")
        print(f"Appended {n} new article(s) to {PATHS.cache_dir / 'corpus.jsonl'}.")
        print("Next: build the index with —")
        print("    python scripts/build_rag_index.py --fresh        # first time (all categories)")
        print("    python scripts/fetch_news_corpus.py ... --build   # or index news directly")


# ── Index build (reuses IndexGrower) ─────────────────────────────────────────

def _build_index(articles: list[Article]) -> None:
    """Chunk, embed and append ``articles`` to the existing FAISS + BM25 index.

    Reuses ``IndexGrower`` end-to-end (buffer → confirm → flush), which also
    appends the raw articles to ``corpus.jsonl`` — so the index, the BM25
    sidecar and the corpus all stay in sync with one code path.
    """
    index_path = PATHS.cache_dir / "knowledge"
    if not index_path.with_suffix(".faiss").exists():
        print(
            f"No index at {index_path}.faiss. Build the base index first:\n"
            "    python scripts/build_rag_index.py --fresh\n"
            "then re-run with --build to add the harvested news on top."
        )
        return

    # Heavy imports (torch / sentence-transformers) only on the --build path.
    from polimibot.rag.embedder import Embedder
    from polimibot.rag.index_grower import IndexGrower
    from polimibot.rag.retriever import Retriever

    print(f"Loading existing index from {index_path}.faiss …")
    retriever = Retriever.from_saved(index_path)
    embedder = Embedder(retriever._embedder.spec)   # same space as the index
    grower = IndexGrower(
        retriever, embedder, index_path,
        corpus_path=PATHS.cache_dir / "corpus.jsonl",
    )

    added = 0
    for i, article in enumerate(articles):
        qid = f"news_harvest_{i}"
        chunks = grower.buffer(article, qid)
        if chunks:                       # not already indexed
            grower.confirm(qid)          # embed + append in-memory
            added += 1
        else:
            grower.discard(qid)
    grower.flush()                       # persist FAISS + BM25 + corpus.jsonl
    print(f"Indexed {added} new news article(s); index now has "
          f"{retriever.n_chunks} chunks total.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _date_window(args: argparse.Namespace) -> tuple[_dt.date, _dt.date]:
    if args.from_date or args.to_date:
        to_date = _dt.date.fromisoformat(args.to_date) if args.to_date else _dt.date.today()
        from_date = _dt.date.fromisoformat(args.from_date) if args.from_date else to_date - _dt.timedelta(days=args.days)
    else:
        to_date = _dt.date.today()
        from_date = to_date - _dt.timedelta(days=args.days)
    return from_date, to_date


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest Guardian news into the RAG corpus/index.")
    p.add_argument("--days", type=int, default=30,
                   help="Days back from today to harvest (ignored if --from/--to given).")
    p.add_argument("--from", dest="from_date", default=None,
                   help="Inclusive start date YYYY-MM-DD (overrides --days).")
    p.add_argument("--to", dest="to_date", default=None,
                   help="Inclusive end date YYYY-MM-DD (defaults to today).")
    p.add_argument("--sections", default=None,
                   help="Comma-separated Guardian section ids, e.g. 'world,uk-news,business'.")
    p.add_argument("--query", default=None,
                   help="Optional full-text filter; omit to harvest everything in the window.")
    p.add_argument("--page-size", type=int, default=50, dest="page_size",
                   help="Results per page (Guardian max 50).")
    p.add_argument("--max-pages", type=int, default=20, dest="max_pages",
                   help="Safety cap on pages fetched (max_pages * page_size articles).")
    p.add_argument("--build", action="store_true",
                   help="Also chunk/embed/append into the FAISS+BM25 index (needs rag extras + an existing index).")
    return p.parse_args()


if __name__ == "__main__":
    main()
