"""Live Wikipedia search fallback for RAGStrategy.

Used when the offline FAISS/BM25 index yields a top score below the
configured ``min_score`` threshold — meaning the index doesn't have a good
answer for this question.  Rather than degrading silently to the bare LLM,
``LiveSearchFallback.search()`` fires a real-time Wikipedia API query and
returns fresh ``Article`` objects that can be formatted as context identical
to offline RAG passages.

Design constraints
──────────────────
- Hard wall-clock timeout (default 5 s).  The game server gives 30 s per
  question; a stray network call must never burn more than a small fraction
  of that budget.  On timeout or any network/API error the method returns
  ``[]`` — callers degrade gracefully without exceptions.
- Returns at most ``max_articles`` Wikipedia article *summaries* (not full
  pages).  Full pages can be 50 000 words; summaries are 3–10 sentences —
  enough for trivia context, much faster to fetch.
- Reuses ``clean_wikipedia_text()`` from ``corpus.py`` so the text is on the
  same level of cleanliness as the offline corpus (citation markers stripped,
  trailing meta-sections removed).
- Returns ``corpus.Article`` objects so ``IndexGrower`` can ingest them using
  exactly the same ``chunk_text`` pipeline as the original build script.

Usage
─────
    fallback = LiveSearchFallback(timeout_seconds=5.0, max_articles=2)
    articles = fallback.search("Julius Caesar Rubicon", category=Category.HISTORY)
    # → [Article(title="Julius Caesar", text="...", category=..., url="..."), ...]
    # or [] on any failure / no results
"""
from __future__ import annotations

import threading
from typing import Optional

from ..config import Category
from .corpus import Article, clean_wikipedia_text


class LiveSearchFallback:
    """Query Wikipedia API in real-time and return Article objects.

    Thread-safety: ``search()`` is re-entrant — each call runs its own
    daemon thread for the timeout guard, so multiple concurrent callers
    (e.g. in an evaluation loop) do not interfere.

    Args:
        timeout_seconds: hard wall-clock limit per ``search()`` call.
            If Wikipedia does not respond within this window the call
            returns ``[]``.  Default: 5 s.
        max_articles: maximum number of Wikipedia articles to fetch per
            query.  Each article requires one ``wikipedia.summary()`` call
            (fast) and optionally one ``wikipedia.page()`` call when
            ``use_summary_only=False`` (slower but richer context).
            Default: 2.
        use_summary_only: when True (default), fetch the Wikipedia intro
            summary instead of the full article.  Summaries are ~200–600
            words — ideal for single-question trivia context without the
            latency of a full-page download.
        search_results: how many candidate titles to ask Wikipedia for
            before filtering.  A wider net improves the chance of finding
            an on-topic page.  Default: 5.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        max_articles: int = 2,
        use_summary_only: bool = True,
        search_results: int = 5,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_articles = max_articles
        self.use_summary_only = use_summary_only
        self.search_results = search_results

    # ── Public API ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        category: Optional[Category] = None,
    ) -> list[Article]:
        """Search Wikipedia live for articles relevant to ``query``.

        The call is guarded by a daemon-thread timeout so it never blocks
        the game loop beyond ``self.timeout_seconds``.

        Args:
            query: free-text query string (typically the question text, or
                question + option text from RAGStrategy's multi-query path).
            category: when set, the returned Article objects are tagged with
                this category so ``IndexGrower`` stores them under the right
                category label and the retriever's category filter works for
                future queries.

        Returns:
            List of up to ``max_articles`` Article objects.  Returns ``[]``
            on timeout, network error, empty results, or import failure
            (wikipedia library not installed).
        """
        if not query or not query.strip():
            return []

        box: dict[str, object] = {}

        def _worker() -> None:
            try:
                box["result"] = self._fetch(query, category=category)
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(self.timeout_seconds)

        if th.is_alive():
            # Thread is still running — timeout.  It will eventually finish
            # in the background; we just discard the result.
            return []

        if "error" in box:
            return []

        return box.get("result", [])  # type: ignore[return-value]

    # ── Internal fetch logic ──────────────────────────────────────────────

    def _fetch(
        self,
        query: str,
        *,
        category: Optional[Category],
    ) -> list[Article]:
        """Perform the actual Wikipedia API calls (runs inside the daemon thread).

        Strategy:
          1. ``wikipedia.search(query, results=search_results)`` → candidate titles.
          2. For each title (up to ``max_articles``):
             a. ``wikipedia.summary(title)`` (fast — intro paragraph only).
             b. Optionally ``wikipedia.page(title).content`` for fuller text.
          3. Apply ``clean_wikipedia_text()`` for consistency with offline corpus.
          4. Wrap in ``Article`` with the supplied category tag.

        Disambiguation errors are handled by trying the next candidate in the
        list; ``PageError`` is skipped.  Any single-article failure does not
        abort the whole batch.
        """
        try:
            import wikipedia  # lazy — only needed at game time when triggered
        except ImportError:
            return []

        wikipedia.set_lang("en")

        try:
            titles = wikipedia.search(query, results=self.search_results)
        except Exception:  # noqa: BLE001
            return []

        if not titles:
            return []

        articles: list[Article] = []
        for title in titles:
            if len(articles) >= self.max_articles:
                break
            article = self._fetch_one(title, category=category)
            if article is not None:
                articles.append(article)

        return articles

    def _fetch_one(
        self,
        title: str,
        *,
        category: Optional[Category],
    ) -> Optional[Article]:
        """Fetch a single Wikipedia article (summary or full page).

        Returns ``None`` on any failure — the caller skips and tries the next
        candidate title.
        """
        try:
            import wikipedia
        except ImportError:
            return None

        try:
            if self.use_summary_only:
                # ``auto_suggest=False`` prevents Wikipedia from silently
                # redirecting to a different article (e.g. typo correction
                # that picks a completely different entity).
                text = wikipedia.summary(title, auto_suggest=False)
                url  = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            else:
                page = wikipedia.page(title, auto_suggest=False)
                text = page.content
                url  = page.url

            text = clean_wikipedia_text(text)
            if not text.strip():
                return None

            return Article(
                title=title,
                text=text,
                category=category if category is not None else Category.HISTORY,
                url=url,
            )

        except Exception:  # noqa: BLE001 — PageError, DisambiguationError, network
            return None
