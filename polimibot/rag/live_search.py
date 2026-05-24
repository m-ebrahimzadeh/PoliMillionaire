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

import concurrent.futures as _cf
from typing import Optional

from ..config import Category
from .corpus import Article, clean_wikipedia_text


class LiveSearchFallback:
    """Query Wikipedia API in real-time and return Article objects.

    Thread-safety: ``search()`` is re-entrant — each call uses its own
    one-worker ``ThreadPoolExecutor`` for the timeout guard, so multiple
    concurrent callers (e.g. in an evaluation loop) do not interfere and
    no leaked daemon threads pile up across calls.

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

        The call is guarded by a ``concurrent.futures`` executor timeout
        so it never blocks the game loop beyond ``self.timeout_seconds``.

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

        # ``future.cancel()`` can't preempt an in-flight HTTP call (Python
        # offers no cross-thread interrupt for native code), but the
        # executor's context-manager exit cleans up references promptly
        # and a fresh worker is created on the next call — no leaked
        # daemon threads as in the prior implementation.
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._fetch, query, category=category)
            try:
                return future.result(timeout=self.timeout_seconds)
            except _cf.TimeoutError:
                future.cancel()
                print(f"[live_search] TIMEOUT after {self.timeout_seconds}s on query={query!r}")
                return []
            except Exception as _exc:   # noqa: BLE001
                # Diagnostic: surface the real exception class so we can
                # distinguish HTTPError / ChunkedEncodingError / SSL blip /
                # library bug / true no-result from each other. Otherwise
                # every failure looks like "no articles found", in distinguishable, this is not.
                print(
                    f"[live_search] EXCEPTION in search() wrapper: "
                    f"{type(_exc).__name__}: {_exc}  | query={query!r}"
                )
                return []

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
        import time
        try:
            import wikipedia  # lazy — only needed at game time when triggered
        except ImportError:
            return []

        wikipedia.set_lang("en")

        # Single-retry on transient failures of ``wikipedia.search()``.
        # The diagnostic logs identified ``JSONDecodeError`` clusters
        # (empty HTTP body on rate-limit) as the dominant search-call
        # failure mode. A 1 s sleep typically refills Wikipedia's token
        # bucket enough to let the retry through. Truly bad queries (no
        # such article) reach the ``EMPTY`` branch below, not this one.
        titles: list = []
        for attempt in range(2):
            try:
                titles = wikipedia.search(query, results=self.search_results)
                break
            except Exception as _exc:  # noqa: BLE001
                is_last = (attempt == 1)
                if is_last:
                    print(
                        f"[live_search] EXCEPTION in wikipedia.search() "
                        f"(after retry): {type(_exc).__name__}: {_exc}  "
                        f"| query={query!r}"
                    )
                    return []
                print(
                    f"[live_search] RETRY wikipedia.search() after "
                    f"{type(_exc).__name__}: {_exc}  | query={query!r}"
                )
                time.sleep(1.0)

        if not titles:
            # Empty but no exception — Wikipedia legitimately found nothing.
            # Logged so we can distinguish "real empty" from "exception-as-empty".
            print(f"[live_search] EMPTY (no exception) for query={query!r}")
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

        Two recovery mechanisms, beyond bare try/except, this method has:

        1. ``DisambiguationError``: the library's exception exposes
           ``exc.options`` — a list of candidate titles. Retry with the
           first option (MediaWiki's relevance-sorted top choice). Saves
           common cases like ``"Achaeans"`` → ``"Achaeans (Homer)"``,
           ``"Temple of Olympian Zeus"`` → ``"Temple of Olympian Zeus, Athens"``.

        2. Transient network/parse failures (``JSONDecodeError`` on empty
           body when rate-limited, ``HTTPError``, ``ConnectionError``):
           sleep 1 s and retry once. Most rate-limit clusters refill
           within a couple of seconds, so a single short retry recovers
           the majority. ``PageError`` is treated as terminal — retrying
           a non-existent page won't conjure it into existence.

        Returns ``None`` on terminal failure; caller skips to next title.
        """
        import time
        try:
            import wikipedia
        except ImportError:
            return None

        def _do_summary(t: str):
            if self.use_summary_only:
                # ``auto_suggest=False`` prevents Wikipedia from silently
                # redirecting to a different article (e.g. typo correction
                # that picks a completely different entity).
                txt = wikipedia.summary(t, auto_suggest=False)
                u   = f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}"
            else:
                page = wikipedia.page(t, auto_suggest=False)
                txt  = page.content
                u    = page.url
            return txt, u

        fetched_title = title
        text: Optional[str] = None
        url: Optional[str] = None
        for attempt in range(2):
            try:
                text, url = _do_summary(fetched_title)
                break
            except wikipedia.exceptions.DisambiguationError as _exc:
                # Pick the first option from the disambiguation list and
                # retry. The for-loop's attempt counter still advances, so
                # we cap at one disambig hop before giving up — prevents
                # infinite loops on pathological titles.
                options = list(_exc.options) if _exc.options else []
                if not options:
                    print(
                        f"[live_search] DisambiguationError on "
                        f"{fetched_title!r} with no options — giving up"
                    )
                    return None
                new_title = options[0]
                print(
                    f"[live_search] DisambiguationError on {fetched_title!r}; "
                    f"retrying with {new_title!r}"
                )
                fetched_title = new_title
                # Fall through to next iteration; no sleep needed (this
                # isn't a rate-limit failure, just a redirect).
                continue
            except wikipedia.exceptions.PageError as _exc:
                # Page truly doesn't exist — terminal, retry won't help.
                print(f"[live_search] PageError on {fetched_title!r}: {_exc}")
                return None
            except Exception as _exc:  # noqa: BLE001
                is_last = (attempt == 1)
                if is_last:
                    print(
                        f"[live_search] EXCEPTION in _fetch_one"
                        f"({fetched_title!r}) (after retry): "
                        f"{type(_exc).__name__}: {_exc}"
                    )
                    return None
                print(
                    f"[live_search] RETRY _fetch_one({fetched_title!r}) "
                    f"after {type(_exc).__name__}: {_exc}"
                )
                time.sleep(1.0)

        if text is None:
            return None
        text = clean_wikipedia_text(text)
        if not text.strip():
            return None

        return Article(
            title=fetched_title,
            text=text,
            category=category if category is not None else Category.HISTORY,
            url=url,
        )
