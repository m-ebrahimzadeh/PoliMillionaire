"""Online news source for the NEWS category — The Guardian Open Platform.

Why a news API at all
─────────────────────
The other five categories answer well from Wikipedia.  NEWS does not: its
questions reference a *specific dated article* —

    "According to the article published on 2026-05-17, which charity is
     advocating for changes to the benefit cap …?"

Wikipedia cannot retrieve that.  A news API with full-text search, a real
archive, and precise date filtering can.  The Guardian Open Platform is the
best free fit: a free developer key, the full article body via
``show-fields=bodyText``, a complete archive back to 1999, and ``from-date`` /
``to-date`` publication-date filtering.

Two entry points, this module has:

- ``GuardianNewsSource`` — a thin ``requests`` client over the Guardian content
  search endpoint.  ``search()`` powers the online live-search fallback;
  ``fetch_range()`` paginates a date range for the offline harvest
  (``scripts/fetch_news_corpus.py``).  Responses are cached on disk so repeat
  queries (eval replays) cost no quota and need no network.

- ``NewsLiveSearch`` — the object handed to ``RAGStrategy(news_search=…)``.  It
  implements the same ``search(query, *, category) -> list[Article]`` contract
  as ``LiveSearchFallback`` so it drops straight into the existing gated
  live-fallback seam.  Internally it extracts the date from the question, asks
  the Guardian for that date window, and — on an empty result, a missing key,
  or any error — *delegates to the Wikipedia fallback* so behaviour degrades
  gracefully instead of going dark.

Design constraints (mirrors ``live_search.py``)
───────────────────────────────────────────────
- Hard per-request timeout via ``requests`` ``timeout=`` — a stray network call
  never burns the question budget.  On 429 / quota / any error the source
  returns ``[]`` and the caller degrades.
- Returns ``corpus.Article`` objects so ``IndexGrower`` ingests them with the
  exact same ``chunk_text`` pipeline as the offline build.
- The publication date is carried *in-band* as a ``"Published YYYY-MM-DD. "``
  dateline prefix on the article text — so it informs both the passage
  embedding and BM25 — without changing the ``Chunk`` schema.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import requests

# Import the config *module* (not the NEWS/PATHS names) so that a runtime
# ``update_news(...)`` — which rebinds ``config.NEWS`` — is honoured by no-arg
# construction.  Category / NewsConfig are never rebound, so importing those by
# name is fine.
from .. import config as _config
from ..config import Category, NewsConfig
from .corpus import Article

# Sentinel so ``cache_dir=None`` can explicitly disable caching while the
# default still resolves to ``PATHS.news_cache_dir`` at construction time.
_UNSET = object()


# ── Date extraction ─────────────────────────────────────────────────────────

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# "16 May 2026" / "16th May 2026"
_DMY_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b"
)
# "May 16, 2026" / "May 16 2026"
_MDY_RE = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
)


def extract_question_date(text: str) -> Optional[_dt.date]:
    """Pull the first publication date out of a question, if one is stated.

    Handles ISO (``2026-05-17``) and the two common English prose forms
    (``17 May 2026``, ``May 17, 2026``).  Returns ``None`` when no date is
    present — the caller then queries without a date window.

    Args:
        text: the question string.

    Returns:
        A ``datetime.date`` for the first date found, else ``None``.
    """
    if not text:
        return None

    m = _ISO_DATE_RE.search(text)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        try:
            return _dt.date(y, mo, d)
        except ValueError:
            pass

    for rx, order in ((_DMY_RE, "dmy"), (_MDY_RE, "mdy")):
        m = rx.search(text)
        if not m:
            continue
        g = m.groups()
        if order == "dmy":
            day, month_name, year = g
        else:
            month_name, day, year = g
        month = _MONTHS.get(month_name.lower())
        if month is None:
            continue
        try:
            return _dt.date(int(year), month, int(day))
        except ValueError:
            continue

    return None


# ── Guardian client ──────────────────────────────────────────────────────────

class GuardianNewsSource:
    """Query The Guardian content API and return ``Article`` objects.

    Args:
        config: a :class:`~polimibot.config.NewsConfig`.  ``None`` (default)
            resolves the current ``NEWS`` singleton at construction time, so an
            earlier ``update_news(...)`` is honoured.
        cache_dir: directory for the on-disk response cache.  Defaults to
            ``PATHS.news_cache_dir``.  Pass ``None`` to disable caching.
    """

    def __init__(
        self,
        config: Optional[NewsConfig] = None,
        *,
        cache_dir: Optional[Path] = _UNSET,  # type: ignore[assignment]
    ) -> None:
        self.config = config if config is not None else _config.NEWS
        self.cache_dir = (
            _config.PATHS.news_cache_dir if cache_dir is _UNSET else cache_dir
        )
        self._last_call = 0.0
        self._warned_no_key = False
        # Set by ``_request``: True when the most recent call hit a transport
        # failure (no-key / 429 / timeout / network) rather than a clean result.
        self._last_call_failed = False
        # Cumulative health counters for observability (see ``stats``). Kept
        # cheap (a dict of ints) so a live game incurs no measurable overhead.
        self._stats: dict[str, int] = {
            "calls": 0,       # network requests actually sent (cache misses)
            "cache_hits": 0,  # served from the on-disk cache (no network)
            "hits": 0,        # ok response with >=1 result
            "empty_ok": 0,    # ok response with 0 results
            "http_429": 0,    # rate-limited (after the short-retry logic)
            "timeouts": 0,    # request timed out
            "errors": 0,      # non-ok status / network / parse errors
        }

    @property
    def stats(self) -> dict[str, int]:
        """A snapshot copy of the cumulative request counters."""
        return dict(self._stats)

    @property
    def last_call_failed(self) -> bool:
        """Whether the most recent ``_request`` failed at the transport level.

        Lets ``NewsLiveSearch`` distinguish "the Guardian is unavailable right
        now" (don't retry into a throttled endpoint) from "the Guardian
        answered, but found nothing" (a broader query may still help).
        """
        return self._last_call_failed

    # ── Public API ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        from_date: Optional[_dt.date] = None,
        to_date: Optional[_dt.date] = None,
        page_size: Optional[int] = None,
        order_by: str = "relevance",
    ) -> list[Article]:
        """Full-text search, optionally constrained to a publication-date window.

        Args:
            query: free-text query (Guardian supports AND/OR/quoted phrases).
            from_date / to_date: inclusive publication-date bounds.
            page_size: results to request (capped 1–50).  Defaults to
                ``config.max_articles``.
            order_by: ``relevance`` (default), ``newest`` or ``oldest``.

        Returns:
            Up to ``page_size`` Article objects.  ``[]`` on missing key,
            empty result, timeout, 429, or any network/parse error.
        """
        size = page_size or self.config.max_articles
        params = self._base_params(query, order_by=order_by, page_size=size)
        if from_date is not None:
            params["from-date"] = from_date.isoformat()
        if to_date is not None:
            params["to-date"] = to_date.isoformat()

        resp = self._request(params)
        if not resp:
            return []
        results = resp.get("results", []) or []
        articles = [a for a in (self._to_article(r) for r in results) if a is not None]
        return articles[:size]

    def fetch_range(
        self,
        from_date: _dt.date,
        to_date: _dt.date,
        *,
        query: Optional[str] = None,
        sections: Optional[str] = None,
        page_size: int = 50,
        max_pages: int = 20,
        order_by: str = "newest",
    ) -> list[Article]:
        """Paginate a date range — used by the offline harvest script.

        Args:
            from_date / to_date: inclusive publication-date bounds.
            query: optional full-text filter; ``None`` harvests everything
                published in the window (optionally narrowed by ``sections``).
            sections: comma-separated Guardian section ids (e.g.
                ``"world,uk-news,business"``) to focus the harvest.
            page_size: results per page (Guardian max 50).
            max_pages: safety cap on pages fetched (``max_pages * page_size``
                articles at most).
            order_by: page ordering; ``newest`` by default.

        Returns:
            All Article objects gathered across the paged window.
        """
        out: list[Article] = []
        for page in range(1, max_pages + 1):
            params = self._base_params(
                query or "", order_by=order_by, page_size=min(page_size, 50)
            )
            params["from-date"] = from_date.isoformat()
            params["to-date"] = to_date.isoformat()
            params["page"] = str(page)
            if sections:
                params["section"] = sections
            if not query:
                # An empty q would be rejected; drop it and rely on the
                # date window (+ optional section) to scope the harvest.
                params.pop("q", None)

            resp = self._request(params)
            if not resp:
                break
            results = resp.get("results", []) or []
            out.extend(a for a in (self._to_article(r) for r in results) if a is not None)

            total_pages = int(resp.get("pages", page) or page)
            if page >= total_pages or not results:
                break
        return out

    # ── Internals ─────────────────────────────────────────────────────────────

    def _base_params(self, query: str, *, order_by: str, page_size: int) -> dict:
        params = {
            "q": query.strip(),
            "api-key": self.config.guardian_api_key,
            "order-by": order_by,
            "page-size": str(max(1, min(page_size, 50))),
            "show-fields": "bodyText,trailText,headline" if self.config.use_full_body
                           else "trailText,headline",
        }
        return params

    def _request(self, params: dict) -> Optional[dict]:
        """GET the Guardian endpoint with caching + a hard timeout.

        Returns the inner ``response`` object (the dict holding ``results`` /
        ``pages``) on success, or ``None`` on any failure.  Failures are logged
        with their exception class so a 429 / quota / network blip can be told
        apart from a genuine empty result, and ``last_call_failed`` records
        which it was.
        """
        self._last_call_failed = False

        # No key → don't send anything. The public 'test' key is globally
        # throttled (every call 429s) and silently masks a misconfigured /
        # set-too-late key as a "rate limit". Skip the network entirely and let
        # the caller degrade to Wikipedia/offline.
        if not self.config.guardian_api_key:
            if not self._warned_no_key:
                print(
                    "[news_search] GUARDIAN_API_KEY not set — skipping the "
                    "Guardian and degrading to Wikipedia/offline. Set the env "
                    "var *before the strategy is built* (e.g. from the Colab "
                    "secret) to enable the online news path."
                )
                self._warned_no_key = True
            self._last_call_failed = True
            return None

        cache_key = self._cache_key(params)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._stats["cache_hits"] += 1
            return cached

        # Client-side throttle (free tier ~1 req/s).
        elapsed = time.monotonic() - self._last_call
        if 0 < elapsed < self.config.min_delay_seconds:
            time.sleep(self.config.min_delay_seconds - elapsed)

        self._stats["calls"] += 1
        payload = None
        for attempt in range(2):  # at most one retry, only on a short-wait 429
            try:
                r = requests.get(
                    self.config.guardian_base_url,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                self._last_call = time.monotonic()
                if r.status_code == 429:
                    wait = self._retry_after_seconds(r)
                    remaining = self._rate_limit_remaining(r)
                    if (attempt == 0 and wait is not None
                            and wait <= self.config.max_retry_seconds):
                        print(
                            f"[news_search] RATE LIMIT (HTTP 429){remaining} — "
                            f"retrying in {wait:.1f}s"
                        )
                        time.sleep(wait)
                        continue
                    print(
                        f"[news_search] RATE LIMIT (HTTP 429){remaining} — "
                        "backing off to fallback"
                    )
                    self._stats["http_429"] += 1
                    self._last_call_failed = True
                    return None
                r.raise_for_status()
                payload = r.json()
                break
            except requests.Timeout:
                print(f"[news_search] TIMEOUT after {self.config.timeout_seconds}s")
                self._stats["timeouts"] += 1
                self._last_call_failed = True
                return None
            except Exception as exc:  # noqa: BLE001
                print(f"[news_search] EXCEPTION: {type(exc).__name__}: {exc}")
                self._stats["errors"] += 1
                self._last_call_failed = True
                return None

        if payload is None:  # defensive: retries exhausted without a payload
            self._stats["errors"] += 1
            self._last_call_failed = True
            return None

        response = payload.get("response") if isinstance(payload, dict) else None
        if not response or response.get("status") != "ok":
            print(f"[news_search] non-ok response: {str(payload)[:200]}")
            self._stats["errors"] += 1
            self._last_call_failed = True
            return None

        if response.get("results"):
            self._stats["hits"] += 1
        else:
            self._stats["empty_ok"] += 1
        self._cache_put(cache_key, response)
        return response

    @staticmethod
    def _retry_after_seconds(resp) -> Optional[float]:
        """Parse the ``Retry-After`` header (seconds form) into a float."""
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            # HTTP-date form is rare on this endpoint; treat as unknown so the
            # caller gives up rather than guessing a wait.
            return None

    @staticmethod
    def _rate_limit_remaining(resp) -> str:
        """A short ' [remaining day=… minute=…]' suffix for 429 logs, if sent."""
        day = resp.headers.get("X-RateLimit-Remaining-day")
        minute = resp.headers.get("X-RateLimit-Remaining-minute")
        if day is None and minute is None:
            return ""
        return f" [remaining day={day} minute={minute}]"

    def _to_article(self, item: dict) -> Optional[Article]:
        """Map one Guardian result item to an ``Article`` (NEWS-tagged).

        The publication date is prepended as an in-band dateline so it informs
        the passage embedding and BM25 without a Chunk-schema change.
        """
        fields = item.get("fields", {}) or {}
        body = fields.get("bodyText") or fields.get("trailText") or ""
        title = item.get("webTitle") or fields.get("headline") or ""
        if not title or not body.strip():
            return None
        pub = (item.get("webPublicationDate") or "")[:10]  # YYYY-MM-DD
        dateline = f"Published {pub}. " if pub else ""
        text = _clean_news_text(f"{dateline}{body}")
        if not text:
            return None
        return Article(
            title=title,
            text=text,
            category=Category.NEWS,
            url=item.get("webUrl", "") or "",
        )

    # ── Disk cache ──────────────────────────────────────────────────────────

    def _cache_key(self, params: dict) -> str:
        # Exclude the api-key from the cache identity so a key rotation does
        # not invalidate the cache and the key never lands on disk in a path.
        keyable = {k: v for k, v in params.items() if k != "api-key"}
        blob = json.dumps(keyable, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[dict]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — corrupt cache entry → treat as miss
            return None
        if not isinstance(blob, dict):
            return None

        # Legacy bare-response entries (written before TTL support) carry no
        # ``cached_at`` envelope — honour them permanently so existing caches
        # stay valid after the upgrade.
        if "cached_at" not in blob:
            return blob

        response = blob.get("response")
        if not isinstance(response, dict):
            return None

        # Differential freshness. Empty results are the staleness risk (the
        # query may have run before the Guardian indexed a very recent
        # article), so they expire after ``empty_cache_ttl_seconds``. Non-empty
        # windowed results are stable and default to permanent
        # (``cache_ttl_seconds is None``) so eval replays cost no quota.
        ttl = (
            self.config.empty_cache_ttl_seconds
            if not response.get("results")
            else self.config.cache_ttl_seconds
        )
        if ttl is not None and (time.time() - float(blob.get("cached_at", 0.0))) > ttl:
            return None
        return response

    def _cache_put(self, key: str, response: dict) -> None:
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            envelope = {"cached_at": time.time(), "response": response}
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(envelope, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 — cache is best-effort
            print(f"[news_search] WARNING: cache write failed: {exc}")


def _clean_news_text(text: str) -> str:
    """Collapse whitespace in Guardian ``bodyText`` (already plain text)."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Live-search wrapper (the RAGStrategy plug-in) ────────────────────────────

class NewsLiveSearch:
    """Date-aware Guardian live search with a Wikipedia fallback.

    Implements the same ``search(query, *, category) -> list[Article]``
    contract as :class:`~polimibot.rag.live_search.LiveSearchFallback`, so
    ``RAGStrategy`` treats it as a drop-in live source for the NEWS category.

    Lookup order, per query:
      1. Extract the stated date from the question; build a ``±date_window_days``
         publication window around it.
      2. Ask the Guardian for that window (relevance-ordered).
      3. If empty *and* a date was found, retry once without the window
         (the article may carry a slightly different publication date).
      4. If still empty (or no key / an error), delegate to the Wikipedia
         ``LiveSearchFallback`` so we never go dark.

    Args:
        config: NewsConfig; ``None`` (default) resolves the current ``NEWS``
            singleton at construction time (honours ``update_news(...)``).
        guardian: an existing ``GuardianNewsSource`` (constructed from
            ``config`` if omitted).
        wiki_fallback: an existing ``LiveSearchFallback`` for the secondary
            path.  Constructed lazily on first need if omitted; pass ``None``
            *and* set ``use_wiki_fallback=False`` to disable it entirely.
        use_wiki_fallback: when False, skip the Wikipedia secondary path.
    """

    def __init__(
        self,
        config: Optional[NewsConfig] = None,
        *,
        guardian: Optional[GuardianNewsSource] = None,
        wiki_fallback=None,
        use_wiki_fallback: bool = True,
    ) -> None:
        self.config = config if config is not None else _config.NEWS
        self.guardian = guardian or GuardianNewsSource(self.config)
        self.use_wiki_fallback = use_wiki_fallback
        self._wiki = wiki_fallback  # may be lazily built in search()
        # Per-query provenance, read by RAGStrategy into ``extras`` so the
        # observability layer can tell a Guardian hit apart from the Wikipedia
        # fallback (the dashboards otherwise mislabel every News live result as
        # "Wikipedia").  Set fresh on each ``search()`` call.
        self.last_provider: str = "none"   # guardian_window | guardian_broad | wikipedia | none
        self.last_date_extracted: bool = False
        # Cumulative routing counters for observability (see ``stats``):
        # ``queries``, ``with_date`` and one ``provider_<name>`` per outcome.
        self._stats: Counter = Counter()

    @property
    def stats(self) -> dict[str, int]:
        """A snapshot copy of the cumulative routing counters."""
        return dict(self._stats)

    def search(
        self,
        query: str,
        *,
        category: Optional[Category] = None,
    ) -> list[Article]:
        """Return up to ``config.max_articles`` Article objects for ``query``.

        Thin wrapper over :meth:`_search` that folds the outcome into the
        cumulative routing counters (``stats``) for observability.

        Args:
            query: the question text (the stated date is parsed out of it).
            category: passed through to the Wikipedia fallback so its results
                are tagged correctly; Guardian results are always ``NEWS``.
        """
        result = self._search(query, category=category)
        if query and query.strip():
            self._stats["queries"] += 1
            if self.last_date_extracted:
                self._stats["with_date"] += 1
            self._stats[f"provider_{self.last_provider}"] += 1
        return result

    def _search(
        self,
        query: str,
        *,
        category: Optional[Category] = None,
    ) -> list[Article]:
        self.last_provider = "none"
        self.last_date_extracted = False
        if not query or not query.strip():
            return []

        date = extract_question_date(query)
        q = _build_news_query(query)
        self.last_date_extracted = date is not None

        articles: list[Article] = []
        if date is not None:
            window = _dt.timedelta(days=self.config.date_window_days)
            articles = self.guardian.search(
                q, from_date=date - window, to_date=date + window
            )
            if articles:
                self.last_provider = "guardian_window"
                return articles[: self.config.max_articles]
            # If that call hit a 429 / timeout / no-key, the Guardian is
            # unavailable right now — a second unconstrained call would only
            # earn a second 429. Degrade straight to Wikipedia instead.
            if self.guardian.last_call_failed:
                return self._wiki_search(query, category)

        # Healthy but empty (window too tight) or no date stated → broaden once.
        articles = self.guardian.search(q)
        if articles:
            self.last_provider = "guardian_broad"
            return articles[: self.config.max_articles]

        # ── Secondary fallback: Wikipedia ──────────────────────────────────
        return self._wiki_search(query, category)

    def _wiki_search(
        self, query: str, category: Optional[Category]
    ) -> list[Article]:
        """Delegate to the Wikipedia ``LiveSearchFallback`` (the secondary path)."""
        if not self.use_wiki_fallback:
            return []
        wiki = self._wiki_source()
        if wiki is None:
            return []
        results = wiki.search(query, category=category or Category.NEWS)
        if results:
            self.last_provider = "wikipedia"
        return results

    def _wiki_source(self):
        if self._wiki is None and self.use_wiki_fallback:
            from .live_search import LiveSearchFallback
            self._wiki = LiveSearchFallback(
                timeout_seconds=self.config.timeout_seconds,
                max_articles=self.config.max_articles,
            )
        return self._wiki


# Lead-ins that add no retrieval signal — stripped before hitting the Guardian.
_BOILERPLATE_RE = re.compile(
    r"^(?:according to|as reported (?:in|on|by)|based on|per)\s+"
    r"(?:the\s+)?(?:article|report|story|piece)?\s*"
    r"(?:published|reported|released)?\s*(?:on|in|by)?\s*",
    re.IGNORECASE,
)


def _build_news_query(question: str) -> str:
    """Turn a question into a Guardian full-text query.

    Strips the stated date token and a few stock lead-ins ("According to the
    article published on …") that only dilute the search, leaving the salient
    entities/terms.  Guardian's relevance ranking does the rest.
    """
    q = question.strip()
    # Drop date tokens — the window param handles dates; in ``q`` they hurt.
    q = _ISO_DATE_RE.sub(" ", q)
    q = _DMY_RE.sub(" ", q)
    q = _MDY_RE.sub(" ", q)
    q = _BOILERPLATE_RE.sub("", q).strip()
    q = re.sub(r"\s+", " ", q).strip(" ,.")
    return q or question.strip()


# ── Offline harvest helper ───────────────────────────────────────────────────

def harvest_news_range(
    from_date: _dt.date,
    to_date: _dt.date,
    *,
    source: Optional[GuardianNewsSource] = None,
    sections: Optional[str] = None,
    query: Optional[str] = None,
    page_size: int = 50,
    max_pages: int = 20,
    verbose: bool = False,
) -> list[Article]:
    """Harvest Guardian articles across a date window, one day at a time.

    Used by both the offline harvest script (``scripts/fetch_news_corpus.py``)
    and the notebook's index build to seed the offline corpus with recent news.

    The harvest is day-by-day because a single ``fetch_range`` over a multi-day
    window only returns the newest ``page_size * max_pages`` results (the
    Guardian publishes hundreds of pieces a day), silently dropping the older
    end of the range — exactly where the dated News questions live. Articles are
    de-duplicated by title across the whole window (paging / adjacent days can
    re-surface an article).

    Args:
        from_date / to_date: inclusive publication-date bounds.
        source: an existing ``GuardianNewsSource`` (constructed if omitted, so
            the current ``NEWS`` config / key is honoured).
        sections: comma-separated Guardian section ids to focus the harvest
            (e.g. ``"world,uk-news,business"``); ``None`` harvests every section.
        query: optional full-text filter; ``None`` harvests everything in-window.
        page_size / max_pages: per-day pagination budget (see ``fetch_range``).
        verbose: print per-day counts and a final unique total.

    Returns:
        Unique ``Article`` objects (``Category.NEWS``) gathered across the window.
    """
    src = source or GuardianNewsSource()
    seen: set[str] = set()
    unique: list[Article] = []
    total = 0
    day = from_date
    while day <= to_date:
        day_articles = src.fetch_range(
            day, day,
            query=query, sections=sections,
            page_size=page_size, max_pages=max_pages,
        )
        total += len(day_articles)
        for a in day_articles:
            if a.title not in seen:
                seen.add(a.title)
                unique.append(a)
        if verbose:
            print(f"  {day}: {len(day_articles)} article(s)")
        day += _dt.timedelta(days=1)
    if verbose:
        print(f"Fetched {total} results → {len(unique)} unique articles.")
    return unique
