"""Wikipedia corpus builder. Fetch once, chunk as many times as you like.

Separation of concerns:
  - This module: knows *what* to fetch and how to persist raw text.
  - chunker.py:  knows *how* to split text into retrieval units.
  - index.py:    knows *how* to store and search embeddings.
"""
from __future__ import annotations

import json
from random import random
import re
import time
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import Category


# ── Text cleanup ──────────────────────────────────────────────────────────────
# Wikipedia plaintext (via the `wikipedia` library) contains citation markers
# ("[1]", "[42][43]") and trailing meta-sections (References, See also, …)
# that add noise to embeddings and waste prompt-context tokens.
#
# Section headers like "== Early life ==" are KEPT — useful retrieval signal.
# Bumping this regex tail list bumps CLEANUP_VERSION so the index manifest
# (built later in this PR) can detect stale corpora.

CLEANUP_VERSION = 1

# Bumped when the seed list, disambiguation policy, or any other corpus
# *selection* logic changes (separate from CLEANUP_VERSION, which tracks the
# in-place text normalisation only). Surfaced in the index manifest so a
# retriever can spot a stale corpus even when cleanup didn't change.
CORPUS_VERSION = 4   # v4 = concept-first seeds + aliases/competition schema (was v3: category-graph harvest)

_CITATION_RE = re.compile(r"\[\d+\](?:\[\d+\])*")

_TAIL_SECTIONS_RE = re.compile(
    r"^={2,}\s*("
    r"References?|See also|External links?|Notes?|"
    r"Further reading|Bibliography|Sources?|Citations?|"
    r"Footnotes?"
    r")\s*={2,}\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_wikipedia_text(text: str) -> str:
    """Strip Wikipedia noise: citation markers and trailing meta-sections.

    Idempotent — running twice gives the same result, so it's safe to apply
    both at fetch-time (so saved corpora are clean) and at chunk-time
    (defensive pass over older corpora pre-dating this function).
    """
    if not text:
        return text
    # Drop [1], [2][3], etc. inline citation markers.
    text = _CITATION_RE.sub("", text)
    # Truncate from the first tail meta-section onward.
    m = _TAIL_SECTIONS_RE.search(text)
    if m:
        text = text[: m.start()]
    # Collapse excessive whitespace from the cuts.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Domain record ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Article:
    """One Wikipedia article — the raw unit before chunking.

    The trailing fields are optional with safe defaults so every existing
    constructor and on-disk corpus (which omits them) keeps loading:

      ``aliases`` — Wikipedia redirect titles / alternate phrasings for this
          article (e.g. "Dr. Drake Ramoray" → *The One Where Dr. Ramoray Dies*).
          Indexed as extra retrieval keys so a question that names the entity
          by an alias still reaches the article.
      ``competition`` — the exact runtime competition label this article serves
          (e.g. "Ancient History and Politics"), derived from ``category`` via
          ``config.CATEGORIES``. Kept for provenance/trace; the retriever still
          filters on ``category``.
    """
    title: str
    text: str         # full article text (may be many thousands of words)
    category: Category
    url: str = ""
    aliases: tuple[str, ...] = ()
    competition: str = ""


# ── Topic seeds ───────────────────────────────────────────────────────────────
# These are the Wikipedia article titles we seed the corpus with.
# Coverage over depth: prefer breadth of topics over very long articles.
# Maths: favour concept/history articles over pure computation (quiz questions
#         that ask "what is 12×13" derive zero benefit from Wikipedia).

TOPIC_SEEDS: dict[Category, list[str]] = {
    Category.ENTERTAINMENT: [
        "Film", "Academy Awards", "The Beatles", "Michael Jackson",
        "Walt Disney", "Marvel Comics", "James Bond", "Star Wars",
        "The Godfather", "Titanic (1997 film)", "Forrest Gump",
        "Pulp Fiction", "The Shawshank Redemption", "Schindler's List",
        "Elvis Presley", "Madonna (entertainer)", "Bob Dylan",
        "Rolling Stones", "Queen (band)", "David Bowie",
        "Friends (TV series)", "The Simpsons", "Breaking Bad",
        "Game of Thrones", "Seinfeld",
    ],
    Category.HISTORY: [
        "Julius Caesar", "Roman Republic", "Ancient Rome",
        "Alexander the Great", "Ancient Greece", "Peloponnesian War",
        "Napoleon Bonaparte", "French Revolution", "World War I",
        "World War II", "Adolf Hitler", "Winston Churchill",
        "Ancient Egypt", "Cleopatra", "Byzantine Empire",
        "Ottoman Empire", "Mongol Empire", "Genghis Khan",
        "Renaissance", "Age of Exploration", "Christopher Columbus",
        "American Revolution", "Abraham Lincoln", "Cold War",
        "Roman Empire",
    ],
    Category.SCIENCE: [
        "Isaac Newton", "Albert Einstein", "Charles Darwin",
        "Evolution", "DNA", "Photosynthesis", "Periodic table",
        "Chemical element", "Black hole", "Solar System",
        "Quantum mechanics", "Theory of relativity", "Cell (biology)",
        "Human anatomy", "Nervous system", "Immune system",
        "Climate change", "Plate tectonics", "Big Bang",
        "Atom", "Molecule", "Gravity", "Thermodynamics",
        "Electromagnetic spectrum", "Marie Curie",
    ],
    Category.MATHS: [
        "Mathematics", "Prime number", "Pythagorean theorem",
        "Pi", "Calculus", "Isaac Newton", "Gottfried Wilhelm Leibniz",
        "Geometry", "Algebra", "Probability", "Statistics",
        "Fibonacci sequence", "Euclid", "Archimedes",
        "Number theory", "Logarithm", "Trigonometry",
        "Set theory", "Graph theory", "Cryptography",
    ],
    Category.PHILOSOPHY: [
        "Philosophy", "Psychology", "Socrates", "Plato", "Aristotle",
        "Immanuel Kant", "René Descartes", "David Hume",
        "Friedrich Nietzsche", "Jean-Paul Sartre", "Stoicism",
        "Existentialism", "Epistemology", "Metaphysics", "Ethics",
        "Sigmund Freud", "Carl Jung", "B. F. Skinner",
        "Ivan Pavlov", "Cognitive psychology", "Behaviorism",
        "Psychoanalysis", "Cognitive bias", "Classical conditioning",
        "Operant conditioning",
    ],
    Category.NEWS: [
        "News", "Journalism", "Pulitzer Prize", "Time Person of the Year",
        "Nobel Peace Prize", "United Nations",
        "Secretary-General of the United Nations", "European Union",
        "G7", "G20", "Brexit", "September 11 attacks",
        "COVID-19 pandemic", "Watergate scandal",
        "Russian invasion of Ukraine", "Arab Spring",
        "Fall of the Berlin Wall", "Reuters", "Associated Press",
        "BBC News",
    ],
}


# ── Fetcher ───────────────────────────────────────────────────────────────────

def _dedupe_seeds(
    targets: list[Category],
    *,
    verbose: bool,
) -> list[tuple[str, Category]]:
    """Flatten TOPIC_SEEDS into a (title, category) list with cross-category
    duplicates removed.

    A title that appears under multiple categories (e.g. "Isaac Newton" in
    both SCIENCE and MATHS) is kept under its first occurrence in ``targets``
    order. The duplicate is logged so the seeds file can be cleaned up at
    leisure; until then the corpus never contains two copies of the same
    article wearing different category tags.
    """
    seen: set[str] = set()
    flat: list[tuple[str, Category]] = []
    for cat in targets:
        for title in TOPIC_SEEDS[cat]:
            if title in seen:
                if verbose:
                    print(f"  ! '{title}' already seeded in earlier category — skipped duplicate")
                continue
            seen.add(title)
            flat.append((title, cat))
    return flat


def fetch_articles(
    categories: list[Category] | None = None,
    *,
    sleep_seconds: float = 0.3,   # politeness delay between API calls
    verbose: bool = True,
) -> list[Article]:
    """Fetch Wikipedia articles for each topic seed.

    Args:
        categories: subset of categories to fetch (default: all four).
        sleep_seconds: pause between requests to avoid rate-limiting.
        verbose: print progress.

    Returns:
        List of Article objects. Failed fetches are skipped with a warning.
        Cross-category seed duplicates are deduped before fetching.
    """
    import wikipedia  # lazy import — only needed at corpus-build time

    _configure_wikipedia(wikipedia)
    targets = categories or list(TOPIC_SEEDS.keys())
    flat_seeds = _dedupe_seeds(targets, verbose=verbose)

    # Group printout by category for readability while iterating the flat list.
    if verbose:
        for cat in targets:
            n = sum(1 for _, c in flat_seeds if c == cat)
            print(f"\n[{cat.value}] fetching {n} articles…")

    articles: list[Article] = []
    for title, cat in flat_seeds:
        article = _fetch_one(title, cat, verbose=verbose)
        if article is not None:
            articles.append(article)
        time.sleep(sleep_seconds)

    if verbose:
        print(f"\nFetched {len(articles)} articles total.")
    return articles


# Words that carry no entity signal when checking a disambiguation match.
_SEED_STOP = frozenset({
    "the", "a", "an", "and", "of", "in", "on", "to", "for", "is", "are",
})


def _seed_keywords(title: str) -> set[str]:
    """Tokens from a seed title that should appear in the matching page.

    Filters trivial connectives so checks like "Queen (band)" require
    "queen" or "band" — not the bracket itself or "of".
    """
    raw = re.findall(r"\w+", title.lower())
    return {w for w in raw if w not in _SEED_STOP and len(w) > 1}


# Per-title retry is intentionally CHEAP. The dominant failure is an empty-body
# response ("Expecting value: line 1 column 1") which is a *rate-limit window*,
# not a one-off blip — it outlasts a few seconds, so retrying the same title at
# 2s then 4s just burns ~6s and hammers the throttled endpoint without helping
# (live runs show the same title failing all three attempts). One quick retry
# still catches a genuine transient blip; sustained throttling is handled at the
# crawl level by a cooldown once failures cluster (see _make_failure_cooldown).
_FETCH_MAX_ATTEMPTS = 2
_FETCH_BACKOFF = (0.0, 0.5)  # no wait before attempt 1; 0.5s before the single retry

# When empty-body failures arrive in a burst, the per-IP rate limit is active.
# Plowing on just skips every title in the window; instead pause the crawl once
# enough consecutive fetches fail, letting the limit clear so the next titles
# (and a second pass) succeed.
_COOLDOWN_AFTER_CONSECUTIVE_FAILURES = 8
_RATE_LIMIT_COOLDOWN_SECONDS = 30.0


def _make_failure_cooldown(verbose: bool):
    """Return ``note(ok: bool)`` that tracks consecutive fetch failures and, once
    they cross ``_COOLDOWN_AFTER_CONSECUTIVE_FAILURES``, sleeps a one-off cooldown
    so an active rate-limit window can clear. A success resets the streak."""
    state = {"streak": 0}

    def note(ok: bool) -> None:
        if ok:
            state["streak"] = 0
            return
        state["streak"] += 1
        if state["streak"] >= _COOLDOWN_AFTER_CONSECUTIVE_FAILURES:
            if verbose:
                print(f"    … {state['streak']} consecutive failures — pausing "
                      f"{_RATE_LIMIT_COOLDOWN_SECONDS:.0f}s for the rate limit to clear")
            time.sleep(_RATE_LIMIT_COOLDOWN_SECONDS)
            state["streak"] = 0

    return note


def _page_with_retry(title: str, *, verbose: bool):
    """Call ``wikipedia.page(title)`` with automatic retry on transient errors.

    Retries on any exception that looks transient (network / JSON parse /
    HTTP 5xx / rate-limit). Hard failures such as ``PageError`` or
    ``DisambiguationError`` are re-raised immediately so callers can handle
    them specifically.
    """
    import json as _json
    import wikipedia

    # These are intentional "hard" errors — no point retrying them.
    _hard = (wikipedia.DisambiguationError, wikipedia.PageError)

    last_exc: Exception | None = None
    for attempt in range(_FETCH_MAX_ATTEMPTS):
        if attempt > 0:
            delay = _FETCH_BACKOFF[min(attempt, len(_FETCH_BACKOFF) - 1)]
            if verbose:
                print(f"    retrying '{title}' in {delay:.0f}s (attempt {attempt + 1})…")
            time.sleep(delay)
        try:
            return wikipedia.page(title, auto_suggest=False)
        except _hard:
            raise   # propagate immediately — caller handles these
        except Exception as exc:
            last_exc = exc
            # json.JSONDecodeError (empty/bad API response) is the most common
            # transient error; log at debug level and retry.
            if verbose and attempt == 0:
                print(f"    transient error for '{title}': {exc!r} — will retry")

    # All attempts exhausted — re-raise the last exception.
    raise last_exc  # type: ignore[misc]


# Wikimedia's API:Etiquette requires a contact-bearing User-Agent.
_WIKI_API_USER_AGENT = (
    "PoliMillionaire/1.0 "
    "(https://github.com/m-ebrahimzadeh/PoliMillionaire; contact: ebrahimzadeh.meh@gmail.com)"
)

# Minimum spacing between `wikipedia` library API calls. Wikimedia throttles
# anonymous high-volume clients (a shared Colab IP fetching thousands of titles
# trips this), returning empty 0-byte bodies that surface as JSONDecodeError.
# A small global floor keeps the crawl under the throttle far better than the
# per-title back-off alone (which kicks in only *after* a failure).
_WIKI_MIN_REQUEST_INTERVAL_MS = 300


def _configure_wikipedia(wikipedia) -> None:
    """One-time politeness setup for the `wikipedia` library: English, a
    contact-bearing User-Agent (API:Etiquette), and a global minimum request
    interval. Safe to call repeatedly — it just re-asserts the settings."""
    from datetime import timedelta
    wikipedia.set_lang("en")
    # set_user_agent / set_rate_limiting exist on the standard `wikipedia`
    # package; guard so a stripped stub (e.g. in tests) can't break the crawl.
    if hasattr(wikipedia, "set_user_agent"):
        wikipedia.set_user_agent(_WIKI_API_USER_AGENT)
    if hasattr(wikipedia, "set_rate_limiting"):
        wikipedia.set_rate_limiting(
            True, min_wait=timedelta(milliseconds=_WIKI_MIN_REQUEST_INTERVAL_MS))


# Cap on redirect titles kept per article. Most pages have a handful; a few
# popular ones have dozens of near-duplicate redirects that add no signal.
_MAX_ALIASES = 10


def _competition_for(category: Category) -> str:
    """The runtime competition display name for a category (e.g. HISTORY →
    "Ancient History and Politics"), or "" if unmapped. Single source of truth
    is ``config.CATEGORIES``."""
    from ..config import CATEGORIES
    for info in CATEGORIES.values():
        if info.category == category:
            return info.display_name
    return ""


def _fetch_redirects(title: str, *, verbose: bool = False) -> tuple[str, ...]:
    """Best-effort fetch of a page's redirect titles (aliases) via the MediaWiki
    ``prop=redirects`` API.

    Returns up to ``_MAX_ALIASES`` redirect titles (namespace 0 only), or ``()``
    on any failure — aliases are pure enrichment, so a redirect hiccup must
    never sink the article fetch. One short, polite GET with a named UA.
    """
    params = {
        "action": "query", "prop": "redirects", "titles": title,
        "rdlimit": str(_MAX_ALIASES), "rdnamespace": "0",
        "format": "json", "formatversion": "2",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_API_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — aliases are optional enrichment
        if verbose:
            print(f"    (redirects fetch failed for '{title}': {exc})")
        return ()
    out: list[str] = []
    for p in data.get("query", {}).get("pages", []):
        for r in p.get("redirects", []):
            t = (r.get("title") or "").strip()
            if t and t != title:
                out.append(t)
    return tuple(out[:_MAX_ALIASES])


def _make_article(page, category: Category, *, fetch_aliases: bool,
                  verbose: bool) -> Article:
    """Build an enriched Article from a resolved wikipedia page: clean the body,
    derive the competition label, and (optionally) attach redirect aliases."""
    aliases = _fetch_redirects(page.title, verbose=verbose) if fetch_aliases else ()
    return Article(
        title=page.title,
        text=clean_wikipedia_text(page.content),
        category=category,
        url=page.url,
        aliases=aliases,
        competition=_competition_for(category),
    )


def _fetch_one(title: str, category: Category, *, verbose: bool,
               fetch_aliases: bool = True) -> Optional[Article]:
    """Fetch a single Wikipedia page. Returns None on hard failure.

    On a DisambiguationError, walks the option list (instead of blindly
    picking the first) and returns the first option whose page summary
    contains at least one keyword from the seed title.

    Transient network / JSON-parse errors are retried automatically up to
    ``_FETCH_MAX_ATTEMPTS`` times before giving up. When ``fetch_aliases`` is
    True the resolved page's redirect titles are attached as ``Article.aliases``.
    """
    import wikipedia

    try:
        page = _page_with_retry(title, verbose=verbose)
        return _make_article(page, category, fetch_aliases=fetch_aliases,
                             verbose=verbose)

    except wikipedia.DisambiguationError as e:
        keywords = _seed_keywords(title)
        for option in e.options[:5]:   # cap — disambiguation pages can be huge
            try:
                page = _page_with_retry(option, verbose=verbose)
                # page.summary / page.content are *lazy* MediaWiki calls in the
                # `wikipedia` library — on a throttled endpoint they return an
                # empty body and raise JSONDecodeError. They MUST stay inside
                # this try: a failure here was escaping uncaught and killing the
                # entire crawl (and discarding the in-memory harvest). One bad
                # option is just a clean skip to the next.
                preview = (page.summary or page.content[:500]).lower()
                if not keywords or any(k in preview for k in keywords):
                    if verbose and option != title:
                        print(f"  ! disambiguation for '{title}' → resolved to '{option}'")
                    return _make_article(page, category, fetch_aliases=fetch_aliases,
                                         verbose=verbose)
            except Exception:
                continue
        if verbose:
            print(f"  ! disambiguation for '{title}' — no relevant option found, skipped")

    except wikipedia.PageError:
        if verbose:
            print(f"  ! page not found: '{title}' — skipped")

    except Exception as exc:
        if verbose:
            print(f"  ! unexpected error for '{title}': {exc} — skipped")

    return None


# ── Batched, concurrent harvest (fast path for the bulk category crawl) ─────────
# The `wikipedia` library fetches one page at a time and can't batch content, so
# a 5000-title crawl costs ~3 API calls/title plus a sleep between each. The
# MediaWiki ``prop=extracts`` API returns full plaintext for up to 20 titles in a
# single request, so batching cuts the round-trips ~20x; a small thread pool then
# cuts wall-clock another ~Nx. ``exsectionformat=wiki`` keeps the ``== Header ==``
# markup that the chunker parses, so the produced corpus is byte-for-byte the same
# shape as the per-title path.
_EXTRACTS_BATCH_SIZE = 20      # MediaWiki caps prop=extracts at 20 titles/request
_HARVEST_WORKERS_DEFAULT = 5   # concurrent extract batches; polite + fast


def _fetch_extracts_batch(items: list[tuple[str, "Category"]], *,
                          verbose: bool = False) -> list[Article]:
    """Fetch full plaintext for up to ``_EXTRACTS_BATCH_SIZE`` titles in one
    MediaWiki request and return their Articles.

    ``items`` is ``[(title, category), …]``. Pages flagged ``missing`` or carrying
    a ``disambiguation`` pageprop are skipped. Raises whatever ``_api_get`` raises
    on a hard request failure so the caller can retry the whole batch.
    """
    from .category_seeds import _api_get

    title_to_cat = {t: c for t, c in items}
    titles = list(title_to_cat)
    if not titles:
        return []

    base = {
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "extracts|info|pageprops",
        "explaintext": "1", "exsectionformat": "wiki", "exlimit": "max",
        "inprop": "url", "ppprop": "disambiguation", "redirects": "1",
        "titles": "|".join(titles),
    }

    # Trace each returned/resolved title back to the requested one (the API may
    # normalise casing and follow redirects), so we can recover its category.
    req_of: dict[str, str] = {t: t for t in titles}
    pages_by_title: dict[str, dict] = {}
    cont: dict = {}
    while True:
        params = dict(base)
        params.update(cont)
        data = _api_get(params)
        q = data.get("query", {})
        for n in q.get("normalized", []):
            req_of[n["to"]] = req_of.get(n["from"], n["from"])
        for r in q.get("redirects", []):
            req_of[r["to"]] = req_of.get(r["from"], r["from"])
        for p in q.get("pages", []):
            cur = pages_by_title.setdefault(p.get("title", ""), {})
            for k, v in p.items():
                if k != "extract":
                    cur[k] = v
            if p.get("extract"):
                cur["extract"] = cur.get("extract", "") + p["extract"]
        cont = data.get("continue") or {}
        if not cont:
            break

    articles: list[Article] = []
    for t, p in pages_by_title.items():
        if p.get("missing") or "disambiguation" in (p.get("pageprops") or {}):
            continue
        extract = p.get("extract")
        if not extract:
            continue
        cat = title_to_cat.get(req_of.get(t, t))
        if cat is None:
            continue   # couldn't trace back to a requested title — shouldn't happen
        articles.append(Article(
            title=p.get("title", t),
            text=clean_wikipedia_text(extract),
            category=cat,
            url=p.get("fullurl", ""),
            competition=_competition_for(cat),
        ))
    return articles


def _harvest_bulk_concurrent(
    items: list[tuple["Category", str]] | list[tuple[str, "Category"]],
    *,
    workers: int,
    checkpoint_path: Optional[Path],
    checkpoint_every: int,
    batch_size: int = _EXTRACTS_BATCH_SIZE,
    verbose: bool,
) -> list[Article]:
    """Fetch ``[(title, category), …]`` concurrently in batched extract requests.

    Splits the work into ``batch_size`` batches run over a thread pool,
    with a single second pass over any batch whose request hard-failed (a
    rate-limit casualty). Checkpoints the running corpus every ``checkpoint_every``
    articles so a crash is never total loss.
    """
    from concurrent.futures import ThreadPoolExecutor

    batches = [items[i:i + batch_size]
               for i in range(0, len(items), batch_size)]

    def _run(batch):
        """Return (articles, failed_batch_or_None)."""
        try:
            result = _fetch_extracts_batch(batch, verbose=verbose)
            time.sleep(random.uniform(1, 3.5))  # <--- ADD THIS: Random stagger (0.5s to 1.5s)
            return result, None
        except Exception as exc:   # whole-batch request failure → retry-able
            if verbose:
                print(f"    batch failed ({len(batch)} titles): {exc!r} — will retry")
            time.sleep(random.uniform(6.0, 10.0))  # <--- ADD THIS: Random backoff on failure
            return [], batch

    def _drain(batch_list, label):
        out: list[Article] = []
        failed: list = []
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for arts, fail in pool.map(_run, batch_list):
                out.extend(arts)
                if fail:
                    failed.append(fail)
                done += 1
                if verbose and done % 10 == 0:
                    print(f"  ... {label}: {done}/{len(batch_list)} batches "
                          f"({len(out)} articles)")
                if (checkpoint_path is not None and out
                        and len(out) % checkpoint_every < batch_size
                        and len(out) >= checkpoint_every):
                    save_raw_corpus(out, checkpoint_path)
        return out, failed

    articles, failed = _drain(batches, "fetching")
    if failed:
        if verbose:
            print(f"\nSecond pass: retrying {len(failed)} failed batches "
                  f"({sum(len(b) for b in failed)} titles) on a calmer connection…")
        recovered, still_failed = _drain(failed, "second pass")
        articles.extend(recovered)
        if verbose and still_failed:
            print(f"  {sum(len(b) for b in still_failed)} titles still unreachable — skipped.")

    if checkpoint_path is not None and articles:
        save_raw_corpus(articles, checkpoint_path)
    return articles


def _fetch_redirects_batch(titles: list[str], *,
                           verbose: bool = False) -> dict[str, tuple[str, ...]]:
    """Fetch redirect aliases for many titles via batched ``prop=redirects``.

    Returns ``{resolved_title: (alias, …)}`` capped at ``_MAX_ALIASES``. Aliases
    are pure enrichment, so a failed batch simply yields none for its titles.
    """
    from .category_seeds import _api_get

    _REDIR_BATCH = 50   # no extracts here → the higher 50-title limit applies
    out: dict[str, tuple[str, ...]] = {}
    for i in range(0, len(titles), _REDIR_BATCH):
        chunk = titles[i:i + _REDIR_BATCH]
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "prop": "redirects", "rdlimit": "max", "rdnamespace": "0",
            "redirects": "1", "titles": "|".join(chunk),
        }
        try:
            data = _api_get(params)
        except Exception as exc:   # noqa: BLE001 — aliases are optional
            if verbose:
                print(f"    (redirects batch failed: {exc!r})")
            continue
        for p in data.get("query", {}).get("pages", []):
            t = p.get("title")
            rs = [r.get("title") for r in (p.get("redirects") or []) if r.get("title")]
            if t and rs:
                out[t] = tuple(rs[:_MAX_ALIASES])
    return out


def fetch_articles_from_categories(
    categories: list[Category] | None = None,
    *,
    cache_path: Optional[Path] = None,
    max_per_category: int = 500,
    max_depth: int = 0,
    sleep_seconds: float = 0.3,
    fetch_aliases: bool = True,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 250,
    harvest_workers: int = _HARVEST_WORKERS_DEFAULT,
    batch_size: int = _EXTRACTS_BATCH_SIZE,
    verbose: bool = True,
) -> list[Article]:
    """Fetch Wikipedia articles seeded from the MediaWiki category graph.

    Drop-in alternative to ``fetch_articles()``. Instead of consuming the
    hand-curated ``TOPIC_SEEDS`` (~95 titles total), this calls the
    MediaWiki ``categorymembers`` API via ``category_seeds.harvest_titles``
    and then fetches the ~500-2000 titles per category through the fast
    **batched + concurrent** extracts path (``_harvest_bulk_concurrent`` →
    ``_fetch_extracts_batch``): full plaintext for 20 titles per request, over
    a small thread pool, instead of one ``wikipedia.page()`` call per title.

    A title appearing under multiple categories is kept under its first
    occurrence in ``categories`` order — same dedup policy as
    ``_dedupe_seeds`` so the corpus never has two copies of the same
    article wearing different category tags.

    Args:
        categories: subset of categories to fetch (default: all four).
        cache_path: optional JSON file for the harvested title list.
            Cache policy is monotonic — categories present in the file
            are reused, missing categories are harvested fresh.
        max_per_category: cap on titles per seed-category before fetching.
        max_depth: subcategory recursion depth for the harvester. 0 means
            "this category only" (the safe default — depth >= 2 produces
            tens of thousands of titles).
        sleep_seconds: retained for API compatibility; the batched path needs no
            per-title sleep (batching already cuts the request count ~20x), so it
            is unused here.
        fetch_aliases: when True (default), attach redirect titles as
            ``Article.aliases`` — but only for the curated ``CONCEPT_TITLES``,
            fetched in one batched ``prop=redirects`` query. Aliases matter most
            for the proven gap concepts (e.g. "Dr. Drake Ramoray"); the bulk
            titles skip them. Set False to skip aliases entirely.
        checkpoint_path: if set, the running article list is rewritten to this
            path periodically, so a mid-crawl failure leaves a durable partial
            corpus instead of losing the whole harvest.
        checkpoint_every: article interval between checkpoint writes (default 250).
        harvest_workers: concurrent extract batches (default
            ``_HARVEST_WORKERS_DEFAULT``). Higher = faster but less polite.
        batch_size: titles per extract request (default ``_EXTRACTS_BATCH_SIZE``,
            the anonymous MediaWiki cap of 20). Lower values mean more requests
            but smaller response bodies — useful if the API returns truncated
            extracts on large batches.
        verbose: print progress.

    Returns:
        List of Article objects. Failed fetches skipped with a warning.
    """
    # Local import keeps the dependency cycle clean: category_seeds is a
    # leaf module that imports nothing from corpus / chunker / index.
    from .category_seeds import harvest_titles, CONCEPT_TITLES

    import wikipedia  # lazy — only needed when this function actually runs

    _configure_wikipedia(wikipedia)
    targets = categories or list(Category)

    # Step 1: title harvest (cache-aware).
    harvested = harvest_titles(
        categories=targets,
        cache_path=cache_path,
        max_per_category=max_per_category,
        max_depth=max_depth,
        verbose=verbose,
    )

    # Step 1b: prepend the explicit concept titles (guaranteed inclusion).
    # These bypass the category caps and lead the per-category list so they win
    # the cross-category dedup below and are fetched first. Merged here (not in
    # harvest_titles) so they're never lost to the harvested-titles cache.
    # Track them so only these (not the bulk titles) pay the redirect-fetch call.
    curated_titles: set[str] = set()
    for cat in targets:
        explicit = CONCEPT_TITLES.get(cat, [])
        if explicit:
            curated_titles.update(explicit)
            harvested[cat] = explicit + harvested.get(cat, [])

    # Step 2: flatten with cross-category dedup. First category in
    # ``targets`` wins for any duplicate title.
    seen: set[str] = set()
    flat: list[tuple[str, Category]] = []
    for cat in targets:
        for title in harvested.get(cat, []):
            if title in seen:
                if verbose:
                    print(f"  ! '{title}' duplicate across categories — kept under earlier topic")
                continue
            seen.add(title)
            flat.append((title, cat))

    if verbose:
        print(f"\nFetching bodies for {len(flat)} unique articles…")
        for cat in targets:
            n = sum(1 for _, c in flat if c == cat)
            print(f"  [{cat.value}] {n} articles to fetch")

    # Step 3: fetch bodies through the fast batched + concurrent extracts path
    # (~20x fewer round-trips than per-title, plus thread-pool concurrency).
    articles = _harvest_bulk_concurrent(
        flat, workers=harvest_workers, batch_size=batch_size,
        checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every,
        verbose=verbose,
    )

    # Step 3b: enrich just the curated concept titles with redirect aliases, in
    # one batched prop=redirects query (alt-phrasing matters most there).
    if fetch_aliases and curated_titles:
        curated_present = [a.title for a in articles if a.title in curated_titles]
        if curated_present:
            if verbose:
                print(f"\nFetching redirect aliases for {len(curated_present)} curated titles…")
            alias_map = _fetch_redirects_batch(curated_present, verbose=verbose)
            if alias_map:
                from dataclasses import replace as _replace
                articles = [
                    _replace(a, aliases=alias_map[a.title]) if a.title in alias_map else a
                    for a in articles
                ]
                if checkpoint_path is not None and articles:
                    save_raw_corpus(articles, checkpoint_path)

    if verbose:
        print(f"\nFetched {len(articles)} / {len(flat)} articles successfully.")
    return articles


def fetch_articles_by_title(
    titles_by_category: dict[Category, list[str]],
    *,
    existing_titles: Optional[set[str]] = None,
    fetch_aliases: bool = True,
    resolve: bool = True,
    sleep_seconds: float = 0.3,
    verbose: bool = True,
) -> list[Article]:
    """Fetch an explicit, per-category set of article titles via ``_fetch_one``.

    Used for the log-mined gap queue (``build_rag_index.py --gap-queue``): a
    ``{Category: [title, ...]}`` map of articles to add directly, reusing the
    same retry/disambiguation/cleanup pipeline as the category crawl. Titles in
    ``existing_titles`` (already in the index) and cross-title duplicates are
    skipped. Failed fetches are skipped with a warning.

    When ``resolve`` is True (default) each title is first canonicalised via
    ``wikipedia.search(title, results=1)`` and the top hit is fetched instead of
    the raw string. Gap candidates mined from question text are often fragments
    ("Beatles To", "does the relationship") or casing variants ("the bystander
    effect"); resolving turns the valid ones into real titles and drops the rest
    *before* they reach ``wikipedia.page()`` — so no fragment triggers a
    disambiguation/404 storm. Set False to fetch the literal strings.
    """
    import wikipedia  # lazy — only needed when this actually runs
    _configure_wikipedia(wikipedia)
    can_search = resolve and hasattr(wikipedia, "search")

    existing = set(existing_titles or ())
    seen: set[str] = set()
    out: list[Article] = []
    cooldown = _make_failure_cooldown(verbose)
    for cat, titles in titles_by_category.items():
        for title in titles:
            if title in seen or title in existing:
                continue
            seen.add(title)
            fetch_title = title
            if can_search:
                try:
                    hits = wikipedia.search(title, results=1)
                except Exception:
                    hits = []
                if not hits:
                    if verbose:
                        print(f"  ! gap title '{title}' → no search hit — skipped")
                    continue
                fetch_title = hits[0]
                if fetch_title in existing or fetch_title in seen:
                    continue   # resolved to an article we already have / queued
                seen.add(fetch_title)
            try:
                art = _fetch_one(fetch_title, cat, verbose=verbose, fetch_aliases=fetch_aliases)
            except Exception as exc:   # belt-and-braces — see fetch_articles_from_categories
                if verbose:
                    print(f"  ! unhandled error for '{fetch_title}': {exc!r} — skipped")
                art = None
            if art is not None:
                out.append(art)
            cooldown(art is not None)   # pause if empty-body failures cluster
            time.sleep(sleep_seconds)
    if verbose:
        n_req = sum(len(v) for v in titles_by_category.values())
        print(f"Gap queue: fetched {len(out)} / {n_req} requested titles.")
    return out


# ── Persistence ───────────────────────────────────────────────────────────────

def save_raw_corpus(articles: list[Article], path: Path) -> None:
    """Persist raw articles to JSONL. Safe to overwrite.

    ``aliases``/``competition`` are written only when non-empty so corpora with
    no enrichment stay byte-for-byte compatible with the v3 reader.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in articles:
            row = {
                "title": a.title, "text": a.text,
                "category": a.category.value, "url": a.url,
            }
            if a.aliases:
                row["aliases"] = list(a.aliases)
            if a.competition:
                row["competition"] = a.competition
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(articles)} articles → {path}")


def load_raw_corpus(path: Path) -> list[Article]:
    """Load previously saved raw corpus from JSONL.

    Tolerant of both v3 rows (title/text/category/url only) and v4 rows that
    add ``aliases``/``competition`` — missing fields fall back to their
    dataclass defaults.
    """
    articles = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            articles.append(Article(
                title=d["title"], text=d["text"],
                category=Category(d["category"]), url=d.get("url", ""),
                aliases=tuple(d.get("aliases", ())),
                competition=d.get("competition", ""),
            ))
    return articles