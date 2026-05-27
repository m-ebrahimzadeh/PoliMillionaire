"""Wikipedia category-graph harvester.

Bridges the gap between the hand-curated ``TOPIC_SEEDS`` in ``corpus.py``
(~95 articles total, breadth-limited by what a human can list) and the
real coverage needed for trivia retrieval (hundreds to thousands of
entity articles per topic).

The MediaWiki ``categorymembers`` API exposes Wikipedia's category graph
for free, no auth required. Seeding from ~20 well-chosen categories per
topic yields ~500-2000 article titles per category — enough that the
"right article isn't in the index" failure mode largely disappears.

Design notes
────────────
- **Output is titles only.** This module does NOT fetch article bodies.
  Hand it to ``corpus.fetch_articles_from_categories()`` which reuses
  the existing ``_fetch_one`` machinery (retry, disambiguation,
  citation cleanup) per title.

- **Subcategories are NOT recursed by default.** ``Category:World_War_II``
  contains 50+ subcategories (battles, leaders, weapons, ...); a naive
  recursion balloons the title list into the tens of thousands and most
  subcat content is off-topic for *trivia*. Pass ``max_depth > 0`` if you
  want it; in practice ``max_depth=1`` for very narrow seed categories
  ("Category:Chemical_elements") is the sweet spot.

- **Per-category cap.** Even with ``cmtype=page`` some categories have
  10k+ members. Cap at ``max_per_category`` (default 500), taking the
  first N as MediaWiki returns them. The default sort is alphabetical by
  sortkey, which tends to put canonical/short-titled articles first.

- **JSON-on-disk cache with partial-coverage refresh.** The cache file
  is a per-category dict. On load, any category present in the cache is
  reused; any category missing is harvested fresh and merged back in.
  The cache grows monotonically across runs.

- **Polite by default.** Sets an informative ``User-Agent`` per the
  Wikimedia API:Etiquette guideline; sleeps between paginated requests
  and retries transient failures with exponential back-off.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import urllib.error
import urllib.parse
import urllib.request

from ..config import Category


# Wikimedia's API:Etiquette requires a contact-info-bearing User-Agent.
# An IP can be silently rate-limited or blocked without it.
_USER_AGENT = (
    "PoliMillionaire-RAG/1.0 "
    "(https://github.com/your-org/polimibot; "
    "contact: ebrahimzadeh.meh@gmail.com)"
)

# MediaWiki caps cmlimit at 500 for non-bot accounts; we use that as the
# pagination step size. Lower values just mean more round-trips.
_CM_PAGE_SIZE = 500

# Polite delay between paginated requests to the same category. 0.2 s is
# well under the documented limits but keeps us a good citizen.
_REQUEST_DELAY = 0.2

# Default per-category title cap. Larger categories ("World War II",
# "Chemical elements") have thousands of members; the long tail is
# almost always off-topic for trivia. Override per category via
# ``CATEGORY_SEEDS``'s tuple-form entries when a category genuinely
# warrants a deeper crawl.
_DEFAULT_PER_CATEGORY_CAP = 500

# Retry policy for transient MediaWiki API failures (empty body, 5xx,
# JSON parse error, transient network blip). Matches the resilience of
# corpus._page_with_retry so a single bad response doesn't sink a whole
# category harvest.
_API_MAX_ATTEMPTS = 3
_API_BACKOFF = (1.0, 2.0, 4.0)


# ── Seed categories per topic ─────────────────────────────────────────────
# Categories were chosen to maximise *entity coverage for trivia*, not
# encyclopaedic completeness. Prefer categories of named entities (people,
# works, elements, events) over conceptual ones ("Category:Physics" pulls
# in thousands of theoretical articles with poor question-answering value).
#
# Each entry is either:
#   - "Category_Name"                   → use default per-category cap
#   - ("Category_Name", explicit_cap)   → override the cap for this seed
#
# Underscores in category names are required by the MediaWiki API.

CATEGORY_SEEDS: dict[Category, list] = {
    Category.HISTORY: [
        "Roman_emperors",
        "Ancient_Roman_generals",
        "Pharaohs",
        "Ancient_Greek_generals",
        "Ancient_battles",
        ("Battles_of_World_War_II", 300),
        ("Battles_of_World_War_I", 200),
        "Medieval_English_monarchs",
        "Kings_of_France",
        "Founding_Fathers_of_the_United_States",
        "Presidents_of_the_United_States",
        "Prime_Ministers_of_the_United_Kingdom",
        "Byzantine_emperors",
        "Ottoman_sultans",
        "Mongol_Empire",
        "Crusades",
        ("Treaties_of_the_19th_century", 200),
        ("Treaties_of_the_20th_century", 200),
        "Explorers",
        "Renaissance_artists",
    ],
    Category.SCIENCE: [
        "Chemical_elements",
        "Nobel_laureates_in_Physics",
        "Nobel_laureates_in_Chemistry",
        "Nobel_laureates_in_Physiology_or_Medicine",
        "Planets_of_the_Solar_System",
        "Moons_of_the_Solar_System",
        "Constellations",
        ("Mammals", 300),
        ("Birds", 300),
        "Human_organ_systems",
        "Human_anatomy",
        "Infectious_diseases",
        "Subatomic_particles",
        "Fundamental_physics_concepts",
        "Branches_of_biology",
        "Branches_of_chemistry",
        "Geological_periods",
        "Plate_tectonics",
        "Famous_inventions",
        "Inventors",
    ],
    Category.ENTERTAINMENT: [
        ("Best_Picture_Academy_Award_winners", 200),
        ("Academy_Award_for_Best_Director_winners", 200),
        ("Academy_Award_for_Best_Actor_winners", 200),
        ("Academy_Award_for_Best_Actress_winners", 200),
        "Grammy_Award_for_Album_of_the_Year",
        "Grammy_Award_for_Record_of_the_Year",
        ("Rock_and_Roll_Hall_of_Fame_inductees", 300),
        "American_sitcoms",
        "British_sitcoms",
        "HBO_original_programming",
        ("Films_directed_by_Steven_Spielberg", 100),
        ("Films_directed_by_Martin_Scorsese", 100),
        ("Films_directed_by_Stanley_Kubrick", 100),
        ("Films_directed_by_Alfred_Hitchcock", 100),
        "The_Beatles_albums",
        "Pink_Floyd_albums",
        "Queen_(band)_albums",
        "Walt_Disney_Animation_Studios_films",
        "Pixar_films",
        "James_Bond_films",
    ],
    # MATHS deliberately receives a smaller seed list: trivia maths is
    # procedural (compute the p-value, evaluate the sum), not factual.
    # The audit recommends routing MATHS through the tool/agent path
    # entirely; the corpus here exists only to support the small minority
    # of definitional questions ("Who invented calculus?").
    Category.MATHS: [
        "Fields_Medalists",
        "Abel_Prize_laureates",
        ("Mathematical_constants", 100),
        ("Theorems_in_geometry", 200),
        ("Theorems_in_number_theory", 200),
        "Mathematicians_by_century",
        "Number_systems",
    ],
    Category.PHILOSOPHY: [
        "Ancient_Greek_philosophers",
        "Ancient_Roman_philosophers",
        "Medieval_philosophers",
        "Modern_philosophers",
        "Continental_philosophers",
        "Analytic_philosophers",
        "Existentialist_philosophers",
        "Stoicism",
        "Branches_of_philosophy",
        "Ethical_theories",
        "Metaphysics",
        "Epistemology",
        "Schools_of_psychology",
        "Cognitive_scientists",
        "Psychoanalysts",
        "Behaviorism",
        ("Cognitive_biases", 200),
        ("Psychological_experiments", 200),
        "Founders_of_psychology",
        "Theories_in_psychology",
    ],
    # NEWS is the weakest category for Wikipedia: recent events are sparse,
    # late, and POV-flagged. Seed a small static corpus of long-running
    # news anchors (prize winners, "person of the year" lists, major events
    # by decade) and lean on the live-search fallback in
    # rag/live_search.py for anything time-sensitive.
    Category.NEWS: [
        ("Pulitzer_Prize_for_Public_Service_winners", 200),
        ("Pulitzer_Prize_winners", 200),
        "Time_Persons_of_the_Year",
        "Nobel_Peace_Prize_laureates",
        ("Major_news_events_of_the_21st_century", 200),
        ("Years_in_politics", 200),
        "G7_summits",
        "United_Nations_Secretaries-General",
        "Heads_of_state_of_the_European_Union",
        "Political_scandals",
    ],
}


# ── Public API ───────────────────────────────────────────────────────────

def harvest_titles(
    categories: Optional[Iterable[Category]] = None,
    *,
    cache_path: Optional[Path] = None,
    max_per_category: int = _DEFAULT_PER_CATEGORY_CAP,
    max_depth: int = 0,
    verbose: bool = True,
) -> dict[Category, list[str]]:
    """Resolve ``CATEGORY_SEEDS`` to a deduplicated title list per topic.

    Cache policy is monotonic: categories present in the cache file are
    reused; categories missing from it are harvested fresh and merged
    back in. Delete the file to force a full refresh.

    Args:
        categories: which topics to harvest. Default: all four.
        cache_path: optional JSON file to read from / write to.
        max_per_category: default cap on titles per seed-category. Tuple
            entries in ``CATEGORY_SEEDS`` override this per-category.
        max_depth: subcategory recursion depth. ``0`` (default) means
            "this category only". ``1`` walks one level of subcats.
            Beware: depth>=2 explodes the title list.
        verbose: print progress per seed-category.

    Returns:
        ``{Category.HISTORY: ["Julius Caesar", "Augustus", ...], ...}``
        Cross-topic duplicates are NOT removed here — the caller
        (``fetch_articles_from_categories``) handles cross-category
        deduplication so the assignment of an ambiguous title to one
        topic is deterministic.
    """
    targets = list(categories) if categories is not None else list(CATEGORY_SEEDS.keys())

    # Load the existing cache (if any) and figure out which categories
    # we still need to harvest.
    cached: dict[Category, list[str]] = {}
    if cache_path is not None and cache_path.is_file():
        cached = _load_cache(cache_path)

    cached_targets = [c for c in targets if c in cached]
    missing_targets = [c for c in targets if c not in cached]

    if verbose and cache_path is not None:
        if cached_targets and missing_targets:
            cached_names = ", ".join(c.value for c in cached_targets)
            missing_names = ", ".join(c.value for c in missing_targets)
            print(f"Cache: hit for [{cached_names}]; refreshing [{missing_names}].")
        elif cached_targets:
            print(f"Cache hit for all targets — using {cache_path}")
        elif cache_path.is_file():
            print(f"Cache present but covers none of the requested categories — refreshing.")

    out: dict[Category, list[str]] = {c: list(cached[c]) for c in cached_targets}

    for cat in missing_targets:
        seeds = CATEGORY_SEEDS.get(cat, [])
        if verbose:
            print(f"\n[{cat.value}] harvesting from {len(seeds)} seed categories…")

        titles: list[str] = []
        seen: set[str] = set()
        for seed in seeds:
            # Normalise seed entries: either bare string or (name, cap) tuple.
            if isinstance(seed, tuple):
                cat_name, cap = seed
            else:
                cat_name, cap = seed, max_per_category

            try:
                fetched = _fetch_category_members(
                    cat_name, limit=cap, max_depth=max_depth, verbose=verbose,
                )
            except Exception as exc:  # noqa: BLE001 — never abort the whole harvest
                if verbose:
                    print(f"  ! failed to fetch Category:{cat_name}: {exc}")
                continue

            kept = 0
            for title in fetched:
                if title in seen:
                    continue
                seen.add(title)
                titles.append(title)
                kept += 1
            if verbose:
                print(f"  + Category:{cat_name}: {len(fetched)} → {kept} new (total {len(titles)})")

        out[cat] = titles
        if verbose:
            print(f"[{cat.value}] {len(titles)} unique titles harvested.")

    # Persist if we either added new categories or had no cache file yet.
    if cache_path is not None and (missing_targets or not cache_path.is_file()):
        # Merge with anything in the cache that wasn't in the targets, so
        # the cache file remains a superset across runs.
        if cache_path.is_file():
            full_cache = _load_cache(cache_path)
        else:
            full_cache = {}
        for cat, titles in out.items():
            full_cache[cat] = titles
        _save_cache(full_cache, cache_path)
        if verbose:
            print(f"\nSaved harvested titles → {cache_path}")

    return out


# ── MediaWiki API plumbing ────────────────────────────────────────────────

def _fetch_category_members(
    category: str,
    *,
    limit: int,
    max_depth: int,
    verbose: bool,
    _depth: int = 0,
) -> list[str]:
    """Page through ``list=categorymembers`` for one category.

    Returns up to ``limit`` page titles (cmtype=page; subcats and files
    are filtered out). Handles MediaWiki's continuation tokens. When
    ``max_depth > 0``, recurses into subcategories once the page list is
    exhausted, adding their pages until ``limit`` is reached.
    """
    titles: list[str] = []
    subcats: list[str] = []
    cont_params: dict[str, str] = {}

    while len(titles) < limit:
        params = {
            "action":    "query",
            "list":      "categorymembers",
            "cmtitle":   f"Category:{category}",
            # Mix page + subcat in one call so we can recurse cheaply when
            # max_depth > 0; the type field on each member tells us which.
            "cmtype":    "page|subcat" if max_depth > _depth else "page",
            "cmlimit":   str(min(_CM_PAGE_SIZE, limit - len(titles))),
            "cmprop":    "title|type",
            "format":    "json",
            "formatversion": "2",
        }
        params.update(cont_params)

        data = _api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            mtype = m.get("type", "page")
            mtitle = m.get("title", "")
            if not mtitle:
                continue
            if mtype == "subcat":
                # "Category:Foo" → "Foo" for recursion.
                subcats.append(mtitle.split(":", 1)[-1].replace(" ", "_"))
            elif mtype == "page":
                titles.append(mtitle)
                if len(titles) >= limit:
                    break

        cont = data.get("continue")
        if not cont or len(titles) >= limit:
            break
        cont_params = {k: str(v) for k, v in cont.items() if k != "continue"}
        time.sleep(_REQUEST_DELAY)

    # Optional one-level recursion into subcategories.
    if max_depth > _depth and subcats and len(titles) < limit:
        seen_in_titles: set[str] = set(titles)
        for sub in subcats:
            if len(titles) >= limit:
                break
            try:
                sub_titles = _fetch_category_members(
                    sub,
                    limit=limit - len(titles),
                    max_depth=max_depth,
                    verbose=verbose,
                    _depth=_depth + 1,
                )
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"    ! subcat fetch failed for '{sub}': {exc}")
                continue
            for t in sub_titles:
                if t not in seen_in_titles:
                    seen_in_titles.add(t)
                    titles.append(t)
                    if len(titles) >= limit:
                        break
            time.sleep(_REQUEST_DELAY)

    return titles


def _api_get(params: dict) -> dict:
    """GET against the English Wikipedia API with retry/backoff.

    Retries on URLError, HTTPError 5xx, JSONDecodeError, or empty body.
    HTTPError 4xx is re-raised immediately — those won't fix themselves.
    """
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    last_exc: Exception | None = None
    for attempt in range(_API_MAX_ATTEMPTS):
        if attempt > 0:
            delay = _API_BACKOFF[min(attempt, len(_API_BACKOFF) - 1)]
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read()
            if not body:
                last_exc = ValueError("empty response body from MediaWiki API")
                continue
            return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 4xx is a hard error; don't waste retries.
            if 400 <= exc.code < 500:
                raise
            last_exc = exc
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_exc = exc

    # All attempts exhausted.
    raise last_exc  # type: ignore[misc]


# ── Cache I/O ─────────────────────────────────────────────────────────────

def _save_cache(harvested: dict[Category, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {cat.value: titles for cat, titles in harvested.items()}
    path.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _load_cache(path: Path) -> dict[Category, list[str]]:
    """Load every category present in the cache file.

    Unknown category names (e.g. a future enum addition not yet in the
    file) are silently skipped — the caller compares the returned keys
    against its target list to decide what still needs harvesting.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[Category, list[str]] = {}
    for key, titles in raw.items():
        try:
            cat = Category(key)
        except ValueError:
            continue   # cached an unknown enum value — ignore
        out[cat] = list(titles)
    return out
