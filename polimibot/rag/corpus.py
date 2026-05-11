"""Wikipedia corpus builder. Fetch once, chunk as many times as you like.

Separation of concerns:
  - This module: knows *what* to fetch and how to persist raw text.
  - chunker.py:  knows *how* to split text into retrieval units.
  - index.py:    knows *how* to store and search embeddings.
"""
from __future__ import annotations

import json
import re
import time
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
CORPUS_VERSION = 2

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
    """One Wikipedia article — the raw unit before chunking."""
    title: str
    text: str         # full article text (may be many thousands of words)
    category: Category
    url: str = ""


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

    wikipedia.set_lang("en")
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


def _fetch_one(title: str, category: Category, *, verbose: bool) -> Optional[Article]:
    """Fetch a single Wikipedia page. Returns None on hard failure.

    On a DisambiguationError, walks the option list (instead of blindly
    picking the first) and returns the first option whose page summary
    contains at least one keyword from the seed title.
    """
    import wikipedia

    try:
        page = wikipedia.page(title, auto_suggest=False)
        return Article(title=page.title, text=clean_wikipedia_text(page.content),
                       category=category, url=page.url)

    except wikipedia.DisambiguationError as e:
        keywords = _seed_keywords(title)
        for option in e.options[:5]:   # cap — disambiguation pages can be huge
            try:
                page = wikipedia.page(option, auto_suggest=False)
            except Exception:
                continue
            # Use the first paragraph (cheap) as the relevance check; full
            # body is fine too but more work, and the summary usually
            # contains the entity nouns we care about.
            preview = (page.summary or page.content[:500]).lower()
            if not keywords or any(k in preview for k in keywords):
                if verbose and option != title:
                    print(f"  ! disambiguation for '{title}' → resolved to '{option}'")
                return Article(title=page.title,
                               text=clean_wikipedia_text(page.content),
                               category=category, url=page.url)
        if verbose:
            print(f"  ! disambiguation for '{title}' — no relevant option found, skipped")

    except wikipedia.PageError:
        if verbose:
            print(f"  ! page not found: '{title}' — skipped")

    except Exception as exc:
        if verbose:
            print(f"  ! unexpected error for '{title}': {exc} — skipped")

    return None


# ── Persistence ───────────────────────────────────────────────────────────────

def save_raw_corpus(articles: list[Article], path: Path) -> None:
    """Persist raw articles to JSONL. Safe to overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps({
                "title": a.title, "text": a.text,
                "category": a.category.value, "url": a.url,
            }, ensure_ascii=False) + "\n")
    print(f"Saved {len(articles)} articles → {path}")


def load_raw_corpus(path: Path) -> list[Article]:
    """Load previously saved raw corpus from JSONL."""
    articles = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            articles.append(Article(
                title=d["title"], text=d["text"],
                category=Category(d["category"]), url=d.get("url", ""),
            ))
    return articles