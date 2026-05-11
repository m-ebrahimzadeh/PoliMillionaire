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
    """
    import wikipedia  # lazy import — only needed at corpus-build time

    wikipedia.set_lang("en")
    targets = categories or list(TOPIC_SEEDS.keys())
    articles: list[Article] = []

    for cat in targets:
        seeds = TOPIC_SEEDS[cat]
        if verbose:
            print(f"\n[{cat.value}] fetching {len(seeds)} articles…")

        for title in seeds:
            article = _fetch_one(title, cat, verbose=verbose)
            if article is not None:
                articles.append(article)
            time.sleep(sleep_seconds)

    if verbose:
        print(f"\nFetched {len(articles)} articles total.")
    return articles


def _fetch_one(title: str, category: Category, *, verbose: bool) -> Optional[Article]:
    """Fetch a single Wikipedia page. Returns None on hard failure."""
    import wikipedia

    try:
        page = wikipedia.page(title, auto_suggest=False)
        return Article(title=page.title, text=clean_wikipedia_text(page.content),
                       category=category, url=page.url)

    except wikipedia.DisambiguationError as e:
        # e.g. "Mercury" → try the first unambiguous option
        if verbose:
            print(f"  ! disambiguation for '{title}', trying '{e.options[0]}'")
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return Article(title=page.title, text=clean_wikipedia_text(page.content),
                           category=category, url=page.url)
        except Exception:
            pass

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