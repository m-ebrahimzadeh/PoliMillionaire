"""Mine run logs for corpus gaps -> a fetch queue of Wikipedia titles.

The bot's own run logs are the highest-precision signal for *what the corpus is
missing*: every question it got wrong, or where offline retrieval was gated out,
names a concept the index failed to cover. This script reads those JSONL logs,
keeps the gap questions, extracts candidate Wikipedia titles from each, and
writes a per-category ``gap_titles.json`` that ``build_rag_index.py --gap-queue``
fetches into the corpus -- a self-correcting loop on top of the static seeds.

Gap criteria (a question is a gap when ANY holds):
  * ``correct`` is False                          - the model answered wrong
  * ``extras.gated_by_min_score`` is True         - offline retrieval was gated
  * ``extras.top_score`` < ``--top-score-floor``  - retrieval support was weak

NEWS (handled by the live Guardian path) and MATHS (procedural / out of scope)
are excluded.

Usage
-----
    # Offline: emit extracted candidate phrases (many are already valid titles)
    python scripts/mine_corpus_gaps.py run1.jsonl run2.jsonl --out data/cache/gap_titles.json

    # Online: canonicalise candidates to real article titles via Wikipedia search
    python scripts/mine_corpus_gaps.py run*.jsonl --out data/cache/gap_titles.json --resolve
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

# Make the package importable when run as a bare script from the repo root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polimibot.config import CATEGORIES, Category  # noqa: E402


# Competitions whose gaps we do NOT backfill into the static corpus.
_EXCLUDED_COMPETITIONS = {"News", "Maths"}


def _competition_to_category(name: str) -> Optional[Category]:
    """Map a server competition display name to its Category, or None."""
    for info in CATEGORIES.values():
        if info.display_name == name:
            return info.category
    return None


def is_gap(record: dict, *, top_score_floor: float) -> bool:
    """True when a question record signals a corpus gap (see module docstring)."""
    if record.get("run_kind") != "question":
        return False
    if record.get("competition_name") in _EXCLUDED_COMPETITIONS:
        return False
    if record.get("correct") is False:
        return True
    extras = record.get("extras") or {}
    if extras.get("gated_by_min_score") is True:
        return True
    top = extras.get("top_score")
    if isinstance(top, (int, float)) and top < top_score_floor:
        return True
    return False


# A quoted span: 'single' or "double" — these almost always name the concept
# ("will to power", "bystander effect", "Dr. Drake Ramoray").
_QUOTED_RE = re.compile(r"['\"]([^'\"]{3,60})['\"]")

# A run of Capitalised words (proper nouns / titles): "Roman Empire",
# "Pink Floyd", "Carl Rogers", "Ariarathes V". Allows interior of/the/and.
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9.]+(?:\s+(?:of|the|and|in|de|von)\s+|\s+)"
    r"(?:[A-Z][a-zA-Z0-9.]+|[IVX]+)\b)+"
)

# Lone capitalised technical terms worth keeping even when not multi-word.
_SINGLE_CAP_RE = re.compile(r"\b([A-Z][a-z]{4,})\b")

# Lowercase concept noun-phrases named by a tell-tale suffix. Catches the many
# questions whose key concept is lowercase ("the bystander effect", "water
# cycle", "just-world hypothesis", "domino theory") that the capitalisation/
# quote heuristics miss. Wikipedia search then canonicalises the casing.
_CONCEPT_SUFFIX_RE = re.compile(
    r"\b((?:[a-z][a-z'-]+\s+){1,2}"
    r"(?:effect|theory|hypothesis|fallacy|syndrome|principle|paradox|dilemma|"
    r"bias|disorder|cycle|reaction|law|imperative|dissonance|relationship))\b"
)

# Leading filler stripped off a matched concept phrase ("describes the bystander
# effect" → "bystander effect"; "the water cycle" → "water cycle").
_CONCEPT_LEADING_FILLER = {
    "the", "a", "an", "of", "this", "that", "its", "his", "her", "their",
    "which", "describes", "best", "concept", "term", "fundamental",
}


def _strip_leading_filler(phrase: str) -> str:
    toks = phrase.split()
    while len(toks) > 1 and toks[0].lower() in _CONCEPT_LEADING_FILLER:
        toks = toks[1:]
    return " ".join(toks)

_STOP_PHRASES = {
    "which", "what", "how", "the following", "best describes", "according to",
}


def extract_candidates(question_text: str, options: Iterable[str]) -> list[str]:
    """Pull candidate Wikipedia-title phrases out of a question + its options.

    Pure and deterministic — quoted spans, proper-noun runs, and salient
    capitalised terms from both the stem and the options, deduped in order.
    Resolution to canonical titles is a separate (optional, online) step.
    """
    text = question_text or ""
    opts = " ".join(o for o in options if o)
    cands: list[str] = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        p = phrase.strip().strip(".,;:!?").strip()
        key = p.lower()
        if len(p) < 3 or key in _STOP_PHRASES or key in seen:
            return
        seen.add(key)
        cands.append(p)

    for m in _QUOTED_RE.finditer(text):
        _add(m.group(1))
    for m in _PROPER_NOUN_RE.finditer(f"{text} {opts}"):
        _add(m.group(0))
    for m in _CONCEPT_SUFFIX_RE.finditer(text.lower()):
        _add(_strip_leading_filler(m.group(1)))
    for m in _SINGLE_CAP_RE.finditer(text):
        _add(m.group(1))
    return cands


def load_gap_candidates(
    paths: list[Path], *, top_score_floor: float,
) -> dict[Category, list[str]]:
    """Read JSONL logs → ``{Category: [candidate title, ...]}`` for gap questions.

    Deduped per category, preserving first-seen order.
    """
    out: dict[Category, list[str]] = {}
    seen: dict[Category, set[str]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate non-JSONL lines (pretty-printed logs)
                if not is_gap(rec, top_score_floor=top_score_floor):
                    continue
                cat = _competition_to_category(rec.get("competition_name", ""))
                if cat is None:
                    continue
                bucket = out.setdefault(cat, [])
                seent = seen.setdefault(cat, set())
                for cand in extract_candidates(rec.get("question_text", ""),
                                               rec.get("options", []) or []):
                    if cand.lower() not in seent:
                        seent.add(cand.lower())
                        bucket.append(cand)
    return out


def resolve_titles(phrases: list[str], *, verbose: bool = False) -> list[str]:
    """Canonicalise candidate phrases to real article titles via Wikipedia
    search (online). Unresolved phrases are dropped. Deduped, order-preserving."""
    import wikipedia
    wikipedia.set_lang("en")
    wikipedia.set_user_agent("PoliMillionaire-RAG/1.0 (contact: ebrahimzadeh.meh@gmail.com)")
    out: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        try:
            hits = wikipedia.search(phrase, results=1)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  ! search failed for {phrase!r}: {exc}")
            continue
        if hits and hits[0].lower() not in seen:
            seen.add(hits[0].lower())
            out.append(hits[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path, help="run JSONL log file(s)")
    ap.add_argument("--out", type=Path, default=Path("data/cache/gap_titles.json"))
    ap.add_argument("--top-score-floor", type=float, default=0.5,
                    help="treat retrieval below this top_score as a gap (default 0.5)")
    ap.add_argument("--resolve", action="store_true",
                    help="canonicalise candidates to real titles via Wikipedia search (online)")
    args = ap.parse_args()

    by_cat = load_gap_candidates(args.logs, top_score_floor=args.top_score_floor)

    serialisable: dict[str, list[str]] = {}
    for cat, phrases in by_cat.items():
        titles = resolve_titles(phrases, verbose=True) if args.resolve else phrases
        if titles:
            serialisable[cat.value] = titles
        print(f"[{cat.value}] {len(phrases)} candidates -> {len(titles)} titles")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    total = sum(len(v) for v in serialisable.values())
    print(f"\nWrote {total} gap titles across {len(serialisable)} categories -> {args.out}")


if __name__ == "__main__":
    main()
