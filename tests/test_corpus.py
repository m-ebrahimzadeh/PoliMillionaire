"""Wikipedia text cleanup. Pure regex — no FAISS, no network, no GPU."""
from __future__ import annotations

import pytest

from polimibot.rag.corpus import CLEANUP_VERSION, clean_wikipedia_text


def test_clean_wiki_strips_citation_markers():
    text = "Caesar crossed the Rubicon[1] in 49 BC[2][3] before the war."
    out = clean_wikipedia_text(text)
    assert "[1]" not in out
    assert "[2][3]" not in out
    assert "Caesar crossed the Rubicon" in out


def test_clean_wiki_strips_citations_but_keeps_brackets_around_letters():
    """[1] / [42] should go; "(C)" or "[citation needed]" outside numeric
    markers should be left alone. Currently the regex is numeric-only,
    which is right for `wikipedia` library output."""
    text = "Answer (C) is correct[1]. Also [citation needed] elsewhere."
    out = clean_wikipedia_text(text)
    assert "[1]" not in out
    assert "(C)" in out
    assert "[citation needed]" in out  # untouched


def test_clean_wiki_drops_references_section():
    text = (
        "Newton was an English mathematician.\n\n"
        "== Early life ==\n"
        "Newton was born in 1643.\n\n"
        "== References ==\n"
        "1. Smith, J. (2020). Newton.\n"
    )
    out = clean_wikipedia_text(text)
    assert "Newton was born in 1643" in out      # body kept
    assert "Early life" in out                   # body section header kept
    assert "References" not in out               # tail section dropped
    assert "Smith, J." not in out


@pytest.mark.parametrize("header", [
    "References", "See also", "External links", "Notes",
    "Further reading", "Bibliography", "Sources", "Citations",
    "Footnotes",
])
def test_clean_wiki_drops_each_tail_section_alias(header):
    text = f"Body text here.\n\n== {header} ==\nDrop me.\n"
    out = clean_wikipedia_text(text)
    assert "Drop me" not in out, f"failed to drop section: {header}"
    assert "Body text here" in out


def test_clean_wiki_only_first_tail_section_truncates():
    """The earliest tail section wins — subsequent ones inside the cut don't matter."""
    text = (
        "Body.\n\n"
        "== See also ==\n"
        "Goes away.\n\n"
        "== References ==\n"
        "Also goes away.\n"
    )
    out = clean_wikipedia_text(text)
    assert out.strip().endswith("Body.")


def test_clean_wiki_is_idempotent():
    text = "First pass[1].\n\n== References ==\nDrop.\n"
    once = clean_wikipedia_text(text)
    twice = clean_wikipedia_text(once)
    assert once == twice


def test_clean_wiki_handles_empty():
    assert clean_wikipedia_text("") == ""
    assert clean_wikipedia_text(None) is None  # type: ignore[arg-type]


def test_clean_wiki_collapses_excess_whitespace():
    text = "Line one.\n\n\n\n\nLine two.    Triple   spaces."
    out = clean_wikipedia_text(text)
    assert "\n\n\n" not in out
    assert "   " not in out


def test_cleanup_version_is_positive_int():
    """CLEANUP_VERSION is recorded in the index manifest so a downstream
    caller can detect stale corpora when the regex set changes."""
    assert isinstance(CLEANUP_VERSION, int)
    assert CLEANUP_VERSION >= 1
