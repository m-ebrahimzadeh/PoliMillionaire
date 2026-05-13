"""Unit tests for LiveSearchFallback — no network, no wikipedia required.

All tests mock the ``wikipedia`` module so they run fully offline and are
safe in CI environments without internet access.
"""
from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from polimibot.config import Category
from polimibot.rag.live_search import LiveSearchFallback


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wikipedia_mock(
    search_results: list[str],
    summary_text: str = "Caesar crossed the Rubicon in 49 BC.",
    raise_on_summary: Exception | None = None,
) -> types.ModuleType:
    """Build a minimal wikipedia module mock."""
    wp = MagicMock()
    wp.search.return_value = search_results
    if raise_on_summary is not None:
        wp.summary.side_effect = raise_on_summary
    else:
        wp.summary.return_value = summary_text
    wp.set_lang.return_value = None
    return wp


# ── Construction ──────────────────────────────────────────────────────────────

def test_live_search_default_construction():
    fb = LiveSearchFallback()
    assert fb.timeout_seconds == 5.0
    assert fb.max_articles == 2
    assert fb.use_summary_only is True


def test_live_search_custom_params():
    fb = LiveSearchFallback(timeout_seconds=3.0, max_articles=1, search_results=3)
    assert fb.timeout_seconds == 3.0
    assert fb.max_articles == 1
    assert fb.search_results == 3


# ── Happy path ────────────────────────────────────────────────────────────────

def test_live_search_returns_articles_on_success():
    """When wikipedia returns results, we get Article objects back."""
    wp_mock = _make_wikipedia_mock(
        search_results=["Julius Caesar", "Roman Republic"],
        summary_text="Caesar crossed the Rubicon in 49 BC.",
    )
    fb = LiveSearchFallback(max_articles=2)
    with patch.dict(sys.modules, {"wikipedia": wp_mock}):
        articles = fb.search("Who crossed the Rubicon?", category=Category.HISTORY)

    assert len(articles) == 2
    assert articles[0].title == "Julius Caesar"
    assert "Caesar" in articles[0].text
    assert articles[0].category == Category.HISTORY


def test_live_search_respects_max_articles():
    """max_articles=1 must cap even if Wikipedia returns more."""
    wp_mock = _make_wikipedia_mock(
        search_results=["Julius Caesar", "Roman Republic", "Pompey"],
    )
    fb = LiveSearchFallback(max_articles=1)
    with patch.dict(sys.modules, {"wikipedia": wp_mock}):
        articles = fb.search("Rubicon")

    assert len(articles) == 1


def test_live_search_tags_category():
    """Articles are tagged with the supplied category."""
    wp_mock = _make_wikipedia_mock(search_results=["Big Bang"])
    fb = LiveSearchFallback(max_articles=1)
    with patch.dict(sys.modules, {"wikipedia": wp_mock}):
        articles = fb.search("origin of universe", category=Category.SCIENCE)

    assert articles[0].category == Category.SCIENCE


def test_live_search_returns_empty_on_no_results():
    """wikipedia.search returning [] → empty list."""
    wp_mock = _make_wikipedia_mock(search_results=[])
    fb = LiveSearchFallback()
    with patch.dict(sys.modules, {"wikipedia": wp_mock}):
        articles = fb.search("some obscure query")

    assert articles == []


# ── Failure / degradation ─────────────────────────────────────────────────────

def test_live_search_returns_empty_on_network_error():
    """Any exception from wikipedia → returns [], no crash."""
    wp_mock = _make_wikipedia_mock(
        search_results=["Julius Caesar"],
        raise_on_summary=ConnectionError("network down"),
    )
    fb = LiveSearchFallback()
    with patch.dict(sys.modules, {"wikipedia": wp_mock}):
        articles = fb.search("Caesar")

    assert articles == []


def test_live_search_returns_empty_on_search_exception():
    """Exception from wikipedia.search itself → returns []."""
    wp_mock = MagicMock()
    wp_mock.set_lang.return_value = None
    wp_mock.search.side_effect = RuntimeError("API error")

    fb = LiveSearchFallback()
    with patch.dict(sys.modules, {"wikipedia": wp_mock}):
        articles = fb.search("anything")

    assert articles == []


def test_live_search_returns_empty_when_wikipedia_not_installed():
    """ImportError on ``import wikipedia`` → returns [], no crash."""
    fb = LiveSearchFallback()
    # Temporarily hide wikipedia from sys.modules.
    saved = sys.modules.pop("wikipedia", None)
    try:
        with patch.dict(sys.modules, {"wikipedia": None}):
            articles = fb.search("Caesar")
        assert articles == []
    finally:
        if saved is not None:
            sys.modules["wikipedia"] = saved


def test_live_search_returns_empty_on_timeout():
    """When the fetch takes longer than timeout, returns []."""

    def _slow_fetch(*args, **kwargs):
        threading.Event().wait(10)  # blocks until test teardown
        return []

    fb = LiveSearchFallback(timeout_seconds=0.05)  # very short timeout
    # Patch _fetch directly so we don't need wikipedia installed.
    with patch.object(fb, "_fetch", side_effect=_slow_fetch):
        articles = fb.search("slow query")

    assert articles == []


def test_live_search_empty_query_returns_empty():
    """Blank query string → immediate [] without touching wikipedia."""
    fb = LiveSearchFallback()
    articles = fb.search("   ")
    assert articles == []

    articles = fb.search("")
    assert articles == []


# ── Article content ───────────────────────────────────────────────────────────

def test_live_search_cleans_citation_markers():
    """clean_wikipedia_text() strips [1][2] citation markers."""
    wp_mock = _make_wikipedia_mock(
        search_results=["Test"],
        summary_text="Caesar[1] crossed[2][3] the Rubicon.",
    )
    fb = LiveSearchFallback(max_articles=1)
    with patch.dict(sys.modules, {"wikipedia": wp_mock}):
        articles = fb.search("Caesar", category=Category.HISTORY)

    assert "[1]" not in articles[0].text
    assert "[2]" not in articles[0].text
    assert "Caesar" in articles[0].text


def test_live_search_skips_empty_summary():
    """An article whose summary cleans to empty text is skipped."""
    wp_mock = _make_wikipedia_mock(
        search_results=["Empty", "Julius Caesar"],
    )
    # First summary is whitespace only; second is real content.
    wp_mock.summary.side_effect = ["   \n  ", "Caesar crossed the Rubicon."]

    fb = LiveSearchFallback(max_articles=2)
    with patch.dict(sys.modules, {"wikipedia": wp_mock}):
        articles = fb.search("Caesar")

    # The empty-summary article is skipped; the second one is returned.
    assert len(articles) == 1
    assert articles[0].title == "Julius Caesar"
