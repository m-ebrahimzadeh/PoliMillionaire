"""Unit tests for the Guardian news source — no network required.

All HTTP is mocked at ``requests.get`` (the real ``requests`` module is kept so
its exception classes stay valid for the ``except`` clauses), so these run
fully offline and are safe in CI.
"""
from __future__ import annotations

import datetime as _dt
import types
from unittest.mock import MagicMock, patch

import pytest

from polimibot.config import Category, NewsConfig
from polimibot.rag import news_search as ns
from polimibot.rag.corpus import Article
from polimibot.rag.news_search import (
    GuardianNewsSource, NewsLiveSearch, _build_news_query, extract_question_date,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg(**kw) -> NewsConfig:
    base = dict(guardian_api_key="testkey", min_delay_seconds=0.0)
    base.update(kw)
    return NewsConfig(**base)


def _result(title="Living museum in Brazil", date="2026-05-16",
            body="Museu de Favela describes itself as a living museum.",
            url="https://www.theguardian.com/world/2026/may/16/museu-de-favela"):
    return {
        "webTitle": title,
        "webUrl": url,
        "webPublicationDate": f"{date}T10:00:00Z",
        "sectionName": "World news",
        "fields": {"bodyText": body, "trailText": "teaser"},
    }


def _payload(results, *, pages=1, status="ok"):
    return {"response": {"status": status, "pages": pages, "results": results}}


def _resp(payload, *, status_code=200, raise_exc=None):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = payload
    if raise_exc is not None:
        r.raise_for_status.side_effect = raise_exc
    else:
        r.raise_for_status.return_value = None
    return r


# ── Date extraction ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("published on 2026-05-17, which charity", _dt.date(2026, 5, 17)),
    ("The 2026-05-16 article mentions a museum", _dt.date(2026, 5, 16)),
    ("as reported on 2026-05-18?", _dt.date(2026, 5, 18)),
    ("an event on 16th May 2026 in London", _dt.date(2026, 5, 16)),
    ("released May 16, 2026 by the charity", _dt.date(2026, 5, 16)),
    ("no date here at all", None),
    ("not a date 2026-13-40 here", None),  # invalid → None
])
def test_extract_question_date(text, expected):
    assert extract_question_date(text) == expected


def test_build_news_query_strips_date_and_boilerplate():
    q = _build_news_query(
        "According to the article published on 2026-05-17, which charity is "
        "advocating for changes to the benefit cap?"
    )
    assert "2026-05-17" not in q
    assert "according to" not in q.lower()
    assert "charity" in q
    assert "benefit cap" in q


# ── GuardianNewsSource.search — happy path ──────────────────────────────────────

def test_search_maps_results_to_articles(tmp_path):
    src = GuardianNewsSource(_cfg(), cache_dir=tmp_path)
    with patch.object(ns.requests, "get",
                      return_value=_resp(_payload([_result()]))) as get:
        articles = src.search("museum brazil living",
                              from_date=_dt.date(2026, 5, 15),
                              to_date=_dt.date(2026, 5, 17))

    assert len(articles) == 1
    a = articles[0]
    assert isinstance(a, Article)
    assert a.category == Category.NEWS
    assert a.title == "Living museum in Brazil"
    assert a.text.startswith("Published 2026-05-16.")
    assert "Museu de Favela" in a.text
    assert a.url.startswith("https://www.theguardian.com")

    # Date window + query are passed through as Guardian params.
    params = get.call_args.kwargs["params"]
    assert params["from-date"] == "2026-05-15"
    assert params["to-date"] == "2026-05-17"
    assert params["q"] == "museum brazil living"
    assert "bodyText" in params["show-fields"]


def test_search_respects_page_size(tmp_path):
    src = GuardianNewsSource(_cfg(max_articles=2), cache_dir=tmp_path)
    results = [_result(title=f"Article {i}") for i in range(5)]
    with patch.object(ns.requests, "get", return_value=_resp(_payload(results))):
        articles = src.search("anything")
    assert len(articles) == 2


def test_search_skips_items_without_body(tmp_path):
    src = GuardianNewsSource(_cfg(), cache_dir=tmp_path)
    bad = {"webTitle": "No body", "webUrl": "u", "webPublicationDate": "2026-05-16T0:0:0Z",
           "fields": {"bodyText": "  "}}
    with patch.object(ns.requests, "get",
                      return_value=_resp(_payload([bad, _result()]))):
        articles = src.search("q")
    assert len(articles) == 1
    assert articles[0].title == "Living museum in Brazil"


# ── GuardianNewsSource — graceful degradation ───────────────────────────────────

def test_search_empty_on_http_error(tmp_path):
    src = GuardianNewsSource(_cfg(), cache_dir=tmp_path)
    import requests
    with patch.object(ns.requests, "get",
                      return_value=_resp(_payload([]), raise_exc=requests.HTTPError("500"))):
        assert src.search("q") == []


def test_search_empty_on_429(tmp_path):
    src = GuardianNewsSource(_cfg(), cache_dir=tmp_path)
    with patch.object(ns.requests, "get",
                      return_value=_resp(_payload([_result()]), status_code=429)):
        assert src.search("q") == []


def test_search_empty_on_timeout(tmp_path):
    import requests
    src = GuardianNewsSource(_cfg(), cache_dir=tmp_path)
    with patch.object(ns.requests, "get", side_effect=requests.Timeout()):
        assert src.search("q") == []


def test_search_empty_on_non_ok_status(tmp_path):
    src = GuardianNewsSource(_cfg(), cache_dir=tmp_path)
    with patch.object(ns.requests, "get",
                      return_value=_resp(_payload([_result()], status="error"))):
        assert src.search("q") == []


# ── Disk cache ──────────────────────────────────────────────────────────────────

def test_search_uses_disk_cache(tmp_path):
    """A repeated identical query is served from cache — no second HTTP call."""
    src = GuardianNewsSource(_cfg(), cache_dir=tmp_path)
    with patch.object(ns.requests, "get",
                      return_value=_resp(_payload([_result()]))) as get:
        a1 = src.search("museum", from_date=_dt.date(2026, 5, 15), to_date=_dt.date(2026, 5, 17))
        a2 = src.search("museum", from_date=_dt.date(2026, 5, 15), to_date=_dt.date(2026, 5, 17))

    assert get.call_count == 1            # second call hit the cache
    assert a1[0].title == a2[0].title
    assert list(tmp_path.glob("*.json"))  # cache file written


# ── fetch_range (harvest) ───────────────────────────────────────────────────────

def test_fetch_range_paginates(tmp_path):
    src = GuardianNewsSource(_cfg(), cache_dir=tmp_path)
    page1 = _resp(_payload([_result(title="p1a"), _result(title="p1b")], pages=2))
    page2 = _resp(_payload([_result(title="p2a")], pages=2))
    with patch.object(ns.requests, "get", side_effect=[page1, page2]) as get:
        articles = src.fetch_range(_dt.date(2026, 5, 1), _dt.date(2026, 5, 31),
                                   sections="world", page_size=2, max_pages=5)
    assert get.call_count == 2
    assert [a.title for a in articles] == ["p1a", "p1b", "p2a"]


# ── NewsLiveSearch — Guardian + Wikipedia fallback ──────────────────────────────

class _FakeGuardian:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, *, from_date=None, to_date=None, page_size=None, order_by="relevance"):
        self.calls.append((query, from_date, to_date))
        return list(self.results)


class _FakeWiki:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, *, category=None):
        self.calls.append((query, category))
        return list(self.results)


def test_news_live_search_uses_guardian_date_window():
    g_article = Article("g", "Published 2026-05-17. body", Category.NEWS, "u")
    g = _FakeGuardian([g_article])
    w = _FakeWiki([])
    nls = NewsLiveSearch(_cfg(date_window_days=1), guardian=g, wiki_fallback=w)

    out = nls.search("According to the 2026-05-17 article, who won?", category=Category.NEWS)

    assert out == [g_article]
    assert w.calls == []                  # Wikipedia not consulted
    # ±1 day window around the stated date.
    _query, frm, to = g.calls[0]
    assert frm == _dt.date(2026, 5, 16)
    assert to == _dt.date(2026, 5, 18)


def test_news_live_search_falls_back_to_wikipedia():
    w_article = Article("w", "wiki text", Category.NEWS, "u")
    g = _FakeGuardian([])                 # Guardian finds nothing
    w = _FakeWiki([w_article])
    nls = NewsLiveSearch(_cfg(), guardian=g, wiki_fallback=w)

    out = nls.search("On 2026-05-17 something happened", category=Category.NEWS)

    assert out == [w_article]             # degraded to Wikipedia
    assert len(g.calls) >= 1              # Guardian was tried first
    assert w.calls and w.calls[0][1] == Category.NEWS


def test_news_live_search_empty_query():
    nls = NewsLiveSearch(_cfg(), guardian=_FakeGuardian([]), wiki_fallback=_FakeWiki([]))
    assert nls.search("   ") == []


# ── RAGStrategy routing ─────────────────────────────────────────────────────────

class _FakeSource:
    def __init__(self):
        self.calls = []

    def search(self, query, *, category=None):
        self.calls.append((query, category))
        return []


def _rag_with_sources():
    from polimibot.models.mock import MockLLM
    from polimibot.strategies.rag_strategy import RAGStrategy
    news, wiki = _FakeSource(), _FakeSource()
    strat = RAGStrategy(
        MockLLM(), types.SimpleNamespace(),
        use_multi_query=False,
        min_score=1.0,            # force-gate + skip offline (no retriever needed)
        news_search=news,
    )
    strat._live_search = wiki     # inject Wikipedia source for non-news categories
    return strat, news, wiki


def test_rag_strategy_routes_news_to_news_search():
    from polimibot.strategies.base import StrategyInput
    strat, news, wiki = _rag_with_sources()
    strat.answer(StrategyInput(
        question="According to the 2026-05-17 article, who won?",
        options=("A", "B", "C", "D"), level=5, category=Category.NEWS,
    ))
    assert len(news.calls) == 1
    assert len(wiki.calls) == 0


def test_rag_strategy_routes_other_categories_to_wikipedia():
    from polimibot.strategies.base import StrategyInput
    strat, news, wiki = _rag_with_sources()
    strat.answer(StrategyInput(
        question="Who crossed the Rubicon?",
        options=("A", "B", "C", "D"), level=5, category=Category.HISTORY,
    ))
    assert len(wiki.calls) == 1
    assert len(news.calls) == 0
