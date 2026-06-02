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


# ── Cross-category seed dedup ───────────────────────────────────────────────


def test_dedupe_seeds_drops_cross_category_duplicates():
    """A title that lives under multiple Category entries is fetched once,
    under its first occurrence in the iteration order."""
    from polimibot.config import Category
    from polimibot.rag.corpus import TOPIC_SEEDS, _dedupe_seeds
    # Pick any title that we expect to be duplicated. If the seeds change
    # so that no duplicates remain, the test should still pass (just
    # verifies the dedup function itself preserves the first category).
    flat = _dedupe_seeds(list(TOPIC_SEEDS.keys()), verbose=False)
    titles = [t for t, _ in flat]
    # No duplicates.
    assert len(titles) == len(set(titles))
    # Every title still appears.
    expected = set()
    for seeds in TOPIC_SEEDS.values():
        expected.update(seeds)
    assert set(titles) == expected


def test_dedupe_seeds_first_category_wins():
    """The first category in iteration order owns shared titles."""
    from polimibot.config import Category
    from polimibot.rag.corpus import TOPIC_SEEDS, _dedupe_seeds
    # If "Isaac Newton" appears in both SCIENCE and MATHS and SCIENCE is
    # listed first in TOPIC_SEEDS, it should resolve to SCIENCE.
    cats = list(TOPIC_SEEDS.keys())
    flat = _dedupe_seeds(cats, verbose=False)
    by_title = {t: c for t, c in flat}
    if "Isaac Newton" in by_title:
        # Whichever category lists "Isaac Newton" first in cats order wins.
        first_owner = next(
            cat for cat in cats if "Isaac Newton" in TOPIC_SEEDS[cat]
        )
        assert by_title["Isaac Newton"] == first_owner


# ── Smart disambiguation ────────────────────────────────────────────────────


def test_fetch_one_walks_disambiguation_options(monkeypatch):
    """When the seed title hits a DisambiguationError, _fetch_one should
    try options and return the first whose page summary contains a
    keyword from the seed title — not blindly pick options[0]."""
    from polimibot.config import Category
    from polimibot.rag import corpus as corpus_mod

    # Build a stand-in wikipedia module.
    class _DummyDisambig(Exception):
        def __init__(self, options):
            self.options = options

    class _DummyPageError(Exception):
        pass

    class _DummyPage:
        def __init__(self, title, content, summary, url=""):
            self.title = title
            self.content = content
            self.summary = summary
            self.url = url

    # Map page-name → page (or DisambiguationError, or PageError).
    pages = {
        "Queen": _DummyDisambig(["Queen (band)", "Queen (monarch)"]),
        "Queen (band)": _DummyPage(
            title="Queen (band)",
            content="Queen are a British rock band formed in London.",
            summary="Queen are a British rock band formed in London.",
        ),
        "Queen (monarch)": _DummyPage(
            title="Queen (monarch)",
            content="A queen regnant is a female monarch.",
            summary="A queen regnant is a female monarch.",
        ),
    }

    def _page(name, auto_suggest=False):
        result = pages[name]
        if isinstance(result, _DummyDisambig):
            raise result
        if isinstance(result, _DummyPageError):
            raise result
        return result

    class _DummyWiki:
        DisambiguationError = _DummyDisambig
        PageError = _DummyPageError
        page = staticmethod(_page)
        @staticmethod
        def set_lang(_): pass

    # Inject into sys.modules so corpus._fetch_one's `import wikipedia` resolves to us.
    import sys
    monkeypatch.setitem(sys.modules, "wikipedia", _DummyWiki)

    # Seed "Queen (band)" → we want the BAND, not the monarch. Disambig
    # options are [band, monarch]; both pass content; first-option logic
    # returns band by chance. Reverse the option order and rerun to make
    # sure the keyword-walk picks the right one.
    pages["Queen"] = _DummyDisambig(["Queen (monarch)", "Queen (band)"])
    article = corpus_mod._fetch_one("Queen (band)", Category.ENTERTAINMENT, verbose=False)
    assert article is not None
    # Either resolution can match if both contain the keyword "queen", so
    # also verify the keyword check actually filtered: if we provide
    # "Pythagorean theorem" as a seed with options that share no keyword
    # with the title's tokens, we'd skip — covered by the next test.


def test_fetch_one_skips_when_no_disambig_option_matches(monkeypatch):
    """If no disambiguation option's summary contains a seed keyword,
    return None instead of grabbing a random page."""
    from polimibot.config import Category
    from polimibot.rag import corpus as corpus_mod

    class _Dis(Exception):
        def __init__(self, options): self.options = options
    class _PageErr(Exception):
        pass

    class _Page:
        def __init__(self, title, content, summary):
            self.title, self.content, self.summary, self.url = title, content, summary, ""

    pages = {
        "Pythagorean theorem": _Dis(["Random Article", "Other Random"]),
        "Random Article":      _Page("Random Article", "Cats and dogs are pets.", "Cats and dogs are pets."),
        "Other Random":        _Page("Other Random", "Trains run on rails.",      "Trains run on rails."),
    }

    def _page(name, auto_suggest=False):
        result = pages[name]
        if isinstance(result, _Dis): raise result
        return result

    class _Wiki:
        DisambiguationError = _Dis
        PageError = _PageErr
        page = staticmethod(_page)
        @staticmethod
        def set_lang(_): pass

    import sys
    monkeypatch.setitem(sys.modules, "wikipedia", _Wiki)

    out = corpus_mod._fetch_one("Pythagorean theorem", Category.MATHS, verbose=False)
    assert out is None


def _install_disambig_wiki(monkeypatch, seed, options, lookup):
    """Install a stand-in `wikipedia` module where `wikipedia.page(seed)` raises
    a DisambiguationError over `options`, and each option resolves via `lookup`.
    Returns nothing — call `_fetch_one` afterwards."""
    import sys

    class _Dis(Exception):
        def __init__(self, opts): self.options = opts
    class _PageErr(Exception):
        pass

    def _page(name, auto_suggest=False):
        if name == seed:
            raise _Dis(options)
        return lookup[name]

    class _Wiki:
        DisambiguationError = _Dis
        PageError = _PageErr
        page = staticmethod(_page)
        @staticmethod
        def set_lang(_): pass

    monkeypatch.setitem(sys.modules, "wikipedia", _Wiki)


class _BoomSummaryPage:
    """A page whose *lazy* .summary access raises like a throttled/empty
    MediaWiki response — the exact failure that crashed the Colab crawl."""
    def __init__(self, title):
        self.title, self.url, self.content = title, "", "body text"
    @property
    def summary(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


class _PlainPage:
    def __init__(self, title, summary):
        self.title, self.url, self.content, self.summary = title, "", summary, summary


def test_fetch_one_skips_disambig_option_with_failing_lazy_summary(monkeypatch):
    """Regression (§8a): when a disambiguation option's lazy .summary raises,
    _fetch_one must skip it and walk on — not let the error escape and kill the
    whole crawl. Here the first option booms, the second resolves cleanly."""
    from polimibot.config import Category
    from polimibot.rag import corpus as corpus_mod

    _install_disambig_wiki(
        monkeypatch, seed="Charlie",
        options=["Charlie (elephant)", "Charlie Chaplin"],
        lookup={
            "Charlie (elephant)": _BoomSummaryPage("Charlie (elephant)"),
            "Charlie Chaplin": _PlainPage("Charlie Chaplin",
                                          "Charlie Chaplin was a comic actor."),
        },
    )
    article = corpus_mod._fetch_one("Charlie", Category.ENTERTAINMENT,
                                    verbose=False, fetch_aliases=False)
    assert article is not None
    assert article.title == "Charlie Chaplin"


def test_fetch_one_returns_none_when_every_disambig_option_errors(monkeypatch):
    """Regression (§8a): if *every* option's lazy fetch raises, _fetch_one returns
    None — it never propagates the exception out of the per-title fetch."""
    from polimibot.config import Category
    from polimibot.rag import corpus as corpus_mod

    _install_disambig_wiki(
        monkeypatch, seed="Charlie",
        options=["Charlie (elephant)", "Charlie (parrot)"],
        lookup={
            "Charlie (elephant)": _BoomSummaryPage("Charlie (elephant)"),
            "Charlie (parrot)": _BoomSummaryPage("Charlie (parrot)"),
        },
    )
    out = corpus_mod._fetch_one("Charlie", Category.ENTERTAINMENT,
                                verbose=False, fetch_aliases=False)
    assert out is None


def test_configure_wikipedia_sets_ua_and_rate_limiting():
    """§8b: _configure_wikipedia must set a contact UA and enable rate limiting
    (the throttle defence) — and tolerate a stub missing the optional setters."""
    from datetime import timedelta
    from polimibot.rag import corpus

    calls = {}

    class _FullWiki:
        @staticmethod
        def set_lang(lang): calls["lang"] = lang
        @staticmethod
        def set_user_agent(ua): calls["ua"] = ua
        @staticmethod
        def set_rate_limiting(on, min_wait=None):
            calls["rate"] = (on, min_wait)

    corpus._configure_wikipedia(_FullWiki)
    assert calls["lang"] == "en"
    assert "PoliMillionaire" in calls["ua"] and "contact:" in calls["ua"]
    on, min_wait = calls["rate"]
    assert on is True and isinstance(min_wait, timedelta) and min_wait.microseconds > 0

    # A stripped stub with only set_lang must not raise.
    class _BareWiki:
        @staticmethod
        def set_lang(lang): pass
    corpus._configure_wikipedia(_BareWiki)  # no exception = pass


def test_fetch_from_categories_checkpoints_partial_harvest(tmp_path, monkeypatch):
    """§8c: the crawl rewrites a checkpoint file every `checkpoint_every` fetches,
    so a crash mid-harvest leaves a durable partial corpus on disk."""
    import sys
    from polimibot.config import Category
    from polimibot.rag import corpus as corpus_mod
    from polimibot.rag import category_seeds as cs

    titles = [f"Article {n}" for n in range(5)]
    monkeypatch.setattr(cs, "harvest_titles",
                        lambda categories, **kw: {Category.SCIENCE: list(titles)})
    monkeypatch.setattr(cs, "CONCEPT_TITLES", {})

    class _Page:
        def __init__(self, title):
            self.title, self.url = title, ""
            self.content = self.summary = f"Body of {title}."

    class _Wiki:
        class DisambiguationError(Exception): ...
        class PageError(Exception): ...
        @staticmethod
        def set_lang(_): pass
        @staticmethod
        def page(name, auto_suggest=False): return _Page(name)
    monkeypatch.setitem(sys.modules, "wikipedia", _Wiki)

    ckpt = tmp_path / "corpus.partial.jsonl"
    arts = corpus_mod.fetch_articles_from_categories(
        categories=[Category.SCIENCE], cache_path=None, fetch_aliases=False,
        checkpoint_path=ckpt, checkpoint_every=2, sleep_seconds=0, verbose=False,
    )
    assert len(arts) == 5
    assert ckpt.exists(), "checkpoint file should exist after a multi-fetch crawl"
    # Checkpoints fired at i=2 and i=4 → at least 4 articles are durable on disk.
    assert len(corpus_mod.load_raw_corpus(ckpt)) >= 4


def test_corpus_version_is_positive_int():
    from polimibot.rag.corpus import CORPUS_VERSION
    assert isinstance(CORPUS_VERSION, int) and CORPUS_VERSION >= 2


# ── Article schema round-trip (aliases + competition, backward-compatible) ─────

def test_save_load_roundtrips_aliases_and_competition(tmp_path):
    from polimibot.rag.corpus import Article, save_raw_corpus, load_raw_corpus
    from polimibot.config import Category

    arts = [
        Article(
            title="The One Where Dr. Ramoray Dies", text="Plot…",
            category=Category.ENTERTAINMENT, url="http://x",
            aliases=("Dr. Drake Ramoray", "Ramoray"),
            competition="Entertainment",
        ),
    ]
    path = tmp_path / "corpus.jsonl"
    save_raw_corpus(arts, path)
    loaded = load_raw_corpus(path)
    assert loaded[0].aliases == ("Dr. Drake Ramoray", "Ramoray")
    assert loaded[0].competition == "Entertainment"
    assert loaded[0].category == Category.ENTERTAINMENT


def test_load_tolerates_v3_rows_without_new_fields(tmp_path):
    """v3 corpora (no aliases/competition keys) must still load, with defaults."""
    path = tmp_path / "v3.jsonl"
    path.write_text(
        '{"title": "Gold", "text": "Au.", "category": "science", "url": ""}\n',
        encoding="utf-8",
    )
    from polimibot.rag.corpus import load_raw_corpus
    loaded = load_raw_corpus(path)
    assert loaded[0].aliases == ()
    assert loaded[0].competition == ""


def test_save_omits_empty_new_fields(tmp_path):
    """No aliases/competition → row stays v3-shaped (no extra keys written)."""
    import json
    from polimibot.rag.corpus import Article, save_raw_corpus
    from polimibot.config import Category

    path = tmp_path / "c.jsonl"
    save_raw_corpus([Article("Gold", "Au.", Category.SCIENCE)], path)
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert "aliases" not in row and "competition" not in row


# ── Alias / competition enrichment at fetch time (no network) ──────────────────

def test_competition_for_maps_category_to_display_name():
    from polimibot.rag.corpus import _competition_for
    from polimibot.config import Category
    assert _competition_for(Category.HISTORY) == "Ancient History and Politics"
    assert _competition_for(Category.PHILOSOPHY) == "Philosophy and Psychology"
    assert _competition_for(Category.SCIENCE) == "Science and Nature"


def test_make_article_sets_competition_and_skips_aliases_when_disabled():
    """_make_article derives the competition label and, with fetch_aliases=False,
    attaches no aliases (and makes no network call)."""
    from polimibot.rag.corpus import _make_article
    from polimibot.config import Category

    class _FakePage:
        title = "Bystander effect"
        content = "The bystander effect is a social psychological phenomenon."
        url = "https://en.wikipedia.org/wiki/Bystander_effect"

    art = _make_article(_FakePage(), Category.PHILOSOPHY,
                        fetch_aliases=False, verbose=False)
    assert art.competition == "Philosophy and Psychology"
    assert art.aliases == ()
    assert art.title == "Bystander effect"


def test_fetch_redirects_returns_empty_on_failure(monkeypatch):
    """A redirect-API hiccup must never raise — aliases are optional."""
    import polimibot.rag.corpus as corpus

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(corpus.urllib.request, "urlopen", boom)
    assert corpus._fetch_redirects("Anything", verbose=False) == ()
