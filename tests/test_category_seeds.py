"""Concept-seed harvesting logic — no network (the MediaWiki call is patched)."""
from __future__ import annotations

import polimibot.rag.category_seeds as cs
from polimibot.config import Category


def test_concept_seeds_cover_conceptual_categories_only():
    """Concept seeds target the conceptual competitions; MATHS (procedural) and
    NEWS (live Guardian path) intentionally get none."""
    assert cs.CONCEPT_SEEDS.get(Category.MATHS, []) == []
    assert cs.CONCEPT_SEEDS.get(Category.NEWS, []) == []
    # Philosophy & Psychology is the dominant, most conceptual category.
    assert len(cs.CONCEPT_SEEDS[Category.PHILOSOPHY]) >= 10
    for cat in (Category.SCIENCE, Category.HISTORY, Category.ENTERTAINMENT):
        assert cs.CONCEPT_SEEDS[cat]


def test_harvest_seeds_dedupes_and_respects_caps(monkeypatch):
    """_harvest_seeds normalises bare/tuple entries, dedupes across categories,
    and forwards the per-seed cap + depth to the fetcher."""
    calls = []

    def fake_fetch(category, *, limit, max_depth, verbose, _depth=0):
        calls.append((category, limit, max_depth))
        return {
            "Cognitive_biases": ["Bystander effect", "Just-world hypothesis"],
            "Psychotherapy": ["Just-world hypothesis", "Person-centered therapy"],
        }.get(category, [])

    monkeypatch.setattr(cs, "_fetch_category_members", fake_fetch)

    titles: list[str] = []
    seen: set[str] = set()
    cs._harvest_seeds(
        [("Cognitive_biases", 50), "Psychotherapy"],
        titles=titles, seen=seen, default_cap=500, max_depth=1, verbose=False,
    )

    # Tuple cap honoured; bare seed gets the default cap; depth forwarded.
    assert ("Cognitive_biases", 50, 1) in calls
    assert ("Psychotherapy", 500, 1) in calls
    # Cross-seed dedup: "Just-world hypothesis" appears once.
    assert titles == ["Bystander effect", "Just-world hypothesis", "Person-centered therapy"]


def test_harvest_seeds_skips_failed_category(monkeypatch):
    """A category that raises (e.g. doesn't exist) is skipped, not fatal."""
    def fake_fetch(category, *, limit, max_depth, verbose, _depth=0):
        if category == "Does_Not_Exist":
            raise RuntimeError("404")
        return ["Real Article"]

    monkeypatch.setattr(cs, "_fetch_category_members", fake_fetch)
    titles: list[str] = []
    cs._harvest_seeds(
        ["Does_Not_Exist", "Good_Category"],
        titles=titles, seen=set(), default_cap=10, max_depth=0, verbose=False,
    )
    assert titles == ["Real Article"]
