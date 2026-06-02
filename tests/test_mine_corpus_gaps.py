"""scripts/mine_corpus_gaps.py — gap detection + candidate extraction (no network)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mine_corpus_gaps.py"
_spec = importlib.util.spec_from_file_location("mine_corpus_gaps", _SCRIPT)
mcg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcg)

from polimibot.config import Category


def _q(**kw) -> dict:
    base = {"run_kind": "question", "competition_name": "Science and Nature",
            "correct": True, "extras": {}}
    base.update(kw)
    return base


# ── is_gap ─────────────────────────────────────────────────────────────────

def test_is_gap_true_for_wrong_answer():
    assert mcg.is_gap(_q(correct=False), top_score_floor=0.5)


def test_is_gap_true_when_gated():
    assert mcg.is_gap(_q(extras={"gated_by_min_score": True}), top_score_floor=0.5)


def test_is_gap_true_for_low_top_score():
    assert mcg.is_gap(_q(extras={"top_score": 0.02}), top_score_floor=0.5)


def test_is_gap_false_for_confident_correct_baseline():
    assert not mcg.is_gap(_q(correct=True, extras={"margin": 0.9}), top_score_floor=0.5)


def test_is_gap_excludes_news_and_maths():
    assert not mcg.is_gap(_q(competition_name="News", correct=False), top_score_floor=0.5)
    assert not mcg.is_gap(_q(competition_name="Maths", correct=False), top_score_floor=0.5)


def test_is_gap_ignores_non_question_records():
    assert not mcg.is_gap({"run_kind": "summary", "correct": False}, top_score_floor=0.5)


# ── extract_candidates ───────────────────────────────────────────────────────

def test_extract_candidates_quoted_and_proper_noun():
    cands = mcg.extract_candidates(
        "How does Joey's character Dr. Drake Ramoray's role on 'Days of Our Lives' end?",
        ["He quits", "He falls down an elevator shaft"],
    )
    assert "Days of Our Lives" in cands
    assert any("Ramoray" in c for c in cands)


def test_extract_candidates_lowercase_concept_suffix():
    cands = mcg.extract_candidates("Which of the following best describes the bystander effect?", [])
    assert "bystander effect" in [c.lower() for c in cands]


def test_extract_candidates_proper_noun_run():
    cands = mcg.extract_candidates(
        "What was the significance of the Battle of Actium in 31 BC?", [])
    assert "Battle of Actium" in cands


def test_extract_candidates_drops_boundary_fragments():
    """§8d: proper-noun runs that end (or start) on a function word are sentence
    fragments, not titles — they must not survive extraction."""
    cands = mcg.extract_candidates(
        "Did The Beatles To stardom rise, like Of Human folly and Washington It?",
        [])
    lowers = [c.lower() for c in cands]
    assert "beatles to" not in lowers
    assert "of human" not in lowers
    assert "washington it" not in lowers


def test_extract_candidates_keeps_leading_article_titles():
    """A leading article is fine for a real title ("The Beatles")."""
    cands = mcg.extract_candidates("How did The Beatles change pop music forever?", [])
    assert any(c.lower() == "the beatles" for c in cands)


def test_extract_candidates_possessive_apostrophe_not_a_quote():
    """§8d: a possessive apostrophe must not open a quoted span and swallow the
    sentence ("s approach to learning his lines for")."""
    cands = mcg.extract_candidates(
        "What was Brando's approach to learning his lines for 'On the Waterfront'?",
        [])
    lowers = [c.lower() for c in cands]
    assert not any(c.startswith("s approach") for c in lowers)
    assert "On the Waterfront" in cands   # the real quoted title still resolves


def test_extract_candidates_no_generic_relationship_phrase():
    """§8d: 'relationship' is no longer a concept suffix — filler like
    'does the relationship' must not be extracted."""
    cands = mcg.extract_candidates(
        "How does the relationship between supply and demand work?", [])
    assert not any("relationship" in c.lower() for c in cands)


# ── load_gap_candidates ──────────────────────────────────────────────────────

def test_load_gap_candidates_buckets_by_category(tmp_path):
    log = tmp_path / "run.jsonl"
    rows = [
        {"run_kind": "manifest"},  # ignored
        _q(competition_name="Ancient History and Politics", correct=False,
           question_text="What was the significance of the Battle of Actium?",
           options=[]),
        _q(competition_name="News", correct=False,           # excluded
           question_text="Which whale was found dead?", options=[]),
        _q(correct=True, extras={"margin": 0.99},            # not a gap
           question_text="The Sun is a Star.", options=[]),
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    by_cat = mcg.load_gap_candidates([log], top_score_floor=0.5)
    assert Category.HISTORY in by_cat
    assert "Battle of Actium" in by_cat[Category.HISTORY]
    assert Category.NEWS not in by_cat
    assert Category.SCIENCE not in by_cat   # the only science row wasn't a gap


def test_competition_to_category_maps_display_names():
    assert mcg._competition_to_category("Ancient History and Politics") == Category.HISTORY
    assert mcg._competition_to_category("Unknown Competition") is None
