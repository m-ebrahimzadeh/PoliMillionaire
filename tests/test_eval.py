"""Gold set tests. No filesystem side-effects — uses tmp_path."""
import json
from pathlib import Path

import pytest

from polimibot.eval.gold_set import (
    GoldItem, harvest_gold_set, save_gold_set, load_gold_set
)
from polimibot.config import Category


def _write_run(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_harvest_direct_correct(tmp_path: Path):
    """correct=True → chosen_index is gold."""
    _write_run(tmp_path / "run1.jsonl", [
        {"run_kind": "manifest"},
        {
            "run_kind": "question", "competition_id": 2, "level": 3,
            "question_text": "What is H2O?", "options": ["Oxygen","Hydrogen","Water","CO2"],
            "chosen_index": 2, "correct": True,
        },
    ])
    items = harvest_gold_set(tmp_path)
    assert len(items) == 1
    assert items[0].correct_index == 2
    assert items[0].category == Category.SCIENCE


def test_harvest_deduplicates_same_question(tmp_path: Path):
    """Same question in two runs → one gold item."""
    rec = {
        "run_kind": "question", "competition_id": 1, "level": 5,
        "question_text": "Who crossed the Rubicon?", "options": ["A","B","C","D"],
        "chosen_index": 0, "correct": True,
    }
    _write_run(tmp_path / "run1.jsonl", [rec])
    _write_run(tmp_path / "run2.jsonl", [rec])
    items = harvest_gold_set(tmp_path)
    assert len(items) == 1


def test_harvest_elimination(tmp_path: Path):
    """Three wrong attempts → 4th option is gold by elimination."""
    q = "What is 2+2?"
    opts = ["3", "5", "6", "4"]
    _write_run(tmp_path / "run1.jsonl", [
        {"run_kind": "question", "competition_id": 3, "level": 1,
         "question_text": q, "options": opts, "chosen_index": 0, "correct": False},
    ])
    _write_run(tmp_path / "run2.jsonl", [
        {"run_kind": "question", "competition_id": 3, "level": 1,
         "question_text": q, "options": opts, "chosen_index": 1, "correct": False},
    ])
    _write_run(tmp_path / "run3.jsonl", [
        {"run_kind": "question", "competition_id": 3, "level": 1,
         "question_text": q, "options": opts, "chosen_index": 2, "correct": False},
    ])
    items = harvest_gold_set(tmp_path)
    assert len(items) == 1
    assert items[0].correct_index == 3    # only remaining


def test_wrong_only_not_included(tmp_path: Path):
    """A question seen wrong once (2 options remain) → not included."""
    _write_run(tmp_path / "run1.jsonl", [
        {"run_kind": "question", "competition_id": 0, "level": 1,
         "question_text": "?", "options": ["a","b","c","d"],
         "chosen_index": 1, "correct": False},
    ])
    items = harvest_gold_set(tmp_path)
    assert items == []


def test_save_and_load_roundtrip(tmp_path: Path):
    item = GoldItem(
        question_text="What is Au?",
        options=("Silver", "Iron", "Gold", "Copper"),
        correct_index=2,
        competition_id=2,
        level=7,
        category=Category.SCIENCE,
        source_run="run_abc.jsonl",
    )
    path = tmp_path / "gold_set.jsonl"
    save_gold_set([item], path)
    loaded = load_gold_set(path)
    assert len(loaded) == 1
    assert loaded[0].correct_index == 2
    assert loaded[0].category == Category.SCIENCE
    assert loaded[0].options == ("Silver", "Iron", "Gold", "Copper")