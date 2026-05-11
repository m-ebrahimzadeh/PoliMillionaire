"""Gold set tests. No filesystem side-effects — uses tmp_path."""
import json
from pathlib import Path

import pytest

from polimibot.eval.gold_set import (
    GoldItem, GoldSet, harvest_gold_set, save_gold_set, load_gold_set
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


# ── GoldSet view ──────────────────────────────────────────────────────

def _gs_item(idx: int, level: int, cat: Category, cid: int) -> GoldItem:
    return GoldItem(
        question_text=f"Q{idx}",
        options=("a", "b", "c", "d"),
        correct_index=0,
        competition_id=cid,
        level=level,
        category=cat,
        source_run="x",
    )


def _gs_fixture() -> GoldSet:
    """6 maths × levels 1..3 (2 each) + 4 science × levels 1..2 (2 each)."""
    items = []
    idx = 0
    for level in (1, 2, 3):
        for _ in range(2):
            items.append(_gs_item(idx, level, Category.MATHS, cid=3)); idx += 1
    for level in (1, 2):
        for _ in range(2):
            items.append(_gs_item(idx, level, Category.SCIENCE, cid=2)); idx += 1
    return GoldSet(items)


def test_goldset_len_iter():
    gs = _gs_fixture()
    assert len(gs) == 10
    assert sum(1 for _ in gs) == 10


def test_goldset_filter_category_with_enum_and_string():
    gs = _gs_fixture()
    by_enum = gs.filter_category(Category.MATHS)
    by_str  = gs.filter_category("maths")
    assert len(by_enum) == 6 == len(by_str)
    assert all(g.category == Category.MATHS for g in by_enum)


def test_goldset_filter_level_range():
    gs = _gs_fixture()
    easy = gs.filter_level(max_level=2)
    assert len(easy) == 8       # 2+2 maths + 2+2 science
    assert all(g.level <= 2 for g in easy)


def test_goldset_filter_competition():
    gs = _gs_fixture()
    only_3 = gs.filter_competition(3)
    assert len(only_3) == 6
    assert all(g.competition_id == 3 for g in only_3)


def test_goldset_take_per_level_caps_each_level():
    gs = _gs_fixture()
    out = gs.take_per_level(1, seed=0)
    # 3 maths levels (1,2,3) + 2 science levels (1,2) = 5 distinct
    # per-level pairs, BUT levels {1,2} are shared across maths+science —
    # take_per_level caps within each LEVEL bucket regardless of category,
    # so levels {1,2,3} → 3 items total.
    assert len(out) == 3
    levels = sorted(g.level for g in out)
    assert levels == [1, 2, 3]


def test_goldset_take_per_category_caps_each_category():
    gs = _gs_fixture()
    out = gs.take_per_category(2, seed=0)
    cats = sorted(g.category.value for g in out)
    assert cats == ["maths", "maths", "science", "science"]


def test_goldset_sample_is_deterministic():
    gs = _gs_fixture()
    a = [g.question_text for g in gs.sample(4, seed=42)]
    b = [g.question_text for g in gs.sample(4, seed=42)]
    c = [g.question_text for g in gs.sample(4, seed=7)]
    assert a == b
    assert a != c   # different seed → different sample, almost surely


def test_goldset_sample_returns_all_when_n_too_big():
    gs = _gs_fixture()
    out = gs.sample(99, seed=0)
    assert len(out) == 10


def test_goldset_split_partitions_without_overlap():
    gs = _gs_fixture()
    a, b = gs.split(0.7, seed=0)
    assert len(a) == 7 and len(b) == 3
    a_ids = {g.question_text for g in a}
    b_ids = {g.question_text for g in b}
    assert a_ids.isdisjoint(b_ids)
    assert a_ids | b_ids == {g.question_text for g in gs}


def test_goldset_split_bad_fraction_raises():
    gs = _gs_fixture()
    with pytest.raises(ValueError):
        gs.split(0.0)
    with pytest.raises(ValueError):
        gs.split(1.5)


def test_goldset_by_category_and_by_level_keys():
    gs = _gs_fixture()
    cats = gs.by_category()
    assert set(cats.keys()) == {Category.MATHS, Category.SCIENCE}
    assert len(cats[Category.MATHS]) == 6

    levels = gs.by_level()
    assert sorted(levels.keys()) == [1, 2, 3]


def test_goldset_chain_filter_then_sample():
    gs = _gs_fixture()
    out = gs.filter_category(Category.MATHS).filter_level(min_level=2).sample(2, seed=0)
    assert len(out) == 2
    assert all(g.category == Category.MATHS and g.level >= 2 for g in out)


def test_goldset_union_dedup_by_question_identity():
    a = GoldSet([_gs_item(0, 1, Category.MATHS, 3), _gs_item(1, 1, Category.MATHS, 3)])
    b = GoldSet([_gs_item(1, 1, Category.MATHS, 3), _gs_item(2, 1, Category.MATHS, 3)])
    # Items 0,1,2 expected; item 1 is in both but deduped.
    assert len(a + b) == 3


def test_goldset_difference_holds_out_test_set():
    full = _gs_fixture()
    test = full.sample(3, seed=0)
    train = full - test
    assert len(train) == 7
    test_ids = {g.question_text for g in test}
    assert all(g.question_text not in test_ids for g in train)


def test_goldset_save_load_roundtrip(tmp_path):
    gs = _gs_fixture()
    path = tmp_path / "subset.jsonl"
    gs.filter_category(Category.MATHS).save(path)
    loaded = GoldSet.load(path)
    assert len(loaded) == 6
    assert all(g.category == Category.MATHS for g in loaded)


def test_goldset_counts_and_print_stats(capsys):
    gs = _gs_fixture()
    by_cat = gs.counts_by_category()
    assert by_cat == {"maths": 6, "science": 4}
    by_lvl = gs.counts_by_level()
    assert by_lvl == {1: 4, 2: 4, 3: 2}
    gs.print_stats()
    out = capsys.readouterr().out
    assert "maths" in out and "science" in out and "total" in out


def test_goldset_drops_straight_into_evaluate_strategy():
    """Smoke test: a GoldSet should be acceptable wherever list[GoldItem] is."""
    gs = _gs_fixture().filter_category(Category.MATHS).take(3)
    from polimibot.strategies.base import Strategy, StrategyInput, StrategyOutput

    class _AlwaysA(Strategy):
        name = "always_A"
        def answer(self, inp): return StrategyOutput(chosen_index=0, confidence=1.0)

    report = evaluate_strategy(_AlwaysA(), gs, verbose=False)
    assert report.n_total == 3





# ── evaluator tests ────────────────────────────────────────────────

from polimibot.eval.evaluator import evaluate_strategy, _ece
from polimibot.strategies.base import Strategy, StrategyInput, StrategyOutput
from polimibot.config import Category


def _make_gold(correct_index: int, competition_id: int = 2) -> GoldItem:
    return GoldItem(
        question_text="Q?", options=("a", "b", "c", "d"),
        correct_index=correct_index, competition_id=competition_id,
        level=1, category=Category.SCIENCE,
    )


class _FixedStrategy(Strategy):
    """Always picks the same index with fixed confidence."""
    def __init__(self, index: int, confidence: float = 0.9):
        self.name = f"fixed_{index}"
        self._index = index
        self._confidence = confidence
    def answer(self, inp: StrategyInput) -> StrategyOutput:
        return StrategyOutput(chosen_index=self._index, confidence=self._confidence)


def test_evaluate_perfect_strategy():
    gold = [_make_gold(correct_index=1) for _ in range(10)]
    report = evaluate_strategy(_FixedStrategy(index=1), gold, verbose=False)
    assert report.accuracy == 1.0
    assert report.n_total == 10


def test_evaluate_always_wrong():
    gold = [_make_gold(correct_index=1) for _ in range(10)]
    report = evaluate_strategy(_FixedStrategy(index=0), gold, verbose=False)
    assert report.accuracy == 0.0


def test_per_category_breakdown():
    gold = (
        [_make_gold(1, competition_id=2) for _ in range(5)] +   # science, correct
        [_make_gold(1, competition_id=3) for _ in range(5)]     # maths, correct
    )
    report = evaluate_strategy(_FixedStrategy(index=1), gold, verbose=False)
    assert "science" in report.by_category
    assert "maths" in report.by_category
    assert report.by_category["science"].accuracy == 1.0


def test_ece_perfect_calibration():
    # Model says 0.5 confidence and is right exactly half the time
    confs = [0.5] * 100
    corrects = [True] * 50 + [False] * 50
    assert _ece(confs, corrects) < 0.05   # near zero


def test_ece_overconfident():
    # Model always says 0.99 confidence but only gets 50% right
    confs = [0.99] * 100
    corrects = [True] * 50 + [False] * 50
    assert _ece(confs, corrects) > 0.4    # large gap


def test_report_save_roundtrip(tmp_path):
    gold = [_make_gold(1) for _ in range(4)]
    report = evaluate_strategy(_FixedStrategy(1), gold, verbose=False)
    path = tmp_path / "report.json"
    report.save(path)
    import json
    d = json.loads(path.read_text())
    assert d["accuracy"] == 1.0
    assert "by_category" in d
    assert "samples" not in d    # excluded from file