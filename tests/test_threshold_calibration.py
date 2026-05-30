"""Path-aware min_score calibration.

Pins the calibration formula behaviour so the in-notebook tuning section
and ``scripts/calibrate_min_score.py`` (both consumers of this module)
stay in sync with the algorithm RAGStrategy's gating depends on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from polimibot.eval.threshold_calibration import (
    calibrate_threshold,
    load_pairs_from_runs,
)


def test_calibrate_threshold_zero_threshold_keeps_everyone():
    # τ=0.0 ungates every score in [0, 1] → expected == observed accuracy.
    rows = calibrate_threshold(
        scores=[0.1, 0.5, 0.9],
        corrects=[False, True, True],
        bare_baseline_acc=0.0,
    )
    by_tau = {r["tau"]: r for r in rows}
    assert by_tau[0.0]["p_ungated"] == 1.0
    assert by_tau[0.0]["acc_ungated"] == pytest.approx(2 / 3)
    assert by_tau[0.0]["expected"]    == pytest.approx(2 / 3)


def test_calibrate_threshold_high_threshold_gates_everyone():
    # τ=1.0 gates every score < 1.0 → expected == bare_baseline_acc.
    rows = calibrate_threshold(
        scores=[0.1, 0.5, 0.9],
        corrects=[True, True, True],
        bare_baseline_acc=0.42,
    )
    by_tau = {r["tau"]: r for r in rows}
    assert by_tau[1.0]["p_ungated"] == 0.0
    assert by_tau[1.0]["expected"]  == pytest.approx(0.42)


def test_calibrate_threshold_handles_negative_logits():
    # bge-reranker emits raw logits — negative scores are normal. The
    # algorithm must not assume a [0, 1] scale.
    rows = calibrate_threshold(
        scores=[-2.5, -1.0, 0.5, 2.0],
        corrects=[False, False, True, True],
        bare_baseline_acc=0.1,
    )
    # The optimum should ungate the two high-scoring (and correct) entries.
    best = rows[0]
    assert best["acc_ungated"] >= 0.5
    assert any(r["tau"] < 0 for r in rows)  # negative candidates considered


def test_calibrate_threshold_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calibrate_threshold(
            scores=[0.1, 0.2],
            corrects=[True],
            bare_baseline_acc=0.5,
        )


def test_calibrate_threshold_empty_input_returns_empty():
    assert calibrate_threshold([], [], bare_baseline_acc=0.5) == []


def test_calibrate_threshold_rows_sorted_by_expected_desc():
    rows = calibrate_threshold(
        scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        corrects=[False, False, True, True, True],
        bare_baseline_acc=0.3,
    )
    expecteds = [r["expected"] for r in rows]
    assert expecteds == sorted(expecteds, reverse=True)


def test_load_pairs_from_runs_skips_rows_without_top_score(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"correct": True,  "extras": {"top_score": 0.8}}) + "\n"
        + json.dumps({"correct": False, "extras": {}})                  + "\n"  # no score
        + json.dumps({"correct": True})                                  + "\n"  # no extras
        + json.dumps({"correct": False, "extras": {"top_score": 0.1}}) + "\n",
        encoding="utf-8",
    )
    pairs = load_pairs_from_runs(path)
    assert pairs == [(0.8, True), (0.1, False)]
