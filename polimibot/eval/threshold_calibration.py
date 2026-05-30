"""Path-aware ``min_score`` threshold calibration.

For a given retrieval path (dense / hybrid-RRF / cross-encoder), find the
threshold τ that maximises the *gated-policy* expected accuracy:

    expected_acc(τ) =
        acc(score ≥ τ) · P(score ≥ τ)
      + bare_baseline_acc · P(score < τ)

Reads ``(top_score, correct)`` pairs from one or more RAG run logs and
returns candidate thresholds with their expected accuracy. Used by both
:mod:`scripts.calibrate_min_score` (CLI) and the in-notebook tuning section
(reuses the already-loaded LLM/retriever, so iteration is fast).

The algorithm is path-agnostic: it operates on whatever score scale lives in
``extras.top_score`` of the supplied run log. RAGStrategy writes the active
path's score there, so the scale automatically matches the threshold knob
for that recipe (cosine for dense, RRF for hybrid, raw CE logit for rerank).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def calibrate_threshold(
    scores: Sequence[float],
    corrects: Sequence[bool],
    *,
    bare_baseline_acc: float,
) -> list[dict]:
    """Return candidate τ rows sorted by expected gated-policy accuracy.

    Args:
        scores: per-question top retrieval score (any scale — see module
            docstring; the caller's responsibility to pass scores from one
            retrieval path at a time).
        corrects: per-question correctness of the RAG strategy's answer
            when it answered un-gated. Same length as ``scores``.
        bare_baseline_acc: accuracy of the bare-LLM baseline on the same
            evaluation slice. Drives the expected-accuracy formula by
            standing in for the fallback arm's accuracy on gated questions.

    Returns:
        List of ``{"tau", "p_ungated", "acc_ungated", "expected"}`` dicts,
        one per candidate threshold (every observed score, plus 0.0 and 1.0
        as bookends), sorted by ``expected`` descending. Caller picks the
        first row for the optimum.
    """
    import numpy as np

    if len(scores) != len(corrects):
        raise ValueError(
            f"scores ({len(scores)}) and corrects ({len(corrects)}) "
            "must have the same length"
        )
    if not scores:
        return []

    scores_arr  = np.asarray(scores, dtype=float)
    correct_arr = np.asarray(corrects, dtype=bool)

    # Candidate thresholds: every observed score (rounded to 3 dp so identical
    # scores collapse), plus {0.0, 1.0} as scale-agnostic bookends. The 0/1
    # bookends are harmless on raw-logit reranker scales — they just become
    # additional candidates within or beyond the observed range.
    candidates = sorted(set(np.round(scores_arr, 3).tolist()) | {0.0, 1.0})

    rows: list[dict] = []
    n_total = len(scores_arr)
    for tau in candidates:
        ungated   = scores_arr >= tau
        n_ungated = int(ungated.sum())
        acc_ungated = float(correct_arr[ungated].mean()) if n_ungated > 0 else 0.0
        p_ungated   = n_ungated / n_total
        expected    = (
            acc_ungated * p_ungated
            + bare_baseline_acc * (1.0 - p_ungated)
        )
        rows.append({
            "tau":         float(tau),
            "p_ungated":   p_ungated,
            "acc_ungated": acc_ungated,
            "expected":    expected,
        })

    rows.sort(key=lambda r: r["expected"], reverse=True)
    return rows


def load_pairs_from_runs(*run_paths: Path) -> list[tuple[float, bool]]:
    """Pull ``(top_score, correct)`` tuples from one or more run JSONLs.

    Each line must be a dict with ``correct: bool`` at the top level and
    ``extras.top_score: float``. Lines missing either field are skipped —
    that covers baseline rows (no retrieval, no top_score), gated rows that
    suppressed the score, and any malformed entries.
    """
    pairs: list[tuple[float, bool]] = []
    for path in run_paths:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                extras = rec.get("extras") or {}
                score  = extras.get("top_score")
                if score is None:
                    continue
                pairs.append((float(score), bool(rec.get("correct", False))))
    return pairs
