"""Reliability diagram and Expected Calibration Error.

Calibration asks: "When the model says it is 90% confident,
is it right 90% of the time?" A perfectly calibrated model's
diagonal line sits on the identity line y = x.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class CalibrationResult:
    ece: float                  # Expected Calibration Error ∈ [0, 1]
    bin_confidences: list[float]
    bin_accuracies: list[float]
    bin_counts: list[int]
    n_bins: int


def compute_calibration(
    confidences: Sequence[float],
    corrects: Sequence[bool],
    n_bins: int = 10,
) -> CalibrationResult:
    """Bin predictions by confidence; compare mean confidence to accuracy per bin.

    Args:
        confidences: top-letter probability for each question (∈ [0, 1]).
        corrects:    whether the chosen answer was correct.
        n_bins:      number of equal-width bins (default 10).

    Returns:
        CalibrationResult with ECE and per-bin stats.
    """
    # Sorted, equal-width bins: [0,0.1), [0.1,0.2), … [0.9,1.0]
    # Boundary of each bin: correct, this order must be.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    conf_arr = np.asarray(confidences, dtype=float)
    corr_arr = np.asarray(corrects, dtype=float)

    bin_confs, bin_accs, bin_counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf_arr >= lo) & (conf_arr < hi)
        if lo == bins[-2]:          # include right edge in last bin
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        n = mask.sum()
        bin_counts.append(int(n))
        if n == 0:
            bin_confs.append(float((lo + hi) / 2))
            bin_accs.append(0.0)
        else:
            bin_confs.append(float(conf_arr[mask].mean()))
            bin_accs.append(float(corr_arr[mask].mean()))

    total = len(conf_arr)
    ece = float(
        sum(
            cnt * abs(acc - conf)
            for cnt, acc, conf in zip(bin_counts, bin_accs, bin_confs)
        )
        / max(total, 1)
    )
    return CalibrationResult(
        ece=ece,
        bin_confidences=bin_confs,
        bin_accuracies=bin_accs,
        bin_counts=bin_counts,
        n_bins=n_bins,
    )


def plot_calibration(
    result: CalibrationResult,
    title: str = "Reliability Diagram",
    output_path: Path | None = None,
) -> None:
    """Render and optionally save a reliability diagram.

    Requires matplotlib. Imported lazily so the module is importable
    in headless eval environments.
    """
    import matplotlib.pyplot as plt  # lazy import, intentional this is

    fig, ax = plt.subplots(figsize=(5, 5))

    # Identity line — perfect calibration, this represents.
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    # Bar chart of bin accuracy, width proportional to bin size — density, shows this.
    bin_width = 1.0 / result.n_bins
    ax.bar(
        result.bin_confidences,
        result.bin_accuracies,
        width=bin_width * 0.8,
        alpha=0.7,
        label="Model",
        color="steelblue",
        edgecolor="white",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE = {result.ece:.4f}")
    ax.legend()
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    plt.show()


def calibration_from_runs(run_path: Path, n_bins: int = 10) -> CalibrationResult:
    """Load a run-log JSONL and compute calibration from stored confidence values.

    Expects a file produced by ``RunLogger`` (one JSON object per line).
    Only ``run_kind == "question"`` records contribute, and only those with
    both ``confidence`` (float) and ``correct`` (bool) populated. Records
    where ``correct is None`` (server didn't reveal) are skipped.

    Note: this used to be called ``calibration_from_gold_set`` and pointed
    at gold-set JSONL — but gold items have no confidence/correct fields,
    so the old function silently produced empty data. Always use a run
    log here.
    """
    confidences, corrects = [], []
    with run_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("run_kind") and row.get("run_kind") != "question":
                continue
            conf = row.get("confidence")
            corr = row.get("correct")
            if conf is None or corr is None:
                continue
            confidences.append(float(conf))
            corrects.append(bool(corr))
    return compute_calibration(confidences, corrects, n_bins=n_bins)