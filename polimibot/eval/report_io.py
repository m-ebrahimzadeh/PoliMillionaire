"""Persist and load EvalReport instances.

One JSON file per strategy run. Loaded by make_leaderboard.py to
build the consolidated comparison table. Trivial, but centralised
this logic must be — else scattered saves will diverge in format.
"""
from __future__ import annotations

import json
from pathlib import Path

from polimibot.config import ts
from polimibot.eval.evaluator import EvalReport


def save_report(report: EvalReport, name: str, eval_dir: Path) -> Path:
    """Serialise *report* to ``eval_dir/{name}_{ts()}.json``.

    A UTC timestamp is embedded in the filename so each run produces a
    new file and no previous results are silently overwritten.

    Args:
        report:   completed EvalReport from evaluate_strategy().
        name:     slug identifying the strategy, e.g. 'baseline_zs'.
        eval_dir: directory for eval artefacts (created if absent).

    Returns:
        Path to the written file.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    out = eval_dir / f"{name}_{ts()}.json"
    report.save(out)
    return out


def load_report(path: Path) -> dict:
    """Load a serialised EvalReport dict. Raw dict returned, this is."""
    return json.loads(path.read_text())


def model_slug(model_id: str | None, *, mock: bool = False) -> str:
    """Filesystem-safe short tag for a model id (e.g. 'Qwen/Qwen2.5-7B-Instruct' → 'qwen2.5-7b-instruct')."""
    if mock or not model_id:
        return "mock"
    tail = model_id.rsplit("/", 1)[-1]
    return tail.lower().replace(" ", "-")