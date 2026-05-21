"""Persist and load EvalReport instances.

One JSON file per strategy run. Loaded by make_leaderboard.py to
build the consolidated comparison table. Trivial, but centralised
this logic must be — else scattered saves will diverge in format.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from polimibot.eval.evaluator import EvalReport


def save_report(report: EvalReport, name: str, eval_dir: Path) -> Path:
    """Serialise *report* to ``eval_dir/{name}.json``.

    The *name* (i.e. ``report_id``) is expected to already embed a UTC
    timestamp via ``make_report_id()``, so no suffix is added here.

    Args:
        report:   completed EvalReport from evaluate_strategy().
        name:     slug identifying the run, e.g. the output of make_report_id().
        eval_dir: directory for eval artefacts (created if absent).

    Returns:
        Path to the written file.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    out = eval_dir / f"{name}.json"
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


def _collect_strategy_flags(strategy) -> set[str]:
    """Recursively inspect a strategy tree and return the set of active feature tokens."""
    from polimibot.strategies.rag_strategy import RAGStrategy
    from polimibot.strategies.tiered_strategy import TieredStrategy
    from polimibot.strategies.ensemble_strategy import EnsembleStrategy
    from polimibot.strategies.tool_strategy import ToolStrategy
    from polimibot.strategies.agent_strategy import AgentStrategy

    seen: set[int] = set()
    flags: set[str] = set()

    def _visit(s) -> None:
        if id(s) in seen:
            return
        seen.add(id(s))
        if isinstance(s, RAGStrategy):
            flags.add('rag')
            if s.use_reranker:      flags.add('rerank')
            if s.use_hybrid:        flags.add('hybrid')
            if s.use_multi_query:   flags.add('mq')
            if s.use_live_fallback: flags.add('livesearch')
        elif isinstance(s, TieredStrategy):
            for arm in (s.easy, s.medium, s.hard, s.maths_override):
                if arm is not None:
                    _visit(arm)
        elif isinstance(s, EnsembleStrategy):
            for arm in s.strategies:
                _visit(arm)
        elif isinstance(s, ToolStrategy):
            flags.add('math')
        elif isinstance(s, AgentStrategy):
            flags.add('agent')

    _visit(strategy)
    return flags


def make_report_id(
    strategy,
    model_id: str | None,
    prompt_style,
    *,
    mock: bool = False,
) -> str:
    """Return a self-describing, filesystem-safe run identifier.

    Format: ``{strategy}__{model}__{style}__{flags}__{utc_ts}``

    Examples::

        tiered__qwen2.5-7b-4bit__zero_shot__rag-rerank-hybrid-mq-livesearch__20260521_183042
        baseline__mock__few_shot__base__20260521_183042
    """
    short_tag = strategy.name.split('[', 1)[0]
    mslug_str = model_slug(model_id, mock=mock)
    style_val = prompt_style.value if hasattr(prompt_style, 'value') else str(prompt_style)
    active    = _collect_strategy_flags(strategy)
    ordered   = [f for f in ('rag', 'rerank', 'hybrid', 'mq', 'math', 'agent', 'livesearch')
                 if f in active]
    flag_str  = '-'.join(ordered) if ordered else 'base'
    ts_str    = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f'{short_tag}__{mslug_str}__{style_val}__{flag_str}__{ts_str}'