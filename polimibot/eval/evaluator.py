"""Offline strategy evaluator. Replays gold items, computes EvalReport.

Caller is responsible for warm_up() / shutdown() — this function is
composable: run multiple evaluate_strategy() calls on the same loaded model.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from ..config import Category
from ..strategies.base import Strategy, StrategyInput, StrategyOutput
from .gold_set import GoldItem


# ── Calibration ───────────────────────────────────────────────────────────

def _ece(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    Groups predictions into n_bins buckets by stated confidence,
    then averages the |accuracy - confidence| gap weighted by bucket size.
    Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, corrects):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))
    ece = 0.0
    n = len(confidences)
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc  = sum(int(ok) for _, ok in bucket) / len(bucket)
        ece += abs(avg_acc - avg_conf) * len(bucket) / n
    return round(ece, 6)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * p))
    return s[idx]


# ── Result records ────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One question's outcome. Kept for post-hoc error analysis.

    ``extras`` is the full ``StrategyOutput.extras`` dict — it carries
    everything the strategy computed internally (retrieved passages, gate
    status, live-search results, per-arm votes, tier routing, etc.).
    Stored here so callers can run post-hoc diagnostics without re-running
    the strategy.  Defaults to empty dict for strategies that don't emit extras.
    """
    question_text: str
    options: tuple[str, ...]
    correct_index: int
    predicted_index: int
    correct: bool
    confidence: float
    elapsed_seconds: float
    category: Optional[Category]
    level: int
    extras: dict = field(default_factory=dict)


@dataclass
class CategoryStats:
    n: int
    correct: int
    accuracy: float
    ece: float


@dataclass
class ToolStats:
    """Aggregate metrics for one tool (e.g. maths_tool, sympy_direct)."""
    n_fired: int          # questions where this tool returned an answer
    n_correct: int        # of those, how many were correct
    accuracy: float       # n_correct / n_fired  (0.0 if n_fired == 0)


@dataclass
class EvalReport:
    """Aggregate metrics from one evaluate_strategy() run."""
    strategy_name: str
    n_total: int
    accuracy: float
    ece: float
    by_category: dict[str, CategoryStats]
    latency_p50: float
    latency_p95: float
    latency_mean: float
    tool_stats: dict[str, ToolStats] = field(default_factory=dict)
    n_tool_answered: int = 0     # questions answered by any tool (not fallback LLM)
    n_llm_answered: int = 0      # questions that fell through to the LLM
    samples: list[EvalSample] = field(default_factory=list, repr=False)

    def print_summary(self) -> None:
        """Human-readable report to stdout."""
        print(f"\n{'─'*55}")
        print(f"  Strategy : {self.strategy_name}")
        print(f"  N        : {self.n_total}")
        print(f"  Accuracy : {self.accuracy:.1%}")
        print(f"  ECE      : {self.ece:.4f}  (lower = better calibrated)")
        print(f"  Latency  : p50={self.latency_p50:.2f}s  p95={self.latency_p95:.2f}s")
        if self.n_tool_answered or self.tool_stats:
            pct = self.n_tool_answered / self.n_total if self.n_total else 0.0
            print(f"\n  Tool coverage : {self.n_tool_answered}/{self.n_total} "
                  f"({pct:.1%} answered by a tool, "
                  f"{self.n_llm_answered} fell to LLM)")
            for tname, ts in sorted(self.tool_stats.items()):
                print(f"    {tname:<20} fired={ts.n_fired}  "
                      f"correct={ts.n_correct}  acc={ts.accuracy:.1%}")
        print(f"\n  Per-category:")
        for cat, stats in sorted(self.by_category.items()):
            print(f"    {cat:<16} acc={stats.accuracy:.1%}  ece={stats.ece:.4f}  n={stats.n}")
        print(f"{'─'*55}\n")

    def save(self, path: Path) -> None:
        """Write report as JSON (samples excluded — too large)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("samples")                        # keep file small
        # Category keys are enums → already strings via asdict
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        print(f"Report saved → {path}")


# ── Evaluator ────────────────────────────────────────────────────────────

def evaluate_strategy(
    strategy: Strategy,
    gold_set: list[GoldItem],
    *,
    verbose: bool = True,
    index_grower=None,   # Optional[IndexGrower] — avoids circular import
) -> EvalReport:
    """Replay gold_set through strategy offline. No live API calls.

    Args:
        strategy: already warm; caller owns warm_up/shutdown lifecycle.
        gold_set: list of GoldItems from load_gold_set() or harvest_gold_set().
        verbose: print a progress dot every 10 questions.
        index_grower: optional IndexGrower instance.  When provided, mirrors
            the online learning loop: confirms live-fetched articles on correct
            answers and discards them on wrong ones, then flushes to disk after
            all questions are processed.  Only has an effect when the strategy
            also has an index_grower attached and live-search fires.

    Returns:
        EvalReport with accuracy, ECE, latency, per-category breakdown.
    """
    samples: list[EvalSample] = []

    for i, item in enumerate(gold_set):
        inp = StrategyInput(
            question=item.question_text,
            options=item.options,
            level=item.level,
            category=item.category,
            competition_id=item.competition_id,
        )
        t0 = time.monotonic()
        out: StrategyOutput = strategy.answer(inp)
        elapsed = time.monotonic() - t0

        correct = (out.chosen_index == item.correct_index)

        if index_grower is not None:
            _qid = f"lvl_{item.level}"
            if correct:
                index_grower.confirm(_qid)
            else:
                index_grower.discard(_qid)

        samples.append(EvalSample(
            question_text=item.question_text,
            options=item.options,
            correct_index=item.correct_index,
            predicted_index=out.chosen_index,
            correct=correct,
            confidence=out.confidence,
            elapsed_seconds=elapsed,
            category=item.category,
            level=item.level,
            extras=out.extras or {},  # preserve full pipeline trace for diagnostics
        ))

        if verbose and (i + 1) % 10 == 0:
            running_acc = sum(s.correct for s in samples) / len(samples)
            print(f"  [{i+1}/{len(gold_set)}]  running acc={running_acc:.1%}")

    if index_grower is not None:
        index_grower.flush()

    return _build_report(strategy.name, samples)


def _build_report(strategy_name: str, samples: list[EvalSample]) -> EvalReport:
    n = len(samples)
    if n == 0:
        return EvalReport(strategy_name, 0, 0.0, 0.0, {}, 0.0, 0.0, 0.0)

    accuracy = sum(s.correct for s in samples) / n
    ece = _ece([s.confidence for s in samples], [s.correct for s in samples])

    # Per-category
    by_cat: dict[str, list[EvalSample]] = defaultdict(list)
    for s in samples:
        key = s.category.value if s.category else "unknown"
        by_cat[key].append(s)

    cat_stats: dict[str, CategoryStats] = {}
    for cat_key, cat_samples in by_cat.items():
        cat_acc = sum(s.correct for s in cat_samples) / len(cat_samples)
        cat_ece = _ece(
            [s.confidence for s in cat_samples],
            [s.correct for s in cat_samples],
        )
        cat_stats[cat_key] = CategoryStats(
            n=len(cat_samples),
            correct=sum(s.correct for s in cat_samples),
            accuracy=cat_acc,
            ece=cat_ece,
        )

    # ── Tool-hit metrics ──────────────────────────────────────────────────
    tool_hits: dict[str, list[bool]] = defaultdict(list)  # tool_name → [correct, ...]
    for s in samples:
        t = s.extras.get("tool")
        if t:
            tool_hits[t].append(s.correct)

    tool_stats: dict[str, ToolStats] = {}
    for tname, results in tool_hits.items():
        fired   = len(results)
        correct = sum(results)
        tool_stats[tname] = ToolStats(
            n_fired=fired,
            n_correct=correct,
            accuracy=correct / fired if fired else 0.0,
        )

    n_tool_answered = sum(len(v) for v in tool_hits.values())
    n_llm_answered  = n - n_tool_answered

    latencies = [s.elapsed_seconds for s in samples]
    return EvalReport(
        strategy_name=strategy_name,
        n_total=n,
        accuracy=accuracy,
        ece=ece,
        by_category=cat_stats,
        latency_p50=_percentile(latencies, 0.5),
        latency_p95=_percentile(latencies, 0.95),
        latency_mean=sum(latencies) / n,
        tool_stats=tool_stats,
        n_tool_answered=n_tool_answered,
        n_llm_answered=n_llm_answered,
        samples=samples,
    )