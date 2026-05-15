"""Retrieval observability utilities.

Two public entry points:

    show_trace(sample, idx)      — pretty-print every pipeline detail for a
                                   single EvalSample (question, retrieved
                                   passages, gate status, live-search outcome,
                                   tier routing, ensemble arm votes).

    retrieval_dashboard(report)  — aggregate health metrics across an entire
                                   EvalReport: gate rate, live-search success,
                                   top-score distributions, gated vs. ungated
                                   accuracy, category breakdown.

Both functions are no-ops on strategies that don't emit extras (e.g.
BaselineLLMStrategy) — they degrade gracefully to a minimal summary.

Usage in the notebook (after Section 2.2):

    from polimibot.observability import show_trace, retrieval_dashboard

    # Full dashboard after eval:
    retrieval_dashboard(report)

    # Inspect the three most confidently wrong answers:
    wrong = [s for s in report.samples if not s.correct]
    for i, s in enumerate(sorted(wrong, key=lambda x: -x.confidence)[:3]):
        show_trace(s, idx=i)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports — only used for type hints.
    from .eval.evaluator import EvalReport, EvalSample


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe(d: dict, key: str, default="—"):
    """Return d[key] if present and not None, else default."""
    v = d.get(key)
    return default if v is None else v


# ── Per-question trace ────────────────────────────────────────────────────

def show_trace(sample: "EvalSample", idx: int = 0) -> None:
    """Pretty-print the full pipeline trace for one EvalSample.

    Works with any strategy. When extras are absent (non-RAG strategies),
    prints a compact summary of the decision only.

    Args:
        sample: An EvalSample from EvalReport.samples.
        idx:    Display index (cosmetic only — helps correlate with loops).
    """
    e = sample.extras or {}
    status = "✓ CORRECT" if sample.correct else "✗ WRONG"
    cat = sample.category.value if sample.category else "unknown"
    options_letters = ("A", "B", "C", "D")

    print(f"{'═' * 72}")
    print(f"  [{idx}]  {status}   Level {sample.level}  |  Category: {cat}")
    print(f"{'─' * 72}")
    print(f"  Q:  {sample.question_text[:120]}")
    print()
    for i, opt in enumerate(sample.options):
        marker = "→" if i == sample.predicted_index else " "
        correct_marker = "✓" if i == sample.correct_index else " "
        print(f"    {marker} {correct_marker} {options_letters[i]}. {opt}")
    print()
    print(f"  Predicted : {options_letters[sample.predicted_index]}  "
          f"(correct: {options_letters[sample.correct_index]})   "
          f"Confidence: {sample.confidence:.2%}   "
          f"Elapsed: {sample.elapsed_seconds:.2f}s")

    # Only print retrieval details when extras exist.
    if not e:
        print(f"{'═' * 72}\n")
        return

    print(f"{'─' * 72}")
    print("  RETRIEVAL")
    query = _safe(e, "query")
    if len(str(query)) > 110:
        query = str(query)[:110] + "…"
    print(f"    Query sent    : {query}")
    print(f"    Passages      : {_safe(e, 'n_passages')}  |  "
          f"Top source: {_safe(e, 'top_source')}  |  "
          f"Top score: {_safe(e, 'top_score')}")
    print(f"    Category filt : {_safe(e, 'category_filter', 'off')}  |  "
          f"Hybrid: {_safe(e, 'hybrid', False)}  |  "
          f"Multi-query: {_safe(e, 'multi_query', False)}  |  "
          f"Reranked: {_safe(e, 'reranked', False)}")
    print(f"    Gated         : {_safe(e, 'gated_by_min_score', False)}  |  "
          f"Threshold: {_safe(e, 'min_score_threshold')}")
    print(f"    Decoding      : {_safe(e, 'decoding_path')}  |  "
          f"Parse OK: {_safe(e, 'parse_ok')}")

    # Live-search block
    live_fired = e.get("live_search_fired", False)
    if live_fired is not False or e.get("live_search_articles"):
        print(f"{'─' * 72}")
        print("  LIVE SEARCH")
        live_q = e.get("live_search_query")
        if live_q is not None:
            print(f"    Query         : {str(live_q)[:100]}")
        print(f"    Fired         : {live_fired}")
        articles = e.get("live_search_articles") or []
        print(f"    Articles      : {articles if articles else '(none)'}")
        print(f"    Latency       : {_safe(e, 'live_search_latency')}s")

    # Passage list
    passages = e.get("passages") or []
    if passages:
        print(f"{'─' * 72}")
        print("  TOP PASSAGES")
        for i, p in enumerate(passages[:5]):
            src = p.get("source", "?")
            score = p.get("score", "?")
            chunk_id = p.get("chunk_id", "?")
            preview = p.get("text_preview", "")
            line = f"    [{i + 1}] {src}  (score={score}, chunk={chunk_id})"
            if preview:
                line += f"\n        ↳ {preview[:120]}"
            print(line)

    # Tier routing
    tier = e.get("tier_selected")
    if tier:
        print(f"{'─' * 72}")
        print("  TIER ROUTING")
        print(f"    Tier selected    : {tier}")
        print(f"    Escalated        : {_safe(e, 'escalated', False)}")
        margin = e.get("margin")
        if margin is not None:
            print(f"    Margin           : {margin:.4f}  "
                  f"(threshold: {_safe(e, 'escalation_threshold')})")

    # Ensemble arm votes
    arm_votes = e.get("arm_votes")
    if arm_votes:
        print(f"{'─' * 72}")
        print("  ENSEMBLE ARMS")
        for arm_name, arm_info in arm_votes.items():
            chosen = arm_info.get("chosen_index", "?")
            conf   = arm_info.get("confidence", 0.0)
            letter = ("A", "B", "C", "D")[chosen] if isinstance(chosen, int) else "?"
            print(f"    {arm_name:<30} → {letter}  conf={conf:.2%}")
        fused = e.get("fused_probs")
        if fused:
            probs_str = "  ".join(
                f"{l}={fused.get(l, 0):.3f}" for l in ("A", "B", "C", "D")
            )
            print(f"    Fused probs: {probs_str}")

    print(f"{'═' * 72}\n")


# ── Aggregate dashboard ───────────────────────────────────────────────────

def retrieval_dashboard(report: "EvalReport") -> None:
    """Print a compact retrieval-health dashboard for an EvalReport.

    Covers:
    - Gate rate (how often offline retrieval was gated)
    - Live-search success rate and latency
    - Mean top_score for correct vs. wrong questions
    - Gated vs. ungated accuracy (were gates helpful?)
    - Tier routing frequency (when tiered strategy used)
    - Ensemble arm win rate (when ensemble strategy used)
    - Per-category gate rate breakdown

    No-op (prints a notice) if no samples have extras.
    """
    samples = report.samples
    n = len(samples)
    if n == 0:
        print("Dashboard: no samples in report.")
        return

    has_extras = any(s.extras for s in samples)
    if not has_extras:
        print(f"Dashboard: no extras in report samples — "
              f"strategy '{report.strategy_name}' does not emit pipeline traces.")
        return

    print(f"\n{'═' * 68}")
    print(f"  RETRIEVAL HEALTH DASHBOARD — {report.strategy_name}")
    print(f"  Overall accuracy: {report.accuracy:.1%}  |  N = {n}")
    print(f"{'═' * 68}")

    # ── Gate stats ────────────────────────────────────────────────────────
    rag_samples = [s for s in samples if s.extras.get("n_passages") is not None]
    if rag_samples:
        n_rag = len(rag_samples)
        n_gated = sum(1 for s in rag_samples if s.extras.get("gated_by_min_score"))
        print(f"\n  RETRIEVAL GATING  (RAG questions: {n_rag}/{n})")
        print(f"    Gate rate     : {n_gated}/{n_rag}  ({n_gated/n_rag:.1%})")

        # Mean top_score correct vs wrong
        scores_correct = [s.extras["top_score"] for s in rag_samples
                          if s.correct and s.extras.get("top_score") is not None]
        scores_wrong   = [s.extras["top_score"] for s in rag_samples
                          if not s.correct and s.extras.get("top_score") is not None]
        if scores_correct:
            print(f"    top_score (correct questions) : "
                  f"mean={sum(scores_correct)/len(scores_correct):.4f}  "
                  f"min={min(scores_correct):.4f}  max={max(scores_correct):.4f}")
        if scores_wrong:
            print(f"    top_score (wrong questions)   : "
                  f"mean={sum(scores_wrong)/len(scores_wrong):.4f}  "
                  f"min={min(scores_wrong):.4f}  max={max(scores_wrong):.4f}")

        # Gated vs. ungated accuracy
        gated_s   = [s for s in rag_samples if s.extras.get("gated_by_min_score")]
        ungated_s = [s for s in rag_samples if not s.extras.get("gated_by_min_score")]
        if gated_s:
            acc_g = sum(s.correct for s in gated_s) / len(gated_s)
            print(f"    Accuracy when gated           : {acc_g:.1%}  "
                  f"(n={len(gated_s)})")
        if ungated_s:
            acc_u = sum(s.correct for s in ungated_s) / len(ungated_s)
            print(f"    Accuracy when NOT gated       : {acc_u:.1%}  "
                  f"(n={len(ungated_s)})")

        # Per-category gate rate
        from collections import defaultdict
        cat_gated: dict[str, list] = defaultdict(list)
        for s in rag_samples:
            key = s.category.value if s.category else "unknown"
            cat_gated[key].append(s.extras.get("gated_by_min_score", False))
        if len(cat_gated) > 1:
            print(f"\n  GATE RATE BY CATEGORY")
            for cat, flags in sorted(cat_gated.items()):
                g = sum(flags)
                print(f"    {cat:<16} {g}/{len(flags)}  ({g/len(flags):.0%})")

    # ── Live-search stats ─────────────────────────────────────────────────
    live_samples = [s for s in samples if s.extras.get("live_search_fired") is not None]
    if live_samples:
        n_fired    = sum(1 for s in live_samples if s.extras.get("live_search_fired"))
        n_success  = sum(1 for s in live_samples
                         if s.extras.get("live_search_fired")
                         and s.extras.get("live_search_articles"))
        latencies  = [s.extras["live_search_latency"] for s in live_samples
                      if s.extras.get("live_search_latency") is not None]
        print(f"\n  LIVE SEARCH")
        print(f"    Fired / total : {n_fired}/{len(live_samples)}")
        if n_fired:
            print(f"    Success rate  : {n_success}/{n_fired}  "
                  f"({n_success/n_fired:.1%})")
        if latencies:
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[len(latencies_sorted) // 2]
            p95 = latencies_sorted[min(int(len(latencies_sorted) * 0.95),
                                       len(latencies_sorted) - 1)]
            print(f"    Latency       : p50={p50:.2f}s  p95={p95:.2f}s")

    # ── Tier routing stats ────────────────────────────────────────────────
    tier_samples = [s for s in samples if s.extras.get("tier_selected")]
    if tier_samples:
        from collections import Counter
        tier_counts = Counter(s.extras["tier_selected"] for s in tier_samples)
        escalated = sum(1 for s in tier_samples if s.extras.get("escalated"))
        print(f"\n  TIER ROUTING  (n={len(tier_samples)})")
        for tier, cnt in sorted(tier_counts.items()):
            print(f"    {tier:<16} {cnt}  ({cnt/len(tier_samples):.0%})")
        print(f"    Escalations   : {escalated}  "
              f"({escalated/len(tier_samples):.1%})")

    # ── Ensemble arm stats ────────────────────────────────────────────────
    ensemble_samples = [s for s in samples if s.extras.get("arm_votes")]
    if ensemble_samples:
        from collections import defaultdict
        arm_wins: dict[str, int] = defaultdict(int)
        for s in ensemble_samples:
            votes: dict = s.extras["arm_votes"]
            if votes:
                # The arm whose letter matches the final chosen index "won"
                final_letter = ("A", "B", "C", "D")[s.predicted_index]
                for arm_name, arm_info in votes.items():
                    ci = arm_info.get("chosen_index")
                    if isinstance(ci, int) and ("A", "B", "C", "D")[ci] == final_letter:
                        arm_wins[arm_name] += 1
        if arm_wins:
            print(f"\n  ENSEMBLE ARMS  (n={len(ensemble_samples)})")
            for arm, wins in sorted(arm_wins.items(), key=lambda x: -x[1]):
                print(f"    {arm:<30} agrees with final: "
                      f"{wins}/{len(ensemble_samples)}  "
                      f"({wins/len(ensemble_samples):.0%})")

    print(f"{'═' * 68}\n")
