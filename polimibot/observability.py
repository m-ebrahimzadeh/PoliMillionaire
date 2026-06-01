"""Retrieval observability utilities.

Three public entry points:

    print_retrieval_summary(extras) — compact 2-4 line per-question summary
                                      printed during a live game session.
                                      Called from runner.py after each answer.

    show_trace(sample, idx)      — pretty-print every pipeline detail for a
                                   single EvalSample (question, retrieved
                                   passages, gate status, live-search outcome,
                                   tier routing, ensemble arm votes).

    retrieval_dashboard(report)  — aggregate health metrics across an entire
                                   EvalReport: gate rate, live-search success,
                                   top-score distributions, gated vs. ungated
                                   accuracy, category breakdown.

All functions are no-ops on strategies that don't emit extras (e.g.
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


# ── Live-game per-question retrieval summary ─────────────────────────────

def print_retrieval_summary(extras: dict) -> None:
    """Print a compact retrieval summary during a live game session.

    Called from ``runner.py`` after each strategy answer when ``verbose=True``.
    Prints 2-4 indented lines covering:

    - Whether offline RAG was used or gated (and why).
    - If live search fired: query, articles found, latency, passages kept.
    - If live search found nothing or all passages were below threshold: warning.
    - Decoding path and parse status.

    Gracefully no-ops when ``extras`` is empty (non-RAG strategies).

    Args:
        extras: the ``out.extras`` dict from a :class:`StrategyOutput`.
    """
    if not extras:
        return

    # ── Offline retrieval line ─────────────────────────────────────────────
    gated      = extras.get("gated_by_min_score", False)
    top_score  = extras.get("top_score")
    threshold  = extras.get("min_score_threshold")
    n_passages = extras.get("n_passages", 0)
    top_src    = extras.get("top_source")

    if gated:
        score_str = f"{top_score:.4f}" if top_score is not None else "n/a"
        thresh_str = f"{threshold}" if threshold is not None else "?"
        print(f"  🔎 GATED  (offline top_score={score_str} < {thresh_str})")
    else:
        score_str = f"{top_score:.4f}" if top_score is not None else "n/a"
        src_str   = f'  src: "{top_src}"' if top_src else ""
        print(f"  🔎 OFFLINE  top_score={score_str}  passages={n_passages}{src_str}")

    # ── Live-search block (only shown when live search was considered) ─────
    live_fired   = extras.get("live_search_fired", False)
    live_latency = extras.get("live_search_latency")
    live_articles = extras.get("live_search_articles") or []
    live_query   = extras.get("live_search_query")
    all_below    = extras.get("live_all_below_threshold", False)

    # live_search_latency being set means the API was called (even if 0 articles)
    live_attempted = live_latency is not None
    live_top_score = extras.get("live_top_score")

    if live_attempted or live_fired:
        lat_str = f"{live_latency:.2f}s" if live_latency is not None else "?"
        if live_query:
            q_preview = str(live_query)[:80] + ("…" if len(str(live_query)) > 80 else "")
            print(f"     ↳ live query : \"{q_preview}\"")

        if not live_articles:
            print(f"     ↳ Wikipedia  : (no articles found)  [{lat_str}]")
        else:
            art_str = ", ".join(f'"{a}"' for a in live_articles[:3])
            if len(live_articles) > 3:
                art_str += f", +{len(live_articles) - 3} more"
            print(f"     ↳ Wikipedia  : {art_str}  [{lat_str}]")

            if all_below:
                score_str = f"  best={live_top_score:.4f}" if live_top_score is not None else ""
                print(f"     ↳ ⚠ all live passages below threshold{score_str} — no context sent to LLM")
            elif live_fired:
                live_n = extras.get("n_passages", 0)
                score_str = f"  top_score={live_top_score:.4f}" if live_top_score is not None else ""
                print(f"     ↳ live passages used: {live_n}{score_str} ✓")

    # ── Tool hit line (shown only when a deterministic tool answered) ─────
    tool_name = extras.get("tool")
    if tool_name:
        # Direct tools store "expr"; rewrite path stores "rewrite_expr"
        expr   = extras.get("expr") or extras.get("rewrite_expr", "")
        result = extras.get("result", "")
        path   = extras.get("path", "")
        expr_str   = f"  expr={str(expr)[:60]}"    if expr   else ""
        result_str = f"  result={str(result)[:30]}" if result and result != "complex" else ""
        path_str   = f"  [{path}]"                  if path and path not in ("direct_tool",) else ""
        print(f"     TOOL HIT [{tool_name}]{path_str}{expr_str}{result_str}")

    # ── Decoding / parse line ──────────────────────────────────────────────
    decoding  = extras.get("decoding_path", "—")
    parse_ok  = extras.get("parse_ok")
    parse_str = "parse OK ✓" if parse_ok else ("parse FAIL ✗" if parse_ok is False else "—")
    print(f"     decoding={decoding}  {parse_str}")


# ── Post-game summary ────────────────────────────────────────────────────

def print_game_summary(result: "GameResult") -> None:
    """Print a full post-game diagnostic summary from a :class:`GameResult`.

    Covers 6 sections:

    ①  Context source breakdown — how many questions used offline RAG,
        live search, or got no context (gated + live failed), and the
        accuracy in each case.
    ②  Per-category breakdown — question count, accuracy, and context-
        source split per category (only shown when >1 category present).
    ③  Difficulty curve — accuracy and mean top_score grouped by level
        ranges (1-5, 6-10, 11-15) so you can spot where the bot starts
        failing.
    ④  Live-search efficiency — gate rate, Wikipedia hit rate, threshold
        pass rate, wasted searches, mean latency and top score.
    ⑤  Confidence calibration — quick sanity check (high-conf correct vs.
        high-conf wrong counts).
    ⑥  Worst misses — up to 3 wrong answers ordered by confidence (most
        confidently wrong first), with the context source that was used.

    Sections ②–⑥ only appear when the relevant data exists (e.g. ② is
    hidden for single-category competitions; ④ is hidden when live search
    was never attempted).

    Args:
        result: a :class:`GameResult` returned by :func:`play_game`.
                Requires ``result.question_records`` to be populated
                (always the case when using the current runner).
    """
    from collections import defaultdict

    records = result.question_records
    n = len(records)
    if n == 0:
        print("print_game_summary: no question records available.")
        return

    W = 74  # box width

    # ── Classify each question ────────────────────────────────────────────
    # Three mutually-exclusive context sources:
    #   offline  — not gated; used offline RAG passages
    #   live     — gated, live fired, passages above threshold
    #   none     — gated, live didn't succeed (no articles / all below thresh)
    def _classify(rec) -> str:
        e = rec.extras or {}
        if not e.get("gated_by_min_score", False):
            return "offline"
        if e.get("live_search_fired") and not e.get("live_all_below_threshold"):
            return "live"
        return "none"

    classes = [_classify(r) for r in records]
    n_offline = classes.count("offline")
    n_live    = classes.count("live")
    n_none    = classes.count("none")

    def _acc(recs):
        if not recs:
            return None
        correct = sum(1 for r in recs if r.correct is True)
        return correct / len(recs)

    offline_recs = [r for r, c in zip(records, classes) if c == "offline"]
    live_recs    = [r for r, c in zip(records, classes) if c == "live"]
    none_recs    = [r for r, c in zip(records, classes) if c == "none"]

    n_correct = sum(1 for r in records if r.correct is True)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _pct(a, b):
        return f"{a/b:.0%}" if b else "—"

    def _acc_str(recs):
        a = _acc(recs)
        return f"{a:.0%}" if a is not None else "—"

    def _bar(w, filled):
        """Simple ASCII bar of width w, filled fraction."""
        n_fill = round(w * filled) if filled else 0
        return "█" * n_fill + "░" * (w - n_fill)

    print(f"\n{'╔' + '═' * W + '╗'}")
    print(f"  POST-GAME SUMMARY")
    print(f"  Strategy : {result.strategy_name}")
    print(f"  Result   : Level {result.final_level}  |  "
          f"{n_correct}/{n} correct ({_pct(n_correct, n)})  |  "
          f"Earned: {result.earned_amount:,.0f}")
    print(f"  Time     : {result.elapsed_seconds:.1f}s total  |  "
          f"avg {result.elapsed_seconds/n:.1f}s/question")
    print(f"{'╠' + '═' * W + '╣'}")

    # ── ① Context source breakdown ────────────────────────────────────────
    print(f"\n  ① CONTEXT SOURCE BREAKDOWN")
    header = f"  {'Source':<30} {'N':>3}  {'Correct':>7}  {'Accuracy':>8}"
    print(header)
    print(f"  {'-'*60}")
    rows = [
        ("📚 Offline RAG",  n_offline, offline_recs),
        ("🌐 Live search",  n_live,    live_recs),
        ("⚠️  No context",   n_none,    none_recs),
    ]
    for label, count, recs in rows:
        acc = _acc_str(recs)
        bar = _bar(12, count / n if n else 0)
        print(f"  {label:<30} {count:>3}  {_pct(count,n):>7}   {acc:>7}  {bar}")

    # ── ② Per-category breakdown ──────────────────────────────────────────
    cat_map: dict[str, list] = defaultdict(list)
    for rec, cls in zip(records, classes):
        cat = rec.extras.get("category_filter") or "unknown"
        cat_map[cat].append((rec, cls))

    if len(cat_map) > 1:
        print(f"\n  ② PER-CATEGORY BREAKDOWN")
        print(f"  {'Category':<14} {'N':>3}  {'Acc':>5}  {'Offline':>7}  {'Live':>5}  {'No-ctx':>6}")
        print(f"  {'-'*52}")
        for cat in sorted(cat_map):
            recs_cls = cat_map[cat]
            cat_recs  = [r for r, _ in recs_cls]
            cat_off   = sum(1 for _, c in recs_cls if c == "offline")
            cat_live  = sum(1 for _, c in recs_cls if c == "live")
            cat_none  = sum(1 for _, c in recs_cls if c == "none")
            cat_acc   = _acc_str(cat_recs)
            print(f"  {cat:<14} {len(cat_recs):>3}  {cat_acc:>5}  "
                  f"{cat_off:>7}  {cat_live:>5}  {cat_none:>6}")

    # ── ③ Difficulty curve ────────────────────────────────────────────────
    bands = [(1, 5), (6, 10), (11, 15)]
    band_data = []
    for lo, hi in bands:
        band_recs = [r for r in records if lo <= r.level <= hi]
        if not band_recs:
            continue
        scores = [r.extras.get("top_score") for r in band_recs
                  if r.extras.get("top_score") is not None]
        avg_score = sum(scores) / len(scores) if scores else None
        band_data.append((lo, hi, band_recs, avg_score))

    if len(band_data) > 1:
        print(f"\n  ③ DIFFICULTY CURVE  (by level range)")
        print(f"  {'Levels':<12} {'N':>3}  {'Accuracy':>8}  {'Avg top_score':>14}")
        print(f"  {'-'*44}")
        for lo, hi, band_recs, avg_score in band_data:
            acc = _acc_str(band_recs)
            sc  = f"{avg_score:.4f}" if avg_score is not None else "—"
            print(f"  {f'{lo}-{hi}':<12} {len(band_recs):>3}  {acc:>8}  {sc:>14}")

    # ── ④ Live-search efficiency ──────────────────────────────────────────
    live_attempted = [r for r in records
                      if r.extras.get("live_search_latency") is not None]
    if live_attempted:
        n_gated      = sum(1 for r in records
                           if r.extras.get("gated_by_min_score"))
        n_fired      = sum(1 for r in records
                           if r.extras.get("live_search_fired"))
        n_above      = len(live_recs)
        n_all_below  = sum(1 for r in records
                           if r.extras.get("live_all_below_threshold"))
        latencies    = [r.extras["live_search_latency"] for r in live_attempted]
        live_scores  = [r.extras["live_top_score"] for r in live_recs
                        if r.extras.get("live_top_score") is not None]

        acc_live = _acc_str(live_recs)
        acc_none = _acc_str(none_recs)
        boost    = ""
        a_live   = _acc(live_recs)
        a_none   = _acc(none_recs)
        if a_live is not None and a_none is not None:
            boost = f"  (vs no-context: {a_none:.0%}, boost +{(a_live - a_none)*100:.0f}pp)"

        print(f"\n  ④ LIVE SEARCH EFFICIENCY")
        print(f"    Gated (offline weak)        : {n_gated}/{n}  ({_pct(n_gated, n)})")
        print(f"    Wikipedia found articles    : {n_fired}/{n_gated}  ({_pct(n_fired, n_gated)})")
        if n_fired:
            print(f"    Passages above threshold    : {n_above}/{n_fired}  ({_pct(n_above, n_fired)})")
        if n_all_below:
            print(f"    Wasted (all below thresh)   : {n_all_below}/{n_fired}  ({_pct(n_all_below, n_fired)})")
        if latencies:
            print(f"    Mean latency                : {sum(latencies)/len(latencies):.2f}s")
        if live_scores:
            print(f"    Mean live top_score (kept)  : {sum(live_scores)/len(live_scores):.4f}")
        if n_live:
            print(f"    Live accuracy               : {acc_live}{boost}")

    # ── ⑤ Confidence calibration ──────────────────────────────────────────
    hi_conf = [r for r in records if r.confidence is not None and r.confidence >= 0.80]
    lo_conf = [r for r in records if r.confidence is not None and r.confidence < 0.50]
    if hi_conf or lo_conf:
        print(f"\n  ⑤ CONFIDENCE CALIBRATION")
        if hi_conf:
            hi_correct = sum(1 for r in hi_conf if r.correct is True)
            hi_wrong   = len(hi_conf) - hi_correct
            print(f"    High conf (≥80%)  correct : {hi_correct}/{len(hi_conf)}  "
                  f"({_pct(hi_correct, len(hi_conf))})")
            if hi_wrong:
                print(f"    High conf (≥80%)  wrong   : {hi_wrong}/{len(hi_conf)}  "
                      f"({_pct(hi_wrong, len(hi_conf))})  ← overconfidence")
        if lo_conf:
            lo_correct = sum(1 for r in lo_conf if r.correct is True)
            print(f"    Low conf  (<50%)  correct : {lo_correct}/{len(lo_conf)}  "
                  f"({_pct(lo_correct, len(lo_conf))})")
            lo_wrong = len(lo_conf) - lo_correct
            if lo_wrong:
                print(f"    Low conf  (<50%)  wrong   : {lo_wrong}/{len(lo_conf)}  "
                      f"({_pct(lo_wrong, len(lo_conf))})")

    # ── ⑥ Worst misses ────────────────────────────────────────────────────
    wrong = [r for r in records if r.correct is False]
    wrong_sorted = sorted(wrong,
                          key=lambda r: r.confidence if r.confidence is not None else 0.0,
                          reverse=True)
    if wrong_sorted:
        print(f"\n  ⑥ WORST MISSES  (wrong, ordered by confidence)")
        letters = ("A", "B", "C", "D")
        for r in wrong_sorted[:3]:
            cat   = r.extras.get("category_filter") or "?"
            conf  = f"{r.confidence:.0%}" if r.confidence is not None else "?"
            pred  = letters[r.chosen_index] if 0 <= r.chosen_index < 4 else "?"
            src   = _classify(r)
            src_icon = {"offline": "📚", "live": "🌐", "none": "⚠️"}.get(src, "?")
            q_short = r.question_text[:70] + ("…" if len(r.question_text) > 70 else "")
            top_sc = r.extras.get("top_score")
            ts_str = f"  top_score={top_sc:.4f}" if top_sc is not None else ""
            live_sc = r.extras.get("live_top_score")
            lts_str = f"  live_top={live_sc:.4f}" if live_sc is not None else ""
            print(f"    L{r.level} [{cat}] → {pred} (conf {conf})  {src_icon}")
            print(f"      Q: {q_short}")
            print(f"      {ts_str}{lts_str}")

    print(f"\n{'╚' + '═' * W + '╝'}\n")


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
    # When live search fired but ALL passages were below threshold, the LLM
    # received no context.  We display the merged offline+live debug pool
    # (tagged by source) so you can see what was scored and rejected.
    all_below = e.get("live_all_below_threshold", False)
    debug_pool = e.get("debug_passages") or []
    passages = e.get("passages") or []

    if all_below and debug_pool:
        thresh = e.get("min_score_threshold", "?")
        print(f"{'─' * 72}")
        print(f"  TOP PASSAGES  ⚠ ALL below threshold ({thresh}) — NOT sent to LLM")
        print(f"    (merged offline + live; best scored first)")
        for i, p in enumerate(debug_pool[:6]):
            src      = p.get("source", "?")
            score    = p.get("score", "?")
            chunk_id = p.get("chunk_id", "?")
            pool_src = p.get("pool", "?")
            preview  = p.get("text_preview", "")
            tag      = "[LIVE]" if pool_src == "live" else "[OFFLINE]"
            line = f"    [{i + 1}] {src}  (score={score}, chunk={chunk_id})  {tag}"
            if preview:
                line += f"\n        ↳ {preview[:120]}"
            print(line)
    elif passages:
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
        # "No results" = gated questions where the search was attempted but
        # Wikipedia returned no articles (live_search_fired stays False while
        # live_search_latency is populated, proving the API call was made).
        n_no_results = sum(
            1 for s in live_samples
            if s.extras.get("gated_by_min_score")
            and not s.extras.get("live_search_fired")
            and s.extras.get("live_search_latency") is not None
        )
        n_gated_total = sum(1 for s in live_samples if s.extras.get("gated_by_min_score"))
        latencies  = [s.extras["live_search_latency"] for s in live_samples
                      if s.extras.get("live_search_latency") is not None]
        print(f"\n  LIVE SEARCH")
        print(f"    Fired / total : {n_fired}/{len(live_samples)}")
        if n_no_results:
            pct = n_no_results / n_gated_total if n_gated_total else 0.0
            print(f"    No results    : {n_no_results}/{n_gated_total}  "
                  f"({pct:.1%})  ← gated but Wikipedia returned nothing")
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
