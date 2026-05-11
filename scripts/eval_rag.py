"""Ablation: Baseline LLM vs RAG on the frozen gold set.

Both strategies share one model load — this is the controlled comparison.
Run build_rag_index.py first if data/cache/knowledge.faiss does not exist.

Usage
-----
    python scripts/eval_rag.py --mock          # CPU smoke test
    python scripts/eval_rag.py                 # real model on Colab GPU
    python scripts/eval_rag.py --k 5           # retrieve 5 passages instead of 3
    python scripts/eval_rag.py --categories history science
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polimibot.config import PATHS, Category
from polimibot.eval.evaluator import EvalReport, evaluate_strategy
from polimibot.eval.gold_set import load_gold_set
from polimibot.eval.report_io import model_slug, save_report
from polimibot.models.mock import MockLLM
from polimibot.prompts.templates import PromptStyle
from polimibot.strategies.llm_baseline import BaselineLLMStrategy
from polimibot.strategies.rag_strategy import RAGStrategy


def main() -> int:
    args = _parse_args()
    PATHS.ensure()

    gold_path  = PATHS.eval_dir  / "gold_set.jsonl"
    index_path = PATHS.cache_dir / "knowledge"

    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not gold_path.exists():
        print(f"Gold set not found at {gold_path}.\n"
              "Run: python scripts/build_gold_set.py", file=sys.stderr)
        return 1

    if not args.mock and not index_path.with_suffix(".faiss").exists():
        print(f"FAISS index not found at {index_path}.faiss.\n"
              "Run: python scripts/build_rag_index.py", file=sys.stderr)
        return 1

    # ── Load gold set ────────────────────────────────────────────────────────
    gold = load_gold_set(gold_path)
    if args.categories:
        keep = {Category(c) for c in args.categories}
        gold = [g for g in gold if g.category in keep]
    print(f"Gold set: {len(gold)} items", end="")
    if args.categories:
        print(f"  (filtered to: {', '.join(args.categories)})", end="")
    print()

    if not gold:
        print("No gold items after filtering — check --categories values.", file=sys.stderr)
        return 1

    # ── Build LLM (once) ─────────────────────────────────────────────────────
    if args.mock:
        llm = MockLLM(name="mock", correctness=0.6)
        print("Using MockLLM (no GPU required)\n")
    else:
        from polimibot.models.llm import LLM, LLMSpec
        llm = LLM.load(LLMSpec(model_id=args.model, load_in_4bit=not args.no_4bit))

    # ── Build retriever ───────────────────────────────────────────────────────
    if args.mock:
        # MockRetriever: always returns empty passages → RAG degrades to baseline
        # This still validates the pipeline runs end-to-end without a real index.
        from polimibot.rag.retriever import Retriever
        from polimibot.rag.index import FAISSIndex
        from polimibot.rag.embedder import Embedder, EmbedderSpec

        class _EmptyRetriever:
            n_chunks = 0
            def retrieve(self, query: str, k: int = 3, *, category=None):
                return []

        retriever = _EmptyRetriever()  # type: ignore[assignment]
        print("MockRetriever: 0 passages (index not built — RAG == baseline here)\n")
    else:
        from polimibot.rag.retriever import Retriever
        from polimibot.rag.embedder import EmbedderSpec
        retriever = Retriever.from_saved(index_path, embedder_spec=EmbedderSpec())
        print(f"Retriever loaded: {retriever.n_chunks} chunks indexed\n")

    # ── Strategies ────────────────────────────────────────────────────────────
    style   = PromptStyle(args.style)
    baseline = BaselineLLMStrategy(llm, style=style, use_score_options=True)
    rag      = RAGStrategy(llm, retriever, k=args.k, style=style)  # type: ignore[arg-type]

    mslug = model_slug(args.model, mock=args.mock)
    base_slug = f"baseline_{'fs' if style == PromptStyle.FEW_SHOT else 'zs'}__{mslug}"
    rag_slug  = f"rag__{mslug}"

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print("Evaluating: BASELINE")
    baseline.warm_up()
    report_baseline = evaluate_strategy(baseline, gold, verbose=True)
    # Persist immediately — leaderboard builder reads these files.
    save_report(report_baseline, name=base_slug, eval_dir=PATHS.eval_dir)

    print("\n" + "=" * 55)
    print("Evaluating: RAG")
    # No warm_up() for RAG — LLM already warmed; retriever is CPU-only.
    report_rag = evaluate_strategy(rag, gold, verbose=True)
    save_report(report_rag, name=rag_slug, eval_dir=PATHS.eval_dir)

    baseline.shutdown()

    # ── Side-by-side comparison ───────────────────────────────────────────────
    _print_comparison(report_baseline, report_rag)
    return 0


def _print_comparison(baseline: EvalReport, rag: EvalReport) -> None:
    """Print a delta table: what RAG changed relative to baseline."""
    def _pct(v: float) -> str:
        return f"{v:.1%}"

    def _delta(a: float, b: float, *, higher_is_better: bool = True) -> str:
        d = b - a
        sign = "+" if d >= 0 else ""
        marker = ""
        if higher_is_better and d > 0.02:   marker = " ✓"
        if higher_is_better and d < -0.02:  marker = " ✗"
        if not higher_is_better and d < -0.005: marker = " ✓"
        if not higher_is_better and d > 0.005:  marker = " ✗"
        return f"{sign}{d:+.1%}{marker}"

    print("\n" + "═" * 65)
    print(f"  {'Metric':<22} {'Baseline':>10}  {'RAG':>10}  {'Δ (RAG-base)':>14}")
    print("─" * 65)
    print(f"  {'Accuracy':<22} {_pct(baseline.accuracy):>10}  {_pct(rag.accuracy):>10}  "
          f"{_delta(baseline.accuracy, rag.accuracy):>14}")
    print(f"  {'ECE (↓ better)':<22} {baseline.ece:>10.4f}  {rag.ece:>10.4f}  "
          f"{_delta(baseline.ece, rag.ece, higher_is_better=False):>14}")
    print(f"  {'Latency p50 (s)':<22} {baseline.latency_p50:>10.2f}  {rag.latency_p50:>10.2f}  "
          f"{_delta(baseline.latency_p50, rag.latency_p50, higher_is_better=False):>14}")
    print(f"  {'Latency p95 (s)':<22} {baseline.latency_p95:>10.2f}  {rag.latency_p95:>10.2f}  "
          f"{_delta(baseline.latency_p95, rag.latency_p95, higher_is_better=False):>14}")

    # Per-category accuracy delta
    all_cats = sorted(set(baseline.by_category) | set(rag.by_category))
    if all_cats:
        print("─" * 65)
        print(f"  {'Category accuracy':<22} {'Baseline':>10}  {'RAG':>10}  {'Δ':>14}")
        print("─" * 65)
        for cat in all_cats:
            b_acc = baseline.by_category[cat].accuracy if cat in baseline.by_category else float("nan")
            r_acc = rag.by_category[cat].accuracy      if cat in rag.by_category      else float("nan")
            print(f"  {cat:<22} {_pct(b_acc):>10}  {_pct(r_acc):>10}  "
                  f"{_delta(b_acc, r_acc):>14}")

    print("═" * 65)
    print(f"  Reports saved to {PATHS.eval_dir}/\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mock",       action="store_true")
    p.add_argument("--model",      default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--no-4bit",    action="store_true", dest="no_4bit")
    p.add_argument("--k",          type=int, default=3,
                   help="Number of passages to retrieve per question")
    p.add_argument("--style",      default="zero_shot",
                   choices=["zero_shot", "few_shot"])
    p.add_argument("--categories", nargs="+",
                   choices=[c.value for c in Category])
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())