"""Retrieval-only evaluation.

Without a recall@k harness, every RAG change is a guess. This module lets
the user label which Wikipedia article each gold question SHOULD have
retrieved, then measure how often the retriever (or any retriever) puts
that article in the top-k.

Workflow:

  1. Build the labeling template — one row per gold question, with
     retriever-suggested candidate titles to choose from:

        from polimibot.eval import GoldSet
        from polimibot.eval.retrieval import build_labeling_template, save_retrieval_gold

        gold = GoldSet.load(PATHS.eval_dir / 'gold_set.jsonl')
        items = build_labeling_template(gold, retriever=retriever, k_candidates=5)
        save_retrieval_gold(items, PATHS.eval_dir / 'retrieval_gold.jsonl')

  2. Open the JSONL, fill in ``gold_article_title`` for each item by hand.
     A null value means "no article suffices" and is skipped in recall.

  3. Measure:

        from polimibot.eval.retrieval import load_retrieval_gold, evaluate_retrieval

        items = load_retrieval_gold(PATHS.eval_dir / 'retrieval_gold.jsonl')
        report = evaluate_retrieval(retriever, items, ks=(1, 3, 5, 10))
        report.print_summary()

The harness accepts a custom ``query_fn`` so the same labeled set serves
multiple retrieval recipes (question-only, question+options, HyDE, …).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from ..config import Category
from ..logging_utils import load_jsonl
from .gold_set import GoldItem, GoldSet


# ── Domain records ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalGoldItem:
    """One question + the Wikipedia article title that should be retrieved.

    ``gold_article_title=None`` means unlabeled — skipped in recall scores.
    ``candidates`` is a convenience field populated by
    ``build_labeling_template`` so the labeler can pick from a short list
    rather than typing titles from memory.
    """
    question_text: str
    options: tuple[str, ...]
    correct_index: int
    competition_id: int
    level: int
    category: Optional[Category] = None
    gold_article_title: Optional[str] = None
    candidates: tuple[str, ...] = ()

    @property
    def is_labeled(self) -> bool:
        return self.gold_article_title is not None


@dataclass
class RetrievalSample:
    """Per-question outcome for error analysis."""
    question_text: str
    gold_article_title: str
    retrieved_titles: list[str]
    retrieved_scores: list[float]
    found_at_rank: Optional[int]   # 1-indexed; None = not in top-k_retrieve
    category: Optional[str]
    level: int


@dataclass
class RetrievalReport:
    """Aggregate retrieval metrics."""
    retriever_name: str
    n_total: int                            # all items, labeled or not
    n_labeled: int                          # contributes to recall
    n_unlabeled_skipped: int                # passed through but ignored
    ks: tuple[int, ...]
    recall_at: dict[int, float]             # k -> recall ∈ [0, 1]
    mrr: float                              # Mean Reciprocal Rank
    by_category: dict[str, dict[int, float]] = field(default_factory=dict)
    samples: list[RetrievalSample] = field(default_factory=list, repr=False)

    def print_summary(self) -> None:
        print(f"\n{'─' * 55}")
        print(f"  Retriever : {self.retriever_name}")
        print(f"  Labeled   : {self.n_labeled} / {self.n_total} "
              f"({self.n_unlabeled_skipped} unlabeled, skipped)")
        print(f"  MRR       : {self.mrr:.4f}")
        for k in sorted(self.ks):
            print(f"  recall@{k:<2} : {self.recall_at[k]:.1%}")
        if self.by_category:
            print(f"\n  Per-category recall@{max(self.ks)}:")
            for cat in sorted(self.by_category):
                v = self.by_category[cat].get(max(self.ks), 0.0)
                print(f"    {cat:<16} {v:.1%}")
        print(f"{'─' * 55}\n")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("samples")
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False))


# ── Persistence ──────────────────────────────────────────────────────────


def save_retrieval_gold(items: Iterable[RetrievalGoldItem], path: Path) -> None:
    """Write the labeling set to a JSONL file. Overwrites existing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            d = asdict(item)
            if d["category"] is not None:
                d["category"] = d["category"].value  # enum → string
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved retrieval gold → {path}")


def load_retrieval_gold(path: Path) -> list[RetrievalGoldItem]:
    """Load a labeling-set JSONL."""
    items = []
    for rec in load_jsonl(path):
        cat = rec.get("category")
        items.append(RetrievalGoldItem(
            question_text=rec["question_text"],
            options=tuple(rec["options"]),
            correct_index=rec["correct_index"],
            competition_id=rec["competition_id"],
            level=rec["level"],
            category=Category(cat) if cat else None,
            gold_article_title=rec.get("gold_article_title"),
            candidates=tuple(rec.get("candidates", ())),
        ))
    return items


# ── Template builder ─────────────────────────────────────────────────────


def build_labeling_template(
    gold_set: GoldSet,
    *,
    retriever: Any = None,
    k_candidates: int = 5,
) -> list[RetrievalGoldItem]:
    """Turn a GoldSet into a labeling-stub list of RetrievalGoldItem.

    Each item starts with ``gold_article_title=None`` — the user fills it
    in by editing the JSONL. If ``retriever`` is provided, the top
    ``k_candidates`` retrieved article titles are stored under
    ``candidates`` so the labeler can pick from a short list rather than
    typing the title from memory.

    The query used for candidate suggestions mirrors RAGStrategy's
    current convention: ``question + " ".join(options)``. The labeler
    can override per item.
    """
    out: list[RetrievalGoldItem] = []
    for g in gold_set:
        candidates: tuple[str, ...] = ()
        if retriever is not None:
            query = f"{g.question_text} {' '.join(g.options)}"
            cat = g.category.value if g.category is not None else None
            try:
                # Prefer category-filtered candidates if the retriever supports it.
                try:
                    hits = retriever.retrieve(query, k=k_candidates, category=cat)
                except TypeError:
                    hits = retriever.retrieve(query, k=k_candidates)
            except Exception:
                hits = []
            # Dedup while preserving rank order.
            seen: set[str] = set()
            titles: list[str] = []
            for chunk, _score in hits:
                if chunk.source not in seen:
                    seen.add(chunk.source)
                    titles.append(chunk.source)
            candidates = tuple(titles)

        out.append(RetrievalGoldItem(
            question_text=g.question_text,
            options=g.options,
            correct_index=g.correct_index,
            competition_id=g.competition_id,
            level=g.level,
            category=g.category,
            gold_article_title=None,
            candidates=candidates,
        ))
    return out


# ── Harness ──────────────────────────────────────────────────────────────


def _default_query(item: RetrievalGoldItem) -> str:
    """Mirror RAGStrategy's current query construction."""
    return f"{item.question_text} {' '.join(item.options)}"


def evaluate_retrieval(
    retriever: Any,
    items: Sequence[RetrievalGoldItem],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    query_fn: Callable[[RetrievalGoldItem], str] = _default_query,
    retriever_name: str = "retriever",
    k_retrieve: Optional[int] = None,
    use_category_filter: bool = True,
    use_reranker: bool = False,
    use_hybrid: bool = False,
) -> RetrievalReport:
    """Compute recall@k and MRR for ``retriever`` against ``items``.

    Args:
        retriever: anything with
            ``.retrieve(query, k, *, category=None) -> list[(Chunk, score)]``.
            Older retrievers that don't accept ``category`` are still
            supported when ``use_category_filter=False``.
        items: labeling set. Items with ``gold_article_title=None`` are
            counted in ``n_unlabeled_skipped`` and excluded from scores.
        ks: which recall thresholds to report.
        query_fn: how to build the retrieval query from each item. Lets you
            run the same labeled set against different query recipes
            (question-only, multi-query, HyDE, …).
        k_retrieve: how many results to ask the retriever for. Defaults to
            ``max(ks)`` so every threshold is computable from one call.
        use_category_filter: when True, pass the item's category to the
            retriever. Set False to ablate the filter and see the raw
            uncategorised retrieval performance.
        use_reranker: when True, ask the retriever to rerank its
            oversearched pool with its attached cross-encoder. Same
            ablation pattern as ``use_category_filter`` — keep the
            labels fixed, toggle the recipe.
        use_hybrid: when True, ask the retriever to RRF-fuse dense +
            BM25 results. Same ablation pattern.
    """
    if not items:
        return RetrievalReport(
            retriever_name=retriever_name,
            n_total=0, n_labeled=0, n_unlabeled_skipped=0,
            ks=tuple(sorted(ks)), recall_at={k: 0.0 for k in ks}, mrr=0.0,
        )

    ks_sorted = tuple(sorted(set(ks)))
    if k_retrieve is None:
        k_retrieve = max(ks_sorted)

    samples: list[RetrievalSample] = []
    n_unlabeled = 0
    for item in items:
        if not item.is_labeled:
            n_unlabeled += 1
            continue

        query = query_fn(item)
        category = (
            item.category.value
            if (use_category_filter and item.category is not None)
            else None
        )
        kw: dict = {"k": k_retrieve, "category": category}
        if use_reranker:
            kw["rerank"] = True
        if use_hybrid:
            kw["hybrid"] = True
        try:
            hits = retriever.retrieve(query, **kw) or []
        except TypeError:
            # Legacy retrievers without the newer kwargs — fall back to
            # the lowest-common-denominator signature so the harness
            # doesn't crash on older mocks.
            try:
                hits = retriever.retrieve(query, k=k_retrieve, category=category) or []
            except TypeError:
                hits = retriever.retrieve(query, k=k_retrieve) or []

        # Unique titles in rank order (a single article often produces
        # multiple chunks; for recall, the first chunk hit defines the rank).
        seen: set[str] = set()
        unique_titles: list[str] = []
        unique_scores: list[float] = []
        for chunk, score in hits:
            if chunk.source not in seen:
                seen.add(chunk.source)
                unique_titles.append(chunk.source)
                unique_scores.append(float(score))

        found_at: Optional[int] = None
        gold_norm = (item.gold_article_title or "").strip()
        for rank, title in enumerate(unique_titles, start=1):
            if title.strip() == gold_norm:
                found_at = rank
                break

        samples.append(RetrievalSample(
            question_text=item.question_text,
            gold_article_title=item.gold_article_title or "",
            retrieved_titles=unique_titles,
            retrieved_scores=unique_scores,
            found_at_rank=found_at,
            category=item.category.value if item.category else None,
            level=item.level,
        ))

    n_labeled = len(samples)
    if n_labeled == 0:
        return RetrievalReport(
            retriever_name=retriever_name,
            n_total=len(items), n_labeled=0,
            n_unlabeled_skipped=n_unlabeled,
            ks=ks_sorted,
            recall_at={k: 0.0 for k in ks_sorted}, mrr=0.0,
        )

    recall_at: dict[int, float] = {}
    for k in ks_sorted:
        hits_at_k = sum(
            1 for s in samples
            if s.found_at_rank is not None and s.found_at_rank <= k
        )
        recall_at[k] = hits_at_k / n_labeled

    mrr = sum(
        (1.0 / s.found_at_rank) if s.found_at_rank else 0.0
        for s in samples
    ) / n_labeled

    # Per-category recall (one row per category).
    by_cat_samples: dict[str, list[RetrievalSample]] = defaultdict(list)
    for s in samples:
        by_cat_samples[s.category or "unknown"].append(s)
    by_category: dict[str, dict[int, float]] = {}
    for cat, cat_samples in by_cat_samples.items():
        by_category[cat] = {
            k: sum(
                1 for s in cat_samples
                if s.found_at_rank is not None and s.found_at_rank <= k
            ) / len(cat_samples)
            for k in ks_sorted
        }

    return RetrievalReport(
        retriever_name=retriever_name,
        n_total=len(items),
        n_labeled=n_labeled,
        n_unlabeled_skipped=n_unlabeled,
        ks=ks_sorted,
        recall_at=recall_at,
        mrr=round(mrr, 4),
        by_category=by_category,
        samples=samples,
    )


# ── Post-hoc from run logs ───────────────────────────────────────────────


def recall_from_runs(
    run_path: Path,
    items: Sequence[RetrievalGoldItem],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    retriever_name: str = "from_runs",
) -> RetrievalReport:
    """Compute recall@k from already-logged retrieval triples in a run JSONL.

    Reads ``extras['passages']`` (the list of {source, chunk_id, score}
    triples that RAGStrategy now logs through the runner) and matches
    each question against the labeled set by ``question_text``. Lets you
    re-score historical runs against new labels without re-running the
    LLM or the retriever.
    """
    # Index labeled items by question_text for O(1) lookup.
    by_text: dict[str, RetrievalGoldItem] = {
        it.question_text: it for it in items if it.is_labeled
    }
    if not by_text:
        return RetrievalReport(
            retriever_name=retriever_name,
            n_total=len(items), n_labeled=0,
            n_unlabeled_skipped=len(items),
            ks=tuple(sorted(ks)),
            recall_at={k: 0.0 for k in ks}, mrr=0.0,
        )

    ks_sorted = tuple(sorted(set(ks)))
    samples: list[RetrievalSample] = []
    for rec in load_jsonl(run_path):
        if rec.get("run_kind") != "question":
            continue
        qtext = rec.get("question_text", "")
        item = by_text.get(qtext)
        if item is None:
            continue
        passages = (rec.get("extras", {}) or {}).get("passages", [])
        if not passages:
            continue

        # Same rank-unique logic as evaluate_retrieval.
        seen: set[str] = set()
        unique_titles: list[str] = []
        unique_scores: list[float] = []
        for p in passages:
            src = p.get("source", "")
            if src and src not in seen:
                seen.add(src)
                unique_titles.append(src)
                unique_scores.append(float(p.get("score", 0.0)))

        gold_norm = (item.gold_article_title or "").strip()
        found_at: Optional[int] = None
        for rank, title in enumerate(unique_titles, start=1):
            if title.strip() == gold_norm:
                found_at = rank
                break

        samples.append(RetrievalSample(
            question_text=qtext,
            gold_article_title=item.gold_article_title or "",
            retrieved_titles=unique_titles,
            retrieved_scores=unique_scores,
            found_at_rank=found_at,
            category=item.category.value if item.category else None,
            level=item.level,
        ))

    if not samples:
        return RetrievalReport(
            retriever_name=retriever_name,
            n_total=len(items), n_labeled=0,
            n_unlabeled_skipped=len(items),
            ks=ks_sorted,
            recall_at={k: 0.0 for k in ks_sorted}, mrr=0.0,
        )

    n_labeled = len(samples)
    recall_at = {
        k: sum(1 for s in samples
               if s.found_at_rank is not None and s.found_at_rank <= k) / n_labeled
        for k in ks_sorted
    }
    mrr = sum(
        (1.0 / s.found_at_rank) if s.found_at_rank else 0.0
        for s in samples
    ) / n_labeled

    by_cat_samples: dict[str, list[RetrievalSample]] = defaultdict(list)
    for s in samples:
        by_cat_samples[s.category or "unknown"].append(s)
    by_category: dict[str, dict[int, float]] = {
        cat: {
            k: sum(1 for s in cs
                   if s.found_at_rank is not None and s.found_at_rank <= k) / len(cs)
            for k in ks_sorted
        }
        for cat, cs in by_cat_samples.items()
    }

    return RetrievalReport(
        retriever_name=retriever_name,
        n_total=len(items),
        n_labeled=n_labeled,
        n_unlabeled_skipped=len(items) - n_labeled,
        ks=ks_sorted,
        recall_at=recall_at,
        mrr=round(mrr, 4),
        by_category=by_category,
        samples=samples,
    )
