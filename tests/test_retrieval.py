"""Retrieval-only metric tests. No GPU, no network, no FAISS."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from polimibot.config import Category
from polimibot.eval.gold_set import GoldItem, GoldSet
from polimibot.eval.retrieval import (
    RetrievalGoldItem,
    build_labeling_template,
    evaluate_retrieval,
    evaluate_retrieval_multi_query,
    load_retrieval_gold,
    recall_from_runs,
    save_retrieval_gold,
)
from polimibot.rag.chunker import Chunk


# ── Mocks ────────────────────────────────────────────────────────────────


class _FakeRetriever:
    """Returns a scripted list of (Chunk, score) per query, in declared order."""

    def __init__(self, hits_by_query: dict[str, list[tuple[str, float]]]):
        self._hits = hits_by_query

    def retrieve(self, query: str, k: int = 5, *, category=None):
        # category accepted for parity with Retriever.retrieve; the mock
        # ignores it (per-test scripts decide what to return).
        triples = self._hits.get(query, [])[:k]
        return [
            (Chunk(text="…", source=src, chunk_id=i, category=category), score)
            for i, (src, score) in enumerate(triples)
        ]


def _gold(idx: int, *, cat=Category.HISTORY, level=3, title="Article") -> RetrievalGoldItem:
    return RetrievalGoldItem(
        question_text=f"Q{idx}",
        options=("a", "b", "c", "d"),
        correct_index=0,
        competition_id=1,
        level=level,
        category=cat,
        gold_article_title=title,
        candidates=(),
    )


def _query(item: RetrievalGoldItem) -> str:
    # Tests use question_text alone so the mock dispatch is easy to script.
    return item.question_text


# ── Recall@k semantics ───────────────────────────────────────────────────


def test_recall_at_k_when_gold_is_rank_1():
    items = [_gold(0, title="A")]
    retriever = _FakeRetriever({"Q0": [("A", 0.9), ("B", 0.5)]})
    report = evaluate_retrieval(retriever, items, ks=(1, 3), query_fn=_query)
    assert report.recall_at[1] == 1.0
    assert report.recall_at[3] == 1.0
    assert report.mrr == 1.0


def test_recall_at_k_when_gold_is_rank_3():
    items = [_gold(0, title="C")]
    retriever = _FakeRetriever({"Q0": [("A", 0.9), ("B", 0.7), ("C", 0.4), ("D", 0.1)]})
    report = evaluate_retrieval(retriever, items, ks=(1, 3, 5), query_fn=_query)
    assert report.recall_at[1] == 0.0
    assert report.recall_at[3] == 1.0
    assert report.recall_at[5] == 1.0
    # MRR is rounded to 4 dp in the report for clean JSON; allow slack.
    assert report.mrr == pytest.approx(1 / 3, abs=1e-3)


def test_recall_at_k_when_gold_missing_entirely():
    items = [_gold(0, title="Z")]
    retriever = _FakeRetriever({"Q0": [("A", 0.9), ("B", 0.7), ("C", 0.4)]})
    report = evaluate_retrieval(retriever, items, ks=(1, 3, 5), query_fn=_query)
    for k in (1, 3, 5):
        assert report.recall_at[k] == 0.0
    assert report.mrr == 0.0


def test_recall_averaged_over_items():
    # Two questions: one hit at rank 1, one miss.
    items = [_gold(0, title="A"), _gold(1, title="X")]
    retriever = _FakeRetriever({
        "Q0": [("A", 0.9), ("B", 0.5)],
        "Q1": [("B", 0.9), ("C", 0.5)],
    })
    report = evaluate_retrieval(retriever, items, ks=(1, 3), query_fn=_query)
    assert report.recall_at[1] == 0.5
    assert report.recall_at[3] == 0.5
    assert report.mrr == pytest.approx(0.5)


def test_unlabeled_items_skipped():
    """gold_article_title=None means 'no article suffices' — exclude from scores."""
    items = [
        _gold(0, title="A"),                               # labeled, hit
        RetrievalGoldItem(                                  # unlabeled
            question_text="Q1", options=("a","b","c","d"),
            correct_index=0, competition_id=1, level=3,
            category=Category.HISTORY, gold_article_title=None,
        ),
    ]
    retriever = _FakeRetriever({"Q0": [("A", 1.0)]})
    report = evaluate_retrieval(retriever, items, ks=(1,), query_fn=_query)
    assert report.n_total == 2
    assert report.n_labeled == 1
    assert report.n_unlabeled_skipped == 1
    assert report.recall_at[1] == 1.0   # not 0.5 — the unlabeled item is ignored, not counted as a miss


def test_duplicate_chunks_from_same_article_count_as_rank_1():
    """A long article produces multiple chunks; recall should treat the
    first-appearance rank as the article's rank, not penalise every chunk
    individually."""
    items = [_gold(0, title="Newton")]
    retriever = _FakeRetriever({
        "Q0": [("Newton", 0.9), ("Newton", 0.88), ("Newton", 0.85), ("Einstein", 0.4)],
    })
    report = evaluate_retrieval(retriever, items, ks=(1,), query_fn=_query)
    assert report.recall_at[1] == 1.0


# ── Per-category breakdown ───────────────────────────────────────────────


def test_per_category_recall_separates_categories():
    items = [
        _gold(0, cat=Category.HISTORY, title="A"),     # hit
        _gold(1, cat=Category.HISTORY, title="Z"),     # miss
        _gold(2, cat=Category.SCIENCE, title="C"),     # hit
    ]
    retriever = _FakeRetriever({
        "Q0": [("A", 1.0)],
        "Q1": [("B", 1.0)],
        "Q2": [("C", 1.0)],
    })
    report = evaluate_retrieval(retriever, items, ks=(1,), query_fn=_query)
    assert report.by_category["history"][1] == 0.5
    assert report.by_category["science"][1] == 1.0


# ── Save / load roundtrip ────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path):
    items = [
        _gold(0, title="Newton"),
        RetrievalGoldItem(                                  # unlabeled with candidates
            question_text="Q1", options=("a","b","c","d"),
            correct_index=2, competition_id=2, level=7,
            category=Category.SCIENCE, gold_article_title=None,
            candidates=("DNA", "Cell (biology)", "Photosynthesis"),
        ),
    ]
    path = tmp_path / "retrieval_gold.jsonl"
    save_retrieval_gold(items, path)
    loaded = load_retrieval_gold(path)
    assert len(loaded) == 2
    assert loaded[0].gold_article_title == "Newton"
    assert loaded[0].category == Category.HISTORY
    assert loaded[1].gold_article_title is None
    assert loaded[1].candidates == ("DNA", "Cell (biology)", "Photosynthesis")


# ── Labeling template builder ────────────────────────────────────────────


def test_build_labeling_template_without_retriever():
    gold = GoldSet([GoldItem(
        question_text="Q0", options=("a","b","c","d"),
        correct_index=0, competition_id=1, level=3,
        category=Category.HISTORY, source_run="x",
    )])
    items = build_labeling_template(gold)
    assert len(items) == 1
    assert items[0].gold_article_title is None      # unlabeled
    assert items[0].candidates == ()                # no retriever → no candidates
    assert items[0].category == Category.HISTORY


def test_build_labeling_template_with_retriever_populates_candidates():
    gold = GoldSet([GoldItem(
        question_text="Who crossed the Rubicon?",
        options=("Pompey", "Caesar", "Augustus", "Cicero"),
        correct_index=1, competition_id=1, level=3,
        category=Category.HISTORY, source_run="x",
    )])
    retriever = _FakeRetriever({
        # build_labeling_template uses question + " ".join(options) as query.
        "Who crossed the Rubicon? Pompey Caesar Augustus Cicero": [
            ("Julius Caesar", 0.9),
            ("Roman Republic", 0.7),
            ("Pompey", 0.5),
        ],
    })
    items = build_labeling_template(gold, retriever=retriever, k_candidates=3)
    assert items[0].candidates == ("Julius Caesar", "Roman Republic", "Pompey")


def test_build_labeling_template_dedupes_candidates():
    """Multiple chunks from the same article should produce one candidate title."""
    gold = GoldSet([GoldItem(
        question_text="Q0", options=("a","b","c","d"),
        correct_index=0, competition_id=1, level=3,
        category=Category.HISTORY, source_run="x",
    )])
    retriever = _FakeRetriever({
        "Q0 a b c d": [("Newton", 0.9), ("Newton", 0.85), ("Einstein", 0.7), ("Newton", 0.6)],
    })
    items = build_labeling_template(gold, retriever=retriever, k_candidates=5)
    assert items[0].candidates == ("Newton", "Einstein")


# ── Post-hoc from run logs ───────────────────────────────────────────────


def _write_run(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_recall_from_runs_uses_logged_passages(tmp_path: Path):
    """The runner now propagates RAGStrategy's full top-k into extras.passages.
    recall_from_runs should be able to recompute recall@k from that alone."""
    run_path = tmp_path / "run.jsonl"
    _write_run(run_path, [
        {"run_kind": "manifest"},
        {
            "run_kind": "question",
            "question_text": "Q0",
            "extras": {
                "passages": [
                    {"source": "Wrong", "chunk_id": 0, "score": 0.9},
                    {"source": "Newton", "chunk_id": 1, "score": 0.6},
                ],
            },
        },
    ])
    labeled = [_gold(0, title="Newton")]
    report = recall_from_runs(run_path, labeled, ks=(1, 3))
    assert report.recall_at[1] == 0.0      # rank 1 was Wrong
    assert report.recall_at[3] == 1.0      # rank 2 was Newton — under top-3
    assert report.mrr == pytest.approx(0.5)


def test_recall_from_runs_skips_questions_without_labels(tmp_path: Path):
    run_path = tmp_path / "run.jsonl"
    _write_run(run_path, [
        {"run_kind": "question", "question_text": "Q0",
         "extras": {"passages": [{"source": "A", "chunk_id": 0, "score": 1.0}]}},
        {"run_kind": "question", "question_text": "Q_unlabeled",
         "extras": {"passages": [{"source": "X", "chunk_id": 0, "score": 1.0}]}},
    ])
    labeled = [_gold(0, title="A")]   # only Q0 is in the labeled set
    report = recall_from_runs(run_path, labeled, ks=(1,))
    assert report.n_labeled == 1
    assert report.recall_at[1] == 1.0


# ── Report serialisation ─────────────────────────────────────────────────


def test_evaluate_retrieval_forwards_rerank_flag():
    """use_reranker=True must reach the retriever as rerank=True."""
    captured: dict = {}

    class _Captor:
        def retrieve(self, query, k=5, *, category=None, rerank=False, **_):
            captured["rerank"] = rerank
            captured["category"] = category
            return [(Chunk(text="t", source="X", chunk_id=0), 1.0)]

    items = [_gold(0, title="X")]
    evaluate_retrieval(_Captor(), items, ks=(1,), use_reranker=True,
                       query_fn=_query)
    assert captured["rerank"] is True


def test_evaluate_retrieval_default_rerank_false():
    captured: dict = {}

    class _Captor:
        def retrieve(self, query, k=5, *, category=None, rerank=False, **_):
            captured["rerank"] = rerank
            return [(Chunk(text="t", source="X", chunk_id=0), 1.0)]

    items = [_gold(0, title="X")]
    evaluate_retrieval(_Captor(), items, ks=(1,), query_fn=_query)
    assert captured["rerank"] is False


def test_evaluate_retrieval_tolerates_legacy_retriever_without_rerank_kwarg():
    """Older retrievers that only accept (query, k, *, category=None) shouldn't
    crash the harness when use_reranker=True — fall back to dense."""

    class _LegacyRetriever:
        def retrieve(self, query, k=5, *, category=None):
            return [(Chunk(text="t", source="X", chunk_id=0), 1.0)]

    items = [_gold(0, title="X")]
    report = evaluate_retrieval(_LegacyRetriever(), items, ks=(1,),
                                 use_reranker=True, query_fn=_query)
    # No crash — hit at rank 1.
    assert report.recall_at[1] == 1.0


def test_report_save_omits_samples(tmp_path: Path):
    items = [_gold(0, title="A")]
    retriever = _FakeRetriever({"Q0": [("A", 1.0)]})
    report = evaluate_retrieval(retriever, items, ks=(1,), query_fn=_query)
    path = tmp_path / "report.json"
    report.save(path)
    d = json.loads(path.read_text())
    assert "recall_at" in d
    assert "samples" not in d   # too large for the report file


# ── Multi-query harness (mirrors RAGStrategy's runtime recipe) ────────────


def _mq_gold(idx: int, options: tuple[str, ...], *, title: str,
             cat=Category.HISTORY) -> RetrievalGoldItem:
    return RetrievalGoldItem(
        question_text=f"Q{idx}",
        options=options,
        correct_index=0,
        competition_id=1,
        level=3,
        category=cat,
        gold_article_title=title,
        candidates=(),
    )


def test_multi_query_fans_out_one_call_per_query():
    """RAGStrategy issues [question] + [question+opt for opt in options]
    — 5 calls for a 4-option MCQ. Harness must do the same."""
    calls: list[str] = []

    class _Counter:
        def retrieve(self, query, k=5, *, category=None, **_):
            calls.append(query)
            return [(Chunk(text="…", source="A", chunk_id=0), 1.0)]

    items = [_mq_gold(0, options=("opt_a", "opt_b", "opt_c", "opt_d"), title="A")]
    evaluate_retrieval_multi_query(_Counter(), items, ks=(1,))
    assert calls == [
        "Q0",
        "Q0 opt_a", "Q0 opt_b", "Q0 opt_c", "Q0 opt_d",
    ]


def test_multi_query_rrf_fusion_picks_consensus_winner():
    """When every per-query list ranks article X near the top, RRF should
    push X to rank 1 even if no single query ranked it first."""
    # X is rank-2 in every list — but rank-1 differs across lists, so the
    # rank-1 entries split votes. X wins by consensus.
    per_query = {
        "Q0":         [("Y1", 0.9), ("X", 0.8)],
        "Q0 a":       [("Y2", 0.9), ("X", 0.8)],
        "Q0 b":       [("Y3", 0.9), ("X", 0.8)],
        "Q0 c":       [("Y4", 0.9), ("X", 0.8)],
        "Q0 d":       [("Y5", 0.9), ("X", 0.8)],
    }

    class _ByQuery:
        def retrieve(self, query, k=5, *, category=None, **_):
            return [
                (Chunk(text="…", source=src, chunk_id=i), score)
                for i, (src, score) in enumerate(per_query.get(query, []))
            ]

    items = [_mq_gold(0, options=("a", "b", "c", "d"), title="X")]
    report = evaluate_retrieval_multi_query(_ByQuery(), items, ks=(1, 3))
    assert report.recall_at[1] == 1.0
    assert report.mrr == 1.0


def test_multi_query_passes_hybrid_through_per_call():
    captured: list[dict] = []

    class _Captor:
        def retrieve(self, query, k=5, *, category=None, **kw):
            captured.append({"query": query, "category": category, **kw})
            return [(Chunk(text="…", source="A", chunk_id=0), 1.0)]

    items = [_mq_gold(0, options=("a", "b", "c", "d"), title="A")]
    evaluate_retrieval_multi_query(_Captor(), items, ks=(1,),
                                    use_hybrid=True,
                                    use_category_filter=True)
    # Every per-query call gets hybrid=True and category=history (the item's).
    for call in captured:
        assert call.get("hybrid") is True
        assert call["category"] == "history"
    # And the per-query call must NOT pass rerank=True — rerank fires once
    # over the fused pool, not per query (matches RAGStrategy).
    for call in captured:
        assert call.get("rerank") is None or call.get("rerank") is False


def test_multi_query_reranks_fused_pool_with_question_as_anchor():
    """When use_reranker=True, the harness calls retriever.rerank_pool ONCE,
    with the item's question_text (not a fanned-out query)."""
    rerank_calls: list[str] = []

    class _ReranklingRetriever:
        def retrieve(self, query, k=5, *, category=None, **_):
            return [(Chunk(text="…", source="A", chunk_id=0), 0.5)]

        def rerank_pool(self, query, pool, *, k):
            rerank_calls.append(query)
            # Trivial pass-through; relevance not the point of this test.
            return list(pool)[:k]

    items = [_mq_gold(0, options=("a", "b", "c", "d"), title="A")]
    evaluate_retrieval_multi_query(
        _ReranklingRetriever(), items, ks=(1,),
        use_reranker=True, rerank_oversearch=2,
    )
    assert rerank_calls == ["Q0"]   # exactly one rerank, anchored on the question


def test_multi_query_oversearch_widens_per_query_pool():
    """rerank_oversearch=N → each per-query call asks for N*k_retrieve hits.
    Without reranker, per-query k stays at k_retrieve."""
    captured_ks: list[int] = []

    class _Captor:
        def retrieve(self, query, k=5, *, category=None, **_):
            captured_ks.append(k)
            return [(Chunk(text="…", source="A", chunk_id=0), 1.0)]

        def rerank_pool(self, query, pool, *, k):
            return list(pool)[:k]

    items = [_mq_gold(0, options=("a", "b", "c", "d"), title="A")]

    captured_ks.clear()
    evaluate_retrieval_multi_query(_Captor(), items, ks=(1, 3),
                                    use_reranker=True, rerank_oversearch=4)
    # max(ks)=3, oversearch=4 → per-query k must be 12.
    assert set(captured_ks) == {12}

    captured_ks.clear()
    evaluate_retrieval_multi_query(_Captor(), items, ks=(1, 3),
                                    use_reranker=False, rerank_oversearch=4)
    # Without reranker, oversearch has no effect — per-query k stays at 3.
    assert set(captured_ks) == {3}


def test_multi_query_unlabeled_items_skipped():
    items = [
        _mq_gold(0, options=("a", "b", "c", "d"), title="A"),
        RetrievalGoldItem(
            question_text="Q1", options=("a","b","c","d"),
            correct_index=0, competition_id=1, level=3,
            category=Category.HISTORY, gold_article_title=None,
        ),
    ]

    class _Hit:
        def retrieve(self, query, k=5, *, category=None, **_):
            return [(Chunk(text="…", source="A", chunk_id=0), 1.0)]

    report = evaluate_retrieval_multi_query(_Hit(), items, ks=(1,))
    assert report.n_total == 2
    assert report.n_labeled == 1
    assert report.n_unlabeled_skipped == 1
    assert report.recall_at[1] == 1.0
