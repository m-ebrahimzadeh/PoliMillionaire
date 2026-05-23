"""Public retrieval interface. Strategies call this; they never touch FAISS directly."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from .bm25 import BM25Index
from .chunker import CHUNKER_VERSION, Chunk
from .embedder import Embedder, EmbedderSpec
from .fusion import reciprocal_rank_fusion
from .index import FAISSIndex
from .reranker import CrossEncoderReranker


def _diversify_by_source(
    items: list[tuple[Chunk, float]],
    *,
    k: int,
) -> list[tuple[Chunk, float]]:
    """Source-aware top-k selection.

    Pass 1: keep the highest-scored chunk per source until ``k`` slots
    fill or the pool runs out. Pass 2: if pass 1 was short of ``k``,
    backfill from the remaining chunks (those from already-seen sources)
    in their original order.

    The pool is assumed to arrive already ordered by relevance. The
    result preserves that order within each pass; it just collapses
    runs of same-source chunks down to one until later slots are needed.
    """
    if k <= 0 or not items:
        return []
    seen: set[str] = set()
    primary: list[tuple[Chunk, float]] = []
    leftover: list[tuple[Chunk, float]] = []
    for chunk, score in items:
        if chunk.source not in seen:
            seen.add(chunk.source)
            primary.append((chunk, score))
        else:
            leftover.append((chunk, score))
        if len(primary) >= k:
            break
    out = primary[:k]
    if len(out) < k:
        out = out + leftover[:k - len(out)]
    return out


def _check_manifest_compat(manifest: dict, spec: EmbedderSpec) -> None:
    """Refuse to load an index built with an incompatible embedder.

    Hard-fails on model_name or dim mismatch (those silently corrupt
    scores). Warns on normalize / chunker_version drift (less catastrophic
    but worth surfacing — chunk text changed but vectors still align).
    """
    expected = manifest.get("embedder_model_name")
    if expected and expected != spec.model_name:
        raise ValueError(
            f"Index was built with embedder '{expected}', but you're "
            f"querying with '{spec.model_name}'. Vectors live in "
            f"incompatible spaces. Rebuild the index, or pass the matching "
            f"EmbedderSpec(model_name={expected!r})."
        )
    expected_norm = manifest.get("normalize")
    if expected_norm is not None and expected_norm != spec.normalize:
        warnings.warn(
            f"Index was built with normalize={expected_norm}, but the "
            f"current EmbedderSpec has normalize={spec.normalize}. "
            f"Scores will be inconsistent.",
            RuntimeWarning,
            stacklevel=3,
        )
    # Prefix drift silently corrupts scores on asymmetric models (BGE/E5):
    # the query and passage vectors land in mismatched halves of the space.
    # Hard-fail like model_name. Absent fields in legacy manifests are
    # treated as "no prefix" so older indices keep loading.
    for field_name in ("query_prefix", "passage_prefix"):
        indexed_prefix = manifest.get(f"embedder_{field_name}")
        if indexed_prefix is None:
            continue
        spec_prefix = getattr(spec, field_name)
        if indexed_prefix != spec_prefix:
            raise ValueError(
                f"Index was built with {field_name}={indexed_prefix!r}, but "
                f"the current EmbedderSpec has {field_name}={spec_prefix!r}. "
                f"Asymmetric-model prefixes must match between index build "
                f"and query time — vectors live in incompatible halves of "
                f"the embedding space otherwise. Rebuild the index, or pass "
                f"the matching EmbedderSpec."
            )
    indexed_chunker = manifest.get("chunker_version")
    if indexed_chunker is not None and indexed_chunker != CHUNKER_VERSION:
        warnings.warn(
            f"Index was built with chunker_version={indexed_chunker}, but "
            f"the current chunker is version {CHUNKER_VERSION}. The chunk "
            f"text shape may differ from what the embeddings encode — "
            f"rebuild the index for best retrieval quality.",
            RuntimeWarning,
            stacklevel=3,
        )
    indexed_corpus = manifest.get("corpus_version")
    # Imported lazily to avoid pulling the wikipedia dependency into the
    # retriever import graph; corpus_version is just a constant int.
    if indexed_corpus is not None:
        from .corpus import CORPUS_VERSION
        if indexed_corpus != CORPUS_VERSION:
            warnings.warn(
                f"Index was built with corpus_version={indexed_corpus}, but "
                f"the current corpus seeds/disambiguation policy is at "
                f"version {CORPUS_VERSION}. Article selection may differ "
                f"from what's indexed — rebuild for the freshest coverage.",
                RuntimeWarning,
                stacklevel=3,
            )


class Retriever:
    """Given a query string, return the k most relevant chunks.

    Two construction paths:
      - Build from scratch: Retriever(index, embedder)
      - Load pre-built:     Retriever.from_saved(path)

    The same Embedder instance should be used for both indexing and querying —
    querying with a different model produces garbage results (vectors live in
    incompatible spaces).
    """

    def __init__(
        self,
        index: FAISSIndex,
        embedder: Embedder,
        *,
        reranker: Optional[CrossEncoderReranker] = None,
        bm25: Optional[BM25Index] = None,
        rrf_weights: Optional[tuple[float, float]] = None,
    ) -> None:
        if index.dim != embedder.dim:
            raise ValueError(
                f"Index dim={index.dim} != embedder dim={embedder.dim}. "
                "Must use the same model for indexing and querying."
            )
        self._index = index
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = bm25
        # Optional (dense_w, bm25_w) weighting for RRF fusion in hybrid
        # mode. None ⇒ classic 1:1 symmetric fusion. Bump bm25_w for
        # entity-heavy queries; bump dense_w for conceptual ones.
        self._rrf_weights = rrf_weights

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    @property
    def has_bm25(self) -> bool:
        return self._bm25 is not None

    # Oversearch factors. When a category filter or a reranker is in play
    # we ask the index for more chunks than the caller wants, then trim
    # down. Pure dense IndexFlatIP doesn't support an in-FAISS ID mask
    # cleanly across versions, and Python filtering on a small index is
    # cheap. The default rerank oversearch matches the cross-encoder
    # literature's "retrieve 5× more, rerank to k".
    _CATEGORY_OVERSEARCH = 8
    _DEFAULT_RERANK_OVERSEARCH = 5
    # Diversify needs a wider pool than k or it has nothing to swap in.
    # 2× k is enough to dedupe overlapping windows from the same article
    # without paying for a full 5× rerank-style oversearch.
    _DIVERSIFY_OVERSEARCH = 2

    def retrieve(
        self,
        query: str,
        k: int = 3,
        *,
        bm25_query: Optional[str] = None,
        category: Optional[str] = None,
        hybrid: bool = False,
        rerank: bool = False,
        rerank_oversearch: Optional[int] = None,
        diversify: bool = True,
    ) -> list[tuple[Chunk, float]]:
        """Return top-k (Chunk, score) for the given query string.

        Args:
            query: free-text query used for DENSE retrieval (and BM25 too
                when ``bm25_query`` is None — the symmetric default).
            k: number of passages to return.
            bm25_query: optional LEXICAL query string used only for the
                BM25 path inside ``hybrid``. Useful when the dense path
                wants a hypothesis (HyDE) but the BM25 path needs the
                user's original lexical signal (proper nouns, dates).
                Ignored when ``hybrid=False``.
            category: when set, restrict results to chunks whose
                ``Chunk.category`` matches this string.
            hybrid: when True, query BOTH the dense index AND the
                attached BM25 index, then RRF-fuse the two ranked lists.
                Score-scale-independent — dense cosine and BM25 scores
                aren't comparable, but ranks are. Requires bm25=
                at construction.
            rerank: when True, oversearch and rerank the pool with the
                attached cross-encoder. Composes with ``hybrid``: dense
                + BM25 are RRF-fused first, then the reranker scores
                the fused pool.
            rerank_oversearch: how many times k to ask the underlying
                retrievers for before reranking. Default: 5.
            diversify: when True (default), apply a source-level dedup
                to the final top-k slice — the highest-scored chunk per
                source comes first, then any leftover chunks fill the
                remaining slots. Stops top-k from being three overlapping
                windows of the same article. Disable for ablations.

        Returns:
            Up to ``k`` (Chunk, score) pairs. Score units:
              dense-only           → cosine
              hybrid (no rerank)   → RRF
              rerank (any source)  → cross-encoder
        """
        if rerank and self._reranker is None:
            raise ValueError(
                "rerank=True but no reranker is attached. Construct "
                "with Retriever(index, embedder, reranker=...)."
            )
        if hybrid and self._bm25 is None:
            raise ValueError(
                "hybrid=True but no BM25 index is attached. Construct "
                "with Retriever(index, embedder, bm25=...)."
            )

        rerank_x = rerank_oversearch or self._DEFAULT_RERANK_OVERSEARCH
        # How many chunks each underlying retriever should surface so
        # the reranker / fusion / diversify step has enough headroom.
        if rerank:
            target_pool = k * rerank_x
        elif diversify:
            target_pool = k * self._DIVERSIFY_OVERSEARCH
        else:
            target_pool = k

        # 1. Dense retrieval (always — it's the primary signal).
        dense_hits = self._dense_search(query, target_pool, category)

        # 2. Optional BM25 retrieval + RRF fusion.
        if hybrid:
            lex_query = bm25_query if bm25_query is not None else query
            bm25_hits = self._bm25.search(  # type: ignore[union-attr]
                lex_query, k=target_pool, category=category,
            )
            pool = reciprocal_rank_fusion(
                [dense_hits, bm25_hits],
                k=target_pool,
                weights=self._rrf_weights,
            )
        else:
            pool = dense_hits[:target_pool]

        # 3. Optional cross-encoder rerank. We rerank the full pool (no
        #    top_k truncation) so the post-rerank diversify pass has the
        #    largest possible material to pull substitutes from when the
        #    top of the ranking is dominated by a single source.
        if rerank:
            pool = self._reranker.rerank(query, pool, top_k=None)  # type: ignore[union-attr]

        if diversify:
            return _diversify_by_source(pool, k=k)
        return pool[:k]

    def _dense_search(
        self,
        query: str,
        k_pool: int,
        category: Optional[str],
    ) -> list[tuple[Chunk, float]]:
        """Encode + FAISS search + optional Python-side category filter.

        The category filter is post-hoc Python (FAISS' IndexFlatIP has no
        cross-version ID mask). We first ask for ``k_pool *
        _CATEGORY_OVERSEARCH`` chunks; if the matching slice doesn't fill
        ``k_pool``, we retry with the full index. Cap at one retry — a
        category with very few chunks just returns fewer than k, which
        downstream code already tolerates.
        """
        n_total = self._index.n_chunks or 1
        # Prefer the asymmetric path when the embedder offers it; fall back
        # to ``encode`` so simple test mocks (which only define .encode)
        # keep working.
        encode_fn = getattr(self._embedder, "encode_query", None) \
            or self._embedder.encode
        query_vec = encode_fn([query])
        if category is None:
            return self._index.search(
                query_vec, k=min(k_pool, n_total),
            )
        k_dense = min(k_pool * self._CATEGORY_OVERSEARCH, n_total)
        raw = self._index.search(query_vec, k=k_dense)
        filtered = [(c, s) for c, s in raw if c.category == category]
        if len(filtered) < k_pool and k_dense < n_total:
            # The 8× oversearch didn't surface enough of the target
            # category — scan the full index and refilter.
            raw = self._index.search(query_vec, k=n_total)
            filtered = [(c, s) for c, s in raw if c.category == category]
        return filtered[:k_pool]

    def append_chunks(
        self,
        chunks: list[Chunk],
        embeddings,
    ) -> None:
        """Hot-append new chunks to both the FAISS and BM25 indices.

        Called by ``IndexGrower.confirm()`` after a correct answer is confirmed.
        Updates both the dense and (when present) lexical indices in memory so
        that subsequent ``retrieve()`` calls in the same session can find the
        new material.

        Does NOT persist to disk — call ``IndexGrower.flush()`` (or
        ``FAISSIndex.save`` + ``BM25Index.save`` directly) at session end.

        Args:
            chunks: new Chunk objects (already deduped by the caller).
            embeddings: float32 ndarray of shape (len(chunks), dim), produced
                by the same embedder used to build the index (caller
                responsibility — no spec check here; IndexGrower enforces it).
        """
        if not chunks:
            return
        self._index.append(chunks, embeddings)
        if self._bm25 is not None:
            self._bm25.append(chunks)

    def rerank_pool(
        self,
        query: str,
        pool: list[tuple[Chunk, float]],
        *,
        k: int,
    ) -> list[tuple[Chunk, float]]:
        """Score an already-assembled pool with the cross-encoder and return top-k.

        Exposed so callers (e.g. RAGStrategy's multi-query path) can fuse
        across queries first, then rerank the merged pool once — rather than
        reranking inside each per-query retrieve() call and then discarding
        those cross-encoder scores in an outer RRF pass.

        Requires a reranker to be attached; raises otherwise.
        """
        if self._reranker is None:
            raise ValueError(
                "rerank_pool() called but no reranker is attached. "
                "Construct with Retriever(index, embedder, reranker=...)."
            )
        return self._reranker.rerank(query, pool, top_k=k)

    @property
    def n_chunks(self) -> int:
        """How many chunks are indexed."""
        return self._index.n_chunks

    def iter_sources(self) -> set[str]:
        """Return the set of source article titles currently indexed.

        Exposed so callers (notably IndexGrower) can dedup new articles
        without reaching into ``_index._chunks`` directly.
        """
        return {c.source for c in self._index._chunks}

    @classmethod
    def from_saved(
        cls,
        index_path: Path,
        *,
        embedder_spec: EmbedderSpec | None = None,
    ) -> "Retriever":
        """Convenience constructor: load index from disk + spin up embedder.

        If the index has a manifest, the embedder spec is checked against
        it and a mismatch raises before any retrieval happens (model_name
        / dim mismatches silently corrupt scores otherwise).
        """
        spec = embedder_spec or EmbedderSpec()
        index = FAISSIndex.load(index_path)
        if index.manifest is not None:
            _check_manifest_compat(index.manifest, spec)
        embedder = Embedder(spec)
        return cls(index, embedder)