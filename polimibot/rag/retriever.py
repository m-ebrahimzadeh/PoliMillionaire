"""Public retrieval interface. Strategies call this; they never touch FAISS directly."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from .chunker import Chunk
from .embedder import Embedder, EmbedderSpec
from .index import FAISSIndex
from .reranker import CrossEncoderReranker


def _check_manifest_compat(manifest: dict, spec: EmbedderSpec) -> None:
    """Refuse to load an index built with an incompatible embedder.

    Hard-fails on model_name or dim mismatch (those silently corrupt
    scores). Warns on normalize / chunk_size drift (less catastrophic
    but worth surfacing).
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
    ) -> None:
        if index.dim != embedder.dim:
            raise ValueError(
                f"Index dim={index.dim} != embedder dim={embedder.dim}. "
                "Must use the same model for indexing and querying."
            )
        self._index = index
        self._embedder = embedder
        self._reranker = reranker

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    # Oversearch factors. When a category filter or a reranker is in play
    # we ask the index for more chunks than the caller wants, then trim
    # down. Pure dense IndexFlatIP doesn't support an in-FAISS ID mask
    # cleanly across versions, and Python filtering on a small index is
    # cheap. The default rerank oversearch matches the cross-encoder
    # literature's "retrieve 5× more, rerank to k".
    _CATEGORY_OVERSEARCH = 8
    _DEFAULT_RERANK_OVERSEARCH = 5

    def retrieve(
        self,
        query: str,
        k: int = 3,
        *,
        category: Optional[str] = None,
        rerank: bool = False,
        rerank_oversearch: Optional[int] = None,
    ) -> list[tuple[Chunk, float]]:
        """Return top-k (Chunk, score) for the given query string.

        Args:
            query: free-text query.
            k: number of passages to return.
            category: when set, restrict results to chunks whose
                ``Chunk.category`` matches this string. Chunks with
                ``category=None`` are excluded under a filter — call
                without ``category`` to include them. Pass the string
                value, e.g. ``Category.MATHS.value``.
            rerank: when True, oversearch by ``rerank_oversearch × k``
                and rerank the pool with the attached cross-encoder.
                Returned scores are CROSS-ENCODER scores (not cosine).
                Raises if no reranker was set on construction.
            rerank_oversearch: how many times k to ask the dense index
                for before reranking. Default: 5. Larger = more recall
                headroom for the reranker at higher latency cost.

        Returns:
            Up to ``k`` (Chunk, score) pairs. May return fewer if the
            category filter is active and the oversearched pool didn't
            contain enough matching chunks. Score units depend on
            ``rerank``: cosine when False, cross-encoder when True.
        """
        if rerank and self._reranker is None:
            raise ValueError(
                "rerank=True but no reranker is attached. Construct "
                "with Retriever(index, embedder, reranker=...)."
            )

        rerank_x = rerank_oversearch or self._DEFAULT_RERANK_OVERSEARCH
        # How many chunks the reranker needs to see, post-filter.
        target_pool = k * rerank_x if rerank else k
        n_total = self._index.n_chunks or 1

        query_vec = self._embedder.encode([query])  # (1, dim)

        if category is None:
            hits = self._index.search(
                query_vec, k=min(target_pool, n_total)
            )
        else:
            # Oversearch dense by category factor on top of the
            # rerank-target pool — both gated by total chunks.
            k_dense = min(target_pool * self._CATEGORY_OVERSEARCH, n_total)
            raw = self._index.search(query_vec, k=k_dense)
            hits = [(c, s) for c, s in raw if c.category == category][:target_pool]

        if rerank:
            # type-narrow: we asserted self._reranker is not None above.
            return self._reranker.rerank(query, hits, top_k=k)  # type: ignore[union-attr]
        return hits[:k]

    @property
    def n_chunks(self) -> int:
        """How many chunks are indexed."""
        return self._index.n_chunks

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