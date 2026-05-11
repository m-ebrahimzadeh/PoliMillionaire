"""Public retrieval interface. Strategies call this; they never touch FAISS directly."""
from __future__ import annotations

import warnings
from pathlib import Path

from .chunker import Chunk
from .embedder import Embedder, EmbedderSpec
from .index import FAISSIndex


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

    def __init__(self, index: FAISSIndex, embedder: Embedder) -> None:
        if index.dim != embedder.dim:
            raise ValueError(
                f"Index dim={index.dim} != embedder dim={embedder.dim}. "
                "Must use the same model for indexing and querying."
            )
        self._index = index
        self._embedder = embedder

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return top-k (Chunk, cosine_score) for the given query string.

        Scores are in [0, 1] because both query and chunk vectors are L2-normalized.
        """
        query_vec = self._embedder.encode([query])  # (1, dim)
        return self._index.search(query_vec, k=k)

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