"""Public retrieval interface. Strategies call this; they never touch FAISS directly."""
from __future__ import annotations

from pathlib import Path

from .chunker import Chunk
from .embedder import Embedder, EmbedderSpec
from .index import FAISSIndex


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
        """Convenience constructor: load index from disk + spin up embedder."""
        spec = embedder_spec or EmbedderSpec()
        embedder = Embedder(spec)
        index = FAISSIndex.load(index_path)
        return cls(index, embedder)