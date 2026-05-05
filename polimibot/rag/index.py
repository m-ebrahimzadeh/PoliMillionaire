"""FAISS index: build from chunks, persist to disk, load back."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from .chunker import Chunk


class FAISSIndex:
    """Flat exact inner-product index over chunk embeddings.

    Persistence convention: two files share the same stem:
        {stem}.faiss  — the binary FAISS index
        {stem}.jsonl  — chunk metadata (text, source, chunk_id)

    Usage:
        idx = FAISSIndex(dim=384)
        idx.add(chunks, embeddings)
        idx.save(Path("data/cache/knowledge"))
        # later:
        idx = FAISSIndex.load(Path("data/cache/knowledge"))
        results = idx.search(query_vec, k=3)
    """

    def __init__(self, dim: int, _faiss_index=None) -> None:
        """
        Args:
            dim: embedding dimension (must match the embedder used to build it)
            _faiss_index: internal — pass a pre-loaded faiss index to avoid rebuilding
        """
        import faiss
        self.dim = dim
        self._index = _faiss_index if _faiss_index is not None else faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks with their precomputed embeddings (already L2-normalized).

        Args:
            chunks: list of Chunk objects (same order as embeddings rows)
            embeddings: float32 array of shape (len(chunks), dim)
        """
        if embeddings.shape != (len(chunks), self.dim):
            raise ValueError(
                f"Shape mismatch: expected ({len(chunks)}, {self.dim}), "
                f"got {embeddings.shape}"
            )
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs.

        Args:
            query_vec: float32 array of shape (1, dim), L2-normalized
            k: number of results

        Returns:
            List of (Chunk, cosine_score) sorted descending by score.
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]   # FAISS expects (1, dim)
        scores, ids = self._index.search(query_vec, k)
        return [
            (self._chunks[i], float(scores[0][j]))
            for j, i in enumerate(ids[0])
            if i >= 0  # -1 means the index had fewer than k entries
        ]

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    def save(self, path: Path) -> None:
        """Write {path}.faiss and {path}.jsonl."""
        import faiss
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(
                    {"text": c.text, "source": c.source, "chunk_id": c.chunk_id},
                    ensure_ascii=False,
                ) + "\n")
        print(f"Saved {self.n_chunks} chunks → {path}.{{faiss,jsonl}}")

    @classmethod
    def load(cls, path: Path) -> "FAISSIndex":
        """Load from {path}.faiss + {path}.jsonl."""
        import faiss
        faiss_index = faiss.read_index(str(path.with_suffix(".faiss")))
        chunks: list[Chunk] = []
        with path.with_suffix(".jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                chunks.append(Chunk(text=d["text"], source=d["source"], chunk_id=d["chunk_id"]))
        obj = cls(dim=faiss_index.d, _faiss_index=faiss_index)
        obj._chunks = chunks
        return obj