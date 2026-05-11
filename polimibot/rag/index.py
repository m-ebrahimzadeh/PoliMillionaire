"""FAISS index: build from chunks, persist to disk, load back."""
from __future__ import annotations

import datetime as _dt
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from .chunker import Chunk


class FAISSIndex:
    """Flat exact inner-product index over chunk embeddings.

    Persistence convention: three files share the same stem:
        {stem}.faiss          — the binary FAISS index
        {stem}.jsonl          — chunk metadata (text, source, chunk_id)
        {stem}.manifest.json  — build manifest (embedder, chunking, versions)

    The manifest is optional on load (legacy indices without one still load
    with a warning), but mandatory at build time going forward — without it,
    a downstream caller can't verify they're querying with a compatible
    embedder. Mismatches silently corrupt scores; the manifest is the fence.

    Usage:
        idx = FAISSIndex(dim=384)
        idx.add(chunks, embeddings)
        idx.save(Path("data/cache/knowledge"), manifest={
            "embedder_model_name": "all-MiniLM-L6-v2",
            "embedder_dim": 384,
            "normalize": True,
            "chunk_size": 300,
            "chunk_overlap": 50,
            "n_articles": 95,
            "text_cleanup_version": 1,
        })
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
        self.manifest: Optional[dict] = None

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

    def save(self, path: Path, *, manifest: Optional[dict] = None) -> None:
        """Write {path}.faiss, {path}.jsonl, and (if provided) {path}.manifest.json.

        ``manifest`` should record everything a future caller needs to
        confirm they're querying compatibly: embedder model name + dim,
        chunking parameters, text-cleanup version, build timestamp,
        library versions. ``Retriever.from_saved`` validates these.
        """
        import faiss
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for c in self._chunks:
                row = {
                    "text": c.text,
                    "source": c.source,
                    "chunk_id": c.chunk_id,
                }
                if c.category is not None:
                    row["category"] = c.category
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if manifest is not None:
            full = dict(manifest)
            full.setdefault("n_chunks", self.n_chunks)
            full.setdefault(
                "build_timestamp",
                _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            )
            # Snapshot library versions so a downstream caller can spot a
            # tokenizer upgrade that would shift embeddings.
            for mod_name, key in (
                ("sentence_transformers", "sentence_transformers_version"),
                ("transformers",          "transformers_version"),
                ("faiss",                 "faiss_version"),
            ):
                if key in full:
                    continue
                try:
                    mod = __import__(mod_name)
                    full[key] = getattr(mod, "__version__", "unknown")
                except ImportError:
                    pass
            self.manifest = full
            path.with_suffix(".manifest.json").write_text(
                json.dumps(full, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print(f"Saved {self.n_chunks} chunks → {path}.{{faiss,jsonl"
              + (",manifest.json" if manifest is not None else "")
              + "}}")

    @classmethod
    def load(cls, path: Path) -> "FAISSIndex":
        """Load from {path}.faiss + {path}.jsonl (+ optional {path}.manifest.json).

        Manifests pre-date this PR for some indices; load proceeds with a
        warning if absent. Use ``Retriever.from_saved`` to enforce
        embedder compatibility at load time.
        """
        import faiss
        faiss_index = faiss.read_index(str(path.with_suffix(".faiss")))
        chunks: list[Chunk] = []
        with path.with_suffix(".jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                chunks.append(Chunk(
                    text=d["text"],
                    source=d["source"],
                    chunk_id=d["chunk_id"],
                    category=d.get("category"),   # absent on legacy chunks
                ))
        obj = cls(dim=faiss_index.d, _faiss_index=faiss_index)
        obj._chunks = chunks

        manifest_path = path.with_suffix(".manifest.json")
        if manifest_path.is_file():
            obj.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            warnings.warn(
                f"Index at {path} has no .manifest.json — embedder "
                f"compatibility cannot be verified. Rebuild to enable checks.",
                RuntimeWarning,
                stacklevel=2,
            )
        return obj