"""Sentence-transformer wrapper. Isolated here so swap to a different model is a one-liner."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedderSpec:
    """Config for the embedding model. Frozen → safe to share across objects."""
    model_name: str = "all-MiniLM-L6-v2"   # 80 MB, 384-dim, CPU-friendly
    batch_size: int = 64
    normalize: bool = True   # L2-normalize → cosine sim becomes dot product (required for IndexFlatIP)


class Embedder:
    """Wraps SentenceTransformer. One instance per process — load once, reuse."""

    def __init__(self, spec: EmbedderSpec | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.spec = spec or EmbedderSpec()
        self._model = SentenceTransformer(self.spec.model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings. Returns float32 array of shape (N, dim).

        If spec.normalize=True (default), each vector has unit L2 norm,
        so inner product == cosine similarity.
        """
        vecs = self._model.encode(
            texts,
            batch_size=self.spec.batch_size,
            normalize_embeddings=self.spec.normalize,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)