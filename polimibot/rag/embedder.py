"""Sentence-transformer wrapper. Asymmetric query/passage encoding for
retrievers that benefit from a model-specific prompt prefix (BGE, E5, …).

Why the asymmetry: BGE was trained with a query-side instruction prompt
("Represent this sentence for searching relevant passages: ") and a bare
passage. Encoding queries without that prefix at retrieval time silently
degrades cosine scores — the query and passage vectors live in slightly
twisted halves of the embedding space. The prefix lives in EmbedderSpec
so the indexer and the retriever can be checked for agreement at load
time via the manifest.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# Model-specific instruction prefix used on QUERIES (not passages).
# Switching model? Update both ``model_name`` and ``query_prefix`` on the
# EmbedderSpec; using the wrong prefix silently degrades cosine scores.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbedderSpec:
    """Config for the embedding model. Frozen → safe to share across objects.

    ``query_prefix`` / ``passage_prefix`` are prepended before encoding.
    For BGE: query prefix is the instruction string above, passage prefix
    is empty. For E5: ``"query: "`` / ``"passage: "``. For MiniLM: both
    empty. Mismatches between indexer and retriever corrupt scores — the
    manifest carries both values and ``_check_manifest_compat`` hard-fails
    on drift.
    """
    model_name: str = "BAAI/bge-small-en-v1.5"   # 384-dim, ~130 MB, CPU-friendly
    batch_size: int = 64
    normalize: bool = True   # L2-normalize → cosine sim becomes dot product
    query_prefix: str = _BGE_QUERY_PREFIX
    passage_prefix: str = ""


class Embedder:
    """Wraps SentenceTransformer. One instance per process — load once, reuse.

    Use ``encode_query`` for retrieval queries and ``encode_passage`` for
    documents at indexing time. The two methods apply the spec-defined
    prefixes; calling the wrong one on an asymmetric model silently
    degrades retrieval quality.
    """

    def __init__(self, spec: EmbedderSpec | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.spec = spec or EmbedderSpec()
        self._model = SentenceTransformer(self.spec.model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], *, prefix: str) -> np.ndarray:
        """Internal encoder. Prepends the prefix only when non-empty —
        avoids wasting tokenizer cycles in the symmetric-model case.
        """
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
        vecs = self._model.encode(
            prefixed,
            batch_size=self.spec.batch_size,
            normalize_embeddings=self.spec.normalize,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def encode_query(self, texts: list[str]) -> np.ndarray:
        """Embed retrieval queries with the model's query prefix."""
        return self._encode(texts, prefix=self.spec.query_prefix)

    def encode_passage(self, texts: list[str]) -> np.ndarray:
        """Embed corpus passages with the model's passage prefix."""
        return self._encode(texts, prefix=self.spec.passage_prefix)
