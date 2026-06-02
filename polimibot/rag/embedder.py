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
from typing import Optional


def _prefixes_for_model(model_name: str) -> tuple[str, str]:
    """Return ``(query_prefix, passage_prefix)`` for a model family.

    Asymmetric retrievers (BGE, E5) corrupt cosine scores when the
    query/passage prefixes don't match what the model was trained with.
    This map keeps the prefixes in sync with the model_name so callers
    only need to specify the model.
    """
    name = model_name.lower()
    # BGE-M3 is a different family from the bge-*-en-v1.5 line: its dense
    # retrieval is trained WITHOUT a query instruction. Prepending the v1.5
    # "Represent this sentence…" prompt twists query vs passage space and
    # silently degrades cosine scores — so it must be checked BEFORE the
    # generic ``"bge" in name`` branch below.
    if "bge-m3" in name or "bge_m3" in name:
        return ("", "")
    if "bge" in name:
        return ("Represent this sentence for searching relevant passages: ", "")
    if "e5" in name:
        return ("query: ", "passage: ")
    # MiniLM, mpnet, and other symmetric models — no prefixes.
    return ("", "")


def _resolve_fp16(flag: Optional[bool]) -> bool:
    """Decide whether to load encoder weights in fp16.

    ``flag`` wins when set explicitly. When ``None`` (the default) we auto-pick:
    fp16 on CUDA (halves the encoder's VRAM with negligible recall loss), fp32
    on CPU (half-precision matmul is unsupported / slow there, and the test
    suite stubs ``torch.cuda.is_available`` to False so CI stays fp32).
    """
    if flag is not None:
        return flag
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


@dataclass(frozen=True)
class EmbedderSpec:
    """Config for the embedding model. Frozen → safe to share across objects.

    ``query_prefix`` / ``passage_prefix`` are prepended before encoding.
    Leave them as ``None`` to let :func:`_prefixes_for_model` derive the
    right values from ``model_name`` (bge-*-en-v1.5 → instruction prefix;
    bge-m3 → empty; E5 → ``"query: "/"passage: "``; MiniLM/mpnet → empty).
    Mismatches between indexer and retriever corrupt scores — the manifest
    carries both values and ``_check_manifest_compat`` hard-fails on drift.
    """
    model_name: str = "BAAI/bge-base-en-v1.5"   # 768-dim, ~440 MB
    batch_size: int = 64
    normalize: bool = True   # L2-normalize → cosine sim becomes dot product
    query_prefix: Optional[str] = None    # None → auto-derive from model_name
    passage_prefix: Optional[str] = None  # None → auto-derive from model_name
    fp16: Optional[bool] = None  # None → auto (fp16 on CUDA, fp32 on CPU)

    def __post_init__(self) -> None:
        if self.query_prefix is None or self.passage_prefix is None:
            q, p = _prefixes_for_model(self.model_name)
            if self.query_prefix is None:
                object.__setattr__(self, "query_prefix", q)
            if self.passage_prefix is None:
                object.__setattr__(self, "passage_prefix", p)


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
        # Half-precision weights on GPU — ~2x smaller VRAM, no measurable recall
        # loss. SentenceTransformer is an nn.Module, so .half() converts in place.
        if _resolve_fp16(self.spec.fp16):
            self._model = self._model.half()
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
