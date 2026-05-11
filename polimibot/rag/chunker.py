"""Text splitting. Retrieval quality depends on chunk design more than model choice."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Chunk:
    """Atomic retrieval unit. Immutable so it's safe to cache and share.

    ``category`` is the source-document's category (entertainment / history /
    science / maths) — used by the retriever's category filter to surface
    only on-topic passages for the question being asked. Stored as a
    plain string for JSON-friendliness; ``None`` means "unknown / generic".
    """
    text: str
    source: str             # document title / filename — shown in the prompt
    chunk_id: int           # position within that document (0-based)
    category: Optional[str] = None


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 300,   # words per chunk
    overlap: int = 50,       # words shared between consecutive chunks
    category: Optional[str] = None,
) -> list[Chunk]:
    """Split text into overlapping word-windows.

    Why overlap? If the answer to a question straddles two chunks
    (e.g., a sentence that begins at word 298), one chunk will always
    contain it in full.

    Args:
        text: raw document text
        source: label embedded in every chunk (appears in RAG prompts)
        chunk_size: target window size in words
        overlap: stride = chunk_size - overlap
        category: optional category tag stamped on every chunk produced from
            this document. The retriever's category filter reads it.

    Returns:
        List of Chunk objects, in document order.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    stride = max(1, chunk_size - overlap)

    for cid, start in enumerate(range(0, len(words), stride)):
        chunk_words = words[start : start + chunk_size]
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            source=source,
            chunk_id=cid,
            category=category,
        ))
        if start + chunk_size >= len(words):
            break  # last window consumed all remaining words

    return chunks