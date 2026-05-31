"""Text splitting. Retrieval quality depends on chunk design more than model choice.

Sentence-aware: a chunk boundary lands at a sentence end whenever a sentence
fits in the remaining window. Sentences longer than the window are word-split.

Section-aware: Wikipedia-style ``== Header ==`` lines split the body and the
header text is prepended once to the first chunk of its section, so the
heading travels with its content instead of floating mid-window.

Min-chunk filter: a short tail window (default: ``chunk_size // 5`` words)
gets merged into its predecessor. Stops the last window of a document from
producing a 10-word vector that competes for top-k slots with full chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional


# Bumped whenever the chunker's output changes for the same input. Surfaced
# in the index manifest so a stale on-disk index can be detected by callers.
CHUNKER_VERSION = 2

# Bumped when the embedding-INPUT transform (``embedding_text``) changes.
# Tracked separately from CHUNKER_VERSION because it does NOT alter the stored
# ``Chunk.text`` — only the string handed to the passage embedder. Recorded in
# the index manifest so a grounding change is visible to a future loader.
EMBED_TEXT_VERSION = 1


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


def embedding_text(chunk: Chunk) -> str:
    """Return the string fed to the passage embedder — chunk grounded in source.

    ``Chunk.text`` is kept pure: display, BM25, and the prompt all use it
    verbatim. The *embedded* form, however, is prefixed with the source title
    so every passage vector is anchored to its entity. Trivia questions name
    the entity ("Who painted the Mona Lisa?"), so grounding each passage
    embedding in its article title sharpens cross-article matching — a chunk
    reading "was painted in 1503" embeds as "Mona Lisa: was painted in 1503".

    This must be applied identically at EVERY embed site (index build,
    live-search scoring, IndexGrower) or vectors land in inconsistent spaces.
    Callers import this helper rather than inlining the format so the
    convention stays in one place; ``EMBED_TEXT_VERSION`` tracks changes to it.
    """
    source = chunk.source.strip() if chunk.source else ""
    if not source:
        return chunk.text
    return f"{source}: {chunk.text}"


# Wikipedia section header line: "== Early life ==", "=== Career ===", ...
_HEADER_RE = re.compile(r"^={2,}\s*(.+?)\s*={2,}\s*$", re.MULTILINE)

# Sentence boundary: ., !, ? followed by whitespace. Conservative — won't
# perfectly handle "Mr. Smith" or "e.g. foo", but a stray split costs one
# slightly-short chunk that the min-chunk filter then absorbs.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split text on Wikipedia-style section headers.

    Returns ``[(header_text, body), ...]``. The first pair's header is ``""``
    when text begins with body content. Empty bodies are dropped.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]
    out: list[tuple[str, str]] = []
    first_start = matches[0].start()
    if first_start > 0:
        prelude = text[:first_start].strip()
        if prelude:
            out.append(("", prelude))
    for i, m in enumerate(matches):
        header_text = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            out.append((header_text, body))
    return out


def _split_sentences(body: str) -> list[str]:
    """Split a section body into sentences. Empty sentences are dropped."""
    parts = _SENTENCE_BOUNDARY_RE.split(body.strip())
    return [p.strip() for p in parts if p.strip()]


def _pack_sentences(
    sentences: list[str],
    *,
    chunk_size: int,
    overlap: int,
) -> Iterator[list[str]]:
    """Greedy sentence packer.

    Emits lists of words sized ≤ ``chunk_size``, with the last ``overlap``
    words of each emitted window carried into the next as its prefix.
    A sentence longer than ``chunk_size`` is word-split internally to honour
    the size cap; the carryover-overlap discipline still applies between the
    final word-window of the long sentence and the next sentence.
    """
    buf_words: list[str] = []
    for sent in sentences:
        words = sent.split()
        if not words:
            continue

        # Long sentence: word-window it. Flush any buffered short sentences
        # first so the long sentence's windows aren't padded with stale words.
        if len(words) > chunk_size:
            if buf_words:
                yield buf_words
                buf_words = []
            stride = max(1, chunk_size - overlap)
            last_window: list[str] = []
            for start in range(0, len(words), stride):
                window = words[start:start + chunk_size]
                yield window
                last_window = window
                if start + chunk_size >= len(words):
                    break
            # Seed the next iteration's buffer with the tail of the long
            # sentence so we keep word-level overlap across the boundary.
            if overlap > 0 and last_window:
                buf_words = last_window[-overlap:]
            continue

        # Short sentence: append, flushing the buffer first if it would
        # overflow. Buffer flush re-seeds with the overlap tail.
        if len(buf_words) + len(words) > chunk_size:
            yield buf_words
            buf_words = buf_words[-overlap:] if overlap > 0 else []
        buf_words.extend(words)

    if buf_words:
        yield buf_words


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 300,   # words per chunk
    overlap: int = 50,       # words shared between consecutive chunks
    min_chunk_words: Optional[int] = None,
    category: Optional[str] = None,
) -> list[Chunk]:
    """Split text into overlapping word-windows that respect sentence and
    section boundaries.

    Args:
        text: raw document text. Wikipedia-style ``== Header ==`` lines are
            treated as section boundaries; the header text is prepended to
            the first chunk of its section rather than floating mid-chunk.
        source: label embedded in every chunk (appears in RAG prompts).
        chunk_size: target window size in words.
        overlap: words shared between consecutive windows.
        min_chunk_words: chunks below this size are merged into the preceding
            chunk. Defaults to ``max(1, chunk_size // 5)`` so callers using
            small chunk_size values in tests don't end up with a single
            mega-merged output.
        category: optional category tag stamped on every chunk produced from
            this document. The retriever's category filter reads it.

    Returns:
        List of Chunk objects, in document order. ``chunk_id`` is the 0-based
        position after any min-chunk merges, so ids are always contiguous.
    """
    if not text or not text.strip():
        return []
    if min_chunk_words is None:
        min_chunk_words = max(1, chunk_size // 5)

    sections = _split_sections(text)

    chunks: list[Chunk] = []
    cid = 0
    for header, body in sections:
        sentences = _split_sentences(body)
        first_in_section = True
        for window_words in _pack_sentences(
            sentences, chunk_size=chunk_size, overlap=overlap,
        ):
            chunk_text_str = " ".join(window_words)
            if first_in_section and header:
                # Prepend the section header to the first chunk of the
                # section only — keeps the header as a localised signal.
                chunk_text_str = f"{header}\n{chunk_text_str}"
                first_in_section = False
            chunks.append(Chunk(
                text=chunk_text_str,
                source=source,
                chunk_id=cid,
                category=category,
            ))
            cid += 1

    # Tail-merge any chunks below the minimum size into their predecessor.
    # Headers stay with whichever chunk they were originally attached to.
    if min_chunk_words > 1 and len(chunks) > 1:
        merged: list[Chunk] = [chunks[0]]
        for c in chunks[1:]:
            if len(c.text.split()) < min_chunk_words:
                prev = merged[-1]
                merged[-1] = Chunk(
                    text=f"{prev.text} {c.text}",
                    source=prev.source,
                    chunk_id=prev.chunk_id,
                    category=prev.category,
                )
            else:
                merged.append(c)
        # Re-number ids contiguously so the merge isn't visible to callers.
        chunks = [
            Chunk(text=c.text, source=c.source, chunk_id=i, category=c.category)
            for i, c in enumerate(merged)
        ]

    return chunks
