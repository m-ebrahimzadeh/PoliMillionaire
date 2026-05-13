"""Self-growing RAG index — the learning-loop manager.

Lifecycle
─────────
  1. ``buffer(article, question_id)``
       Called by RAGStrategy when live search finds something.
       Chunks the article, stores it in a "pending" buffer keyed by
       question_id, and returns the chunks immediately so the strategy
       can format them as context for the current question.

  2. ``confirm(question_id)``
       Called by runner.py when the game server confirms the answer was
       *correct*.  Embeds the pending chunks and appends them to the
       in-memory FAISS + BM25 indices so that repeat questions in the
       same session hit the offline path next time.  Articles for which
       ``confirm()`` is never called (wrong answers) are quietly discarded.

  3. ``flush()``
       Called once at session end.  Persists the grown index to disk by
       re-saving the FAISS + BM25 files and appending the new raw articles
       to ``corpus.jsonl``.

Design choices
──────────────
- **Only confirmed-correct articles are learned.**  Avoids polluting the
  index with content that led to a wrong answer.
- **Embed at confirm-time, not buffer-time.**  Embedding is the expensive
  step (~100 ms per article on CPU).  Doing it after the game server
  confirms correctness means the embedding cost is paid only for useful
  material, and the latency impact on the *current* question is zero
  (the game has already moved on by then, or the flush is end-of-session).
- **In-memory append is immediate.**  Once confirmed, chunks are added to
  the live Retriever so same-session repeat questions benefit.
- **Disk flush is deferred to session end.**  Writing FAISS + BM25 +
  corpus.jsonl during a timed question risks I/O stalling the game loop.
- **Thread-safe.**  A single ``threading.Lock`` serialises all mutations
  to the pending buffer and to the in-memory index.  The embedding call
  inside ``confirm()`` is CPU-bound but brief; holding the lock across it
  is acceptable since the game never calls ``confirm()`` and ``retrieve()``
  truly concurrently (they run in the same event loop).
- **Dedup on article title.**  Before buffering, the grower checks whether
  a title is already in the index (checked against existing chunk sources)
  or in the pending buffer; duplicates are skipped silently.

Usage (wired in runner.py)
──────────────────────────
    grower = IndexGrower(retriever, embedder, PATHS.cache_dir / "knowledge")

    # RAGStrategy calls this internally when live search fires:
    chunks = grower.buffer(article, question_id="q_lvl3")

    # runner.py calls this after outcome.correct is True:
    grower.confirm("q_lvl3")

    # runner.py calls this once at game end:
    grower.flush()
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .chunker import Chunk, chunk_text
from .corpus import Article, save_raw_corpus, load_raw_corpus
from .embedder import Embedder
from .retriever import Retriever


# Default chunking parameters mirror build_rag_index.py defaults so
# newly learned chunks are dimensionally consistent with the original index.
_DEFAULT_CHUNK_SIZE    = 300
_DEFAULT_CHUNK_OVERLAP = 50


@dataclass
class _PendingEntry:
    """An article that has been fetched but not yet confirmed correct."""
    article: Article
    chunks:  list[Chunk]   # pre-chunked, awaiting embedding


class IndexGrower:
    """Manages the buffer → confirm → flush learning loop.

    Args:
        retriever:   the live Retriever instance (will be mutated in-memory
                     on ``confirm()``).
        embedder:    must be the *same* Embedder instance (or spec) used to
                     build the existing index — vectors must live in the same
                     space.
        index_path:  path stem used by ``FAISSIndex.save()`` and
                     ``BM25Index.save()``, e.g.
                     ``PATHS.cache_dir / "knowledge"``.
        corpus_path: path to the raw corpus JSONL file for persistence.
                     Defaults to ``index_path.parent / "corpus.jsonl"``.
        chunk_size:  words per chunk (default 300, mirrors build script).
        chunk_overlap: overlap in words between adjacent chunks (default 50).
    """

    def __init__(
        self,
        retriever: Retriever,
        embedder: Embedder,
        index_path: Path,
        *,
        corpus_path: Optional[Path] = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._retriever   = retriever
        self._embedder    = embedder
        self._index_path  = index_path
        self._corpus_path = corpus_path or (index_path.parent / "corpus.jsonl")
        self._chunk_size  = chunk_size
        self._chunk_overlap = chunk_overlap

        # question_id → _PendingEntry (not yet confirmed).
        self._pending: dict[str, _PendingEntry] = {}

        # Articles that have been confirmed and appended to the live index.
        # Kept for flush() so we know what to persist.
        self._confirmed: list[Article] = []

        # Set of article titles already present in the live index (offline
        # + newly confirmed).  Used to skip duplicate live-search results.
        self._indexed_titles: set[str] = self._collect_existing_titles()

        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def buffer(self, article: Article, question_id: str) -> list[Chunk]:
        """Buffer an article for potential learning.

        Called by RAGStrategy immediately after live search succeeds.
        Returns the pre-chunked text so the strategy can build context
        for the current question *right now* without waiting for confirm().

        If the article title is already in the index or already pending,
        returns the existing chunks (or an empty list for already-indexed
        articles — the offline retriever will serve them anyway).

        Args:
            article:     Article object returned by LiveSearchFallback.
            question_id: opaque string key used to link this buffer entry
                         to a later ``confirm()`` call.  Typically the
                         question level string, e.g. ``"lvl_3"``.

        Returns:
            List of Chunk objects derived from the article text.
        """
        with self._lock:
            # Skip if already indexed (offline or confirmed-this-session).
            if article.title in self._indexed_titles:
                return []

            # Skip if already pending under a different question_id.
            for entry in self._pending.values():
                if entry.article.title == article.title:
                    return entry.chunks

            # Chunk the article using the same parameters as the build script.
            chunks = chunk_text(
                article.text,
                source=article.title,
                chunk_size=self._chunk_size,
                overlap=self._chunk_overlap,
                category=article.category.value if article.category else None,
            )

            if not chunks:
                return []

            self._pending[question_id] = _PendingEntry(
                article=article,
                chunks=chunks,
            )
            return chunks

    def confirm(self, question_id: str) -> None:
        """Confirm that the live-fetched article for *question_id* was helpful.

        Called by runner.py after ``outcome.correct is True``.  Embeds the
        pending chunks and appends them to the live FAISS + BM25 indices in
        memory so repeat questions benefit immediately.

        If ``question_id`` is not in the pending buffer (e.g. the question
        was answered from the offline index, or the live search found
        nothing), this is a silent no-op.

        Args:
            question_id: key supplied to the matching ``buffer()`` call.
        """
        with self._lock:
            entry = self._pending.pop(question_id, None)
            if entry is None:
                return

            # Guard against a race where confirm is called twice.
            if entry.article.title in self._indexed_titles:
                return

        # Embed outside the lock — CPU-bound but brief (~100 ms on CPU).
        # The game has already moved on; this runs after outcome receipt.
        try:
            texts = [c.text for c in entry.chunks]
            embeddings = self._embedder.encode_passage(texts)
        except Exception:  # noqa: BLE001
            # Embedding failure must not crash the runner.
            return

        with self._lock:
            # Double-check after re-acquiring (another thread might have
            # confirmed the same title via a different question_id).
            if entry.article.title in self._indexed_titles:
                return

            self._retriever.append_chunks(entry.chunks, embeddings)
            self._confirmed.append(entry.article)
            self._indexed_titles.add(entry.article.title)

    def discard(self, question_id: str) -> None:
        """Explicitly discard a pending entry (e.g. answer was wrong).

        runner.py may call this when ``outcome.correct is False`` to free
        memory immediately rather than waiting for session end.  Calling it
        is optional — un-confirmed entries are discarded automatically when
        the ``IndexGrower`` is garbage-collected.

        Args:
            question_id: key supplied to the matching ``buffer()`` call.
        """
        with self._lock:
            self._pending.pop(question_id, None)

    def flush(self) -> None:
        """Persist the grown index to disk.

        Writes:
          - Updated FAISS vectors + metadata JSONL (replaces the old files).
          - Updated BM25 sidecar JSONL (if the retriever has a BM25 index).
          - Appended raw corpus JSONL (new articles appended to the end).

        This is a full overwrite of the FAISS + BM25 files, which is safe
        because the in-memory index already contains all original chunks
        *plus* the new confirmed chunks.

        Call once at session end (after the game loop exits).  Safe to call
        even if no articles were confirmed (writes the unchanged index back).
        """
        with self._lock:
            confirmed_snapshot = list(self._confirmed)

        if not confirmed_snapshot:
            return  # nothing learned — skip the I/O entirely

        # 1. Persist FAISS + BM25 using the retriever's internal index objects.
        #    We reconstruct the manifest from the existing one (if present) so
        #    we don't lose provenance information from the original build.
        faiss_index  = self._retriever._index
        bm25_index   = self._retriever._bm25

        existing_manifest = faiss_index.manifest or {}
        # Update the chunk count; preserve all other manifest fields.
        updated_manifest = dict(existing_manifest)
        updated_manifest["n_chunks"] = faiss_index.n_chunks

        try:
            faiss_index.save(self._index_path, manifest=updated_manifest)
        except Exception as exc:  # noqa: BLE001
            print(f"[IndexGrower] WARNING: failed to persist FAISS index: {exc}")
            return

        if bm25_index is not None:
            try:
                bm25_index.save(self._index_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[IndexGrower] WARNING: failed to persist BM25 index: {exc}")

        # 2. Append new articles to corpus.jsonl so a future full rebuild
        #    picks them up without needing another live-search pass.
        try:
            self._append_to_corpus(confirmed_snapshot)
        except Exception as exc:  # noqa: BLE001
            print(f"[IndexGrower] WARNING: failed to append to corpus.jsonl: {exc}")

        n = len(confirmed_snapshot)
        print(
            f"[IndexGrower] Flushed {n} new article(s) to "
            f"{self._index_path}.{{faiss,jsonl}}"
            + (f" + {self._index_path}.bm25.jsonl" if bm25_index is not None else "")
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def n_pending(self) -> int:
        """Number of buffered (not yet confirmed) articles."""
        with self._lock:
            return len(self._pending)

    @property
    def n_learned(self) -> int:
        """Number of articles confirmed and appended this session."""
        with self._lock:
            return len(self._confirmed)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _collect_existing_titles(self) -> set[str]:
        """Scan the live FAISS index for already-indexed article titles.

        Builds a set of ``Chunk.source`` values so ``buffer()`` can do O(1)
        dedup without iterating all chunks every call.
        """
        return {c.source for c in self._retriever._index._chunks}

    def _append_to_corpus(self, articles: list[Article]) -> None:
        """Append ``articles`` to the raw corpus JSONL.

        Reads the existing corpus to check for duplicates (by title), then
        appends only the new ones.  Uses ``save_raw_corpus()`` append mode
        so the file grows without rewriting existing content.
        """
        # Load existing titles from corpus for dedup.
        existing_titles: set[str] = set()
        if self._corpus_path.is_file():
            try:
                for a in load_raw_corpus(self._corpus_path):
                    existing_titles.add(a.title)
            except Exception:  # noqa: BLE001
                pass  # corpus may be absent or corrupt — just append

        new_articles = [a for a in articles if a.title not in existing_titles]
        if not new_articles:
            return

        # Append mode — open the file and write new lines.
        import json
        self._corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with self._corpus_path.open("a", encoding="utf-8") as f:
            for a in new_articles:
                f.write(json.dumps({
                    "title":    a.title,
                    "text":     a.text,
                    "category": a.category.value if a.category else "history",
                    "url":      a.url,
                }, ensure_ascii=False) + "\n")
