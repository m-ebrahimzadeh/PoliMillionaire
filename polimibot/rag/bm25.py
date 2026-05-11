"""BM25 sparse retrieval — the lexical complement to dense FAISS retrieval.

Why bother when we have dense embeddings? Trivia heavily rewards lexical
exact match on proper nouns ("Pythagorean theorem", "Schindler's List",
"49 BC"). Dense bi-encoders compress meaning across a whole sentence
and often lose the precise-string signal. BM25 weights each token by
its corpus rarity (IDF) and finds chunks containing the rare query
tokens directly — usually the best signal for entity-style questions.

Used in concert with dense via RRF (reciprocal rank fusion) — see
``polimibot.rag.fusion`` (next commit) and the ``hybrid=True`` toggle
on ``Retriever.retrieve`` (commit after that).

Implementation: pure-Python BM25Okapi over an inverted index. No new
dependencies; ~30 ms per query on a ~25k-chunk corpus, which is fine
sequentially. The formula:

    score(q, d) = Σ_{t ∈ q} IDF(t) · TF(t, d) · (k1 + 1)
                              ───────────────────────────────
                              TF(t, d) + k1 · (1 − b + b · |d| / avgdl)

    IDF(t) = log( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )

with the standard parameters k1 = 1.5, b = 0.75.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .chunker import Chunk


@dataclass(frozen=True)
class BM25Spec:
    """BM25 hyperparameters.

    k1 controls term-frequency saturation (higher = TF matters more).
    b controls length normalisation (0 = none, 1 = full).
    Defaults are the standard Okapi values, robust across corpora.
    """
    k1: float = 1.5
    b: float = 0.75


# Token regex: lowercase, ASCII + Unicode word chars, drop punctuation.
# Conservative — no stemming. Stemming hurts proper-noun retrieval
# (would map "Caesars" → "caesar", merging distinct entity senses).
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Pure-function tokeniser. Lowercased word tokens, no stemming."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Sparse lexical index over a chunk corpus.

    Build:
        idx = BM25Index.build(chunks)
        idx.save(Path("data/cache/knowledge"))  # writes {stem}.bm25.jsonl

    Load:
        idx = BM25Index.load(Path("data/cache/knowledge"))
        hits = idx.search("Caesar Rubicon", k=5, category="history")

    Persistence shares the FAISS stem so a single ``Retriever.from_saved``
    can pick up both indices for hybrid retrieval.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        spec: Optional[BM25Spec] = None,
        *,
        _precomputed: Optional[dict] = None,
    ) -> None:
        self.spec = spec or BM25Spec()
        self._chunks: List[Chunk] = list(chunks)

        if _precomputed is not None:
            # Loaded from disk — use the precomputed state verbatim.
            self._doc_tokens: List[List[str]] = _precomputed["doc_tokens"]
            self._doc_len:    List[int]       = _precomputed["doc_len"]
            self._avgdl:      float           = _precomputed["avgdl"]
            self._postings:   dict[str, dict[int, int]] = _precomputed["postings"]
            self._idf:        dict[str, float]          = _precomputed["idf"]
            return

        # Build from scratch.
        self._doc_tokens = [tokenize(c.text) for c in self._chunks]
        self._doc_len = [len(t) for t in self._doc_tokens]
        n_docs = len(self._chunks)
        self._avgdl = (sum(self._doc_len) / n_docs) if n_docs else 0.0

        # Inverted index: token -> {doc_id: term_frequency}
        self._postings = defaultdict(dict)
        for doc_id, toks in enumerate(self._doc_tokens):
            for tok, tf in Counter(toks).items():
                self._postings[tok][doc_id] = tf
        # Freeze to plain dict for cleaner serialisation.
        self._postings = {t: dict(posts) for t, posts in self._postings.items()}

        # IDF table (BM25-flavoured; "+1" inside the log prevents negatives).
        self._idf = {
            tok: math.log((n_docs - len(posts) + 0.5) / (len(posts) + 0.5) + 1.0)
            for tok, posts in self._postings.items()
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write ``{path}.bm25.jsonl``. One JSON object per line —

            line 0: header with spec + corpus stats
            line 1..N: per-chunk row (text + source + chunk_id + category + tokens)

        The token list IS stored so load doesn't have to retokenise. Cheap
        on disk (~2× the text size for English Wikipedia) and means
        consistency across runs even if ``tokenize`` ever changes.
        """
        out = path.with_suffix(".bm25.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            header = {
                "kind": "bm25_header",
                "k1":   self.spec.k1,
                "b":    self.spec.b,
                "n_docs": len(self._chunks),
                "avgdl":  self._avgdl,
                "version": 1,
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for chunk, tokens, dl in zip(self._chunks, self._doc_tokens, self._doc_len):
                row = {
                    "text":     chunk.text,
                    "source":   chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "category": chunk.category,
                    "tokens":   tokens,
                    "doc_len":  dl,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved BM25 index ({len(self._chunks)} chunks) → {out}")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load ``{path}.bm25.jsonl``. Reconstructs postings + IDF in-memory."""
        in_path = path.with_suffix(".bm25.jsonl")
        chunks: list[Chunk] = []
        doc_tokens: list[list[str]] = []
        doc_len: list[int] = []
        spec = BM25Spec()
        with in_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                if i == 0 and rec.get("kind") == "bm25_header":
                    spec = BM25Spec(k1=rec.get("k1", 1.5), b=rec.get("b", 0.75))
                    continue
                chunks.append(Chunk(
                    text=rec["text"],
                    source=rec["source"],
                    chunk_id=rec["chunk_id"],
                    category=rec.get("category"),
                ))
                doc_tokens.append(rec["tokens"])
                doc_len.append(rec["doc_len"])

        n_docs = len(chunks)
        avgdl = (sum(doc_len) / n_docs) if n_docs else 0.0

        postings: dict[str, dict[int, int]] = defaultdict(dict)
        for doc_id, toks in enumerate(doc_tokens):
            for tok, tf in Counter(toks).items():
                postings[tok][doc_id] = tf
        postings = {t: dict(p) for t, p in postings.items()}

        idf = {
            tok: math.log((n_docs - len(p) + 0.5) / (len(p) + 0.5) + 1.0)
            for tok, p in postings.items()
        }

        return cls(
            chunks,
            spec=spec,
            _precomputed={
                "doc_tokens": doc_tokens,
                "doc_len":    doc_len,
                "avgdl":      avgdl,
                "postings":   postings,
                "idf":        idf,
            },
        )

    # ── Search ───────────────────────────────────────────────────────────

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        category: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Top-k BM25-scored chunks. Optional category mask.

        Returns ``(Chunk, score)`` pairs in descending BM25 score order.
        BM25 scores are positive unbounded — typical magnitudes 0–30 on
        Wikipedia chunks; not comparable to cosine.
        """
        q_tokens = tokenize(query)
        # Score only docs that contain at least one query token — saves
        # huge time vs. scoring every doc against every (likely-absent) term.
        scores: dict[int, float] = defaultdict(float)
        k1, b = self.spec.k1, self.spec.b
        avgdl = self._avgdl or 1.0

        for tok in q_tokens:
            posts = self._postings.get(tok)
            if not posts:
                continue
            idf = self._idf.get(tok, 0.0)
            for doc_id, tf in posts.items():
                dl = self._doc_len[doc_id]
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc_id] += idf * tf * (k1 + 1) / denom

        if not scores:
            return []

        # Sort by score descending, take top-k, apply category filter if any.
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out: List[Tuple[Chunk, float]] = []
        for doc_id, score in ranked:
            chunk = self._chunks[doc_id]
            if category is not None and chunk.category != category:
                continue
            out.append((chunk, float(score)))
            if len(out) >= k:
                break
        return out
