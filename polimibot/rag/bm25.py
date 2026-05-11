"""BM25 sparse retrieval — the lexical complement to dense FAISS retrieval.

Why bother when we have dense embeddings? Trivia heavily rewards lexical
exact match on proper nouns ("Pythagorean theorem", "Schindler's List",
"49 BC"). Dense bi-encoders compress meaning across a whole sentence
and often lose the precise-string signal. BM25 weights each token by
its corpus rarity (IDF) and finds chunks containing the rare query
tokens directly — usually the best signal for entity-style questions.

Used in concert with dense via RRF (reciprocal rank fusion) — see
``polimibot.rag.fusion`` and the ``hybrid=True`` toggle on
``Retriever.retrieve``.

Implementation: pure-Python BM25Okapi over a positional inverted index.
No new dependencies. Per-doc postings store *positions* (not just term
frequencies) so the search step can add a proximity bonus when two
query tokens co-occur near each other — captures "Pythagorean
theorem"-style phrase signal without indexing bigrams. The base BM25
formula is unchanged:

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
from typing import List, Optional, Sequence, Tuple

from .chunker import Chunk


# Bumped whenever the on-disk format or the tokenisation pipeline changes.
# v2 = positional postings + stopword removal (was v1: tf-only postings).
BM25_VERSION = 2


@dataclass(frozen=True)
class BM25Spec:
    """BM25 hyperparameters.

    k1 controls term-frequency saturation (higher = TF matters more).
    b controls length normalisation (0 = none, 1 = full).
    proximity_alpha is the weight on the phrase-proximity bonus added on top
    of the base BM25 score for docs where two query tokens co-occur within
    proximity_window positions. Set proximity_alpha=0 to disable proximity
    scoring entirely. Defaults are robust across corpora.
    """
    k1: float = 1.5
    b: float = 0.75
    proximity_alpha: float = 0.25
    proximity_window: int = 10


# Token regex: lowercase, ASCII + Unicode word chars, drop punctuation.
# Conservative — no stemming. Stemming hurts proper-noun retrieval
# (would map "Caesars" → "caesar", merging distinct entity senses).
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


# A conservative English stoplist. These tokens have near-zero discriminative
# power for trivia retrieval and inflate posting-list scan cost. Negation
# words ("not", "no", "never") and comparatives ("than", "more", "less") are
# DELIBERATELY kept — they carry semantic weight in MCQ questions.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "was", "were", "will", "with",
})


def tokenize(text: str, *, drop_stopwords: bool = True) -> List[str]:
    """Lowercase word tokens, optionally stopword-filtered. Pure function."""
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in toks if t not in _STOPWORDS]
    return toks


def _build_positional_postings(
    doc_tokens: List[List[str]],
) -> dict[str, dict[int, List[int]]]:
    """token -> {doc_id: [positions]} from the tokenised corpus."""
    postings: dict[str, dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for doc_id, toks in enumerate(doc_tokens):
        for pos, tok in enumerate(toks):
            postings[tok][doc_id].append(pos)
    return {t: dict(p) for t, p in postings.items()}


def _idf_table(
    postings: dict[str, dict[int, List[int]]],
    n_docs: int,
) -> dict[str, float]:
    return {
        tok: math.log((n_docs - len(posts) + 0.5) / (len(posts) + 0.5) + 1.0)
        for tok, posts in postings.items()
    }


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
            self._doc_tokens: List[List[str]] = _precomputed["doc_tokens"]
            self._doc_len:    List[int]       = _precomputed["doc_len"]
            self._avgdl:      float           = _precomputed["avgdl"]
            self._postings:   dict[str, dict[int, List[int]]] = _precomputed["postings"]
            self._idf:        dict[str, float]                = _precomputed["idf"]
            return

        # Build from scratch.
        self._doc_tokens = [tokenize(c.text) for c in self._chunks]
        self._doc_len = [len(t) for t in self._doc_tokens]
        n_docs = len(self._chunks)
        self._avgdl = (sum(self._doc_len) / n_docs) if n_docs else 0.0
        self._postings = _build_positional_postings(self._doc_tokens)
        self._idf = _idf_table(self._postings, n_docs)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write ``{path}.bm25.jsonl`` (header + one row per chunk).

        Tokens are stored so load doesn't have to re-tokenise (cheap, and
        guarantees consistency across versions of ``tokenize``). Positions
        are recomputed at load time from the saved tokens.
        """
        out = path.with_suffix(".bm25.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            header = {
                "kind": "bm25_header",
                "k1":   self.spec.k1,
                "b":    self.spec.b,
                "proximity_alpha":  self.spec.proximity_alpha,
                "proximity_window": self.spec.proximity_window,
                "n_docs": len(self._chunks),
                "avgdl":  self._avgdl,
                "version": BM25_VERSION,
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
        """Load ``{path}.bm25.jsonl``. Reconstructs positional postings + IDF.

        Refuses to load sidecars from earlier format versions (tokenisation
        differed; loading them would silently corrupt scores).
        """
        in_path = path.with_suffix(".bm25.jsonl")
        chunks: list[Chunk] = []
        doc_tokens: list[list[str]] = []
        doc_len: list[int] = []
        spec = BM25Spec()
        with in_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                if i == 0 and rec.get("kind") == "bm25_header":
                    version = rec.get("version", 1)
                    if version != BM25_VERSION:
                        raise ValueError(
                            f"BM25 sidecar at {in_path} is version {version}, "
                            f"expected {BM25_VERSION}. Rebuild with "
                            f"`python scripts/build_rag_index.py --refetch` "
                            f"(or omit --refetch and just re-run the BM25 step)."
                        )
                    spec = BM25Spec(
                        k1=rec.get("k1", 1.5),
                        b=rec.get("b", 0.75),
                        proximity_alpha=rec.get("proximity_alpha", 0.25),
                        proximity_window=rec.get("proximity_window", 10),
                    )
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
        postings = _build_positional_postings(doc_tokens)
        idf = _idf_table(postings, n_docs)

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
        if not q_tokens:
            return []
        k1, b = self.spec.k1, self.spec.b
        avgdl = self._avgdl or 1.0

        # Base BM25. Score only docs that contain at least one query token —
        # iterating per-token postings is cheap; iterating every doc is not.
        scores: dict[int, float] = defaultdict(float)
        # Track per-doc which query tokens hit and their positions, for the
        # proximity step. Map: doc_id -> list of (query_token_index, positions).
        doc_hits: dict[int, list[tuple[int, List[int]]]] = defaultdict(list)
        for qi, tok in enumerate(q_tokens):
            posts = self._postings.get(tok)
            if not posts:
                continue
            idf = self._idf.get(tok, 0.0)
            for doc_id, positions in posts.items():
                tf = len(positions)
                dl = self._doc_len[doc_id]
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc_id] += idf * tf * (k1 + 1) / denom
                doc_hits[doc_id].append((qi, positions))

        # Proximity bonus. For every doc with 2+ distinct query tokens hit,
        # find the closest co-occurrence between any pair and add
        # alpha * (idf_a + idf_b) / (1 + gap) when gap <= proximity_window.
        # Keeps cost O(pairs × positions) — tiny on Wikipedia chunks.
        alpha = self.spec.proximity_alpha
        window = self.spec.proximity_window
        if alpha > 0 and window > 0:
            for doc_id, hits in doc_hits.items():
                if len(hits) < 2:
                    continue
                for i in range(len(hits)):
                    qi_a, pos_a = hits[i]
                    idf_a = self._idf.get(q_tokens[qi_a], 0.0)
                    for j in range(i + 1, len(hits)):
                        qi_b, pos_b = hits[j]
                        if qi_a == qi_b:
                            continue  # same query token, skip
                        gap = _min_gap(pos_a, pos_b)
                        if gap <= window:
                            idf_b = self._idf.get(q_tokens[qi_b], 0.0)
                            scores[doc_id] += alpha * (idf_a + idf_b) / (1 + gap)

        if not scores:
            return []

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


def _min_gap(positions_a: List[int], positions_b: List[int]) -> int:
    """Minimum |pa - pb| over the cartesian product. Both lists are sorted
    (positions come from enumerate(tokens)). Two-pointer scan is O(n+m).
    """
    if not positions_a or not positions_b:
        return 10**9
    i = j = 0
    best = 10**9
    while i < len(positions_a) and j < len(positions_b):
        a, bp = positions_a[i], positions_b[j]
        gap = abs(a - bp)
        if gap < best:
            best = gap
            if best == 0:
                return 0
        if a < bp:
            i += 1
        else:
            j += 1
    return best
