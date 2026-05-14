"""Wrong set: incorrectly-answered questions mined from run logs.

Symmetrical companion to gold_set.py.  Where the gold set collects questions
the bot answered *correctly* (confirmed ground truth), the wrong set collects
questions the bot answered *incorrectly* — complete with which answer was
chosen and, when recoverable from other runs, what the correct answer is.

Use cases:
  • Error analysis by category / level / answer position.
  • Targeted re-evaluation after strategy changes.
  • Building hard-negative training examples.

Two layers of API:

  WrongItem           : frozen record of one incorrectly-answered question.
  load_wrong_set      : free function returning list[WrongItem].
  harvest_wrong_set   : free function building a list from run logs.
  save_wrong_set      : free function persisting a list to JSONL.

  WrongSet            : chainable view over a list[WrongItem].  Filter by
                        category / level / competition, sample, split,
                        balance per-level / per-category, print stats.
                        Iterable, has __len__, so it drops straight into
                        analysis loops.

Example:
    full    = WrongSet.load(PATHS.eval_dir / 'wrong_set.jsonl')
    full.print_stats()
    maths   = full.filter_category(Category.MATHS)
    hard    = full.filter_level(min_level=11)
    known   = full.filter(lambda w: w.correct_index >= 0)  # correct answer known
    train, test = full.split(0.8, seed=42)
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from ..config import CATEGORIES, Category
from ..logging_utils import load_jsonl


@dataclass(frozen=True)
class WrongItem:
    """One question that was answered incorrectly.  Immutable by design.

    Attributes:
        question_text:  Full text of the question.
        options:        Tuple of 4 answer strings, original order preserved.
        wrong_index:    0-based index of the answer that was (incorrectly) chosen.
        correct_index:  0-based index of the correct answer, if known from other
                        runs or from the same session; -1 when unknown.
        competition_id: Server competition id.
        level:          Question level (1–15).
        category:       Derived from competition_id; None when unmapped.
        source_run:     Name of the first run file in which this wrong answer
                        was recorded.
    """
    question_text: str
    options: tuple[str, ...]     # always 4, original order preserved
    wrong_index: int             # 0-based: 0=A, 1=B, 2=C, 3=D
    correct_index: int           # 0-based; -1 = unknown
    competition_id: int
    level: int
    category: Optional[Category] = None
    source_run: str = ""


def _category_for(competition_id: int) -> Optional[Category]:
    info = CATEGORIES.get(competition_id)
    return info.category if info else None


def harvest_wrong_set(runs_dir: Path) -> list[WrongItem]:
    """Mine all *.jsonl run files in *runs_dir* → WrongItems.

    Algorithm:
      1. Collect every question record where ``correct=False``.
         Deduplicate by ``(competition_id, question_text, wrong_index)`` —
         the same mistake from different sessions counts once.
      2. Cross-reference against all records in the same logs to recover
         ``correct_index`` when possible:
           a. Direct: another record for the same question has ``correct=True``.
           b. Elimination: 3 of 4 options have been seen wrong → 4th is correct.
    """
    # key: (competition_id, question_text)
    # confirmed_correct[key] = correct_index (from direct or elimination pass)
    confirmed_correct: dict[str, int] = {}
    wrong_indices_seen: dict[str, set[int]] = {}   # for elimination
    meta: dict[str, dict] = {}                     # raw metadata per question

    # wrong_records: list of (key, wrong_index, source_run)
    # We defer building WrongItem until we've done the full cross-reference pass.
    wrong_records: list[tuple[str, int, str]] = []
    wrong_record_set: set[tuple[str, int]] = set()   # dedup by (key, wrong_idx)

    for fp in sorted(runs_dir.glob("*.jsonl")):
        for rec in load_jsonl(fp):
            if rec.get("run_kind") != "question":
                continue
            cid = rec.get("competition_id", -1)
            text = rec.get("question_text", "")
            if not text:
                continue
            key = f"{cid}|{text}"
            idx = rec.get("chosen_index", -1)
            is_correct = rec.get("correct")

            # Cache metadata on first sight of this question
            if key not in meta:
                meta[key] = {
                    "options": tuple(rec.get("options", [])),
                    "competition_id": cid,
                    "level": rec.get("level", 0),
                    "source_run": fp.name,
                }

            if is_correct is True and key not in confirmed_correct:
                confirmed_correct[key] = idx

            if is_correct is False and 0 <= idx < 4:
                wrong_indices_seen.setdefault(key, set()).add(idx)
                record_key = (key, idx)
                if record_key not in wrong_record_set:
                    wrong_record_set.add(record_key)
                    wrong_records.append((key, idx, fp.name))

    # Elimination pass: 3 wrong → 4th must be correct
    for key, wrong_set in wrong_indices_seen.items():
        if key in confirmed_correct:
            continue
        m = meta.get(key, {})
        if len(m.get("options", ())) != 4:
            continue
        remaining = set(range(4)) - wrong_set
        if len(remaining) == 1:
            confirmed_correct[key] = next(iter(remaining))

    # Build WrongItem list (deduplicated by question+wrong_index)
    items: list[WrongItem] = []
    for key, wrong_idx, source_run in wrong_records:
        m = meta.get(key, {})
        correct_idx = confirmed_correct.get(key, -1)
        items.append(WrongItem(
            question_text=key.split("|", 1)[1],
            options=m.get("options", ()),
            wrong_index=wrong_idx,
            correct_index=correct_idx,
            competition_id=m.get("competition_id", -1),
            level=m.get("level", 0),
            category=_category_for(m.get("competition_id", -1)),
            source_run=source_run,
        ))

    return items


def save_wrong_set(items: list[WrongItem], path: Path) -> None:
    """Write wrong items to a JSONL file.  Overwrites existing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            d = asdict(item)
            # Category enum → string for JSON; None stays None
            if d["category"] is not None:
                d["category"] = d["category"]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} wrong items → {path}")


def load_wrong_set(path: Path) -> list[WrongItem]:
    """Load a previously saved wrong set from JSONL."""
    items = []
    for rec in load_jsonl(path):
        cat = rec.get("category")
        items.append(WrongItem(
            question_text=rec["question_text"],
            options=tuple(rec["options"]),
            wrong_index=rec["wrong_index"],
            correct_index=rec.get("correct_index", -1),
            competition_id=rec["competition_id"],
            level=rec["level"],
            category=Category(cat) if cat else None,
            source_run=rec.get("source_run", ""),
        ))
    return items


# ── WrongSet — chainable view over a list[WrongItem] ───────────────────────

# Accept both Category enums and their string values in *_filter args.
CategoryLike = Union[Category, str]


def _coerce_category(c: CategoryLike) -> Category:
    """Allow callers to pass strings ('maths') or enums (Category.MATHS)."""
    return c if isinstance(c, Category) else Category(c)


def _identity_key(w: WrongItem) -> tuple[int, str, int]:
    """Stable dedup key: (competition_id, question_text, wrong_index)."""
    return (w.competition_id, w.question_text, w.wrong_index)


class WrongSet:
    """Chainable view over a list[WrongItem].

    Every filter / sampler / splitter returns a NEW WrongSet — the original
    is never mutated, so you can branch analyses freely:

        full        = WrongSet.load(path)
        maths       = full.filter_category(Category.MATHS)
        hard_maths  = maths.filter_level(min_level=11)
        known       = full.filter(lambda w: w.correct_index >= 0)
        pilot       = full.take_per_level(3, seed=0)
        train, test = full.split(0.8, seed=42)

    Iterable and ``__len__``-able.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[WrongItem] = ()) -> None:
        self._items: list[WrongItem] = list(items)

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "WrongSet":
        """Load from a wrong-set JSONL written by ``save_wrong_set``."""
        return cls(load_wrong_set(path))

    @classmethod
    def harvest(cls, runs_dir: Path) -> "WrongSet":
        """Mine run logs into a fresh WrongSet."""
        return cls(harvest_wrong_set(runs_dir))

    # ── Iterable / list-like ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, key):
        result = self._items[key]
        return WrongSet(result) if isinstance(key, slice) else result

    def __repr__(self) -> str:
        return f"WrongSet(n={len(self._items)})"

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def items(self) -> list[WrongItem]:
        """Defensive copy of the underlying list."""
        return list(self._items)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write this view's items to JSONL (overwrites existing)."""
        save_wrong_set(self._items, path)

    # ── Filters (each returns a new WrongSet — chainable) ─────────────────

    def filter(self, predicate: Callable[[WrongItem], bool]) -> "WrongSet":
        """Arbitrary predicate filter."""
        return WrongSet(w for w in self._items if predicate(w))

    def filter_category(self, *categories: CategoryLike) -> "WrongSet":
        """Keep only items whose category is in ``categories``.

        Accepts either Category enums or their string values:
            ws.filter_category(Category.MATHS)
            ws.filter_category('maths', 'science')
        """
        keep = {_coerce_category(c) for c in categories}
        return WrongSet(w for w in self._items if w.category in keep)

    def filter_level(self, *, min_level: int = 1, max_level: int = 15) -> "WrongSet":
        """Keep only items with ``min_level <= level <= max_level``."""
        return WrongSet(
            w for w in self._items if min_level <= w.level <= max_level
        )

    def filter_competition(self, *competition_ids: int) -> "WrongSet":
        """Keep only items from the given competition ids."""
        keep = set(competition_ids)
        return WrongSet(w for w in self._items if w.competition_id in keep)

    def filter_known_correct(self) -> "WrongSet":
        """Keep only items where the correct answer was recovered (correct_index >= 0)."""
        return WrongSet(w for w in self._items if w.correct_index >= 0)

    # ── Sampling (each returns a new WrongSet) ─────────────────────────────

    def take(self, n: int) -> "WrongSet":
        """Deterministic first-n slice (preserves order)."""
        return WrongSet(self._items[:n])

    def shuffle(self, *, seed: int = 0) -> "WrongSet":
        """Return a shuffled copy. Reproducible with the seed."""
        rng = random.Random(seed)
        out = list(self._items)
        rng.shuffle(out)
        return WrongSet(out)

    def sample(self, n: int, *, seed: int = 0) -> "WrongSet":
        """Uniform random n-item sample without replacement.

        Returns all items unchanged if ``n >= len(self)``.
        """
        if n >= len(self._items):
            return WrongSet(self._items)
        rng = random.Random(seed)
        return WrongSet(rng.sample(self._items, n))

    def take_per_level(self, n: int, *, seed: int = 0) -> "WrongSet":
        """At most ``n`` items per level. Random within each level."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        rng = random.Random(seed)
        bucket: dict[int, list[WrongItem]] = defaultdict(list)
        for w in self._items:
            bucket[w.level].append(w)
        out: list[WrongItem] = []
        for level in sorted(bucket):
            items = bucket[level]
            out.extend(items if len(items) <= n else rng.sample(items, n))
        return WrongSet(out)

    def take_per_category(self, n: int, *, seed: int = 0) -> "WrongSet":
        """At most ``n`` items per category. Random within each category."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        rng = random.Random(seed)
        bucket: dict[Optional[Category], list[WrongItem]] = defaultdict(list)
        for w in self._items:
            bucket[w.category].append(w)
        out: list[WrongItem] = []
        for cat in sorted(bucket, key=lambda c: c.value if c else "~unknown"):
            items = bucket[cat]
            out.extend(items if len(items) <= n else rng.sample(items, n))
        return WrongSet(out)

    # ── Splits ───────────────────────────────────────────────────────────

    def split(
        self, fraction: float, *, seed: int = 0,
    ) -> tuple["WrongSet", "WrongSet"]:
        """Shuffled train/test split.

        First returned WrongSet gets ``fraction`` of the items; second gets
        the rest. The shuffle is seeded — same seed → same split.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        rng = random.Random(seed)
        shuffled = list(self._items)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * fraction)
        return WrongSet(shuffled[:cut]), WrongSet(shuffled[cut:])

    def by_category(self) -> dict[Category, "WrongSet"]:
        """Group items by category. Items with ``category=None`` are skipped."""
        groups: dict[Category, list[WrongItem]] = defaultdict(list)
        for w in self._items:
            if w.category is not None:
                groups[w.category].append(w)
        return {cat: WrongSet(items) for cat, items in groups.items()}

    def by_level(self) -> dict[int, "WrongSet"]:
        """Group items by level (1..15). Returns dict sorted by level."""
        groups: dict[int, list[WrongItem]] = defaultdict(list)
        for w in self._items:
            groups[w.level].append(w)
        return {lvl: WrongSet(groups[lvl]) for lvl in sorted(groups)}

    # ── Stats ────────────────────────────────────────────────────────────

    def counts_by_category(self) -> dict[str, int]:
        """{'maths': N, 'science': N, ...}. ``'unknown'`` for items with None."""
        return dict(Counter(
            w.category.value if w.category else "unknown" for w in self._items
        ))

    def counts_by_level(self) -> dict[int, int]:
        """{1: N, 2: N, ...}. Sorted by level."""
        return dict(sorted(Counter(w.level for w in self._items).items()))

    def counts(self) -> dict[str, dict[int, int]]:
        """Cross-tab: {category: {level: count}}."""
        nested: dict[str, Counter] = defaultdict(Counter)
        for w in self._items:
            key = w.category.value if w.category else "unknown"
            nested[key][w.level] += 1
        return {cat: dict(sorted(c.items())) for cat, c in nested.items()}

    def correct_recovery_rate(self) -> float:
        """Fraction of items where the correct answer is known (correct_index >= 0)."""
        if not self._items:
            return 0.0
        return sum(1 for w in self._items if w.correct_index >= 0) / len(self._items)

    def print_stats(self) -> None:
        """Pretty-print a category × level counts matrix plus recovery rate."""
        if not self._items:
            print("WrongSet is empty.")
            return
        matrix = self.counts()
        levels = sorted({lvl for c in matrix.values() for lvl in c})
        cat_w = max((len(k) for k in matrix.keys()), default=8) + 2

        header = (
            f"{'category'.ljust(cat_w)}"
            + " ".join(f"L{l:>2}" for l in levels)
            + "  total"
        )
        print(header)
        print("-" * len(header))
        for cat in sorted(matrix.keys()):
            row = matrix[cat]
            cells = " ".join(f"{row.get(l, 0):>3}" for l in levels)
            total = sum(row.values())
            print(f"{cat.ljust(cat_w)}{cells}  {total:>5}")
        # Totals row
        totals = [sum(matrix[c].get(l, 0) for c in matrix) for l in levels]
        grand = sum(totals)
        print("-" * len(header))
        print(
            f"{'total'.ljust(cat_w)}"
            + " ".join(f"{t:>3}" for t in totals)
            + f"  {grand:>5}"
        )
        n_known = sum(1 for w in self._items if w.correct_index >= 0)
        print(f"\nCorrect answer known: {n_known}/{grand} "
              f"({n_known / grand:.1%})")

    # ── Set ops (by question+wrong_index identity) ─────────────────────────

    def __add__(self, other: "WrongSet") -> "WrongSet":
        """Union by (competition_id, question_text, wrong_index). Right side wins ties."""
        seen: dict[tuple[int, str, int], WrongItem] = {
            _identity_key(w): w for w in self._items
        }
        for w in other:
            seen[_identity_key(w)] = w
        return WrongSet(seen.values())

    def __sub__(self, other: "WrongSet") -> "WrongSet":
        """Difference by (competition_id, question_text, wrong_index)."""
        exclude = {_identity_key(w) for w in other}
        return WrongSet(w for w in self._items if _identity_key(w) not in exclude)
