"""Gold set: ground-truth questions mined from run logs.

The game server reveals correctness only after submission.
We recover the correct answer index two ways:
  1. Direct: chosen_index where correct=True
  2. Elimination: if all-but-one options have been seen wrong across runs

Save once after your baseline runs. Never re-harvest mid-experiment —
that would silently grow your test set and invalidate comparisons.

Two layers of API:

  GoldItem            : the frozen record of one labelled question.
  load_gold_set       : free function returning list[GoldItem] (compat).
  harvest_gold_set    : free function building a list from run logs.
  save_gold_set       : free function persisting a list to JSONL.

  GoldSet             : chainable view over a list[GoldItem]. Filter by
                        category / level / competition, sample, split,
                        balance per-level / per-category, print stats.
                        Iterable, has __len__, so it drops straight into
                        ``evaluate_strategy``.

Example:
    full   = GoldSet.load(PATHS.eval_dir / 'gold_set.jsonl')
    full.print_stats()
    maths  = full.filter_category(Category.MATHS)
    pilot  = full.take_per_level(3, seed=0)
    train, test = full.split(0.8, seed=42)
    report = evaluate_strategy(strategy, pilot)
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
class GoldItem:
    """One question with a confirmed correct answer. Immutable by design."""
    question_text: str
    options: tuple[str, ...]     # always 4, original order preserved
    correct_index: int           # 0-based: 0=A, 1=B, 2=C, 3=D
    competition_id: int
    level: int
    category: Optional[Category] = None
    source_run: str = ""         # first run file that confirmed this label


def _category_for(competition_id: int) -> Optional[Category]:
    info = CATEGORIES.get(competition_id)
    return info.category if info else None


def _model_from_manifest(manifest: dict) -> str:
    extra = manifest.get("extra", {})
    # First check standard keys
    model = extra.get("model_id") or extra.get("model")
    if model:
        return str(model).lower()
    # Fallback: extract model from strategy string like "rag[model|...]"
    strategy = extra.get("strategy", "")
    if strategy and "[" in strategy:
        # Extract text between [ and first |
        start = strategy.index("[") + 1
        end = strategy.index("|", start) if "|" in strategy[start:] else len(strategy)
        model = strategy[start:end]
        if model:
            return model.lower()
    return ""


def _build_run_filter(
    exclude_models: list[str] | None,
    include_models: list[str] | None,
    run_filter: Callable[[dict], bool] | None,
) -> Callable[[dict], bool] | None:
    checks: list[Callable[[dict], bool]] = []
    if exclude_models:
        ex = [m.lower() for m in exclude_models]
        checks.append(lambda m, _ex=ex: not any(s in _model_from_manifest(m) for s in _ex))
    if include_models:
        inc = [m.lower() for m in include_models]
        checks.append(lambda m, _inc=inc: any(s in _model_from_manifest(m) for s in _inc))
    if run_filter:
        checks.append(run_filter)
    if not checks:
        return None
    return lambda m: all(c(m) for c in checks)


def harvest_gold_set(
    runs_dir: Path,
    *,
    exclude_models: list[str] | None = None,
    include_models: list[str] | None = None,
    run_filter: Callable[[dict], bool] | None = None,
) -> list[GoldItem]:
    """Mine *.jsonl run files in runs_dir → GoldItems.

    Deduplicates by (competition_id, question_text).
    A question is included only when the correct index can be confirmed.

    exclude_models / include_models: substring-match against the model slug
    in the manifest ``extra`` dict (checks 'model_id' then 'model' key).
    run_filter: arbitrary predicate on the raw manifest dict; ANDed with the
    model params.

    Examples::
        harvest_gold_set(runs_dir, exclude_models=["qwen"])
        harvest_gold_set(runs_dir, include_models=["gpt"])
        harvest_gold_set(runs_dir, run_filter=lambda m: m["extra"].get("seed") == 42)
    """
    _filter = _build_run_filter(exclude_models, include_models, run_filter)

    # confirmed[key] = GoldItem
    confirmed: dict[str, GoldItem] = {}
    # wrong_indices[key] = set of indices we've seen answered incorrectly
    wrong_indices: dict[str, set[int]] = {}
    # raw options/metadata for elimination pass
    meta: dict[str, dict] = {}

    for fp in sorted(runs_dir.glob("*.jsonl")):
        recs = iter(load_jsonl(fp))
        first = next(recs, None)
        if first is None:
            continue
        if first.get("run_kind") == "manifest":
            if _filter and not _filter(first):
                continue  # skip entire file
            question_iter: Iterable[dict] = recs
        else:
            # no manifest line — treat first record as a question record
            question_iter = (r for r in [first, *recs])
        for rec in question_iter:
            if rec.get("run_kind") != "question":
                continue
            cid = rec.get("competition_id", -1)
            text = rec.get("question_text", "")
            if not text:
                continue
            key = f"{cid}|{text}"
            idx = rec.get("chosen_index", -1)
            is_correct = rec.get("correct")

            # Cache raw metadata for elimination pass
            if key not in meta:
                meta[key] = {
                    "options": tuple(rec.get("options", [])),
                    "competition_id": cid,
                    "level": rec.get("level", 0),
                    "source_run": fp.name,
                }

            if is_correct is True and key not in confirmed:
                confirmed[key] = GoldItem(
                    question_text=text,
                    options=tuple(rec.get("options", [])),
                    correct_index=idx,
                    competition_id=cid,
                    level=rec.get("level", 0),
                    category=_category_for(cid),
                    source_run=fp.name,
                )
            elif is_correct is False and 0 <= idx < 4:
                wrong_indices.setdefault(key, set()).add(idx)

    # Elimination pass: question seen wrong on 3 of 4 options → 4th is gold
    for key, wrong in wrong_indices.items():
        if key in confirmed:
            continue  # already have direct confirmation
        m = meta.get(key, {})
        options = m.get("options", ())
        if len(options) != 4:
            continue
        remaining = set(range(4)) - wrong
        if len(remaining) == 1:
            correct_idx = next(iter(remaining))
            confirmed[key] = GoldItem(
                question_text=key.split("|", 1)[1],
                options=options,
                correct_index=correct_idx,
                competition_id=m["competition_id"],
                level=m["level"],
                category=_category_for(m["competition_id"]),
                source_run=m["source_run"],
            )

    return list(confirmed.values())


def save_gold_set(items: list[GoldItem], path: Path) -> None:
    """Write gold items to a JSONL file. Overwrites existing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            d = asdict(item)
            # Category enum → string for JSON; None stays None
            if d["category"] is not None:
                d["category"] = d["category"]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} gold items → {path}")


def load_gold_set(path: Path) -> list[GoldItem]:
    """Load previously saved gold set from JSONL."""
    items = []
    for rec in load_jsonl(path):
        cat = rec.get("category")
        items.append(GoldItem(
            question_text=rec["question_text"],
            options=tuple(rec["options"]),
            correct_index=rec["correct_index"],
            competition_id=rec["competition_id"],
            level=rec["level"],
            category=Category(cat) if cat else None,
            source_run=rec.get("source_run", ""),
        ))
    return items


# ── GoldSet — chainable view over a list[GoldItem] ─────────────────────────

# Accept both Category enums and their string values in *_filter args.
CategoryLike = Union[Category, str]


def _coerce_category(c: CategoryLike) -> Category:
    """Allow callers to pass strings ('maths') or enums (Category.MATHS)."""
    return c if isinstance(c, Category) else Category(c)


def _identity_key(g: GoldItem) -> tuple[int, str]:
    """Stable identifier for set-style ops. Dedups across re-harvests."""
    return (g.competition_id, g.question_text)


class GoldSet:
    """Chainable view over a list[GoldItem].

    Every filter / sampler / splitter returns a NEW GoldSet — the original
    is never mutated, so you can branch experiments freely:

        full           = GoldSet.load(path)
        maths          = full.filter_category(Category.MATHS)
        maths_easy     = maths.filter_level(max_level=5)
        balanced_pilot = full.take_per_level(3, seed=0)
        train, test    = full.split(0.8, seed=42)

    Drops into ``evaluate_strategy(strategy, gs)`` directly: GoldSet is
    iterable, has ``__len__``, and yields GoldItem instances.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[GoldItem] = ()) -> None:
        self._items: list[GoldItem] = list(items)

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "GoldSet":
        """Load from a gold-set JSONL written by ``save_gold_set``."""
        return cls(load_gold_set(path))

    @classmethod
    def harvest(cls, runs_dir: Path) -> "GoldSet":
        """Mine run logs into a fresh GoldSet."""
        return cls(harvest_gold_set(runs_dir))

    # ── Iterable / list-like ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, key):
        result = self._items[key]
        return GoldSet(result) if isinstance(key, slice) else result

    def __repr__(self) -> str:
        return f"GoldSet(n={len(self._items)})"

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def items(self) -> list[GoldItem]:
        """Defensive copy of the underlying list. Use this when you need
        ``list[GoldItem]`` rather than a GoldSet (e.g. for older APIs)."""
        return list(self._items)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write this view's items to JSONL (overwrites existing)."""
        save_gold_set(self._items, path)

    # ── Filters (each returns a new GoldSet — chainable) ─────────────────

    def filter(self, predicate: Callable[[GoldItem], bool]) -> "GoldSet":
        """Arbitrary predicate filter. Returns items where predicate(g) is truthy."""
        return GoldSet(g for g in self._items if predicate(g))

    def filter_category(self, *categories: CategoryLike) -> "GoldSet":
        """Keep only items whose category is in ``categories``.

        Accepts either Category enums or their string values:
            gs.filter_category(Category.MATHS)
            gs.filter_category('maths', 'science')
        """
        keep = {_coerce_category(c) for c in categories}
        return GoldSet(g for g in self._items if g.category in keep)

    def filter_level(self, *, min_level: int = 1, max_level: int = 15) -> "GoldSet":
        """Keep only items with ``min_level <= level <= max_level``."""
        return GoldSet(
            g for g in self._items if min_level <= g.level <= max_level
        )

    def filter_competition(self, *competition_ids: int) -> "GoldSet":
        """Keep only items from the given competition ids."""
        keep = set(competition_ids)
        return GoldSet(g for g in self._items if g.competition_id in keep)

    # ── Sampling (each returns a new GoldSet) ─────────────────────────────

    def take(self, n: int) -> "GoldSet":
        """Deterministic first-n slice (preserves order)."""
        return GoldSet(self._items[:n])

    def shuffle(self, *, seed: int = 0) -> "GoldSet":
        """Return a shuffled copy. Reproducible with the seed."""
        rng = random.Random(seed)
        out = list(self._items)
        rng.shuffle(out)
        return GoldSet(out)

    def sample(self, n: int, *, seed: int = 0) -> "GoldSet":
        """Uniform random n-item sample without replacement.

        Returns all items unchanged if ``n >= len(self)``.
        """
        if n >= len(self._items):
            return GoldSet(self._items)
        rng = random.Random(seed)
        return GoldSet(rng.sample(self._items, n))

    def take_per_level(self, n: int, *, seed: int = 0) -> "GoldSet":
        """At most ``n`` items per level. Random within each level.

        Levels with fewer than ``n`` items are taken whole. Useful for
        building difficulty-balanced pilot sets where you want, e.g., 3
        questions at each level 1..15 — yielding 45 if every level has
        enough items, fewer otherwise.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        rng = random.Random(seed)
        bucket: dict[int, list[GoldItem]] = defaultdict(list)
        for g in self._items:
            bucket[g.level].append(g)
        out: list[GoldItem] = []
        for level in sorted(bucket):
            items = bucket[level]
            out.extend(items if len(items) <= n else rng.sample(items, n))
        return GoldSet(out)

    def take_per_category(self, n: int, *, seed: int = 0) -> "GoldSet":
        """At most ``n`` items per category. Random within each category."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        rng = random.Random(seed)
        bucket: dict[Optional[Category], list[GoldItem]] = defaultdict(list)
        for g in self._items:
            bucket[g.category].append(g)
        out: list[GoldItem] = []
        for cat in sorted(bucket, key=lambda c: c.value if c else "~unknown"):
            items = bucket[cat]
            out.extend(items if len(items) <= n else rng.sample(items, n))
        return GoldSet(out)

    # ── Splits ───────────────────────────────────────────────────────────

    def split(
        self, fraction: float, *, seed: int = 0,
    ) -> tuple["GoldSet", "GoldSet"]:
        """Shuffled train/test split.

        First returned GoldSet gets ``fraction`` of the items; second gets
        the rest. The shuffle is seeded — same seed → same split.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        rng = random.Random(seed)
        shuffled = list(self._items)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * fraction)
        return GoldSet(shuffled[:cut]), GoldSet(shuffled[cut:])

    def by_category(self) -> dict[Category, "GoldSet"]:
        """Group items by category. Items with ``category=None`` are skipped."""
        groups: dict[Category, list[GoldItem]] = defaultdict(list)
        for g in self._items:
            if g.category is not None:
                groups[g.category].append(g)
        return {cat: GoldSet(items) for cat, items in groups.items()}

    def by_level(self) -> dict[int, "GoldSet"]:
        """Group items by level (1..15). Returns dict sorted by level."""
        groups: dict[int, list[GoldItem]] = defaultdict(list)
        for g in self._items:
            groups[g.level].append(g)
        return {lvl: GoldSet(groups[lvl]) for lvl in sorted(groups)}

    # ── Stats ────────────────────────────────────────────────────────────

    def counts_by_category(self) -> dict[str, int]:
        """{'maths': N, 'science': N, ...}. ``'unknown'`` for items with None."""
        return dict(Counter(
            g.category.value if g.category else "unknown" for g in self._items
        ))

    def counts_by_level(self) -> dict[int, int]:
        """{1: N, 2: N, ...}. Sorted by level."""
        return dict(sorted(Counter(g.level for g in self._items).items()))

    def counts(self) -> dict[str, dict[int, int]]:
        """Cross-tab: {category: {level: count}}."""
        nested: dict[str, Counter] = defaultdict(Counter)
        for g in self._items:
            key = g.category.value if g.category else "unknown"
            nested[key][g.level] += 1
        return {cat: dict(sorted(c.items())) for cat, c in nested.items()}

    def print_stats(self) -> None:
        """Pretty-print a category × level counts matrix."""
        if not self._items:
            print("GoldSet is empty.")
            return
        matrix = self.counts()
        levels = sorted({l for c in matrix.values() for l in c})
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

    # ── Set ops (by question identity) ────────────────────────────────────

    def __add__(self, other: "GoldSet") -> "GoldSet":
        """Union by (competition_id, question_text). Right side wins ties."""
        seen: dict[tuple[int, str], GoldItem] = {
            _identity_key(g): g for g in self._items
        }
        for g in other:
            seen[_identity_key(g)] = g
        return GoldSet(seen.values())

    def __sub__(self, other: "GoldSet") -> "GoldSet":
        """Difference by (competition_id, question_text). Use for held-out sets:
            train = full - test
        """
        exclude = {_identity_key(g) for g in other}
        return GoldSet(g for g in self._items if _identity_key(g) not in exclude)