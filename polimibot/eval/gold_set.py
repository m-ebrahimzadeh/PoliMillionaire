"""Gold set: ground-truth questions mined from run logs.

The game server reveals correctness only after submission.
We recover the correct answer index two ways:
  1. Direct: chosen_index where correct=True
  2. Elimination: if all-but-one options have been seen wrong across runs

Save once after your baseline runs. Never re-harvest mid-experiment —
that would silently grow your test set and invalidate comparisons.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

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


def harvest_gold_set(runs_dir: Path) -> list[GoldItem]:
    """Mine all *.jsonl run files in runs_dir → GoldItems.

    Deduplicates by (competition_id, question_text).
    A question is included only when the correct index can be confirmed.
    """
    # confirmed[key] = GoldItem
    confirmed: dict[str, GoldItem] = {}
    # wrong_indices[key] = set of indices we've seen answered incorrectly
    wrong_indices: dict[str, set[int]] = {}
    # raw options/metadata for elimination pass
    meta: dict[str, dict] = {}

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