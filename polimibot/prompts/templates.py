"""Prompt templates. Versioned, testable, swappable they must be.

A PromptStyle is an experiment condition — not a magic string.
Only this module knows what text goes to the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

from ..config import Category

LETTERS = ("A", "B", "C", "D")


# ── Prompt styles ──────────────────────────────────────────────────────────

class PromptStyle(str, Enum):
    """Ablatable prompting variants. Add new ones here; nothing else changes."""
    ZERO_SHOT     = "zero_shot"       # no examples, no CoT
    ZERO_SHOT_COT = "zero_shot_cot"   # no examples, think step-by-step
    FEW_SHOT      = "few_shot"        # 1 curated example per category
    FEW_SHOT_COT  = "few_shot_cot"    # 1 example with reasoning trace


# ── System prompts ─────────────────────────────────────────────────────────

_CATEGORY_SYSTEM: Dict[Category, str] = {
    Category.ENTERTAINMENT: (
        "You are an expert on movies, music, television, and pop culture. "
        "Answer multiple-choice trivia accurately."
    ),
    Category.HISTORY: (
        "You are an expert on ancient history, classical civilisations, "
        "and political history. Answer multiple-choice trivia accurately."
    ),
    Category.SCIENCE: (
        "You are an expert on biology, chemistry, physics, and the natural world. "
        "Answer multiple-choice trivia accurately."
    ),
    Category.MATHS: (
        "You are a careful mathematician. "
        "Compute precisely — do not guess. "
        "Solve step by step, then state your final answer."
    ),
}

_GENERIC_SYSTEM = (
    "You are a careful, knowledgeable quiz contestant. "
    "From four options labelled A, B, C and D, pick the single best answer."
)

# ── Output instructions ────────────────────────────────────────────────────

_DIRECT = (
    "Reply with exactly one line: 'Answer: <letter>' "
    "where <letter> is one of A, B, C, D. No other text."
)

_COT = (
    "Think step by step in at most 3 short sentences, "
    "then on the final line write exactly 'Answer: <letter>' "
    "where <letter> is one of A, B, C, D."
)


# ── Few-shot examples ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class FewShotExample:
    question: str
    options: tuple[str, ...]   # exactly 4
    answer_letter: str         # "A" .. "D"
    rationale: Optional[str] = None   # only used in CoT styles


# One hand-curated example per category.
# Not real game questions — similar style and difficulty, leakage avoided.
_FEW_SHOT_BANK: Dict[Category, FewShotExample] = {
    Category.ENTERTAINMENT: FewShotExample(
        question="Which 1994 film features a character named Forrest Gump?",
        options=("Philadelphia", "Pulp Fiction", "Forrest Gump", "The Shawshank Redemption"),
        answer_letter="C",
        rationale="The film is named after its protagonist Forrest Gump.",
    ),
    Category.HISTORY: FewShotExample(
        question="In which year did Julius Caesar cross the Rubicon?",
        options=("63 BC", "49 BC", "44 BC", "31 BC"),
        answer_letter="B",
        rationale="Caesar crossed the Rubicon in 49 BC, triggering the Roman civil war.",
    ),
    Category.SCIENCE: FewShotExample(
        question="What is the chemical symbol for gold?",
        options=("Ag", "Fe", "Au", "Cu"),
        answer_letter="C",
        rationale="Gold's symbol Au comes from the Latin 'aurum'.",
    ),
    Category.MATHS: FewShotExample(
        question="What is 15% of 200?",
        options=("25", "30", "35", "40"),
        answer_letter="B",
        rationale="15% of 200 = 0.15 × 200 = 30.",
    ),
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_options(options: Sequence[str]) -> str:
    return "\n".join(f"{l}. {t}" for l, t in zip(LETTERS, options))


def _system_prompt(category: Optional[Category]) -> str:
    return _CATEGORY_SYSTEM.get(category, _GENERIC_SYSTEM) if category else _GENERIC_SYSTEM


def _user_turn(question: str, options: Sequence[str], *, cot: bool) -> str:
    instr = _COT if cot else _DIRECT
    return f"Question: {question}\n\nOptions:\n{_format_options(options)}\n\n{instr}"


def _few_shot_turns(example: FewShotExample, *, cot: bool) -> List[Dict[str, str]]:
    """One user+assistant pair for the in-context example."""
    user = _user_turn(example.question, example.options, cot=cot)
    if cot and example.rationale:
        assistant = f"{example.rationale}\nAnswer: {example.answer_letter}"
    else:
        assistant = f"Answer: {example.answer_letter}"
    return [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ── Public API ─────────────────────────────────────────────────────────────

def build_messages(
    question: str,
    options: Sequence[str],
    *,
    category: Optional[Category] = None,
    style: PromptStyle = PromptStyle.ZERO_SHOT,
) -> List[Dict[str, str]]:
    """Assemble chat messages for a single MCQ question.

    This is the only function the strategy layer calls.
    Swap the style → different experiment condition, same interface.
    """
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}")

    cot = style in (PromptStyle.ZERO_SHOT_COT, PromptStyle.FEW_SHOT_COT)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_prompt(category)}
    ]

    if style in (PromptStyle.FEW_SHOT, PromptStyle.FEW_SHOT_COT):
        example = _FEW_SHOT_BANK.get(category) if category else None
        if example:
            messages.extend(_few_shot_turns(example, cot=cot))

    messages.append({"role": "user", "content": _user_turn(question, options, cot=cot)})
    return messages


def build_messages_with_context(
    question: str,
    options: Sequence[str],
    context: str,
    *,
    category: Optional[Category] = None,
    style: PromptStyle = PromptStyle.ZERO_SHOT,
) -> List[Dict[str, str]]:
    """Like build_messages, but prepends retrieved passages to the user turn.

    Args:
        context: pre-formatted retrieval results (caller's responsibility).
                 Empty string → degrades gracefully to build_messages behaviour.
        style: CoT styles are supported here (use_score_options must be False).
    """
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}")
    if not context:
        return build_messages(question, options, category=category, style=style)

    cot = style in (PromptStyle.ZERO_SHOT_COT, PromptStyle.FEW_SHOT_COT)
    instr = _COT if cot else _DIRECT

    context_block = f"Context (from Wikipedia):\n{context}"
    user_content = (
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{_format_options(options)}\n\n"
        f"{instr}"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_prompt(category)},
        {"role": "user",   "content": user_content},
    ]
    return messages




# ── Answer parsing ─────────────────────────────────────────────────────────

# Patterns tried in priority order. Most specific first.
_PATTERNS = (
    re.compile(r"answer\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\b(?:final|correct)\s+answer\s*(?:is)?\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\boption\s+\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"^([A-D])\s*[\.\)]", re.MULTILINE),  # "B." or "B)" at line start
    re.compile(r"\b([A-D])\b"),                       # lone letter — last resort
)


def parse_answer(text: str) -> Optional[int]:
    """Extract 0-based option index from model output. Returns None if unparseable.

    Tries patterns from most specific to least; stops at first match.
    Converts letter → 0-based index: A→0, B→1, C→2, D→3.
    """
    if not text:
        return None
    for pattern in _PATTERNS:
        m = pattern.search(text)
        if m:
            return ord(m.group(1).upper()) - ord("A")
    return None