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
    ZERO_SHOT     = "zero_shot"       # no examples, no CoT — direct "Answer: X"
    ZERO_SHOT_COT = "zero_shot_cot"   # no examples, numbered-step reasoning + \boxed
    FEW_SHOT      = "few_shot"        # 1 curated example per category
    FEW_SHOT_COT  = "few_shot_cot"    # 1 example with reasoning trace + \boxed
    ELIMINATION   = "elimination"     # option-by-option scaffolding + \boxed


# ── System prompts ─────────────────────────────────────────────────────────
# Each category prompt is *instructive*, not flattering. It teaches the model
# the gotchas of that category — generic praise ("You are an expert") leaves
# accuracy on the table.

_CATEGORY_SYSTEM: Dict[Category, str] = {
    Category.ENTERTAINMENT: (
        "You are answering multiple-choice trivia on film, music, television, "
        "and pop culture. Trivia distractors often pair real co-stars from the "
        "same era, films by the same director, or bands sharing members. "
        "Verify the specific year, director, album, or actor before committing."
    ),
    Category.HISTORY: (
        "You are answering multiple-choice trivia on ancient history, classical "
        "civilisations, and political history. Watch for plausible distractors "
        "that confuse adjacent centuries, similar-sounding dynasties, or "
        "namesake successors (e.g. Caesar vs Augustus, Henry V vs Henry VIII). "
        "Verify the specific century or reign before committing."
    ),
    Category.SCIENCE: (
        "You are answering multiple-choice trivia on biology, chemistry, "
        "physics, and the natural world. Be careful with units (atoms vs "
        "molecules, kg vs g, joules vs calories) and distinguish necessary "
        "from sufficient conditions in causal claims."
    ),
    Category.MATHS: (
        "You are a careful mathematician. Compute precisely — do not guess. "
        "Verify each calculation before committing."
    ),
    Category.PHILOSOPHY: (
        "You are answering multiple-choice trivia on philosophy and psychology. "
        "Distractors often confuse adjacent schools (Stoicism vs Epicureanism, "
        "rationalism vs empiricism), a philosopher with their teacher or student "
        "(Socrates / Plato / Aristotle), or psychology subfields (cognitive vs "
        "behaviourist vs psychoanalytic). Watch for named-after attributions "
        "(Pavlovian vs Skinnerian conditioning, Jungian vs Freudian concepts). "
        "Verify the specific philosopher, school, or experiment before committing."
    ),
    Category.NEWS: (
        "You are answering multiple-choice trivia on recent and historical news "
        "events. Distractors typically swap the year, the country, an official's "
        "title, or the outcome of an event. Verify the specific date and the "
        "primary actor before committing. If a question concerns events past "
        "your knowledge cutoff, prefer the option most consistent with the "
        "facts you do know rather than guessing."
    ),
}

_GENERIC_SYSTEM = (
    "You are a careful, knowledgeable quiz contestant. "
    "From four options labelled A, B, C and D, pick the single best answer."
)

# ── Output instructions ────────────────────────────────────────────────────

_DIRECT = (
    "Begin your reply with 'Answer:' followed by exactly one letter from "
    "A, B, C, D. Do not include any text before 'Answer:'."
)

_COT = (
    "Solve step by step using this structure:\n"
    "  Step 1: Restate what the question asks.\n"
    "  Step 2: Compute, eliminate, or recall the relevant fact.\n"
    "  Step 3: End with \\boxed{X} on its own line, where X is one of "
    "A, B, C, D."
)

_ELIMINATION = (
    "Evaluate each option in one short sentence, then commit:\n"
    "  A: <why right or wrong>\n"
    "  B: <why right or wrong>\n"
    "  C: <why right or wrong>\n"
    "  D: <why right or wrong>\n"
    "Then on a new line write \\boxed{X} with the best answer."
)


def _instruction_for(style: PromptStyle) -> str:
    """Pick the output-instruction block for the given style."""
    if style == PromptStyle.ELIMINATION:
        return _ELIMINATION
    if style in (PromptStyle.ZERO_SHOT_COT, PromptStyle.FEW_SHOT_COT):
        return _COT
    return _DIRECT


def _example_answer_format(style: PromptStyle, rationale: Optional[str], letter: str) -> str:
    """How a few-shot assistant turn should phrase its answer.

    Must match the instruction the model just received — so the example
    teaches the format, not contradicts it.
    """
    if style in (PromptStyle.ZERO_SHOT_COT, PromptStyle.FEW_SHOT_COT) and rationale:
        return f"{rationale}\n\\boxed{{{letter}}}"
    return f"Answer: {letter}"


# ── Few-shot examples ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class FewShotExample:
    question: str
    options: tuple[str, ...]   # exactly 4
    answer_letter: str         # "A" .. "D"
    rationale: Optional[str] = None   # only used in CoT styles


# Hand-curated examples per category. Picked to:
#   - exercise the kind of reasoning the real gold set demands
#     (not just answer recall — modular arithmetic, dates, units…)
#   - avoid word-overlap between the question and the correct option
#     (otherwise the model learns "pick the option that mentions the
#     question's noun" — a brittle heuristic that misfires on distractors).
#   - balance the correct-answer letter across A/B/C/D to avoid
#     majority-label bias from a single in-context example.
#
# History gets multiple examples spanning distinct periods so the few-shot
# signal generalises across the gold set rather than anchoring on Roman
# antiquity. The other categories carry a single example wrapped in a
# 1-element list for uniform API.
_FEW_SHOT_BANK: Dict[Category, List["FewShotExample"]] = {
    Category.ENTERTAINMENT: [
        FewShotExample(
            question="Which director directed both 'Jaws' and 'Schindler's List'?",
            options=("Martin Scorsese", "Steven Spielberg", "Francis Ford Coppola", "Stanley Kubrick"),
            answer_letter="B",
            rationale="Steven Spielberg directed Jaws (1975) and Schindler's List (1993).",
        ),
    ],
    Category.HISTORY: [
        # Ancient — Caesar crossing the Rubicon, single-event date.
        FewShotExample(
            question="In which year did Julius Caesar cross the Rubicon?",
            options=("63 BC", "49 BC", "44 BC", "31 BC"),
            answer_letter="B",
            rationale="Caesar crossed the Rubicon in 49 BC, triggering the Roman civil war.",
        ),
        # Medieval — Norman Conquest, single-event date.
        FewShotExample(
            question="In what year did William the Conqueror defeat King Harold at the Battle of Hastings?",
            options=("1066", "1086", "1099", "1054"),
            answer_letter="A",
            rationale=(
                "The Battle of Hastings took place on 14 October 1066, leading to "
                "the Norman conquest of England."
            ),
        ),
        # Early modern — French Revolution, distinguishing trigger event from later phases.
        FewShotExample(
            question="Which event marked the beginning of the French Revolution in 1789?",
            options=(
                "The Reign of Terror",
                "The execution of Louis XVI",
                "The storming of the Bastille",
                "Napoleon's coup",
            ),
            answer_letter="C",
            rationale=(
                "The storming of the Bastille on 14 July 1789 is conventionally "
                "regarded as the beginning of the French Revolution; the Reign of "
                "Terror and the execution of Louis XVI came years later."
            ),
        ),
        # 20th-century — Cold War end, date precision across nearby years.
        FewShotExample(
            question="In which year did the Berlin Wall fall, signalling the end of the Cold War in Europe?",
            options=("1985", "1989", "1991", "1993"),
            answer_letter="B",
            rationale=(
                "The Berlin Wall fell on 9 November 1989, two years before the "
                "formal dissolution of the Soviet Union in 1991."
            ),
        ),
    ],
    Category.SCIENCE: [
        FewShotExample(
            question="What is the chemical symbol for gold?",
            options=("Ag", "Fe", "Au", "Cu"),
            answer_letter="C",
            rationale="Gold's symbol Au comes from the Latin 'aurum'.",
        ),
    ],
    Category.MATHS: [
        FewShotExample(
            question="What is the units digit of 3^100?",
            options=("1", "3", "7", "9"),
            answer_letter="A",
            rationale=(
                "Units digits of 3^n cycle (3, 9, 7, 1) with period 4. "
                "100 mod 4 = 0, so we take the last value in the cycle: 1."
            ),
        ),
    ],
    Category.PHILOSOPHY: [
        FewShotExample(
            question="Which philosopher introduced the 'categorical imperative'?",
            options=("David Hume", "Immanuel Kant", "John Stuart Mill", "Friedrich Nietzsche"),
            answer_letter="B",
            rationale=(
                "The categorical imperative is the central concept of Kant's "
                "deontological ethics, set out in the 1785 Groundwork."
            ),
        ),
    ],
    Category.NEWS: [
        FewShotExample(
            question="In which year did the United Kingdom formally leave the European Union?",
            options=("2016", "2018", "2020", "2022"),
            answer_letter="C",
            rationale=(
                "The UK voted to leave in 2016, but formal withdrawal under "
                "Article 50 took effect on 31 January 2020."
            ),
        ),
    ],
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_options(options: Sequence[str]) -> str:
    return "\n".join(f"{l}. {t}" for l, t in zip(LETTERS, options))


def _system_prompt(category: Optional[Category]) -> str:
    return _CATEGORY_SYSTEM.get(category, _GENERIC_SYSTEM) if category else _GENERIC_SYSTEM


def _user_turn(question: str, options: Sequence[str], *, style: PromptStyle) -> str:
    instr = _instruction_for(style)
    return f"Question: {question}\n\nOptions:\n{_format_options(options)}\n\n{instr}"


def _few_shot_turns(example: FewShotExample, *, style: PromptStyle) -> List[Dict[str, str]]:
    """One user+assistant pair for the in-context example.

    The assistant turn's answer format matches the instruction the model just
    received — so the few-shot teaches the format, not contradicts it.
    """
    user = _user_turn(example.question, example.options, style=style)
    assistant = _example_answer_format(style, example.rationale, example.answer_letter)
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

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_prompt(category)}
    ]

    if style in (PromptStyle.FEW_SHOT, PromptStyle.FEW_SHOT_COT):
        examples = _FEW_SHOT_BANK.get(category) if category else None
        if examples:
            # All curated examples are emitted in sequence — one user+assistant
            # pair each. Categories with a single example produce one pair;
            # History produces four diverse priming cases spanning ancient → modern.
            for example in examples:
                messages.extend(_few_shot_turns(example, style=style))

    messages.append({"role": "user", "content": _user_turn(question, options, style=style)})
    return messages


def build_messages_with_context(
    question: str,
    options: Sequence[str],
    context: str,
    *,
    category: Optional[Category] = None,
    style: PromptStyle = PromptStyle.ZERO_SHOT,
) -> List[Dict[str, str]]:
    """Like build_messages, but appends retrieved passages after the question.

    Ordering rationale: chat-tuned models attend most strongly to the most
    recent tokens. Putting the question + options BEFORE the context means
    the question never has to compete with retrieval for attention. The
    context is framed as 'reference material', not 'authoritative source',
    so off-topic retrievals don't pull the model toward fabricated answers.

    Args:
        context: pre-formatted retrieval results (caller's responsibility).
                 Empty string → degrades gracefully to build_messages behaviour.
        style: CoT / ELIMINATION styles are supported here
               (use_score_options must be False for those).
    """
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}")
    if not context:
        return build_messages(question, options, category=category, style=style)

    instr = _instruction_for(style)

    user_content = (
        f"Question: {question}\n\n"
        f"Options:\n{_format_options(options)}\n\n"
        f"Reference material (may or may not be relevant):\n{context}\n\n"
        f"{instr}"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_prompt(category)},
        {"role": "user",   "content": user_content},
    ]
    return messages




# ── Answer parsing ─────────────────────────────────────────────────────────

# Structured patterns. Most specific first; first-match wins for these because
# they require keywords ("answer", "boxed{}", "option") that the model uses
# only at decision time. The bare-letter last-resort is handled separately —
# it must take the LAST match, not the first, so a free-text setup like
# "Let A be the area..." doesn't outrank the actual answer at the end.
_STRUCTURED_PATTERNS = (
    # Qwen-Math / DeepSeek-Math RL models terminate with \boxed{X}; treat as
    # canonical and place at the top so it wins against any earlier letters.
    re.compile(r"\\boxed\{\s*([A-D])\s*\}", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\b(?:final|correct)\s+answer\s*(?:is)?\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\boption\s+\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"^([A-D])\s*[\.\)]", re.MULTILINE),  # "B." or "B)" at line start
)

# Bare-letter pattern, separately handled — the last match wins.
_BARE_LETTER = re.compile(r"\b([A-D])\b")


def parse_answer(text: str) -> Optional[int]:
    """Extract 0-based option index from model output. Returns None if unparseable.

    Cascade:
      1. Structured patterns (\\boxed{X}, "Answer: X", "final answer is X",
         "Option X", line-leading "X." / "X)") — first match wins.
      2. Bare letter \\b([A-D])\\b — LAST match wins (so a CoT setup like
         "Let A be the area, so the answer is C" picks C, not A).

    Converts letter → 0-based index: A→0, B→1, C→2, D→3.
    """
    if not text:
        return None
    for pattern in _STRUCTURED_PATTERNS:
        m = pattern.search(text)
        if m:
            return ord(m.group(1).upper()) - ord("A")
    matches = _BARE_LETTER.findall(text)
    if matches:
        return ord(matches[-1].upper()) - ord("A")
    return None
