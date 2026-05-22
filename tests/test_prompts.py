import pytest
from polimibot.prompts.templates import (
    PromptStyle, build_messages, parse_answer, FewShotExample, LETTERS
)
from polimibot.config import Category


def _build(style, category=None):
    return build_messages(
        "What is the capital of France?",
        ("Rome", "London", "Paris", "Berlin"),
        category=category,
        style=style,
    )


def test_zero_shot_has_system_and_one_user_turn():
    msgs = _build(PromptStyle.ZERO_SHOT)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]


def test_few_shot_inserts_example_turns():
    msgs = _build(PromptStyle.FEW_SHOT, category=Category.HISTORY)
    roles = [m["role"] for m in msgs]
    # system, user (example), assistant (example), user (real question)
    assert roles == ["system", "user", "assistant", "user"]


def test_cot_instruction_present_in_user_turn():
    msgs = _build(PromptStyle.ZERO_SHOT_COT)
    user_content = msgs[-1]["content"]
    # Numbered-step scaffold replaces "at most 3 sentences" guidance.
    assert "step by step" in user_content.lower()
    assert "Step 1" in user_content
    assert "boxed" in user_content.lower()


def test_category_system_prompt_used_when_given():
    msgs_sci = _build(PromptStyle.ZERO_SHOT, category=Category.SCIENCE)
    msgs_gen = _build(PromptStyle.ZERO_SHOT, category=None)
    assert msgs_sci[0]["content"] != msgs_gen[0]["content"]


def test_maths_system_prompt_carries_no_format_instruction():
    """Format is owned by the user-turn instruction (_DIRECT / _COT /
    _ELIMINATION). The MATHS system prompt — like the other three category
    prompts — must carry domain guidance only, not output format. Letting
    format leak into the system prompt creates contradictions
    (MATHS + ZERO_SHOT: system says \\boxed{X}, user says 'Answer: X') and
    duplication (MATHS + ZERO_SHOT_COT: same instruction twice)."""
    msgs = _build(PromptStyle.ZERO_SHOT, category=Category.MATHS)
    sys_content = msgs[0]["content"].lower()
    assert "boxed" not in sys_content
    assert "answer:" not in sys_content


def test_cot_user_turn_carries_boxed_for_maths():
    """Format lives in the style block. For MATHS + ZERO_SHOT_COT, the user
    instruction must still demand \\boxed{X} — the regression we'd see if
    someone deletes it from _COT thinking the MATHS system prompt covers it."""
    msgs = _build(PromptStyle.ZERO_SHOT_COT, category=Category.MATHS)
    user_content = msgs[-1]["content"]
    assert "boxed" in user_content.lower()


def test_elimination_style_lists_all_letters():
    msgs = _build(PromptStyle.ELIMINATION)
    user_content = msgs[-1]["content"]
    for letter in "ABCD":
        assert f"{letter}:" in user_content
    assert "boxed" in user_content.lower()


def test_few_shot_cot_assistant_uses_boxed():
    msgs = _build(PromptStyle.FEW_SHOT_COT, category=Category.HISTORY)
    assistant = next(m for m in msgs if m["role"] == "assistant")
    # The example must teach the format the instruction demands: \boxed{X}.
    assert "\\boxed" in assistant["content"]


def test_few_shot_non_cot_assistant_uses_answer_prefix():
    msgs = _build(PromptStyle.FEW_SHOT, category=Category.HISTORY)
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"].startswith("Answer:")


def test_entertainment_few_shot_avoids_word_leakage():
    """The Jaws/Schindler's-List example tests the model's recall, not its
    ability to pattern-match question words to option text."""
    from polimibot.prompts.templates import _FEW_SHOT_BANK
    ex = _FEW_SHOT_BANK[Category.ENTERTAINMENT]
    correct_text = ex.options[ord(ex.answer_letter) - ord("A")]
    # No word in the correct option's text appears verbatim in the question.
    q_words = {w.lower().strip(".,'\"") for w in ex.question.split()}
    a_words = {w.lower().strip(".,'\"") for w in correct_text.split()}
    overlap = q_words & a_words
    assert not overlap, f"few-shot leaks answer: overlap={overlap}"


def test_wrong_option_count_raises():
    with pytest.raises(ValueError):
        build_messages("q?", ("A", "B", "C"), style=PromptStyle.ZERO_SHOT)


@pytest.mark.parametrize("text,expected", [
    ("Answer: B",                    1),
    ("The correct answer is C.",     2),
    ("I think option A is correct.", 0),
    ("answer:d",                     3),  # case-insensitive
    ("A.",                           0),  # line-start pattern
    ("Forrest Gump (C)",             2),  # parenthesized letter
])
def test_parse_answer_handles_variants(text, expected):
    assert parse_answer(text) == expected


def test_parse_answer_returns_none_on_garbage():
    assert parse_answer("I have no idea honestly") is None
    assert parse_answer("") is None


def test_parse_answer_priority_most_specific_wins():
    # "Answer: B" contains lone "A" too — should match "B", not "A"
    assert parse_answer("Answer: B, not A") == 1


# ── Regression coverage for ELIMINATION-style outputs (audit P10.4) ─────────
# Locks in the current behaviour where structured patterns (\boxed, "Answer:",
# line-leading "X." / "X)") win over bare letters anywhere in the text, AND
# where the bare-letter LAST-match rule picks the right answer for CoT outputs
# whose reasoning trace mentions other letters earlier.


def test_parse_answer_elimination_with_boxed_wins_over_letter_setup():
    """ELIMINATION style: per-option lines start with 'A.'/'B.'/... — the
    \\boxed{X} marker at the end must override the earlier line-start matches.
    """
    text = (
        "A. not correct because X.\n"
        "B. correct! Newton's second law applies here.\n"
        "C. incorrect, that's a different theorem.\n"
        "D. misreads the question.\n"
        "\\boxed{B}"
    )
    assert parse_answer(text) == 1


def test_parse_answer_elimination_line_start_picks_first_per_option():
    """Without a structured terminator, the line-start 'X.' pattern still
    wins (first match) — useful when the model produces only the per-option
    lines and no final 'Answer:' marker.
    """
    text = (
        "A. correct: this is the canonical definition.\n"
        "B. wrong.\n"
        "C. wrong.\n"
        "D. wrong.\n"
    )
    # Structured patterns are tried first; the line-leading "A." matches.
    assert parse_answer(text) == 0


def test_parse_answer_cot_setup_then_answer_picks_last_bare_letter():
    """CoT-style reasoning: the trace can mention several letters; the
    final 'Therefore C.' must win via the bare-letter LAST-match rule.
    """
    text = "Let A be the area, then by symmetry the answer is C."
    assert parse_answer(text) == 2


def test_parse_answer_cot_trailing_therefore_letter():
    """No structured marker, just a bare 'Therefore D.' at the tail —
    the last bare letter wins."""
    text = (
        "We rule out A because the dates don't match. "
        "B is plausible but B's reign was earlier. "
        "C is a distractor. Therefore D."
    )
    assert parse_answer(text) == 3