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
    assert "step by step" in user_content.lower()


def test_category_system_prompt_used_when_given():
    msgs_sci = _build(PromptStyle.ZERO_SHOT, category=Category.SCIENCE)
    msgs_gen = _build(PromptStyle.ZERO_SHOT, category=None)
    assert msgs_sci[0]["content"] != msgs_gen[0]["content"]


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