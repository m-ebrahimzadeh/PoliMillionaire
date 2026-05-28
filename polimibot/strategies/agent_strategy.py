"""AgentStrategy — ReAct-style iterative tool-calling loop.

Protocol (per question, bounded by max_iterations + time_budget):
  1. Prompt the LLM with a tool-aware system message.
  2. Generate free text.
  3a. If output contains   CALL: calc(<expr>)    → execute via safe_eval, inject result, re-prompt.
  3b. If output contains   CALL: solve(<expr>)   → execute via sympy_solve, inject result, re-prompt.
  3c. If output contains   Answer: <letter>      → done.
  4. If iterations exhausted → parse last text; if still unparseable, abstain.

Two tools are wired:
  calc  — fast arithmetic (safe_eval AST sandbox); use for direct numeric computation.
  solve — symbolic math (SymPy); use for equations, modular arithmetic, exact combinatorics.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from ..models.llm import LLM
from ..models.mock import MockLLM
from ..prompts.templates import parse_answer
from ..tools.calculator import safe_eval
from ..tools.sympy_tool import sympy_solve
from .base import Strategy, StrategyInput, StrategyOutput

_AnyLLM = LLM | MockLLM

# ── System prompt ─────────────────────────────────────────────────────────────
# The protocol the model must follow is described concisely.
# One CALL per turn; wait for result; final line must be "Answer: <letter>".

_AGENT_SYSTEM = """\
You are a careful quiz contestant. You have two tools:

  CALL: calc(<expression>)
    Fast arithmetic. Use for direct numeric computation.
    Example: CALL: calc(15/100 * 400)

  CALL: solve(<expression>)
    Symbolic math via SymPy. Use for:
      - equations with an unknown   e.g. solve("3*x + 7 - 22", "x")
      - modular / exact arithmetic  e.g. solve("Mod(3**100, 10)")
      - exact combinatorics         e.g. solve("binomial(10, 3)")
      - geometric / trig formulas   e.g. solve("pi * 5**2")
    Example: CALL: solve(Mod(3**100, 10))

Rules:
  1. Think step by step in at most 3 sentences.
  2. If computation is needed, emit EXACTLY one tool call on its own line, then STOP.
     Choose calc for plain arithmetic, solve for algebra or exact symbolic results.
  3. When you know the final answer, write EXACTLY: Answer: <letter>
     where <letter> is one of A, B, C, D.
  4. Never guess a number — use the appropriate tool if unsure.
  5. If the question is purely factual (a name, a theorem, a definition), skip
     the tools and answer directly.
"""

# ── Call-detection helpers ────────────────────────────────────────────────────

def _extract_call(text: str) -> Optional[tuple[str, str]]:
    """Find 'CALL: tool_name(expr)' in text, handling nested parens correctly.

    Returns (tool_name, expression) or None if no call found.
    Balanced-paren scan ensures 'calc(factorial(5))' extracts correctly.
    """
    m = re.search(r"CALL\s*:\s*(\w+)\s*\(", text, re.IGNORECASE)
    if not m:
        return None
    tool_name = m.group(1).lower()
    start = m.end()   # index of first char INSIDE the outer '('
    depth = 1
    for i, ch in enumerate(text[start:]):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return tool_name, text[start : start + i]
    return None   # unbalanced — malformed call


def _run_calc(expression: str) -> str:
    """Execute expression via safe_eval. Returns result string or error message."""
    try:
        result = safe_eval(expression)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def _run_solve(expression: str) -> str:
    """Execute expression via sympy_solve. Returns result string or error message."""
    try:
        return sympy_solve(expression)
    except Exception as exc:
        return f"Error: {exc}"


# ── Strategy ──────────────────────────────────────────────────────────────────

class AgentStrategy(Strategy):
    """ReAct agent: LLM emits CALL: markers; tool results injected as user turns.

    Args:
        llm: loaded LLM or MockLLM.
        max_iterations: maximum tool calls before forcing a final answer.
        per_turn_tokens: max_new_tokens per generation step.
    """

    def __init__(
        self,
        llm: _AnyLLM,
        *,
        max_iterations: int = 3,
        per_turn_tokens: int = 200,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.per_turn_tokens = per_turn_tokens
        self.name = (
            f"agent[{getattr(llm, 'name', 'llm')}"
            f"|max_iter={max_iterations}]"
        )

    def warm_up(self) -> None:
        dummy = StrategyInput(
            question="What is 2+2?",
            options=("3", "4", "5", "6"),
            level=1,
        )
        self.answer(dummy)

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        t_start = time.monotonic()
        # Leave a 1-second safety margin before the hard deadline.
        deadline = t_start + max(2.0, inp.time_budget_seconds - 1.0)

        messages = self._initial_messages(inp)
        trace: list[str] = []
        n_tool_calls = 0

        for iteration in range(self.max_iterations + 1):
            if time.monotonic() >= deadline:
                break

            resp = self.llm.generate(
                messages,
                max_new_tokens=self.per_turn_tokens,
                temperature=0.0,
            )
            text = resp.text.strip()
            trace.append(f"[step {iteration}] {text}")

            # Append model turn to conversation history
            messages = list(messages) + [
                {"role": "assistant", "content": text}
            ]

            # ── Branch 1: tool call detected ─────────────────────────────
            call = _extract_call(text)
            if call and time.monotonic() < deadline:
                tool_name, expression = call
                if tool_name == "calc":
                    observation = _run_calc(expression)
                elif tool_name == "solve":
                    observation = _run_solve(expression)
                else:
                    observation = f"Unknown tool '{tool_name}' — available tools: calc, solve."
                n_tool_calls += 1
                trace.append(f"[obs {iteration}] {tool_name}({expression}) → {observation}")

                # Inject result as a new user turn
                messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            f"Tool result: {observation}\n"
                            "Now give your final answer."
                        ),
                    }
                ]
                continue   # next iteration: model sees the observation

            # ── Branch 2: final answer detected ──────────────────────────
            idx = parse_answer(text)
            if idx is not None:
                return StrategyOutput(
                    chosen_index=idx,
                    confidence=0.70,   # free generation: lower confidence than logit-scoring
                    rationale="\n".join(trace),
                    extras={
                        "n_tool_calls": n_tool_calls,
                        "iterations": iteration + 1,
                    },
                )

            # ── Branch 3: neither — nudge the model ──────────────────────
            messages = list(messages) + [
                {
                    "role": "user",
                    "content": "Please write your final answer: 'Answer: <letter>'.",
                }
            ]

        # ── Exhausted — salvage what we can ──────────────────────────────
        rationale = "\n".join(trace)
        for step_text in reversed(trace):
            idx = parse_answer(step_text)
            if idx is not None:
                return StrategyOutput(
                    chosen_index=idx,
                    confidence=0.40,
                    rationale=rationale,
                    extras={"n_tool_calls": n_tool_calls, "exhausted": True},
                )

        return StrategyOutput(
            chosen_index=0,
            confidence=0.25,
            rationale=rationale,
            is_abstain=True,
            extras={"n_tool_calls": n_tool_calls, "exhausted": True},
        )

    def _initial_messages(self, inp: StrategyInput) -> list[dict]:
        options_text = "\n".join(
            f"{letter}. {opt}"
            for letter, opt in zip("ABCD", inp.options)
        )
        user = (
            f"Question: {inp.question}\n\n"
            f"Options:\n{options_text}\n\n"
            "Think step by step. Use CALL: calc(...) for arithmetic or "
            "CALL: solve(...) for algebra/equations/modular problems, "
            "then write your Answer."
        )
        return [
            {"role": "system", "content": _AGENT_SYSTEM},
            {"role": "user",   "content": user},
        ]