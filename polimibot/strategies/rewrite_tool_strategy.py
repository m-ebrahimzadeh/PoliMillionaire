"""RewriteToolStrategy — tools-first, pin-then-rewrite pipeline for math questions.

Full workflow per question:

  1. Run tools on the original question text (MathsTool → SympyDirectTool, ~0ms).
     If a tool fires → return immediately. LLM never called.
  2. LLM answers via logit-scoring (~0.8s). Result is "pinned" as fallback.
  3. If confidence >= threshold → return pinned (LLM is sure, rewrite not worth it).
  4. Ask the LLM to REWRITE the question into a single SymPy expression (~0.8s).
     Pass it directly to sympy_solve() — bypassing _PATTERNS entirely — and match
     against options. If it fires → return with confidence=0.92.
  5. If everything abstains or time runs out → return the pinned LLM answer.

Why tools run twice (sort of)?
  Step 1 runs the 23 regex patterns against the original question text.
  Step 4 runs sympy_solve() directly on a rewritten expression — a completely
  different path. The LLM may produce "solve(3*y*(y+5) - 2*(4*y+5), y)" for a
  question phrased in a way no pattern matches. Same tools, different input.

Why tools first?
  The patterns are deterministic and near-instant. A question like "What is
  log_8(2)?" or "(i+1)(5-5i)(5+5i) =" is answered in <1ms with 0.97 confidence
  regardless of what the LLM would have said. Never waste an LLM call on a
  question a tool can answer with certainty.

Confidence assignments:
  - Direct tool fires (step 1):   tool_out.confidence  (0.97 SympyDirect, 0.99 MathsTool)
  - Rewrite path fires (step 4):  0.92  (slightly lower — semantic rewrite risk)
  - Pinned LLM answer (step 5):   original logit confidence unchanged
"""
from __future__ import annotations

import re
import time
from fractions import Fraction
from typing import Optional

from ..config import Category
from ..models.llm import LLM
from ..models.mock import MockLLM
from ..prompts.templates import PromptStyle, build_messages
from ..tools.base import Tool
from ..tools.maths_tool import _match_options
from ..tools.sympy_tool import sympy_solve, _BLOCKED
from .base import Strategy, StrategyInput, StrategyOutput

_AnyLLM = LLM | MockLLM


# ── Rewrite prompt ────────────────────────────────────────────────────────────
# Short, concrete, example-driven. The model must output exactly one line of
# valid Python/SymPy — no prose, no LaTeX, no explanation.

_REWRITE_SYSTEM = (
    "You are a SymPy expression converter. "
    "Given a math problem, output ONLY a single SymPy-parseable Python expression. "
    "No explanation. No prose. Output exactly one line.\n"
    "Examples:\n"
    "  binomial(25, 3)\n"
    "  Mod(2**87, 7)\n"
    "  factorial(7)\n"
    "  log(2, 8)\n"
    "  solve(x**2 - 5*x + 6, x)[0]\n"
    "  diff(x**3 + exp(x), x).subs(x, 0)\n"
    "  (3*5 - 2) / 7"
)


class RewriteToolStrategy(Strategy):
    """LLM-pin + direct tools + LLM-rewrite + tools pipeline.

    Args:
        llm:                  The loaded language model (same instance used elsewhere).
        tools:                Tool chain to attempt — tried in order.
        confidence_threshold: Skip stages 3-5 when primary LLM confidence >= this.
                              Default 0.70 — questions above this are usually correct.
        rewrite_time_reserve: Seconds to reserve before deadline for the rewrite LLM
                              call. If less time remains, skip to pinned answer.
        rewrite_max_new_tokens: Token budget for the rewrite generation (60 is enough
                              for any single-line SymPy expression).
    """

    def __init__(
        self,
        llm: _AnyLLM,
        tools: list[Tool],
        *,
        confidence_threshold: float = 0.70,
        rewrite_time_reserve: float = 2.0,
        rewrite_max_new_tokens: int = 60,
    ) -> None:
        self.llm                    = llm
        self.tools                  = tools
        self.confidence_threshold   = confidence_threshold
        self.rewrite_time_reserve   = rewrite_time_reserve
        self.rewrite_max_new_tokens = rewrite_max_new_tokens
        tool_names = "+".join(t.name for t in tools)
        self.name = (
            f"rewrite_tool[{getattr(llm, 'name', 'llm')}"
            f"|thresh={confidence_threshold:.0%}"
            f"|tools={tool_names}]"
        )

    def warm_up(self) -> None:
        """Absorb CUDA JIT on a trivial question."""
        dummy = StrategyInput(
            question="What is 2+2?",
            options=("3", "4", "5", "6"),
            level=1,
        )
        self.answer(dummy)

    # ── Public entry point ────────────────────────────────────────────────────

    def answer(self, inp: StrategyInput) -> StrategyOutput:
        deadline = time.monotonic() + inp.time_budget_seconds - 1.5

        # ── Step 1: tools on original question text (always, ~0ms) ───────────
        for tool in self.tools:
            if time.monotonic() >= deadline:
                return StrategyOutput(chosen_index=0, confidence=0.25,
                                      extras={"path": "timeout_step1"})
            if not tool.can_handle(inp):
                continue
            tool_out = tool.use(inp)
            if tool_out is not None:
                return tool_out   # deterministic answer — done, no LLM call

        # ── Step 2: LLM logit answer (~0.8s) — pinned as fallback ────────────
        pinned = self._primary_answer(inp)

        # ── Step 3: high-confidence fast exit ────────────────────────────────
        if pinned.confidence >= self.confidence_threshold:
            return pinned

        # ── Step 4: rewrite + sympy_solve (maths only) ───────────────────────
        if inp.category != Category.MATHS:
            return self._pinned(pinned, "non_maths_no_rewrite")

        if time.monotonic() + self.rewrite_time_reserve >= deadline:
            return self._pinned(pinned, "no_time_for_rewrite")

        rewrite_expr = self._get_rewrite(inp)
        if rewrite_expr is not None:
            idx = self._eval_rewrite(rewrite_expr, inp.options)
            if idx is not None:
                return StrategyOutput(
                    chosen_index=idx,
                    confidence=0.92,
                    rationale=f"[rewrite_tool] {rewrite_expr}",
                    extras={
                        "tool":         "rewrite_tool",
                        "path":         "rewrite_tool",
                        "rewrite_expr": rewrite_expr,
                        "pinned_index": pinned.chosen_index,
                        "pinned_conf":  pinned.confidence,
                    },
                )

        # ── Step 5: all paths failed — return pinned ─────────────────────────
        return self._pinned(pinned, "all_abstained")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _primary_answer(self, inp: StrategyInput) -> StrategyOutput:
        """Single logit-scored LLM call. Mirrors BaselineLLMStrategy._answer_via_logits."""
        messages = build_messages(
            inp.question, inp.options,
            category=inp.category,
            style=PromptStyle.FEW_SHOT,
        )
        try:
            result = self.llm.score_options(messages)
            return StrategyOutput(
                chosen_index=result.chosen_index,
                confidence=result.top_prob,
                extras={
                    "probs":  result.probs,
                    "margin": result.margin,
                    "path":   "llm_primary",
                },
            )
        except Exception as _exc:
            # Log the actual error so we can diagnose it, then fall back.
            import traceback
            print(f"[RewriteToolStrategy] _primary_answer failed: {_exc}")
            traceback.print_exc()
            return StrategyOutput(chosen_index=0, confidence=0.25,
                                  extras={"path": "llm_primary_failed",
                                          "error": str(_exc)})

    def _get_rewrite(self, inp: StrategyInput) -> Optional[str]:
        """Ask the LLM to rewrite the question as a SymPy expression."""
        opts = "  ".join(f"{l}) {v}" for l, v in zip("ABCD", inp.options))
        user_content = (
            f"Math problem: {inp.question}\n"
            f"Options: {opts}\n\n"
            "Output one SymPy expression whose evaluated result matches an option. "
            "One line only."
        )
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        try:
            resp = self.llm.generate(
                messages,
                max_new_tokens=self.rewrite_max_new_tokens,
                temperature=0.0,
                stop_strings=["\n\n", "\n#"],
            )
            return self._strip_rewrite(resp.text)
        except Exception:
            return None

    @staticmethod
    def _strip_rewrite(raw: str) -> Optional[str]:
        """Extract the first plausible SymPy expression line from LLM output."""
        # Remove markdown code fences
        raw = re.sub(r'```[a-z]*\n?', '', raw).strip()
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if not re.search(r'[a-zA-Z0-9_(]', line):
                continue
            # Reject prose: common English words signal a failed rewrite
            if re.search(
                r'\b(the|is|of|to|and|for|that|this|you|we|it|a|an)\b',
                line, re.I,
            ):
                return None
            # Must contain at least one operator or parenthesis to be an expression
            if not re.search(r'[+\-*/()=,]', line):
                return None
            return line
        return None

    @staticmethod
    def _eval_rewrite(expr: str, options: tuple) -> Optional[int]:
        """Evaluate a raw SymPy expression and match against options."""
        if _BLOCKED.search(expr):
            return None
        try:
            result_str = sympy_solve(expr)
        except (ValueError, Exception):
            return None

        nums = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', result_str)
        if not nums:
            return None

        floats = [float(n) for n in nums]
        pos = [f for f in floats if f >= 0]
        value = pos[0] if pos else floats[0]

        # Numeric match (handles integers and floats with tolerance)
        idx = _match_options(value, options)
        if idx is not None:
            return idx

        # Fraction fallback (for options like \frac{1}{3})
        try:
            from ..tools.sympy_direct_tool import SympyDirectTool
            frac = Fraction(value).limit_denominator(1000)
            idx = SympyDirectTool._match_fraction(frac.numerator, frac.denominator, options)
        except Exception:
            idx = None
        return idx

    @staticmethod
    def _pinned(pinned: StrategyOutput, reason: str) -> StrategyOutput:
        return StrategyOutput(
            chosen_index=pinned.chosen_index,
            confidence=pinned.confidence,
            rationale=pinned.rationale,
            extras={**(pinned.extras or {}), "path": "pinned", "pinned_reason": reason},
        )
