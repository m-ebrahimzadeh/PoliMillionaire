"""MathsTool: deterministic arithmetic solver for maths-category questions.

Pipeline:
  1. Normalize NL text to Python-arithmetic syntax ("15% of 200" → "15/100*200")
  2. Extract the expression (regex — conservative, abstains readily)
  3. Evaluate with safe_eval
  4. Match result against numeric option values
  5. Return exact match or None (abstain)

Design choice: precision over recall.
If we can't parse confidently, we abstain and let the LLM handle it.
A wrong computation is worse than a wrong LLM guess (both wrong, but
one is confidently wrong — hurts calibration).
"""
from __future__ import annotations

import re
from typing import Optional

from ..config import Category
from ..strategies.base import StrategyInput, StrategyOutput
from .base import Tool
from .calculator import safe_eval


# ── Text → arithmetic normalization ──────────────────────────────────────────
# Applied in order. Patterns are conservative: only match when unambiguous.

_NORMALIZATIONS = [
    # Percentage: "15% of 200" → "15/100*200"
    (re.compile(r'(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)', re.I), r'\1/100*\2'),
    (re.compile(r'(\d+(?:\.\d+)?)\s*percent\s+of\s+(\d+(?:\.\d+)?)', re.I), r'\1/100*\2'),
    # Power: "2 to the power of 8" → "2**8"
    (re.compile(r'(\d+(?:\.\d+)?)\s+to\s+the\s+power\s+of\s+(\d+(?:\.\d+)?)', re.I), r'\1**\2'),
    # Square root: "square root of 144" → "sqrt(144)"
    (re.compile(r'square\s+root\s+of\s+(\d+(?:\.\d+)?)', re.I), r'sqrt(\1)'),
    # Named powers
    (re.compile(r'\bsquared\b', re.I), '**2'),
    (re.compile(r'\bcubed\b', re.I), '**3'),
    # Verbal operators
    (re.compile(r'\btimes\b', re.I), '*'),
    (re.compile(r'\bdivided\s+by\b', re.I), '/'),
    (re.compile(r'\bplus\b', re.I), '+'),
    (re.compile(r'\bminus\b', re.I), '-'),
    # "×" and "÷" symbols
    (re.compile(r'×'), '*'),
    (re.compile(r'÷'), '/'),
]

# Question patterns that signal a computable expression follows
_COMPUTE_PREFIX = re.compile(
    r'(?:what\s+is|calculate|compute|find\s+the\s+value\s+of|evaluate|simplify)'
    r'\s+(.+)',
    re.IGNORECASE,
)

# Strip anything that isn't arithmetic syntax after normalization
_KEEP = re.compile(r'[^0-9+\-*/().,a-zA-Z_\s]')


def _normalize(text: str) -> str:
    for pattern, repl in _NORMALIZATIONS:
        text = pattern.sub(repl, text)
    return text


def _extract_expression(question: str) -> Optional[str]:
    """Normalize and extract an arithmetic expression. Returns None if ambiguous."""
    q = _normalize(question)
    m = _COMPUTE_PREFIX.search(q)
    if not m:
        return None
    fragment = m.group(1).strip().rstrip('?').strip()
    # Remove trailing prose ("...where x = 5")
    fragment = re.split(r'\bwhere\b|\bif\b|\bgiven\b', fragment)[0].strip()
    # Drop characters that aren't arithmetic
    cleaned = _KEEP.sub(' ', fragment).strip()
    # Collapse whitespace but keep multi-char function names (sqrt, factorial, ...)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else None


def _parse_option_value(text: str) -> Optional[float]:
    """Parse an option string as a float. Returns None if non-numeric."""
    # Strip currency, commas, trailing units (e.g. "30 kg", "€1,000")
    cleaned = re.sub(r'[€$£,]', '', text)
    cleaned = re.split(r'\s+[a-zA-Z]', cleaned)[0].strip()  # "30 meters" → "30"
    cleaned = cleaned.rstrip('.')
    try:
        return float(cleaned)
    except ValueError:
        return None


def _match_options(
    result: float, options: tuple[str, ...],
) -> Optional[int]:
    """Return 0-based index of the option whose numeric value equals result.
    
    Uses relative tolerance to handle float imprecision.
    Returns None if no option matches or if options aren't all numeric.
    """
    parsed = [_parse_option_value(opt) for opt in options]
    numeric_count = sum(1 for v in parsed if v is not None)
    # Require at least 3 of 4 options to be numeric — otherwise it's a
    # non-numeric question where a number happened to appear in the text.
    if numeric_count < 3:
        return None
    tol = max(1e-9, abs(result) * 1e-9)
    for i, v in enumerate(parsed):
        if v is not None and abs(v - result) <= tol:
            return i
    return None


class MathsTool(Tool):
    """Deterministic arithmetic solver. Only answers when it can prove the answer.
    
    Coverage expectation: ~30–50% of maths questions (those with computable
    answers and numeric options). The rest fall through to the LLM.
    """
    name = "maths_tool"

    def can_handle(self, inp: StrategyInput) -> bool:
        # Only attempt maths-category questions.
        # Fast guard: no regex here.
        return inp.category == Category.MATHS

    def use(self, inp: StrategyInput) -> Optional[StrategyOutput]:
        expr = _extract_expression(inp.question)
        if expr is None:
            return None  # couldn't extract → abstain

        try:
            result = safe_eval(expr)
        except Exception:
            return None  # parse/math error → abstain

        if not isinstance(result, (int, float)) or not _is_finite(result):
            return None

        idx = _match_options(float(result), inp.options)
        if idx is None:
            return None  # computed value doesn't match any option → abstain

        return StrategyOutput(
            chosen_index=idx,
            confidence=0.99,   # deterministic → near-certain
            rationale=f"MathsTool: {expr} = {result}",
            extras={"tool": "maths_tool", "expr": expr, "result": result},
        )


def _is_finite(x: float) -> bool:
    import math
    return math.isfinite(x)