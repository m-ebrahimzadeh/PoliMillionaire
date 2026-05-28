"""SympyDirectTool: symbolic solver that runs before the LLM, no agent loop.

Extends MathsTool's coverage to question types that safe_eval cannot handle:
  - Equations / unknowns   "If 3x(x+5) = 2(3x+x+5), find x"
  - Modular arithmetic     "What is the remainder when 2^87 is divided by 7?"
  - Combinatorics          "In how many ways can 3 be chosen from 25, of which 5 are damaged?"
  - Polynomial roots       "What is the maximum of 4(x+7)(2-x)?"
  - Geometry               "Opposite vertices (1,2) and (7,4) — area of square?"

Design: same precision-over-recall contract as MathsTool.
  - No LLM involved at any point — purely deterministic.
  - Abstains immediately on any parse/eval failure.
  - Only returns an answer when the numeric result matches an option exactly.
  - Sub-millisecond on warm SymPy; ~2s cold import (lazy, paid once per session).
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..config import Category
from ..strategies.base import StrategyInput, StrategyOutput
from .base import Tool
from .maths_tool import _parse_option_value, _match_options
from .sympy_tool import sympy_solve, _BLOCKED


# ── Pattern library ───────────────────────────────────────────────────────────
# Each entry is (compiled_regex, builder_fn).
# builder_fn(match) → sympy expression string, or None to skip.
# Patterns are tried in order; first match that produces a valid SymPy result
# and aligns with an option wins.  Conservative: abstain on anything ambiguous.

def _mod_pattern():
    """remainder when A^B is divided by C  →  Mod(A**B, C)"""
    pat = re.compile(
        r'remainder\s+when\s+(\d+)\s*\^?\*?\*?\s*(\d+)\s+is\s+divided\s+by\s+(\d+)',
        re.I,
    )
    def build(m): return f"Mod({m.group(1)}**{m.group(2)}, {m.group(3)})"
    return pat, build

def _comb_pattern():
    """C(n, k) phrasing — "chosen from", "ways to choose", "combinations of" """
    pat = re.compile(
        r'(\d+)\s+(?:be\s+)?chosen\s+from\s+(\d+)'
        r'|ways?\s+to\s+choose\s+(\d+)\s+from\s+(\d+)'
        r'|combinations?\s+of\s+(\d+)\s+(?:from|out\s+of)\s+(\d+)',
        re.I,
    )
    def build(m):
        groups = [g for g in m.groups() if g is not None]
        if len(groups) >= 2:
            k, n = int(groups[0]), int(groups[1])
            return f"binomial({n}, {k})"
        return None
    return pat, build

def _perm_pattern():
    """arrangements / permutations of n distinct items"""
    pat = re.compile(
        r'(\d+)\s+(?:different\s+)?(?:items?|books?|people|objects?|letters?)'
        r'\s+(?:be\s+)?arranged',
        re.I,
    )
    def build(m): return f"factorial({m.group(1)})"
    return pat, build

def _max_poly_pattern():
    """maximum value of a polynomial in one variable"""
    pat = re.compile(
        r'maximum\s+value\s+of\s+(.+?)(?:\s*,\s*over|\s*for\s+all|\s*\?|$)',
        re.I,
    )
    def build(m):
        expr = m.group(1).strip().rstrip('?').strip()
        # Rewrite to SymPy: find critical point, evaluate
        # Return a special marker so _eval_max() handles it
        return f"__max__:{expr}"
    return pat, build

def _area_square_from_vertices_pattern():
    """area of square given two opposite vertices — any word order."""
    # Matches any question containing two coordinate pairs and 'opposite vertices'
    pat = re.compile(
        r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
        r'(?:.*?)'
        r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
        r'(?=.*?(?:opposite\s+vertices|vertices\s+of\s+a\s+square))',
        re.I | re.DOTALL,
    )
    def build(m):
        x1, y1 = float(m.group(1)), float(m.group(2))
        x2, y2 = float(m.group(3)), float(m.group(4))
        diag_sq = (x2 - x1)**2 + (y2 - y1)**2
        area = diag_sq / 2
        return f"__literal__:{area}"
    return pat, build

def _linear_interp_pattern():
    """linear decrease from year Y1 (value V1) to year Y2 (value V2) — find value at Y3"""
    pat = re.compile(
        r'(\d{4}).*?(\d[\d,]*)\s+cases.*?(\d{4}).*?(\d[\d,]*)\s+cases'
        r'.*?(\d{4})',
        re.I | re.DOTALL,
    )
    def build(m):
        y1, v1 = int(m.group(1)), int(m.group(2).replace(',', ''))
        y2, v2 = int(m.group(3)), int(m.group(4).replace(',', ''))
        y3 = int(m.group(5))
        if y1 == y2 or not (min(y1, y2) <= y3 <= max(y1, y2)):
            return None
        value = v1 + (v2 - v1) * (y3 - y1) / (y2 - y1)
        return f"__literal__:{value}"
    return pat, build

def _latex_binom_pattern():
    r"""\\binom{n}{k} or \\dbinom{n}{k} outside of $...$."""
    pat = re.compile(r'\\d?binom\s*\{(\d+)\}\s*\{(\d+)\}', re.I)
    def build(m): return f"binomial({m.group(1)}, {m.group(2)})"
    return pat, build

def _latex_log_pattern():
    r"""\\log_{base} arg  e.g.  \log_8 2"""
    pat = re.compile(r'\\log_\{?(\d+)\}?\s+(\d+)', re.I)
    def build(m): return f"log({m.group(2)}, {m.group(1)})"
    return pat, build

def _direct_sympy_pattern():
    """LaTeX inline $expr$ — short purely-numeric/algebraic expressions only."""
    pat = re.compile(r'\$([^$]{1,80})\$', re.I)
    def build(m):
        expr = m.group(1).strip()
        # Skip if it contains prose words or LaTeX commands (handled by dedicated patterns)
        if re.search(r'[a-zA-Z]{4,}', expr):
            return None
        expr = (expr
                .replace('^', '**')
                .replace('\\cdot', '*')
                .replace('\\times', '*')
                .replace('\\div', '/')
                .replace('{', '(').replace('}', ')')
                .replace('\\', ''))
        return expr
    return pat, build


_PATTERNS = [
    _mod_pattern(),
    _latex_binom_pattern(),      # before _comb_pattern and _direct_sympy_pattern
    _latex_log_pattern(),        # before _direct_sympy_pattern
    _comb_pattern(),
    _perm_pattern(),
    _max_poly_pattern(),
    _area_square_from_vertices_pattern(),
    _linear_interp_pattern(),
    _direct_sympy_pattern(),
]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _eval_max(expr_body: str) -> Optional[float]:
    """Find the maximum of a single-variable polynomial expression."""
    try:
        sp = __import__('sympy')
        # Detect the free variable (x is conventional)
        x = sp.Symbol('x')
        ns = {str(x): x, 'pi': sp.pi, 'e': sp.E}
        body = (expr_body.strip()
                .replace('^', '**')
                .replace('{', '(').replace('}', ')')
                .replace('\\', ''))
        parsed = sp.sympify(body, locals=ns)
        if not parsed.free_symbols:
            return float(sp.N(parsed, 15))
        var = list(parsed.free_symbols)[0]
        deriv = sp.diff(parsed, var)
        crits = sp.solve(deriv, var)
        if not crits:
            return None
        vals = [float(sp.N(parsed.subs(var, c), 15)) for c in crits
                if sp.im(c) == 0]  # real critical points only
        return max(vals) if vals else None
    except Exception:
        return None


def _try_evaluate(expr_str: str) -> Optional[float]:
    """Evaluate a SymPy expression string to a float. Returns None on any error."""
    if not expr_str:
        return None

    # Special markers set by pattern builders
    if expr_str.startswith('__literal__:'):
        try:
            return float(expr_str[len('__literal__:'):])
        except ValueError:
            return None

    if expr_str.startswith('__max__:'):
        return _eval_max(expr_str[len('__max__:'):])

    # Safety gate — reuse sympy_tool's block list
    if _BLOCKED.search(expr_str):
        return None

    try:
        result_str = sympy_solve(expr_str)
        # sympy_solve returns a string; parse the first float from it
        # Handles: "5", "120", "[3, -3]", "78.539..."
        nums = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', result_str)
        if not nums:
            return None
        # Prefer positive solutions (most MCQ answers are positive)
        floats = [float(n) for n in nums]
        pos = [f for f in floats if f >= 0]
        return pos[0] if pos else floats[0]
    except Exception:
        return None


# ── Tool ─────────────────────────────────────────────────────────────────────

class SympyDirectTool(Tool):
    """Symbolic solver that runs before the LLM — no agent loop, no network.

    Tries each pattern in order against the question text. First pattern that
    produces a numeric result matching an option returns with confidence=0.97.
    Abstains (returns None) if no pattern fires or no option matches.

    Sits after MathsTool in the ToolStrategy chain:
      MathsTool  →  SympyDirectTool  →  BaselineLLMStrategy
    MathsTool covers simple arithmetic; SympyDirectTool covers symbolic/modular/
    combinatorial forms; baseline covers everything else.
    """
    name = "sympy_direct"

    def can_handle(self, inp: StrategyInput) -> bool:
        return inp.category == Category.MATHS

    def use(self, inp: StrategyInput) -> Optional[StrategyOutput]:
        question = inp.question

        for pattern, builder in _PATTERNS:
            m = pattern.search(question)
            if not m:
                continue
            expr = builder(m)
            if not expr:
                continue
            result = _try_evaluate(expr)
            if result is None or not math.isfinite(result):
                continue
            idx = _match_options(result, inp.options)
            if idx is not None:
                return StrategyOutput(
                    chosen_index=idx,
                    confidence=0.97,
                    rationale=f"SympyDirectTool: {expr} = {result}",
                    extras={"tool": "sympy_direct", "expr": expr, "result": result},
                )

        return None  # no pattern matched or no option aligned → abstain
