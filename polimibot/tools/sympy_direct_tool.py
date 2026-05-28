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


# ── New patterns from live-game gap analysis ──────────────────────────────────

def _equation_solve_pattern():
    """Single-variable equation: 'If A = B, then x = ?'
    Covers: '3^(x-3) + 10 = 19', '2x^2+5x+12 = 19-7x', etc."""
    pat = re.compile(
        r'(?:if\s+)?'
        r'([0-9x\^+\-*/().\s]+=[0-9x\^+\-*/().\s]+)'
        r'(?:,?\s*(?:then\s+)?(?:what\s+is|find|solve\s+for)?\s*x\s*=)?',
        re.I,
    )
    def build(m):
        raw = m.group(1).strip()
        if 'x' not in raw.lower():
            return None
        if '=' in raw and '==' not in raw:
            parts = raw.split('=', 1)
            lhs = re.sub(r'\s+', '', parts[0]).replace('^', '**')
            rhs = re.sub(r'\s+', '', parts[1]).replace('^', '**')
            # Prose check on the raw equation parts only (before adding the marker)
            combined = lhs + rhs
            if len(combined) > 100 or re.search(r'[a-wyzA-WYZ]{2,}', combined):
                return None
            return f"__eq_solve__:({lhs}) - ({rhs})"
        else:
            expr = raw.replace('^', '**')
            if len(expr) > 100 or re.search(r'[a-wyzA-WYZ]{2,}', expr):
                return None
            return expr
    return pat, build


def _quadratic_difference_pattern():
    """Positive difference between roots of ax^2+bx+c=0."""
    pat = re.compile(
        r'(?:quadratic|equation)\s+'
        r'([0-9x\^+\-*/().\s]+=\s*[0-9x\^+\-*/().\s]+)'
        r'.*?(?:positive\s+difference|difference\s+between\s+(?:the\s+)?(?:two\s+)?solutions?)',
        re.I | re.DOTALL,
    )
    def build(m):
        raw = m.group(1).strip()
        if '=' not in raw or 'x' not in raw.lower():
            return None
        parts = raw.split('=', 1)
        lhs = re.sub(r'\s+', '', parts[0]).replace('^', '**')
        rhs = re.sub(r'\s+', '', parts[1]).replace('^', '**')
        return f"__diff_roots__:({lhs}) - ({rhs})"
    return pat, build


_WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}

def _lcm_backup_pattern():
    """LCM / simultaneous events: 'every N minutes ... every M minutes ... same time'
    Handles both digit and word forms: 'every five minutes'."""
    num = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
    pat = re.compile(
        rf'every\s+{num}\s+minutes?.*?every\s+{num}\s+minutes?'
        r'.*?(?:same\s+time|simultaneously|together|how\s+many\s+times)',
        re.I | re.DOTALL,
    )
    def build(m):
        def to_int(s):
            return _WORD_TO_NUM.get(s.lower(), None) or int(s)
        try:
            a, b = to_int(m.group(1)), to_int(m.group(2))
        except (ValueError, TypeError):
            return None
        return f"__lcm_count__:{a},{b}"
    return pat, build


_NUM_WORDS = {
    'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
    'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
    'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
    'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,
    'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,
}

def _words_to_digits(text: str) -> str:
    """Replace spelled-out numbers with digits. Handles simple cases only."""
    def replace(m):
        w = m.group(0).lower()
        return str(_NUM_WORDS[w]) if w in _NUM_WORDS else m.group(0)
    return re.sub(r'\b(' + '|'.join(_NUM_WORDS.keys()) + r')\b', replace, text, flags=re.I)


def _inclusion_exclusion_pattern():
    """Simple inclusion-exclusion: total=N, |A|=a, |B|=b, |neither|=k → |A∩B|.
    Handles both digit and word forms ('twenty boxes', '13 boxes')."""
    pat = re.compile(
        r'(?:has\s+|are\s+)?(\d+)\s+(?:total\s+)?(?:boxes?|students?|people|items?|objects?|members?)'
        r'.*?(\d+).*?contain'
        r'.*?(\d+).*?contain'
        r'.*?(\d+).*?neither',
        re.I | re.DOTALL,
    )
    def build(m):
        total   = int(m.group(1))
        a       = int(m.group(2))
        b       = int(m.group(3))
        neither = int(m.group(4))
        both = a + b - (total - neither)
        if both < 0:
            return None
        return f"__literal__:{both}"

    class _WordNormPattern:
        """Wraps the regex to first normalize number words to digits."""
        def __init__(self, inner_pat, inner_build):
            self._pat = inner_pat
            self._build = inner_build
            self.pattern = inner_pat.pattern  # for debug printing
        def search(self, text):
            return self._pat.search(_words_to_digits(text))

    return _WordNormPattern(pat, build), build


def _r_squared_ratio_pattern():
    """R² ratio: 'correlation r1 ... correlation r2 ... how many times'
    Answer = (r1/r2)^2"""
    pat = re.compile(
        r'correlation\s+of\s+(\d+(?:\.\d+)?)'
        r'(?:.*?)'
        r'correlation\s+of\s+(\d+(?:\.\d+)?)',
        re.I | re.DOTALL,
    )
    def build(m):
        r1, r2 = float(m.group(1)), float(m.group(2))
        if r2 == 0:
            return None
        ratio = (r1 / r2) ** 2
        return f"__literal__:{ratio}"
    return pat, build


def _sum_of_squares_formula_pattern():
    """Sum-of-squares telescoping: given formula for 1^2+...+n^2, find 21^2+...+40^2"""
    pat = re.compile(
        r'(\d+)\^2\s*\+\s*(\d+)\^2\s*\+.*?\+\s*(\d+)\^2',
        re.I,
    )
    def build(m):
        # Extract all exponent bases from the question
        all_nums = re.findall(r'(\d+)\^2', m.string)
        if len(all_nums) < 2:
            return None
        nums = [int(n) for n in all_nums]
        low, high = min(nums), max(nums)
        if high - low < 2:
            return None
        # sum(k^2, k=low..high) = sum(k^2, k=1..high) - sum(k^2, k=1..low-1)
        def s(n): return n * (n + 1) * (2 * n + 1) // 6
        result = s(high) - s(low - 1)
        return f"__literal__:{result}"
    return pat, build


def _bare_expression_pattern():
    """Bare arithmetic/complex expression ending with '='.
    Catches: '(i + 1)(5 - 5i)(5 + 5i) =', '(2+3i)(1-i) ='"""
    pat = re.compile(
        r'([()\d\s+\-*/.,i^]+)'     # expression containing only numeric/operator chars
        r'\s*=\s*$',                # ends with '='
        re.I | re.MULTILINE,
    )
    def build(m):
        expr = m.group(1).strip()
        if not expr or len(expr) > 80:
            return None
        if not re.search(r'[\di]', expr, re.I):
            return None
        expr = (expr
                .replace('–', '-').replace('−', '-')  # en-dash, minus sign
                .replace('^', '**')
                .replace('i', 'I'))
        expr = _insert_mul(expr)
        return expr
    return pat, build


_PATTERNS = [
    _mod_pattern(),
    _latex_binom_pattern(),           # before _comb_pattern and _direct_sympy_pattern
    _latex_log_pattern(),             # before _direct_sympy_pattern
    _sum_of_squares_formula_pattern(),# specific — before generic equation solver
    _quadratic_difference_pattern(),  # specific — before generic equation solver
    _lcm_backup_pattern(),
    _inclusion_exclusion_pattern(),
    _r_squared_ratio_pattern(),
    _comb_pattern(),
    _perm_pattern(),
    _equation_solve_pattern(),        # generic — after specific patterns
    _max_poly_pattern(),
    _area_square_from_vertices_pattern(),
    _linear_interp_pattern(),
    _bare_expression_pattern(),       # last — most permissive
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


def _insert_mul(expr: str) -> str:
    """Insert explicit * where Python/SymPy requires it but humans omit it.
    '2x' → '2*x', ')('' → ')*(', but never breaks function names like 'Mod('.
    """
    # digit immediately followed by 'x' (the unknown variable): 2x → 2*x
    expr = re.sub(r'(\d)\s*(x)\b', r'\1*\2', expr, flags=re.I)
    # closing paren immediately before opening paren: )( → )*(
    expr = re.sub(r'\)\s*\(', r')*(', expr)
    # single letter (not part of a longer word) followed by (: x( → x*(
    # Use negative lookbehind to avoid matching multi-char function names
    expr = re.sub(r'(?<![a-zA-Z])([a-zA-Z])\s*\(', r'\1*(', expr)
    return expr


def _eval_eq_solve(expr_body: str) -> Optional[float]:
    """Solve a single-variable equation expressed as LHS - RHS = 0."""
    try:
        sp = __import__('sympy')
        x = sp.Symbol('x')
        body = _insert_mul(expr_body)
        parsed = sp.sympify(body, locals={'x': x})
        solutions = sp.solve(parsed, x)
        real_sols = [float(sp.re(s)) for s in solutions if sp.Abs(sp.im(s)) < 1e-9]
        if not real_sols:
            return None
        # Return the positive solution if available, otherwise the first
        pos = [s for s in real_sols if s > 0]
        return pos[0] if pos else real_sols[0]
    except Exception:
        return None


def _eval_diff_roots(expr_body: str) -> Optional[float]:
    """Return |r1 - r2| for a single-variable polynomial (positive root difference)."""
    try:
        sp = __import__('sympy')
        x = sp.Symbol('x')
        body = _insert_mul(expr_body)
        parsed = sp.sympify(body, locals={'x': x, 'X': x})
        roots = sp.solve(parsed, x)
        real_roots = [float(sp.re(r)) for r in roots if sp.Abs(sp.im(r)) < 1e-9]
        if len(real_roots) < 2:
            return None
        return abs(max(real_roots) - min(real_roots))
    except Exception:
        return None


def _eval_lcm_count(params: str) -> Optional[float]:
    """Number of coincidences in 24h for two periodic events with periods a,b minutes."""
    try:
        a, b = (int(x) for x in params.split(','))
        import math as _math
        lcm = a * b // _math.gcd(a, b)
        count = (24 * 60) // lcm
        return float(count)
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

    if expr_str.startswith('__diff_roots__:'):
        return _eval_diff_roots(expr_str[len('__diff_roots__:'):])

    if expr_str.startswith('__eq_solve__:'):
        return _eval_eq_solve(expr_str[len('__eq_solve__:'):])

    if expr_str.startswith('__lcm_count__:'):
        return _eval_lcm_count(expr_str[len('__lcm_count__:'):])

    # Safety gate — reuse sympy_tool's block list
    if _BLOCKED.search(expr_str):
        return None

    try:
        result_str = sympy_solve(_insert_mul(expr_str))
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

    @staticmethod
    def _to_sympy_complex(s: str):
        """Parse a complex expression string into a SymPy object."""
        sp = __import__('sympy')
        # Normalise: i/j → I, insert * before I when preceded by a digit
        s = s.replace('j', 'i').replace('i', 'I')
        s = re.sub(r'(\d)\s*I\b', r'\1*I', s)  # '50I' → '50*I', '5I' → '5*I'
        s = _insert_mul(s)
        return sp.sympify(s)

    @staticmethod
    def _match_complex(result_str: str, options: tuple) -> Optional[int]:
        """Match a complex-number result string against options like '50+50i'."""
        try:
            sp = __import__('sympy')
            r = SympyDirectTool._to_sympy_complex(result_str)
            re_val = float(sp.re(r))
            im_val = float(sp.im(r))
        except Exception:
            return None
        for i, opt in enumerate(options):
            try:
                o = SympyDirectTool._to_sympy_complex(opt)
                if (abs(float(sp.re(o)) - re_val) < 1e-6 and
                        abs(float(sp.im(o)) - im_val) < 1e-6):
                    return i
            except Exception:
                continue
        return None

    def use(self, inp: StrategyInput) -> Optional[StrategyOutput]:
        question = inp.question

        for pattern, builder in _PATTERNS:
            m = pattern.search(question)
            if not m:
                continue
            expr = builder(m)
            if not expr:
                continue

            # Try real numeric match first
            result = _try_evaluate(expr)
            if result is not None and math.isfinite(result):
                idx = _match_options(result, inp.options)
                if idx is not None:
                    return StrategyOutput(
                        chosen_index=idx,
                        confidence=0.97,
                        rationale=f"SympyDirectTool: {expr} = {result}",
                        extras={"tool": "sympy_direct", "expr": expr, "result": result},
                    )

            # Try complex match as fallback (e.g. options like '50+50i')
            raw_expr = expr
            if not expr.startswith('__'):
                idx = self._match_complex(raw_expr, inp.options)
                if idx is not None:
                    return StrategyOutput(
                        chosen_index=idx,
                        confidence=0.97,
                        rationale=f"SympyDirectTool[complex]: {expr}",
                        extras={"tool": "sympy_direct", "expr": expr, "result": "complex"},
                    )

        return None  # no pattern matched or no option aligned → abstain
