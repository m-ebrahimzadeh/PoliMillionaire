"""SympyDirectTool: symbolic solver that runs before the LLM, no agent loop.

Extends MathsTool's coverage to question types that safe_eval cannot handle:
  - Equations / unknowns   "If 3x(x+5) = 2(3x+x+5), find x"
  - Modular arithmetic     "What is the remainder when 2^87 is divided by 7?"
  - Combinatorics          "In how many ways can 3 be chosen from 25, of which 5 are damaged?"
  - Polynomial roots       "What is the maximum of 4(x+7)(2-x)?"
  - Geometry               "Opposite vertices (1,2) and (7,4) — area of square?"
  - Repeating decimals     "Express 0.1̄7̄ as a fraction" / "reciprocal of 0.7̄"
  - Linear systems         "725x+727y=1500 and 729x+731y=1508, find x-y"
  - LaTeX expressions      "Evaluate log_8 2", "range of y=5+3sin(π-x)", nested fractions

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
    r"""remainder when EXPR is divided by C  →  Mod(EXPR, C)

    Handles:
      "remainder when 2^87 is divided by 7"          → Mod(2**87, 7)
      "remainder when $2^{87}+3$ is divided by $7$"  → Mod(2**87+3, 7)
      "remainder when 2^87 - 1 is divided by 7"      → Mod(2**87-1, 7)
    """
    pat = re.compile(
        r'remainder\s+when\s+'
        r'\$?'                                          # optional opening $
        r'(\d+)'                                        # base
        r'\s*(?:\^|\*\*)\s*\{?(\d+)\}?'               # ^B or **B (with optional {})
        r'\s*([+\-]\s*\d+)?'                            # optional +C or -C
        r'\$?'                                          # optional closing $
        r'\s+is\s+divided\s+by\s+\$?(\d+)\$?',        # divided by D
        re.I,
    )
    def build(m):
        base     = m.group(1)
        exp      = m.group(2)
        addend   = m.group(3)          # e.g. "+3" or "-1", may be None
        divisor  = m.group(4)
        expr = f"{base}**{exp}"
        if addend:
            expr = f"{expr}{addend.replace(' ', '')}"
        return f"Mod({expr}, {divisor})"
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
    """maximum value of a polynomial in one variable.
    Strips surrounding $...$ LaTeX delimiters before passing to _eval_max."""
    pat = re.compile(
        r'maximum\s+(?:possible\s+)?value\s+of\s+(.+?)(?:\s*,\s*over|\s*for\s+all|\s*\?|$)',
        re.I,
    )
    def build(m):
        expr = m.group(1).strip().rstrip('?').strip()
        # Strip LaTeX $...$ delimiters if present
        expr = re.sub(r'^\$(.+)\$$', r'\1', expr)
        # Convert LaTeX notation
        expr = _latex_to_sympy(expr)
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
        # Guard: reject bare single-variable expressions (e.g. just "x") —
        # these produce meaningless results and cause false positives.
        stripped = expr.strip()
        if re.fullmatch(r'[a-zA-Z]', stripped):
            return None
        # Must contain at least one digit or operator to be worth evaluating.
        if not re.search(r'[\d+\-*/]', stripped):
            return None
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


# ── Addition 1: Repeating decimal patterns ────────────────────────────────────

def _repeating_decimal_pattern():
    r"""Convert repeating-decimal notation to a fraction.

    Handles three forms that appear in live questions:
      \overline{d}          → pure repeat:   0.\overline{7}   = 7/9
      d.\overline{d}        → mixed:         0.1\overline{7}  = 8/45
      reciprocal of …       → wraps result:  reciprocal of 0.\overline{7} = 9/7

    Algorithm (no SymPy needed — pure integer arithmetic):
      For  A.BC̄  (A = integer part, B = non-repeating decimals, C = repeating block):
        value = (ABC - AB) / (99...9 × 10^len(B))
      where the number of 9s equals len(C).
    """
    pat = re.compile(
        r'(reciprocal\s+of\s+)?'              # optional "reciprocal of"
        r'(\d+(?:\.\d*)?)'                    # leading digits, e.g. "0.1" or "0"
        r'\\overline\s*\{(\d+)\}',            # \overline{repeating block}
        re.I,
    )
    def build(m):
        want_reciprocal = bool(m.group(1))
        leading = m.group(2)          # e.g. "0.1"
        repeat  = m.group(3)          # e.g. "7"

        # Split leading into integer part and non-repeating decimal part
        if '.' in leading:
            int_part, non_rep = leading.split('.', 1)
        else:
            int_part, non_rep = leading, ''

        int_val  = int(int_part) if int_part else 0
        len_nr   = len(non_rep)
        len_rep  = len(repeat)

        # Numerator = (all digits without decimal) - (digits without repeating part)
        all_digits = int(int_part + non_rep + repeat) if (int_part + non_rep + repeat) else 0
        no_rep_digits = int(int_part + non_rep) if (int_part + non_rep) else 0

        numer = all_digits - no_rep_digits
        denom = int('9' * len_rep) * (10 ** len_nr)

        if denom == 0:
            return None

        # Reduce fraction
        from math import gcd
        g = gcd(abs(numer), denom)
        numer //= g
        denom //= g

        if want_reciprocal:
            if numer == 0:
                return None
            numer, denom = denom, numer

        return f"__fraction__:{numer}/{denom}"
    return pat, build


# ── Addition 2: Linear system of equations ────────────────────────────────────

def _linear_system_pattern():
    r"""Two-equation linear system → solve for a simple expression of the unknowns.

    Matches questions like:
      "725x + 727y = 1500 and 729x + 731y = 1508, what is x - y?"
      "$725x + 727y = 1500$ and $729x + 731y = 1508$. Find x + y."

    Extracts coefficients from the two equations, solves with SymPy, then
    evaluates the requested linear combination (x-y, x+y, x, y, etc.).
    """
    # Matches two equations each of the form  Ax ± By = C  (integer coefficients)
    _eq = r'(-?\d+)\s*x\s*([+\-])\s*(\d+)\s*y\s*=\s*(-?\d+)'
    pat = re.compile(
        rf'{_eq}'                                       # equation 1
        r'(?:\s*(?:and|,|\.|;)\s*\$?)'                  # separator
        rf'{_eq}'                                       # equation 2
        r'.*?(?:find|what\s+is|compute|determine)'      # question verb
        r'\s*(?:the\s+(?:value\s+of\s+)?)?'
        r'(x\s*[-+]\s*y|x\s*\*\s*y|x|y)',              # what to evaluate
        re.I | re.DOTALL,
    )
    def build(m):
        a1  = int(m.group(1))
        s1  = 1 if m.group(2) == '+' else -1
        b1  = s1 * int(m.group(3))
        c1  = int(m.group(4))
        a2  = int(m.group(5))
        s2  = 1 if m.group(6) == '+' else -1
        b2  = s2 * int(m.group(7))
        c2  = int(m.group(8))
        target = re.sub(r'\s+', '', m.group(9).lower())  # 'x-y', 'x+y', 'x', 'y'
        return f"__linsys__:{a1},{b1},{c1},{a2},{b2},{c2},{target}"
    return pat, build


# ── Addition 3: LaTeX expression evaluator ────────────────────────────────────

# Maps LaTeX commands to SymPy-compatible equivalents.
# Applied before sympify so the expression evaluates correctly.
_LATEX_REPLACEMENTS = [
    # Trig functions
    (re.compile(r'\\sin\b'),  'sin'),
    (re.compile(r'\\cos\b'),  'cos'),
    (re.compile(r'\\tan\b'),  'tan'),
    (re.compile(r'\\ln\b'),   'log'),
    (re.compile(r'\\exp\b'),  'exp'),
    (re.compile(r'\\sqrt\b'), 'sqrt'),
    # Constants
    (re.compile(r'\\pi\b'),   'pi'),
    (re.compile(r'\\infty\b'),'oo'),
    # Operators
    (re.compile(r'\\cdot\b'), '*'),
    (re.compile(r'\\times\b'),'*'),
    (re.compile(r'\\div\b'),  '/'),
    (re.compile(r'\\pm\b'),   '+'),   # approximate — abstain if result ambiguous
    # Fractions: \frac{a}{b} → (a)/(b)
    # Applied iteratively to handle nested fractions
    (re.compile(r'\\[dc]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}'), r'(\1)/(\2)'),
    # \left( \right) — remove sizing commands
    (re.compile(r'\\(?:left|right)\s*([(){}\[\]|])'), r'\1'),
    # Superscripts: ^{expr} → **(expr)
    (re.compile(r'\^\{([^{}]+)\}'), r'**(\1)'),
    # Subscripts in log: \log_{base} → handled by dedicated pattern; strip here
    (re.compile(r'\\log_\{?(\w+)\}?'), r'log_\1_'),  # marker, resolved below
    # Remove remaining backslash commands we don't recognise
    (re.compile(r'\\[a-zA-Z]+'), ''),
    # Braces → parens
    (re.compile(r'\{'), '('),
    (re.compile(r'\}'), ')'),
]

_LOG_BASE_MARKER = re.compile(r'log_(\w+)_\s*\(?(\w+)\)?')


def _latex_to_sympy(expr: str) -> str:
    """Convert a LaTeX math expression string to a SymPy-parseable string."""
    # Resolve nested \frac / \cfrac / \dfrac from innermost outward.
    # Repeat until no more \frac patterns remain (handles arbitrary nesting depth).
    _frac_pat = re.compile(r'\\[dc]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
    for _ in range(8):
        new_expr = _frac_pat.sub(r'(\1)/(\2)', expr)
        if new_expr == expr:
            break
        expr = new_expr

    # Apply remaining replacements once
    for pat, repl in _LATEX_REPLACEMENTS:
        # Skip frac — already handled above
        if hasattr(pat, 'pattern') and 'frac' in pat.pattern:
            continue
        expr = pat.sub(repl, expr)

    # Resolve log_base_ markers: log_8_(2) → log(2,8)
    expr = _LOG_BASE_MARKER.sub(lambda m: f"log({m.group(2)},{m.group(1)})", expr)
    expr = _insert_mul(expr)
    return expr.strip()


def _latex_expr_pattern():
    r"""Evaluate a LaTeX expression from a question of the form:
      "Evaluate $\log_8 2$"
      "Find $-\frac{1}{-3}\cdot\frac{1}{\frac{1}{-3}}$"
      "What is the range of y = 5 + 3\sin(\pi - x)?"  [range queries handled separately]
      "Calculate $\left(\frac{1}{a}\right)^4 \cdot 2 \cdot a^4 + a^{2+1-3}$ when a=42"

    For range questions the result is a pair; handled by __range__ marker.
    For substitution questions ("when a=42") the variable is substituted first.
    """
    pat = re.compile(
        r'(?:evaluate|calculate|compute|find|simplify)\s+'
        r'\$([^$]{1,200})\$'           # LaTeX expression in $...$
        r'(?:\s+when\s+([a-z])\s*=\s*(-?\d+(?:\.\d+)?))?',  # optional substitution
        re.I,
    )
    def build(m):
        latex = m.group(1).strip()
        sub_var = m.group(2)
        sub_val = m.group(3)
        expr = _latex_to_sympy(latex)
        if sub_var and sub_val:
            return f"__subst__:{expr}|{sub_var}={sub_val}"
        # Reject if still contains long alphabetic words after conversion
        if re.search(r'[a-zA-Z]{5,}', expr):
            return None
        return expr
    return pat, build


def _trig_range_pattern():
    r"""Range of a trig function like y = A + B*sin(expr) or y = A + B*cos(expr).
    Answer is [A-|B|, A+|B|] since sin/cos range is [-1, 1].

    Matches: "range of y = 5 + 3*sin(pi - x)" → [2, 8]
    sin(π - x) = sin(x), so the phase doesn't affect the range.
    """
    pat = re.compile(
        r'range\s+of\s+(?:the\s+function\s+)?'
        r'y\s*=\s*(-?\d+(?:\.\d+)?)\s*([+\-])\s*(\d+(?:\.\d+)?)'
        r'\s*\*?\s*(?:sin|cos)\s*\(',
        re.I,
    )
    def build(m):
        A   = float(m.group(1))
        sgn = 1.0 if m.group(2) == '+' else -1.0
        B   = sgn * float(m.group(3))
        lo  = A - abs(B)
        hi  = A + abs(B)
        return f"__range__:{lo},{hi}"
    return pat, build


# ── Addition 4: three-way LCM (lights blinking at 3 different periods) ───────

def _lcm3_pattern():
    """Three simultaneous periodic events — find how many times they coincide.
    'red blinks every 2 seconds, yellow every 3, blue every 5 ... 7 minutes'
    Handles word and digit period forms; extracts total duration from minutes/seconds.
    """
    num = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
    pat = re.compile(
        rf'every\s+{num}\s+seconds?'
        r'.*?'
        rf'every\s+{num}\s+seconds?'
        r'.*?'
        rf'every\s+{num}\s+seconds?'
        r'.*?'
        r'(\d+)\s*(?:-\s*)?(?:minute|min)\b',
        re.I | re.DOTALL,
    )
    def build(m):
        def to_int(s):
            return _WORD_TO_NUM.get(s.lower(), None) or int(s)
        try:
            a, b, c = to_int(m.group(1)), to_int(m.group(2)), to_int(m.group(3))
            total_min = int(m.group(4))
        except (ValueError, TypeError):
            return None
        return f"__lcm3_count__:{a},{b},{c},{total_min}"
    return pat, build


# ── Addition 5: tangent line to a curve at a point ────────────────────────────

def _tangent_line_pattern():
    r"""Tangent line to y=f(x) at x=a.
    'equation of the line tangent to y = x + e^x at x = 0'
    Differentiates symbolically, evaluates slope and y-intercept, returns
    a __tangent__ marker matched against options of the form 'y = mx + b'.
    """
    pat = re.compile(
        r'(?:equation\s+of\s+)?(?:the\s+)?line\s+tangent\s+to\s+'
        r'(?:the\s+graph\s+of\s+)?'
        r'y\s*=\s*([^,?]+?)'           # f(x) expression
        r'\s+at\s+x\s*=\s*(-?\d+(?:\.\d+)?)',  # x = a
        re.I,
    )
    def build(m):
        fx_raw = m.group(1).strip()
        x_val  = m.group(2).strip()
        # Convert to SymPy-parseable form
        fx = _latex_to_sympy(fx_raw)
        return f"__tangent__:{fx}|{x_val}"
    return pat, build


# ── Addition 6: functional equation h(ax+b)=cx+d → solve h(x)=x ─────────────

def _functional_eq_pattern():
    r"""h(ax+b) = cx+d, find x where h(x)=x.
    'Let h(4x-1) = 2x+7. For what value of x is h(x) = x?'
    Strategy: substitute u=ax+b → x=(u-b)/a, express h(u)=(c(u-b)/a)+d,
    then solve h(x)=x symbolically.
    """
    pat = re.compile(
        r'(?:let\s+)?h\s*\(\s*(-?\d+)\s*x\s*([+\-]\s*\d+)\s*\)'
        r'\s*=\s*(-?\d+)\s*x\s*([+\-]\s*\d+)'
        r'.*?h\s*\(\s*x\s*\)\s*=\s*x',
        re.I | re.DOTALL,
    )
    def build(m):
        a  = int(m.group(1))
        b  = int(m.group(2).replace(' ', ''))   # e.g. "-1"
        c  = int(m.group(3))
        d  = int(m.group(4).replace(' ', ''))   # e.g. "+7"
        return f"__func_eq__:{a},{b},{c},{d}"
    return pat, build


_PATTERNS = [
    # ── Addition 1: repeating decimals (high-confidence failures) ──
    _repeating_decimal_pattern(),
    # ── Addition 2: linear systems ──
    _linear_system_pattern(),
    # ── Addition 3: LaTeX evaluation ──
    _trig_range_pattern(),            # specific trig range — before generic latex
    _latex_expr_pattern(),            # general LaTeX evaluate/calculate
    # ── Additions 4-6: new patterns ──
    _tangent_line_pattern(),          # before equation_solve (more specific)
    _functional_eq_pattern(),         # before equation_solve (more specific)
    _lcm3_pattern(),                  # three-period LCM — before two-period
    # ── Existing patterns ──
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

def _eval_lcm3_count(params: str) -> Optional[float]:
    """Number of simultaneous events for 3 periodic processes in a time window.
    params: 'a,b,c,total_minutes'  (periods in seconds, window in minutes)
    Includes start (t=0) per problem convention.
    """
    try:
        import math as _math
        a, b, c, total_min = (int(x) for x in params.split(','))
        lcm_ab  = a * b // _math.gcd(a, b)
        lcm_abc = lcm_ab * c // _math.gcd(lcm_ab, c)
        total_sec = total_min * 60
        # Number of multiples of lcm_abc in [0, total_sec] inclusive
        count = total_sec // lcm_abc + 1
        return float(count)
    except Exception:
        return None


def _eval_tangent(spec: str) -> Optional[tuple[float, float]]:
    """Differentiate f(x), evaluate at x=a, return (slope, intercept)."""
    try:
        sp = __import__('sympy')
        fx_str, x_str = spec.split('|', 1)
        x   = sp.Symbol('x')
        a   = sp.Rational(x_str.strip())
        fx  = sp.sympify(_insert_mul(fx_str), locals={'x': x, 'e': sp.E, 'pi': sp.pi})
        df  = sp.diff(fx, x)
        m   = float(sp.N(df.subs(x, a), 15))    # slope
        y0  = float(sp.N(fx.subs(x, a), 15))    # y at x=a
        b   = y0 - m * float(a)                 # intercept
        return (m, b)
    except Exception:
        return None


def _eval_func_eq(params: str) -> Optional[float]:
    """Solve h(ax+b)=cx+d for h(x)=x.
    Derives h(u) = c*(u-b)/a + d, then solves c*(x-b)/a + d = x.
    """
    try:
        sp = __import__('sympy')
        a, b, c, d = (int(v) for v in params.split(','))
        x = sp.Symbol('x')
        # h(u) expressed by substituting u = ax+b → x = (u-b)/a into h(ax+b)=cx+d
        u = sp.Symbol('u')
        h_u = sp.Rational(c) * (u - b) / sp.Rational(a) + d
        # Solve h(x) = x
        sols = sp.solve(h_u.subs(u, x) - x, x)
        real_sols = [float(sp.re(s)) for s in sols if sp.Abs(sp.im(s)) < 1e-9]
        if not real_sols:
            return None
        pos = [s for s in real_sols if s > 0]
        return pos[0] if pos else real_sols[0]
    except Exception:
        return None


def _eval_fraction(spec: str) -> Optional[float]:
    """Return float value of a fraction string like '8/45'."""
    try:
        n, d = spec.split('/')
        return float(int(n)) / float(int(d))
    except Exception:
        return None


def _eval_linsys(spec: str) -> Optional[float]:
    """Solve  a1*x + b1*y = c1,  a2*x + b2*y = c2  and evaluate target."""
    try:
        sp = __import__('sympy')
        parts = spec.split(',')
        a1, b1, c1 = int(parts[0]), int(parts[1]), int(parts[2])
        a2, b2, c2 = int(parts[3]), int(parts[4]), int(parts[5])
        target = parts[6]
        x, y = sp.symbols('x y')
        sol = sp.solve([a1*x + b1*y - c1, a2*x + b2*y - c2], [x, y])
        if not sol:
            return None
        xv = float(sol[x])
        yv = float(sol[y])
        if target == 'x-y':   return xv - yv
        if target == 'x+y':   return xv + yv
        if target == 'x*y':   return xv * yv
        if target == 'x':     return xv
        if target == 'y':     return yv
        return None
    except Exception:
        return None


def _eval_subst(spec: str) -> Optional[float]:
    """Evaluate  expr  after substituting  var=val  (e.g. a=42)."""
    try:
        sp = __import__('sympy')
        expr_part, sub_part = spec.split('|', 1)
        var_name, val_str = sub_part.split('=', 1)
        var = sp.Symbol(var_name.strip())
        val = sp.Rational(val_str.strip())
        parsed = sp.sympify(_insert_mul(expr_part), locals={var_name.strip(): var})
        result = parsed.subs(var, val)
        return float(sp.N(result, 15))
    except Exception:
        return None


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
    '2x' → '2*x', '4(' → '4*(', ')('' → ')*(', but never breaks function names.
    """
    # digit immediately followed by 'x' (the unknown variable): 2x → 2*x
    expr = re.sub(r'(\d)\s*(x)\b', r'\1*\2', expr, flags=re.I)
    # digit immediately before opening paren: 4( → 4*(
    expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
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

    if expr_str.startswith('__lcm3_count__:'):
        return _eval_lcm3_count(expr_str[len('__lcm3_count__:'):])

    if expr_str.startswith('__tangent__:'):
        return None  # handled via __tangent__ special path in use()

    if expr_str.startswith('__func_eq__:'):
        return _eval_func_eq(expr_str[len('__func_eq__:'):])

    if expr_str.startswith('__fraction__:'):
        return _eval_fraction(expr_str[len('__fraction__:'):])

    if expr_str.startswith('__linsys__:'):
        return _eval_linsys(expr_str[len('__linsys__:'):])

    if expr_str.startswith('__subst__:'):
        return _eval_subst(expr_str[len('__subst__:'):])

    # __range__ is not a float — handled separately in use()
    if expr_str.startswith('__range__:'):
        return None

    # Reject bare numbers — a lone integer or float is not a meaningful expression
    # and would spuriously match options (e.g. "17" matching option "B. 17").
    if re.fullmatch(r'-?\d+(?:\.\d+)?', expr_str.strip()):
        return None

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

    @staticmethod
    def _match_fraction(numer: int, denom: int, options: tuple) -> Optional[int]:
        """Match a fraction n/d against options that may be written as LaTeX fractions."""
        target = numer / denom
        # Try numeric match first
        for i, opt in enumerate(options):
            # Parse LaTeX fraction: \frac{a}{b}
            fm = re.search(r'\\(?:d?c?frac|frac)\s*\{(-?\d+)\}\s*\{(\d+)\}', opt)
            if fm:
                try:
                    val = int(fm.group(1)) / int(fm.group(2))
                    if abs(val - target) < 1e-9:
                        return i
                except (ValueError, ZeroDivisionError):
                    pass
            # Plain numeric
            try:
                if abs(float(opt) - target) < 1e-9:
                    return i
            except ValueError:
                pass
        return None

    @staticmethod
    def _match_linear(slope: float, intercept: float, options: tuple) -> Optional[int]:
        """Match a tangent line y=mx+b against options like 'y = 2x + 1' or 'y = 2x'."""
        _num = r'(-?\d+(?:\.\d+)?)'
        # Pattern: y = mx + b  or  y = mx  (intercept may be absent if 0)
        pat_full = re.compile(rf'y\s*=\s*{_num}\s*x\s*([+\-])\s*{_num}')
        pat_no_b = re.compile(rf'y\s*=\s*{_num}\s*x\s*$')
        for i, opt in enumerate(options):
            m = pat_full.search(opt)
            if m:
                try:
                    om = float(m.group(1))
                    ob = float(m.group(3)) * (1 if m.group(2) == '+' else -1)
                    if abs(om - slope) < 1e-6 and abs(ob - intercept) < 1e-6:
                        return i
                except ValueError:
                    pass
            m2 = pat_no_b.search(opt)
            if m2:
                try:
                    om = float(m2.group(1))
                    if abs(om - slope) < 1e-6 and abs(intercept) < 1e-6:
                        return i
                except ValueError:
                    pass
        return None

    @staticmethod
    def _match_range(lo: float, hi: float, options: tuple) -> Optional[int]:
        """Match a [lo, hi] range against options like '2 ≤ y ≤ 8' or '-2 <= y <= 8'."""
        _num = r'(-?\d+(?:\.\d+)?)'
        pat = re.compile(rf'{_num}\s*(?:≤|<=|<)\s*\w\s*(?:≤|<=|<)\s*{_num}')
        for i, opt in enumerate(options):
            m = pat.search(opt)
            if m:
                try:
                    olo, ohi = float(m.group(1)), float(m.group(2))
                    if abs(olo - lo) < 1e-6 and abs(ohi - hi) < 1e-6:
                        return i
                except ValueError:
                    pass
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

            # ── Special: fraction result ─────────────────────────────────
            if expr.startswith('__fraction__:'):
                spec = expr[len('__fraction__:'):]
                try:
                    n, d = spec.split('/')
                    idx = self._match_fraction(int(n), int(d), inp.options)
                except (ValueError, AttributeError):
                    idx = None
                if idx is not None:
                    return StrategyOutput(
                        chosen_index=idx,
                        confidence=0.97,
                        rationale=f"SympyDirectTool[fraction]: {spec}",
                        extras={"tool": "sympy_direct", "expr": expr, "result": spec},
                    )
                continue

            # ── Special: tangent line ────────────────────────────────────
            if expr.startswith('__tangent__:'):
                mb = _eval_tangent(expr[len('__tangent__:'):])
                if mb is not None:
                    slope, intercept = mb
                    idx = self._match_linear(slope, intercept, inp.options)
                    if idx is not None:
                        return StrategyOutput(
                            chosen_index=idx,
                            confidence=0.97,
                            rationale=f"SympyDirectTool[tangent]: y={slope}x+{intercept}",
                            extras={"tool": "sympy_direct", "expr": expr,
                                    "result": f"slope={slope},intercept={intercept}"},
                        )
                continue

            # ── Special: range result ────────────────────────────────────
            if expr.startswith('__range__:'):
                parts = expr[len('__range__:'):].split(',')
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                    idx = self._match_range(lo, hi, inp.options)
                except (ValueError, IndexError):
                    idx = None
                if idx is not None:
                    return StrategyOutput(
                        chosen_index=idx,
                        confidence=0.97,
                        rationale=f"SympyDirectTool[range]: [{lo}, {hi}]",
                        extras={"tool": "sympy_direct", "expr": expr,
                                "result": f"[{lo},{hi}]"},
                    )
                continue

            # ── Standard: real numeric match ─────────────────────────────
            result = _try_evaluate(expr)
            if result is not None and math.isfinite(result):
                idx = _match_options(result, inp.options)
                if idx is None:
                    # Options may be LaTeX fractions — try fraction matching
                    from math import gcd as _gcd
                    # Approximate result as a simple fraction and try
                    from fractions import Fraction
                    try:
                        frac = Fraction(result).limit_denominator(1000)
                        idx = self._match_fraction(frac.numerator,
                                                   frac.denominator,
                                                   inp.options)
                    except Exception:
                        pass
                if idx is not None:
                    return StrategyOutput(
                        chosen_index=idx,
                        confidence=0.97,
                        rationale=f"SympyDirectTool: {expr} = {result}",
                        extras={"tool": "sympy_direct", "expr": expr, "result": result},
                    )

            # ── Fallback: complex match ───────────────────────────────────
            if not expr.startswith('__'):
                idx = self._match_complex(expr, inp.options)
                if idx is not None:
                    return StrategyOutput(
                        chosen_index=idx,
                        confidence=0.97,
                        rationale=f"SympyDirectTool[complex]: {expr}",
                        extras={"tool": "sympy_direct", "expr": expr, "result": "complex"},
                    )

        return None  # no pattern matched or no option aligned → abstain
