"""StatsTool: deterministic statistics solver using scipy.stats.

Covers question types that neither MathsTool nor SympyDirectTool can handle:
  - Binomial P(X >= k)    "probability that at least 3 of 5 cyclones become hurricanes"
  - Binomial P(X = k)     "probability that exactly 2 of 4 tests are positive"
  - Normal percentile     "28th percentile of N(9.8, 2.1) — how far from mean?"
  - Normal P(X < x)       "P(mean < 3.9) for sample of 40 from N(4, 0.25)"
  - Normal inverse        "80% weigh more than 10000 — find mean/std"

Design: same precision-over-recall contract as MathsTool and SympyDirectTool.
  - Lazy import: scipy loaded only on first call (~50ms cold, <1ms warm).
  - Abstains immediately on any parse failure or ambiguous phrasing.
  - Only returns when the computed value matches an option within tolerance.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..config import Category
from ..strategies.base import StrategyInput, StrategyOutput
from .base import Tool
from .maths_tool import _match_options, _parse_option_value


# ── Lazy scipy import ─────────────────────────────────────────────────────────

_scipy_stats = None


def _stats():
    global _scipy_stats
    if _scipy_stats is None:
        import scipy.stats as _s
        _scipy_stats = _s
    return _scipy_stats


# ── Word-to-number conversion ─────────────────────────────────────────────────

_W2N = {
    'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
    'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
    'fourteen':14,'fifteen':15,'twenty':20,'thirty':30,'forty':40,
    'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,
}

def _to_float(s: str) -> Optional[float]:
    """Convert a string that may be a word-number or digit-number to float."""
    s = s.strip().lower()
    if s in _W2N:
        return float(_W2N[s])
    try:
        return float(s)
    except ValueError:
        return None


def _words_to_digits_local(text: str) -> str:
    """Replace word-numbers with digits in a text string."""
    def repl(m):
        w = m.group(0).lower()
        return str(_W2N[w]) if w in _W2N else m.group(0)
    return re.sub(r'\b(' + '|'.join(_W2N.keys()) + r')\b', repl, text, flags=re.I)


# ── Parameter extraction helpers ─────────────────────────────────────────────

def _extract_probability(text: str) -> Optional[float]:
    """Extract a probability expressed as decimal, fraction, or ratio in context.
    Handles: '5.1 became hurricanes' from '8.7 cyclones', '.323', '0.323', '70%', '2/7'.
    """
    # Explicit "X became/are/were Y" rate pattern — handles "5.1 became hurricanes"
    # out of "8.7 cyclones" by finding the smaller and larger numbers with a verb
    rate_m = re.search(
        r'(\d+(?:\.\d+)?)\s+(?:became|are|were|out\s+of)\s+\S*\s*(?:\S+\s+)?'
        r'(\d+(?:\.\d+)?)'
        r'|(\d+(?:\.\d+)?)\s+of\s+(?:the\s+)?(\d+(?:\.\d+)?)',
        text, re.I,
    )
    if rate_m:
        pairs = [
            (rate_m.group(1), rate_m.group(2)),
            (rate_m.group(3), rate_m.group(4)),
        ]
        for a_s, b_s in pairs:
            if a_s and b_s:
                a, b = float(a_s), float(b_s)
                if 0 < a < b:
                    return a / b
                if 0 < b < a:
                    return b / a

    # Fallback: adjacent small numbers that form a plausible probability ratio
    nums_found = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    seen = set()
    candidates = []
    for n in nums_found:
        f = float(n)
        if f not in seen and 0.01 <= f <= 100:
            seen.add(f)
            candidates.append(f)
    for i in range(len(candidates) - 1):
        a, b = candidates[i], candidates[i + 1]
        # Check both orderings (sentence may mention larger denominator first)
        if 0 < a < b and a / b < 1:
            return a / b
        if 0 < b < a and b / a < 1:
            return b / a

    # Explicit fraction: 2/7
    frac = re.search(r'\b(\d+)\s*/\s*(\d+)\b', text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if 0 < num < den:
            return num / den

    # Plain decimal — allow leading dot: ".323" or "0.323"
    plain = re.search(r'(?<!\d)\.(\d+)\b|(?<!\d)\b0\.(\d+)\b', text)
    if plain:
        digits = plain.group(1) or plain.group(2)
        val = float('0.' + digits)
        if 0 < val < 1:
            return val

    # Percentage
    pct = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if pct:
        return float(pct.group(1)) / 100

    return None


def _extract_n_trials(text: str) -> Optional[int]:
    """Extract the number of trials from phrasing like 'five cyclones', '3 at-bats'.
    Normalises word-numbers first, then searches for digit + trial noun.
    """
    normed = _words_to_digits_local(text)
    trial_words = (
        r'cyclones?|tosses?|trials?|rolls?|children|patients?|flips?'
        r'|at[\s\-]?bats?|games?|attempts?|observations?|people|students?'
    )
    # Prefer "if there are N <noun>" phrasing
    m = re.search(rf'(?:there\s+are\s+|are\s+)?(\d+)\s+(?:{trial_words})', normed, re.I)
    if m:
        return int(m.group(1))
    # "N hits in N at-bats"
    m2 = re.search(r'(\d+)\s+(?:hits?|successes?)\s+in\s+(\d+)', normed, re.I)
    if m2:
        return int(m2.group(2))
    return None


def _extract_k_successes_repeated(text: str) -> Optional[int]:
    """Extract k from 'three hits in three at-bats' style (repeated number)."""
    normed = _words_to_digits_local(text)
    m = re.search(r'(\d+)\s+(?:hits?|successes?)\s+in\s+\1', normed, re.I)
    if m:
        return int(m.group(1))
    return None


def _parse_option_approx(text: str) -> Optional[float]:
    """Parse an option that may contain units or prose: '1.22 ounces below the mean'."""
    m = re.search(r'(-?\d+(?:\.\d+)?)', text)
    return float(m.group(1)) if m else None


def _match_options_approx(value: float, options: tuple,
                           tol: float = 0.01) -> Optional[int]:
    """Match value against options that may contain prose/units. Uses absolute tol."""
    for i, opt in enumerate(options):
        v = _parse_option_approx(opt)
        if v is not None and abs(v - value) <= tol:
            return i
    return None


def _match_prob(result: float, options: tuple) -> Optional[int]:
    """Match a probability result against options.
    Uses absolute tolerance of 0.005 (half a percentage point) for small values,
    which handles rounding in option display (e.g. 0.0057 → option '0.0057').
    Falls back to standard _match_options for larger values.
    """
    # Round to 4 significant figures and try
    rounded = round(result, 4)
    idx = _match_options(rounded, options)
    if idx is not None:
        return idx
    # Absolute tolerance for probability options
    return _match_options_approx(result, options, tol=0.005)


# ── Pattern functions ─────────────────────────────────────────────────────────

def _try_binomial_at_least(question: str, options: tuple) -> Optional[int]:
    """P(X >= k) for binomial(n, p).
    Matches: 'at least K' in N trials with probability P.
    """
    normed = _words_to_digits_local(question)

    k_match = re.search(r'at\s+least\s+(\d+)', normed, re.I)
    if not k_match:
        return None
    k = int(k_match.group(1))

    n = _extract_n_trials(normed)
    if n is None:
        return None

    p = _extract_probability(question)
    if p is None or not (0 < p < 1):
        return None

    result = _stats().binom.sf(k - 1, n, p)
    return _match_prob(result, options)


def _try_binomial_repeated_success(question: str, options: tuple) -> Optional[int]:
    """P(X = k in k trials) — 'k hits in k at-bats', 'three hits in three at-bats'.
    This is simply p^k when each trial is independent.
    """
    k = _extract_k_successes_repeated(question)
    if k is None:
        return None

    p = _extract_probability(question)
    if p is None or not (0 < p < 1):
        return None

    result = p ** k
    return _match_prob(result, options)


def _try_binomial_exactly(question: str, options: tuple) -> Optional[int]:
    """P(X = k) for binomial(n, p).
    Matches: 'exactly K' in N trials with probability P.
    """
    normed = _words_to_digits_local(question)

    k_match = re.search(r'exactly\s+(\d+)', normed, re.I)
    if not k_match:
        return None
    k = int(k_match.group(1))

    n = _extract_n_trials(normed)
    if n is None:
        return None

    p = _extract_probability(question)
    if p is None or not (0 < p < 1):
        return None

    result = _stats().binom.pmf(k, n, p)
    return _match_prob(result, options)


def _try_normal_percentile(question: str, options: tuple) -> Optional[int]:
    """Nth percentile of N(mu, sigma) — how far from the mean?
    Computes norm.ppf(p) * sigma and matches against options (which may have units).
    """
    normed = _words_to_digits_local(question)

    pct_match = re.search(r'(\d+)(?:st|nd|rd|th)\s+percentile', normed, re.I)
    if not pct_match:
        return None
    percentile = int(pct_match.group(1)) / 100.0

    mu_match  = re.search(r'mean\s+(?:of\s+)?(\d+(?:\.\d+)?)', normed, re.I)
    std_match = re.search(r'standard\s+deviation\s+(?:of\s+)?(\d+(?:\.\d+)?)', normed, re.I)
    if not mu_match or not std_match:
        return None

    mu  = float(mu_match.group(1))
    std = float(std_match.group(1))

    z        = _stats().norm.ppf(percentile)
    value    = mu + z * std
    distance = round(abs(z * std), 2)

    # Try exact numeric match on raw value or distance
    idx = _match_options(value, options)
    if idx is not None:
        return idx
    idx = _match_options(distance, options)
    if idx is not None:
        return idx
    # Try prose options like "1.22 ounces below the mean"
    return _match_options_approx(distance, options)


def _try_normal_sample_mean(question: str, options: tuple) -> Optional[int]:
    """P(sample mean < x) for N(mu, sigma/sqrt(n)).
    Matches: 'average N ... std S ... sample of N ... less than X'.
    Handles word-numbers like 'four ounces', 'Forty jars'.
    """
    normed = _words_to_digits_local(question)

    mu_match  = re.search(r'(?:average|mean)\s+(?:of\s+)?(\d+(?:\.\d+)?)', normed, re.I)
    std_match = re.search(r'standard\s+deviation\s+(?:of\s+)?(\d+(?:\.\d+)?)', normed, re.I)
    n_match   = re.search(
        r'(\d+)\s+(?:jars?|samples?|observations?|subjects?|tablets?|items?)'
        r'|(?:sample\s+of\s+|selected\s+at\s+random\s*)(\d+)',
        normed, re.I,
    )
    x_match   = re.search(r'less\s+than\s+(\d+(?:\.\d+)?)', normed, re.I)

    if not all([mu_match, std_match, n_match, x_match]):
        return None

    mu  = float(mu_match.group(1))
    std = float(std_match.group(1))
    n   = int(n_match.group(1) or n_match.group(2))
    x   = float(x_match.group(1))

    se     = std / math.sqrt(n)
    result = _stats().norm.cdf(x, loc=mu, scale=se)
    return _match_prob(result, options)


# ── Tool ─────────────────────────────────────────────────────────────────────

class StatsTool(Tool):
    """Deterministic statistics solver using scipy.stats.

    Sits after SympyDirectTool in the tool chain:
      MathsTool → SympyDirectTool → StatsTool → LLM fallback

    Covers binomial and normal distribution computations that appear
    frequently in the MATHS category but are outside SymPy's scope.
    """
    name = "stats_tool"

    # Ordered list of (pattern_fn, description) — tried in order, first match wins
    _PATTERNS = [
        (_try_binomial_at_least,         "binomial P(X>=k)"),
        (_try_binomial_repeated_success, "binomial p^k"),
        (_try_binomial_exactly,          "binomial P(X=k)"),
        (_try_normal_percentile,         "normal percentile"),
        (_try_normal_sample_mean,        "normal sample mean"),
    ]

    def can_handle(self, inp: StrategyInput) -> bool:
        return inp.category == Category.MATHS

    def use(self, inp: StrategyInput) -> Optional[StrategyOutput]:
        question = inp.question

        for pattern_fn, label in self._PATTERNS:
            try:
                idx = pattern_fn(question, inp.options)
            except Exception:
                continue
            if idx is not None:
                return StrategyOutput(
                    chosen_index=idx,
                    confidence=0.97,
                    rationale=f"StatsTool[{label}]",
                    extras={"tool": "stats_tool", "pattern": label},
                )

        return None
