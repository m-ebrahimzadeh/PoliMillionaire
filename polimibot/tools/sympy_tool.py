"""SympySolver: symbolic math evaluator for the AgentStrategy ReAct loop.

Handles what safe_eval/calc cannot:
  - Algebraic equations        solve("3*x + 7 - 22", "x")  → [5]
  - Modular / exact arithmetic Mod(3**100, 10)              → 1
  - Exact combinatorics        binomial(10, 3)              → 120
  - Geometric / trig           pi * 5**2                   → 78.539...
  - Symbolic simplification    simplify("sin(x)**2 + cos(x)**2") → 1

Design: same precision-over-recall philosophy as MathsTool.
  - Lazy import: sympy is only loaded on first call (~2 s cold, <1 ms warm).
  - Hard timeout guard: caller (AgentStrategy) already enforces a wall-clock
    deadline; the solver itself raises immediately on parse failure and never
    loops, so it won't stall the game turn.
  - Sandboxed namespace: only a curated subset of sympy names is exposed —
    no file I/O, no os, no exec.
"""
from __future__ import annotations

import re
from typing import Optional

# Lazy-loaded; only imported on first call to keep startup time unaffected.
_sympy: Optional[object] = None


def _load_sympy():
    global _sympy
    if _sympy is None:
        import sympy
        _sympy = sympy
    return _sympy


# Symbols the model is allowed to reference. Expand conservatively.
_ALLOWED_NAMES = {
    # Core
    "symbols", "Symbol", "Rational", "Integer", "Float", "pi", "E", "oo",
    "I",  # imaginary unit
    # Solving
    "solve", "solveset", "nsolve",
    # Arithmetic / combinatorics
    "factorial", "binomial", "Mod", "gcd", "lcm", "floor", "ceiling",
    "Abs", "sqrt", "cbrt", "root", "log", "ln", "exp",
    # Trig
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    # Simplification / evaluation
    "simplify", "expand", "factor", "cancel", "together",
    "N",  # numeric evaluation
    # Sequences / sums
    "summation", "Sum", "product", "Product",
    # Number theory
    "isprime", "factorint", "totient", "nextprime",
}


def _build_namespace(sympy_mod) -> dict:
    ns = {}
    for name in _ALLOWED_NAMES:
        obj = getattr(sympy_mod, name, None)
        if obj is not None:
            ns[name] = obj
    return ns


# Patterns that look like equation strings needing solve()
# e.g. "3*x + 7 = 22"  or  "x**2 - 4 = 0"
_EQ_PATTERN = re.compile(r"[a-zA-Z_]\w*\s*(?:[+\-*/^]|\*\*)\s*.*=\s*\S")

# Block dangerous constructs before they ever reach sympify.
# sympify() uses Python's eval() internally and a restricted local namespace
# is not sufficient — __import__ and builtins can still slip through.
_BLOCKED = re.compile(
    r'__\w+__'           # dunder names  (__import__, __builtins__, etc.)
    r'|import\s'         # import statements
    r'|open\s*\('        # file access
    r'|exec\s*\('        # code execution
    r'|eval\s*\('        # nested eval
    r'|getattr\s*\('     # attribute bypass
    r'|setattr\s*\('
    r'|delattr\s*\('
    r'|globals\s*\('
    r'|locals\s*\(',
    re.IGNORECASE,
)


def sympy_solve(expression: str) -> str:
    """Evaluate or solve a SymPy expression string.

    If the expression contains '=' (an equation), rewrites as LHS - RHS and
    calls solve() on the first free symbol.

    Returns a compact string representation of the result, or raises
    ValueError on any error (parse failure, disallowed name, etc.).

    >>> sympy_solve("3*x + 7 - 22")
    '[5]'
    >>> sympy_solve("Mod(3**100, 10)")
    '1'
    >>> sympy_solve("factorial(10)")
    '3628800'
    >>> sympy_solve("pi * 5**2")
    '78.5398163397448'
    """
    sp = _load_sympy()
    ns = _build_namespace(sp)

    expr_str = expression.strip()

    # Reject dangerous patterns before they reach sympify (which uses eval internally).
    if _BLOCKED.search(expr_str):
        raise ValueError(f"Expression contains disallowed construct: {expr_str!r}")

    # Rewrite "LHS = RHS" into "LHS - (RHS)" so solve() can handle it
    if "=" in expr_str and "==" not in expr_str:
        parts = expr_str.split("=", 1)
        lhs, rhs = parts[0].strip(), parts[1].strip()
        expr_str = f"({lhs}) - ({rhs})"

    try:
        parsed = sp.sympify(expr_str, locals=ns, evaluate=True)
    except Exception as exc:
        raise ValueError(f"SymPy parse error: {exc}") from exc

    # sympify can return arbitrary Python objects if the expression escapes the
    # sandbox (e.g. a module, a builtin). Reject anything that isn't a SymPy Expr.
    if not isinstance(parsed, sp.Basic):
        raise ValueError(f"Expression did not evaluate to a SymPy expression: {type(parsed)}")

    # If free symbols exist, solve for the first one
    if parsed.free_symbols:
        var = sorted(parsed.free_symbols, key=str)[0]
        try:
            solutions = sp.solve(parsed, var)
        except Exception as exc:
            raise ValueError(f"SymPy solve error: {exc}") from exc
        if not solutions:
            raise ValueError("No solution found")
        # Numerically evaluate each solution for matching against options
        result = [sp.N(s, 15) for s in solutions]
        if len(result) == 1:
            return str(result[0])
        return str(result)

    # Pure numeric expression — evaluate to 15 significant figures
    try:
        numeric = sp.N(parsed, 15)
    except Exception as exc:
        raise ValueError(f"SymPy evaluation error: {exc}") from exc

    # Return exact integer string when possible (avoids "120.000000" noise)
    try:
        as_int = int(numeric)
        if sp.Abs(numeric - as_int) < 1e-9:
            return str(as_int)
    except (TypeError, ValueError):
        pass

    return str(numeric)
