"""Safe arithmetic evaluator using Python's AST — no exec(), no subprocess.

Why AST instead of eval()? eval("__import__('os').system('rm -rf /')") works.
AST parsing lets us whitelist exactly which node types are allowed.
The model's output goes through here, so safety matters.
"""
from __future__ import annotations

import ast
import math
import operator as op
from typing import Any, Dict, Union

# Whitelisted binary operators only. No bitwise, no shifts.
_BINARY_OPS: Dict[type, Any] = {
    ast.Add: op.add,   ast.Sub: op.sub,
    ast.Mult: op.mul,  ast.Div: op.truediv,
    ast.Mod: op.mod,   ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv,
}

_UNARY_OPS: Dict[type, Any] = {
    ast.USub: op.neg, ast.UAdd: op.pos,
}

# Whitelisted names and functions. Expand carefully.
_ALLOWED: Dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "factorial": math.factorial,
    "log": math.log, "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "gcd": math.gcd, "comb": math.comb, "perm": math.perm,
    "pi": math.pi, "e": math.e,
}


def safe_eval(expression: str) -> Union[int, float]:
    """Evaluate a numeric Python expression in a restricted AST sandbox.
    
    Raises ValueError on disallowed constructs, TypeError/ZeroDivisionError
    on runtime math errors. Caller should catch Exception broadly.
    
    >>> safe_eval("15 / 100 * 200")
    30.0
    >>> safe_eval("factorial(5)")
    120
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Cannot parse: {expression!r}") from e
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        fn = _BINARY_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Operator {type(node.op).__name__} not allowed")
        return fn(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unary {type(node.op).__name__} not allowed")
        return fn(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function names allowed")
        fn = _ALLOWED.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function '{node.func.id}' not in whitelist")
        return fn(*[_eval_node(a) for a in node.args])

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED:
            return _ALLOWED[node.id]
        raise ValueError(f"Name '{node.id}' not in whitelist")

    raise ValueError(f"AST node {type(node).__name__} not allowed")