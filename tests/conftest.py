"""Test-suite scaffolding.

This dev environment doesn't have torch installed (GPU work happens on
Colab). Most tests don't actually need torch — they go through MockLLM
or pure-Python paths — but importing ``polimibot`` pulls
``polimibot.models.llm`` which top-imports ``torch``. Stub it before
the package imports so the test suite collects cleanly without GPU
dependencies.

If torch IS installed (CI, Colab, full dev box), this no-ops and the
real package is used.
"""
from __future__ import annotations

import sys
import types


def _install_torch_stub() -> None:
    try:
        import torch  # noqa: F401
        return  # real torch present — leave it alone
    except ImportError:
        pass

    stub = types.ModuleType("torch")
    stub.bfloat16 = "bfloat16"

    class _InferenceMode:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    stub.inference_mode = lambda: _InferenceMode()
    stub.softmax = lambda x, dim: x
    stub.tensor = lambda x: x

    class _Cuda:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def empty_cache(): pass
        @staticmethod
        def synchronize(): pass

    stub.cuda = _Cuda()
    sys.modules["torch"] = stub


_install_torch_stub()
