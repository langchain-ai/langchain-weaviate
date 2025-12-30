# tests/unit_tests/test_math_simsimd_fallback.py
import importlib
import sys

import numpy as np
import pytest

MODULE_PATH = "langchain_weaviate._math"


def _reload_math_with_numpy_fallback(monkeypatch):
    """
    Reload the math module simulating that simsimd and scipy are NOT available,
    so the pure-NumPy fallback is used.
    """
    # Ensure simsimd and scipy are not importable
    if "simsimd" in sys.modules:
        monkeypatch.delitem(sys.modules, "simsimd", raising=False)
    if "scipy" in sys.modules:
        monkeypatch.delitem(sys.modules, "scipy", raising=False)
    if "scipy.spatial" in sys.modules:
        monkeypatch.delitem(sys.modules, "scipy.spatial", raising=False)

    orig_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Force ImportError for simsimd and any scipy import
        # so we hit the numpy fallback
        if name == "simsimd" or name.startswith("scipy"):
            raise ImportError(f"{name} not available in this test environment")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    # Unload the math module if previously imported so it re-executes top-level code.
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


def test_cdist_numpy_fallback(monkeypatch):
    math_mod = _reload_math_with_numpy_fallback(monkeypatch)

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])

    # cdist returns cosine distance; identical unit vectors -> distance 0 on diagonal
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = math_mod.cdist(X, Y, metric="cosine")
    assert np.allclose(Z, expected)


def test_cdist_unsupported_metric_raises(monkeypatch):
    math_mod = _reload_math_with_numpy_fallback(monkeypatch)

    X = np.array([[1.0, 0.0]])
    Y = np.array([[0.0, 1.0]])

    with pytest.raises(ValueError):
        # NumPy fallback only supports metric='cosine' by design
        math_mod.cdist(X, Y, metric="euclidean")
