# tests/unit_tests/test_math_simsimd_fallback.py
from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

MODULE_PATH = "langchain_weaviate._math"


def _reload_math_with_scipy_only(monkeypatch: Any) -> Any:
    """
    Reload the math module simulating that simsimd is NOT available but scipy IS,
    so the SciPy cdist is used (line 27).
    """
    # Ensure simsimd is not importable, but scipy is
    if "simsimd" in sys.modules:
        monkeypatch.delitem(sys.modules, "simsimd", raising=False)

    orig_import = __import__

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[Any, ...] = (),
        level: int = 0,
    ) -> Any:
        # Force ImportError for simsimd only, allow scipy
        if name == "simsimd":
            raise ImportError(f"{name} not available in this test environment")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    # Unload the math module if previously imported so it re-executes top-level code.
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


def _reload_math_with_numpy_fallback(monkeypatch: Any) -> Any:
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

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[Any, ...] = (),
        level: int = 0,
    ) -> Any:
        # Force ImportError for simsimd and any scipy import so we hit the
        # numpy fallback
        if name == "simsimd" or name.startswith("scipy"):
            raise ImportError(f"{name} not available in this test environment")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    # Unload the math module if previously imported so it re-executes top-level code.
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


def test_cdist_numpy_fallback(monkeypatch: Any) -> None:
    math_mod = _reload_math_with_numpy_fallback(monkeypatch)

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])

    # cdist returns cosine distance; identical unit vectors -> distance 0 on diagonal
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = math_mod.cdist(X, Y, metric="cosine")
    assert np.allclose(Z, expected)


def test_cdist_unsupported_metric_raises(monkeypatch: Any) -> None:
    math_mod = _reload_math_with_numpy_fallback(monkeypatch)

    X = np.array([[1.0, 0.0]])
    Y = np.array([[0.0, 1.0]])

    with pytest.raises(ValueError):
        # NumPy fallback only supports metric='cosine' by design
        math_mod.cdist(X, Y, metric="euclidean")


def test_numpy_fallback_invalid_dimensions(monkeypatch: Any) -> None:
    """Test line 40: ValueError when X and Y are not 2-D arrays."""
    math_mod = _reload_math_with_numpy_fallback(monkeypatch)

    # Test with 1-D array
    X = np.array([1.0, 0.0])  # 1-D instead of 2-D
    Y = np.array([[1.0, 0.0]])

    with pytest.raises(ValueError, match="X and Y must be 2-D arrays"):
        math_mod.cdist(X, Y, metric="cosine")

    # Test with 3-D array
    X = np.array([[[1.0, 0.0]]])  # 3-D instead of 2-D
    Y = np.array([[1.0, 0.0]])

    with pytest.raises(ValueError, match="X and Y must be 2-D arrays"):
        math_mod.cdist(X, Y, metric="cosine")


def test_scipy_fallback_path(monkeypatch: Any) -> None:
    """Test line 27: scipy.spatial.distance.cdist is used when simsimd not available.
    
    This tests that the scipy fallback (line 27: _cdist_impl = _scipy_cdist)
    works correctly when simsimd is not available but scipy is.
    """
    # Check if scipy is available
    try:
        import scipy.spatial.distance  # noqa: F401
    except ImportError:
        pytest.skip("scipy not available, cannot test scipy fallback path")

    math_mod = _reload_math_with_scipy_only(monkeypatch)

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])

    # cdist returns cosine distance; identical unit vectors -> distance 0 on diagonal
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = math_mod.cdist(X, Y, metric="cosine")
    assert np.allclose(Z, expected)

    # Verify that scipy is being used by checking _simsimd_available is False
    assert not math_mod._simsimd_available


def test_cosine_similarity_float_return_path(monkeypatch: Any) -> None:
    """Test line 100: cosine_similarity handles float return from _cdist_impl.
    
    This test mocks _cdist_impl to return a float instead of an array,
    ensuring the isinstance(Z, float) check and array conversion works.
    """
    from langchain_weaviate import _math

    # Save original implementation
    original_cdist = _math._cdist_impl

    def mock_cdist_returns_float(
        X: np.ndarray, Y: np.ndarray, metric: str = "cosine"
    ) -> float:
        # Return a float instead of an array for 1x1 case
        if X.shape[0] == 1 and Y.shape[0] == 1:
            return 0.0  # cosine distance of 0 means identical vectors
        return original_cdist(X, Y, metric=metric)

    monkeypatch.setattr(_math, "_cdist_impl", mock_cdist_returns_float)

    X = np.array([[1.0, 0.0]])
    Y = np.array([[1.0, 0.0]])

    result = _math.cosine_similarity(X, Y)

    # Result should be an array, not a float
    assert isinstance(result, np.ndarray)
    # cosine distance 0 -> similarity 1
    assert result.shape == (1,)
    assert np.allclose(result, [1.0])

