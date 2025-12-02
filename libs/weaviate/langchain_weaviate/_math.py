"""Math utils."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]

# ---- Backend selection: try simsimd -> scipy.cdist -> numpy fallback ----
_simsimd_available = False
try:
    import simsimd  # type: ignore

    _simsimd_available = True
    _cdist_impl = simsimd.cdist  # type: ignore
except Exception:
    # Could be ImportError or OSError (binary incompatibility)
    _simsimd_available = False
    try:
        from scipy.spatial.distance import cdist as _scipy_cdist  # type: ignore

        _cdist_impl = _scipy_cdist
    except Exception:
        # Minimal NumPy fallback supporting cosine distance only
        def _numpy_cdist(X: np.ndarray, Y: np.ndarray, metric: str = "cosine"):
            if metric != "cosine":
                raise ValueError("NumPy fallback only supports metric='cosine'")

            X = np.asarray(X, dtype=float)
            Y = np.asarray(Y, dtype=float)

            if X.ndim != 2 or Y.ndim != 2:
                raise ValueError("X and Y must be 2-D arrays")

            # normalize rows (avoid division by zero)
            def _normalize(A: np.ndarray):
                norms = np.linalg.norm(A, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return A / norms

            Xn = _normalize(X)
            Yn = _normalize(Y)

            sim = Xn @ Yn.T
            sim = np.clip(sim, -1.0, 1.0)
            # cosine distance = 1 - similarity
            return 1.0 - sim

        _cdist_impl = _numpy_cdist

if not _simsimd_available:
    logger.warning(
        "simsimd not available — falling back to SciPy/NumPy implementation for vector math. "
        "To enable the accelerated path, install the optional dependency 'simsimd' "
        "(note: simsimd may require newer glibc on some systems)."
    )

#  Exposed distance function 
def cdist(X: np.ndarray, Y: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compatibility wrapper. Returns pairwise distances between rows of X and Y.
    Uses simsimd.cdist if available, else SciPy, else NumPy fallback.
    """
    return np.asarray(_cdist_impl(X, Y, metric=metric))    


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""

    # handle empty inputs (preserve previous behavior)
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # _cdist_impl returns distances (cosine distance), so 1 - distance = similarity
    Z = 1 - np.array(_cdist_impl(X, Y, metric="cosine"))

    # maintain previous behavior: if a scalar (float) is returned, convert to 1-D array
    if isinstance(Z, float):
        return np.array([Z])
    return Z


def cosine_similarity_top_k(
    X: Matrix,
    Y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Row-wise cosine similarity with optional top-k and score threshold filtering.

    Args:
        X: Matrix.
        Y: Matrix, same width as X.
        top_k: Max number of results to return.
        score_threshold: Minimum cosine similarity of results.

    Returns:
        Tuple of two lists. First contains two-tuples of indices `(X_idx, Y_idx)`,
            second contains corresponding cosine similarities.
    """
    if len(X) == 0 or len(Y) == 0:
        return [], []

    score_array = cosine_similarity(X, Y)
    score_threshold = score_threshold or -1.0
    score_array[score_array < score_threshold] = 0
    top_k = min(top_k or len(score_array), np.count_nonzero(score_array))
    if top_k == 0:
        return [], []
    top_k_idxs = np.argpartition(score_array, -top_k, axis=None)[-top_k:]
    top_k_idxs = top_k_idxs[np.argsort(score_array.ravel()[top_k_idxs])][::-1]
    ret_idxs = np.unravel_index(top_k_idxs, score_array.shape)
    scores = score_array.ravel()[top_k_idxs].tolist()
    return list(zip(*ret_idxs)), scores  # type: ignore
