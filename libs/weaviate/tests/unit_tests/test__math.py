import numpy as np
import pytest

from langchain_weaviate._math import cosine_similarity, cosine_similarity_top_k


def test_cosine_similarity() -> None:
    X = np.array([[1, 0], [0, 1]])
    Y = np.array([[0, 1], [1, 0]])

    result = cosine_similarity(X, Y)
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(result, expected, rtol=1e-2)

    # Test handling of empty matrices
    assert cosine_similarity([], []).size == 0

    # Test handling of shape validation
    with pytest.raises(ValueError):
        cosine_similarity(np.array([[1, 0]]), np.array([[1, 0, 0]]))

    # Test single value case
    X_single = np.array([[1, 0]])
    Y_single = np.array([[0, 1]])
    result_single = cosine_similarity(X_single, Y_single)
    assert np.allclose(result_single, np.array([0.0]), rtol=1e-2)


def test_cosine_similarity_top_k() -> None:
    X = np.array([[1, 0], [0, 1]])
    Y = np.array([[0, 1], [1, 0]])

    expected_result_indices = [(0, 1), (1, 0)]
    expected_result_scores = [1.0, 1.0]

    result_indices, result_scores = cosine_similarity_top_k(X, Y)

    assert result_indices == expected_result_indices
    assert np.allclose(result_scores, expected_result_scores, rtol=1e-2)

    # Test with score threshold
    result_indices, result_scores = cosine_similarity_top_k(X, Y, score_threshold=0.5)
    assert result_indices == expected_result_indices
    assert np.allclose(result_scores, expected_result_scores, rtol=1e-2)

    # Test with empty matrices
    empty_indices, empty_scores = cosine_similarity_top_k([], [])
    assert empty_indices == []
    assert empty_scores == []

    # Test with top_k limit
    limited_indices, limited_scores = cosine_similarity_top_k(X, Y, top_k=1)
    assert len(limited_indices) == 1
    assert len(limited_scores) == 1
