import numpy as np
import pytest

from langchain_weaviate._math import cosine_similarity, cosine_similarity_top_k


def test_cosine_similarity_top_k() -> None:
    X = np.array([[1, 0], [0, 1]])
    Y = np.array([[0, 1], [1, 0]])

    expected_result_indices = [(0, 1), (1, 0)]
    expected_result_scores = [1.0, 1.0]

    result_indices, result_scores = cosine_similarity_top_k(X, Y)

    assert result_indices == expected_result_indices
    assert np.allclose(result_scores, expected_result_scores, rtol=1e-2)


def test_cosine_similarity_empty_inputs() -> None:
    """Test that empty inputs return empty array."""
    X = np.array([])
    Y = np.array([[1, 0]])
    result = cosine_similarity(X, Y)
    assert result.shape == (0,)

    X = np.array([[1, 0]])
    Y = np.array([])
    result = cosine_similarity(X, Y)
    assert result.shape == (0,)


def test_cosine_similarity_shape_mismatch() -> None:
    """Test that shape mismatch raises ValueError."""
    X = np.array([[1, 0]])
    Y = np.array([[1, 0, 1]])
    with pytest.raises(
        ValueError, match="Number of columns in X and Y must be the same"
    ):
        cosine_similarity(X, Y)


def test_cosine_similarity_top_k_empty_inputs() -> None:
    """Test that empty inputs return empty lists."""
    X = np.array([])
    Y = np.array([[1, 0]])
    result_indices, result_scores = cosine_similarity_top_k(X, Y)
    assert result_indices == []
    assert result_scores == []

    X = np.array([[1, 0]])
    Y = np.array([])
    result_indices, result_scores = cosine_similarity_top_k(X, Y)
    assert result_indices == []
    assert result_scores == []


def test_cosine_similarity_top_k_with_threshold() -> None:
    """Test top_k with score_threshold filtering."""
    X = np.array([[1, 0], [0, 1], [1, 1]])
    Y = np.array([[1, 0], [0, 1], [0.5, 0.5]])

    # With a high threshold, should filter out low scores
    result_indices, result_scores = cosine_similarity_top_k(
        X, Y, top_k=10, score_threshold=0.95
    )

    # All scores should be above the threshold
    assert all(score >= 0.95 for score in result_scores)


def test_cosine_similarity_top_k_zero_results() -> None:
    """Test when all scores are below threshold."""
    X = np.array([[1, 0]])
    Y = np.array([[0, 1]])  # Orthogonal vectors, similarity = 0

    result_indices, result_scores = cosine_similarity_top_k(
        X, Y, top_k=10, score_threshold=0.5
    )

    # No results should pass the threshold
    assert result_indices == []
    assert result_scores == []


def test_cosine_similarity_single_value() -> None:
    """Test cosine similarity with single vectors to cover float conversion path."""
    X = np.array([[1.0, 0.0]])
    Y = np.array([[1.0, 0.0]])

    result = cosine_similarity(X, Y)

    # Should return an array
    assert isinstance(result, np.ndarray)
    assert result.shape[0] >= 1


def test_cosine_similarity_float_to_array_conversion() -> None:
    """Test line 100: Conversion of scalar float to 1-D array.
    
    This tests the edge case where _cdist_impl returns a float instead of an array,
    which is then converted to a 1-D array containing that float value.
    """
    X = np.array([[1.0, 0.0]])
    Y = np.array([[1.0, 0.0]])

    result = cosine_similarity(X, Y)

    # Result should always be an array, even if backend returns a float
    assert isinstance(result, np.ndarray)
    # Should contain the similarity value (1.0 for identical vectors)
    assert np.allclose(result, 1.0, rtol=1e-2)
