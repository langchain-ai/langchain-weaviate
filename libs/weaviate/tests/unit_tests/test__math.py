import numpy as np

from langchain_weaviate._math import cosine_similarity_top_k


def test_cosine_similarity_top_k() -> None:
    X = np.array([[1, 0], [0, 1]])
    Y = np.array([[0, 1], [1, 0]])

    expected_result_indices = [(0, 1), (1, 0)]
    expected_result_scores = [1.0, 1.0]

    result_indices, result_scores = cosine_similarity_top_k(X, Y)

    assert result_indices == expected_result_indices
    assert np.allclose(result_scores, expected_result_scores, rtol=1e-2)
