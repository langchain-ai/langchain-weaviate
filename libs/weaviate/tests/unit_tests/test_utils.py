import numpy as np
from numpy.typing import NDArray

from langchain_weaviate.utils import maximal_marginal_relevance


def test_maximal_marginal_relevance_1d_query() -> None:
    """Test MMR with 1D query embedding."""
    query_embedding = np.array([1.0, 0.0])  # 1D
    embedding_list = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.5, 0.5]),
    ]

    result = maximal_marginal_relevance(query_embedding, embedding_list, k=2)
    assert len(result) <= 2
    assert all(isinstance(idx, int) for idx in result)


def test_maximal_marginal_relevance_2d_query() -> None:
    """Test MMR with 2D query embedding."""
    query_embedding = np.array([[1.0, 0.0]])  # 2D
    embedding_list = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.5, 0.5]),
    ]

    result = maximal_marginal_relevance(query_embedding, embedding_list, k=2)
    assert len(result) <= 2
    assert all(isinstance(idx, int) for idx in result)


def test_maximal_marginal_relevance_empty_list() -> None:
    """Test MMR with empty embedding list."""
    query_embedding = np.array([1.0, 0.0])
    embedding_list: list[NDArray] = []

    result = maximal_marginal_relevance(query_embedding, embedding_list, k=2)
    assert result == []


def test_maximal_marginal_relevance_k_zero() -> None:
    """Test MMR with k=0."""
    query_embedding = np.array([1.0, 0.0])
    embedding_list = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    result = maximal_marginal_relevance(query_embedding, embedding_list, k=0)
    assert result == []
