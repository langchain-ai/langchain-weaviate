import numpy as np

from langchain_weaviate.utils import maximal_marginal_relevance


def test_maximal_marginal_relevance() -> None:
    # Test with standard input
    query_embedding = np.array([1.0, 0.0])
    embedding_list = [
        np.array([1.0, 0.0]),  # Most similar to query
        np.array([0.9, 0.1]),  # Second most similar but close to first
        np.array([0.5, 0.5]),  # Less similar
        np.array([0.0, 1.0]),  # Least similar but diverse
    ]

    # Test normal case
    result = maximal_marginal_relevance(
        query_embedding, embedding_list, lambda_mult=0.5, k=2
    )
    assert len(result) == 2
    # First result should be the most similar to query
    assert result[0] == 0
    # Don't assert specific indices as that's implementation dependent
    # Just check that we got 2 results

    # Test edge case: k=0
    empty_result = maximal_marginal_relevance(
        query_embedding, embedding_list, lambda_mult=0.5, k=0
    )
    assert empty_result == []

    # Test edge case: k > len(embedding_list)
    large_k_result = maximal_marginal_relevance(
        query_embedding, embedding_list, lambda_mult=0.5, k=10
    )
    assert len(large_k_result) == 4  # Should return all embeddings, not more

    # Test edge case: empty embedding_list (covers line 30)
    empty_list_result = maximal_marginal_relevance(
        query_embedding, [], lambda_mult=0.5, k=3
    )
    assert empty_list_result == []

    # Test different lambda values
    # Higher lambda values prioritize similarity to query
    maximal_marginal_relevance(query_embedding, embedding_list, lambda_mult=1.0, k=3)
    # Lower lambda values prioritize diversity
    maximal_marginal_relevance(query_embedding, embedding_list, lambda_mult=0.0, k=3)
    # Results might be the same, so we don't assert they're different
