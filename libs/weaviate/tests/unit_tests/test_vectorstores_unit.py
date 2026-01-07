import datetime
from typing import Union

import numpy as np
import pytest

from langchain_weaviate.vectorstores import (
    WeaviateVectorStore,
    _default_score_normalizer,
    _json_serializable,
)


@pytest.mark.parametrize("val, expected_result", [(1e6, 1.0), (-1e6, 0.0)])
def test_default_score_normalizer(val: float, expected_result: float) -> None:
    assert np.isclose(_default_score_normalizer(val), expected_result, atol=1e-6)


@pytest.mark.parametrize(
    "value, expected_result",
    [
        (datetime.datetime(2022, 1, 1, 12, 0, 0), "2022-01-01T12:00:00"),
        ("test", "test"),
        (123, 123),
        (None, None),
    ],
)
def test_json_serializable(
    value: Union[datetime.datetime, str, int, None],
    expected_result: Union[str, int, None],
) -> None:
    assert _json_serializable(value) == expected_result


def test_from_texts_raises_value_error_when_client_is_none() -> None:
    with pytest.raises(
        ValueError, match="client must be an instance of WeaviateClient"
    ):
        WeaviateVectorStore.from_texts(
            texts=["sample text"], embedding=None, client=None
        )


def test_select_relevance_score_fn_with_custom_function() -> None:
    """Test that custom relevance_score_fn is used when provided."""
    from unittest.mock import MagicMock

    def custom_score_fn(score: float) -> float:
        return score * 2.0

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_client.collections.get.return_value.config.get.return_value.multi_tenancy_config.enabled = (
        False
    )

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="test",
        text_key="text",
        relevance_score_fn=custom_score_fn,
    )

    result_fn = vectorstore._select_relevance_score_fn()
    assert result_fn == custom_score_fn
    assert result_fn(0.5) == 1.0


def test_max_marginal_relevance_search_without_embeddings() -> None:
    """Test that MMR search raises error when no embeddings provided."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_client.collections.get.return_value.config.get.return_value.multi_tenancy_config.enabled = (
        False
    )

    vectorstore = WeaviateVectorStore(
        client=mock_client, index_name="test", text_key="text", embedding=None
    )

    with pytest.raises(ValueError, match="requires a suitable Embeddings object"):
        vectorstore.max_marginal_relevance_search("query", k=4)
