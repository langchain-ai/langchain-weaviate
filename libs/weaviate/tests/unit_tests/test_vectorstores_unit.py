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
