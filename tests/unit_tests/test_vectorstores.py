import numpy as np
import pytest

from langchain_weaviate.vectorstores import _default_score_normalizer


@pytest.mark.parametrize("val, expected_result", [(1e6, 1.0), (-1e6, 0.0)])
def test_default_score_normalizer(val, expected_result):
    assert np.isclose(_default_score_normalizer(val), expected_result, atol=1e-6)
