"""Fake Embedding class for testing purposes."""

import json
import math
from typing import Dict, List

from langchain_core.embeddings import Embeddings, FakeEmbeddings


class ConsistentFakeEmbeddings(FakeEmbeddings):
    """Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts."""

    def __init__(self) -> None:
        with open("./tests/unit_tests/fake_docs.json", "r") as f:
            self.known_texts: Dict[str, List[float]] = json.load(f)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return consistent embeddings for each text seen so far."""

        return [self.known_texts[text] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        return self.embed_documents([text])[0]


class AngularTwoDimensionalEmbeddings(Embeddings):
    """
    From angles (as strings in units of pi) to unit embedding vectors on a circle.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Make a list of texts into a list of embedding vectors.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Convert input text to a 'vector' (list of floats).
        If the text is a number, use it as the angle for the
        unit vector in units of pi.
        Any other input text becomes the singular result [0, 0] !
        """
        try:
            angle = float(text)
            return [math.cos(angle * math.pi), math.sin(angle * math.pi)]
        except ValueError:
            # Assume: just test string, no attention is paid to values.
            return [0.0, 0.0]
