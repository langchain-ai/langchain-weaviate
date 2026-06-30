"""Test package metadata and imports."""

from langchain_weaviate import __version__, WeaviateVectorStore


def test_version_is_string() -> None:
    """Verify that __version__ is a non-empty string."""
    assert isinstance(__version__, str)
    assert __version__
    assert __version__ == "0.0.7"


def test_all_exports() -> None:
    """Verify that the public API surface is available."""
    assert WeaviateVectorStore is not None
