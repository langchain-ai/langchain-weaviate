from langchain_weaviate.vectorstores import WeaviateVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    WeaviateVectorStore()
