"""Test Weaviate functionality."""
import logging
import os
import uuid
from typing import Generator, Union

import pytest
from langchain_core.documents import Document
import requests

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.weaviate import Weaviate
from .fake_embeddings import FakeEmbeddings

logging.basicConfig(level=logging.DEBUG)

def setup_module(module):
    if 'OPENAI_API_KEY' not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY must be defined before running tests")

def is_ready(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False

@pytest.fixture(scope="function")
def weaviate_url(docker_ip, docker_services):
    port = docker_services.port_for("weaviate", 8080)
    url = "http://{}:{}".format(docker_ip, port)

    ready_endpoint = url + "/v1/.well-known/ready"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_ready(ready_endpoint)
    )

    yield url
    
@pytest.fixture
def embedding_openai():
    yield OpenAIEmbeddings()

def test_similarity_search_without_metadata(
        weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search without metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = Weaviate.from_texts(
            texts,
            embedding_openai,
            weaviate_url=weaviate_url,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

def test_similarity_search_with_metadata(
        weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Weaviate.from_texts(
            texts, embedding_openai, metadatas=metadatas, weaviate_url=weaviate_url
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": 0})]

def test_similarity_search_with_metadata_and_filter(
        weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Weaviate.from_texts(
            texts, embedding_openai, metadatas=metadatas, weaviate_url=weaviate_url
        )
        output = docsearch.similarity_search(
            "foo",
            k=2,
            where_filter={"path": ["page"], "operator": "Equal", "valueNumber": 0},
        )
        assert output == [Document(page_content="foo", metadata={"page": 0})]

def test_similarity_search_with_metadata_and_additional(
        weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search with metadata and additional."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Weaviate.from_texts(
            texts, embedding_openai, metadatas=metadatas, weaviate_url=weaviate_url
        )
        output = docsearch.similarity_search(
            "foo",
            k=1,
            additional=["certainty"],
        )
        assert output == [
            Document(
                page_content="foo",
                metadata={"page": 0, "_additional": {"certainty": 1.0000003576278687}},
            )
        ]

def test_similarity_search_with_uuids(
        weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search with uuids."""
        texts = ["foo", "bar", "baz"]
        # Weaviate replaces the object if the UUID already exists
        uuids = [uuid.uuid5(uuid.NAMESPACE_DNS, "same-name") for text in texts]

        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Weaviate.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            weaviate_url=weaviate_url,
            uuids=uuids,
        )
        output = docsearch.similarity_search("foo", k=2)
        assert len(output) == 1

def test_max_marginal_relevance_search(
        weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and MRR search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        docsearch = Weaviate.from_texts(
            texts, embedding_openai, metadatas=metadatas, weaviate_url=weaviate_url
        )
        # if lambda=1 the algorithm should be equivalent to standard ranking
        standard_ranking = docsearch.similarity_search("foo", k=2)
        output = docsearch.max_marginal_relevance_search(
            "foo", k=2, fetch_k=3, lambda_mult=1.0
        )
        assert output == standard_ranking

        # if lambda=0 the algorithm should favour maximal diversity
        output = docsearch.max_marginal_relevance_search(
            "foo", k=2, fetch_k=3, lambda_mult=0.0
        )
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
        ]

def test_max_marginal_relevance_search_by_vector(
        weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and MRR search by vector."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        docsearch = Weaviate.from_texts(
            texts, embedding_openai, metadatas=metadatas, weaviate_url=weaviate_url
        )
        foo_embedding = embedding_openai.embed_query("foo")

        # if lambda=1 the algorithm should be equivalent to standard ranking
        standard_ranking = docsearch.similarity_search("foo", k=2)
        output = docsearch.max_marginal_relevance_search_by_vector(
            foo_embedding, k=2, fetch_k=3, lambda_mult=1.0
        )
        assert output == standard_ranking

        # if lambda=0 the algorithm should favour maximal diversity
        output = docsearch.max_marginal_relevance_search_by_vector(
            foo_embedding, k=2, fetch_k=3, lambda_mult=0.0
        )
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
        ]

def test_max_marginal_relevance_search_with_filter(
        weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and MRR search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        docsearch = Weaviate.from_texts(
            texts, embedding_openai, metadatas=metadatas, weaviate_url=weaviate_url
        )
        where_filter = {"path": ["page"], "operator": "Equal", "valueNumber": 0}
        # if lambda=1 the algorithm should be equivalent to standard ranking
        standard_ranking = docsearch.similarity_search(
            "foo", k=2, where_filter=where_filter
        )
        output = docsearch.max_marginal_relevance_search(
            "foo", k=2, fetch_k=3, lambda_mult=1.0, where_filter=where_filter
        )
        assert output == standard_ranking

        # if lambda=0 the algorithm should favour maximal diversity
        output = docsearch.max_marginal_relevance_search(
            "foo", k=2, fetch_k=3, lambda_mult=0.0, where_filter=where_filter
        )
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
        ]

def test_add_texts_with_given_embedding(weaviate_url: str) -> None:
        texts = ["foo", "bar", "baz"]
        embedding = FakeEmbeddings()

        docsearch = Weaviate.from_texts(
            texts, embedding=embedding, weaviate_url=weaviate_url
        )

        docsearch.add_texts(["foo"])
        output = docsearch.similarity_search_by_vector(
            embedding.embed_query("foo"), k=2
        )
        assert output == [
            Document(page_content="foo"),
            Document(page_content="foo"),
        ]

def test_add_texts_with_given_uuids(weaviate_url: str) -> None:
        texts = ["foo", "bar", "baz"]
        embedding = FakeEmbeddings()
        uuids = [uuid.uuid5(uuid.NAMESPACE_DNS, text) for text in texts]

        docsearch = Weaviate.from_texts(
            texts,
            embedding=embedding,
            weaviate_url=weaviate_url,
            uuids=uuids,
        )

        # Weaviate replaces the object if the UUID already exists
        docsearch.add_texts(["foo"], uuids=[uuids[0]])
        output = docsearch.similarity_search_by_vector(
            embedding.embed_query("foo"), k=2
        )
        assert output[0] == Document(page_content="foo")
        assert output[1] != Document(page_content="foo")