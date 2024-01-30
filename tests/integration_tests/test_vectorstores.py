"""Test Weaviate functionality."""
import logging
import os
import uuid
from typing import List, Union

import pytest
import requests
import weaviate
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_weaviate.vectorstores import WeaviateVectorStore

from .fake_embeddings import FakeEmbeddings

logging.basicConfig(level=logging.DEBUG)


def setup_module(module):
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY must be defined before running tests")


def is_ready(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False


@pytest.fixture(scope="function")
def weaviate_client(docker_ip, docker_services):
    http_port = docker_services.port_for("weaviate", 8080)
    grpc_port = docker_services.port_for("weaviate", 50051)
    url = f"http://{docker_ip}:{http_port}"

    ready_endpoint = url + "/v1/.well-known/ready"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_ready(ready_endpoint)
    )

    client = weaviate.WeaviateClient(
        connection_params=weaviate.connect.ConnectionParams.from_params(
            http_host=docker_ip,
            http_port=http_port,
            http_secure=False,
            grpc_host=docker_ip,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
    )

    client.connect()
    yield client
    client.close()


@pytest.fixture
def embedding_openai():
    yield OpenAIEmbeddings()


@pytest.fixture
def texts():
    return ["foo", "bar", "baz"]


@pytest.fixture
def embedding():
    return FakeEmbeddings()


def test_similarity_search_without_metadata(
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    """Test end to end construction and search without metadata."""
    texts = ["foo", "bar", "baz"]
    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embedding_openai,
        client=weaviate_client,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_similarity_search_with_metadata(
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    """Test end to end construction and search with metadata."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding_openai, metadatas=metadatas, client=weaviate_client
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_similarity_search_with_metadata_and_filter(
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    """Test end to end construction and search with metadata."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding_openai, metadatas=metadatas, client=weaviate_client
    )
    output = docsearch.similarity_search(
        "foo", k=2, filters=weaviate.classes.query.Filter.by_property("page").equal(0)
    )
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_similarity_search_with_metadata_and_additional(
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    """Test end to end construction and search with metadata and additional."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding_openai, metadatas=metadatas, client=weaviate_client
    )
    output = docsearch.similarity_search(
        "foo",
        k=1,
        return_metadata=["creation_time"],
    )

    assert len(output) == 1
    doc = output[0]
    assert doc.page_content == "foo"
    assert "creation_time" in doc.metadata


def test_similarity_search_with_uuids(
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    """Test end to end construction and search with uuids."""
    texts = ["foo", "bar", "baz"]
    # Weaviate replaces the object if the UUID already exists
    uuids = [uuid.uuid5(uuid.NAMESPACE_DNS, "same-name") for text in texts]

    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embedding_openai,
        metadatas=metadatas,
        client=weaviate_client,
        uuids=uuids,
    )
    output = docsearch.similarity_search("foo", k=2)
    assert len(output) == 1


def test_similarity_search_by_text(
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    embedding_openai: OpenAIEmbeddings,
) -> None:
    """Test end to end construction and search by text."""

    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding_openai, client=weaviate_client, by_text=True
    )

    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert "foo" in output[0].page_content


def test_max_marginal_relevance_search(
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]

    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding_openai, metadatas=metadatas, client=weaviate_client
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
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    """Test end to end construction and MRR search by vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]

    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding_openai, metadatas=metadatas, client=weaviate_client
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
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]

    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding_openai, metadatas=metadatas, client=weaviate_client
    )

    is_page_0_filter = weaviate.classes.query.Filter.by_property("page").equal(0)
    # if lambda=1 the algorithm should be equivalent to standard ranking
    standard_ranking = docsearch.similarity_search("foo", k=2, filters=is_page_0_filter)
    output = docsearch.max_marginal_relevance_search(
        "foo", k=2, fetch_k=3, lambda_mult=1.0, filters=is_page_0_filter
    )
    assert output == standard_ranking

    # if lambda=0 the algorithm should favour maximal diversity
    output = docsearch.max_marginal_relevance_search(
        "foo", k=2, fetch_k=3, lambda_mult=0.0, filters=is_page_0_filter
    )
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
    ]


def test_add_texts_with_given_embedding(
    weaviate_client: weaviate.WeaviateClient, texts, embedding
) -> None:
    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding=embedding, client=weaviate_client
    )

    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search_by_vector(embedding.embed_query("foo"), k=2)
    assert output == [
        Document(page_content="foo"),
        Document(page_content="foo"),
    ]


def test_add_texts_with_given_uuids(
    weaviate_client: weaviate.WeaviateClient, texts, embedding
) -> None:
    uuids = [uuid.uuid5(uuid.NAMESPACE_DNS, text) for text in texts]

    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embedding=embedding,
        client=weaviate_client,
        uuids=uuids,
    )

    # Weaviate replaces the object if the UUID already exists
    docsearch.add_texts(["foo"], uuids=[uuids[0]])
    output = docsearch.similarity_search_by_vector(embedding.embed_query("foo"), k=2)
    assert output[0] == Document(page_content="foo")
    assert output[1] != Document(page_content="foo")


def test_add_texts_with_metadata(
    weaviate_client: weaviate.WeaviateClient, texts, embedding
) -> None:
    """
    Test that the text's metadata ends up in Weaviate too
    """

    index_name = f"TestIndex_{uuid.uuid4().hex}"

    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embedding=embedding,
        client=weaviate_client,
        index_name=index_name,
    )

    ids = docsearch.add_texts(["qux"], metadatas=[{"page": 1}])

    expected_result = {
        "page": 1,
        "text": "qux",
    }

    doc = weaviate_client.collections.get(index_name).query.fetch_object_by_id(ids[0])
    result = doc.properties

    assert result == expected_result


def test_add_text_with_given_id(
    weaviate_client: weaviate.WeaviateClient, texts, embedding
) -> None:
    """
    Test that the text's id ends up in Weaviate too
    """

    index_name = f"TestIndex_{uuid.uuid4().hex}"
    doc_id = uuid.uuid4()

    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embedding=embedding,
        client=weaviate_client,
        index_name=index_name,
    )

    docsearch.add_texts(["qux"], ids=[doc_id])

    doc = weaviate_client.collections.get(index_name).query.fetch_object_by_id(doc_id)

    assert str(doc.uuid) == str(doc_id)


def test_similarity_search_with_score(
    weaviate_client: weaviate.WeaviateClient, embedding_openai: OpenAIEmbeddings
) -> None:
    texts = ["cat", "dog"]

    # create a weaviate instance without an embedding
    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding=None, client=weaviate_client
    )

    with pytest.raises(ValueError, match="_embedding cannot be None"):
        docsearch.similarity_search_with_score("foo", k=2)

    weaviate_client.collections.delete_all()

    # now create an instance with an embedding
    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding_openai, client=weaviate_client, by_text=False
    )

    results = docsearch.similarity_search_with_score("kitty", k=1)

    assert len(results) == 1

    doc, score = results[0]

    assert isinstance(score, float)
    assert score > 0
    assert doc.page_content == "cat"


@pytest.mark.parametrize(
    "use_multi_tenancy, tenant", [(True, "TestTenant"), (False, None)]
)
def test_delete(
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    embedding: FakeEmbeddings,
    use_multi_tenancy: bool,
    tenant: Union[str, None],
) -> None:
    index_name = f"Index_{uuid.uuid4().hex}"

    docsearch = WeaviateVectorStore(
        client=weaviate_client,
        index_name=index_name,
        text_key="text",
        embedding=embedding,
        use_multi_tenancy=use_multi_tenancy,
    )
    docids = docsearch.add_texts(texts, tenant=tenant)

    total_docs_before_delete = (
        weaviate_client.collections.get(index_name)
        .with_tenant(tenant)
        .aggregate.over_all(total_count=True)
        .total_count
    )
    docsearch.delete(docids, tenant=tenant)

    total_docs_after_delete = (
        weaviate_client.collections.get(index_name)
        .with_tenant(tenant)
        .aggregate.over_all(total_count=True)
        .total_count
    )

    assert total_docs_before_delete == len(texts)
    assert total_docs_after_delete == 0

    with pytest.raises(ValueError, match="No ids provided to delete"):
        docsearch.delete()


@pytest.mark.parametrize("use_multi_tenancy", [True, False])
def test_enable_multi_tenancy(
    use_multi_tenancy,
    weaviate_client: weaviate.WeaviateClient,
    embedding: FakeEmbeddings,
) -> None:
    index_name = f"Index_{uuid.uuid4().hex}"

    _ = WeaviateVectorStore(
        client=weaviate_client,
        index_name=index_name,
        text_key="text",
        embedding=embedding,
        use_multi_tenancy=use_multi_tenancy,
    )

    schema = weaviate_client.collections.get(index_name).config.get(simple=False)
    assert schema.multi_tenancy_config.enabled == use_multi_tenancy


def test_tenant_exists(
    weaviate_client: weaviate.WeaviateClient,
    embedding: FakeEmbeddings,
) -> None:
    index_name = "TestTenant"
    tenant_name = "Foo"
    tenant = weaviate.classes.tenants.Tenant(name=tenant_name)

    # a collection with mt enabled
    docsearch_with_mt = WeaviateVectorStore(
        client=weaviate_client,
        index_name=index_name,
        text_key="text",
        embedding=embedding,
        use_multi_tenancy=True,
    )

    assert not docsearch_with_mt._does_tenant_exist(tenant_name)

    weaviate_client.collections.get(index_name).tenants.create([tenant])

    assert docsearch_with_mt._does_tenant_exist(tenant_name)

    # make another collection without mt enabled
    docsearch_no_mt = WeaviateVectorStore(
        client=weaviate_client,
        index_name="Bar",
        text_key="text",
        embedding=embedding,
        use_multi_tenancy=False,
    )

    with pytest.raises(AssertionError, match="Cannot check for tenant existence"):
        docsearch_no_mt._does_tenant_exist(tenant_name)


def test_add_texts_with_multi_tenancy(
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    embedding: FakeEmbeddings,
    caplog,
) -> None:
    index_name = "TestMultiTenancy"
    tenant_name = "Foo"
    create_tenant_log_msg = f"Tenant {tenant_name} does not exist in index {index_name}"

    docsearch = WeaviateVectorStore(
        client=weaviate_client,
        index_name=index_name,
        text_key="text",
        embedding=embedding,
        use_multi_tenancy=True,
    )

    assert tenant_name not in weaviate_client.collections.get(index_name).tenants.get()

    with caplog.at_level(logging.INFO):
        docsearch.add_texts(texts, tenant=tenant_name)
        assert create_tenant_log_msg in caplog.text

    caplog.clear()

    assert tenant_name in weaviate_client.collections.get(index_name).tenants.get()

    assert weaviate_client.collections.get(index_name).with_tenant(
        tenant_name
    ).aggregate.over_all(total_count=True).total_count == len(texts)

    # index again
    # this should not create a new tenant
    with caplog.at_level(logging.INFO):
        docsearch.add_texts(texts, tenant=tenant_name)
        assert create_tenant_log_msg not in caplog.text

    assert (
        weaviate_client.collections.get(index_name)
        .with_tenant(tenant_name)
        .aggregate.over_all(total_count=True)
        .total_count
        == len(texts) * 2
    )


@pytest.mark.parametrize(
    "use_multi_tenancy, tenant_name", [(True, "Foo"), (False, None)]
)
def test_simple_from_texts(
    use_multi_tenancy,
    tenant_name,
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    embedding: FakeEmbeddings,
) -> None:
    index_name = f"Index_{uuid.uuid4().hex}"

    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embedding=embedding,
        client=weaviate_client,
        index_name=index_name,
        tenant=tenant_name,
    )

    assert docsearch._multi_tenancy_enabled == use_multi_tenancy


def test_search_with_multi_tenancy(
    weaviate_client: weaviate.WeaviateClient, texts: List[str], embedding_openai
) -> None:
    index_name = f"Index_{uuid.uuid4().hex}"
    tenant_name = "Foo"

    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embedding=embedding_openai,
        client=weaviate_client,
        index_name=index_name,
        tenant=tenant_name,
    )

    # Note: we only test MT with similarity_search because all public search methods
    # in VectorStore takes kwargs to pass the tenant name, and internally calls
    # _perform_search to validate and run the search query

    # search without tenant with MT enabled
    with pytest.raises(
        ValueError, match="Must use tenant context when multi-tenancy is enabled"
    ):
        docsearch.similarity_search("foo", k=1)

    # search with tenant with MT enabled
    docsearch.similarity_search("foo", k=1, tenant=tenant_name)

    # search with tenant with MT disabled
    docsearch._multi_tenancy_enabled = (
        False  # doesn't actually do anything to weaviate's schema
    )

    with pytest.raises(
        ValueError, match="Cannot use tenant context when multi-tenancy is not enabled"
    ):
        docsearch.similarity_search("foo", k=1, tenant=tenant_name)

    # search without tenant with MT disabled
    # weaviate will throw an error because MT is still enabled and we didn't pass
    # a tenant
    with pytest.raises(
        ValueError, match="has multi-tenancy enabled, but request was without tenant"
    ):
        docsearch.similarity_search("foo", k=1)
