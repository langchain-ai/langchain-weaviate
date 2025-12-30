"""Test Weaviate functionality."""

import logging
import re
import uuid
from typing import Any, List, Optional, Union, cast

import pytest
import requests
import weaviate  # type: ignore
from langchain_core.documents import Document

from langchain_weaviate.vectorstores import WeaviateVectorStore

from .fake_embeddings import ConsistentFakeEmbeddings, FakeEmbeddings

logging.basicConfig(level=logging.DEBUG)


def setup_module(module: Any) -> None:
    pass


def is_ready(url: str) -> bool:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        pass
    return False


@pytest.fixture(scope="function")
def weaviate_client(docker_ip: Any, docker_services: Any) -> Any:
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
def texts() -> List[str]:
    return ["foo", "bar", "baz"]


@pytest.fixture
def embedding() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def consistent_embedding() -> ConsistentFakeEmbeddings:
    return ConsistentFakeEmbeddings()


def test_similarity_search_without_metadata(
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and search without metadata."""
    texts = ["foo", "bar", "baz"]
    docsearch = WeaviateVectorStore.from_texts(
        texts,
        consistent_embedding,
        client=weaviate_client,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_similarity_search_with_metadata(
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and search with metadata."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = WeaviateVectorStore.from_texts(
        texts, consistent_embedding, metadatas=metadatas, client=weaviate_client
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_similarity_search_with_metadata_and_filter(
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and search with metadata."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = WeaviateVectorStore.from_texts(
        texts, consistent_embedding, metadatas=metadatas, client=weaviate_client
    )
    output = docsearch.similarity_search(
        "foo", k=2, filters=weaviate.classes.query.Filter.by_property("page").equal(0)
    )
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_similarity_search_with_metadata_and_additional(
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and search with metadata and additional."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = WeaviateVectorStore.from_texts(
        texts, consistent_embedding, metadatas=metadatas, client=weaviate_client
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
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and search with uuids."""
    texts = ["foo", "bar", "baz"]
    # Weaviate replaces the object if the UUID already exists
    uuids = [uuid.uuid5(uuid.NAMESPACE_DNS, "same-name") for text in texts]

    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = WeaviateVectorStore.from_texts(
        texts,
        consistent_embedding,
        metadatas=metadatas,
        client=weaviate_client,
        uuids=uuids,
    )
    output = docsearch.similarity_search("foo", k=2)
    assert len(output) == 1


def test_similarity_search_by_text(
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and search by text."""

    docsearch = WeaviateVectorStore.from_texts(
        texts,
        consistent_embedding,
        client=weaviate_client,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert "foo" in output[0].page_content


def test_max_marginal_relevance_search(
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]

    docsearch = WeaviateVectorStore.from_texts(
        texts, consistent_embedding, metadatas=metadatas, client=weaviate_client
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
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and MRR search by vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]

    docsearch = WeaviateVectorStore.from_texts(
        texts, consistent_embedding, metadatas=metadatas, client=weaviate_client
    )
    foo_embedding = consistent_embedding.embed_query("foo")

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
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]

    docsearch = WeaviateVectorStore.from_texts(
        texts, consistent_embedding, metadatas=metadatas, client=weaviate_client
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
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    embedding: FakeEmbeddings,
) -> None:
    docsearch = WeaviateVectorStore.from_texts(
        texts, embedding=embedding, client=weaviate_client
    )

    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", alpha=1, k=2)
    assert output == [
        Document(page_content="foo"),
        Document(page_content="foo"),
    ]


def test_add_texts_with_given_uuids(
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    embedding: FakeEmbeddings,
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
    output = docsearch.similarity_search("foo", alpha=1, k=2)
    assert output[0] == Document(page_content="foo")
    assert output[1] != Document(page_content="foo")


def test_add_texts_with_metadata(
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    embedding: FakeEmbeddings,
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
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    embedding: FakeEmbeddings,
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
    weaviate_client: weaviate.WeaviateClient,
    consistent_embedding: ConsistentFakeEmbeddings,
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
        texts, consistent_embedding, client=weaviate_client
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
    use_multi_tenancy: bool,
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
    caplog: Any,
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
    use_multi_tenancy: bool,
    tenant_name: Optional[str],
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
    weaviate_client: weaviate.WeaviateClient,
    texts: List[str],
    consistent_embedding: ConsistentFakeEmbeddings,
) -> None:
    index_name = f"Index_{uuid.uuid4().hex}"
    tenant_name = "Foo"

    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embedding=consistent_embedding,
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


def test_embedding_property(
    weaviate_client: Any, consistent_embedding: ConsistentFakeEmbeddings
) -> None:
    index_name = "test_index"
    text_key = "text"

    docsearch = WeaviateVectorStore(
        client=weaviate_client,
        index_name=index_name,
        text_key=text_key,
        embedding=consistent_embedding,
    )

    assert type(docsearch.embeddings) == type(consistent_embedding)


def test_documents_with_many_properties(
    weaviate_client: Any, consistent_embedding: ConsistentFakeEmbeddings
) -> None:
    data = [
        {
            "aliases": ["Big Tech Co", "Tech Giant"],
            "categoryid": "101",
            "name": "Tech Innovations Drive Market Surge",
            "page_content": "The latest product launch by Big Tech Co "
            "has exceeded expectations, "
            "pushing its stock to record highs and invigorating the tech sector.",
            "ticker": "BTCH",
        },
        {
            "aliases": ["Global Energy Leader", "Energy Corp"],
            "categoryid": "102",
            "name": "Energy Corp Announces Renewable Initiative",
            "page_content": "In a bold move towards sustainability, "
            "Energy Corp has unveiled plans "
            "to significantly increase its investment in renewable energy sources, "
            "sparking investor interest.",
            "ticker": "GEL",
        },
        {
            "aliases": ["Pharma Pioneer", "Healthcare Innovator"],
            "categoryid": "103",
            "name": "Breakthrough Drug Approval",
            "page_content": "Pharma Pioneer's latest drug has received FDA approval, "
            "setting the stage "
            "for a major shift in treatment options and a positive outlook for the "
            "company's stock.",
            "ticker": "PPHI",
        },
    ]

    uuids = [uuid.uuid4().hex for _ in range(3)]
    properties = set(data[0].keys())

    index_name = f"TestIndex_{uuid.uuid4().hex}"
    text_key = "page_content"

    # since text_key is a separate field in a LangChain Document,
    # we remove it from the properties
    properties.remove(text_key)

    docsearch = WeaviateVectorStore(
        client=weaviate_client,
        index_name=index_name,
        # in default schema, "page_content" is stored in "text" property
        text_key="text",
        embedding=consistent_embedding,
    )

    texts: List[str] = [cast(str, doc["page_content"]) for doc in data]
    metadatas = [{k: doc[k] for k in doc if k != "page_content"} for doc in data]
    doc_ids = docsearch.add_texts(texts, metadatas=metadatas, uuids=uuids)

    weaviate_client.collections.get(index_name).query.fetch_object_by_id(doc_ids[0])

    # by default, all the properties are returned
    doc = docsearch.similarity_search("foo", k=1)[0]
    assert set(doc.metadata.keys()) == properties

    # you can also specify which properties to return
    doc = docsearch.similarity_search("foo", k=1, return_properties=["ticker"])[0]
    assert set(doc.metadata.keys()) == {"ticker"}

    # returning the uuids requires a different method
    doc = docsearch.similarity_search(
        "foo", k=1, return_uuids=True, return_properties=["ticker", "categoryid"]
    )[0]
    assert set(doc.metadata.keys()) == {"uuid", "ticker", "categoryid"}


def test_ingest_bad_documents(
    weaviate_client: Any, consistent_embedding: ConsistentFakeEmbeddings, caplog: Any
) -> None:
    # try to ingest 2 documents
    docs = [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"_additional": 1}),
    ]
    uuids = [weaviate.util.generate_uuid5(doc) for doc in docs]

    index_name = f"TestIndex_{uuid.uuid4().hex}"
    text_key = "page_content"

    with caplog.at_level(logging.ERROR):
        _ = WeaviateVectorStore.from_documents(
            documents=docs,
            embedding=consistent_embedding,
            client=weaviate_client,
            index_name=index_name,
            text_key=text_key,
            uuids=uuids,
        )

        good_doc_uuid, bad_doc_uuid = uuids

        # the bad doc should generate a log message
        pattern = r"ERROR.*Failed to add object: {}".format(bad_doc_uuid)
        assert re.search(pattern, caplog.text)
        assert good_doc_uuid not in caplog.text

        # the good doc should still be ingested
        total_docs = (
            weaviate_client.collections.get(index_name)
            .aggregate.over_all(total_count=True)
            .total_count
        )
        assert total_docs == 1
        assert weaviate_client.collections.get(index_name).query.fetch_object_by_id(
            good_doc_uuid
        )


@pytest.mark.parametrize("auto_limit, expected_num_docs", [(0, 4), (1, 3)])
def test_autocut(weaviate_client: Any, auto_limit: int, expected_num_docs: int) -> None:
    index_name = f"TestIndex_{uuid.uuid4().hex}"
    text_key = "page_content"

    # 4 documents, 3 of which are semantically similar to each other and 1 is very
    # dissimilar. So, by default, query should return all 4 documents but with
    # auto_limit=1, the dissimilar document should be cut off
    texts = [
        "Renewable energy sources like solar and wind power play a crucial role in "
        "combating climate change.",
        "The transition to green energy technologies is vital for reducing global "
        "carbon emissions.",
        "Investing in sustainable energy solutions is key to achieving environmental "
        "conservation goals.",
        "Ancient civilizations developed complex societies without the use of modern "
        "technology.",
    ]

    query = "How does the use of renewable resources impact ecological sustainability?"

    docsearch = WeaviateVectorStore.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        client=weaviate_client,
        index_name=index_name,
        text_key=text_key,
    )

    def run_similarity_test(search_method: str) -> None:
        f = getattr(docsearch, search_method)
        results = f(
            query=query,
            auto_limit=auto_limit,
            k=len(texts),
            fusion_type=weaviate.classes.query.HybridFusion.RELATIVE_SCORE,
        )

        actual_num_docs = len(results)

        assert actual_num_docs == expected_num_docs

    run_similarity_test("similarity_search")

    run_similarity_test("similarity_search_with_score")


def test_invalid_search_param(
    weaviate_client: Any, consistent_embedding: ConsistentFakeEmbeddings
) -> None:
    index_name = f"TestIndex_{uuid.uuid4().hex}"
    text_key = "page"
    weaviate_vector_store = WeaviateVectorStore(
        weaviate_client, index_name, text_key, consistent_embedding
    )

    with pytest.raises(ValueError) as excinfo:
        weaviate_vector_store._perform_search(query=None, k=5)

    assert str(excinfo.value) == "Either query or vector must be provided."

    with pytest.raises(ValueError) as excinfo:
        weaviate_vector_store._perform_search(query=None, vector=None, k=5)

    assert str(excinfo.value) == "Either query or vector must be provided."

    weaviate_vector_store._perform_search(query="hello", vector=None, k=5)

    weaviate_vector_store._perform_search(query=None, vector=[1, 2, 3], k=5)
