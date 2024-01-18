from __future__ import annotations

import datetime
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from uuid import uuid4

import numpy as np
import weaviate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from weaviate.util import get_valid_uuid

from .utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import weaviate


def _default_schema(index_name: str) -> Dict:
    return {
        "class": index_name,
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
            }
        ],
    }


def _default_score_normalizer(val: float) -> float:
    return 1 - 1 / (1 + np.exp(val))


def _json_serializable(value: Any) -> Any:
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return value


class WeaviateVectorStore(VectorStore):
    """`Weaviate` vector store.

    To use, you should have the ``weaviate-client`` python package installed.

    Example:
        .. code-block:: python

            import weaviate
            from langchain_community.vectorstores import Weaviate

            client = weaviate.Client(url=os.environ["WEAVIATE_URL"], ...)
            weaviate = Weaviate(client, index_name, text_key)

    """

    def __init__(
        self,
        client: weaviate.WeaviateClient,
        index_name: str,
        text_key: str,
        embedding: Optional[Embeddings] = None,
        attributes: Optional[List[str]] = None,
        relevance_score_fn: Optional[
            Callable[[float], float]
        ] = _default_score_normalizer,
        by_text: bool = True,
    ):
        """Initialize with Weaviate client."""

        if not isinstance(client, weaviate.WeaviateClient):
            raise ValueError(
                f"client should be an instance of weaviate.WeaviateClient, got {type(client)}"
            )
        self._client = client
        self._index_name = index_name
        self._embedding = embedding
        self._text_key = text_key
        self._query_attrs = [self._text_key]
        self.relevance_score_fn = relevance_score_fn
        self._by_text = by_text
        if attributes is not None:
            self._query_attrs.extend(attributes)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return (
            self.relevance_score_fn
            if self.relevance_score_fn
            else _default_score_normalizer
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload texts with metadata (properties) to Weaviate."""
        from weaviate.util import get_valid_uuid

        ids = []
        embeddings: Optional[List[List[float]]] = None
        if self._embedding:
            if not isinstance(texts, list):
                texts = list(texts)
            embeddings = self._embedding.embed_documents(texts)

        with self._client.batch as batch:
            for i, text in enumerate(texts):
                data_properties = {self._text_key: text}
                if metadatas is not None:
                    for key, val in metadatas[i].items():
                        data_properties[key] = _json_serializable(val)

                # Allow for ids (consistent w/ other methods)
                # # Or uuids (backwards compatible w/ existing arg)
                # If the UUID of one of the objects already exists
                # then the existing object will be replaced by the new object.
                _id = get_valid_uuid(uuid4())
                if "uuids" in kwargs:
                    _id = kwargs["uuids"][i]
                elif "ids" in kwargs:
                    _id = kwargs["ids"][i]

                batch.add_object(
                    collection=self._index_name,
                    properties=data_properties,
                    uuid=_id,
                    vector=embeddings[i] if embeddings else None,
                    tenant=kwargs.get("tenant"),
                )

                ids.append(_id)
        return ids

    def _perform_search(
        self,
        query: str,
        k: int,
        return_score=False,
        search_method: Literal["hybrid", "near_vector"] = "hybrid",
        **kwargs: Any,
    ) -> List[Union[Document, Tuple[Document, float]]]:
        """
        Perform a similarity search.

        Parameters:
        query (str): The query string to search for.
        k (int): The number of results to return.
        return_score (bool, optional): Whether to return the score along with the document. Defaults to False.
        search_method (Literal['hybrid', 'near_vector'], optional): The search method to use. Can be 'hybrid' or 'near_vector'. Defaults to 'hybrid'.
        **kwargs: Additional parameters to pass to the search method. These parameters will be directly passed to the underlying Weaviate client's search method.

        Returns:
        List[Union[Document, Tuple[Document, float]]]: A list of documents that match the query. If return_score is True, each document is returned as a tuple with the document and its score.

        Raises:
        ValueError: If _embedding is None or an invalid search method is provided.
        """
        if self._embedding is None:
            raise ValueError("_embedding cannot be None for similarity_search")

        if "return_metadata" in kwargs and "score" not in kwargs["return_metadata"]:
            kwargs["return_metadata"].append("score")
        else:
            kwargs["return_metadata"] = ["score"]

        try:
            if search_method == "hybrid":
                embedding = self._embedding.embed_query(query)
                result = self._client.collections.get(self._index_name).query.hybrid(
                    query=query, vector=embedding, limit=k, **kwargs
                )
            elif search_method == "near_vector":
                result = self._client.collections.get(
                    self._index_name
                ).query.near_vector(limit=k, **kwargs)
            else:
                raise ValueError(f"Invalid search method: {search_method}")
        except weaviate.exceptions.WeaviateQueryException as e:
            raise ValueError(f"Error during query: {e}")

        docs = []
        for obj in result.objects:
            text = obj.properties.pop(self._text_key)
            filtered_metadata = {
                k: v
                for k, v in obj.metadata.__dict__.items()
                if v is not None and k != "score"
            }
            merged_props = {
                **obj.properties,
                **filtered_metadata,
                **({"vector": obj.vector} if obj.vector else {}),
            }
            doc = Document(page_content=text, metadata=merged_props)
            if not return_score:
                docs.append(doc)
            else:
                score = obj.metadata.score
                docs.append((doc, score))

        return docs

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Additional keyword arguments will be passed to the `hybrid()` function of the weaviate client.

        Returns:
            List of Documents most similar to the query.
        """

        result = self._perform_search(query, k, **kwargs)
        return result

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Look up similar documents by embedding vector in Weaviate."""

        return self._perform_search(
            query=None,
            k=k,
            near_vector=embedding,
            search_method="near_vector",
            **kwargs,
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._embedding is not None:
            embedding = self._embedding.embed_query(query)
        else:
            raise ValueError(
                "max_marginal_relevance_search requires a suitable Embeddings object"
            )

        return self.max_marginal_relevance_search_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        results = self._perform_search(
            query=None,
            k=fetch_k,
            include_vector=True,
            near_vector=embedding,
            search_method="near_vector",
            **kwargs,
        )

        embeddings = [result.metadata["vector"] for result in results]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )

        docs = []

        for idx in mmr_selected:
            text = results[idx].page_content
            results[idx].metadata.pop("vector")
            docs.append(Document(page_content=text, metadata=results[idx].metadata))

        return docs

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Return list of documents most similar to the query
        text and cosine distance in float for each.
        Lower score represents more similarity.
        """

        results = self._perform_search(query, k, return_score=True, **kwargs)

        return results

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        client: weaviate.WeaviateClient = None,
        metadatas: Optional[List[dict]] = None,
        *,
        weaviate_api_key: Optional[str] = None,
        batch_size: Optional[int] = 25,
        index_name: Optional[str] = None,
        text_key: str = "text",
        by_text: bool = False,
        relevance_score_fn: Optional[
            Callable[[float], float]
        ] = _default_score_normalizer,
        **kwargs: Any,
    ) -> WeaviateVectorStore:
        """Construct Weaviate wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the Weaviate instance.
            3. Adds the documents to the newly created Weaviate index.

        This is intended to be a quick way to get started.

        Args:
            texts: Texts to add to vector store.
            embedding: Text embedding model to use.
            metadatas: Metadata associated with each text.
            client: weaviate.Client to use.
            weaviate_url: The Weaviate URL. If using Weaviate Cloud Services get it
                from the ``Details`` tab. Can be passed in as a named param or by
                setting the environment variable ``WEAVIATE_URL``. Should not be
                specified if client is provided.
            weaviate_api_key: The Weaviate API key. If enabled and using Weaviate Cloud
                Services, get it from ``Details`` tab. Can be passed in as a named param
                or by setting the environment variable ``WEAVIATE_API_KEY``. Should
                not be specified if client is provided.
            batch_size: Size of batch operations.
            index_name: Index name.
            text_key: Key to use for uploading/retrieving text to/from vectorstore.
            by_text: Whether to search by text or by embedding.
            relevance_score_fn: Function for converting whatever distance function the
                vector store uses to a relevance score, which is a normalized similarity
                score (0 means dissimilar, 1 means similar).
            **kwargs: Additional named parameters to pass to ``Weaviate.__init__()``.

        Example:
            .. code-block:: python

                from langchain_community.embeddings import OpenAIEmbeddings
                from langchain_community.vectorstores import Weaviate

                embeddings = OpenAIEmbeddings()
                weaviate = Weaviate.from_texts(
                    texts,
                    embeddings,
                    client=client
                )
        """

        index_name = index_name or f"LangChain_{uuid4().hex}"
        schema = _default_schema(index_name)
        # check whether the index already exists
        if not client.collections.exists(index_name):
            client.collections.create_from_dict(schema)

        embeddings = embedding.embed_documents(texts) if embedding else None
        attributes = list(metadatas[0].keys()) if metadatas else None

        # If the UUID of one of the objects already exists
        # then the existing object will be replaced by the new object.
        if "uuids" in kwargs:
            uuids = kwargs.pop("uuids")
        else:
            uuids = [get_valid_uuid(uuid4()) for _ in range(len(texts))]

        with client.batch as batch:
            for i, text in enumerate(texts):
                data_properties = {
                    text_key: text,
                }
                if metadatas is not None:
                    for key in metadatas[i].keys():
                        data_properties[key] = metadatas[i][key]

                _id = uuids[i]

                # if an embedding strategy is not provided, we let
                # weaviate create the embedding. Note that this will only
                # work if weaviate has been installed with a vectorizer module
                # like text2vec-contextionary for example
                params = {
                    "uuid": _id,
                    "properties": data_properties,
                    "collection": index_name,
                }
                if embeddings is not None:
                    params["vector"] = embeddings[i]

                batch.add_object(**params)

            batch.flush()

        return cls(
            client,
            index_name,
            text_key,
            embedding=embedding,
            attributes=attributes,
            relevance_score_fn=relevance_score_fn,
            by_text=by_text,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        # TODO: Check if this can be done in bulk
        for id in ids:
            self._client.collections.get(self._index_name).data.delete_by_id(id)
