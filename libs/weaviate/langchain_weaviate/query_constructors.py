"""Translate LangChain structured queries into Weaviate filters."""

from datetime import datetime, timezone
from typing import Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from weaviate.classes.query import Filter
from weaviate.collections.classes.filters import _Filters


class WeaviateTranslator(Visitor):
    """Translate `Weaviate` internal query language elements to valid filters.

    This powers the LangChain self-querying retriever for
    :class:`~langchain_weaviate.vectorstores.WeaviateVectorStore`. Because the
    vectorstore is built on ``weaviate-client`` v4, the translator emits
    :class:`weaviate.classes.query.Filter` objects (passed through to the
    ``filters`` argument of the underlying search) rather than the legacy v3
    GraphQL ``where`` filter dictionaries.

    Example:
        .. code-block:: python

            from langchain.retrievers.self_query.base import SelfQueryRetriever
            from langchain_weaviate.query_constructors import WeaviateTranslator

            retriever = SelfQueryRetriever.from_llm(
                llm,
                vectorstore,
                document_contents,
                metadata_field_info,
                structured_query_translator=WeaviateTranslator(),
            )
    """

    allowed_operators = [Operator.AND, Operator.OR]
    """Subset of allowed logical operators."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GTE,
        Comparator.LTE,
        Comparator.LT,
        Comparator.GT,
    ]
    """Subset of allowed logical comparators."""

    def visit_operation(self, operation: Operation) -> _Filters:
        self._validate_func(operation.operator)
        operands = [arg.accept(self) for arg in operation.arguments]
        if operation.operator == Operator.AND:
            return Filter.all_of(operands)
        return Filter.any_of(operands)

    def visit_comparison(self, comparison: Comparison) -> _Filters:
        self._validate_func(comparison.comparator)
        prop = Filter.by_property(comparison.attribute)
        value = comparison.value
        if isinstance(value, dict) and value.get("type") == "date":
            # An ISO 8601 date is converted to a timezone-aware datetime, which
            # the weaviate-client v4 filter API expects for date properties.
            value = datetime.strptime(value["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        # https://weaviate.io/developers/weaviate/api/graphql/filters
        comparator_map = {
            Comparator.EQ: prop.equal,
            Comparator.NE: prop.not_equal,
            Comparator.GT: prop.greater_than,
            Comparator.GTE: prop.greater_or_equal,
            Comparator.LT: prop.less_than,
            Comparator.LTE: prop.less_or_equal,
        }
        return comparator_map[comparison.comparator](value)

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs: dict = {}
        else:
            kwargs = {"filters": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
