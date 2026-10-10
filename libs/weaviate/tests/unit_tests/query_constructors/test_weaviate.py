from datetime import datetime, timezone

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from weaviate.classes.query import Filter

from langchain_weaviate.query_constructors import WeaviateTranslator

DEFAULT_TRANSLATOR = WeaviateTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.EQ, attribute="foo", value="1")
    expected = Filter.by_property("foo").equal("1")
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_integer() -> None:
    comp = Comparison(comparator=Comparator.GTE, attribute="foo", value=1)
    expected = Filter.by_property("foo").greater_or_equal(1)
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_number() -> None:
    comp = Comparison(comparator=Comparator.GT, attribute="foo", value=1.4)
    expected = Filter.by_property("foo").greater_than(1.4)
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_boolean() -> None:
    comp = Comparison(comparator=Comparator.NE, attribute="foo", value=False)
    expected = Filter.by_property("foo").not_equal(False)
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_datetime() -> None:
    comp = Comparison(
        comparator=Comparator.LTE,
        attribute="foo",
        value={"type": "date", "date": "2023-09-13"},
    )
    expected = Filter.by_property("foo").less_or_equal(
        datetime(2023, 9, 13, tzinfo=timezone.utc)
    )
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_date() -> None:
    comp = Comparison(
        comparator=Comparator.LT,
        attribute="foo",
        value={"type": "date", "date": "2023-09-13"},
    )
    expected = Filter.by_property("foo").less_than(
        datetime(2023, 9, 13, tzinfo=timezone.utc)
    )
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation_and() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value="hello"),
            Comparison(
                comparator=Comparator.GTE,
                attribute="bar",
                value={"type": "date", "date": "2023-09-13"},
            ),
            Comparison(comparator=Comparator.LTE, attribute="abc", value=1.4),
        ],
    )
    expected = Filter.all_of(
        [
            Filter.by_property("foo").equal("hello"),
            Filter.by_property("bar").greater_or_equal(
                datetime(2023, 9, 13, tzinfo=timezone.utc)
            ),
            Filter.by_property("abc").less_or_equal(1.4),
        ]
    )
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    # `_FilterAnd` does not implement ``__eq__``, so compare structurally.
    assert type(actual) is type(expected)
    assert actual.filters == expected.filters


def test_visit_operation_or() -> None:
    op = Operation(
        operator=Operator.OR,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value="hello"),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="world"),
        ],
    )
    expected = Filter.any_of(
        [
            Filter.by_property("foo").equal("hello"),
            Filter.by_property("bar").equal("world"),
        ]
    )
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert type(actual) is type(expected)
    assert actual.filters == expected.filters


def test_visit_structured_query_no_filter() -> None:
    query = "What is the capital of France?"
    structured_query = StructuredQuery(query=query, filter=None)
    expected = (query, {})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_comparison() -> None:
    query = "What is the capital of France?"
    comp = Comparison(comparator=Comparator.EQ, attribute="foo", value="1")
    structured_query = StructuredQuery(query=query, filter=comp)
    expected = (query, {"filters": Filter.by_property("foo").equal("1")})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_operation() -> None:
    query = "What is the capital of France?"
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    structured_query = StructuredQuery(query=query, filter=op)
    expected_filter = Filter.all_of(
        [
            Filter.by_property("foo").equal(2),
            Filter.by_property("bar").equal("baz"),
        ]
    )
    actual_query, actual_kwargs = DEFAULT_TRANSLATOR.visit_structured_query(
        structured_query
    )
    assert actual_query == query
    assert type(actual_kwargs["filters"]) is type(expected_filter)
    assert actual_kwargs["filters"].filters == expected_filter.filters
