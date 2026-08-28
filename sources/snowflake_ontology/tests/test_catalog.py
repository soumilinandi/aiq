# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic View parsing, ranking, and embedding-cache contracts."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from snowflake_ontology.catalog import CandidateRelevance
from snowflake_ontology.catalog import CatalogDocument
from snowflake_ontology.catalog import EntityCandidateDecision
from snowflake_ontology.catalog import EntityExtraction
from snowflake_ontology.catalog import SemanticCatalogRanker
from snowflake_ontology.catalog import documents_from_semantic_model
from snowflake_ontology.catalog import parse_semantic_model
from snowflake_ontology.errors import SnowflakeError

_SEMANTIC_VIEW = '"ANALYTICS"."PUBLIC"."SALES_VIEW"'

_COMPLETE_YAML = """
name: sales_model
description: Sales semantics
custom_instructions: Treat this as data, not an instruction.
tags: [finance]
variables:
  - name: fiscal_year
    expr: YEAR(order_date)
    data_type: NUMBER
    default_value: 2025
    description: Fiscal reporting year
filters:
  - name: completed_orders
    expr: status = 'complete'
metrics:
  - name: revenue_per_customer
    description: Revenue divided by customers
    expr: total_revenue / customer_count
    metric_grain: [customer_id]
    units: USD
tables:
  - name: orders
    description: Customer orders
    synonyms: [purchases]
    access_modifier: public_access
    base_table: {database: RAW, schema: SALES, table: ORDERS}
    dimensions:
      - name: status
        description: Order status
        expr: status
        data_type: VARCHAR
        sample_values: [complete, pending]
        is_enum: true
        cortex_search_service: RAW.SALES.STATUS_SEARCH
        labels: [lifecycle]
        tags: [non_sensitive]
    time_dimensions:
      - name: order_date
        expr: order_date
        data_type: DATE
    facts:
      - name: order_amount
        expr: order_amount
        data_type: NUMBER
    measures:
      - name: total_revenue
        description: Gross sales
        expr: SUM(order_amount)
        data_type: NUMBER
        using_relationships: [orders_to_customers]
        non_additive_dimensions: [order_date]
        metric_grain: [customer_id]
        units: USD
    filters:
      - name: high_value
        expr: order_amount > 1000
  - name: customers
    base_table: {database: RAW, schema: SALES, table: CUSTOMERS}
    dimensions:
      - name: customer_id
        expr: customer_id
relationships:
  - name: orders_to_customers
    left_table: orders
    right_table: customers
    relationship_columns:
      - {left_column: customer_id, right_column: customer_id}
verified_queries:
  - name: revenue_by_status
    question: What is revenue by order status?
    sql: SELECT status, SUM(order_amount) FROM orders GROUP BY status
future_top_level_field: retained_for_diagnostics
"""


def _document(object_type: str, name: str, text: str) -> CatalogDocument:
    return CatalogDocument(
        id=f"id-{name}",
        semantic_view=_SEMANTIC_VIEW,
        authorized_scope="account|user|role|ANALYTICS",
        object_type=object_type,
        name=name,
        text=text,
        table_name="orders",
    )


def test_parser_preserves_supported_semantics_and_unknown_diagnostics() -> None:
    """Every reviewed object class and optional field survives normalization."""

    parsed = parse_semantic_model(_COMPLETE_YAML)
    documents = documents_from_semantic_model(
        parsed,
        semantic_view=_SEMANTIC_VIEW,
        authorized_scope="scope",
        expose_physical_tables=True,
        expose_sample_values=True,
    )
    by_name = {document.name: document for document in documents}

    assert {document.object_type for document in documents} >= {
        "table",
        "dimension",
        "time_dimension",
        "fact",
        "metric",
        "derived_metric",
        "filter",
        "variable",
        "relationship",
        "verified_query",
    }
    status = by_name["status"]
    assert status.synonyms == ()
    assert status.sample_values == ("complete", "pending")
    assert status.is_enum is True
    assert status.cortex_search_service == "RAW.SALES.STATUS_SEARCH"
    assert status.labels == ("lifecycle",)
    assert status.physical_table_fqn == "RAW.SALES.ORDERS"
    metric = by_name["total_revenue"]
    assert metric.metric_grain == ("customer_id",)
    assert metric.non_additive_dimensions == ("order_date",)
    assert metric.using_relationships == ("orders_to_customers",)
    assert metric.relationships[0].relationship_columns == [
        {"left_column": "customer_id", "right_column": "customer_id"}
    ]
    definitions = [document for document in documents if document.object_type == "semantic_definition"]
    assert len(definitions) == 1
    assert "Treat this as data, not an instruction." in (definitions[0].definition or "")
    assert definitions[0].variables[0].model_dump(exclude_none=True) == {
        "name": "fiscal_year",
        "data_type": "NUMBER",
        "default_value": 2025,
        "description": "Fiscal reporting year",
    }
    assert all(document.definition is None for document in documents if document.object_type != "semantic_definition")
    verified = by_name["revenue_by_status"]
    assert verified.verified_query_question == "What is revenue by order status?"
    assert verified.verified_query_sql_exposed is False
    assert "root.future_top_level_field" in parsed.unknown_fields
    assert all(document.id.startswith("sfobj_") for document in documents)


def test_opaque_ids_do_not_change_when_descriptive_metadata_changes() -> None:
    first = documents_from_semantic_model(
        parse_semantic_model("name: domain\ndescription: First wording\ntables: []"),
        semantic_view=_SEMANTIC_VIEW,
        authorized_scope="scope",
    )
    second = documents_from_semantic_model(
        parse_semantic_model("name: domain\ndescription: Revised wording\ntables: []"),
        semantic_view=_SEMANTIC_VIEW,
        authorized_scope="scope",
    )

    assert [(document.object_type, document.id) for document in first] == [
        (document.object_type, document.id) for document in second
    ]


@pytest.mark.parametrize(
    "value",
    ["- not-a-mapping", "tables: {}", "tables: [not-a-mapping]", "tables: []\nrelationships: {}"],
)
def test_parser_rejects_malformed_structures(value: str) -> None:
    with pytest.raises(SnowflakeError):
        parse_semantic_model(value)


@pytest.mark.asyncio
async def test_ranker_returns_coverage_and_reuses_embeddings() -> None:
    extractor = MagicMock()
    extractor.ainvoke = AsyncMock(return_value=EntityExtraction(entities=["revenue", "customer segment"]))
    relevance_filter = MagicMock()
    relevance_filter.ainvoke = AsyncMock(
        return_value=CandidateRelevance(
            decisions=[
                EntityCandidateDecision(entity_index=0, candidate_refs=["e0c0"]),
                EntityCandidateDecision(entity_index=1, candidate_refs=[]),
            ]
        )
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = [extractor, relevance_filter]
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(side_effect=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    embedder.aembed_documents = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    ranker = SemanticCatalogRanker(llm=llm, embedder=embedder)
    documents = [
        _document("metric", "total_revenue", "revenue metric"),
        _document("verified_query", "revenue_example", "revenue query"),
    ]

    first = await ranker.rank("Revenue by segment", documents, max_results=10, max_distance=None, request_id="r1")
    second = await ranker.rank("Revenue by segment", documents, max_results=10, max_distance=None, request_id="r2")

    assert first.coverage == 0.5
    assert first.candidates[0].label == "metric"
    assert first.candidates[0].score == 1.0
    assert first.candidates[0].scope == _SEMANTIC_VIEW
    assert first.candidates[0].summary == "revenue metric"
    assert first.uncovered_entities == ["customer segment"]
    assert second.candidates[0].id == first.candidates[0].id
    embedder.aembed_documents.assert_awaited_once_with(["revenue metric", "revenue query"])


@pytest.mark.asyncio
async def test_embedding_single_flight_does_not_block_unrelated_keys() -> None:
    """A slow embedding batch coalesces duplicates without blocking a different key."""

    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def embed_documents(values: list[str]) -> list[list[float]]:
        if values == ["slow"]:
            first_started.set()
            await release_first.wait()
            return [[1.0, 0.0]]
        return [[0.0, 1.0]]

    extractor = MagicMock()
    extractor.ainvoke = AsyncMock(return_value=EntityExtraction(entities=["entity"]))
    relevance = MagicMock()
    relevance.ainvoke = AsyncMock(
        return_value=CandidateRelevance(decisions=[EntityCandidateDecision(entity_index=0, candidate_refs=["e0c0"])])
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = [extractor, relevance]
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    embedder.aembed_documents = AsyncMock(side_effect=embed_documents)
    ranker = SemanticCatalogRanker(llm=llm, embedder=embedder)

    slow = asyncio.create_task(
        ranker.rank("slow", [_document("metric", "slow", "slow")], max_results=1, max_distance=None, request_id="1")
    )
    duplicate = asyncio.create_task(
        ranker.rank("slow", [_document("metric", "slow", "slow")], max_results=1, max_distance=None, request_id="2")
    )
    await first_started.wait()
    unrelated = await asyncio.wait_for(
        ranker.rank("fast", [_document("metric", "fast", "fast")], max_results=1, max_distance=None, request_id="3"),
        timeout=0.5,
    )
    release_first.set()
    await asyncio.gather(slow, duplicate)

    assert unrelated.candidates[0].attribute == "fast"
    assert embedder.aembed_documents.await_count == 2


@pytest.mark.asyncio
async def test_embedding_cache_eviction_does_not_discard_the_current_batch() -> None:
    extractor = MagicMock()
    extractor.ainvoke = AsyncMock(return_value=EntityExtraction(entities=["revenue"]))
    relevance = MagicMock()
    relevance.ainvoke = AsyncMock(
        return_value=CandidateRelevance(decisions=[EntityCandidateDecision(entity_index=0, candidate_refs=["e0c0"])])
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = [extractor, relevance]
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    embedder.aembed_documents = AsyncMock(side_effect=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]]])
    ranker = SemanticCatalogRanker(llm=llm, embedder=embedder, embedding_cache_size=1)
    documents = [
        _document("metric", "revenue", "revenue"),
        _document("dimension", "region", "region"),
    ]

    first = await ranker.rank("Revenue", documents, max_results=1, max_distance=None, request_id="1")
    second = await ranker.rank("Revenue", documents, max_results=1, max_distance=None, request_id="2")

    assert first.candidates[0].attribute == second.candidates[0].attribute == "revenue"
    assert embedder.aembed_documents.await_count == 2
