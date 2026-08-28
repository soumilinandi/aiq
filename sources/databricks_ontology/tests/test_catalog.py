# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Genie metadata normalization and semantic ranking."""

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from databricks_ontology.catalog import CandidateRelevance
from databricks_ontology.catalog import CatalogDocument
from databricks_ontology.catalog import EntityCandidateDecision
from databricks_ontology.catalog import EntityExtraction
from databricks_ontology.catalog import SemanticCatalogRanker
from databricks_ontology.catalog import _DocumentEmbeddingCache
from databricks_ontology.catalog import documents_from_space
from databricks_ontology.errors import DatabricksError


def test_documents_from_space_extracts_semantic_objects(genie_space: dict) -> None:
    """Extract queryable objects and one non-duplicated semantic definition."""

    documents = documents_from_space(genie_space)

    labels = {document.label for document in documents}
    assert {"ColumnAttribute", "GenieSpace", "Metric", "MetricView", "SemanticDefinition", "Table"} <= labels
    assert "Instruction" not in labels
    definitions = [document for document in documents if document.label == "SemanticDefinition"]
    assert len(definitions) == 1
    assert "Use completed fiscal periods" in (definitions[0].definition or "")
    assert any(document.attribute == "customer_id" for document in documents)
    assert all(document.space_id == "space-1" for document in documents)


def test_opaque_ids_do_not_change_when_descriptive_metadata_changes(genie_space: dict) -> None:
    revised = {**genie_space, "description": "Revised domain wording"}

    first = documents_from_space(genie_space)
    second = documents_from_space(revised)

    assert [(document.label, document.attribute, document.id) for document in first] == [
        (document.label, document.attribute, document.id) for document in second
    ]


@pytest.mark.asyncio
async def test_semantic_ranker_computes_entity_coverage_from_embedding_matches() -> None:
    """Rank independent entity-to-object matches rather than matching stop words."""

    extractor = MagicMock()
    extractor.ainvoke = AsyncMock(return_value=EntityExtraction(entities=["customer revenue", "sales region"]))
    relevance_filter = MagicMock()
    relevance_filter.ainvoke = AsyncMock(
        return_value=CandidateRelevance(
            decisions=[
                EntityCandidateDecision(entity_index=0, candidate_refs=["e0c0"]),
                EntityCandidateDecision(entity_index=1, candidate_refs=["e1c0"]),
            ]
        )
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = [extractor, relevance_filter]
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(side_effect=[[1.0, 0.0], [0.0, 1.0]])
    embedder.aembed_documents = AsyncMock(return_value=[[0.95, 0.05], [0.1, 0.9], [-1.0, 0.0]])
    documents = [
        CatalogDocument("Metric", "recognized_revenue", "Revenue", "metric-1", "space-1", "customer revenue"),
        CatalogDocument("ColumnAttribute", "region", "Customers", "column-1", "space-1", "sales region"),
        CatalogDocument("Table", "inventory", "Inventory", "table-1", "space-2", "warehouse stock"),
    ]
    ranker = SemanticCatalogRanker(llm=llm, embedder=embedder)

    response = await ranker.rank(
        "Revenue by customer region",
        documents,
        max_results=10,
        max_distance=None,
        request_id="request-1",
    )

    assert response.request_id == "request-1"
    assert response.coverage == 1
    assert [candidate.id for candidate in response.candidates] == ["metric-1", "column-1"]
    assert response.candidates[0].label == "Metric"
    assert response.candidates[0].attribute == "recognized_revenue"
    assert response.candidates[0].term == "Revenue"
    assert response.candidates[0].scope == "space-1"
    assert response.candidates[0].summary == "customer revenue"
    assert response.candidates[0].score > response.candidates[1].score
    assert response.uncovered_entities == []


@pytest.mark.asyncio
async def test_semantic_ranker_reranks_top_fifteen_candidates_per_entity() -> None:
    """Bound embedding retrieval and preserve the LLM's relevance order."""

    extractor = MagicMock()
    extractor.ainvoke = AsyncMock(return_value=EntityExtraction(entities=["revenue"]))
    relevance_filter = MagicMock()
    relevance_filter.ainvoke = AsyncMock(
        return_value=CandidateRelevance(
            decisions=[EntityCandidateDecision(entity_index=0, candidate_refs=["e0c14", "e0c4"])]
        )
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = [extractor, relevance_filter]
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    embedder.aembed_documents = AsyncMock(return_value=[[1.0, 0.0]] * 16)
    documents = [
        CatalogDocument("Metric", f"metric_{index:02}", "Revenue", f"metric-{index:02}", "space-1", f"metric {index}")
        for index in range(16)
    ]
    ranker = SemanticCatalogRanker(llm=llm, embedder=embedder)

    response = await ranker.rank("revenue", documents, max_results=10, max_distance=None, request_id=None)

    assert [candidate.id for candidate in response.candidates] == ["metric-14", "metric-04"]


@pytest.mark.asyncio
async def test_semantic_ranker_reports_uncovered_entities() -> None:
    """Do not claim coverage when an extracted entity has no embedding match."""

    extractor = MagicMock()
    extractor.ainvoke = AsyncMock(return_value=EntityExtraction(entities=["seismic activity"]))
    relevance_filter = MagicMock()
    relevance_filter.ainvoke = AsyncMock(
        return_value=CandidateRelevance(decisions=[EntityCandidateDecision(entity_index=0, candidate_refs=[])])
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = [extractor, relevance_filter]
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    embedder.aembed_documents = AsyncMock(return_value=[[0.0, 1.0]])
    ranker = SemanticCatalogRanker(llm=llm, embedder=embedder)

    response = await ranker.rank(
        "seismic activity",
        [CatalogDocument("Table", "orders", "Sales", "table-1", "space-1", "customer orders")],
        max_results=10,
        max_distance=None,
        request_id=None,
    )

    assert response.coverage == 0
    assert response.candidates == []
    assert response.uncovered_entities == ["seismic activity"]


@pytest.mark.asyncio
async def test_semantic_ranker_does_not_count_non_queryable_objects_as_coverage() -> None:
    """Return relevant context without treating a Space or example as field-level coverage."""

    extractor = MagicMock()
    extractor.ainvoke = AsyncMock(return_value=EntityExtraction(entities=["customer revenue"]))
    relevance_filter = MagicMock()
    relevance_filter.ainvoke = AsyncMock(
        return_value=CandidateRelevance(decisions=[EntityCandidateDecision(entity_index=0, candidate_refs=["e0c0"])])
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = [extractor, relevance_filter]
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    embedder.aembed_documents = AsyncMock(return_value=[[1.0, 0.0]])
    ranker = SemanticCatalogRanker(llm=llm, embedder=embedder)

    response = await ranker.rank(
        "customer revenue",
        [CatalogDocument("SampleQuestion", "Revenue by customer", "Sales", "sample-1", "space-1", "revenue")],
        max_results=10,
        max_distance=None,
        request_id=None,
    )

    assert response.coverage == 0
    assert [candidate.id for candidate in response.candidates] == ["sample-1"]
    assert response.uncovered_entities == ["customer revenue"]


@pytest.mark.asyncio
async def test_semantic_ranker_reuses_embeddings_without_widening_authorized_scope() -> None:
    """Reuse cached vectors while ranking only documents supplied by the current request."""

    extractor = MagicMock()
    extractor.ainvoke = AsyncMock(return_value=EntityExtraction(entities=["revenue"]))
    relevance_filter = MagicMock()
    relevance_filter.ainvoke = AsyncMock(
        return_value=CandidateRelevance(decisions=[EntityCandidateDecision(entity_index=0, candidate_refs=["e0c0"])])
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = [extractor, relevance_filter]
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    embedder.aembed_documents = AsyncMock(return_value=[[1.0, 0.0], [0.9, 0.1]])
    ranker = SemanticCatalogRanker(llm=llm, embedder=embedder)
    full_scope = [
        CatalogDocument("Metric", "total_revenue", "Sales", "metric-1", "space-1", "total revenue"),
        CatalogDocument("Metric", "net_revenue", "Finance", "metric-2", "space-2", "net revenue"),
    ]

    first = await ranker.rank("revenue", full_scope, max_results=10, max_distance=None, request_id="request-1")
    restricted = await ranker.rank("revenue", full_scope[1:], max_results=10, max_distance=None, request_id="request-2")

    assert [candidate.id for candidate in first.candidates] == ["metric-1"]
    assert [candidate.id for candidate in restricted.candidates] == ["metric-2"]
    embedder.aembed_documents.assert_awaited_once_with(["total revenue", "net revenue"])


@pytest.mark.asyncio
async def test_embedding_cache_does_not_hold_lock_during_remote_inference() -> None:
    """Allow cached reads to proceed while an unrelated embedding request is in flight."""

    inference_started = asyncio.Event()
    release_inference = asyncio.Event()

    async def embed_documents(texts: list[str]) -> list[list[float]]:
        if texts == ["cached"]:
            return [[1.0, 0.0]]
        inference_started.set()
        await release_inference.wait()
        return [[0.0, 1.0]]

    embedder = MagicMock()
    embedder.aembed_documents = AsyncMock(side_effect=embed_documents)
    cache = _DocumentEmbeddingCache(embedder=embedder, max_entries=10)
    cached = CatalogDocument("Metric", "cached", "cached", "cached", "space-1", "cached")
    missing = CatalogDocument("Metric", "missing", "missing", "missing", "space-1", "missing")
    await cache.embed([cached])

    pending = asyncio.create_task(cache.embed([missing]))
    await inference_started.wait()
    assert await asyncio.wait_for(cache.embed([cached]), timeout=0.1) == [[1.0, 0.0]]

    release_inference.set()
    assert await pending == [[0.0, 1.0]]


def test_documents_from_space_bounds_returned_semantic_context(genie_space: dict) -> None:
    """Bound normalized semantic metadata instead of returning a serialized Space dump."""

    definition = json.loads(genie_space["serialized_space"])
    definition["data_sources"]["tables"][0]["description"] = ["x" * 3_000]
    genie_space["serialized_space"] = json.dumps(definition)

    table = next(document for document in documents_from_space(genie_space) if document.label == "Table")

    assert len(table.text) == 2_000


def test_documents_from_space_rejects_malformed_definition() -> None:
    """Reject malformed serialized Space JSON as an invalid provider response."""

    with pytest.raises(DatabricksError):
        documents_from_space({"space_id": "space-1", "serialized_space": "{"})
