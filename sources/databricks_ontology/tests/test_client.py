# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed Databricks SDK adapter."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from databricks.sdk import errors as sdk_errors
from databricks_ontology.catalog import documents_from_space
from databricks_ontology.client import DatabricksClient
from databricks_ontology.errors import DatabricksError
from databricks_ontology.errors import DatabricksErrorCode
from databricks_ontology.models import CatalogCandidate
from databricks_ontology.models import CatalogSearchRequest
from databricks_ontology.models import CatalogSearchResponse
from databricks_ontology.models import TextToSQLRequest


class SdkModel:
    """Minimal generated-SDK-model test double."""

    def __init__(self, payload: dict) -> None:
        """Store one wire-format payload."""

        self._payload = payload

    def as_dict(self) -> dict:
        """Return a copy of the wire-format payload."""

        return dict(self._payload)


class SdkWait:
    """Minimal immediate SDK waiter returned by start/create operations."""

    def __init__(self, payload: dict) -> None:
        """Expose the initial response without invoking the blocking waiter."""

        self.response = SdkModel(payload)


def _ranker() -> MagicMock:
    """Return a semantic-ranker double with one authorized candidate."""

    ranker = MagicMock()

    async def rank(_question: str, documents: list, **_kwargs: object) -> CatalogSearchResponse:
        document = next(
            (value for value in documents if value.label == "ColumnAttribute"),
            next(value for value in documents if value.label in {"Table", "SemanticDefinition", "GenieSpace"}),
        )
        return CatalogSearchResponse(
            request_id="request-1",
            coverage=1,
            candidates=[
                CatalogCandidate(
                    id=document.id,
                    label=document.label,
                    attribute=document.attribute,
                    term=document.term,
                    score=0.95,
                )
            ],
        )

    ranker.rank = AsyncMock(side_effect=rank)
    return ranker


def _space(space_id: str = "space-1", identifier: str = "sales.analytics.orders") -> dict:
    return {
        "space_id": space_id,
        "title": "Analytics",
        "serialized_space": json.dumps({"data_sources": {"tables": [{"identifier": identifier}]}}),
    }


def _object_id(space: dict) -> str:
    return next(document.id for document in documents_from_space(space) if document.label == "Table")


def test_catalog_request_accepts_uniform_optional_controls() -> None:
    """Expose an optional distance control without applying an uncalibrated default."""

    request = CatalogSearchRequest(
        question="customer revenue",
        database_name="benchmark_db",
        max_distance=0.5,
    )

    assert request.database_name == "benchmark_db"
    assert request.max_distance == 0.5
    assert CatalogSearchRequest(question="customer revenue").max_distance is None


@pytest.mark.asyncio
async def test_catalog_search_uses_typed_genie_and_unity_catalog_apis(genie_space: dict) -> None:
    """Discover visible Spaces and enrich them through the typed WorkspaceClient surface."""

    workspace = MagicMock()
    workspace.genie.list_spaces.return_value = SdkModel({"spaces": [{"space_id": "space-1"}]})
    workspace.genie.get_space.return_value = SdkModel(genie_space)
    workspace.tables.get.return_value = SdkModel(
        {
            "full_name": "sales.analytics.customers",
            "comment": "Authorized Unity Catalog metadata",
            "columns": [{"name": "recognized_revenue", "comment": "Booked customer revenue"}],
        }
    )
    factory = MagicMock(return_value=workspace)
    ranker = _ranker()
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=ranker,
        space_ids=("space-1",),
        workspace_factory=factory,
    )

    result = await client.catalog_search(
        CatalogSearchRequest(question="customer revenue"),
        token="test-token",
        trace_headers={"x-request-id": "request-1"},
    )

    assert result.coverage == 1.0
    assert result.candidates[0].id in {document.id for document in documents_from_space(genie_space)}
    assert result.candidates[0].capabilities == ["text_to_sql"]
    assert result.warnings == []
    factory.assert_called_once_with("test-token", {"x-request-id": "request-1"})
    workspace.genie.get_space.assert_called_once_with("space-1", include_serialized_space=True)
    assert workspace.tables.get.call_count == 2
    documents = ranker.rank.await_args.args[1]
    assert any("Authorized Unity Catalog metadata" in document.text for document in documents)


def test_catalog_capabilities_use_one_shared_space_scope() -> None:
    client = DatabricksClient(workspace_url="https://workspace.example", catalog_ranker=_ranker())

    assert client._catalog_capabilities() == ["text_to_sql"]


@pytest.mark.asyncio
async def test_catalog_search_preserves_space_metadata_when_unity_catalog_is_forbidden(genie_space: dict) -> None:
    """Treat per-object Unity Catalog denial as an enrichment warning, not a total failure."""

    workspace = MagicMock()
    workspace.genie.list_spaces.return_value = SdkModel({"spaces": [{"space_id": "space-1"}]})
    workspace.genie.get_space.return_value = SdkModel(genie_space)
    workspace.tables.get.side_effect = sdk_errors.PermissionDenied("not authorized")
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        workspace_factory=lambda _token, _headers: workspace,
    )

    result = await client.catalog_search(
        CatalogSearchRequest(question="customer revenue"),
        token="test-token",
    )

    assert result.candidates
    assert result.warnings == ["Unity Catalog metadata enrichment was unavailable for 2 of 2 objects."]


@pytest.mark.asyncio
async def test_catalog_search_resolves_database_name_to_genie_space(genie_space: dict) -> None:
    """Filter discovered Genie Spaces using their referenced table metadata."""

    workspace = MagicMock()
    workspace.genie.get_space.side_effect = lambda space_id, **_kwargs: SdkModel(
        _space_with_table(genie_space, space_id, f"catalog.{space_id}.orders")
    )
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        space_ids=("space-1", "space-2"),
        enrich_unity_catalog=False,
        workspace_factory=lambda _token, _headers: workspace,
    )

    await client.catalog_search(
        CatalogSearchRequest(question="customer revenue", database_name="SPACE-2"),
        token="test-token",
    )

    workspace.genie.list_spaces.assert_not_called()
    assert workspace.genie.get_space.call_count == 2


@pytest.mark.parametrize(
    ("identifier", "selector", "matches"),
    [
        ("main.sales.orders", "sales", True),
        ("main.sales.orders", "main.sales", True),
        ("main.sales.orders", "MAIN.SALES.ORDERS", True),
        ("main.sales_archive.orders", "sales", False),
        ("main.sales_orders", "sales", False),
        ("`main`.`Sales.Analytics`.`orders`", "`main`.`sales.analytics`", True),
        ("`main`.`sales``ops`.`orders`", "`sales``ops`", True),
        ("main.sales.orders", "orders", False),
        ("main.sales.orders", "main..sales", False),
    ],
)
def test_database_matching_uses_exact_databricks_identifier_components(
    identifier: str,
    selector: str,
    matches: bool,
) -> None:
    """Match catalogs and schemas exactly without invented prefix normalization."""

    assert DatabricksClient._matches_database(identifier, selector) is matches


@pytest.mark.asyncio
async def test_catalog_search_surfaces_space_discovery_truncation() -> None:
    """Tell callers when the configured Space bound leaves authorized Spaces unsearched."""

    workspace = MagicMock()
    workspace.genie.list_spaces.return_value = SdkModel(
        {"spaces": [{"space_id": "space-1"}], "next_page_token": "more-spaces"}
    )
    workspace.genie.get_space.return_value = SdkModel(
        _space_with_table({"serialized_space": "{}"}, "space-1", "main.sales.orders")
    )
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        max_spaces=1,
        enrich_unity_catalog=False,
        workspace_factory=lambda _token, _headers: workspace,
    )

    result = await client.catalog_search(CatalogSearchRequest(question="revenue"), token="test-token")

    assert result.truncated is True
    assert any("Space discovery" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_catalog_search_surfaces_unity_catalog_enrichment_truncation(genie_space: dict) -> None:
    """Tell callers when only part of the referenced table metadata was enriched."""

    workspace = MagicMock()
    workspace.genie.list_spaces.return_value = SdkModel({"spaces": [{"space_id": "space-1"}]})
    workspace.genie.get_space.return_value = SdkModel(genie_space)
    workspace.tables.get.return_value = SdkModel({"full_name": "sales.analytics.customers"})
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        max_catalog_tables=1,
        workspace_factory=lambda _token, _headers: workspace,
    )

    result = await client.catalog_search(CatalogSearchRequest(question="revenue"), token="test-token")

    assert result.truncated is True
    assert workspace.tables.get.call_count == 1
    assert any("metadata enrichment stopped" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_text_to_sql_polls_genie_and_returns_bounded_rows() -> None:
    """Poll Genie asynchronously and normalize its bounded query attachment."""

    workspace = MagicMock()
    semantic_space = _space(identifier="sales.analytics.orders")
    workspace.genie.get_space.return_value = SdkModel(semantic_space)
    workspace.genie.start_conversation.return_value = SdkWait(
        {"conversation_id": "conversation-1", "message_id": "message-1"}
    )
    workspace.genie.get_message.side_effect = [
        SdkModel(
            {
                "conversation_id": "conversation-1",
                "message_id": "message-1",
                "status": "EXECUTING_QUERY",
            }
        ),
        SdkModel(
            {
                "conversation_id": "conversation-1",
                "message_id": "message-1",
                "status": "COMPLETED",
                "attachments": [
                    {"attachment_id": "text-1", "text": {"content": "Revenue by region."}},
                    {
                        "attachment_id": "query-1",
                        "query": {
                            "query": (
                                "SELECT region, SUM(revenue) AS total_revenue "
                                "FROM sales.analytics.orders GROUP BY region"
                            )
                        },
                    },
                ],
            }
        ),
    ]
    workspace.genie.get_message_attachment_query_result.return_value = SdkModel(
        {
            "statement_response": {
                "manifest": {
                    "schema": {
                        "columns": [
                            {"name": "region", "type_name": "STRING"},
                            {"name": "total_revenue", "type_name": "DECIMAL"},
                        ]
                    }
                },
                "result": {"data_array": [["east", "100"], ["west", "75"]]},
            }
        }
    )
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        space_ids=("space-1",),
        enrich_unity_catalog=False,
        default_max_rows=1,
        genie_poll_interval_seconds=0,
        workspace_factory=lambda _token, _headers: workspace,
    )

    result = await client.text_to_sql(
        TextToSQLRequest(
            question="Show revenue by region",
            object_ids=[_object_id(semantic_space)],
            max_rows=10,
        ),
        token="test-token",
        trace_headers={"x-request-id": "request-1"},
    )

    assert result.request_id == "request-1"
    assert result.space_id == "space-1"
    assert result.outcome == "query_evidence"
    assert result.status == "success"
    assert result.total_ms is not None
    payload = result.model_dump()
    assert "conversation_id" not in payload
    assert "message_id" not in payload
    assert "timings" not in payload
    assert result.sql is not None and result.sql.startswith("SELECT region")
    assert result.rows == [{"region": "east", "total_revenue": "100"}]
    assert result.truncated is True
    assert result.objects_used == ["sales.analytics.orders"]
    workspace.genie.start_conversation.assert_called_once_with(
        "space-1",
        "Show revenue by region",
        enable_visualization=False,
    )
    assert workspace.genie.get_message.call_count == 2
    workspace.genie.get_space.assert_called_once_with("space-1", include_serialized_space=True)
    workspace.genie.get_message_attachment_query_result.assert_called_once_with(
        "space-1",
        "conversation-1",
        "message-1",
        "query-1",
    )


@pytest.mark.asyncio
async def test_text_to_sql_preserves_clarification_without_sql() -> None:
    """Return Genie's textual clarification without treating it as executed evidence."""

    workspace = MagicMock()
    semantic_space = _space()
    workspace.genie.get_space.return_value = SdkModel(semantic_space)
    workspace.genie.start_conversation.return_value = SdkWait({"conversation_id": "c1", "message_id": "m1"})
    workspace.genie.get_message.return_value = SdkModel(
        {
            "conversation_id": "c1",
            "message_id": "m1",
            "status": "COMPLETED",
            "attachments": [{"attachment_id": "a1", "text": {"content": "Which fiscal period?"}}],
        }
    )
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        space_ids=("space-1",),
        enrich_unity_catalog=False,
        workspace_factory=lambda _token, _headers: workspace,
    )

    result = await client.text_to_sql(
        TextToSQLRequest(question="Show revenue", object_ids=[_object_id(semantic_space)]), token="test-token"
    )

    assert result.sql is None
    assert result.outcome == "clarification"
    assert result.rows == []
    assert result.response == "Which fiscal period?"
    assert result.warnings


@pytest.mark.asyncio
async def test_text_to_sql_resolves_database_name_to_genie_space() -> None:
    """Filter provider metadata instead of using an unscoped default Space."""

    workspace = MagicMock()
    base_space = {
        "title": "Analytics",
        "serialized_space": json.dumps({"data_sources": {"tables": []}}),
    }
    workspace.genie.get_space.side_effect = lambda space_id, **_kwargs: SdkModel(
        _space_with_table(base_space, space_id, f"catalog.{space_id}.orders")
    )
    workspace.genie.start_conversation.return_value = SdkWait({"conversation_id": "c1", "message_id": "m1"})
    workspace.genie.get_message.return_value = SdkModel(
        {
            "conversation_id": "c1",
            "message_id": "m1",
            "status": "COMPLETED",
            "attachments": [{"attachment_id": "a1", "text": {"content": "Scoped response."}}],
        }
    )
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        space_ids=("space-1", "space-2"),
        enrich_unity_catalog=False,
        workspace_factory=lambda _token, _headers: workspace,
    )
    selected_space = _space_with_table(base_space, "space-2", "catalog.space-2.orders")

    result = await client.text_to_sql(
        TextToSQLRequest(
            question="Show revenue",
            database_name="space-2",
            object_ids=[_object_id(selected_space)],
        ),
        token="test-token",
    )

    assert result.space_id == "space-2"
    workspace.genie.start_conversation.assert_called_once()
    assert workspace.genie.start_conversation.call_args.args[0] == "space-2"


@pytest.mark.asyncio
async def test_text_to_sql_repairs_failed_message_in_same_conversation() -> None:
    """Use one bounded follow-up in the original conversation after a failed attempt."""

    workspace = MagicMock()
    semantic_space = _space()
    workspace.genie.get_space.return_value = SdkModel(semantic_space)
    workspace.genie.start_conversation.return_value = SdkWait(
        {"conversation_id": "conversation-1", "message_id": "message-1"}
    )
    workspace.genie.create_message.return_value = SdkWait(
        {"conversation_id": "conversation-1", "message_id": "message-2"}
    )
    workspace.genie.get_message.side_effect = [
        SdkModel(
            {
                "conversation_id": "conversation-1",
                "message_id": "message-1",
                "status": "FAILED",
                "error": {"type": "SQL_EXECUTION_ERROR", "error": "Column `amount` cannot be resolved."},
                "attachments": [
                    {
                        "attachment_id": "query-1",
                        "query": {"query": "SELECT amount FROM sales.analytics.orders"},
                    }
                ],
            }
        ),
        SdkModel(
            {
                "conversation_id": "conversation-1",
                "message_id": "message-2",
                "status": "COMPLETED",
                "attachments": [
                    {
                        "attachment_id": "query-2",
                        "query": {"query": "SELECT * FROM sales.analytics.orders"},
                    }
                ],
            }
        ),
    ]
    workspace.genie.get_message_attachment_query_result.return_value = SdkModel(
        {"statement_response": {"manifest": {"schema": {"columns": []}}, "result": {"data_array": []}}}
    )
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        space_ids=("space-1",),
        enrich_unity_catalog=False,
        genie_poll_interval_seconds=0,
        workspace_factory=lambda _token, _headers: workspace,
    )

    result = await client.text_to_sql(
        TextToSQLRequest(question="Show orders", object_ids=[_object_id(semantic_space)]), token="test-token"
    )

    assert result.outcome == "query_evidence"
    assert result.conversation_id == "conversation-1"
    assert result.message_id == "message-2"
    workspace.genie.create_message.assert_called_once()
    assert workspace.genie.create_message.call_args.args[:2] == ("space-1", "conversation-1")
    repair_prompt = workspace.genie.create_message.call_args.args[2]
    assert "SQL_EXECUTION_ERROR" in repair_prompt
    assert "Column `amount` cannot be resolved." in repair_prompt
    assert "SELECT amount FROM sales.analytics.orders" in repair_prompt


@pytest.mark.asyncio
async def test_text_to_sql_requests_can_poll_concurrently() -> None:
    """Allow one pending Genie message to yield while another request completes."""

    workspace = MagicMock()
    semantic_space = _space()
    workspace.genie.get_space.return_value = SdkModel(semantic_space)
    workspace.genie.start_conversation.side_effect = lambda _space, question, **_kwargs: SdkWait(
        {
            "conversation_id": f"conversation-{question}",
            "message_id": f"message-{question}",
        }
    )
    polls = {"slow": 0}

    def get_message(space_id: str, conversation_id: str, message_id: str) -> SdkModel:
        assert space_id == "space-1"
        question = conversation_id.removeprefix("conversation-")
        if question == "slow" and polls["slow"] == 0:
            polls["slow"] += 1
            return SdkModel({"conversation_id": conversation_id, "message_id": message_id, "status": "ASKING_AI"})
        return SdkModel(
            {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "status": "COMPLETED",
                "attachments": [{"text": {"content": f"Clarify {question}"}}],
            }
        )

    workspace.genie.get_message.side_effect = get_message
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        space_ids=("space-1",),
        enrich_unity_catalog=False,
        genie_poll_interval_seconds=0.01,
        workspace_factory=lambda _token, _headers: workspace,
    )

    object_ids = [_object_id(semantic_space)]
    slow = asyncio.create_task(
        client.text_to_sql(TextToSQLRequest(question="slow", object_ids=object_ids), token="test-token")
    )
    await asyncio.sleep(0)
    fast = asyncio.create_task(
        client.text_to_sql(TextToSQLRequest(question="fast", object_ids=object_ids), token="test-token")
    )
    fast_result = await fast
    assert not slow.done()
    slow_result = await slow

    assert fast_result.response == "Clarify fast"
    assert slow_result.response == "Clarify slow"


@pytest.mark.asyncio
async def test_text_to_sql_polling_honors_genie_deadline() -> None:
    """Stop polling pending messages at the configured end-to-end deadline."""

    workspace = MagicMock()
    semantic_space = _space()
    workspace.genie.get_space.return_value = SdkModel(semantic_space)
    workspace.genie.start_conversation.return_value = SdkWait(
        {"conversation_id": "conversation-1", "message_id": "message-1"}
    )
    workspace.genie.get_message.return_value = SdkModel(
        {"conversation_id": "conversation-1", "message_id": "message-1", "status": "ASKING_AI"}
    )
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        space_ids=("space-1",),
        enrich_unity_catalog=False,
        genie_timeout_seconds=0.01,
        genie_poll_interval_seconds=1,
        workspace_factory=lambda _token, _headers: workspace,
    )

    with pytest.raises(DatabricksError) as caught:
        await client.text_to_sql(
            TextToSQLRequest(question="Show revenue", object_ids=[_object_id(semantic_space)]), token="test-token"
        )

    assert caught.value.code == DatabricksErrorCode.TIMEOUT


def test_text_to_sql_request_rejects_removed_execute_option() -> None:
    """Do not imply that AI-Q can prevent Genie from executing generated SQL."""

    with pytest.raises(ValueError, match="execute"):
        TextToSQLRequest(question="Show revenue", object_ids=["dbobj_candidate"], execute=False)


@pytest.mark.asyncio
async def test_text_to_sql_rejects_stale_and_mixed_but_accepts_selected_space_ids() -> None:
    analytical = _space("space-1", "main.analytics.deployments")
    other = _space("space-2", "main.finance.contracts")
    risk = _space("space-3", "main.risk.deployment_events")
    workspace = MagicMock()
    spaces = {space["space_id"]: space for space in (analytical, other, risk)}
    workspace.genie.get_space.side_effect = lambda space_id, **_kwargs: SdkModel(spaces[space_id])
    client = DatabricksClient(
        workspace_url="https://workspace.example",
        catalog_ranker=_ranker(),
        space_ids=("space-1", "space-2", "space-3"),
        enrich_unity_catalog=False,
        workspace_factory=lambda _token, _headers: workspace,
    )

    with pytest.raises(DatabricksError, match="unavailable in the authorized scope"):
        await client._resolve_space_id(
            workspace,
            TextToSQLRequest(question="Show deployments", object_ids=["dbobj_stale"]),
            request_id="request-stale",
        )
    with pytest.raises(DatabricksError, match="one Genie Space"):
        await client._resolve_space_id(
            workspace,
            TextToSQLRequest(
                question="Compare domains",
                object_ids=[_object_id(analytical), _object_id(other)],
            ),
            request_id="request-mixed",
        )
    resolved = await client._resolve_space_id(
        workspace,
        TextToSQLRequest(question="Analyze risk", object_ids=[_object_id(risk)]),
        request_id="request-risk",
    )

    assert resolved == "space-3"


def _space_with_table(space: dict, space_id: str, identifier: str) -> dict:
    """Return a Genie definition referencing one database-prefixed table."""

    definition = json.loads(str(space["serialized_space"]))
    definition.setdefault("data_sources", {})["tables"] = [{"identifier": identifier}]
    definition["data_sources"]["metric_views"] = []
    return {**space, "space_id": space_id, "serialized_space": json.dumps(definition)}
