# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snowflake scope, Agent transport, trajectory, and result-safety contracts."""

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests
from snowflake.connector.errors import DatabaseError
from snowflake.connector.errors import OperationalError
from snowflake_ontology.catalog import CatalogDocument
from snowflake_ontology.client import SemanticViewRef
from snowflake_ontology.client import SnowflakeClient
from snowflake_ontology.errors import SnowflakeError
from snowflake_ontology.errors import SnowflakeErrorCode
from snowflake_ontology.models import SemanticVariableContext
from snowflake_ontology.models import TextToSQLRequest

_VIEW = '"DB"."SCHEMA"."SALES_VIEW"'


def _client(**overrides: object) -> SnowflakeClient:
    values = {
        "account": "account",
        "user": "user",
        "role": "AIQ_READ_ONLY",
        "warehouse": "warehouse",
        "semantic_views": (_VIEW, '"OTHER"."SCHEMA"."OTHER_VIEW"'),
        "max_semantic_views": 10,
        "catalog_ranker": MagicMock(),
    }
    values.update(overrides)
    return SnowflakeClient(**values)  # type: ignore[arg-type]


def _document(
    name: str,
    *,
    semantic_view: str = _VIEW,
    synonyms: tuple[str, ...] = (),
) -> CatalogDocument:
    return CatalogDocument(
        id=f"id-{name}",
        semantic_view=semantic_view,
        authorized_scope="scope",
        object_type="metric",
        name=name,
        text=f"metric {name}",
        description=f"Description for {name}",
        synonyms=synonyms,
        expression=f"SUM({name})",
        table_name="orders",
    )


def _request(question: str = "Revenue", **values: object) -> TextToSQLRequest:
    return TextToSQLRequest(question=question, object_ids=["id-gross_revenue"], **values)


def _agent_payload(*, failed_first: bool = False, rows: list[list[object]] | None = None) -> dict:
    content: list[dict] = [
        {
            "type": "tool_use",
            "tool_use": {
                "type": "cortex_analyst_text_to_sql",
                "name": "analyst_tool_1",
                "input": {"question": "revenue"},
            },
        }
    ]
    if failed_first:
        content.extend(
            [
                {
                    "type": "tool_use",
                    "tool_use": {
                        "type": "system_execute_sql",
                        "name": "system_execute_sql",
                        "input": {"sql": "SELECT missing_column FROM DB.SCHEMA.ORDERS"},
                    },
                },
                {
                    "type": "tool_result",
                    "tool_result": {
                        "type": "system_execute_sql",
                        "name": "system_execute_sql",
                        "status": "error",
                        "content": [
                            {
                                "type": "json",
                                "json": {
                                    "sql": "SELECT missing_column FROM DB.SCHEMA.ORDERS",
                                    "query_id": "failed-query",
                                    "error": {"code": "002003", "message": "Invalid identifier"},
                                },
                            }
                        ],
                    },
                },
            ]
        )
    sql = "SELECT A AS VALUE, B AS VALUE FROM DB.SCHEMA.ORDERS"
    content.extend(
        [
            {
                "type": "tool_use",
                "tool_use": {
                    "type": "system_execute_sql",
                    "name": "system_execute_sql",
                    "input": {"sql": sql},
                },
            },
            {
                "type": "tool_result",
                "tool_result": {
                    "type": "system_execute_sql",
                    "name": "system_execute_sql",
                    "status": "success",
                    "content": [
                        {
                            "type": "json",
                            "json": {
                                "query_id": "successful-query",
                                "sql": sql,
                                "verified_query_used": True,
                                "verified_query_name": "revenue_verified",
                                "verified_query_confidence": 0.97,
                                "question_category": "analytics",
                                "model_name": "cortex-analyst-model",
                                "semantic_model_selection": {"semantic_view": _VIEW, "confidence": 0.96},
                                "search_metadata": {"service": "literal_search", "matches": 2},
                                "result_set": {
                                    "data": rows if rows is not None else [["10", "20"]],
                                    "resultSetMetaData": {
                                        "numRows": len(rows) if rows is not None else 1,
                                        "rowType": [
                                            {"name": "VALUE", "type": "FIXED"},
                                            {"name": "VALUE", "type": "FIXED"},
                                        ],
                                    },
                                },
                            },
                        }
                    ],
                },
            },
            {"type": "text", "text": "Revenue was returned."},
        ]
    )
    return {"status": "completed", "content": content, "warnings": [{"message": "Agent warning."}]}


def _failed_agent_payload() -> dict:
    payload = _agent_payload(failed_first=True)
    return {"status": "completed", "content": payload["content"][:3]}


def _query_payload(sql: str, *, columns: list[str] | None = None, rows: list[list[object]] | None = None) -> dict:
    columns = columns or ["DEPLOYMENT_CODE", "CONTRACT_VALUE"]
    rows = rows or [["D-001", "100"]]
    payload = _agent_payload(rows=rows)
    payload["content"][-3]["tool_use"]["input"]["sql"] = sql
    result = payload["content"][-2]["tool_result"]["content"][0]["json"]
    result["sql"] = sql
    result["result_set"]["resultSetMetaData"]["rowType"] = [{"name": name, "type": "TEXT"} for name in columns]
    return payload


_ANCHOR_QUESTION = (
    "As of 2026-08-12, which non-live deployments due within the next 180 days should be ranked by contract value?"
)
_VALID_ANCHOR_SQL = (
    "SELECT deployment_code, contract_value FROM deployments "
    "WHERE status <> 'live' AND TO_DATE(target_go_live_date) > CAST('2026-08-12' AS DATE) "
    "AND TO_DATE(target_go_live_date) <= CAST('2027-02-08' AS DATE) ORDER BY contract_value DESC"
)


def test_execution_grounding_rehydrates_governed_view_definition() -> None:
    """Send view instructions and typed variables alongside selected catalog objects."""

    selected = _document("gross_revenue")
    definition = CatalogDocument(
        id="id-definition",
        semantic_view=_VIEW,
        authorized_scope="scope",
        object_type="semantic_definition",
        name="sales_model",
        text="Use the governed fiscal year.",
        definition="Use the governed fiscal year.",
        variables=(SemanticVariableContext(name="fiscal_year", data_type="NUMBER", default_value=2025),),
    )
    unrelated = CatalogDocument(
        id="id-other-definition",
        semantic_view='"OTHER"."SCHEMA"."OTHER_VIEW"',
        authorized_scope="scope",
        object_type="semantic_definition",
        name="other_model",
        text="Other semantics",
    )

    grounded = SnowflakeClient._execution_grounding_documents([selected, definition, unrelated], [selected])
    payload = json.loads(SnowflakeClient._grounding_json(grounded))

    assert [document.id for document in grounded] == ["id-definition", selected.id]
    definition_payload = next(item for item in payload["candidates"] if item["id"] == "id-definition")
    assert definition_payload["variables"] == [{"name": "fiscal_year", "data_type": "NUMBER", "default_value": 2025}]


def test_catalog_capabilities_use_one_shared_view_scope() -> None:
    assert _client()._catalog_capabilities() == ["text_to_sql"]


@pytest.mark.parametrize(
    ("identifier", "database", "matches"),
    [
        ("DB.SCHEMA.VIEW", "db", True),
        ("DB.SCHEMA.VIEW", "schema", True),
        ('"Mixed.DB"."Schema"."View"', '"Mixed.DB"', True),
        ('"DB"."Source.Schema"."View"', '"source.schema"', True),
        ("DB.SCHEMA.BENCHMARK_DB_SEMANTIC", "benchmark_db", False),
        ("OTHER.DB.VIEW", "DB", True),
        ('"DB"."SCHEMA"."VIEW"', "DB_EXTRA", False),
    ],
)
def test_database_scope_matches_exact_database_or_schema_components(
    identifier: str,
    database: str,
    matches: bool,
) -> None:
    assert SnowflakeClient._matches_database(identifier, database) is matches


def test_database_scope_rejects_an_unmatched_allowlist() -> None:
    with pytest.raises(SnowflakeError, match="requested database"):
        _client()._scope_semantic_views([SemanticViewRef(_VIEW)], "OTHER")


def test_discovery_scopes_server_query_before_applying_limit() -> None:
    client = _client(semantic_views=(), max_semantic_views=25)
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        {"database_name": "DB", "schema_name": "S", "name": "TARGET", "last_altered": "target"},
        *[
            {"database_name": "DB", "schema_name": "S", "name": f"V{index}", "last_altered": str(index)}
            for index in range(1, 26)
        ],
    ]
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    views, warnings, truncated = client._resolve_semantic_views(connection, "DB")

    cursor.execute.assert_called_once_with('SHOW SEMANTIC VIEWS IN DATABASE "DB" LIMIT 26')
    assert views[0].fqn == '"DB"."S"."TARGET"'
    assert len(views) == 25
    assert truncated is True
    assert warnings


def test_semantic_metadata_cache_reuses_version_and_refreshes_on_change() -> None:
    client = _client(semantic_views=())
    cursor = MagicMock()
    cursor.fetchone.side_effect = [("tables: []",), ("tables: []",)]
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    first = client._semantic_model(connection, SemanticViewRef(_VIEW, "version-1"), "scope")
    cached = client._semantic_model(connection, SemanticViewRef(_VIEW, "version-1"), "scope")
    refreshed = client._semantic_model(connection, SemanticViewRef(_VIEW, "version-2"), "scope")

    assert first is cached
    assert refreshed is not first
    assert cursor.execute.call_count == 2


def test_semantic_metadata_cache_is_not_shared_across_roles() -> None:
    cursor = MagicMock()
    cursor.fetchone.return_value = ("tables: []",)
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    _client(semantic_views=(), role="ROLE_A")._semantic_model(connection, SemanticViewRef(_VIEW, "version-1"), "scope")
    _client(semantic_views=(), role="ROLE_B")._semantic_model(connection, SemanticViewRef(_VIEW, "version-1"), "scope")

    assert cursor.execute.call_count == 2


def test_semantic_metadata_cache_refreshes_after_ttl_expiry() -> None:
    client = _client(semantic_views=(), semantic_cache_ttl_seconds=1)
    cursor = MagicMock()
    cursor.fetchone.side_effect = [("tables: []",), ("tables: []",)]
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    with patch("snowflake_ontology.client.time.monotonic", side_effect=[0.0, 0.0, 2.0, 2.0]):
        first = client._semantic_model(connection, SemanticViewRef(_VIEW, "version"), "scope")
        refreshed = client._semantic_model(connection, SemanticViewRef(_VIEW, "version"), "scope")

    assert refreshed is not first
    assert cursor.execute.call_count == 2


@pytest.mark.asyncio
async def test_text_to_sql_routes_only_the_catalog_selected_view() -> None:
    client = _client()
    selected = _document("gross_revenue")
    other = _document("other_revenue", semantic_view='"OTHER"."SCHEMA"."OTHER_VIEW"')
    client._semantic_documents = MagicMock(  # type: ignore[method-assign]
        return_value=(
            [selected, other],
            [SemanticViewRef(selected.semantic_view), SemanticViewRef(other.semantic_view)],
            [],
            [],
            False,
        )
    )
    client._post_agent = MagicMock(  # type: ignore[method-assign]
        return_value=(_agent_payload(), "snowflake-request")
    )

    result = await client.text_to_sql(
        TextToSQLRequest(question="Revenue", object_ids=[selected.id]),
        token="test-token",
    )

    assert result.status == "success"
    body = client._post_agent.call_args.kwargs["body"]
    assert [tool["tool_spec"]["name"] for tool in body["tools"]] == ["analyst_tool_1"]
    assert body["tool_resources"]["analyst_tool_1"]["semantic_view"] == selected.semantic_view
    assert selected.name in body["instructions"]["orchestration"]
    assert other.name not in body["instructions"]["orchestration"]


@pytest.mark.asyncio
async def test_text_to_sql_runs_one_repair_after_failed_provider_run() -> None:
    client = _client()
    selected = _document("gross_revenue")
    client._semantic_documents = MagicMock(  # type: ignore[method-assign]
        return_value=([selected], [SemanticViewRef(selected.semantic_view)], [], [], False)
    )
    client._post_agent = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            (_failed_agent_payload(), "initial-request"),
            (_agent_payload(), "repair-request"),
        ]
    )

    result = await client.text_to_sql(
        TextToSQLRequest(question="Revenue", object_ids=[selected.id]),
        token="test-token",
    )

    assert result.status == "success"
    assert client._post_agent.call_count == 2
    repair_body = client._post_agent.call_args_list[1].kwargs["body"]
    repair_question = repair_body["messages"][0]["content"][0]["text"]
    assert "single failed-run repair" in repair_question
    assert [attempt.status for attempt in result.attempts] == ["error", "success"]
    assert result.repair_count == 1
    assert any("performed one Cortex repair run" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_text_to_sql_does_not_repair_a_valid_empty_result() -> None:
    client = _client()
    selected = _document("gross_revenue")
    client._semantic_documents = MagicMock(  # type: ignore[method-assign]
        return_value=([selected], [SemanticViewRef(selected.semantic_view)], [], [], False)
    )
    client._post_agent = MagicMock(  # type: ignore[method-assign]
        return_value=(_agent_payload(rows=[]), "initial-request")
    )

    result = await client.text_to_sql(
        TextToSQLRequest(question="Revenue", object_ids=[selected.id]),
        token="test-token",
    )

    assert result.status == "empty_result"
    client._post_agent.assert_called_once()


@pytest.mark.asyncio
async def test_text_to_sql_limits_failed_run_to_one_repair() -> None:
    client = _client()
    selected = _document("gross_revenue")
    client._semantic_documents = MagicMock(  # type: ignore[method-assign]
        return_value=([selected], [SemanticViewRef(selected.semantic_view)], [], [], False)
    )
    client._post_agent = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            (_failed_agent_payload(), "initial-request"),
            (_failed_agent_payload(), "repair-request"),
        ]
    )

    result = await client.text_to_sql(
        TextToSQLRequest(question="Revenue", object_ids=[selected.id]),
        token="test-token",
    )

    assert result.status == "failed"
    assert client._post_agent.call_count == 2
    assert len(result.attempts) == 2
    assert result.repair_count == 1


@pytest.mark.asyncio
async def test_text_to_sql_does_not_semantically_reinterpret_a_successful_result() -> None:
    client = _client()
    selected = _document("total_contract_value_musd", synonyms=("contract value at risk",))
    client._semantic_documents = MagicMock(  # type: ignore[method-assign]
        return_value=([selected], [SemanticViewRef(selected.semantic_view)], [], [], False)
    )
    client._post_agent = MagicMock(  # type: ignore[method-assign]
        return_value=(_query_payload(_VALID_ANCHOR_SQL), "initial-request")
    )

    result = await client.text_to_sql(
        TextToSQLRequest(question=_ANCHOR_QUESTION, object_ids=[selected.id]),
        token="test-token",
    )

    assert result.status == "success"
    assert result.repair_count == 0
    client._post_agent.assert_called_once()


@pytest.mark.asyncio
async def test_text_to_sql_retains_provider_executed_sql_without_posthoc_statement_classification() -> None:
    client = _client()
    selected = _document("gross_revenue")
    invalid = _agent_payload()
    invalid["content"][-2]["tool_result"]["content"][0]["json"]["sql"] = "DELETE FROM DB.SCHEMA.ORDERS"
    client._semantic_documents = MagicMock(  # type: ignore[method-assign]
        return_value=([selected], [SemanticViewRef(selected.semantic_view)], [], [], False)
    )
    client._post_agent = MagicMock(return_value=(invalid, "initial-request"))  # type: ignore[method-assign]

    result = await client.text_to_sql(
        TextToSQLRequest(question="Revenue", object_ids=[selected.id]),
        token="test-token",
    )

    assert result.status == "success"
    assert result.sql == "DELETE FROM DB.SCHEMA.ORDERS"
    client._post_agent.assert_called_once()


def test_text_to_sql_requires_a_catalog_selection() -> None:
    with pytest.raises(ValueError, match="object_ids"):
        TextToSQLRequest(question="Revenue")


def test_catalog_object_resolution_uses_current_authorized_mapping() -> None:
    client = _client()
    first = _document("gross_revenue")
    second = _document("net_revenue")
    client._semantic_documents = MagicMock(  # type: ignore[method-assign]
        return_value=([first, second], [SemanticViewRef(_VIEW)], [], [], False)
    )

    resolved = client.resolve_catalog_objects([second.id, first.id, second.id], token="test-token")

    assert [document.id for document in resolved] == [second.id, first.id]
    with pytest.raises(SnowflakeError, match="unavailable in the authorized scope"):
        client.resolve_catalog_objects(["sfobj_stale"], token="test-token")


def test_text_to_sql_accepts_selected_view_and_rejects_mixed_views() -> None:
    selected_view = '"DB"."SCHEMA"."RISK_VIEW"'
    analytical = _document("gross_revenue")
    other_analytical = _document("contract_value", semantic_view='"OTHER"."SCHEMA"."OTHER_VIEW"')
    risk = _document("miss_probability", semantic_view=selected_view)
    client = _client(
        semantic_views=(_VIEW, other_analytical.semantic_view, selected_view),
    )
    client._semantic_documents = MagicMock(  # type: ignore[method-assign]
        return_value=(
            [analytical, other_analytical, risk],
            [SemanticViewRef(_VIEW), SemanticViewRef(other_analytical.semantic_view), SemanticViewRef(selected_view)],
            [],
            [],
            False,
        )
    )
    client._post_agent = MagicMock(return_value=(_agent_payload(), "snowflake-request"))  # type: ignore[method-assign]

    result = client._text_to_sql_sync(
        TextToSQLRequest(question="Analyze risk", object_ids=[risk.id]), "token", "request-1"
    )

    assert result.status == "success"
    with pytest.raises(SnowflakeError, match="one Semantic View"):
        client._text_to_sql_sync(
            TextToSQLRequest(question="Compare", object_ids=[analytical.id, other_analytical.id]),
            "token",
            "request-2",
        )
    client._post_agent.assert_called_once()


def test_agent_response_retains_failed_and_successful_attempts_and_duplicate_columns() -> None:
    result = _client()._agent_response(
        _agent_payload(failed_first=True),
        request=_request(),
        request_id="request-1",
        selected_documents=[_document("gross_revenue")],
        tool_views={"analyst_tool_1": _VIEW},
        discovery_warnings=["Discovery warning."],
        started=time.monotonic(),
    )

    assert result.status == "success"
    assert result.total_ms is not None
    payload = result.model_dump()
    assert "attempts" not in payload
    assert "budget" not in payload
    assert "provenance" not in payload
    assert "timings" not in payload
    assert result.query_id == "successful-query"
    assert [column.name for column in result.columns] == ["VALUE", "VALUE"]
    assert result.rows == [["10", "20"]]
    assert [attempt.status for attempt in result.attempts] == ["error", "success"]
    assert [attempt.query_id for attempt in result.attempts] == ["failed-query", "successful-query"]
    assert len(result.result_sets) == 1
    assert result.result_sets[0].rows == [["10", "20"]]
    assert result.attempts[0].error_code == "sql_error"
    assert result.attempts[0].diagnostic_code == "002003"
    assert result.attempts[0].error_message == "Snowflake rejected the SQL execution."
    assert all(attempt.semantic_view == _VIEW for attempt in result.attempts)
    assert result.repair_count == 1
    assert [event.event_type for event in result.trajectory].count("tool_use") == 3
    assert [event.event_type for event in result.trajectory].count("tool_result") == 2
    assert "trajectory" not in payload
    assert result.provenance.selected_tools == ["analyst_tool_1"]
    assert result.provenance.selected_semantic_views == [_VIEW]
    assert result.provenance.selected_object_ids == ["id-gross_revenue"]
    assert result.provenance.verified_query_used is True
    assert result.provenance.verified_query_confidence == 0.97
    assert result.provenance.question_category == "analytics"
    assert result.provenance.selected_models == ["cortex-analyst-model"]
    assert result.provenance.semantic_model_selection == {"semantic_view": _VIEW, "confidence": 0.96}
    assert result.provenance.search_metadata == {"service": "literal_search", "matches": 2}
    assert result.warnings == ["Discovery warning.", "Agent warning."]


def test_agent_response_preserves_every_successful_result_set() -> None:
    first = _agent_payload(rows=[["10", "20"]])
    second = _agent_payload(rows=[["30", "40"]])
    second_execution = second["content"][-2]["tool_result"]["content"][0]["json"]
    second_execution["query_id"] = "second-query"
    payload = {
        "status": "completed",
        "content": [*first["content"], *second["content"]],
    }

    result = _client()._agent_response(
        payload,
        request=_request(),
        request_id="request-1",
        tool_views={"analyst_tool_1": _VIEW},
        discovery_warnings=[],
        started=time.monotonic(),
    )

    assert result.status == "success"
    assert result.rows == [["30", "40"]]
    assert [result_set.query_id for result_set in result.result_sets] == ["successful-query", "second-query"]
    assert [result_set.rows for result_set in result.result_sets] == [[["10", "20"]], [["30", "40"]]]
    assert any("last execution is not necessarily authoritative" in warning for warning in result.warnings)


def test_agent_response_retains_later_provider_success_without_reinterpreting_sql() -> None:
    valid = _agent_payload(rows=[["10", "20"]])
    rejected = _agent_payload(rows=[["30", "40"]])
    rejected_execution = rejected["content"][-2]["tool_result"]["content"][0]["json"]
    rejected_execution["query_id"] = "rejected-query"
    rejected_execution["sql"] = "DELETE FROM DB.SCHEMA.ORDERS"
    rejected["content"][-3]["tool_use"]["input"]["sql"] = rejected_execution["sql"]

    result = _client()._agent_response(
        {"status": "completed", "content": [*valid["content"], *rejected["content"]]},
        request=_request(),
        request_id="request-1",
        discovery_warnings=[],
        started=time.monotonic(),
    )

    assert result.status == "success"
    assert result.query_id == "rejected-query"
    assert result.rows == [["30", "40"]]
    assert [result_set.query_id for result_set in result.result_sets] == ["successful-query", "rejected-query"]
    assert [attempt.status for attempt in result.attempts] == ["success", "success"]
    assert all(event.event_type != "result_rejected" for event in result.trajectory)


def test_agent_response_retains_earlier_provider_success_without_reinterpreting_sql() -> None:
    rejected = _agent_payload(rows=[["30", "40"]])
    rejected_execution = rejected["content"][-2]["tool_result"]["content"][0]["json"]
    rejected_execution["query_id"] = "rejected-query"
    rejected_execution["sql"] = "DELETE FROM DB.SCHEMA.ORDERS"
    rejected["content"][-3]["tool_use"]["input"]["sql"] = rejected_execution["sql"]
    valid = _agent_payload(rows=[["10", "20"]])

    result = _client()._agent_response(
        {"status": "completed", "content": [*rejected["content"], *valid["content"]]},
        request=_request(),
        request_id="request-1",
        discovery_warnings=[],
        started=time.monotonic(),
    )

    assert result.status == "success"
    assert result.query_id == "successful-query"
    assert result.rows == [["10", "20"]]
    assert [result_set.query_id for result_set in result.result_sets] == ["rejected-query", "successful-query"]
    assert len(result.attempts) == 2


def test_agent_response_normalizes_provider_executed_sql_without_posthoc_classification() -> None:
    payload = _agent_payload()
    execution = payload["content"][-2]["tool_result"]["content"][0]["json"]
    execution["sql"] = "DELETE FROM DB.SCHEMA.ORDERS"
    payload["content"][-3]["tool_use"]["input"]["sql"] = execution["sql"]

    result = _client()._agent_response(
        payload,
        request=_request(),
        request_id="request-1",
        discovery_warnings=[],
        started=time.monotonic(),
    )

    assert result.status == "success"
    assert result.query_id == "successful-query"
    assert len(result.result_sets) == 1
    assert result.result_sets[0].sql == "DELETE FROM DB.SCHEMA.ORDERS"
    assert [attempt.status for attempt in result.attempts] == ["success"]
    assert all(event.event_type != "result_rejected" for event in result.trajectory)


def test_agent_response_returns_clarification_as_a_first_class_outcome() -> None:
    payload = {
        "status": "completed",
        "content": [
            {"type": "suggestions", "suggestions": [{"question": "Gross or net revenue?"}]},
            {"type": "text", "text": "Please clarify."},
        ],
    }
    result = _client()._agent_response(
        payload,
        request=_request(),
        request_id="request-1",
        discovery_warnings=[],
        started=time.monotonic(),
    )

    assert result.status == "clarification_required"
    assert result.clarification_suggestions == ["Gross or net revenue?"]
    assert result.sql is None


def test_result_bounds_rows_columns_cells_bytes_binary_and_semistructured_values() -> None:
    client = _client(
        default_max_rows=2,
        max_result_columns=2,
        max_cell_bytes=8,
        max_result_bytes=100,
    )
    payload = _agent_payload(
        rows=[
            ["abcdefghijk", b"secret", {"nested": "value"}],
            ["short", "ok", [1, 2, 3]],
            ["third", "row", "ignored"],
        ]
    )
    execution = payload["content"][-2]["tool_result"]["content"][0]["json"]
    execution["result_set"]["resultSetMetaData"]["rowType"].append({"name": "THIRD", "type": "VARIANT"})
    result = client._agent_response(
        payload,
        request=_request(max_rows=5),
        request_id="request-1",
        discovery_warnings=[],
        started=time.monotonic(),
    )

    assert len(result.columns) == 2
    assert len(result.rows) == 2
    assert result.rows[0] == ["abcdefgh", "[binary "]
    assert result.truncation["rows"] is True
    assert result.truncation["columns"] is True
    assert result.truncation["cells"] is True


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (401, SnowflakeErrorCode.AUTHENTICATION_REQUIRED, False),
        (403, SnowflakeErrorCode.AUTHORIZATION_DENIED, False),
        (408, SnowflakeErrorCode.TIMEOUT, True),
        (429, SnowflakeErrorCode.RATE_LIMITED, True),
        (500, SnowflakeErrorCode.UPSTREAM_ERROR, True),
        (400, SnowflakeErrorCode.UPSTREAM_ERROR, False),
    ],
)
def test_http_error_classification(status: int, code: SnowflakeErrorCode, retryable: bool) -> None:
    response = requests.Response()
    response.status_code = status
    response.headers["retry-after"] = "3"
    error = SnowflakeClient._http_error(response, "request-1")

    assert error.code == code
    assert error.retryable is retryable
    assert error.request_id == "request-1"
    assert error.diagnostic_code == f"cortex_http_{status}"
    assert error.retry_after_seconds == 3


@pytest.mark.parametrize(
    ("exception", "code", "retryable"),
    [
        (DatabaseError(msg="credential=secret", errno=390100, sqlstate="28000"), "authentication_required", False),
        (DatabaseError(msg="credential=secret", errno=394400), "authentication_required", False),
        (DatabaseError(msg="credential=secret", errno=2003, sqlstate="42501"), "authorization_denied", False),
        (OperationalError(msg="credential=secret", errno=250001), "network_error", True),
        (DatabaseError(msg="credential=secret", errno=250002), "configuration_error", False),
        (ValueError("credential=secret"), "upstream_error", True),
    ],
    ids=["authentication", "pat-authentication", "authorization", "network", "configuration", "unexpected"],
)
def test_connector_error_classification_is_typed_and_redacted(
    exception: Exception,
    code: str,
    retryable: bool,
) -> None:
    error = SnowflakeClient._connector_error(exception, operation="connect")

    assert error.code == code
    assert error.retryable is retryable
    assert "secret" not in str(error)


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("Query cancelled; literal=secret", "cancelled"),
        ("Statement timed out; literal=secret", "timeout"),
        ("Warehouse is suspended; literal=secret", "warehouse_error"),
        ("Access denied; literal=secret", "authorization_denied"),
        ("Invalid identifier; literal=secret", "sql_error"),
    ],
)
def test_execution_error_classification_is_typed_and_redacted(message: str, code: str) -> None:
    resolved_code, safe_message = SnowflakeClient._execution_error_category(message)

    assert resolved_code == code
    assert "secret" not in safe_message


def test_stream_transport_calls_agent_run_and_preserves_all_events() -> None:
    client = _client()
    client._host = "account.snowflakecomputing.com"
    response = MagicMock()
    response.status_code = 200
    response.headers = {
        "content-type": "text/event-stream",
        "x-snowflake-request-id": "snowflake-request-1",
    }
    payload = _agent_payload(failed_first=True)
    response.iter_lines.return_value = [
        f"data: {json.dumps({'content': [item]})}".encode() for item in payload["content"]
    ] + [b"data: [DONE]"]
    client._http.post = MagicMock(return_value=response)

    events, request_id = client._post_agent(
        token="test-token",
        request_id="aiq-request-1",
        body={"stream": True},
    )

    url = client._http.post.call_args.args[0]
    assert url.endswith("/api/v2/cortex/agent:run")
    assert client._http.post.call_args.kwargs["stream"] is True
    assert client._http.post.call_args.kwargs["headers"]["Accept"] == "text/event-stream"
    assert len(events) == len(payload["content"])
    assert request_id == "snowflake-request-1"


def test_stream_timeout_preserves_a_completed_sql_result_with_warning() -> None:
    client = _client(orchestration_timeout_seconds=1)
    client._host = "account.snowflakecomputing.com"
    response = MagicMock()
    response.status_code = 200
    response.headers = {
        "content-type": "text/event-stream",
        "x-snowflake-request-id": "snowflake-request-1",
    }
    response.iter_lines.return_value = [
        f"data: {json.dumps(_agent_payload())}".encode(),
        b": keepalive",
    ]
    client._http.post = MagicMock(return_value=response)

    with patch("snowflake_ontology.client.time.monotonic", side_effect=[0.0, 0.0, 0.5, 2.0, 2.0]):
        events, request_id = client._post_agent(token="token", request_id="request", body={})
        result = client._agent_response(
            events,
            request=_request(),
            request_id=request_id,
            discovery_warnings=[],
            started=0.0,
        )

    assert result.status == "success"
    assert result.query_id == "successful-query"
    assert result.rows == [["10", "20"]]
    assert any("timed out after returning a successful SQL result" in warning for warning in result.warnings)
    response.close.assert_called_once_with()


def test_stream_transport_does_not_count_cortex_internal_tool_uses() -> None:
    client = _client()
    client._host = "account.snowflakecomputing.com"
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-type": "text/event-stream"}
    tool_use = {
        "type": "tool_use",
        "tool_use": {"type": "cortex_analyst_text_to_sql", "name": "analyst_tool_1"},
    }
    response.iter_lines.return_value = [
        f"data: {json.dumps({'content': [tool_use]})}",
        f"data: {json.dumps({'content': [tool_use]})}",
    ]
    client._http.post = MagicMock(return_value=response)

    events, _request_id = client._post_agent(token="token", request_id="request", body={})

    assert len(events) == 2
    response.close.assert_not_called()


def test_transport_honors_retry_after_for_rate_limit_within_deadline() -> None:
    client = _client()
    client._host = "account.snowflakecomputing.com"
    limited = MagicMock()
    limited.status_code = 429
    limited.headers = {"retry-after": "0", "x-snowflake-request-id": "limited-request"}
    success = MagicMock()
    success.status_code = 200
    success.headers = {"content-type": "application/json", "x-snowflake-request-id": "success-request"}
    success.json.return_value = {"status": "completed", "content": []}
    client._http.post = MagicMock(side_effect=[limited, success])

    with patch("snowflake_ontology.client.time.sleep") as sleep:
        events, request_id = client._post_agent(token="token", request_id="request", body={})

    assert events == [{"status": "completed", "content": []}]
    assert request_id == "success-request"
    sleep.assert_called_once_with(0.0)


def test_exhausted_orchestration_deadline_is_retryable() -> None:
    """Expose a bounded orchestration deadline as a transient provider failure."""

    client = _client()
    with (
        patch("snowflake_ontology.client.time.monotonic", side_effect=[0.0, 10_000.0]),
        pytest.raises(SnowflakeError) as raised,
    ):
        client._post_agent(token="token", request_id="request", body={})

    assert raised.value.code is SnowflakeErrorCode.TIMEOUT
    assert raised.value.diagnostic_code == "cortex_orchestration_deadline"
    assert raised.value.retryable is True


@pytest.mark.asyncio
async def test_async_cancellation_closes_the_active_agent_stream() -> None:
    client = _client()
    started = threading.Event()
    release = threading.Event()
    response = MagicMock()

    def run(_request: object, _token: str, request_id: str) -> None:
        with client._active_responses_lock:
            client._active_responses[request_id] = response
        started.set()
        release.wait(timeout=1)

    client._request_id = MagicMock(return_value="cancel-request")  # type: ignore[method-assign]
    client._text_to_sql_sync = MagicMock(side_effect=run)  # type: ignore[method-assign]
    task = asyncio.create_task(client.text_to_sql(_request("slow"), token="token"))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()

    response.close.assert_called_once_with()


def test_cancellation_closes_original_and_failed_run_repair_streams() -> None:
    client = _client()
    original = MagicMock()
    repair = MagicMock()
    unrelated = MagicMock()
    with client._active_responses_lock:
        client._active_responses["request"] = original
        client._active_responses["request-repair-1"] = repair
        client._active_responses["other-request"] = unrelated

    client._cancel_request("request")

    original.close.assert_called_once_with()
    repair.close.assert_called_once_with()
    unrelated.close.assert_not_called()
