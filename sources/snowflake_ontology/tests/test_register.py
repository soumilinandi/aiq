# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Snowflake NAT function-group registration."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from snowflake_ontology.models import CatalogSearchResponse
from snowflake_ontology.register import SnowflakeFunctionGroupConfig
from snowflake_ontology.register import snowflake_function_group


class FakeClientContext:
    """Expose a mocked client through an asynchronous context manager."""

    def __init__(self, client: MagicMock) -> None:
        """Store the mocked client."""

        self.client = client

    async def __aenter__(self) -> MagicMock:
        """Return the mocked client."""

        return self.client

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """Finish cleanup without suppressing failures."""


def _config(**overrides: object) -> SnowflakeFunctionGroupConfig:
    """Create the Snowflake function-group config."""

    values = {
        "account": "account",
        "user": "user",
        "role": "AIQ_READ_ONLY",
        "warehouse": "warehouse",
        "access_token": "configured-token",
        "catalog_llm": "catalog-llm",
        "catalog_embedder": "catalog-embedder",
        "include": ["catalog_search"],
    }
    values.update(overrides)
    return SnowflakeFunctionGroupConfig(**values)


def _builder() -> MagicMock:
    """Return a builder double that resolves semantic catalog models."""

    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=MagicMock())
    builder.get_embedder = AsyncMock(return_value=MagicMock())
    return builder


def test_config_uses_one_semantic_view_scope() -> None:
    config = _config(semantic_views=["DB.SCHEMA.SHARED"])

    assert config.semantic_views == ["DB.SCHEMA.SHARED"]


@pytest.mark.asyncio
async def test_catalog_search_uses_configured_token() -> None:
    """Pass a direct programmatic token to the Snowflake client."""

    client = MagicMock()
    client.catalog_search = AsyncMock(
        return_value=CatalogSearchResponse(request_id="request-1", coverage=1, candidates=[])
    )
    config = _config()

    with (
        patch("snowflake_ontology.register.SnowflakeClient", return_value=FakeClientContext(client)),
        patch("snowflake_ontology.register._request_trace_headers", return_value={}),
    ):
        async with snowflake_function_group(config, _builder()) as group:
            tool = (await group.get_accessible_functions())["snowflake__catalog_search"]
            result = json.loads(await tool.ainvoke({"question": "Find deployment data"}))

    assert result["request_id"] == "request-1"
    assert client.catalog_search.await_args.kwargs["token"] == "configured-token"


@pytest.mark.asyncio
async def test_catalog_search_resolves_runtime_environment_token() -> None:
    """Recover a FastAPI-redacted SecretStr from the configured environment reference."""

    client = MagicMock()
    client.catalog_search = AsyncMock(
        return_value=CatalogSearchResponse(request_id="request-1", coverage=1, candidates=[])
    )
    config = _config(access_token="**********", access_token_env="SNOWFLAKE_PAT")

    with (
        patch.dict("os.environ", {"SNOWFLAKE_PAT": "runtime-token"}),
        patch("snowflake_ontology.register.SnowflakeClient", return_value=FakeClientContext(client)),
        patch("snowflake_ontology.register._request_trace_headers", return_value={}),
    ):
        async with snowflake_function_group(config, _builder()) as group:
            tool = (await group.get_accessible_functions())["snowflake__catalog_search"]
            result = json.loads(await tool.ainvoke({"question": "Find deployment data"}))

    assert result["request_id"] == "request-1"
    assert client.catalog_search.await_args.kwargs["token"] == "runtime-token"


@pytest.mark.asyncio
async def test_catalog_search_returns_authentication_error_when_runtime_token_is_missing() -> None:
    """Fail safely when an environment-backed PAT is unavailable."""

    client = MagicMock()
    client.catalog_search = AsyncMock()
    config = _config(access_token="**********", access_token_env="SNOWFLAKE_PAT")

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("snowflake_ontology.register.SnowflakeClient", return_value=FakeClientContext(client)),
    ):
        async with snowflake_function_group(config, _builder()) as group:
            tool = (await group.get_accessible_functions())["snowflake__catalog_search"]
            result = json.loads(await tool.ainvoke({"question": "Find deployment data"}))

    assert result == {
        "status": "error",
        "code": "authentication_required",
        "message": "Snowflake authentication is not configured in the server environment.",
        "retryable": False,
    }
    client.catalog_search.assert_not_awaited()
