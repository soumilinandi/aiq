# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Databricks NAT function-group registration."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from databricks_ontology.errors import DatabricksError
from databricks_ontology.errors import DatabricksErrorCode
from databricks_ontology.models import CatalogSearchResponse
from databricks_ontology.register import DatabricksFunctionGroupConfig
from databricks_ontology.register import databricks_function_group


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


def _config(**overrides) -> DatabricksFunctionGroupConfig:
    """Create the Databricks function-group config."""

    values = {
        "workspace_url": "https://workspace.example",
        "catalog_llm": "catalog-llm",
        "catalog_embedder": "catalog-embedder",
        "access_token": "configured-token",
    }
    values.update(overrides)
    return DatabricksFunctionGroupConfig(**values)


def _builder() -> MagicMock:
    """Return a builder double that resolves semantic catalog models."""

    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=MagicMock())
    builder.get_embedder = AsyncMock(return_value=MagicMock())
    return builder


def test_config_uses_one_genie_space_scope() -> None:
    """Use one Genie Space scope for catalog search and analytical execution."""

    config = _config(space_ids=["space-1"])

    assert config.space_ids == ["space-1"]


@pytest.mark.asyncio
async def test_group_exposes_only_requested_capability() -> None:
    """Use NAT function-group inclusion to expose only selected tools."""

    client = MagicMock()
    config = _config(include=["catalog_search"])

    with patch("databricks_ontology.register.DatabricksClient.from_config", return_value=FakeClientContext(client)):
        async with databricks_function_group(config, _builder()) as group:
            tools = await group.get_accessible_functions()

    assert set(tools) == {"databricks__catalog_search"}


@pytest.mark.asyncio
async def test_catalog_search_uses_configured_token() -> None:
    """Pass the configured development token to the SDK client."""

    client = MagicMock()
    client.catalog_search = AsyncMock(
        return_value=CatalogSearchResponse(
            request_id="request-1",
            coverage=1,
            candidates=[],
        )
    )
    config = _config(
        include=["catalog_search"],
    )

    with (
        patch("databricks_ontology.register.DatabricksClient.from_config", return_value=FakeClientContext(client)),
        patch("databricks_ontology.register._request_trace_headers", return_value={}),
    ):
        async with databricks_function_group(config, _builder()) as group:
            tool = (await group.get_accessible_functions())["databricks__catalog_search"]
            result = json.loads(await tool.ainvoke({"question": "Find customer data"}))

    assert result["request_id"] == "request-1"
    assert client.catalog_search.await_args.kwargs["token"] == "configured-token"


@pytest.mark.asyncio
async def test_catalog_search_resolves_runtime_environment_token() -> None:
    """Recover a FastAPI-redacted SecretStr from the configured environment reference."""

    client = MagicMock()
    client.catalog_search = AsyncMock(
        return_value=CatalogSearchResponse(
            request_id="request-1",
            coverage=1,
            candidates=[],
        )
    )
    config = _config(
        access_token="**********",
        access_token_env="DATABRICKS_TOKEN",
        include=["catalog_search"],
    )

    with (
        patch.dict("os.environ", {"DATABRICKS_TOKEN": "runtime-token"}),
        patch("databricks_ontology.register.DatabricksClient.from_config", return_value=FakeClientContext(client)),
        patch("databricks_ontology.register._request_trace_headers", return_value={}),
    ):
        async with databricks_function_group(config, _builder()) as group:
            tool = (await group.get_accessible_functions())["databricks__catalog_search"]
            result = json.loads(await tool.ainvoke({"question": "Find customer data"}))

    assert result["request_id"] == "request-1"
    assert client.catalog_search.await_args.kwargs["token"] == "runtime-token"


@pytest.mark.asyncio
async def test_provider_failure_has_normalized_error_status() -> None:
    client = MagicMock()
    client.catalog_search = AsyncMock(
        side_effect=DatabricksError(DatabricksErrorCode.RATE_LIMITED, "Databricks is busy.", retryable=True)
    )
    config = _config(include=["catalog_search"])

    with patch("databricks_ontology.register.DatabricksClient.from_config", return_value=FakeClientContext(client)):
        async with databricks_function_group(config, _builder()) as group:
            tool = (await group.get_accessible_functions())["databricks__catalog_search"]
            result = json.loads(await tool.ainvoke({"question": "Find customer data"}))

    assert result == {
        "status": "error",
        "code": "rate_limited",
        "message": "Databricks is busy.",
        "retryable": True,
    }
