# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register Databricks ontology capabilities as a NAT function group."""

import logging
import os
from collections.abc import Mapping

from pydantic import Field
from pydantic import HttpUrl
from pydantic import SecretStr
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionGroupBaseConfig

from .catalog import SemanticCatalogRanker
from .client import FORWARDED_HEADER_NAMES
from .client import DatabricksClient
from .errors import DatabricksError
from .errors import DatabricksErrorCode
from .errors import DatabricksToolError
from .models import CatalogSearchRequest
from .models import TextToSQLRequest

logger = logging.getLogger(__name__)


class DatabricksFunctionGroupConfig(FunctionGroupBaseConfig, name="databricks"):
    """Configuration shared by AI-Q's Databricks ontology tools."""

    workspace_url: HttpUrl
    catalog_llm: LLMRef = Field(
        description="LLM used to extract required business entities from catalog-search questions.",
    )
    catalog_embedder: EmbedderRef = Field(
        description="Embedder used to rank authorized Genie and Unity Catalog ontology objects.",
    )
    catalog_embedding_cache_size: int = Field(
        default=10_000,
        ge=1,
        le=100_000,
        description="Maximum normalized catalog-document embeddings retained by this process.",
    )
    access_token: SecretStr = Field(
        description="Databricks access token supplied through environment-backed workflow configuration.",
    )
    access_token_env: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Optional environment variable resolved when a tool runs. Use for FastAPI profiles because NAT "
            "redacts SecretStr values while materializing its worker config."
        ),
    )
    space_ids: list[str] = Field(
        default_factory=list,
        description="Optional Genie Space allowlist. Empty discovers all Spaces visible to the configured identity.",
    )
    read_timeout_seconds: float = Field(default=60.0, gt=0)
    genie_timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    default_max_rows: int = Field(default=1_000, ge=1)
    max_spaces: int = Field(default=25, ge=1, le=100)
    metadata_concurrency: int = Field(default=5, ge=1, le=20)
    enrich_unity_catalog: bool = Field(
        default=True,
        description="Enrich Genie Space objects with authorized Unity Catalog table and column metadata.",
    )
    max_catalog_tables: int = Field(default=100, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_spaces(self) -> "DatabricksFunctionGroupConfig":
        """Reject ambiguous or duplicate configured Space scope."""

        if len(self.space_ids) != len(set(self.space_ids)):
            raise ValueError("space_ids must not contain duplicates")
        if any(not space_id.strip() for space_id in self.space_ids):
            raise ValueError("space_ids must not contain blank values")
        return self


def _tool_error(error: DatabricksError) -> str:
    """Serialize one safe Databricks provider error."""

    return DatabricksToolError.from_exception(error).model_dump_json(exclude_none=True)


def _resolve_access_token(config: DatabricksFunctionGroupConfig) -> str:
    """Resolve the Databricks token after the FastAPI worker has loaded its config."""

    if config.access_token_env is not None:
        token = os.getenv(config.access_token_env, "")
        if token:
            return token
        raise DatabricksError(
            DatabricksErrorCode.AUTHENTICATION_REQUIRED,
            "Databricks authentication is not configured in the server environment.",
        )

    token = config.access_token.get_secret_value()
    if token and token != "**********":
        return token
    raise DatabricksError(
        DatabricksErrorCode.AUTHENTICATION_REQUIRED,
        "Databricks authentication is not configured for this server runtime.",
    )


def _request_trace_headers() -> Mapping[str, str]:
    """Read allowlisted tracing headers from the active NAT request context."""

    try:
        metadata = Context.get().metadata
        incoming = metadata.headers if metadata else None
    except Exception as exc:
        logger.debug("Unable to read request trace headers (error_type=%s)", type(exc).__name__)
        return {}
    if not incoming:
        return {}
    return {name: value for name, value in incoming.items() if name.lower() in FORWARDED_HEADER_NAMES and value}


@register_function_group(
    config_type=DatabricksFunctionGroupConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def databricks_function_group(config: DatabricksFunctionGroupConfig, _builder: Builder):
    """Build Databricks ontology tools around typed SDK and semantic retrieval clients."""

    catalog_llm = await _builder.get_llm(config.catalog_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    catalog_embedder = await _builder.get_embedder(
        config.catalog_embedder,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )
    ranker = SemanticCatalogRanker(
        llm=catalog_llm,
        embedder=catalog_embedder,
        embedding_cache_size=config.catalog_embedding_cache_size,
    )

    async with DatabricksClient.from_config(config, catalog_ranker=ranker) as client:

        async def catalog_search(request: CatalogSearchRequest) -> str:
            """Find authorized Databricks ontology objects relevant to an enterprise-data question.

            It extracts required business entities and embeds authorized Genie Space and Unity Catalog objects for
            catalog-aware routing. This capability inspects metadata and does not execute the analytical question.
            Candidate IDs are opaque; copy selected `id` values verbatim into `object_ids` on the compatible
            Databricks execution tool.
            """

            try:
                result = await client.catalog_search(
                    request,
                    token=_resolve_access_token(config),
                    trace_headers=_request_trace_headers(),
                )
                return result.model_dump_json(exclude_none=True)
            except DatabricksError as error:
                return _tool_error(error)
            except Exception as exc:
                logger.error("Unexpected Databricks catalog-search failure (error_type=%s)", type(exc).__name__)
                return _tool_error(
                    DatabricksError(
                        DatabricksErrorCode.UPSTREAM_ERROR,
                        "Databricks catalog search failed.",
                    )
                )

        async def text_to_sql(request: TextToSQLRequest) -> str:
            """Generate SQL and return bounded executed rows through Databricks Genie.

            Use for analytical questions over enterprise structured data. Pass catalog candidate IDs unchanged; the
            provider resolves them against current authorized metadata and requires one analytical Genie Space. The
            result includes generated SQL, rows, textual explanation, and request identifiers. Databricks executes the
            SQL using the supplied identity's permissions.
            """

            try:
                result = await client.text_to_sql(
                    request,
                    token=_resolve_access_token(config),
                    trace_headers=_request_trace_headers(),
                )
                return result.model_dump_json(exclude_none=True)
            except DatabricksError as error:
                return _tool_error(error)
            except Exception as exc:
                logger.error("Unexpected Databricks text-to-SQL failure (error_type=%s)", type(exc).__name__)
                return _tool_error(
                    DatabricksError(
                        DatabricksErrorCode.UPSTREAM_ERROR,
                        "Databricks text-to-SQL failed.",
                    )
                )

        group = FunctionGroup(config=config)
        group.add_function(
            "catalog_search",
            catalog_search,
            input_schema=CatalogSearchRequest,
            description=catalog_search.__doc__,
        )
        group.add_function(
            "text_to_sql",
            text_to_sql,
            input_schema=TextToSQLRequest,
            description=text_to_sql.__doc__,
        )
        yield group
