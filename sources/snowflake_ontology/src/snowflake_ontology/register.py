# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register Snowflake ontology capabilities as a NAT function group."""

import logging
import os
from collections.abc import Mapping

from pydantic import Field
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
from .client import SnowflakeClient
from .errors import SnowflakeError
from .errors import SnowflakeErrorCode
from .errors import SnowflakeToolError
from .models import CatalogSearchRequest
from .models import TextToSQLRequest

logger = logging.getLogger(__name__)


class SnowflakeFunctionGroupConfig(FunctionGroupBaseConfig, name="snowflake"):
    """Configuration shared by AI-Q's Snowflake ontology tools."""

    account: str = Field(min_length=1)
    user: str = Field(min_length=1)
    role: str = Field(
        min_length=1,
        description="Dedicated least-privilege Snowflake role used by the deployment-scoped PAT.",
    )
    warehouse: str = Field(min_length=1)
    access_token: SecretStr = Field(
        description="Snowflake PAT supplied through environment-backed workflow configuration.",
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
    semantic_views: list[str] = Field(
        default_factory=list,
        description="Optional Semantic View allowlist. Empty discovers views authorized for the current identity.",
    )
    max_semantic_views: int = Field(default=25, ge=1, le=100)
    catalog_llm: LLMRef = Field(
        description="LLM used to extract required business entities from catalog-search questions.",
    )
    catalog_embedder: EmbedderRef = Field(
        description="Embedder used to rank authorized Snowflake semantic objects.",
    )
    catalog_embedding_cache_size: int = Field(
        default=10_000,
        ge=1,
        le=100_000,
        description="Maximum normalized catalog-document embeddings retained by this process.",
    )
    client_prefetch_threads: int = Field(
        default=1,
        ge=1,
        description="Snowflake result-set download threads. Keep low to avoid bursty staged-result downloads.",
    )
    read_timeout_seconds: float = Field(default=120.0, gt=0)
    query_timeout_seconds: int = Field(default=120, ge=1, le=900)
    orchestration_timeout_seconds: int = Field(default=180, ge=1, le=1_800)
    orchestration_token_budget: int = Field(default=32_000, ge=1, le=128_000)
    default_max_rows: int = Field(default=1_000, ge=1)
    max_result_columns: int = Field(default=100, ge=1, le=1_000)
    max_cell_bytes: int = Field(default=16_384, ge=64, le=1_000_000)
    max_result_bytes: int = Field(default=1_000_000, ge=1_024, le=20_000_000)
    semantic_cache_ttl_seconds: float = Field(default=300.0, gt=0, le=86_400)
    semantic_load_concurrency: int = Field(default=4, ge=1, le=16)
    expose_sample_values: bool = Field(
        default=False,
        description="Expose bounded Semantic View samples under the deployment metadata policy.",
    )

    @model_validator(mode="after")
    def validate_semantic_scope(self) -> "SnowflakeFunctionGroupConfig":
        """Validate the shared Semantic View scope."""

        if len(self.semantic_views) != len(set(self.semantic_views)):
            raise ValueError("semantic_views must not contain duplicates")
        return self


def _tool_error(error: SnowflakeError) -> str:
    """Serialize one safe Snowflake provider error."""

    return SnowflakeToolError.from_exception(error).model_dump_json(exclude_none=True)


def _resolve_access_token(config: SnowflakeFunctionGroupConfig) -> str:
    """Resolve the Snowflake PAT after the FastAPI worker has loaded its config."""

    if config.access_token_env is not None:
        token = os.getenv(config.access_token_env, "")
        if token:
            return token
        raise SnowflakeError(
            SnowflakeErrorCode.AUTHENTICATION_REQUIRED,
            "Snowflake authentication is not configured in the server environment.",
        )

    token = config.access_token.get_secret_value()
    if token and token != "**********":
        return token
    raise SnowflakeError(
        SnowflakeErrorCode.AUTHENTICATION_REQUIRED,
        "Snowflake authentication is not configured for this server runtime.",
    )


def _request_trace_headers() -> Mapping[str, str]:
    """Read allowlisted trace headers from the current NAT context."""

    try:
        metadata = Context.get().metadata
        incoming = metadata.headers if metadata else None
    except Exception as exc:
        logger.debug("Unable to read request trace headers (error_type=%s)", type(exc).__name__)
        return {}
    if not incoming:
        return {}
    return {name: value for name, value in incoming.items() if name.lower() in FORWARDED_HEADER_NAMES and value}


@register_function_group(config_type=SnowflakeFunctionGroupConfig)
async def snowflake_function_group(config: SnowflakeFunctionGroupConfig, builder: Builder):
    """Build Snowflake catalog-search and text-to-SQL tools."""

    catalog_llm = await builder.get_llm(config.catalog_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    catalog_embedder = await builder.get_embedder(config.catalog_embedder, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    ranker = SemanticCatalogRanker(
        llm=catalog_llm,
        embedder=catalog_embedder,
        embedding_cache_size=config.catalog_embedding_cache_size,
    )

    async with SnowflakeClient(
        account=config.account,
        user=config.user,
        role=config.role,
        warehouse=config.warehouse,
        semantic_views=tuple(config.semantic_views),
        max_semantic_views=config.max_semantic_views,
        catalog_ranker=ranker,
        client_prefetch_threads=config.client_prefetch_threads,
        read_timeout_seconds=config.read_timeout_seconds,
        query_timeout_seconds=config.query_timeout_seconds,
        orchestration_timeout_seconds=config.orchestration_timeout_seconds,
        orchestration_token_budget=config.orchestration_token_budget,
        default_max_rows=config.default_max_rows,
        max_result_columns=config.max_result_columns,
        max_cell_bytes=config.max_cell_bytes,
        max_result_bytes=config.max_result_bytes,
        semantic_cache_ttl_seconds=config.semantic_cache_ttl_seconds,
        semantic_load_concurrency=config.semantic_load_concurrency,
        expose_sample_values=config.expose_sample_values,
    ) as client:

        async def catalog_search(request: CatalogSearchRequest) -> str:
            """Rank authorized Snowflake semantic objects for catalog-aware routing.

            Candidate IDs are opaque. Copy selected `id` values verbatim into `object_ids` on the compatible
            Snowflake execution tool; never reconstruct IDs from names or metadata.
            """

            try:
                result = await client.catalog_search(
                    request,
                    token=_resolve_access_token(config),
                    trace_headers=_request_trace_headers(),
                )
                return result.model_dump_json(exclude_none=True)
            except SnowflakeError as error:
                return _tool_error(error)
            except Exception as exc:
                logger.error("Unexpected Snowflake catalog-search failure (error_type=%s)", type(exc).__name__)
                return _tool_error(
                    SnowflakeError(SnowflakeErrorCode.UPSTREAM_ERROR, "Snowflake catalog search failed.")
                )

        async def text_to_sql(request: TextToSQLRequest) -> str:
            """Generate and execute SQL with Cortex Agents under the configured Snowflake identity.

            Pass selected opaque catalog IDs unchanged in `object_ids`. The provider resolves them against current
            authorized Semantic View metadata before routing the analytical question to Cortex.
            """

            try:
                result = await client.text_to_sql(
                    request,
                    token=_resolve_access_token(config),
                    trace_headers=_request_trace_headers(),
                )
                return result.model_dump_json(exclude_none=True)
            except SnowflakeError as error:
                return _tool_error(error)
            except Exception as exc:
                logger.error("Unexpected Snowflake text-to-SQL failure (error_type=%s)", type(exc).__name__)
                return _tool_error(SnowflakeError(SnowflakeErrorCode.UPSTREAM_ERROR, "Snowflake text-to-SQL failed."))

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
