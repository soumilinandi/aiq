# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed Snowflake ontology requests and normalized responses."""

from __future__ import annotations

from typing import Literal
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list[JsonScalar]


class SnowflakeRequest(BaseModel):
    """Strict base model for tool input."""

    model_config = ConfigDict(extra="forbid")


class SnowflakeResponse(BaseModel):
    """Forward-compatible base model for provider output."""

    model_config = ConfigDict(extra="ignore")


class CatalogSearchRequest(SnowflakeRequest):
    """Search configured, authorized Snowflake Semantic View objects."""

    question: str = Field(min_length=1, max_length=4_096)
    database_name: str | None = Field(default=None, min_length=1, max_length=256)
    max_results: int = Field(default=20, ge=1, le=100)
    max_distance: float | None = Field(default=None, gt=0, le=2)


class RelationshipContext(SnowflakeResponse):
    """Relevant relationship metadata for one candidate."""

    name: str
    left_table: str | None = None
    right_table: str | None = None
    relationship_columns: list[dict[str, str]] = Field(default_factory=list)


class SemanticVariableContext(SnowflakeResponse):
    """One bounded Semantic View variable available during analytical generation."""

    name: str
    data_type: str | None = None
    default_value: JsonScalar = None
    description: str | None = None


class CatalogCandidate(SnowflakeResponse):
    """One bounded candidate used only to select an opaque catalog object."""

    id: str = Field(description="Opaque provider-issued identifier.")
    label: str
    attribute: str
    term: str
    scope: str | None = Field(
        default=None,
        description="Provider execution scope that must remain consistent across one grounded tool call.",
    )
    summary: str | None = Field(default=None, max_length=2_000)
    capabilities: list[Literal["text_to_sql"]] = Field(default_factory=list)
    score: float = Field(ge=0, le=1)


class CatalogSearchResponse(SnowflakeResponse):
    """Ranked catalog candidates and routing coverage."""

    status: Literal["success"] = "success"
    request_id: str | None = None
    coverage: float = Field(ge=0, le=1)
    candidates: list[CatalogCandidate]
    uncovered_entities: list[str] = Field(default_factory=list)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class TextToSQLRequest(SnowflakeRequest):
    """Generate and execute SQL through Cortex Agents."""

    question: str = Field(min_length=1, max_length=4_096)
    database_name: str | None = Field(default=None, min_length=1, max_length=256)
    object_ids: list[str] = Field(
        min_length=1,
        max_length=100,
        description="Opaque IDs copied unchanged from the current Snowflake catalog search.",
    )
    execute: Literal[True] = True
    max_rows: int = Field(default=1_000, ge=1)


class ResultColumn(SnowflakeResponse):
    """One ordered column in a bounded Snowflake result."""

    name: str
    data_type: str | None = None


class ExecutionBudget(SnowflakeResponse):
    """Request-level limits applied around Cortex orchestration and returned results."""

    query_timeout_seconds: int
    orchestration_timeout_seconds: int
    token_budget: int
    max_rows: int
    max_columns: int
    max_cell_bytes: int
    max_result_bytes: int


class AgentEvent(SnowflakeResponse):
    """One bounded event retained from the Cortex Agent stream."""

    index: int
    event_type: str
    tool_type: str | None = None
    tool_name: str | None = None
    status: str | None = None
    semantic_view: str | None = None
    sql: str | None = None
    query_id: str | None = None
    request_id: str | None = None
    warning: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SQLAttempt(SnowflakeResponse):
    """One generated SQL execution attempt, including failures."""

    attempt: int
    sql: str
    status: str
    tool_name: str | None = None
    semantic_view: str | None = None
    query_id: str | None = None
    error_code: str | None = None
    diagnostic_code: str | None = None
    error_message: str | None = None
    repair_reason: str | None = None


class SQLResultSet(SnowflakeResponse):
    """One completed SQL execution returned by Cortex Agent Run."""

    status: Literal["success", "empty_result"]
    query_id: str | None = None
    sql: str
    columns: list[ResultColumn] = Field(default_factory=list)
    rows: list[list[JsonValue]] = Field(default_factory=list)
    truncated: bool = False
    truncation: dict[str, int | bool] = Field(default_factory=dict)
    objects_used: list[str] = Field(default_factory=list)


class CortexProvenance(SnowflakeResponse):
    """Routing and confidence metadata emitted by Cortex."""

    selected_tools: list[str] = Field(default_factory=list)
    selected_semantic_views: list[str] = Field(default_factory=list)
    selected_models: list[str] = Field(default_factory=list)
    selected_object_ids: list[str] = Field(default_factory=list)
    verified_query_used: bool | None = None
    verified_query_name: str | None = None
    verified_query_confidence: float | None = None
    question_category: str | None = None
    semantic_model_selection: dict[str, JsonValue] = Field(default_factory=dict)
    search_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TextToSQLResponse(SnowflakeResponse):
    """Normalized Cortex outcome and complete bounded execution trajectory."""

    request_id: str | None = None
    status: Literal["success", "empty_result", "clarification_required", "failed"]
    query_id: str | None = None
    response: str | None = None
    clarification_suggestions: list[str] = Field(default_factory=list)
    sql: str | None = None
    columns: list[ResultColumn] = Field(default_factory=list)
    rows: list[list[JsonValue]] = Field(default_factory=list)
    truncated: bool = False
    truncation: dict[str, int | bool] = Field(default_factory=dict)
    objects_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    attempts: list[SQLAttempt] = Field(default_factory=list, exclude=True)
    result_sets: list[SQLResultSet] = Field(default_factory=list)
    trajectory: list[AgentEvent] = Field(default_factory=list, exclude=True)
    repair_count: int = 0
    budget: ExecutionBudget = Field(exclude=True)
    provenance: CortexProvenance = Field(default_factory=CortexProvenance, exclude=True)
    timings: dict[str, int | float] = Field(default_factory=dict, exclude=True)

    @computed_field
    @property
    def total_ms(self) -> int | float | None:
        """Expose one provider-neutral total duration."""

        value = self.timings.get("total_ms")
        return value if isinstance(value, (int, float)) else None
