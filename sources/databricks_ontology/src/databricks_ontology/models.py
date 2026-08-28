# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts for Databricks ontology capabilities."""

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field


class DatabricksRequest(BaseModel):
    """Base model for agent input to Databricks tools."""

    model_config = ConfigDict(extra="forbid")


class DatabricksResponse(BaseModel):
    """Base model for normalized Databricks tool output."""

    model_config = ConfigDict(extra="ignore")


class CatalogSearchRequest(DatabricksRequest):
    """Search authorized Genie Space ontology objects relevant to a question."""

    question: str = Field(min_length=1, max_length=4_096)
    database_name: str | None = Field(default=None, min_length=1, max_length=256)
    max_results: int = Field(default=20, ge=1, le=100)
    max_distance: float | None = Field(
        default=None,
        gt=0,
        le=2,
        description="Optional cosine-distance cutoff. Omit to rank the nearest semantic objects without a cutoff.",
    )
    space_id: str | None = Field(default=None, min_length=1)


class CatalogCandidate(DatabricksResponse):
    """One bounded candidate used only to select an opaque catalog object."""

    id: str
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


class CatalogSearchResponse(DatabricksResponse):
    """Semantically ranked catalog candidates and experimental coverage."""

    status: Literal["success"] = "success"
    request_id: str | None = None
    coverage: float = Field(ge=0, le=1)
    candidates: list[CatalogCandidate]
    uncovered_entities: list[str] = Field(default_factory=list)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class TextToSQLRequest(DatabricksRequest):
    """Generate SQL and retrieve Genie's bounded query result."""

    question: str = Field(min_length=1, max_length=4_096)
    database_name: str | None = Field(default=None, min_length=1, max_length=256)
    object_ids: list[str] = Field(
        min_length=1,
        description=(
            "Opaque candidate IDs copied verbatim from databricks__catalog_search. "
            "Do not construct, shorten, or translate these values."
        ),
    )
    max_rows: int = Field(default=1_000, ge=1)


class ResultColumn(DatabricksResponse):
    """One column in a bounded Genie SQL result."""

    name: str
    data_type: str | None = None


class TextToSQLResponse(DatabricksResponse):
    """Normalized Genie answer with generated SQL and bounded executed rows."""

    request_id: str | None = None
    outcome: Literal["query_evidence", "clarification"]
    response: str | None = None
    sql: str | None = None
    columns: list[ResultColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    space_id: str
    conversation_id: str | None = Field(default=None, exclude=True)
    message_id: str | None = Field(default=None, exclude=True)
    objects_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, int | float] = Field(default_factory=dict, exclude=True)

    @computed_field
    @property
    def status(self) -> Literal["success", "clarification_required"]:
        """Normalize Genie's valid query or clarification outcome."""

        return "success" if self.outcome == "query_evidence" else "clarification_required"

    @computed_field
    @property
    def total_ms(self) -> int | float | None:
        """Expose one provider-neutral total duration."""

        value = self.timings.get("total_ms")
        return value if isinstance(value, (int, float)) else None
