# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog routing contracts."""

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class CatalogCandidate(BaseModel):
    """A ranked semantic candidate returned by an ontology provider."""

    model_config = ConfigDict(extra="forbid")

    label: str
    attribute: str
    term: str
    id: str
    score: float | None = Field(default=None, ge=0, le=1)
    scope: str | None = None
    summary: str | None = Field(default=None, max_length=2_000)
    capabilities: list[str] = Field(default_factory=list)


class CatalogRoutingResponse(BaseModel):
    """Validated ontology catalog result used to select the research workflow."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    request_id: str | None = None
    coverage: float = Field(ge=0, le=1)
    candidates: list[CatalogCandidate]
    uncovered_entities: list[str] | None = None
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
