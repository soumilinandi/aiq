# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe errors returned by the Databricks ontology provider."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict


class DatabricksErrorCode(StrEnum):
    """Stable error codes exposed to AI-Q tools."""

    AUTHENTICATION_REQUIRED = "authentication_required"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UPSTREAM_ERROR = "upstream_error"


class DatabricksError(RuntimeError):
    """Internal provider failure with a safe agent-facing message."""

    def __init__(
        self,
        code: DatabricksErrorCode,
        message: str,
        *,
        request_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        """Store normalized failure metadata without provider response content."""

        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


class DatabricksToolError(BaseModel):
    """Serialized error returned by a Databricks NAT tool."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    code: DatabricksErrorCode
    message: str
    request_id: str | None = None
    retryable: bool = False
    failure_stage: str | None = None
    attempts: int | None = None

    @classmethod
    def from_exception(cls, error: DatabricksError) -> "DatabricksToolError":
        """Create a safe tool response from an internal exception."""

        return cls(
            code=error.code,
            message=str(error),
            request_id=error.request_id,
            retryable=error.retryable,
        )
