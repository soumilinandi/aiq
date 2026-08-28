# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, redacted errors for the Snowflake ontology provider."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict


class SnowflakeErrorCode(StrEnum):
    """Stable error codes returned by Snowflake tools."""

    AUTHENTICATION_REQUIRED = "authentication_required"
    AUTHORIZATION_DENIED = "authorization_denied"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    WAREHOUSE_ERROR = "warehouse_error"
    SQL_ERROR = "sql_error"
    CANCELLED = "cancelled"
    UPSTREAM_ERROR = "upstream_error"


class SnowflakeError(RuntimeError):
    """One safe provider failure without warehouse content or credentials."""

    def __init__(
        self,
        code: SnowflakeErrorCode,
        message: str,
        *,
        request_id: str | None = None,
        query_id: str | None = None,
        diagnostic_code: str | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.query_id = query_id
        self.diagnostic_code = diagnostic_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class SnowflakeToolError(BaseModel):
    """Serialized provider error returned to an AI-Q agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    code: SnowflakeErrorCode
    message: str
    request_id: str | None = None
    failure_stage: str | None = None
    attempts: int | None = None
    query_id: str | None = None
    diagnostic_code: str | None = None
    retryable: bool = False
    retry_after_seconds: float | None = None

    @classmethod
    def from_exception(cls, error: SnowflakeError) -> "SnowflakeToolError":
        """Convert an internal provider error into safe tool output."""

        return cls(
            code=error.code,
            message=str(error),
            request_id=error.request_id,
            query_id=error.query_id,
            diagnostic_code=error.diagnostic_code,
            retryable=error.retryable,
            retry_after_seconds=error.retry_after_seconds,
        )
