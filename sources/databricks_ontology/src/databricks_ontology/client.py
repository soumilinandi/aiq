# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed Databricks SDK client for ontology search and Genie SQL execution."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk import errors as sdk_errors
from databricks.sdk.config import Config

from .catalog import SemanticCatalogRanker
from .catalog import documents_from_space
from .catalog import table_identifiers_from_space
from .errors import DatabricksError
from .errors import DatabricksErrorCode
from .models import CatalogSearchRequest
from .models import CatalogSearchResponse
from .models import ResultColumn
from .models import TextToSQLRequest
from .models import TextToSQLResponse

FORWARDED_HEADER_NAMES = frozenset({"baggage", "traceparent", "tracestate", "x-correlation-id", "x-request-id"})
_SQL_OBJECT_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+((?:`[^`]+`|[A-Za-z_][\w$]*)(?:\.(?:`[^`]+`|[A-Za-z_][\w$]*)){0,2})",
    re.I,
)

WorkspaceFactory = Callable[[str, Mapping[str, str]], Any]


@dataclass(frozen=True)
class _SpaceLoadResult:
    """Loaded Genie Spaces plus whether discovery stopped at the configured bound."""

    spaces: list[dict[str, Any]]
    truncated: bool = False


class DatabricksClient:
    """NAT-independent adapter over typed Genie and Unity Catalog SDK APIs."""

    def __init__(
        self,
        *,
        workspace_url: str,
        catalog_ranker: SemanticCatalogRanker,
        space_ids: tuple[str, ...] = (),
        read_timeout_seconds: float = 60.0,
        genie_timeout_seconds: float = 120.0,
        max_retries: int = 2,
        default_max_rows: int = 1_000,
        max_spaces: int = 25,
        metadata_concurrency: int = 5,
        enrich_unity_catalog: bool = True,
        max_catalog_tables: int = 100,
        genie_poll_interval_seconds: float = 1.0,
        max_genie_repairs: int = 1,
        workspace_factory: WorkspaceFactory | None = None,
    ) -> None:
        """Configure typed provider calls, semantic retrieval, and result bounds."""

        self._workspace_url = workspace_url.rstrip("/")
        self._catalog_ranker = catalog_ranker
        self._space_ids = space_ids
        self._read_timeout_seconds = read_timeout_seconds
        self._genie_timeout_seconds = genie_timeout_seconds
        self._max_retries = max_retries
        self._default_max_rows = default_max_rows
        self._max_spaces = max_spaces
        self._metadata_concurrency = metadata_concurrency
        self._enrich_unity_catalog = enrich_unity_catalog
        self._max_catalog_tables = max_catalog_tables
        self._genie_poll_interval_seconds = genie_poll_interval_seconds
        self._max_genie_repairs = max_genie_repairs
        self._workspace_factory = workspace_factory or self._create_workspace

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        catalog_ranker: SemanticCatalogRanker,
    ) -> DatabricksClient:
        """Create a client from the Databricks NAT function-group config."""

        return cls(
            workspace_url=str(config.workspace_url),
            catalog_ranker=catalog_ranker,
            space_ids=tuple(config.space_ids),
            read_timeout_seconds=config.read_timeout_seconds,
            genie_timeout_seconds=config.genie_timeout_seconds,
            max_retries=config.max_retries,
            default_max_rows=config.default_max_rows,
            max_spaces=config.max_spaces,
            metadata_concurrency=config.metadata_concurrency,
            enrich_unity_catalog=config.enrich_unity_catalog,
            max_catalog_tables=config.max_catalog_tables,
        )

    async def __aenter__(self) -> DatabricksClient:
        """Retain compatibility with NAT function-group lifecycle management."""

        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """Finish the stateless SDK adapter lifecycle."""

    async def catalog_search(
        self,
        request: CatalogSearchRequest,
        *,
        token: str,
        trace_headers: Mapping[str, str] | None = None,
    ) -> CatalogSearchResponse:
        """Retrieve authorized metadata and semantically rank ontology objects."""

        workspace, request_id = self._workspace(token, trace_headers)
        loaded_spaces = await self._load_spaces(
            workspace,
            requested_space_id=request.space_id,
            request_id=request_id,
            database_name=request.database_name,
            allowed_space_ids=self._space_ids,
        )
        catalog_metadata, warnings, enrichment_truncated = await self._load_unity_catalog_metadata(
            workspace,
            loaded_spaces.spaces,
            request_id=request_id,
        )
        documents = []
        for space in loaded_spaces.spaces:
            space_documents = documents_from_space(space, unity_catalog_metadata=catalog_metadata)
            documents.extend(space_documents)
        response = await self._catalog_ranker.rank(
            request.question,
            documents,
            max_results=request.max_results,
            max_distance=request.max_distance,
            request_id=request_id,
        )
        response.candidates = [
            candidate.model_copy(update={"capabilities": self._catalog_capabilities()})
            for candidate in response.candidates
        ]
        response.truncated = response.truncated or loaded_spaces.truncated or enrichment_truncated
        if loaded_spaces.truncated:
            response.warnings.append(
                f"Genie Space discovery stopped at the configured limit of {self._max_spaces}; results cover only "
                "the loaded Spaces."
            )
        if enrichment_truncated:
            response.warnings.append(
                "Unity Catalog metadata enrichment stopped at the configured limit of "
                f"{self._max_catalog_tables} objects."
            )
        response.warnings.extend(warnings)
        return response

    def _catalog_capabilities(self) -> list[str]:
        """Return enabled execution roles for the shared Genie Space scope."""

        return ["text_to_sql"]

    async def text_to_sql(
        self,
        request: TextToSQLRequest,
        *,
        token: str,
        trace_headers: Mapping[str, str] | None = None,
    ) -> TextToSQLResponse:
        """Run the complete typed Genie conversation and query-result workflow."""

        started = time.monotonic()
        workspace, request_id = self._workspace(token, trace_headers)
        space_id = await self._resolve_space_id(
            workspace,
            request,
            request_id=request_id,
        )
        deadline = time.monotonic() + self._genie_timeout_seconds
        message_payload = await self._start_and_poll_genie_message(
            workspace,
            space_id=space_id,
            question=request.question,
            request_id=request_id,
            deadline=deadline,
        )
        for _attempt in range(self._max_genie_repairs):
            if self._message_status(message_payload) not in {"FAILED", "QUERY_RESULT_EXPIRED"}:
                break
            conversation_id, _message_id = self._conversation_ids(message_payload)
            if not conversation_id:
                break
            message_payload = await self._repair_and_poll_genie_message(
                workspace,
                space_id=space_id,
                conversation_id=conversation_id,
                failed_message=message_payload,
                request_id=request_id,
                deadline=deadline,
            )

        status = self._message_status(message_payload)
        if status != "COMPLETED":
            raise DatabricksError(
                DatabricksErrorCode.UPSTREAM_ERROR,
                f"Databricks Genie ended with status {status or 'UNKNOWN'}.",
                request_id=request_id,
                retryable=status in {"FAILED", "QUERY_RESULT_EXPIRED"},
            )
        conversation_id, message_id = self._conversation_ids(message_payload)
        response_text, sql, attachment_id = self._parse_attachments(message_payload)
        columns: list[ResultColumn] = []
        rows: list[dict[str, Any]] = []
        truncated = False
        warnings: list[str] = []
        max_rows = min(request.max_rows, self._default_max_rows)

        if sql and attachment_id and conversation_id and message_id:
            query_result = await self._sdk_call(
                lambda: workspace.genie.get_message_attachment_query_result(
                    space_id,
                    conversation_id,
                    message_id,
                    attachment_id,
                ),
                request_id=request_id,
                timeout=self._read_timeout_seconds,
            )
            columns, rows, truncated = self._normalize_statement_result(
                self._as_dict(query_result),
                max_rows=max_rows,
            )
        elif not sql:
            warnings.append("Databricks Genie returned a textual response without generated SQL.")
        elif sql:
            warnings.append("Databricks Genie returned SQL without a retrievable query-result attachment.")

        return TextToSQLResponse(
            request_id=request_id,
            outcome="query_evidence" if sql else "clarification",
            response=response_text or None,
            sql=sql or None,
            columns=columns,
            rows=rows,
            truncated=truncated,
            space_id=space_id,
            conversation_id=conversation_id or None,
            message_id=message_id or None,
            objects_used=self._objects_from_sql(sql),
            warnings=warnings,
            timings={"total_ms": round((time.monotonic() - started) * 1_000, 2)},
        )

    async def _start_and_poll_genie_message(
        self,
        workspace: Any,
        *,
        space_id: str,
        question: str,
        request_id: str,
        deadline: float,
    ) -> dict[str, Any]:
        """Start a Genie conversation and asynchronously poll its first message."""

        waiter = await self._sdk_call(
            lambda: workspace.genie.start_conversation(
                space_id,
                question,
                enable_visualization=False,
            ),
            request_id=request_id,
            timeout=self._remaining_timeout(deadline, request_id=request_id),
        )
        payload = self._as_dict(getattr(waiter, "response", waiter))
        conversation_id, message_id = self._conversation_ids(payload)
        return await self._poll_genie_message(
            workspace,
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            request_id=request_id,
            deadline=deadline,
        )

    async def _repair_and_poll_genie_message(
        self,
        workspace: Any,
        *,
        space_id: str,
        conversation_id: str,
        failed_message: Mapping[str, Any],
        request_id: str,
        deadline: float,
    ) -> dict[str, Any]:
        """Ask Genie to repair a failed attempt within the existing conversation."""

        waiter = await self._sdk_call(
            lambda: workspace.genie.create_message(
                space_id,
                conversation_id,
                self._repair_prompt(failed_message),
                enable_visualization=False,
            ),
            request_id=request_id,
            timeout=self._remaining_timeout(deadline, request_id=request_id),
        )
        payload = self._as_dict(getattr(waiter, "response", waiter))
        returned_conversation_id, message_id = self._conversation_ids(payload)
        return await self._poll_genie_message(
            workspace,
            space_id=space_id,
            conversation_id=returned_conversation_id or conversation_id,
            message_id=message_id,
            request_id=request_id,
            deadline=deadline,
        )

    @classmethod
    def _repair_prompt(cls, failed_message: Mapping[str, Any]) -> str:
        """Build a bounded follow-up from the failed message's useful context."""

        details: list[str] = []
        error = failed_message.get("error")
        if isinstance(error, Mapping):
            error_type = str(error.get("type") or "").strip()[:100]
            error_text = str(error.get("error") or "").strip()[:500]
            if error_type:
                details.append(f"Failure type: {error_type}")
            if error_text:
                details.append(f"Failure detail: {error_text}")
        _response, sql, _attachment_id = cls._parse_attachments(failed_message)
        if sql:
            details.append(f"Failed SQL:\n{sql[:2_000]}")
        context = "\n".join(details) or "The prior message ended before producing a valid query result."
        return (
            "Repair the previous analytical query in this conversation. Re-evaluate the original question, "
            f"correct the SQL, and return a valid query result.\n{context}"
        )

    async def _poll_genie_message(
        self,
        workspace: Any,
        *,
        space_id: str,
        conversation_id: str,
        message_id: str,
        request_id: str,
        deadline: float,
    ) -> dict[str, Any]:
        """Poll one Genie message without occupying a worker thread between checks."""

        if not conversation_id or not message_id:
            raise DatabricksError(
                DatabricksErrorCode.INVALID_RESPONSE,
                "Databricks Genie did not return conversation and message IDs.",
                request_id=request_id,
            )
        terminal_statuses = {"CANCELLED", "COMPLETED", "FAILED", "QUERY_RESULT_EXPIRED"}
        while True:
            message = await self._sdk_call(
                lambda: workspace.genie.get_message(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                ),
                request_id=request_id,
                timeout=self._remaining_timeout(deadline, request_id=request_id),
            )
            payload = self._as_dict(message)
            if self._message_status(payload) in terminal_statuses:
                return payload
            await asyncio.sleep(
                min(self._genie_poll_interval_seconds, self._remaining_timeout(deadline, request_id=request_id))
            )

    def _remaining_timeout(self, deadline: float, *, request_id: str) -> float:
        """Return the remaining Genie deadline, capped to one SDK read."""

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DatabricksError(
                DatabricksErrorCode.TIMEOUT,
                "Databricks did not finish before the configured deadline.",
                request_id=request_id,
                retryable=True,
            )
        return min(remaining, self._read_timeout_seconds)

    def _create_workspace(self, token: str, headers: Mapping[str, str]) -> WorkspaceClient:
        """Create one request-scoped SDK client without persisting user credentials."""

        config = Config(
            host=self._workspace_url,
            token=token,
            http_timeout_seconds=self._read_timeout_seconds,
            retry_timeout_seconds=self._read_timeout_seconds * (self._max_retries + 1),
            custom_headers=dict(headers),
            product="aiq-databricks-ontology",
            product_version="0.1.0",
        )
        return WorkspaceClient(config=config)

    def _workspace(
        self,
        token: str,
        trace_headers: Mapping[str, str] | None,
    ) -> tuple[Any, str]:
        """Create an authenticated SDK client and local correlation identifier."""

        if not token:
            raise DatabricksError(
                DatabricksErrorCode.AUTHENTICATION_REQUIRED,
                "Databricks authentication is required.",
            )
        headers = {
            name: value
            for name, value in (trace_headers or {}).items()
            if name.lower() in FORWARDED_HEADER_NAMES and value
        }
        request_id = next(
            (value for name, value in headers.items() if name.lower() == "x-request-id"),
            str(uuid.uuid4()),
        )
        headers.setdefault("x-request-id", request_id)
        return self._workspace_factory(token, headers), request_id

    async def _resolve_space_id(
        self,
        workspace: Any,
        request: TextToSQLRequest,
        *,
        request_id: str,
    ) -> str:
        """Resolve opaque IDs against current authorized metadata and select one analytical Space."""

        loaded_spaces = await self._load_spaces(
            workspace,
            requested_space_id=None,
            request_id=request_id,
            database_name=request.database_name,
        )
        if loaded_spaces.truncated:
            raise DatabricksError(
                DatabricksErrorCode.INVALID_REQUEST,
                "Catalog object resolution cannot cover the complete configured Genie Space scope.",
                request_id=request_id,
            )
        spaces = loaded_spaces.spaces
        catalog_metadata, _warnings, enrichment_truncated = await self._load_unity_catalog_metadata(
            workspace,
            spaces,
            request_id=request_id,
        )
        if enrichment_truncated:
            raise DatabricksError(
                DatabricksErrorCode.INVALID_REQUEST,
                "Catalog object resolution cannot cover the complete configured table scope.",
                request_id=request_id,
            )
        documents = [
            document
            for space in spaces
            for document in documents_from_space(space, unity_catalog_metadata=catalog_metadata)
        ]
        by_id = {document.id: document for document in documents}
        selected_ids = list(dict.fromkeys(request.object_ids))
        if any(object_id not in by_id for object_id in selected_ids):
            raise DatabricksError(
                DatabricksErrorCode.INVALID_REQUEST,
                "One or more catalog objects are unavailable in the authorized scope.",
                request_id=request_id,
            )
        selected_spaces = {by_id[object_id].space_id for object_id in selected_ids}
        if len(selected_spaces) != 1:
            raise DatabricksError(
                DatabricksErrorCode.INVALID_REQUEST,
                "Catalog objects for one execution must resolve to one Genie Space.",
                request_id=request_id,
            )
        return next(iter(selected_spaces))

    async def _load_spaces(
        self,
        workspace: Any,
        *,
        requested_space_id: str | None,
        request_id: str,
        database_name: str | None = None,
        allowed_space_ids: tuple[str, ...] | None = None,
    ) -> _SpaceLoadResult:
        """Load full Space definitions visible to the request-scoped identity."""

        allowed_space_ids = self._space_ids if allowed_space_ids is None else allowed_space_ids
        if requested_space_id:
            if allowed_space_ids and requested_space_id not in allowed_space_ids:
                raise DatabricksError(
                    DatabricksErrorCode.INVALID_REQUEST,
                    "The requested Genie Space is outside the configured provider scope.",
                    request_id=request_id,
                )
            space = await self._sdk_call(
                lambda: workspace.genie.get_space(requested_space_id, include_serialized_space=True),
                request_id=request_id,
                timeout=self._read_timeout_seconds,
            )
            spaces = [self._as_dict(space)]
            return _SpaceLoadResult(self._filter_spaces(spaces, database_name, request_id=request_id))

        space_ids = list(allowed_space_ids)
        truncated = len(space_ids) > self._max_spaces
        if not space_ids:
            space_ids, truncated = await self._list_space_ids(workspace, request_id=request_id)
        semaphore = asyncio.Semaphore(self._metadata_concurrency)

        async def load(space_id: str) -> dict[str, Any]:
            async with semaphore:
                space = await self._sdk_call(
                    lambda: workspace.genie.get_space(space_id, include_serialized_space=True),
                    request_id=request_id,
                    timeout=self._read_timeout_seconds,
                )
                return self._as_dict(space)

        spaces = await asyncio.gather(*(load(space_id) for space_id in space_ids[: self._max_spaces]))
        return _SpaceLoadResult(
            self._filter_spaces(spaces, database_name, request_id=request_id),
            truncated=truncated,
        )

    @classmethod
    def _filter_spaces(
        cls,
        spaces: list[dict[str, Any]],
        database_name: str | None,
        *,
        request_id: str,
    ) -> list[dict[str, Any]]:
        """Filter authorized Genie Spaces by their referenced provider objects."""

        if database_name is None:
            return spaces
        filtered = [
            space
            for space in spaces
            if any(
                cls._matches_database(identifier, database_name) for identifier in table_identifiers_from_space(space)
            )
        ]
        if not filtered:
            raise DatabricksError(
                DatabricksErrorCode.INVALID_REQUEST,
                "No authorized Genie Space contains the requested database.",
                request_id=request_id,
            )
        return filtered

    @staticmethod
    def _matches_database(identifier: str, database_name: str) -> bool:
        """Match exact Databricks catalog/schema identifiers, including backtick quoting."""

        requested = DatabricksClient._identifier_parts(database_name)
        qualified = DatabricksClient._identifier_parts(identifier)
        if not requested or len(qualified) < 2 or len(requested) > len(qualified):
            return False
        namespace = qualified[:-1]
        if len(requested) == 1:
            return requested[0] in namespace
        if len(requested) == len(qualified):
            return requested == qualified
        return requested == namespace[-len(requested) :]

    @staticmethod
    def _identifier_parts(identifier: str) -> tuple[str, ...]:
        """Parse a Databricks multipart identifier without splitting quoted dots."""

        parts: list[str] = []
        current: list[str] = []
        quoted = False
        index = 0
        while index < len(identifier):
            character = identifier[index]
            if character == "`":
                if quoted and index + 1 < len(identifier) and identifier[index + 1] == "`":
                    current.append("`")
                    index += 2
                    continue
                quoted = not quoted
            elif character == "." and not quoted:
                part = "".join(current).strip()
                if not part:
                    return ()
                parts.append(part.casefold())
                current = []
            else:
                current.append(character)
            index += 1
        part = "".join(current).strip()
        if quoted or not part:
            return ()
        parts.append(part.casefold())
        return tuple(parts)

    @staticmethod
    def _space_id(space: dict[str, Any]) -> str:
        """Return a required Genie Space identifier from a loaded definition."""

        space_id = str(space.get("space_id") or space.get("id") or "").strip()
        if not space_id:
            raise DatabricksError(DatabricksErrorCode.INVALID_RESPONSE, "Databricks returned a Space without an ID.")
        return space_id

    async def _list_space_ids(self, workspace: Any, *, request_id: str) -> tuple[list[str], bool]:
        """List authorized Genie Spaces with bounded SDK pagination."""

        space_ids: list[str] = []
        page_token: str | None = None
        while len(space_ids) < self._max_spaces:
            page = await self._sdk_call(
                lambda: workspace.genie.list_spaces(
                    page_size=min(100, self._max_spaces - len(space_ids)),
                    page_token=page_token,
                ),
                request_id=request_id,
                timeout=self._read_timeout_seconds,
            )
            payload = self._as_dict(page)
            for space in payload.get("spaces") or []:
                if isinstance(space, dict) and space.get("space_id"):
                    space_ids.append(str(space["space_id"]))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        return space_ids[: self._max_spaces], bool(page_token or len(space_ids) > self._max_spaces)

    async def _load_unity_catalog_metadata(
        self,
        workspace: Any,
        spaces: list[dict[str, Any]],
        *,
        request_id: str,
    ) -> tuple[dict[str, dict[str, Any]], list[str], bool]:
        """Best-effort enrich Space definitions through the authorized Tables API."""

        if not self._enrich_unity_catalog:
            return {}, [], False
        all_identifiers = list(
            dict.fromkeys(identifier for space in spaces for identifier in table_identifiers_from_space(space))
        )
        identifiers = all_identifiers[: self._max_catalog_tables]
        truncated = len(all_identifiers) > len(identifiers)
        semaphore = asyncio.Semaphore(self._metadata_concurrency)

        async def load(identifier: str) -> tuple[str, dict[str, Any] | None]:
            async with semaphore:
                try:
                    table = await self._sdk_call(
                        lambda: workspace.tables.get(identifier),
                        request_id=request_id,
                        timeout=self._read_timeout_seconds,
                    )
                    return identifier, self._as_dict(table)
                except DatabricksError as error:
                    if error.code in {DatabricksErrorCode.NOT_FOUND, DatabricksErrorCode.PERMISSION_DENIED}:
                        return identifier, None
                    raise

        loaded = await asyncio.gather(*(load(identifier) for identifier in identifiers))
        metadata = {identifier: value for identifier, value in loaded if value is not None}
        failed = len(loaded) - len(metadata)
        warnings = []
        if failed:
            warnings.append(f"Unity Catalog metadata enrichment was unavailable for {failed} of {len(loaded)} objects.")
        return metadata, warnings, truncated

    async def _sdk_call(
        self,
        call: Callable[[], Any],
        *,
        request_id: str,
        timeout: float,
    ) -> Any:
        """Run one blocking SDK operation off-loop and normalize safe errors."""

        try:
            return await asyncio.wait_for(asyncio.to_thread(call), timeout=timeout)
        except TimeoutError as exc:
            raise DatabricksError(
                DatabricksErrorCode.TIMEOUT,
                "Databricks did not finish before the configured deadline.",
                request_id=request_id,
                retryable=True,
            ) from exc
        except sdk_errors.Unauthenticated as exc:
            raise DatabricksError(
                DatabricksErrorCode.AUTHENTICATION_REQUIRED,
                "Databricks authentication was rejected.",
                request_id=request_id,
            ) from exc
        except sdk_errors.PermissionDenied as exc:
            raise DatabricksError(
                DatabricksErrorCode.PERMISSION_DENIED,
                "The Databricks identity is not authorized for this operation.",
                request_id=request_id,
            ) from exc
        except (sdk_errors.NotFound, sdk_errors.ResourceDoesNotExist) as exc:
            raise DatabricksError(
                DatabricksErrorCode.NOT_FOUND,
                "The requested Databricks resource was not found.",
                request_id=request_id,
            ) from exc
        except (sdk_errors.TooManyRequests, sdk_errors.RequestLimitExceeded, sdk_errors.ResourceExhausted) as exc:
            raise DatabricksError(
                DatabricksErrorCode.RATE_LIMITED,
                "Databricks rate-limited the request.",
                request_id=request_id,
                retryable=True,
            ) from exc
        except (sdk_errors.DeadlineExceeded, sdk_errors.OperationTimeout) as exc:
            raise DatabricksError(
                DatabricksErrorCode.TIMEOUT,
                "Databricks did not finish before the configured deadline.",
                request_id=request_id,
                retryable=True,
            ) from exc
        except (sdk_errors.BadRequest, sdk_errors.InvalidParameterValue) as exc:
            raise DatabricksError(
                DatabricksErrorCode.INVALID_REQUEST,
                "Databricks rejected the request.",
                request_id=request_id,
            ) from exc
        except sdk_errors.DatabricksError as exc:
            raise DatabricksError(
                DatabricksErrorCode.UPSTREAM_ERROR,
                "Databricks returned an upstream error.",
                request_id=request_id,
                retryable=isinstance(exc, (sdk_errors.TemporarilyUnavailable, sdk_errors.InternalError)),
            ) from exc

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        """Convert a generated Databricks SDK model to its wire-format dictionary."""

        if isinstance(value, dict):
            return value
        as_dict = getattr(value, "as_dict", None)
        if callable(as_dict):
            payload = as_dict()
            if isinstance(payload, dict):
                return payload
        raise DatabricksError(
            DatabricksErrorCode.INVALID_RESPONSE,
            "Databricks returned an invalid response object.",
        )

    @staticmethod
    def _conversation_ids(payload: Mapping[str, Any]) -> tuple[str, str]:
        """Read flat or nested conversation and message identifiers."""

        conversation = payload.get("conversation") or {}
        message = payload.get("message") or {}
        conversation_id = (
            payload.get("conversation_id") or conversation.get("conversation_id") or conversation.get("id")
        )
        message_id = payload.get("message_id") or message.get("message_id") or message.get("id")
        return str(conversation_id or ""), str(message_id or "")

    @staticmethod
    def _message_status(payload: Mapping[str, Any]) -> str:
        """Normalize generated SDK enums and wire-format message statuses."""

        status = payload.get("status")
        value = getattr(status, "value", status)
        return str(value or "").upper()

    @staticmethod
    def _parse_attachments(message: Mapping[str, Any]) -> tuple[str, str, str]:
        """Extract text, generated SQL, and query attachment ID from a Genie message."""

        text_parts: list[str] = []
        sql = ""
        attachment_id = ""
        for attachment in message.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            text = attachment.get("text") or {}
            content = text.get("content") if isinstance(text, dict) else None
            if isinstance(content, str) and content.strip():
                text_parts.append(content.strip())
            query = attachment.get("query") or {}
            statement = query.get("query") or query.get("statement") if isinstance(query, dict) else None
            if isinstance(statement, str) and statement.strip() and not sql:
                sql = statement.strip()
                attachment_id = str(attachment.get("attachment_id") or attachment.get("id") or "")
                description = query.get("description")
                if isinstance(description, str) and description.strip():
                    text_parts.append(description.strip())
        return "\n".join(dict.fromkeys(text_parts)), sql, attachment_id

    @staticmethod
    def _normalize_statement_result(
        payload: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> tuple[list[ResultColumn], list[dict[str, Any]], bool]:
        """Normalize the first bounded Genie statement-result chunk."""

        statement = payload.get("statement_response") or payload.get("result") or payload
        if not isinstance(statement, dict):
            raise DatabricksError(
                DatabricksErrorCode.INVALID_RESPONSE,
                "Databricks returned an invalid SQL result.",
            )
        manifest = statement.get("manifest") or {}
        schema = manifest.get("schema") or {}
        raw_columns = schema.get("columns") or []
        columns = [
            ResultColumn(
                name=str(column.get("name") or ""),
                data_type=column.get("type_name") or column.get("type_text"),
            )
            for column in raw_columns
            if isinstance(column, dict) and column.get("name")
        ]
        result = statement.get("result") or {}
        raw_rows = result.get("data_array") or []
        names = [column.name for column in columns]
        rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            if isinstance(raw_row, dict):
                rows.append(dict(raw_row))
            elif isinstance(raw_row, (list, tuple)):
                rows.append({name: value for name, value in zip(names, raw_row, strict=False)})
        upstream_truncated = bool(
            result.get("truncated")
            or manifest.get("truncated")
            or result.get("next_chunk_internal_link")
            or result.get("next_chunk_index") is not None
        )
        return columns, rows[:max_rows], upstream_truncated or len(rows) > max_rows

    @staticmethod
    def _objects_from_sql(sql: str) -> list[str]:
        """Conservatively extract physical table identifiers from FROM and JOIN clauses."""

        objects: list[str] = []
        for match in _SQL_OBJECT_RE.finditer(sql):
            identifier = match.group(1).replace("`", "")
            if identifier not in objects:
                objects.append(identifier)
        return objects
