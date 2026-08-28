<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Snowflake ontology provider

This provider exposes:

- `snowflake__catalog_search`, which ranks bounded, typed objects from authorized native Semantic Views.
- `snowflake__text_to_sql`, which calls `POST /api/v2/cortex/agent:run` with only catalog-selected
  `cortex_analyst_text_to_sql` tools and lets Snowflake execute SQL through `system_execute_sql`.

Agent Run owns its internal Analyst and SQL trajectory. AI-Q does not count or stop Cortex-internal tool uses or SQL
executions, and it does not reinterpret successful provider SQL for semantic correctness. A run with no usable
successful execution gets one fresh Cortex repair run; clarification and valid empty-result outcomes do not. The
client retains bounded `tool_use` and `tool_result` events internally and returns every safely normalized successful
SQL result set from completed runs. If one successful execution cannot be normalized, AI-Q records that rejection in
warnings without
discarding other successful result sets. Callers can therefore distinguish repaired, unrepaired, ambiguous, empty,
and timed-out outcomes without assuming the last execution is authoritative.
If the orchestration deadline expires after a successful SQL result event has arrived, AI-Q closes the stream but
normalizes and returns the completed result with a timeout warning. A deadline with no successful result remains a
typed timeout failure.

## Snowflake prerequisites

An administrator must provide a dedicated least-privilege role and warehouse, grant only the intended databases,
schemas, Semantic Views, and underlying tables, and configure warehouse resource monitors outside AI-Q. The role is
the authorization and read-only boundary. Because Agent Run's internal `system_execute_sql` executes inside Snowflake
before AI-Q receives it, post-execution SQL classification is not treated as an authorization control.

The existing deployment-scoped PAT flow is unchanged. Configure `SNOWFLAKE_PAT`; this integration does not add
per-user Snowflake identities or replace authentication architecture.

## Configuration

```yaml
function_groups:
  snowflake:
    _type: snowflake
    account: ${SNOWFLAKE_ACCOUNT}
    user: ${SNOWFLAKE_LOGIN_NAME}
    role: ${SNOWFLAKE_ROLE}
    warehouse: ${SNOWFLAKE_WAREHOUSE}
    access_token: ${SNOWFLAKE_PAT}
    access_token_env: SNOWFLAKE_PAT
    semantic_views:
      - ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA}.AUTHORIZED_VIEW
    catalog_llm: catalog_llm
    catalog_embedder: catalog_embedder
    query_timeout_seconds: 120
    orchestration_timeout_seconds: 180
    orchestration_token_budget: 32000
    default_max_rows: 1000
    max_result_columns: 100
    max_cell_bytes: 16384
    max_result_bytes: 1000000
    semantic_cache_ttl_seconds: 300
    semantic_load_concurrency: 4
    expose_sample_values: false
    include: [catalog_search, text_to_sql]
```

Set `access_token_env` in FastAPI profiles so the tool resolves the PAT from the server environment when it runs.
NAT redacts `SecretStr` values while materializing worker configuration; the environment reference prevents that
redacted representation from being sent to Snowflake. Direct programmatic configurations may omit
`access_token_env` and use `access_token` normally.

`semantic_views` is an optional deployment allowlist. Without it, discovery uses the narrowest requested database
scope and asks Snowflake for `max_semantic_views + 1` rows so truncation is explicit. The optional evaluation-only
`database_name` filter matches a Semantic View's database or schema component; it is not a fuzzy domain selector.

Parsed metadata is cached by account, user, role, authorized scope, Semantic View FQN, and `last_altered` version with
a bounded TTL. Public catalog results hide physical table names; sample values are hidden unless explicitly enabled.
Descriptions, synonyms, samples, and semantic instructions are treated as bounded untrusted catalog data.

## Result contract and limits

Catalog candidate IDs are opaque. Text-to-SQL requires IDs returned by catalog search and resolves them again against
current authorized metadata. Each execution call accepts objects from one Semantic View. Resolved descriptions,
definitions, relationships, and expressions—not model-copied metadata—form the bounded Cortex context. Stale,
unknown, and mixed-view selections are rejected.

Text-to-SQL results contain an ordered column schema and positional rows, so duplicate aliases do not overwrite data.
The response includes `result_sets`, with SQL, columns, rows, query ID, and truncation metadata for every successful
execution that passes response normalization. Cortex attempts, trajectory, provenance, and applied budgets remain
internal diagnostics. The top-level SQL and rows remain the last retained successful result, but callers must compare
all result-set SQL
against the original population, filters, time window, grain, and requested fields. The response also includes
clarification suggestions, selected objects, request/query IDs, warnings, repair count, and normalized total latency.

Row limits bound returned context, not warehouse scans or intermediate work. Warehouse work is bounded separately by
the Snowflake statement timeout, Agent `query_timeout`, the dedicated role/warehouse, and external resource monitors.
Columns, cell bytes, total serialized result bytes, orchestration time, and tokens have independent request-level
limits. Cortex owns its internal tool and SQL execution counts. Client cancellation closes the active Agent stream;
the query timeout remains the hard warehouse execution bound. AI-Q performs at most one additional Agent Run when the
initial provider run fails to return a usable successful execution.

## Validation

```bash
uv run pytest sources/snowflake_ontology/tests -m "not integration"
```
