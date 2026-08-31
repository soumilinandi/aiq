<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Snowflake ontology provider

This package adds two NAT tools to AI-Q:

- `snowflake__catalog_search` discovers relevant objects from authorized Snowflake Semantic Views.
- `snowflake__text_to_sql` generates and executes bounded analytical queries through Cortex Agents.

## Prerequisites

- A Snowflake warehouse and least-privilege role with access to the intended Semantic Views and underlying tables.
- A Snowflake programmatic access token exposed as `SNOWFLAKE_PAT`.
- At least one governed Semantic View. An optional allowlist can restrict discovery to selected views.

## Configuration

```yaml
function_groups:
  snowflake:
    _type: snowflake
    account: ${SNOWFLAKE_ACCOUNT}
    user: ${SNOWFLAKE_USER}
    role: ${SNOWFLAKE_ROLE}
    warehouse: ${SNOWFLAKE_WAREHOUSE}
    access_token: ${SNOWFLAKE_PAT}
    access_token_env: SNOWFLAKE_PAT
    semantic_views:
      - ${SNOWFLAKE_SEMANTIC_VIEW}
    catalog_llm: catalog_llm
    catalog_embedder: catalog_embedder
    include: [catalog_search, text_to_sql]

functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: structured_data
        name: Snowflake Structured Data
        description: Search authorized Snowflake semantics and run analytical queries.
        tools: [snowflake]

  data_science_agent:
    _type: data_science_agent
    llm: data_science_llm
    ontology_provider:
      provider: snowflake
      catalog_tools: [snowflake__catalog_search]
      analytical_tools: [snowflake__text_to_sql]

  intent_classifier:
    _type: context_aware_intent_router
    llm: intent_llm
    catalog_tool: snowflake__catalog_search
    catalog_source_id: structured_data
```

Omit `semantic_views` to discover Semantic Views visible to the configured identity. In API deployments,
`access_token_env` resolves the token when the tool runs; credentials are not included in tool responses or logs.

## Validation

```bash
uv run pytest sources/snowflake_ontology/tests -m "not integration"
```
