<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Databricks ontology provider

This package adds two NAT tools to AI-Q:

- `databricks__catalog_search` discovers relevant objects from authorized Genie Spaces and Unity Catalog metadata.
- `databricks__text_to_sql` generates and executes bounded analytical queries through Genie.

## Prerequisites

- A Databricks SQL warehouse and token exposed as `DATABRICKS_TOKEN`.
- A Genie Space containing the intended Unity Catalog tables, relationships, instructions, and permissions.
- Access to the Space, warehouse, and referenced Unity Catalog objects for the configured identity.

## Configuration

```yaml
function_groups:
  databricks:
    _type: databricks
    workspace_url: https://${DATABRICKS_SERVER_HOSTNAME}
    access_token: ${DATABRICKS_TOKEN}
    access_token_env: DATABRICKS_TOKEN
    space_ids:
      - ${DATABRICKS_GENIE_SPACE_ID}
    catalog_llm: catalog_llm
    catalog_embedder: catalog_embedder
    include: [catalog_search, text_to_sql]

functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: structured_data
        name: Databricks Structured Data
        description: Search authorized Databricks semantics and run analytical queries.
        tools: [databricks]

  data_science_agent:
    _type: data_science_agent
    llm: data_science_llm
    ontology_provider:
      provider: databricks
      catalog_tools: [databricks__catalog_search]
      analytical_tools: [databricks__text_to_sql]

  intent_classifier:
    _type: context_aware_intent_router
    llm: intent_llm
    catalog_tool: databricks__catalog_search
    catalog_source_id: structured_data
```

Omit `space_ids` to discover Genie Spaces visible to the configured identity. In API deployments,
`access_token_env` resolves the token when the tool runs; credentials are not included in tool responses or logs.

## Validation

```bash
uv run pytest sources/databricks_ontology/tests
```
