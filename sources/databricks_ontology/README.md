<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Databricks ontology provider

This NAT function group exposes:

- `databricks__catalog_search`: extracts business entities from a question and ranks authorized Genie Space and
  Unity Catalog metadata with embeddings.
- `databricks__text_to_sql`: asks an authorized Genie Space an analytical question, polls it asynchronously, and
  returns either bounded query evidence or an explicit clarification outcome. Failed Genie messages receive at most
  one targeted repair in the same conversation.

Catalog metadata is authorized for every invocation. `space_ids` optionally restricts the scope; an empty list
discovers Spaces visible to the configured token. Document embeddings are reused through a bounded process-local cache
keyed only by opaque content fingerprints. Public candidates use the provider-neutral ID, label, attribute, term,
score, and capability contract. Bounded discovery and enrichment are reported through `truncated` and `warnings`.
Tool responses expose normalized status, request evidence, and total latency; provider and capability come from the
registered NAT tool identity. Genie conversation and message identifiers remain internal provider diagnostics.

Execution requires catalog candidate IDs. AI-Q resolves them again against current authorized Space metadata and
rejects stale, unknown, and mixed-Space selections. The rehydrated definition selects and grounds a single Space;
Genie independently selects objects within that Space for SQL generation. `space_ids` is shared by catalog search and
text-to-SQL.

## Prerequisites

Before starting AI-Q, a Databricks administrator or data owner must:

1. Provide an accessible Databricks SQL warehouse.
2. Create at least one Genie Space, attach it to that warehouse, and add the relevant Unity Catalog tables and
   instructions.
3. Grant the configured identity access to the SQL warehouse, Genie Space, and underlying Unity Catalog objects.
4. For deployment-scoped token mode, create a personal access token and expose it as `DATABRICKS_TOKEN`.

AI-Q does not create Genie Spaces or grant Databricks access. `space_ids` can explicitly scope the available Spaces;
when omitted, the integration discovers Spaces already visible to the configured identity.

## Preparing semantic context

Do not configure a Genie Space with table identifiers alone. During provisioning, the data owner should:

1. Add descriptions to Unity Catalog tables and columns, including business meaning, units, identifiers, and time
   semantics. These comments remain governed by Unity Catalog and are available to both Genie and catalog search.
2. Add the relevant tables to the Space and describe each table's grain, primary key, important facts, and time fields.
3. Define explicit Genie join relationships and add instructions for domain rules, allowed interpretations, and any
   required filtering or aggregation behavior.
4. Add verified question/SQL examples only when they are reviewed and reusable; they are quality improvements, not a
   substitute for catalog metadata and relationships.

For consistent behavior across ontology providers, generate Snowflake semantic views and Databricks Genie metadata
from the same provider-neutral semantic manifest when possible. AI-Q reads the metadata visible to the active identity;
it does not mutate provider catalogs during a user request.

## Configuration

Install `aiq-databricks-ontology` for catalog search and Genie analytics.

Supply deployment-scoped credentials through environment-backed configuration:

```yaml
function_groups:
  databricks:
    _type: databricks
    workspace_url: ${DATABRICKS_HOST}
    catalog_llm: catalog_llm
    catalog_embedder: catalog_embedder
    access_token: ${DATABRICKS_TOKEN}
    access_token_env: DATABRICKS_TOKEN
    space_ids:
      - ${DATABRICKS_AI_FACTORY_SPACE_ID}
    include: [catalog_search, text_to_sql]
```

Set `access_token_env` in FastAPI profiles so tools resolve the token from the server environment at invocation time;
direct programmatic configurations may omit it and use `access_token` normally.

The current token mode is deployment-scoped. Per-user production authorization must resolve the signed-in user's
Databricks identity at tool invocation time and preserve Databricks authorization and audit enforcement. For local
development, register the function group in `data_source_registry` with `requires_auth: false`. The ontology
configuration can then assign its capabilities without provider-specific routing logic:

```yaml
ontology_provider:
  provider: databricks
  catalog_tools: [databricks__catalog_search]
  analytical_tools: [databricks__text_to_sql]
```
