# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Databricks ontology response fixtures."""

import json

import pytest


@pytest.fixture
def genie_space() -> dict:
    """Return one serialized Genie Space with useful ontology metadata."""

    definition = {
        "version": 2,
        "config": {
            "sample_questions": [
                {"id": "sample1", "question": ["Show revenue by customer region"]},
            ]
        },
        "data_sources": {
            "tables": [
                {
                    "identifier": "sales.analytics.customers",
                    "description": ["Customer master data by region"],
                    "column_configs": [
                        {"column_name": "customer_id", "description": ["Unique customer identifier"]},
                        {"column_name": "region", "description": ["Customer sales region"]},
                    ],
                }
            ],
            "metric_views": [
                {
                    "identifier": "sales.analytics.revenue_metrics",
                    "description": ["Recognized revenue metrics"],
                }
            ],
        },
        "instructions": {
            "text_instructions": [{"id": "instruction1", "content": ["Use completed fiscal periods"]}],
            "example_question_sqls": [
                {
                    "id": "query1",
                    "question": ["Revenue by region"],
                    "sql": ["SELECT region, SUM(revenue) FROM sales GROUP BY region"],
                }
            ],
            "join_specs": [],
            "sql_snippets": {"measures": [{"id": "measure1", "alias": "total_revenue", "sql": ["SUM(revenue)"]}]},
        },
    }
    return {
        "space_id": "space-1",
        "title": "Sales Analytics",
        "description": "Customer and revenue analytics",
        "serialized_space": json.dumps(definition),
    }
