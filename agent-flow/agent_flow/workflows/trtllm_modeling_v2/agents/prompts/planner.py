# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""System prompt for the Planner: orders the work as a task graph."""

from __future__ import annotations

from ...operators.coverage import CATALOG_TESTS_DIR, MODELING_V2_PACKAGE
from ...run.graph import CHECK_KIND, MAX_GOAL_CHARS, WORKER_TAGS
from ..decisions import ending_section
from ._common import tidy
from ._render import render_template
from .tags import TAG_SUMMARIES

_CHECK_SUMMARY = "contract checks and nothing else, run by the orchestrator with no agent turn"

_KIND_ROWS = "\n".join(
    [f"- `{tag}`: {TAG_SUMMARIES[tag]}" for tag in WORKER_TAGS]
    + [f"- `{CHECK_KIND}`: {_CHECK_SUMMARY}"]
)

SYSTEM_PROMPT = tidy(
    render_template(
        "planner.md.j2",
        MODELING_V2_PACKAGE=MODELING_V2_PACKAGE,
        CATALOG_TESTS_DIR=CATALOG_TESTS_DIR,
        MAX_GOAL_CHARS=MAX_GOAL_CHARS,
        kind_rows=_KIND_ROWS,
        ending=ending_section("planner", heading="Your decision"),
    )
)

__all__ = ["SYSTEM_PROMPT"]
