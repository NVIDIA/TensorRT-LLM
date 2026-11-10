# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""System prompt for the planner: asked only when the fixed pipeline cannot go on by itself."""

from __future__ import annotations

from ..decisions import ending_section
from ._common import tidy
from ._render import render_template

#: How many tasks one ``REPAIR`` may add (``run/pipeline.py`` enforces the same number).
MAX_REPAIR_TASKS = 3

SYSTEM_PROMPT = tidy(
    render_template(
        "planner.md.j2",
        MAX_REPAIR_TASKS=MAX_REPAIR_TASKS,
        ending=ending_section("planner", heading="Your decision"),
    )
)

__all__ = ["MAX_REPAIR_TASKS", "SYSTEM_PROMPT"]
