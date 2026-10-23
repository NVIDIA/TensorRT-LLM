# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner/worker workflow that brings up ModelingV2 models and targets.

A planner writes ``plan.md``: tagged tasks plus an ``## Execution Graph`` of
their dependencies. Workers run the tasks in parallel, each in its own git
worktree, and each writes ``task_result_<id>.md``. The planner runs alongside
them and is woken whenever a worker finishes, to extend or revise the plan.
See ``README.md``.
"""

from typing import Any

from .plan import TAGS, PlanError, PlanSpec, parse_plan
from .workflow import ModelingV2OnboardWorkflow, RunOutcome, WorkflowError

__all__ = [
    "TAGS",
    "ModelingV2OnboardWorkflow",
    "PlanError",
    "PlanSpec",
    "RunOutcome",
    "WorkflowError",
    "main",
    "parse_plan",
]


def __getattr__(name: str) -> Any:
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
