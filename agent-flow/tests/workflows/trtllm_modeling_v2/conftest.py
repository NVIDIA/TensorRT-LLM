# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the trtllm_modeling_v2 tests.

`test_workflow.py` built the repo, checkpoint and task-spec fixtures first
and everything since has wanted them. Importing them test-module to
test-module works, but every consumer then has to name them as parameters
*and* import them, which reads to a linter as a redefinition and to a reader
as a dependency between test files that is not really there.

Re-exported here instead, so pytest resolves them by name and no test module
imports another.
"""

from __future__ import annotations

from .test_workflow import checkpoint, repo, task_file  # noqa: F401

__all__ = ["checkpoint", "repo", "task_file"]
