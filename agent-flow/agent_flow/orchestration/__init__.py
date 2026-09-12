# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-agnostic concurrent DAG execution engine.

This package holds the orchestration core: the pure graph data model plus its
parser (:mod:`.graph`), the git-free workspace-isolation abstraction
(:mod:`.isolation`) — the :class:`IsolationProvider` Protocol plus a built-in
:class:`NoOpIsolation` — and a concurrent ready-set scheduler with
checkpoint/resume and a fan-in ``prepare`` hook (:mod:`.scheduler`). It stays
independent of agent, backend, ``layers``, and any concrete-isolation code: the
opt-in git provider lives in :mod:`agent_flow.git_worktree`, which the core never
imports.
"""

from .graph import (
    ExecutionGraph,
    ExecutionGraphError,
    Isolation,
    Node,
    NodeKind,
    NodeState,
    parse_execution_graph,
)
from .isolation import IsolationError, IsolationProvider, NoOpIsolation
from .scheduler import GraphResult, GraphState, NodeOutcome, NodeScheduler, OnTransition, RunNode

__all__ = [
    "ExecutionGraph",
    "ExecutionGraphError",
    "GraphResult",
    "GraphState",
    "Isolation",
    "IsolationError",
    "IsolationProvider",
    "Node",
    "NodeKind",
    "NodeOutcome",
    "NodeScheduler",
    "NodeState",
    "NoOpIsolation",
    "OnTransition",
    "RunNode",
    "parse_execution_graph",
]
