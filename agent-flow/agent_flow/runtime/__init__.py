# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution substrate shared by every workflow.

Two questions live here, and neither of them is a workflow's business:

* **how does one agent turn get delivered and answered** — in this process
  through an ``AgentLayer``, or through a filesystem mailbox a separate
  authenticated session drains;
* **where and how does one command run** — a local subprocess, an ``srun``
  step, or a file dropped into a spool a long-lived allocation watches.

A workflow states *what* it needs run and *what* it needs proven. It should
not carry a branch per cluster protocol, and it should not have to change
when a credential store turns read-only. Everything in this package is
therefore free of workflow vocabulary: nothing here knows what a catalog
entry, a smith or a gate is.

A third concern, older than the other two, already lived at this import
path: running async work on a background loop for a synchronous caller. It
is the same kind of thing — substrate a workflow uses without describing —
so it stayed, as :mod:`~agent_flow.runtime.portal`, and
``from agent_flow.runtime import PortalRunner`` still resolves.
"""

from .commands import (
    CommandExecutor,
    CommandSpec,
    DelegatedCommandExecutor,
    ExecutionCapabilities,
    ExecutionRecord,
    LocalCommandExecutor,
    SlurmStepExecutor,
    SpoolCommandExecutor,
    SubmissionRenderer,
)
from .portal import PortalRunner, build_request
from .turns import (
    DEFAULT_TURN_TIMEOUT_S,
    FileDispatchTurnExecutor,
    InProcessTurnExecutor,
    TurnExecutionError,
    TurnExecutor,
    TurnOutcome,
    TurnRequest,
    TurnTimeout,
)

__all__ = [
    "CommandExecutor",
    "CommandSpec",
    "DEFAULT_TURN_TIMEOUT_S",
    "DelegatedCommandExecutor",
    "SubmissionRenderer",
    "ExecutionCapabilities",
    "ExecutionRecord",
    "FileDispatchTurnExecutor",
    "InProcessTurnExecutor",
    "LocalCommandExecutor",
    "PortalRunner",
    "SlurmStepExecutor",
    "SpoolCommandExecutor",
    "TurnExecutionError",
    "TurnExecutor",
    "TurnOutcome",
    "TurnRequest",
    "TurnTimeout",
    "build_request",
]
