# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent turn transports: one interface, two deliveries."""

from .base import TurnExecutionError, TurnExecutor, TurnOutcome, TurnRequest, TurnTimeout
from .file_dispatch import (
    DEFAULT_TURN_TIMEOUT_S,
    DISPATCH_DIRNAME,
    HEARTBEAT_STALE_S,
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
    FileDispatchTurnExecutor,
    TransportNote,
    job_prefix,
)
from .inprocess import InProcessTurnExecutor, LayerProvider

__all__ = [
    "DEFAULT_TURN_TIMEOUT_S",
    "DISPATCH_DIRNAME",
    "FileDispatchTurnExecutor",
    "HEARTBEAT_STALE_S",
    "InProcessTurnExecutor",
    "LayerProvider",
    "ORCHESTRATOR_EXIT",
    "ORCHESTRATOR_HEARTBEAT",
    "TransportNote",
    "TurnExecutionError",
    "TurnExecutor",
    "TurnOutcome",
    "TurnRequest",
    "TurnTimeout",
    "job_prefix",
]
