# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The external agent transport, for hosts where the orchestrator cannot run an agent itself.

Every turn runs in process by default (``infra/execution/turns.py``). Some
hosts cannot authenticate the agent CLI the orchestrator spawns (a sandbox
makes the credential store read-only, or a rotating token is shared with a
parent session); there ``--agent-transport external`` writes each turn to
``<workspace>/dispatch/`` (:mod:`.transport`) and an authenticated driver
session answers it (:mod:`.driver`, through ``trtllm-modeling-v2 driver``).
Nothing else moves: the stage machine, decision validation and every gate stay
in the orchestrator's process.
"""

from .transport import ExternalAgentTransport

__all__ = ["ExternalAgentTransport"]
