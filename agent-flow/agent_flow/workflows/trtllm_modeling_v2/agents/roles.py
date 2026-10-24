# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The workflow's roles, and which agent each one runs on.

Every role runs on the agent that runs the workflow -- its model and its
reasoning effort -- unless ``task.yaml``'s ``agents`` block routes it
elsewhere (the shared per-role routing in :mod:`agent_flow.agent_runtime`).
A different model for one role is an explicit choice, such as a plan
reviewer on another provider for a cross-model review.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent_flow.agent_runtime import AGENTS_FIELD, AgentConfig, resolve_agent_config

#: Every role a turn can be dispatched to.
ROLES: tuple[str, ...] = (
    "plan_drafter",
    "plan_reviewer",
    "smith",
    "assembler",
    "reviewer",
    "tuner",
    "planner",
    "improver",
)


def resolve_role_agent(task_data: Mapping[str, Any], role: str) -> AgentConfig:
    """Resolve one role's backend, model, reasoning effort and extra MCP servers.

    For an in-process turn, which has no parent agent to inherit from: the
    ``agents`` block, else Claude Code at its default model.
    """
    return resolve_agent_config(task_data, role)


def configured_agent(task_data: Mapping[str, Any], role: str) -> AgentConfig | None:
    """The agent ``task.yaml`` routes ``role`` to, or None to inherit the driving agent's."""
    agents = task_data.get(AGENTS_FIELD)
    if not isinstance(agents, Mapping):
        return None
    roles = agents.get("roles")
    configured = bool(agents.get("defaults")) or (
        isinstance(roles, Mapping) and bool(roles.get(role))
    )
    return resolve_agent_config(task_data, role) if configured else None


__all__ = ["ROLES", "configured_agent", "resolve_role_agent"]
