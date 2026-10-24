# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The workflow's roles, and which agent each one runs on.

Every role runs on the agent that runs the workflow -- Codex when a Codex
session started the orchestrator, else Claude Code -- unless ``task.yaml``'s
``agents`` block routes it elsewhere (the shared per-role routing in
:mod:`agent_flow.agent_runtime`).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping

from agent_flow.agent_runtime import AGENTS_FIELD, AgentConfig, resolve_agent_config
from agent_flow.config import CLAUDE_CODE_DEFAULT_MODEL, BackendKind

#: Every role a turn can be dispatched to.
ROLES: tuple[str, ...] = (
    "surveyor",
    "kernel_writer",
    "assembler",
    "tuner",
    "planner",
)

#: Set by Codex for every command it runs, and by Claude Code for its own.
CODEX_SESSION_ENV = "CODEX_THREAD_ID"
CLAUDE_SESSION_ENV = "CLAUDECODE"
#: Names a Codex model for the roles that configure none; without it Codex chooses.
CODEX_MODEL_ENV = "CODEX_DEFAULT_MODEL"


def driving_backend(environ: Mapping[str, str] | None = None) -> BackendKind:
    """The agent this process was started from: Codex inside a Codex session, else Claude Code.

    A process inside both (Codex started from Claude Code, say) keeps Claude Code;
    ``agents.defaults.backend`` says which one to use.
    """
    env = os.environ if environ is None else environ
    if env.get(CODEX_SESSION_ENV) and not env.get(CLAUDE_SESSION_ENV):
        return "codex"
    return "claude-code"


def resolve_role_agent(
    task_data: Mapping[str, Any], role: str, *, environ: Mapping[str, str] | None = None
) -> AgentConfig:
    """Resolve one role's backend, model, reasoning effort and extra MCP servers.

    For an in-process turn, which has no parent agent to inherit from: the
    ``agents`` block, else the agent that started this process. A role on
    Codex with no model configured gets ``$CODEX_DEFAULT_MODEL``, else ``""``:
    the model the operator's Codex is set to.
    """
    env = os.environ if environ is None else environ
    backend = driving_backend(env)
    default_model = CLAUDE_CODE_DEFAULT_MODEL if backend == "claude-code" else ""
    agent = resolve_agent_config(
        task_data, role, default_backend=backend, default_model=default_model
    )
    if agent.backend == "codex" and not _model_configured(task_data, role, backend):
        agent = dataclasses.replace(agent, model=env.get(CODEX_MODEL_ENV, ""))
    return agent


def _model_configured(task_data: Mapping[str, Any], role: str, default_backend: str) -> bool:
    """Whether the ``agents`` block names ``role``'s model, as ``resolve_agent_config`` reads it."""
    agents = task_data.get(AGENTS_FIELD)
    if not isinstance(agents, Mapping):
        return False
    defaults = agents.get("defaults")
    defaults = defaults if isinstance(defaults, Mapping) else {}
    roles = agents.get("roles")
    override = roles.get(role) if isinstance(roles, Mapping) else None
    override = override if isinstance(override, Mapping) else {}
    if override.get("model") is not None:
        return True
    inherited_backend = defaults.get("backend", default_backend)
    backend = override.get("backend", inherited_backend)
    return defaults.get("model") is not None and backend == inherited_backend


def configured_agent(task_data: Mapping[str, Any], role: str) -> AgentConfig | None:
    """The agent ``task.yaml`` routes ``role`` to, or None to inherit the driving agent's."""
    agents = task_data.get(AGENTS_FIELD)
    if not isinstance(agents, Mapping):
        return None
    roles = agents.get("roles")
    configured = bool(agents.get("defaults")) or (
        isinstance(roles, Mapping) and bool(roles.get(role))
    )
    return resolve_role_agent(task_data, role) if configured else None


__all__ = [
    "CODEX_SESSION_ENV",
    "ROLES",
    "configured_agent",
    "driving_backend",
    "resolve_role_agent",
]
