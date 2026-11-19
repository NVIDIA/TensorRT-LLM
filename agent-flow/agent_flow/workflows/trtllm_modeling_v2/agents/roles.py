# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route each role to the backend, model and reasoning effort the spec states."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_flow.agent_runtime import AGENTS_FIELD, AgentConfig, resolve_agent_config

#: The skills this workflow ships, one directory each; the run materializes them.
PACKAGED_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def packaged_skills() -> tuple[str, ...]:
    """The names of the skills the run materializes: the only ones its roles load."""
    if not PACKAGED_SKILLS_DIR.is_dir():
        return ()
    return tuple(sorted(p.name for p in PACKAGED_SKILLS_DIR.iterdir() if p.is_dir()))


ROLES: tuple[str, ...] = (
    "surveyor",
    "kernel_writer",
    "assembler",
    "planner",
)


class AgentConfigError(ValueError):
    """A role has no model the spec states; no worker was started."""


def resolve_role_agents(
    task_data: Mapping[str, Any], *, roles: Sequence[str] = ROLES
) -> dict[str, AgentConfig]:
    """Every role from ``agents.defaults`` and ``agents.roles.<role>``; nothing is inherited.

    A role whose model the spec does not state is refused: the run never guesses which model
    its workers run on, and a default would hold on one host and not another.
    """
    missing = [role for role in roles if not _model_configured(task_data, role)]
    if missing:
        raise AgentConfigError(
            f"task.yaml states no model for {', '.join(missing)}: write "
            "`agents.defaults: {backend, model, reasoning_effort}` (or the role's own "
            "`agents.roles.<role>.model`); no worker was started"
        )
    return {role: resolve_agent_config(task_data, role) for role in roles}


def resolve_role_agent(task_data: Mapping[str, Any], role: str) -> AgentConfig:
    """Resolve one role; running workflows pin the full mapping with resolve_role_agents."""
    return resolve_role_agents(task_data, roles=(role,))[role]


def _model_configured(task_data: Mapping[str, Any], role: str) -> bool:
    """Whether the spec states the role's model: its own, or the defaults' on the same backend."""
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
    inherited_backend = defaults.get("backend", "claude-code")
    backend = override.get("backend", inherited_backend)
    return defaults.get("model") is not None and backend == inherited_backend


__all__ = [
    "AgentConfigError",
    "PACKAGED_SKILLS_DIR",
    "ROLES",
    "packaged_skills",
    "resolve_role_agent",
    "resolve_role_agents",
]
