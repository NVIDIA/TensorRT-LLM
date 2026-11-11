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


ROLES: tuple[str, ...] = ("surveyor", "kernel_writer", "assembler")


def resolve_role_agents(
    task_data: Mapping[str, Any], *, roles: Sequence[str] = ROLES
) -> dict[str, AgentConfig]:
    """Every role from ``agents.defaults`` and ``agents.roles.<role>``; nothing is inherited."""
    return {role: resolve_agent_config(task_data, role) for role in roles}


def unstated_models(task_data: Mapping[str, Any], roles: Sequence[str] = ROLES) -> list[str]:
    """The roles whose model the spec does not state: its own, or the defaults' on their backend.

    The run never guesses which model its workers run on: a default would hold on one host
    and not another. The spec's loader refuses a spec that leaves one out.
    """
    agents = task_data.get(AGENTS_FIELD)
    agents = agents if isinstance(agents, Mapping) else {}
    defaults = agents.get("defaults")
    defaults = defaults if isinstance(defaults, Mapping) else {}
    overrides = agents.get("roles")
    overrides = overrides if isinstance(overrides, Mapping) else {}
    inherited = defaults.get("backend", "claude-code")
    unstated = []
    for role in roles:
        own = overrides.get(role)
        own = own if isinstance(own, Mapping) else {}
        if own.get("model") is None and (
            defaults.get("model") is None or own.get("backend", inherited) != inherited
        ):
            unstated.append(role)
    return unstated


__all__ = [
    "PACKAGED_SKILLS_DIR",
    "ROLES",
    "packaged_skills",
    "resolve_role_agents",
    "unstated_models",
]
