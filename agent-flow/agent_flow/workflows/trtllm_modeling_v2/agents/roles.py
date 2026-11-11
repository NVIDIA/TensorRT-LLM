# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route each role to the backend, model and reasoning effort the spec states."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_flow.agent_runtime import AGENTS_FIELD, AgentConfig, resolve_agent_config

#: The skills this workflow ships, one directory each; the run materializes them.
PACKAGED_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


#: Where the run's skills go under the workspace, each agent session's cwd:
#: Claude Code reads `.claude/skills`, Codex `.agents/skills`.
SKILL_DIRS: tuple[str, ...] = (".claude/skills", ".agents/skills")


def materialize_skills(workspace: Path) -> None:
    """Copy the packaged skills into ``workspace``, where every backend finds them."""

    def files(root: Path) -> dict[Path, bytes]:
        found = (p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        return {p.relative_to(root): p.read_bytes() for p in found}

    skills = sorted(p for p in PACKAGED_SKILLS_DIR.iterdir() if p.is_dir())
    for destination in (workspace / name for name in SKILL_DIRS):
        destination.mkdir(parents=True, exist_ok=True)
        for skill in skills:
            target = destination / skill.name
            # A resumed run: the copy is current, and the directory may be read-only to
            # this process (a sandbox protects `.claude/skills`).
            if not target.is_dir() or files(skill) != files(target):
                shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(skill, target)


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
