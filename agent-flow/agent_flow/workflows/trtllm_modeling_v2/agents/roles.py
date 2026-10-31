# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route roles to explicit models or the model of the launching agent session."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_flow.agent_runtime import AGENTS_FIELD, AgentConfig, resolve_agent_config
from agent_flow.config import BackendKind

from .session_model import CODEX_SESSION_ENV, ModelSelectionError, session_model

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
    "tuner",
    "planner",
)

CLAUDE_SESSION_ENV = "CLAUDECODE"
CODEX_MODEL_ENV = "CODEX_DEFAULT_MODEL"
CLAUDE_MODEL_ENV = "CLAUDE_CODE_DEFAULT_MODEL"


def driving_backend(environ: Mapping[str, str] | None = None) -> BackendKind:
    """Identify the launching harness; a plain terminal defaults to Claude Code."""
    env = os.environ if environ is None else environ
    if env.get(CODEX_SESSION_ENV) and env.get(CLAUDE_SESSION_ENV):
        raise ModelSelectionError(
            "Both Claude and Codex session markers are present; set agents.defaults.backend "
            "or each role's backend explicitly."
        )
    return "codex" if env.get(CODEX_SESSION_ENV) else "claude-code"


def resolve_role_agents(
    task_data: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    roles: Sequence[str] = ROLES,
) -> dict[str, AgentConfig]:
    """Resolve all roles against one launcher snapshot, preserving explicit overrides."""
    env = dict(os.environ if environ is None else environ)
    block = task_data.get(AGENTS_FIELD) or {}
    defaults = block.get("defaults") or {}
    overrides = block.get("roles") or {}
    if defaults.get("backend"):
        backend = defaults["backend"]
    else:
        try:
            backend = driving_backend(env)
        except ModelSelectionError:
            if defaults.get("model") is not None or not all(
                (overrides.get(role) or {}).get("backend") for role in roles
            ):
                raise
            backend = "claude-code"  # Every role overrides it; no global model to inherit.
    models: dict[BackendKind, str] = {}
    resolved = {}
    for role in roles:
        agent = resolve_agent_config(task_data, role, default_backend=backend, default_model="")
        if not _model_configured(task_data, role, backend):
            if agent.backend not in models:
                key = CODEX_MODEL_ENV if agent.backend == "codex" else CLAUDE_MODEL_ENV
                model = env.get(key, "").strip()
                if not model:
                    try:
                        model = session_model(agent.backend, env)
                    except ModelSelectionError as exc:
                        raise ModelSelectionError(
                            f"Cannot inherit the model for {role} ({agent.backend}): {exc}. "
                            "Set agents.defaults.model or agents.roles."
                            f"{role}.model in task.yaml, or ${key}; no worker was started."
                        ) from exc
                models[agent.backend] = model
            agent = dataclasses.replace(agent, model=models[agent.backend])
        resolved[role] = agent
    return resolved


def resolve_role_agent(
    task_data: Mapping[str, Any], role: str, *, environ: Mapping[str, str] | None = None
) -> AgentConfig:
    """Resolve one role; running workflows pin the full mapping with resolve_role_agents."""
    return resolve_role_agents(task_data, environ=environ, roles=(role,))[role]


def _model_configured(task_data: Mapping[str, Any], role: str, default_backend: str) -> bool:
    """Whether the role inherits an explicit model from a matching backend."""
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


__all__ = [
    "CODEX_SESSION_ENV",
    "PACKAGED_SKILLS_DIR",
    "ROLES",
    "driving_backend",
    "packaged_skills",
    "resolve_role_agent",
    "resolve_role_agents",
]
