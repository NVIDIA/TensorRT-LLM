# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role model and backend selection shared by performance workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL, BackendKind

AGENTS_FIELD = "agents"
_FIELDS = frozenset({"backend", "model", "reasoning_effort", "extra_mcp_servers"})
_BACKENDS = frozenset({"claude-code", "codex"})


@dataclass(frozen=True)
class AgentConfig:
    """Resolved backend configuration for one workflow role."""

    backend: BackendKind
    model: str
    reasoning_effort: str | None = None
    extra_mcp_servers: dict[str, Any] | None = None


def validate_agents(data: Mapping[str, Any], roles: tuple[str, ...]) -> list[str]:
    """Validate the optional ``agents`` block for a workflow's roles."""
    if AGENTS_FIELD not in data:
        return []

    agents = data[AGENTS_FIELD]
    if not isinstance(agents, Mapping):
        return ["'agents' must be a mapping"]

    errors: list[str] = []
    unknown = set(agents) - {"defaults", "roles"}
    if unknown:
        errors.append(f"'agents' has unknown field(s) {sorted(unknown)}")

    _validate_block(agents.get("defaults", {}), "agents.defaults", errors)
    role_blocks = agents.get("roles", {})
    if not isinstance(role_blocks, Mapping):
        errors.append("'agents.roles' must be a mapping")
        return errors

    unknown_roles = set(role_blocks) - set(roles)
    if unknown_roles:
        errors.append(f"'agents.roles' has unknown role(s) {sorted(unknown_roles)}")
    for role, block in role_blocks.items():
        _validate_block(block, f"agents.roles.{role}", errors)
    return errors


def _validate_block(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"'{path}' must be a mapping")
        return

    unknown = set(value) - _FIELDS
    if unknown:
        errors.append(f"'{path}' has unknown field(s) {sorted(unknown)}")
    backend = value.get("backend")
    if backend is not None and backend not in _BACKENDS:
        errors.append(f"'{path}.backend' must be 'claude-code' or 'codex'")
    for field in ("model", "reasoning_effort"):
        item = value.get(field)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            errors.append(f"'{path}.{field}' must be a non-empty string")
    servers = value.get("extra_mcp_servers")
    if servers is not None and not isinstance(servers, Mapping):
        errors.append(f"'{path}.extra_mcp_servers' must be a mapping")


def resolve_agent_config(
    data: Mapping[str, Any],
    role: str,
    *,
    default_backend: BackendKind = "claude-code",
    default_model: str = CLAUDE_CODE_DEFAULT_MODEL,
) -> AgentConfig:
    """Resolve one role from workflow defaults and its optional override."""
    agents = data.get(AGENTS_FIELD, {})
    if not isinstance(agents, Mapping):
        return AgentConfig(default_backend, default_model)
    defaults = agents.get("defaults", {})
    defaults = defaults if isinstance(defaults, Mapping) else {}
    role_blocks = agents.get("roles", {})
    role_blocks = role_blocks if isinstance(role_blocks, Mapping) else {}
    override = role_blocks.get(role, {})
    override = override if isinstance(override, Mapping) else {}

    backend = override.get("backend", defaults.get("backend", default_backend))
    model = override.get("model")
    if model is None:
        inherited_model = defaults.get("model")
        inherited_backend = defaults.get("backend", default_backend)
        if inherited_model is not None and backend == inherited_backend:
            model = inherited_model
        elif backend == default_backend and "backend" not in defaults and "backend" not in override:
            model = default_model
        else:
            model = CODEX_DEFAULT_MODEL if backend == "codex" else CLAUDE_CODE_DEFAULT_MODEL

    servers = override.get("extra_mcp_servers", defaults.get("extra_mcp_servers"))
    return AgentConfig(
        backend=backend,
        model=model,
        reasoning_effort=override.get("reasoning_effort", defaults.get("reasoning_effort")),
        extra_mcp_servers=dict(servers) if isinstance(servers, Mapping) else None,
    )
