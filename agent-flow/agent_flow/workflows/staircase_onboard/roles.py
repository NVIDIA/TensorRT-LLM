# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The staircase roles, and which backend and model each one runs on.

The selection is the shared per-role routing in :mod:`agent_flow.agent_runtime`,
which reads an optional ``agents`` block in ``task.yaml``. This module adds one
staircase-specific default: the ``plan_reviewer`` runs on Codex. The other
roles run on Claude Code, because they need its skill system.

The plan reviewer runs on a different model from the plan drafter on purpose.
A reviewer that shares the author's blind spots reviews its own reasoning back
at itself, and the plan phase has no gate, only this review.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent_flow.agent_runtime import AGENTS_FIELD, AgentConfig, resolve_agent_config
from agent_flow.config import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL, BackendKind

#: Every role a turn can be dispatched to. ``runner`` makes no judgement: it
#: relays a command to a scheduler only an agent can reach. It is listed
#: because it is still a turn on some backend.
ROLES: tuple[str, ...] = (
    "plan_drafter",
    "plan_reviewer",
    "smith",
    "assembler",
    "reviewer",
    "auditor",
    "runner",
)

#: The backend a role runs on when the ``agents`` block does not choose one.
DEFAULT_BACKENDS: Mapping[str, BackendKind] = {role: "claude-code" for role in ROLES} | {
    "plan_reviewer": "codex"
}


def _default_model(backend: BackendKind) -> str:
    return CODEX_DEFAULT_MODEL if backend == "codex" else CLAUDE_CODE_DEFAULT_MODEL


def resolve_role_agent(task_data: Mapping[str, Any], role: str) -> AgentConfig:
    """Resolve one role's backend, model, reasoning effort and extra MCP servers."""
    backend = DEFAULT_BACKENDS[role]
    return resolve_agent_config(
        task_data, role, default_backend=backend, default_model=_default_model(backend)
    )


def validate_role_defaults(data: Mapping[str, Any]) -> list[str]:
    """Reject an ``agents.defaults.model`` that would reach two backends.

    The shared resolver applies ``defaults.model`` to every role whose backend
    matches ``defaults.backend``, and when ``defaults.backend`` is absent it
    compares against each role's own default. The roles here do not share a
    default backend, so a bare ``defaults.model`` would send a Claude model
    name to the Codex plan reviewer, or the reverse.
    """
    agents = data.get(AGENTS_FIELD)
    if not isinstance(agents, Mapping):
        return []
    defaults = agents.get("defaults")
    if isinstance(defaults, Mapping) and "model" in defaults and "backend" not in defaults:
        return [
            "'agents.defaults.model' needs 'agents.defaults.backend' here: the "
            "staircase roles do not share a default backend (plan_reviewer runs on "
            "codex, the others on claude-code), so a model name alone would be "
            "sent to both. Set 'agents.defaults.backend', or set the model per role "
            "under 'agents.roles'."
        ]
    return []


def shares_the_drafters_model(task_data: Mapping[str, Any]) -> str:
    """A warning when the plan reviewer resolves to the plan drafter's model, else ``""``.

    Legal, because a host may not reach a second provider. But it is a weaker
    review, so the run says so instead of recording a cross-model review that
    did not happen.
    """
    drafter = resolve_role_agent(task_data, "plan_drafter")
    reviewer = resolve_role_agent(task_data, "plan_reviewer")
    if (drafter.backend, drafter.model) != (reviewer.backend, reviewer.model):
        return ""
    return (
        f"plan_reviewer and plan_drafter both resolve to {reviewer.backend}/{reviewer.model}. "
        "The plan review is then not cross-model: the reviewer shares the drafter's blind "
        "spots. Route plan_reviewer to another backend or model under 'agents.roles' "
        "if this host can reach one."
    )


__all__ = [
    "DEFAULT_BACKENDS",
    "ROLES",
    "resolve_role_agent",
    "shares_the_drafters_model",
    "validate_role_defaults",
]
