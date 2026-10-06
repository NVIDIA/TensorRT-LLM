# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from agent_flow.agent_runtime import resolve_agent_config, validate_agents
from agent_flow.config import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL

ROLES = ("projector", "analyzer", "reporter")


def test_defaults_and_role_overrides_are_resolved_per_role():
    task = {
        "agents": {
            "defaults": {"backend": "claude-code"},
            "roles": {
                "projector": {
                    "backend": "codex",
                    "model": "gpt-6-astra",
                    "reasoning_effort": "ultra",
                }
            },
        }
    }

    projector = resolve_agent_config(task, "projector")
    reporter = resolve_agent_config(task, "reporter")

    assert (projector.backend, projector.model, projector.reasoning_effort) == (
        "codex",
        "gpt-6-astra",
        "ultra",
    )
    assert (reporter.backend, reporter.model) == (
        "claude-code",
        CLAUDE_CODE_DEFAULT_MODEL,
    )


def test_switching_backend_without_model_uses_that_backends_default():
    config = resolve_agent_config(
        {
            "agents": {
                "defaults": {"backend": "claude-code", "model": "claude-test"},
                "roles": {"analyzer": {"backend": "codex"}},
            }
        },
        "analyzer",
    )
    assert (config.backend, config.model) == ("codex", CODEX_DEFAULT_MODEL)


def test_validate_agents_checks_shape_fields_and_roles():
    errors = validate_agents(
        {
            "agents": {
                "defaults": {"backend": "other", "extra_mcp_servers": []},
                "roles": {"optimizer": {"model": ""}},
            }
        },
        ROLES,
    )
    assert len(errors) == 4
    assert any("backend" in error for error in errors)
    assert any("extra_mcp_servers" in error for error in errors)
    assert any("optimizer" in error for error in errors)
