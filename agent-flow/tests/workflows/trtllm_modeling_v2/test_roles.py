# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role backend routing for the workflow's roles.

The routing itself is the shared `agents` block (`agent_flow.agent_runtime`).
What is specific here is the default assignment, in which every role inherits
the driving agent, the validation of a bare `defaults.model`, and the way a
layer is built for a role that resolves to Codex.
"""

from __future__ import annotations

import pytest

from agent_flow.config import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL
from agent_flow.workflows.trtllm_modeling_v2.agents import roles
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import PromptBundle
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import TurnOutcome
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

from .test_task_schema import _valid_spec, _write
from .test_workflow import _workflow


def test_every_prompt_role_is_routable():
    """A role with a system prompt is a role with a backend, and the reverse."""
    assert set(roles.ROLES) == set(PromptBundle.__dataclass_fields__)


def test_without_an_agents_block_every_role_inherits_the_driving_agent():
    for role in roles.ROLES:
        assert roles.configured_agent({}, role) is None, role
        agent = roles.resolve_role_agent({}, role)
        assert (agent.backend, agent.model) == ("claude-code", CLAUDE_CODE_DEFAULT_MODEL), role
        assert agent.reasoning_effort is None


def test_a_role_override_changes_only_that_role():
    task = {
        "agents": {
            "roles": {
                "plan_reviewer": {"backend": "claude-code", "model": "claude-other"},
                "smith": {"backend": "codex", "reasoning_effort": "high"},
            }
        }
    }
    reviewer = roles.resolve_role_agent(task, "plan_reviewer")
    smith = roles.resolve_role_agent(task, "smith")
    assembler = roles.resolve_role_agent(task, "assembler")
    assert (reviewer.backend, reviewer.model) == ("claude-code", "claude-other")
    assert (smith.backend, smith.model, smith.reasoning_effort) == (
        "codex",
        CODEX_DEFAULT_MODEL,
        "high",
    )
    assert (assembler.backend, assembler.model) == ("claude-code", CLAUDE_CODE_DEFAULT_MODEL)


def test_defaults_with_a_backend_reach_every_role_including_the_reviewer():
    task = {"agents": {"defaults": {"backend": "claude-code", "model": "claude-x"}}}
    for role in roles.ROLES:
        agent = roles.resolve_role_agent(task, role)
        assert (agent.backend, agent.model) == ("claude-code", "claude-x"), role


def test_the_spec_accepts_an_agents_block(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["agents"] = {
        "defaults": {"backend": "claude-code", "model": "claude-x"},
        "roles": {"plan_reviewer": {"backend": "codex", "reasoning_effort": "high"}},
    }
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["agents"]["roles"]["plan_reviewer"]["backend"] == "codex"


@pytest.mark.parametrize(
    ("agents", "message"),
    [
        ({"roles": {"wizard": {"backend": "codex"}}}, "unknown role(s) ['wizard']"),
        ({"roles": {"smith": {"backend": "gemini"}}}, "must be 'claude-code' or 'codex'"),
        ({"roles": {"smith": {"temperature": 0}}}, "unknown field(s) ['temperature']"),
        ("codex", "'agents' must be a mapping"),
    ],
)
def test_the_spec_rejects_a_malformed_agents_block(tmp_path, agents, message):
    spec = _valid_spec(tmp_path)
    spec["agents"] = agents
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert message in str(excinfo.value)


def test_only_a_routed_role_names_its_agent():
    task = {"agents": {"roles": {"plan_reviewer": {"backend": "codex"}}}}
    assert roles.configured_agent(task, "plan_reviewer").backend == "codex"
    assert roles.configured_agent(task, "smith") is None


def test_a_claude_role_keeps_its_stop_hook(tmp_path):
    """The hook refuses a turn that wrote nothing; the file is validated when read."""
    workflow = _workflow(tmp_path)
    workflow.task_data = {"agents": {"roles": {"assembler": {"reasoning_effort": "high"}}}}
    layer = workflow._make_agent("assembler", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model, backend.reasoning_effort) == (
        "claude-code",
        CLAUDE_CODE_DEFAULT_MODEL,
        "high",
    )
    assert backend.hooks and backend.hooks["Stop"]
    assert backend.cwd == workflow.workspace
    assert layer.config.required_tools == ()


def test_a_codex_role_gets_no_claude_hooks(tmp_path):
    """Codex refuses per-client Claude SDK hooks, so a Stop hook would fail the turn."""
    workflow = _workflow(tmp_path)
    workflow.task_data = {
        "agents": {
            "roles": {
                "plan_reviewer": {
                    "backend": "codex",
                    "extra_mcp_servers": {"kb": {"type": "http", "url": "https://kb.test/mcp"}},
                }
            }
        }
    }
    layer = workflow._make_agent("plan_reviewer", "system", session_mode="persistent")
    backend = layer.config.backend
    assert (backend.kind, backend.model) == ("codex", CODEX_DEFAULT_MODEL)
    assert backend.hooks is None
    assert backend.extra_mcp_servers == {"kb": {"type": "http", "url": "https://kb.test/mcp"}}
    assert layer.config.required_tools == ()


def test_only_a_routed_role_carries_an_agent_to_the_transport(tmp_path):
    """An unrouted role runs on the driving agent's model; a routed one says which."""

    class Capture:
        def __init__(self):
            self.requests = []

        def run(self, request):
            self.requests.append(request)
            return TurnOutcome(role=request.role, text="", transport="capture")

    workflow = _workflow(tmp_path)
    workflow._turns = Capture()
    workflow.task_data = {"agents": {"roles": {"smith": {"backend": "codex", "model": "m"}}}}
    workflow._run_turn("plan_reviewer", "round 1", "prompt")
    workflow._run_turn("smith", "onboard x", "prompt")
    sent = [(r.role, r.backend, r.model) for r in workflow._turns.requests]
    assert sent == [("plan_reviewer", None, None), ("smith", "codex", "m")]
