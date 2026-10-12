# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role backend routing for the staircase roles.

The routing itself is the shared `agents` block (`agent_flow.agent_runtime`).
What is specific here is the default assignment, in which the plan reviewer is
on Codex, the validation of a bare `defaults.model` that this assignment needs,
and the way a layer is built for a role that resolves to Codex.
"""

from __future__ import annotations

import pytest

from agent_flow.config import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL
from agent_flow.workflows.staircase_onboard import roles, task_schema
from agent_flow.workflows.staircase_onboard.prompts import PromptBundle

from .test_task_schema import _valid_spec, _write
from .test_workflow import _workflow


def test_every_prompt_role_is_routable():
    """A role with a system prompt is a role with a backend, and the reverse."""
    assert set(roles.ROLES) == set(PromptBundle.__dataclass_fields__)


def test_without_an_agents_block_the_plan_reviewer_runs_on_codex():
    reviewer = roles.resolve_role_agent({}, "plan_reviewer")
    assert (reviewer.backend, reviewer.model) == ("codex", CODEX_DEFAULT_MODEL)
    for role in set(roles.ROLES) - {"plan_reviewer"}:
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
        ({"roles": {"planner": {"backend": "codex"}}}, "unknown role(s) ['planner']"),
        ({"roles": {"smith": {"backend": "gemini"}}}, "must be 'claude-code' or 'codex'"),
        ({"roles": {"smith": {"temperature": 0}}}, "unknown field(s) ['temperature']"),
        ({"defaults": {"model": "claude-x"}}, "needs 'agents.defaults.backend'"),
        ("codex", "'agents' must be a mapping"),
    ],
)
def test_the_spec_rejects_a_malformed_agents_block(tmp_path, agents, message):
    spec = _valid_spec(tmp_path)
    spec["agents"] = agents
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert message in str(excinfo.value)


def test_a_same_model_review_is_reported_not_hidden():
    same = {"agents": {"defaults": {"backend": "claude-code"}}}
    assert "not cross-model" in roles.shares_the_drafters_model(same)
    assert roles.shares_the_drafters_model({}) == ""


def test_a_claude_role_keeps_its_stop_hook(tmp_path):
    workflow = _workflow(tmp_path)
    # Pinned: the default depends on whether this host has an enterprise MCP config.
    workflow._decision_files = False
    workflow.task_data = {"agents": {"roles": {"assembler": {"reasoning_effort": "high"}}}}
    layer = workflow._make_agent(
        "assembler", "system", required_tools=["append_assembler_progress"]
    )
    backend = layer.config.backend
    assert (backend.kind, backend.model, backend.reasoning_effort) == (
        "claude-code",
        CLAUDE_CODE_DEFAULT_MODEL,
        "high",
    )
    assert backend.hooks and backend.hooks["Stop"]
    assert backend.cwd == workflow.workspace
    assert layer.config.required_tools == ()


def test_a_codex_role_requires_its_tool_without_claude_hooks(tmp_path):
    """Codex refuses per-client Claude SDK hooks, so a Stop hook would fail the turn."""
    workflow = _workflow(tmp_path)
    workflow._decision_files = False
    workflow.task_data = {
        "agents": {
            "roles": {
                "plan_reviewer": {
                    "extra_mcp_servers": {"kb": {"type": "http", "url": "https://kb.test/mcp"}}
                }
            }
        }
    }
    layer = workflow._make_agent(
        "plan_reviewer",
        "system",
        required_tools=["append_plan_reviewer_progress"],
        session_mode="persistent",
    )
    backend = layer.config.backend
    assert (backend.kind, backend.model) == ("codex", CODEX_DEFAULT_MODEL)
    assert backend.hooks is None
    assert backend.extra_mcp_servers == {"kb": {"type": "http", "url": "https://kb.test/mcp"}}
    assert layer.config.required_tools == ("append_plan_reviewer_progress",)


def test_a_codex_role_with_a_decision_file_requires_no_tool(tmp_path):
    """Codex has no `Write` tool; the decision file is validated when it is read."""
    workflow = _workflow(tmp_path)
    workflow._decision_files = True
    layer = workflow._make_agent(
        "plan_reviewer", "system", required_tools=["append_plan_reviewer_progress"]
    )
    assert layer.config.backend.hooks is None
    assert layer.config.required_tools == ()


def test_every_turn_carries_the_roles_backend_whatever_the_transport(tmp_path):
    """The external transport hands the turn to a driver that chooses its own model."""
    from agent_flow.runtime import TurnOutcome

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
    assert sent == [("plan_reviewer", "codex", CODEX_DEFAULT_MODEL), ("smith", "codex", "m")]
