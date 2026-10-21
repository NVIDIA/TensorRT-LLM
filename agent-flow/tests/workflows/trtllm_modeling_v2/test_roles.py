# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role backend routing for the workflow's roles."""

from __future__ import annotations

import pytest

from agent_flow.config import CLAUDE_CODE_DEFAULT_MODEL
from agent_flow.workflows.trtllm_modeling_v2.agents import roles
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import PromptBundle
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import TurnOutcome
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

from .test_task_schema import _valid_spec, _write
from .test_workflow import _workflow


def test_every_prompt_role_is_routable():
    assert set(roles.ROLES) == set(PromptBundle.__dataclass_fields__)


def test_without_an_agents_block_every_role_inherits_the_driving_agent():
    for role in roles.ROLES:
        assert roles.configured_agent({}, role) is None, role
        agent = roles.resolve_role_agent({}, role)
        assert (agent.backend, agent.model) == ("claude-code", CLAUDE_CODE_DEFAULT_MODEL), role
        assert agent.reasoning_effort is None


def test_a_run_started_from_codex_runs_its_roles_on_codex():
    """With no model configured, a Codex role runs the model the operator's Codex is set to."""
    codex = {"CODEX_THREAD_ID": "t-1"}
    for role in roles.ROLES:
        agent = roles.resolve_role_agent({}, role, environ=codex)
        assert (agent.backend, agent.model) == ("codex", ""), role
    named = roles.resolve_role_agent({}, "planner", environ={**codex, "CODEX_DEFAULT_MODEL": "m"})
    assert named.model == "m"
    # Inside both (Codex started from Claude Code) the block decides; without one, Claude Code.
    both = roles.resolve_role_agent({}, "planner", environ={**codex, "CLAUDECODE": "1"})
    assert both.backend == "claude-code"
    pinned = {"agents": {"defaults": {"backend": "codex", "model": "gpt-x"}}}
    assert roles.resolve_role_agent(pinned, "planner", environ={}).model == "gpt-x"


def test_a_role_override_changes_only_that_role():
    task = {
        "agents": {
            "roles": {
                "planner": {"backend": "claude-code", "model": "claude-other"},
                "kernel_writer": {"backend": "codex", "reasoning_effort": "high"},
            }
        }
    }
    planner = roles.resolve_role_agent(task, "planner")
    kernel_writer = roles.resolve_role_agent(task, "kernel_writer")
    assembler = roles.resolve_role_agent(task, "assembler")
    assert (planner.backend, planner.model) == ("claude-code", "claude-other")
    assert (kernel_writer.backend, kernel_writer.model, kernel_writer.reasoning_effort) == (
        "codex",
        "",
        "high",
    )
    assert (assembler.backend, assembler.model) == ("claude-code", CLAUDE_CODE_DEFAULT_MODEL)


def test_the_spec_accepts_an_agents_block(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["agents"] = {
        "defaults": {"backend": "claude-code", "model": "claude-x"},
        "roles": {"planner": {"backend": "codex", "reasoning_effort": "high"}},
    }
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["agents"]["roles"]["planner"]["backend"] == "codex"


def test_the_spec_rejects_an_unknown_role(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["agents"] = {"roles": {"wizard": {"backend": "codex"}}}
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown role(s) ['wizard']" in str(excinfo.value)


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
                "planner": {
                    "backend": "codex",
                    "extra_mcp_servers": {"kb": {"type": "http", "url": "https://kb.test/mcp"}},
                }
            }
        }
    }
    layer = workflow._make_agent("planner", "system", session_mode="persistent")
    backend = layer.config.backend
    assert (backend.kind, backend.model) == ("codex", "")
    assert backend.hooks is None
    assert backend.extra_mcp_servers == {"kb": {"type": "http", "url": "https://kb.test/mcp"}}
    assert layer.config.required_tools == ()


def test_only_a_routed_role_carries_an_agent_to_the_transport(tmp_path):
    class Capture:
        def __init__(self):
            self.requests = []

        def run(self, request):
            self.requests.append(request)
            return TurnOutcome(role=request.role, text="", transport="capture")

    workflow = _workflow(tmp_path)
    workflow._turns = Capture()
    workflow.task_data = {
        "agents": {"roles": {"kernel_writer": {"backend": "codex", "model": "m"}}}
    }
    workflow._run_turn("planner", "round 1", "prompt")
    workflow._run_turn("kernel_writer", "onboard x", "prompt")
    sent = [(r.role, r.backend, r.model) for r in workflow._turns.requests]
    assert sent == [("planner", None, None), ("kernel_writer", "codex", "m")]
