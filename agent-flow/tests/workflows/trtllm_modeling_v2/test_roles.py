# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role backend routing for the workflow's roles."""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.agents import roles
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import PromptBundle
from agent_flow.workflows.trtllm_modeling_v2.agents.session_model import ModelSelectionError
from agent_flow.workflows.trtllm_modeling_v2.agents.turns import TurnOutcome
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

from .test_task_schema import _valid_spec, _write
from .test_workflow import _workflow


@pytest.fixture(autouse=True)
def _isolated_driving_agent(monkeypatch):
    for name in (roles.CODEX_SESSION_ENV, roles.CLAUDE_SESSION_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "backend,env",
    [
        ("claude-code", {"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "s-1"}),
        ("codex", {"CODEX_THREAD_ID": "t-1"}),
    ],
)
def test_roles_read_the_launching_model_once(backend, env, monkeypatch):
    seen = []

    def read_model(kind, environ):
        seen.append((kind, dict(environ)))
        return "current-session-model"

    monkeypatch.setattr(roles, "session_model", read_model)
    agents = roles.resolve_role_agents({}, environ=env)
    assert seen == [(backend, env)]
    assert {(a.backend, a.model) for a in agents.values()} == {(backend, "current-session-model")}
    assert all(a.reasoning_effort is None for a in agents.values())


def test_explicit_overrides_do_not_need_a_parent_session(monkeypatch):
    def unavailable(*args):
        raise AssertionError("explicit models must not inspect the parent session")

    monkeypatch.setattr(roles, "session_model", unavailable)
    task = {
        "agents": {
            "defaults": {"backend": "claude-code", "model": "global-model"},
            "roles": {"kernel_writer": {"backend": "codex", "model": "worker-model"}},
        }
    }
    agents = roles.resolve_role_agents(task, environ={})
    assert agents["kernel_writer"].model == "worker-model"
    assert agents["assembler"].model == "global-model"
    named = roles.resolve_role_agent(
        {}, "planner", environ={"CODEX_THREAD_ID": "t", "CODEX_DEFAULT_MODEL": "custom"}
    )
    assert named.model == "custom"


def test_a_backend_switch_never_inherits_the_other_backends_model(monkeypatch):
    monkeypatch.setattr(roles, "session_model", lambda backend, env: "codex-current")
    task = {
        "agents": {
            "defaults": {"backend": "claude-code", "model": "claude-selected"},
            "roles": {"planner": {"backend": "codex"}},
        }
    }
    agents = roles.resolve_role_agents(task, environ={})
    assert agents["planner"].model == "codex-current"
    assert agents["assembler"].model == "claude-selected"
    inferred = {
        "agents": {
            "defaults": {"model": "codex-global"},
            "roles": {"planner": {"backend": "codex"}},
        }
    }
    assert (
        roles.resolve_role_agent(inferred, "planner", environ={"CODEX_THREAD_ID": "t"}).model
        == "codex-global"
    )


def test_an_unknown_or_nested_parent_never_silently_uses_a_default():
    with pytest.raises(ModelSelectionError, match="agents.defaults.model"):
        roles.resolve_role_agents({}, environ={})
    env = {"CLAUDECODE": "1", "CODEX_THREAD_ID": "nested"}
    with pytest.raises(ModelSelectionError, match="Both Claude and Codex"):
        roles.resolve_role_agents({}, environ=env)
    task = {"agents": {"defaults": {"backend": "codex", "model": "chosen"}}}
    assert {a.model for a in roles.resolve_role_agents(task, environ=env).values()} == {"chosen"}


def test_a_role_override_changes_only_that_role_and_the_rest_inherit_the_driving_agent():
    assert set(roles.ROLES) == set(PromptBundle.__dataclass_fields__)  # every prompt is routable
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
    assert (planner.backend, planner.model) == ("claude-code", "claude-other")
    assert (kernel_writer.backend, kernel_writer.model, kernel_writer.reasoning_effort) == (
        "codex",
        "codex-test-model",
        "high",
    )
    for role in set(roles.ROLES) - {"planner", "kernel_writer"}:
        agent = roles.resolve_role_agent(task, role)
        assert (agent.backend, agent.model, agent.reasoning_effort) == (
            "claude-code",
            "claude-test-model",
            None,
        ), role


def test_the_spec_accepts_an_agents_block_of_known_roles_only(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["agents"] = {
        "defaults": {"backend": "claude-code", "model": "claude-x"},
        "roles": {"planner": {"backend": "codex", "reasoning_effort": "high"}},
    }
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["agents"]["roles"]["planner"]["backend"] == "codex"
    spec["agents"] = {"roles": {"wizard": {"backend": "codex"}}}
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown role(s) ['wizard']" in str(excinfo.value)


def test_a_claude_role_keeps_its_stop_hook_and_a_codex_role_gets_none(tmp_path):
    """The hook refuses a turn that wrote nothing; the file is validated when read.

    Codex refuses per-client Claude SDK hooks, so a Stop hook would fail the turn.
    """
    workflow = _workflow(tmp_path)
    kb = {"kb": {"type": "http", "url": "https://kb.test/mcp"}}
    workflow.task_data = {
        "agents": {
            "roles": {
                "assembler": {"reasoning_effort": "high"},
                "planner": {"backend": "codex", "extra_mcp_servers": kb},
            }
        }
    }
    layer = workflow._make_agent("assembler", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model, backend.reasoning_effort) == (
        "claude-code",
        "claude-test-model",
        "high",
    )
    assert backend.hooks and backend.hooks["Stop"]
    assert backend.cwd == workflow.workspace
    assert layer.config.required_tools == ()

    layer = workflow._make_agent("planner", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model) == ("codex", "codex-test-model")
    assert backend.hooks is None
    assert backend.extra_mcp_servers == kb
    assert layer.config.required_tools == ()
    # Without the user's plugins, and, once the agent probe has read them, their skills.
    assert backend.plugins is False and backend.disabled_skills == ()
    workflow._foreign_skills = {"codex": ("imagegen", "cluster-jobs")}
    backend = workflow._make_agent("planner", "system").config.backend
    assert backend.disabled_skills == ("imagegen", "cluster-jobs")


def test_probe_primers_and_turns_share_pinned_models(tmp_path, monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2.run.startup.agent_probe import distinct_agents

    workflow = _workflow(tmp_path)
    first = workflow._role_agents()
    monkeypatch.setenv("CLAUDE_CODE_DEFAULT_MODEL", "changed-after-start")
    assert workflow._role_agents() is first
    for role in roles.ROLES:
        plain = workflow._make_agent(role, "system")
        primer = workflow._make_agent(role, "system", requires_decision=False)
        fork = workflow._make_agent(role, "system", fork_from="primer-session")
        assert (
            plain.config.backend.model
            == primer.config.backend.model
            == fork.config.backend.model
            == "claude-test-model"
        )
    probed = distinct_agents(workflow.task_data, resolved=first)
    assert len(probed) == 1 and probed[0].agent.model == "claude-test-model"


def test_every_turn_records_the_agent_its_role_resolves_to(tmp_path):
    class Capture:
        def __init__(self):
            self.requests = []

        def run(self, request):
            self.requests.append(request)
            return TurnOutcome(role=request.role, text="")

    workflow = _workflow(tmp_path)
    workflow._turns = Capture()
    workflow.task_data = {
        "agents": {"roles": {"kernel_writer": {"backend": "codex", "model": "m"}}}
    }
    workflow._run_turn("planner", "round 1", "prompt")
    workflow._run_turn("kernel_writer", "onboard x", "prompt")
    sent = [(r.role, r.backend, r.model) for r in workflow._turns.requests]
    assert sent == [
        ("planner", "claude-code", "claude-test-model"),
        ("kernel_writer", "codex", "m"),
    ]
