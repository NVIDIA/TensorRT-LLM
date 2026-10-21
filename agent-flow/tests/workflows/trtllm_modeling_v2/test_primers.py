# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Primers: a role's turns start as forks of one briefing session."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.infra.execution.turns import TurnOutcome
from agent_flow.workflows.trtllm_modeling_v2.run import primers

from .test_workflow import _workflow


def test_primers_are_on_unless_the_run_says_no_primers(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2 import cli
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    assert cli._parse_args(["--task", "t.yaml"]).primers is True
    assert cli._parse_args(["--task", "t.yaml", "--no-primers"]).primers is False
    assert TrtllmModelingV2Workflow(workspace=tmp_path / "ws").use_primers is True


class _Executor:
    """Answers each turn, and records its prompt and the layer it was bound to."""

    remote = False

    def __init__(self):
        self.layers = []
        self.prompts = []

    def bound(self, layer):
        self.layers.append(layer)
        return SimpleNamespace(run=self.run)

    def run(self, request):
        self.prompts.append(request.prompt)
        return TurnOutcome(role=request.role, text="ok")


class _Layer:
    def __init__(self, fork_from=None):
        self.fork_from = fork_from
        self.closed = False

    def __exit__(self, *args):
        self.closed = True


def test_a_primed_roles_turn_starts_as_a_fork_and_is_told_the_commit(tmp_path, monkeypatch):
    executor = _Executor()
    workflow = _workflow(tmp_path, primers=True)
    workflow._turns = executor
    made = []

    def make_agent(name, system_prompt, *, requires_decision=True, session_mode="stateless",
                   fork_from=None):  # fmt: skip
        made.append(fork_from)
        return _Layer(fork_from)

    monkeypatch.setattr(workflow, "_make_agent", make_agent)
    workflow._primers = {
        "kernel_writer": primers.Primer("kernel_writer", "sess-1", "c" * 40, "v", "")
    }
    workflow._primer_ready = {"kernel_writer": threading.Event()}
    workflow._primer_ready["kernel_writer"].set()

    assert workflow._run_turn("kernel_writer", "onboard norm/x", "Do the entry.") == "ok"
    assert made == ["sess-1"]
    assert executor.prompts[-1].startswith("**You are now working your own turn")
    assert executor.prompts[-1].rstrip().endswith("Do the entry.")
    assert executor.layers[-1].closed
    # A role with no primer runs as it always did.
    workflow._run_turn("assembler", "core", "Build it.")
    assert executor.prompts[-1] == "Build it."


def test_a_role_on_a_backend_that_cannot_fork_gets_a_summary_only_primer(tmp_path, monkeypatch):
    executor = _Executor()
    workflow = _workflow(tmp_path, primers=True)
    workflow._turns = executor
    workflow.task_data = {"agents": {"defaults": {"backend": "codex"}}}

    def make_agent(name, system_prompt, *, requires_decision=True, session_mode="stateless",
                   fork_from=None):  # fmt: skip
        layer = _Layer(fork_from)
        layer.session_id = "thread-1"
        return layer

    monkeypatch.setattr(workflow, "_make_agent", make_agent)
    ready = threading.Event()
    workflow._build_primer("kernel_writer", "v", ready)
    assert ready.is_set()
    assert "every later kernel_writer turn starts by reading the summary" in executor.prompts[-1]
    assert workflow._primers["kernel_writer"].session_id == ""
