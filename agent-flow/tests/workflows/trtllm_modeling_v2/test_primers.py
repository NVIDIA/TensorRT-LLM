# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Primers: a role's turns start as forks of one briefing session."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.agents.turns import TurnOutcome
from agent_flow.workflows.trtllm_modeling_v2.run.startup import primers

from .test_workflow import _workflow


def test_primers_are_on_unless_the_run_says_no_primers(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2 import cli
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    assert cli._parse_args(["--task", "t.yaml"]).primers is True
    assert cli._parse_args(["--task", "t.yaml", "--no-primers"]).primers is False
    assert TrtllmModelingV2Workflow(workspace=tmp_path / "ws").use_primers is True


class _Executor:
    """Answers each turn, and records its prompt and the layer it was bound to."""

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


def _primed(tmp_path, monkeypatch, primer):
    """A workflow whose kernel_writer has ``primer`` ready, its executor, and the forks it made."""
    executor, made = _Executor(), []
    workflow = _workflow(tmp_path, primers=True)
    workflow._turns = executor
    workflow._primers["kernel_writer"] = primer
    workflow._primer_ready["kernel_writer"] = threading.Event()
    workflow._primer_ready["kernel_writer"].set()

    def make_agent(*args, fork_from=None, **kwargs):
        made.append(fork_from)
        return _Layer(fork_from)

    monkeypatch.setattr(workflow, "_make_agent", make_agent)
    return workflow, executor, made


def test_a_codex_role_keeps_its_primer_thread_for_later_forks(tmp_path, monkeypatch):
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
    workflow._primer_ready["kernel_writer"] = ready
    workflow._build_primer("kernel_writer", "v", ready)
    assert ready.is_set()
    assert "your session is copied to start every later kernel_writer turn" in executor.prompts[-1]
    assert workflow._primers["kernel_writer"].session_id == "thread-1"
    start = workflow._turn_start("kernel_writer", "implement")
    assert start.forked and start.layer.fork_from == "thread-1"
    assert primers.load(workflow.workspace)["kernel_writer"].session_id == "thread-1"


def test_a_failed_fork_keeps_the_summary_fallback_across_restart(tmp_path, monkeypatch):
    summary = tmp_path / "briefing.md"
    summary.write_text("saved source reading")
    primer = primers.Primer("kernel_writer", "old-thread", "a" * 40, "v", str(summary))
    workflow, executor, _ = _primed(tmp_path, monkeypatch, primer)
    original = executor.run

    def run(request):
        if request.prompt.startswith("**You are now working your own turn"):
            raise RuntimeError("thread/fork failed before the turn started")
        return original(request)

    monkeypatch.setattr(executor, "run", run)
    assert workflow._run_turn("kernel_writer", "verify", "do the entry") == "ok"
    assert str(summary) in executor.prompts[-1]
    kept = primers.load(workflow.workspace)["kernel_writer"]
    assert kept.session_id == "" and kept.summary == str(summary)
