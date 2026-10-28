# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight's agent probe: a turn that cannot do its work stops the run before it spends anything.

No real agent runs here; the layers are fakes, and the probe is judged by the file.
"""

from __future__ import annotations

import dataclasses
import subprocess
from typing import Any

import pytest

from agent_flow.agent_runtime import AgentConfig
from agent_flow.logger import get_logger
from agent_flow.types import SessionInitEvent
from agent_flow.workflows.trtllm_modeling_v2.run.startup import agent_probe, bootstrap
from agent_flow.workflows.trtllm_modeling_v2.run.startup.agent_probe import (
    AGENT_PROBE_RECORD,
    AGENT_PROBE_SKIP_ENV,
    ProbedAgent,
    hint_for,
    probe_agents,
    read_probe_file,
)

from .test_workflow import _workflow


@pytest.fixture(autouse=True)
def _probe_runs(monkeypatch):
    """Undo the suite's skip, and pin the CLI identity the probe records."""
    monkeypatch.delenv(AGENT_PROBE_SKIP_ENV, raising=False)
    monkeypatch.setattr(agent_probe, "launched_cli", lambda: ("/opt/claude", "2.1.296"))


class SayingLayer:
    """A turn whose Bash tool failed, and whose model said so and ended normally."""

    def __init__(self, agent, workspace, said: str) -> None:
        self.said = said

    async def aforward(self, prompt: str) -> str:
        return self.said

    def __exit__(self, *args) -> None:
        pass


def test_a_file_never_written_is_a_failure_whatever_the_turn_said(tmp_path):
    problem = read_probe_file(tmp_path / "1.out", "n0nce", "", "/repo")
    assert "did not run the command" in problem


def test_a_failed_probe_stops_the_run_before_any_planner_turn(tmp_path, task_file, monkeypatch):
    said = "Bash failed: Error: connect EPERM /tmp/claude/srt-mux-9.sock"
    monkeypatch.setattr(agent_probe, "make_probe_layer", lambda a, w: SayingLayer(a, w, said))
    shells: list[str] = []
    workflow = _workflow(
        tmp_path,
        skip_substrate_gate=False,
        shell_runner=lambda command, env=None, timeout=0: shells.append(command),
    )
    log = get_logger().console
    assert not bootstrap._run_preflight(workflow, workflow._init_state(task_file, log), log)
    journal = (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")
    assert "agent probe failed" in journal and "srt-mux-9.sock" in journal
    assert "a plain terminal or a tmux session on the login node" in journal
    assert not (tmp_path / "ws" / AGENT_PROBE_RECORD).exists()
    # Nothing past it ran: not the substrate gate, not the survey.
    assert shells == []


@dataclasses.dataclass
class _LayerConfig:
    on_activity: Any = None


class ReadOnlyCodexLayer:
    """A Codex turn whose command ran, approved, in a session the site kept read-only."""

    def __init__(self, agent, workspace) -> None:
        self.config = _LayerConfig()

    async def aforward(self, prompt: str) -> str:
        session = SessionInitEvent(
            backend="codex", sandbox="read-only", approval_policy="untrusted"
        )
        self.config.on_activity("session", session)
        command = next(line.strip() for line in prompt.splitlines() if line.startswith("    "))
        subprocess.run(["bash", "-c", command], check=True)
        return "exit status 0"

    def __exit__(self, *args) -> None:
        pass


def test_a_codex_session_the_site_keeps_read_only_fails_the_probe(tmp_path):
    repo = tmp_path / "repo"
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True)
    probed = ProbedAgent(AgentConfig("codex", "", None), ("planner",))
    (outcome,) = probe_agents(
        [probed], tmp_path / "ws", repo=str(repo), layer_for=ReadOnlyCodexLayer
    )
    assert outcome.settings == "sandbox read-only, approvals untrusted"
    assert "ran read-only" in outcome.problem
    assert "allow no writable sandbox" in hint_for(outcome.problem)
