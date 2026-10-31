# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task worktrees on a real git repository: integration moves the checkout by whole tasks."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitIdentity, CommitPolicy
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import TaskResult
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import (
    REBASE_RETRIES,
    Integration,
)

_ENV = {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    (repo / "b.py").write_text("B = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _commit(tree: Path, name: str, text: str, message: str) -> None:
    (tree / name).write_text(text)
    _git(tree, "commit", "-q", "-am", f"{message}\n\nSigned-off-by: T <t@example.com>")


def test_uncommitted_work_or_a_commit_that_breaks_the_rules_goes_back_to_its_worker(tmp_path):
    """A plain `git commit` in a worktree is audited once, before it can land."""
    repo = _repo(tmp_path)
    policy = CommitPolicy(identity=CommitIdentity("T", "t@example.com"), allowed_paths=("a.py",))
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV, policy=policy)
    tree = integration.open("k1")
    (tree / "a.py").write_text("A = 2\n")
    outcome = integration.integrate("k1")
    assert not outcome.merged and "uncommitted changes" in outcome.detail
    assert not outcome.base  # nothing for a rebase to resolve
    _git(tree, "commit", "-q", "-am", "unsigned")
    outcome = integration.integrate("k1")
    assert not outcome.merged and outcome.repairable
    assert "no Signed-off-by" in outcome.detail and "rebase --signoff" in outcome.detail
    assert (repo / "a.py").read_text() == "A = 1\n"
    _git(tree, "rebase", "-q", "--signoff", _git(repo, "rev-parse", "HEAD"))
    assert integration.integrate("k1").merged


# ------------------------------------------------- a conflict goes back to its worker


class _Journal:
    def __init__(self):
        self.events: list[str] = []

    def event(self, kind, text):
        self.events.append(text)


class _Stage(ExecutionStage):
    """What a task's worktree turn uses of the workflow, with a scripted worker."""

    def __init__(self, workspace, integration, worker):
        super().__init__(
            SimpleNamespace(workspace=workspace, _tls=threading.local(), journal=_Journal())
        )
        self._integration = integration
        self.worker = worker
        #: The worktree block each turn was given, the note on a rebase included.
        self.blocks: list[str] = []

    def _work_task_body(self, task):
        self.blocks.append(self._tls.worktree)
        return TaskResult(self.worker(len(self.blocks)))


def test_a_conflicting_task_goes_back_to_its_worker_and_then_integrates(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    first, second = integration.open("k1"), integration.open("k2")
    _commit(first, "a.py", "A = 2\n", "a")
    # k2's own change predates k1's landing, so its worktree keeps its base.
    _commit(second, "a.py", "A = 3\n", "a again")
    assert integration.integrate("k1").merged
    head = _git(repo, "rev-parse", "HEAD")

    def worker(turn):
        if turn == 1:
            return "UNIT_READY"
        # What the note asks for: rebase onto the head, keep both changes, continue.
        rebased = subprocess.run(["git", "-C", str(second), "rebase", head], capture_output=True)
        assert rebased.returncode != 0
        (second / "a.py").write_text("A = 2\nA3 = 3\n")
        _git(second, "add", "a.py")
        _git(second, "-c", "core.editor=true", "rebase", "--continue")
        return "UNIT_READY"

    stage = _Stage(tmp_path / "ws", integration, worker)
    assert (
        stage._work_task_turn(g.Task(id="k2", kind="feature", goal="mtp")).decision == "UNIT_READY"
    )
    assert (repo / "a.py").read_text() == "A = 2\nA3 = 3\n"
    assert _git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    assert not second.exists()  # integrated, so closed
    first_block, rebase_turn = stage.blocks
    assert "did not integrate" not in first_block
    assert rebase_turn.startswith(first_block)
    assert "did not integrate" in rebase_turn and f"rebase {head}" in rebase_turn
    assert "`a.py`" in rebase_turn and "BLOCKED" in rebase_turn
    events = stage.workflow.journal.events
    assert "conflicts with the integration head" in events[0]
    assert events[1] == f"k2: back to its assembler to rebase onto {head[:10]} and resolve " + (
        f"(1 of {REBASE_RETRIES})"
    )
    assert events[2].startswith("k2: integrated")
