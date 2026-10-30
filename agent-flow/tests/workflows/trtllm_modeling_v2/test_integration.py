# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task worktrees on a real git repository: integration moves the checkout by whole tasks."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.integration import (
    REBASE_RETRIES,
    Integration,
    Outcome,
    worktree_block,
)
from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

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


def test_parallel_tasks_land_one_after_the_other_without_merge_commits(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    first, second = integration.open("k1"), integration.open("k2")
    _commit(first, "a.py", "A = 2\n", "a")
    _commit(second, "b.py", "B = 2\n", "b")
    assert (repo / "a.py").read_text() == "A = 1\n"  # nothing lands before the task finishes
    assert integration.integrate("k1").merged
    outcome = integration.integrate("k2")
    assert outcome.merged and outcome.commits == 1
    assert (repo / "a.py").read_text() == "A = 2\n" and (repo / "b.py").read_text() == "B = 2\n"
    assert _git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    integration.close("k1")
    assert not first.exists() and _git(repo, "rev-parse", "--verify", "mv2/run/k1")


def test_a_conflict_names_the_files_and_the_head_it_met(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    first, second = integration.open("k1"), integration.open("k2")
    _commit(first, "a.py", "A = 2\n", "a")
    _commit(second, "a.py", "A = 3\n", "a again")
    assert integration.integrate("k1").merged
    outcome = integration.integrate("k2")
    assert not outcome.merged and "a.py" in outcome.detail
    assert outcome.conflicts == ("a.py",) and outcome.base == _git(repo, "rev-parse", "HEAD")
    assert (repo / "a.py").read_text() == "A = 2\n"
    assert _git(second, "status", "--porcelain") == ""  # the rebase was aborted


def test_uncommitted_work_is_not_integrated(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k1")
    (tree / "a.py").write_text("A = 9\n")
    outcome = integration.integrate("k1")
    assert not outcome.merged and "uncommitted changes" in outcome.detail
    assert not outcome.base  # nothing for a rebase to resolve


def test_a_task_with_no_commits_integrates_nothing(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    integration.open("review")
    outcome = integration.integrate("review")
    assert outcome.merged and outcome.commits == 0


def test_the_worker_is_told_where_to_work_and_how_to_commit(tmp_path):
    block = worktree_block(tmp_path / "wt", tmp_path / "repo", "/ws/bin/commit.sh", scratch=False)
    assert f"MV2FLOW_COMMIT_TREE={tmp_path / 'wt'} /ws/bin/commit.sh" in block
    assert f"--tree {tmp_path / 'wt'}" in block
    assert "Nothing in it is integrated" in worktree_block(tmp_path, tmp_path, "c", scratch=True)


def test_commit_sh_commits_in_a_worktree_of_its_repository_only(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.infra.committer import (
        CommitIdentity,
        CommitPolicy,
        install_committer,
    )

    repo = _repo(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    policy = CommitPolicy(identity=CommitIdentity(name="T", email="t@example.com", source="test"))
    script = install_committer(tmp_path / "ws", repo, policy=policy)
    tree = Integration(repo, tmp_path / "ws", "mv2/run", _ENV).open("k1")
    (tree / "a.py").write_text("A = 5\n")

    def run(target: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(script), "set A", "a.py"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "MV2FLOW_COMMIT_TREE": str(target)},
        )

    refused = run(other)
    assert refused.returncode != 0 and "is not a worktree of" in refused.stderr
    done = run(tree)
    assert done.returncode == 0, done.stderr
    assert _git(tree, "log", "-1", "--format=%s") == "set A"
    assert _git(repo, "log", "-1", "--format=%s") == "base"


def test_a_fast_forward_that_meets_a_passing_index_lock_is_tried_again(tmp_path, monkeypatch):
    import agent_flow.workflows.trtllm_modeling_v2.run.integration as integration_mod

    monkeypatch.setattr(integration_mod, "LOCK_RETRY_S", 0.0)
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    _commit(integration.open("k1"), "a.py", "A = 2\n", "a")
    real, calls = integration._git, []

    def flaky(cwd, *args):
        if args[:1] == ("merge",) and not calls:
            calls.append(args)
            return subprocess.CompletedProcess(
                args, 128, "", "fatal: Unable to create 'index.lock'"
            )
        return real(cwd, *args)

    integration._git = flaky
    assert integration.integrate("k1").merged and calls
    assert (repo / "a.py").read_text() == "A = 2\n"


def test_a_task_integrated_before_a_restart_is_not_integrated_again(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k1")
    _commit(tree, "a.py", "A = 2\n", "a")
    assert integration.integrate("k1").commits == 1
    (tree / "b.py").unlink()  # the worktree half removed when the process died
    again = Integration(repo, tmp_path / "ws", "mv2/run", _ENV).integrate("k1")
    assert again.merged and again.commits == 0 and "before the restart" in again.detail
    # A branch that moved since it went in is not taken for integrated.
    (tree / "b.py").write_text("B = 1\n")
    _commit(tree, "a.py", "A = 3\n", "a again")
    assert integration.integrate("k1").commits == 1


# ------------------------------------------------- a conflict goes back to its worker


class _Journal:
    def __init__(self):
        self.events: list[str] = []

    def event(self, kind, text):
        self.events.append(text)


class _Stage(PlannedStage):
    """What a task's worktree turn uses of the workflow, with a scripted worker."""

    def __init__(self, workspace, integration, worker):
        self.workspace = workspace
        self._integration = integration
        self._tls = threading.local()
        self.journal = _Journal()
        self.worker = worker
        #: The worktree block each turn was given, the note on a rebase included.
        self.blocks: list[str] = []

    def _work_task_body(self, task):
        self.blocks.append(self._tls.worktree)
        return self.worker(len(self.blocks))


class _FakeIntegration:
    """Integration outcomes in order; a worktree stays until a merge closes it."""

    def __init__(self, root, *outcomes):
        self.repo = root
        self.outcomes = list(outcomes)
        self.calls = 0
        self.closed: list[str] = []

    def open(self, task_id, touches=(), full=False, attempt=1):
        return self.repo / task_id

    def integrate(self, task_id):
        self.calls += 1
        return self.outcomes[min(self.calls, len(self.outcomes)) - 1]

    def close(self, task_id):
        self.closed.append(task_id)


_CONFLICT = Outcome(False, "conflicts in: a.py", base="f00d" * 10, conflicts=("a.py",))
_MERGED = Outcome(True, "integrated (1 commit(s))", 1)


def test_a_conflicting_task_goes_back_to_its_worker_and_then_integrates(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    first, second = integration.open("k1"), integration.open("k2")
    _commit(first, "a.py", "A = 2\n", "a")
    assert integration.integrate("k1").merged
    head = _git(repo, "rev-parse", "HEAD")

    def worker(turn):
        if turn == 1:
            _commit(second, "a.py", "A = 3\n", "a again")
            return "UNIT_READY"
        # What the note asks for: rebase onto the head, keep both changes, continue.
        rebased = subprocess.run(["git", "-C", str(second), "rebase", head], capture_output=True)
        assert rebased.returncode != 0
        (second / "a.py").write_text("A = 2\nA3 = 3\n")
        _git(second, "add", "a.py")
        _git(second, "-c", "core.editor=true", "rebase", "--continue")
        return "UNIT_READY"

    stage = _Stage(tmp_path / "ws", integration, worker)
    assert stage._work_task_turn(g.Task(id="k2", kind="feature", goal="mtp")) == "UNIT_READY"
    assert (repo / "a.py").read_text() == "A = 2\nA3 = 3\n"
    assert _git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    assert not second.exists()  # integrated, so closed
    first_block, rebase_turn = stage.blocks
    assert "did not integrate" not in first_block
    assert rebase_turn.startswith(first_block)
    assert "did not integrate" in rebase_turn and f"rebase {head}" in rebase_turn
    assert "`a.py`" in rebase_turn and "BLOCKED" in rebase_turn
    events = stage.journal.events
    assert "conflicts with the integration head" in events[0]
    assert events[1] == f"k2: back to its assembler to rebase onto {head[:10]} and resolve " + (
        f"(1 of {REBASE_RETRIES})"
    )
    assert events[2].startswith("k2: integrated")


def test_a_conflict_its_worker_cannot_resolve_fails_the_task_after_the_retries(tmp_path):
    integration = _FakeIntegration(tmp_path, _CONFLICT)
    stage = _Stage(tmp_path / "ws", integration, lambda turn: "UNIT_READY")
    decision = stage._work_task_turn(g.Task(id="f", kind="feature", goal="mtp"))
    assert decision == f"INTEGRATION_FAILED\n{_CONFLICT.detail}"
    assert integration.calls == len(stage.blocks) == REBASE_RETRIES + 1
    assert integration.closed == []  # the worktree is kept for whoever merges it


def test_a_worker_that_gives_up_on_the_rebase_ends_with_its_own_decision(tmp_path):
    integration = _FakeIntegration(tmp_path, _CONFLICT, _MERGED)
    stage = _Stage(
        tmp_path / "ws", integration, lambda turn: "UNIT_READY" if turn == 1 else "BLOCKED"
    )
    assert stage._work_task_turn(g.Task(id="f", kind="feature", goal="mtp")) == "BLOCKED"
    assert integration.calls == 1


def test_a_worktree_left_dirty_goes_back_to_its_worker_to_be_put_right(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)

    def worker(turn):
        tree = integration.tree("k1")
        if turn == 1:
            (tree / "a.py").write_text("A = 9\n")  # finished, but left it uncommitted
        else:
            _git(tree, "commit", "-q", "-am", "a\n\nSigned-off-by: T <t@example.com>")
        return "UNIT_READY"

    stage = _Stage(tmp_path / "ws", integration, worker)
    decision = stage._work_task_turn(g.Task(id="k1", kind="feature", goal="mtp"))
    assert decision == "UNIT_READY"
    assert len(stage.blocks) == 2
    assert "could not be integrated as it stands" in stage.blocks[1]
    assert "uncommitted changes" in stage.blocks[1]
    assert (repo / "a.py").read_text() == "A = 9\n"


def test_a_worktree_never_put_right_fails_integration_after_the_retries(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)

    def worker(turn):
        (integration.tree("k1") / "a.py").write_text("A = 9\n")
        return "UNIT_READY"

    stage = _Stage(tmp_path / "ws", integration, worker)
    decision = stage._work_task_turn(g.Task(id="k1", kind="feature", goal="mtp"))
    assert decision.startswith("INTEGRATION_FAILED\nuncommitted changes")
    assert len(stage.blocks) == 1 + REBASE_RETRIES


def test_a_kernel_task_rebases_through_a_fresh_smith_dispatch(tmp_path):
    """With a note the smith is dispatched again, never handed its decision on disk."""

    class KernelStage(_Stage):
        def __init__(self, *args):
            super().__init__(*args)
            self.smiths: list[tuple[str, str]] = []

        def _queue_item(self, task):
            return SimpleNamespace(n=7)

        def _run_one_smith(self, item, note="", task=None):
            self.smiths.append((note, self._tls.worktree))

        def _smith_decision_for(self, item):
            return "LANDED"

    integration = _FakeIntegration(tmp_path, _CONFLICT, _MERGED)
    stage = KernelStage(tmp_path / "ws", integration, lambda turn: "LANDED")
    assert stage._work_task_turn(g.Task(id="k07", kind="kernel", goal="rmsnorm")) == "LANDED"
    ((note, block),) = stage.smiths
    assert "did not integrate" in note and _CONFLICT.base in note
    assert block == stage.blocks[0] and "did not integrate" not in block
    assert integration.closed == ["k07"]


def test_the_run_journals_how_each_task_worktree_was_made(tmp_path):
    repo = _repo(tmp_path)
    stage = _Stage(tmp_path / "ws", None, worker=None)
    stage.task_data = {"trtllm_repo_path": str(repo)}
    stage.task_worktrees = True
    integration = stage._make_integration(SimpleNamespace(slug="run"))
    integration.open("k1", ("a",))
    assert any(event.startswith("k1: worktree") for event in stage.journal.events)


def _stop_rebase_before_its_commits(tree: Path, onto: str) -> None:
    """A rebase as a process that stopped mid-integration leaves it: HEAD at the base."""
    subprocess.run(
        ["git", "-C", str(tree), "rebase", "-i", onto],
        check=True,
        capture_output=True,
        env={**os.environ, **_ENV, "GIT_SEQUENCE_EDITOR": "sed -i '1i break'"},
    )


def test_a_rebase_cut_short_by_a_restart_is_integrated_not_dropped(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    first, second = integration.open("k1"), integration.open("k2")
    _commit(first, "a.py", "A = 2\n", "a")
    _commit(second, "b.py", "B = 2\n", "b")
    assert integration.integrate("k1").merged
    _stop_rebase_before_its_commits(second, _git(repo, "rev-parse", "HEAD"))
    assert _git(second, "rev-list", "--count", "main..HEAD") == "0"
    outcome = integration.integrate("k2")
    assert outcome.merged and outcome.commits == 1
    assert (repo / "b.py").read_text() == "B = 2\n"


def test_a_worktree_off_its_branch_is_not_integrated_as_empty(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k1")
    _commit(tree, "a.py", "A = 2\n", "a")
    _git(tree, "checkout", "-q", "--detach", "main")
    outcome = integration.integrate("k1")
    assert not outcome.merged
    assert "not on its branch" in outcome.detail
