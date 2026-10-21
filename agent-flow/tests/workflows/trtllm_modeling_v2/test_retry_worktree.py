# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task's next attempt starts from the integration head, on a real git repository.

A retry reuses its task's worktree, whose branch was cut when the first
attempt started. Without a rebase the retry tests on a tree that predates
what landed since, the fix it was retried for among it, and its smith ends up
building a throwaway tree of its own for every run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.run.integration import (
    ATTEMPTS_FILENAME,
    Integration,
    worktree_block,
)

from .test_integration import _ENV, _commit, _git, _repo


def _land(integration: Integration, task_id: str, name: str, text: str) -> str:
    """Another task's commit, integrated into the checkout; the new head."""
    tree = integration.open(task_id)
    _commit(tree, name, text, f"{task_id} lands")
    assert integration.integrate(task_id).merged
    integration.close(task_id)
    return _git(integration.repo, "rev-parse", "HEAD")


def _holds(tree: Path, head: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(tree), "merge-base", "--is-ancestor", head, "HEAD"],
            capture_output=True,
        ).returncode
        == 0
    )


def test_a_later_attempt_starts_from_the_integration_head(tmp_path):
    repo = _repo(tmp_path)
    log: list[str] = []
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV, log=log.append)
    tree = integration.open("k-chunk", attempt=1)
    _commit(tree, "a.py", "A = 2\n", "attempt 1: guards and cells")
    head = _land(integration, "eng-fix", "b.py", "B = 2\n")
    assert not _holds(tree, head)

    assert integration.open("k-chunk", attempt=2) == tree

    assert _holds(tree, head)
    assert (tree / "b.py").read_text() == "B = 2\n"  # the fix it was retried for
    assert (tree / "a.py").read_text() == "A = 2\n"  # and the first attempt's work
    assert _git(tree, "log", "-1", "--format=%s") == "attempt 1: guards and cells"
    assert any("rebased onto the integration head" in line for line in log)
    assert "behind the checkout" not in worktree_block(tree, repo, "c", scratch=False)


def test_the_same_attempt_opened_again_is_left_as_its_worker_left_it(tmp_path):
    """A resumed process opens the attempt in flight again; its worker may still be in it."""
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k-chunk", attempt=2)
    _commit(tree, "a.py", "A = 2\n", "work in progress")
    tip = _git(tree, "rev-parse", "HEAD")
    _land(integration, "eng-fix", "b.py", "B = 2\n")

    Integration(repo, tmp_path / "ws", "mv2/run", _ENV).open("k-chunk", attempt=2)

    assert _git(tree, "rev-parse", "HEAD") == tip
    assert (tree / "b.py").read_text() == "B = 1\n"


def test_a_worktree_no_attempt_was_recorded_for_is_left_as_it_is(tmp_path):
    """One a process before the record made: its attempt is unknown, so nothing moves it."""
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k-chunk")
    tip = _git(tree, "rev-parse", "HEAD")
    (integration.root / ATTEMPTS_FILENAME).unlink()
    _land(integration, "eng-fix", "b.py", "B = 2\n")

    integration.open("k-chunk", attempt=2)

    assert _git(tree, "rev-parse", "HEAD") == tip


def test_a_worktree_made_again_on_its_branch_starts_from_the_head(tmp_path):
    """An integrated attempt's worktree is closed; its branch is where the next one opens."""
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k-conv")
    _commit(tree, "a.py", "A = 2\n", "attempt 1")
    assert integration.integrate("k-conv").merged
    integration.close("k-conv")
    head = _land(integration, "core", "b.py", "B = 2\n")

    tree = integration.open("k-conv")

    assert _git(tree, "rev-parse", "HEAD") == head
    assert (tree / "b.py").read_text() == "B = 2\n"


def test_a_retry_that_conflicts_with_the_head_is_left_and_its_worker_told(tmp_path):
    repo = _repo(tmp_path)
    log: list[str] = []
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV, log=log.append)
    tree = integration.open("k-chunk", attempt=1)
    _commit(tree, "a.py", "A = 2\n", "attempt 1")
    tip = _git(tree, "rev-parse", "HEAD")
    head = _land(integration, "eng-fix", "a.py", "A = 3\n")

    integration.open("k-chunk", attempt=2)

    assert _git(tree, "rev-parse", "HEAD") == tip
    assert _git(tree, "status", "--porcelain", "--untracked-files=no") == ""  # rebase aborted
    assert any("stopped on conflicts in a.py" in line for line in log)
    block = worktree_block(tree, repo, "/ws/bin/commit.sh", scratch=False)
    assert "## Your branch is behind the checkout" in block
    assert f"`{head}`" in block and f"git -C {tree} rebase {head}" in block
    assert "1 commit(s) landed" in block


def test_a_retry_with_uncommitted_work_keeps_it_where_it_was(tmp_path):
    repo = _repo(tmp_path)
    log: list[str] = []
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV, log=log.append)
    tree = integration.open("k-chunk", attempt=1)
    tip = _git(tree, "rev-parse", "HEAD")
    (tree / "a.py").write_text("A = 9\n")
    _land(integration, "eng-fix", "b.py", "B = 2\n")

    integration.open("k-chunk", attempt=2)

    assert _git(tree, "rev-parse", "HEAD") == tip
    assert (tree / "a.py").read_text() == "A = 9\n"
    assert any("uncommitted changes" in line for line in log)


def test_a_scratch_worktree_behind_the_head_is_told_without_integration_words(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("why-red")
    _land(integration, "core", "b.py", "B = 2\n")
    block = worktree_block(tree, repo, "c", scratch=True)
    assert "behind the checkout" in block and "when it is integrated" not in block


def test_changed_paths_names_what_the_branch_changes(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k1")
    assert integration.changed_paths("k1") == ()
    _commit(tree, "a.py", "A = 2\n", "a")
    _land(integration, "k2", "b.py", "B = 2\n")  # the head moving adds nothing to k1's
    assert integration.changed_paths("k1") == ("a.py",)
