# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse task worktrees on a repository whose shared config git cannot write.

A read-only, empty ``config.lock`` beside the repository's config is how a
sandbox keeps git from writing it: ``git sparse-checkout set --cone`` in a
linked worktree then fails, as it first turns on per-worktree config there.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agent_flow.workflows.trtllm_modeling_v2.run.integration as integration_mod
from agent_flow.workflows.trtllm_modeling_v2.run.integration import (
    CHECKOUT_WORKERS,
    SPARSE_CONE,
    SPARSE_ON,
    Integration,
    sparse_cone,
    worktree_block,
)

from .test_integration_sparse import _ENV, IN_CONE, OUTSIDE, _commit, _git, _on_disk, _repo


def _lock_config(repo: Path) -> str:
    """Leave the repository's shared config as git cannot write it; return what it says."""
    common = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    lock = common / "config.lock"
    lock.write_text("")
    lock.chmod(0o444)
    return (common / "config").read_text()


def _config(repo: Path) -> str:
    common = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return (common / "config").read_text()


def _skipped(tree: Path) -> set[str]:
    """The index entries git leaves off disk (skip-worktree)."""
    listed = _git(tree, "ls-files", "-t").splitlines()
    return {line[2:] for line in listed if line.startswith("S ")}


def _status(tree: Path) -> str:
    return _git(tree, "status", "--porcelain", "--untracked-files=no")


def test_a_repository_whose_config_cannot_be_locked_still_gets_a_sparse_worktree(tmp_path):
    repo = _repo(tmp_path)
    config = _lock_config(repo)
    events: list[str] = []
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV, log=events.append)
    tree = integration.open("k1")

    assert sparse_cone(tree) == SPARSE_CONE
    # The top-level files and those beside the cone's directories come with it.
    assert _on_disk(tree) == {*IN_CONE, ".gitignore", "tensorrt_llm/libnative.so"}
    assert _skipped(tree) == set(OUTSIDE) and _status(tree) == ""
    assert events == [f"k1: worktree made sparse without a config: {', '.join(SPARSE_CONE)}"]
    assert _config(repo) == config
    # The way through the config is closed here, which is what this test is about.
    refused = subprocess.run(
        ["git", "-C", str(tree), "sparse-checkout", "set", "--cone", "docs"],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "could not lock config file" in refused.stderr


def test_a_worktree_without_a_config_widens_and_then_becomes_whole(tmp_path):
    repo = _repo(tmp_path)
    config = _lock_config(repo)
    events: list[str] = []
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV, log=events.append)
    tree = integration.open("f1")
    _commit(tree, "tensorrt_llm/models/modeling.py", "modeling 2\n", "model")
    (tree / "tensorrt_llm" / "__init__.py").write_text("work in progress\n")

    assert integration.open("f1", ["docs/guide/page.md"]) == tree
    assert "docs/guide" in sparse_cone(tree) and "docs/guide/page.md" in _on_disk(tree)
    assert "docs/guide/page.md" not in _skipped(tree) and "docs/other/notes.md" in _skipped(tree)
    assert integration.open("f1", ["docs/guide/"]) == tree  # nothing more to check out

    assert integration.open("f1", ["cpp/kernels/"]) == tree
    assert sparse_cone(tree) == () and _skipped(tree) == set()
    assert set(IN_CONE + OUTSIDE) <= _on_disk(tree)
    assert (tree / "tensorrt_llm" / "models" / "modeling.py").read_text() == "modeling 2\n"
    assert _status(tree) == "M tensorrt_llm/__init__.py"  # the work in progress stays
    assert events[1:] == [
        "f1: worktree widened to docs/guide without a config",
        "f1: worktree made whole without a config (its task now asks for the whole tree)",
    ]
    assert _config(repo) == config


def _move_on_outside_the_cone(repo: Path) -> None:
    """The checkout moves on while the task works: native code, docs, and a file in the cone."""
    (repo / "cpp" / "kernels" / "rmsnorm.cu").write_text("rmsnorm 2\n")
    (repo / "docs" / "new.md").write_text("new\n")
    (repo / "tests" / "unittest" / "utils" / "util.py").write_text("util 2\n")
    _git(repo, "rm", "-q", "docs/other/notes.md")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "native and docs")


def test_a_worktree_without_a_config_integrates_over_changes_outside_its_cone(tmp_path):
    """The rebase keeps the tree sparse: what the head changed outside the cone stays off disk."""
    repo = _repo(tmp_path)
    config = _lock_config(repo)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k1")
    _commit(tree, "tensorrt_llm/models/modeling.py", "modeling 2\n", "model")
    _commit(tree, "tests/unittest/_torch/test_modeling.py", "test 2\n", "test")
    _move_on_outside_the_cone(repo)
    expected = {p: (repo / p).read_text() for p in _git(repo, "ls-files").split()}
    expected["tensorrt_llm/models/modeling.py"] = "modeling 2\n"
    expected["tests/unittest/_torch/test_modeling.py"] = "test 2\n"

    outcome = integration.integrate("k1")
    assert outcome.merged and outcome.commits == 2
    assert _status(tree) == ""
    assert _git(tree, "log", "--format=%s", "-2").splitlines() == ["test", "model"]
    assert _git(tree, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
    on_disk = _on_disk(tree)
    assert "cpp/kernels/rmsnorm.cu" not in on_disk and "docs/new.md" not in on_disk
    assert {"cpp/kernels/rmsnorm.cu", "docs/new.md"} <= _skipped(tree)
    assert (tree / "tests" / "unittest" / "utils" / "util.py").read_text() == "util 2\n"
    assert set(_git(repo, "ls-files").split()) == set(expected)
    for path, text in expected.items():
        assert (repo / path).read_text() == text, path
    assert _git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    assert _config(repo) == config


def test_a_worker_rebasing_a_worktree_without_a_config_keeps_it_clean(tmp_path):
    """The worker's own ``git rebase``, as :func:`rebase_block` gives it, has no sparse flags.

    It writes what the new base changed outside the cone to disk, the changed
    file and the new one; the tree stays clean and the task's commits whole.
    """
    repo = _repo(tmp_path)
    _lock_config(repo)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k1")
    _commit(tree, "tensorrt_llm/models/modeling.py", "modeling 2\n", "model")
    _move_on_outside_the_cone(repo)
    head = _git(repo, "rev-parse", "HEAD")

    _git(tree, "rebase", head)
    assert _status(tree) == ""
    assert _git(tree, "log", "--format=%s", f"{head}..HEAD") == "model"
    assert (tree / "cpp" / "kernels" / "rmsnorm.cu").read_text() == "rmsnorm 2\n"
    assert (tree / "docs" / "new.md").read_text() == "new\n"
    assert not (tree / "docs" / "other" / "notes.md").exists()
    assert "cpp/CMakeLists.txt" not in _on_disk(tree)
    outcome = integration.integrate("k1")
    assert outcome.merged and outcome.commits == 1
    assert (repo / "tensorrt_llm" / "models" / "modeling.py").read_text() == "modeling 2\n"


def test_a_conflict_in_a_worktree_without_a_config_is_aborted_cleanly(tmp_path):
    repo = _repo(tmp_path)
    _lock_config(repo)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k1")
    _commit(tree, "tensorrt_llm/models/modeling.py", "modeling 2\n", "model")
    (repo / "tensorrt_llm" / "models" / "modeling.py").write_text("modeling 3\n")
    _move_on_outside_the_cone(repo)

    outcome = integration.integrate("k1")
    assert not outcome.merged and outcome.conflicts == ("tensorrt_llm/models/modeling.py",)
    assert _status(tree) == "" and "cpp/kernels/rmsnorm.cu" not in _on_disk(tree)
    assert sparse_cone(tree) == SPARSE_CONE


def test_a_worktree_git_cannot_make_sparse_is_checked_out_whole_and_says_why(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    events: list[str] = []
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV, log=events.append)
    real = integration._git

    def no_sparse(cwd, *args):
        if "sparse-checkout" in args:
            return subprocess.CompletedProcess(args, 1, "", "fatal: no sparse checkout here\n")
        return real(cwd, *args)

    monkeypatch.setattr(integration, "_git", no_sparse)
    tree = integration.open("k1")
    assert sparse_cone(tree) == () and set(IN_CONE + OUTSIDE) <= _on_disk(tree)
    assert events == [
        "k1: worktree made whole: git could not make it sparse (fatal: no sparse checkout here)"
    ]


def test_a_worktree_made_sparse_through_its_config_is_read_and_widened_through_it(tmp_path):
    """A tree ``git sparse-checkout set --cone`` made sparse, where the shared config is writable."""
    repo = _repo(tmp_path)
    events: list[str] = []
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV, log=events.append)
    tree = integration.tree("k1")
    _git(repo, "worktree", "add", "-q", "--no-checkout", "-b", "mv2/run/k1", str(tree), "HEAD")
    _git(tree, "sparse-checkout", "set", "--cone", "--skip-checks", *SPARSE_CONE)
    _git(tree, "reset", "--hard", "-q")

    assert integration.open("k1", ["docs/guide/"]) == tree
    assert sparse_cone(tree) == tuple(sorted(SPARSE_CONE + ("docs/guide",)))
    assert events == ["k1: worktree widened to docs/guide through its config"]
    block = worktree_block(tree, repo, "c", scratch=False)
    assert f"git -C {tree} sparse-checkout add <dir>" in block and "-c core" not in block

    assert integration.open("k1", ["cpp/"]) == tree
    # Its config now says whole, while git keeps the cone file it had.
    assert sparse_cone(tree) == () and set(OUTSIDE) <= _on_disk(tree)
    assert events[-1].startswith("k1: worktree made whole through its config")
    assert _status(tree) == ""


@pytest.mark.parametrize("touches", [[], ["cpp/"]], ids=["sparse", "whole"])
def test_every_checkout_writes_its_files_in_parallel(tmp_path, monkeypatch, touches):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    real, checkouts = integration_mod.git_checked, []

    def record(cwd, *args, check=True):
        if "reset" in args:
            checkouts.append(args)
        return real(cwd, *args, check=check)

    monkeypatch.setattr(integration_mod, "git_checked", record)
    integration.open("k1", touches)
    assert len(checkouts) == 1 and f"checkout.workers={CHECKOUT_WORKERS}" in checkouts[0]
    assert (set(SPARSE_ON) <= set(checkouts[0])) == (not touches)
