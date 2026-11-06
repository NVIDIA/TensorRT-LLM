# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node-local trees (``environment.local_trees``): snapshots, task worktrees, tune trees, jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import localtree, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)
from agent_flow.workflows.trtllm_modeling_v2.run.workspace import snapshots as snapshots_mod
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import Integration
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import TaskSchemaError
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    _validate_environment as validate_environment,
)
from agent_flow.workflows.trtllm_modeling_v2.tune import worktree as tune_worktree

_ENV = {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A checkout with a package, a second directory, and a native build git ignores."""
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "tensorrt_llm" / "__init__.py").write_text("X = 1\n")
    (repo / "docs" / "d.md").write_text("doc\n")
    (repo / ".gitignore").write_text("tensorrt_llm/libs/\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "tensorrt_llm" / "libs").mkdir()
    (repo / "tensorrt_llm" / "libs" / "native.so").write_text("so\n")
    return repo


@pytest.fixture
def local(tmp_path: Path, monkeypatch) -> Path:
    root = localtree.root_for(tmp_path / "node", tmp_path / "workspace", user="u")
    monkeypatch.setenv(localtree.ROOT_ENV, str(root))
    monkeypatch.delenv(localtree.REPO_ENV, raising=False)
    return root


def test_paths_name_what_a_tree_is(tmp_path):
    root = localtree.root_for("/tmp", tmp_path / "ws", user="u")
    assert root.parent == Path("/tmp/mv2flow-trees-u") and root.name.startswith("ws-")
    sha = "a" * 40
    assert localtree.commit_of(root / "snapshots" / sha) == sha
    assert localtree.is_commit_tree(root / "commits" / sha)
    assert not localtree.is_commit_tree(root / "commits" / "abc")
    assert localtree.is_mutable_tree(root / "worktrees" / "t1")
    assert localtree.is_mutable_tree(root / "tune" / "camp-1" / "best")
    assert localtree.tree_name(root / "tune" / "camp-1" / "best") == "tune-camp-1-best"
    assert localtree.parse(tmp_path / "worktrees" / "t1") is None


def test_a_commit_tree_is_made_from_the_checkouts_objects_with_its_recorded_cone(checkout, local):
    head = _git(checkout, "rev-parse", "HEAD")
    tree = localtree.commit_tree(local, "snapshots", head)
    localtree.materialize(tree, checkout, ("tensorrt_llm",))
    assert (tree / "tensorrt_llm" / "__init__.py").read_text() == "X = 1\n"
    assert not (tree / "docs").exists()
    assert (tree / "tensorrt_llm" / "libs").resolve() == (checkout / "tensorrt_llm" / "libs")
    assert not os.access(tree / "tensorrt_llm" / "__init__.py", os.W_OK)
    assert _git(tree, "status", "--porcelain") == ""
    # No object of its own: it borrows the checkout's.
    assert not any((tree / ".git" / "objects" / "pack").iterdir())

    # Another node reads the cone recorded beside the checkout's git directory.
    shutil.rmtree(tree)
    localtree.materialize(tree, checkout)
    assert not (tree / "docs").exists() and localtree.ready(tree)["cone"] == ["tensorrt_llm"]
    # Asked for more, it widens in place.
    localtree.materialize(tree, checkout, None)
    assert (tree / "docs" / "d.md").exists() and localtree.ready(tree)["cone"] is None


def test_a_worktree_freezes_into_a_commit_of_its_edits_without_touching_it(checkout, local):
    wt = local / "worktrees" / "t1"
    _git(checkout, "worktree", "add", "-q", "-b", "mv2/t1", str(wt))
    (wt / "tensorrt_llm" / "__init__.py").write_text("X = 2\n")
    (wt / "tensorrt_llm" / "new.py").write_text("Y = 1\n")
    (wt / ".claude").mkdir()
    (wt / ".claude" / "settings.local.json").write_text("{}")
    before = _git(wt, "status", "--porcelain")

    frozen = localtree.freeze(wt)
    sha = localtree.commit_of(frozen)
    assert frozen.parent == local / "commits" and sha != _git(wt, "rev-parse", "HEAD")
    assert _git(checkout, "rev-parse", "refs/mv2flow/wip/worktrees-t1") == sha
    assert _git(checkout, "show", f"{sha}:tensorrt_llm/new.py") == "Y = 1"
    assert subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", f"{sha}:.claude/settings.local.json"],
        capture_output=True,
    ).returncode
    assert _git(wt, "status", "--porcelain") == before
    # A clean worktree is its HEAD.
    _git(wt, "add", "-A", "--", "tensorrt_llm")
    _git(wt, "commit", "-q", "-m", "work")
    assert localtree.commit_of(localtree.freeze(wt)) == _git(wt, "rev-parse", "HEAD")


def test_spoolctl_submits_a_frozen_tree_and_its_source(checkout, local):
    wt = local / "worktrees" / "t2"
    _git(checkout, "worktree", "add", "-q", "-b", "mv2/t2", str(wt))
    (wt / "tensorrt_llm" / "__init__.py").write_text("X = 3\n")
    fields = spoolctl.tree_fields(wt)
    assert localtree.is_commit_tree(fields["tree"])
    assert Path(fields["source"]) == checkout.resolve()
    # Anything else goes as it is.
    assert spoolctl.tree_fields(checkout) == {"tree": str(checkout)}


def test_the_scheduler_makes_a_commit_tree_on_its_node_before_the_job(checkout, local, tmp_path):
    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    wt = local / "worktrees" / "t3"
    _git(checkout, "worktree", "add", "-q", "-b", "mv2/t3", str(wt))
    (wt / "tensorrt_llm" / "__init__.py").write_text("X = 4\n")
    fields = spoolctl.tree_fields(wt)
    tree = Path(fields["tree"])
    assert not tree.exists()
    header = " ".join(f"{k}={v}" for k, v in fields.items())
    body = 'pwd; python3 -c "import tensorrt_llm; print(tensorrt_llm.X)"'
    (SpoolPaths(spool).pending / "job-1.sh").write_text(
        f"#!/bin/bash\nset -u\n#spool: gpus=1 {header}\n{body}\n"
    )
    environ = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    sched = Scheduler(spool, SchedulerConfig(devices=("0",)), environ=environ)
    receipt = SpoolPaths(spool).receipt("job-1")
    end = time.monotonic() + 30
    while not receipt.exists() and time.monotonic() < end:
        sched.tick()
        time.sleep(0.05)
    log = SpoolPaths(spool).log("job-1").read_text()
    assert receipt.read_text().strip() == "0", log
    assert f"\n{tree}\n" in log and "\n4\n" in log and localtree.ready(tree)


def test_a_missing_commit_tree_with_no_source_is_refused(local, tmp_path):
    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    tree = local / "commits" / ("b" * 40)
    (SpoolPaths(spool).pending / "job-2.sh").write_text(
        f"#!/bin/bash\n#spool: gpus=1 tree={tree}\ntrue\n"
    )
    environ = {
        k: v for k, v in os.environ.items() if not k.startswith("SLURM_") and k != "MV2FLOW_REPO"
    }
    sched = Scheduler(spool, SchedulerConfig(devices=("0",)), environ=environ)
    sched.tick()
    record = json.loads(SpoolPaths(spool).record("job-2").read_text())
    assert "node-local commit tree" in record.get("detail", "")


def test_a_task_worktree_is_node_local_locked_and_comes_back_with_its_edits(checkout, local):
    integration = Integration(checkout, checkout.parent / "workspace", "mv2/run", _ENV)
    tree = integration.open("t4", touches=("tensorrt_llm/x.py",))
    assert tree == local / "worktrees" / "t4"
    assert "locked" in _git(checkout, "worktree", "list", "--porcelain")
    (tree / "tensorrt_llm" / "__init__.py").write_text("X = 5\n")
    localtree.freeze(tree)

    shutil.rmtree(tree)  # the node is gone
    again = integration.open("t4", touches=("tensorrt_llm/x.py",))
    assert again == tree and (tree / "tensorrt_llm" / "__init__.py").read_text() == "X = 5\n"
    assert _git(tree, "diff", "--name-only") == "tensorrt_llm/__init__.py"

    # A whole-tree task keeps its worktree in the workspace.
    whole = integration.open("t5", full=True)
    assert whole == checkout.parent / "workspace" / "worktrees" / "t5"
    integration.close("t4")
    assert not tree.exists()
    assert "t4" not in _git(checkout, "worktree", "list")


def test_snapshots_are_node_local_and_their_reads_are_shared(checkout, local, tmp_path):
    shared = tmp_path / "workspace" / "snapshots"
    snapshots = snapshots_mod.Snapshots(checkout, shared, cone_paths=("docs/d.md",))
    head = _git(checkout, "rev-parse", "HEAD")
    with snapshots.lease(head) as tree:
        assert tree == local / "snapshots" / head
        assert (tree / "docs" / "d.md").exists() and snapshots.commit_of(tree) == head
        pin = snapshots_mod.CheckTree(commit=head, tree=tree, snapshots=snapshots)
        seen = []

        def runner(command, env=None, timeout=1800):
            seen.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        pin.runner(runner)("true")
        assert f"MV2FLOW_READS_DIR={shared / snapshots_mod.READS_DIRNAME}/" in seen[0]
    _git(checkout, "commit", "-q", "--allow-empty", "-m", "next")
    assert snapshots.prune() == [head] and not tree.exists()


def test_tune_trees_go_under_the_node_local_root(checkout, local, tmp_path):
    workdir = tmp_path / "workspace" / "tune"
    root = tune_worktree.trees_root(workdir)
    assert root.parent == local / "tune"
    best = tune_worktree.open_candidate(checkout, root, "best", "HEAD", prefix="tune")
    assert localtree.is_mutable_tree(best)
    assert "locked" in _git(checkout, "worktree", "list", "--porcelain")
    head = _git(checkout, "rev-parse", "HEAD")
    measured = tune_worktree.measured_tree(checkout, root, head)
    assert measured == local / "commits" / head and (measured / "docs" / "d.md").exists()


def test_local_trees_must_be_an_absolute_directory():
    block = {"kind": "local", "python": "python3", "local_trees": "tmp"}
    errors: list[str] = []
    try:
        validate_environment(block, errors)
    except TaskSchemaError as exc:
        errors.append(str(exc))
    assert any("local_trees" in e for e in errors)
    errors = []
    validate_environment({**block, "local_trees": "/tmp"}, errors)
    assert not any("local_trees" in e for e in errors)
