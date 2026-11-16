# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run's trees: pins and their build records, trees on a node's own disk, a job that runs its pin."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra import files
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pintree, provenance, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import scheduler as scheduler_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandSpec
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import (
    select_command_executor,
    shell_runner_for,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    RC_NO_ORIGIN,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import TaskResult
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow
from agent_flow.workflows.trtllm_modeling_v2.run.workspace import integration as integration_mod
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import Integration
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.snapshots import CheckTree
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import _validate_environment

LIB = "tensorrt_llm/libs/libth_common.so"
IMPORT = "python3 -c 'import os, tensorrt_llm; print(\"RAN\", os.getcwd(), tensorrt_llm.WHERE)'"
ENV = {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}


@pytest.fixture(autouse=True)
def _devices_read_clean(monkeypatch):
    """This host has no device to read: every test device reads as free."""
    monkeypatch.setattr(
        scheduler_mod, "read_device_memory_mib", lambda: {str(index): 0 for index in range(8)}
    )
    monkeypatch.setattr(scheduler_mod, "read_device_holders", lambda: {})


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    """A checkout with a package, native sources in two places, and a built library git ignores."""
    repo = tmp_path / "repo"
    files = {
        "tensorrt_llm/__init__.py": "WHERE = 'base'\n",
        "tensorrt_llm/models/modeling.py": "M = 1\n",
        "tensorrt_llm/models/x/csrc/CMakeLists.txt": "add_library(x)\n",
        "tensorrt_llm/models/x/csrc/op.cu": "// op\n",
        "cpp/CMakeLists.txt": "project(engine)\n",
        "cpp/kernels/rmsnorm.cu": "// rmsnorm\n",
        "docs/page.md": "page\n",
        ".gitignore": "tensorrt_llm/libs/\n",
    }
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / LIB).parent.mkdir()
    (repo / LIB).write_bytes(b"build-one!")
    return repo


def _trees(repo: Path, tmp_path: Path, **more) -> tuple[Integration, list[str]]:
    said: list[str] = []
    more.setdefault("pin_age_s", 0)  # a pin's tree nobody names goes at once, unless a test says
    return Integration(repo, tmp_path / "ws", "mv2/run", ENV, log=said.append, **more), said


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _same_size_and_time(path: Path, data: bytes) -> None:
    """Rewrite ``path`` with other bytes of the same length, and put its modification time back."""
    before = path.stat()
    assert len(data) == before.st_size
    time.sleep(0.05)  # past the grain of the file system's clock
    path.write_bytes(data)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


# ------------------------------------------------------------ pin and at


def test_a_pin_is_what_the_tree_holds_now_with_the_content_of_each_library(repo, tmp_path):
    trees, _ = _trees(repo, tmp_path)
    tree = trees.open("k1")
    (tree / "tensorrt_llm" / "models" / "modeling.py").write_text("M = 2\n")  # uncommitted
    (tree / "tensorrt_llm" / "models" / "new.py").write_text("N = 1\n")  # untracked
    (tree / ".mcp.json").write_text("{}")  # a sandbox placeholder
    (tree / "docs" / "page.md").write_text("staged\n")
    _git(tree, "add", "docs/page.md")  # staged by the worker, then changed again
    (tree / "docs" / "page.md").write_text("staged, then edited\n")
    (tree / "docs" / "linked.md").symlink_to("page.md")  # source: a link to a file of the tree
    (tree / "docs" / "out.md").symlink_to("../../outside.md")  # leads out of the tree
    before = (_git(tree, "rev-parse", "HEAD"), _git(tree, "status", "--porcelain"))

    pin = trees.pin("k1")
    # The edits are in the pin's commit; the link to the native build and the placeholder are not.
    assert _git(repo, "show", f"{pin.commit}:tensorrt_llm/models/modeling.py") == "M = 2"
    assert _git(repo, "show", f"{pin.commit}:tensorrt_llm/models/new.py") == "N = 1"
    assert _git(repo, "show", f"{pin.commit}:docs/page.md") == "staged, then edited"
    listed = _git(repo, "ls-tree", "-r", "--name-only", pin.commit).split()
    assert ".mcp.json" not in listed and "tensorrt_llm/libs" not in listed
    assert "docs/linked.md" in listed and "docs/out.md" not in listed
    # Nothing in the task's tree moved: not its branch, its index or its files.
    assert (_git(tree, "rev-parse", "HEAD"), _git(tree, "status", "--porcelain")) == before
    assert pin.commit != before[0]
    # The build record is the content of each library, and the pin is a file every node reads.
    assert dict(pin.build) == {LIB: _sha(b"build-one!")}
    stored = pintree.read_pin(pin.file)
    assert (stored["commit"], stored["build"], stored["id"]) == (pin.commit, pin.build, pin.id)

    with trees.at(pin) as at:
        assert (at / "tensorrt_llm" / "models" / "modeling.py").read_text() == "M = 2\n"
        assert (at / "cpp" / "kernels" / "rmsnorm.cu").exists(), "the whole tree, no cone"
        assert (at / LIB).read_bytes() == b"build-one!"
        assert not os.access(at / "tensorrt_llm" / "__init__.py", os.W_OK) or os.geteuid() == 0
        assert pintree.pin_file_of(at) == str(pin.file)
        assert trees.commit_of(at) == pin.commit
        other, _ = _trees(repo, tmp_path)  # another process of the run
        assert other.prune() == [] and at.exists(), "a tree somebody holds is not removed"
        # The task's tree moves on; the pin does not.
        (tree / "tensorrt_llm" / "models" / "modeling.py").write_text("M = 3\n")
        assert (at / "tensorrt_llm" / "models" / "modeling.py").read_text() == "M = 2\n"
        assert trees.pin("k1").commit != pin.commit
    # Nobody holds it and the head is elsewhere: it goes. The head's own tree stays.
    assert not at.exists()
    head = trees.pin()
    assert head.commit == _git(repo, "rev-parse", "HEAD")
    with trees.at(head) as at_head:
        pass
    assert at_head.exists()


def _listing(root: Path, *skip: str) -> set[str]:
    """Every path under ``root``, relative to it, but what lies under the ``skip`` names."""
    found = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if not any(rel == name or rel.startswith(name + "/") for name in skip):
            found.add(rel)
    return found


def test_trees_are_on_the_nodes_own_disk_and_what_every_node_shares_gains_git_objects_only(
    repo, tmp_path
):
    """No file of any tree on the storage every node shares: the repository hears of no tree."""
    before = _listing(repo / ".git", "objects", "refs", "logs")
    trees, _ = _trees(repo, tmp_path)
    # The spec states no place: this node's own disk, under a path named for the run.
    assert trees.place == Path(pintree.local_root(tmp_path / "ws"))
    assert trees.place.parent == tmp_path / "node" / f"mv2flow-trees-{os.environ.get('USER')}"
    assert trees.place.name.startswith("ws-")
    tree = trees.open("k1")
    assert tree == trees.place / "worktrees" / "k1"
    (tree / "docs" / "page.md").write_text("edited\n")
    _git(tree, "commit", "-q", "-am", "k1: page\n\nSigned-off-by: T <t@example.com>")
    (tree / "tensorrt_llm" / "models" / "modeling.py").write_text("M = 2\n")
    pin = trees.pin("k1")
    with trees.at(pin) as at:
        assert at == trees.place / "pins" / pin.id, "a pin's tree is named for its commit and build"
        assert sorted(p.name for p in (at / ".git").iterdir()) == [
            "HEAD",
            "ORIG_HEAD",
            "commondir",
            "index",
        ]
        assert (at / ".git" / "commondir").read_text().strip() == str(repo / ".git")
        assert _git(at, "rev-parse", "HEAD") == pin.commit
        assert _git(at, "status", "--short", "--untracked-files=no") == ""

    # The tree's git directory is its own: only HEAD, the index and git's own notes live there.
    assert (tree / ".git").is_dir() and (tree / ".git" / "commondir").is_file()
    assert (tree / ".git" / "HEAD").read_text().strip() == "ref: refs/heads/mv2/run/k1"
    # A commit made in the tree is the repository's at once: the branch, and the pin's commit.
    assert _git(repo, "rev-parse", "mv2/run/k1") == _git(tree, "rev-parse", "HEAD")
    assert _git(repo, "cat-file", "-t", pin.commit) == "commit"
    # The repository gained objects and refs, and two small files of the run. No worktree is
    # registered, so nothing lists a tree and no prune on any node can take one away.
    gained = _listing(repo / ".git", "objects", "refs", "logs") - before
    assert gained <= {"mv2flow-trees.json", "mv2flow-worktree.lock", "COMMIT_EDITMSG"}, gained
    assert not (repo / ".git" / "worktrees").exists()
    assert _git(repo, "worktree", "list").count("\n") == 0
    # What every node must see of the run's trees is a few records in the workspace.
    shared = _listing(tmp_path / "ws" / "trees")
    assert {f"pins/{pin.id}.json", "open/k1.json"} <= shared
    assert all(name.endswith(".json") or "." not in name for name in shared), shared
    # A stated place is used as stated, for the trees; the records stay with the workspace.
    stated, _ = _trees(repo, tmp_path, root=tmp_path / "elsewhere")
    assert stated.open("k9") == tmp_path / "elsewhere" / "worktrees" / "k9"
    assert stated.shared == tmp_path / "ws" / "trees"


def test_a_fresh_trees_index_is_trusted_by_its_time_and_an_edit_is_still_seen(repo, tmp_path):
    """Git reads again every file as young as its index: a fresh tree's index is made the younger."""
    tree = tmp_path / "tree"
    pintree.make_tree(repo, tree, _git(repo, "rev-parse", "HEAD"))
    index = tree / ".git" / "index"

    def newest() -> float:
        files = (p for p in tree.rglob("*") if p.is_file() and ".git" not in p.parts)
        return max(p.stat().st_mtime for p in files)

    pintree.settle_index(tree, wait=False)  # a pin's tree: nobody edits it
    assert index.stat().st_mtime > newest() + 1 and _git(tree, "status", "--porcelain") == ""
    (tree / "docs" / "page.md").write_text("edited\n")
    pintree.settle_index(tree, wait=True)  # a task's tree: the checkout's second is waited out
    assert int(index.stat().st_mtime) > int(newest()), "no file shares the index's second"
    assert _git(tree, "status", "--porcelain").endswith("docs/page.md"), "the edit is seen"


def test_an_agent_may_write_where_its_tree_is(repo, tmp_path):
    """A task's tree is no longer inside the workspace: the roles' sessions are told the place."""
    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.task_data = {"trtllm_repo_path": str(repo)}
    local = Path(pintree.local_root(tmp_path / "ws"))
    assert integration_mod.place(workflow.task_data, tmp_path / "ws") == local
    assert str(local) in workflow._writable_roots() and local.is_dir()
    stated = {
        "trtllm_repo_path": str(repo),
        "environment": {"trees": {"root": str(tmp_path / "t")}},
    }
    workflow.task_data = stated
    assert str(tmp_path / "t") in workflow._writable_roots()
    assert str(local) not in workflow._writable_roots()


class _Stage(ExecutionStage):
    """What a task's turn uses of the workflow, with a worker that just finishes."""

    def __init__(self, workspace, integration):
        journal = SimpleNamespace(event=lambda kind, text: None)
        super().__init__(
            SimpleNamespace(workspace=workspace, _tls=threading.local(), journal=journal)
        )
        self._integration = integration
        #: What each turn was told of its tree.
        self.blocks: list[str] = []

    def _work_task_body(self, task):
        self.blocks.append(self._tls.worktree)
        return TaskResult("ready")


def test_a_tree_lost_with_its_node_is_made_again_from_the_last_pin_and_the_worker_is_told(
    repo, tmp_path
):
    trees, _ = _trees(repo, tmp_path)
    tree = trees.open("k1")
    (tree / "docs" / "page.md").write_text("committed\n")
    _git(tree, "commit", "-q", "-am", "k1: page\n\nSigned-off-by: T <t@example.com>")
    (tree / "tensorrt_llm" / "models" / "modeling.py").write_text("M = 2\n")
    (tree / "tensorrt_llm" / "models" / "new.py").write_text("N = 1\n")
    pin = trees.pin("k1")  # as when the worker queues a job against its tree
    (tree / "cpp" / "kernels" / "rmsnorm.cu").write_text("// edited after the pin\n")
    never_pinned = trees.open("k2")
    (never_pinned / "docs" / "page.md").write_text("never committed, never pinned\n")
    assert trees.notice("k1") == ""

    # The controller's job moves to another node: the disk that held its trees is gone.
    shutil.rmtree(trees.place)
    moved, said = _trees(repo, tmp_path)
    again = moved.open("k1")
    assert again == tree and _git(again, "symbolic-ref", "--short", "HEAD") == "mv2/run/k1"
    # Its commits were never on that disk; the last pin's edits are back, uncommitted as they were.
    assert (again / "docs" / "page.md").read_text() == "committed\n"
    assert (again / "tensorrt_llm" / "models" / "modeling.py").read_text() == "M = 2\n"
    assert (again / "tensorrt_llm" / "models" / "new.py").read_text() == "N = 1\n"
    changed = [line.strip() for line in _git(again, "status", "--porcelain").splitlines()]
    assert "M tensorrt_llm/models/modeling.py" in changed, "edited, and not staged"
    assert "?? tensorrt_llm/models/new.py" in changed
    assert _git(again, "diff", "--cached", "--name-only") == ""
    assert moved.pin("k1").commit == pin.commit, "the tree holds what its last pin held"
    # A pin's ref is its tree's branch, apart from the refs of trees on no branch: a job may
    # be called what a branch's first part is called.
    assert _git(repo, "rev-parse", "refs/mv2flow/pins/branch/mv2/run/k1") == pin.commit
    (repo / "docs" / "page.md").write_text("an edit in the checkout, on no task's branch\n")
    _git(repo, "checkout", "-q", "--detach")
    detached = pintree.freeze(repo, "mv2")
    assert _git(repo, "rev-parse", "refs/mv2flow/pins/name/mv2") == detached
    _git(repo, "checkout", "-q", "--force", "main")
    # What was edited after the pin is gone, and the worker is told exactly that.
    assert (again / "cpp" / "kernels" / "rmsnorm.cu").read_text() == "// rmsnorm\n"
    notice = moved.notice("k1")
    assert notice.startswith("## Your tree was lost and made again")
    assert "2 file(s): `tensorrt_llm/models/modeling.py`, `tensorrt_llm/models/new.py`" in notice
    assert "**Lost: every edit you made after that pin.**" in notice
    assert "its tree on" in said[-1] and "2 file(s) of its last pin" in said[-1]
    stage = _Stage(tmp_path / "ws", moved)
    stage._work_task_turn(g.Task(id="k1", kind="feature", goal="x"))
    assert stage.blocks[0].endswith(notice) and "## Your worktree" in stage.blocks[0]

    # A tree that was never pinned comes back at its branch, and the worker hears what that costs.
    back = moved.open("k2")
    assert (back / "docs" / "page.md").read_text() == "page\n"
    assert "**Lost: every edit you had not committed.**" in moved.notice("k2")
    # The task goes on and lands as any other; a landed task has no notice left.
    _git(again, "add", "-A")
    _git(again, "commit", "-q", "-m", "k1: model\n\nSigned-off-by: T <t@example.com>")
    assert moved.land("k1").merged
    moved.close("k1")
    assert moved.notice("k1") == "" and not again.exists()
    assert (repo / "tensorrt_llm" / "models" / "new.py").read_text() == "N = 1\n"


def test_a_pins_tree_is_used_again_and_goes_only_when_no_job_names_it_and_it_is_old(repo, tmp_path):
    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    trees, said = _trees(repo, tmp_path, pin_age_s=600)
    (trees.open("k1") / "docs" / "page.md").write_text("edited\n")
    pin = trees.pin("k1")
    assert pintree.read_pin(pin.file)["keep_s"] == 600
    with trees.at(pin) as tree:
        pass
    # Nobody holds it and the head is elsewhere, but it is young: kept for the next job of the pin.
    assert tree.exists()
    with trees.at(pin) as again:
        assert again == tree
    assert len([line for line in said if line.startswith("tree of ")]) == 1, "made once, used twice"
    # One that no longer holds its commit is made again, and that is said.
    module = tree / "tensorrt_llm" / "__init__.py"
    module.with_name("replacement.py").write_text("WHERE = 'replaced'\n")
    os.replace(module.with_name("replacement.py"), module)  # its directory is not read-only
    with trees.at(pin) as again:
        assert again == tree and module.read_text() == "WHERE = 'base'\n"
    assert len([line for line in said if line.startswith("tree of ")]) == 2

    def aged(seconds: float) -> None:
        lock = pintree._lock_path(trees.place, pin.id)
        os.utime(lock, (time.time() - seconds, time.time() - seconds))

    # Nobody holds it and it is younger than the stated age: it stays.
    aged(300)
    assert trees.prune() == [] and tree.exists()
    # Nobody holds it and it is older: it goes. The node a job runs on applies the same rule
    # before it makes that job's tree (`pintree.at`).
    aged(601)
    assert pintree.sweep(trees.place, keep={pin.commit}, age_s=600) == []
    assert trees.prune() == [pin.commit] and not tree.exists()


def test_asking_a_tree_that_is_used_again_reads_no_file(repo, tmp_path, monkeypatch):
    """A pin's tree is made read-only after its checkout, which moves every file's change time.

    Git would take every file for possibly changed and read each one again, through its
    filters, each time the tree is used: seconds for a real tree, and a filter process per
    large file. A filter that notes each run tells whether anything was read.
    """
    noted = tmp_path / "filter.log"
    (repo / ".gitattributes").write_text("*.md filter=probe\n*.py filter=probe\n")
    _git(repo, "config", "filter.probe.clean", f"sh -c 'echo read >> {noted}; cat'")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-q", "-m", "attributes")
    commit, root = _git(repo, "rev-parse", "HEAD"), tmp_path / "place"
    read_only = pintree._read_only

    def a_second_later(tree):
        time.sleep(1.1)  # as on a real tree, whose checkout and whose chmod are seconds apart
        read_only(tree)

    monkeypatch.setattr(pintree, "_read_only", a_second_later)
    before = noted.read_text() if noted.exists() else ""
    pintree.hold(root, commit, repo=repo).close()
    pintree.hold(root, commit, repo=repo).close()
    tree = pintree.pinned_tree(root, commit)
    assert pintree._intact(tree, commit)
    assert (noted.read_text() if noted.exists() else "") == before, "a file was read again"
    # What it still sees: a file replaced through its directory, and one edited in place.
    page = tree / "docs" / "page.md"
    page.chmod(0o644)
    page.write_text("edited in place\n")
    assert not pintree._intact(tree, commit)


def test_changed_bytes_with_the_same_size_and_time_are_seen_and_copied_bytes_are_no_change(
    repo, tmp_path, monkeypatch
):
    """The build record is by content: blind to where and when, never to what."""
    monkeypatch.setattr(pintree, "RACY_S", 0.0)  # remember digests of files written just now
    remembered = tmp_path / "digests.json"
    first = pintree.build_record(repo, cache=pintree.DigestCache(remembered))
    assert first == {LIB: _sha(b"build-one!")}
    assert len(json.loads(remembered.read_text())) == 1
    identity = pintree.build_id(repo, cache=pintree.DigestCache(remembered))
    assert identity == pintree.record_digest(first)

    # Identical bytes copied elsewhere, at another time: the same build.
    elsewhere = tmp_path / "another-disk" / "tree"
    (elsewhere / LIB).parent.mkdir(parents=True)
    (elsewhere / LIB).write_bytes(b"build-one!")
    os.utime(elsewhere / LIB, (1_000_000, 1_000_000))
    copied = pintree.build_record(elsewhere)
    assert copied == first and pintree.record_diff(first, copied) == []
    assert pintree.pin_id("a" * 40, copied) == pintree.pin_id("a" * 40, first)
    assert pintree.build_id(elsewhere) == identity, "the same build, wherever and whenever"

    # Other bytes of the same size, the time put back: names, sizes and times see nothing.
    _same_size_and_time(repo / LIB, b"build-two!")
    assert pintree.build_id(repo, cache=pintree.DigestCache(remembered)) != identity
    # What a process of a pinned job reports: of exactly the libraries the record names.
    assert pintree.build_id(repo, names=[LIB]) == pintree.record_digest({LIB: _sha(b"build-two!")})
    assert pintree.build_id(repo, names=[LIB, "gone.so"]) != pintree.build_id(repo, names=[LIB])
    assert pintree.build_id(tmp_path / "no-package") is None
    again = pintree.build_record(repo, cache=pintree.DigestCache(remembered))
    assert again == {LIB: _sha(b"build-two!")}
    [said] = pintree.record_diff(first, again)
    assert said.startswith(f"{LIB} is {_sha(b'build-two!')[:12]}, the record has ")
    assert pintree.pin_id("a" * 40, again) != pintree.pin_id("a" * 40, first)


# -------------------------------------------------- a job runs a pin


def _scheduler(spool: Path) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PYTHON"))}
    return Scheduler(spool, SchedulerConfig(devices=("0",), leftover_grace_s=0), environ=env)


def _place(repo: Path, tmp_path: Path) -> Path:
    """Say in the checkout, as a run does, where its trees go; that place."""
    place = tmp_path / "ws"
    config = {"repo": str(repo), "root": str(place)}
    pintree.write_config(repo, config)
    return place


def _run_job(spool: Path, name: str, body: str, header: str) -> tuple[dict, str]:
    paths = SpoolPaths(spool)
    paths.ensure()
    (paths.pending / f"{name}.sh").write_text(f"#!/bin/bash\n#spool: {header}\n{body}\n")
    sched = _scheduler(spool)
    end = time.monotonic() + 60
    while time.monotonic() < end:
        sched.tick()
        if paths.receipt(name).exists() and not sched.jobs:
            return json.loads(paths.record(name).read_text()), paths.log(name).read_text()
        time.sleep(0.02)
    raise AssertionError(paths.log(name).read_text() if paths.log(name).exists() else "no log")


def test_a_job_runs_its_pin_and_has_no_result_when_the_libraries_are_not_the_records(
    repo, tmp_path
):
    spool, place = tmp_path / "spool", _place(repo, tmp_path)
    pin_file = pintree.pin_tree(repo)
    pin = pintree.read_pin(pin_file)
    assert pin["commit"] == _git(repo, "rev-parse", "HEAD") and pin["build"] == {
        LIB: _sha(b"build-one!")
    }
    header = f"gpus=1 wall=60 pin={pin_file} ranks=engine:1"

    record, log = _run_job(spool, "first", IMPORT, header)
    tree = pintree.pinned_tree(place, pin["id"])
    assert str(tree) == pin["tree"] and record["rc"] == 0 and record["pin"] == pin_file
    assert f"RAN {tree} base" in log, "the pin's tree is the working directory and the import"
    assert (tree / "cpp" / "kernels" / "rmsnorm.cu").exists(), "the whole tree, no cone"
    assert pintree.pin_file_of(tree) == pin_file
    [made] = pintree.read_tree_records(SpoolPaths(spool).tree_log("first"))
    assert made["ok"] and made["tree"] == str(tree)
    [origin] = [json.loads(line) for line in SpoolPaths(spool).resolve_log("first").open()]
    # The process reports, by content, the very build the pin records.
    assert origin["build"] == made["build"] == pintree.record_digest(pin["build"])

    # A process that finds other libraries than the verified ones: no result.
    grown = f"{IMPORT}; printf more >> {repo / LIB}; {IMPORT}"
    record, log = _run_job(spool, "midway", grown, header)
    assert record["rc"] == RC_NO_ORIGIN and record["reason"] == "no_origin"
    assert "found other libraries" in record["detail"] and "no result" in record["detail"]

    # The shared build is no longer the one the pin records: the job does not even start.
    (repo / LIB).write_bytes(b"build-one!")
    _same_size_and_time(repo / LIB, b"build-two!")
    record, log = _run_job(spool, "stale", IMPORT, header)
    assert record["rc"] == RC_NO_ORIGIN and record["reason"] == "no_origin"
    assert "are not those its pin records" in record["detail"] and LIB in record["detail"]
    assert "RAN " not in log

    # The same commit pinned again carries the build as it is now, and runs.
    again = pintree.pin_tree(repo)
    assert pintree.read_pin(again)["commit"] == pin["commit"] and again != pin_file
    record, log = _run_job(spool, "repinned", IMPORT, f"gpus=1 wall=60 pin={again}")
    # The same source with another build is another pin, and another place: the first pin's
    # tree, and the pin a job is given for that tree's path, stay what they were.
    elsewhere = pintree.read_pin(again)["tree"]
    assert record["rc"] == 0 and f"RAN {elsewhere} base" in log and elsewhere != str(tree)
    assert pintree.pin_file_of(tree) == pin_file and pintree.pin_file_of(elsewhere) == again
    assert pintree.read_pin(pin_file) == pin, "a pin that was given out never changes"
    # Nor when the run's place for trees moves: that pin gets a file of its own.
    moved = dict(pintree.read_config(repo), root=str(tmp_path / "another-place"))
    pintree.write_config(repo, moved)
    relocated = pintree.pin_tree(repo)
    assert relocated not in (pin_file, again) and pintree.read_pin(again)["root"] == str(place)
    assert pintree.read_pin(relocated)["root"] == str(tmp_path / "another-place")


def test_a_tree_named_at_submission_is_pinned_then_and_a_plain_directory_runs_as_it_is(
    repo, tmp_path
):
    _place(repo, tmp_path)
    (repo / "tensorrt_llm" / "__init__.py").write_text("WHERE = 'edited, uncommitted'\n")
    (repo / "tensorrt_llm" / "new.py").write_text("N = 1\n")  # untracked
    (repo / ".mcp.json").write_text("{}")  # a sandbox placeholder
    before = (_git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain"))
    args = spoolctl.build_parser().parse_args(
        ["--spool", str(tmp_path / "spool"), "launch", "--gpus", "1", "--wall", "60"]
        + ["--tree", str(repo), "job1", f"pytest {repo}/tests -q"]
    )
    header = spoolctl.job_header(args, None, "token")
    assert "tree" not in header
    pin = pintree.read_pin(header["pin"])
    # What the tree held then is a commit no branch moves; the tree itself is as it was.
    assert _git(repo, "show", f"{pin['commit']}:tensorrt_llm/__init__.py").startswith("WHERE = 'ed")
    listed = _git(repo, "ls-tree", "-r", "--name-only", pin["commit"]).split()
    assert "tensorrt_llm/new.py" in listed and ".mcp.json" not in listed
    assert (_git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain")) == before
    assert pin["commit"] != before[0]
    # The same files on the same head are the same pin, however often they are pinned.
    assert spoolctl.tree_header(str(repo), "another-job") == {"pin": header["pin"]}
    # The command's own paths follow the pin, not the tree that moves on.
    command = spoolctl.pinned_command(f"pytest {repo}/tests -q", str(repo), header)
    assert command == f"pytest {pin['tree']}/tests -q"
    (repo / "tensorrt_llm" / "__init__.py").write_text("WHERE = 'moved on'\n")
    record, log = _run_job(
        tmp_path / "spool", "job1", IMPORT, f"gpus=1 wall=60 pin={header['pin']}"
    )
    assert record["rc"] == 0 and "edited, uncommitted" in log and "moved on" not in log

    # A directory no run keeps trees for cannot be pinned: it runs as it is, and says so.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert spoolctl.tree_header(str(plain)) == {"tree": str(plain)}


def test_each_host_of_a_job_answers_for_the_tree_verified_there():
    """Trees on each node's own disk: every process is held to what was verified on its host."""
    tree = "/node/disk/tree"

    def process(host, build):
        origin = f"{tree}/tensorrt_llm/__init__.py"
        return {"module": "tensorrt_llm", "origin": origin, "host": host, "build": build, "pid": 7}

    def verified(host):
        return {"ok": True, "host": host, "tree": tree, "build": "the-pins-build"}

    both = [verified("n1"), verified("n2")]
    ran = [process("n1", "the-pins-build"), process("n2", "the-pins-build")]
    # The same bytes on each node's own disk are the same build, whatever their times there.
    assert provenance.check_pinned(ran, both) == ""
    said = provenance.check_pinned([ran[0], process("n2", "another-build")], both)
    assert said.startswith("a process on n2 (pid 7) found other libraries")
    said = provenance.check_pinned([ran[0], process("n3", "the-pins-build")], both)
    assert (
        said == "a process on n3 imported tensorrt_llm, and no tree of the pin was verified there"
    )
    failed = {"ok": False, "host": "n2", "detail": "the libraries are not those its pin records"}
    said = provenance.check_pinned([ran[0]], [both[0], failed])
    assert said == "the libraries are not those its pin records (on n2)"
    assert provenance.check_pinned([], []) == "no tree was made and verified for the job's pin"


def test_each_node_of_a_job_of_several_nodes_runs_the_pin_itself(repo, tmp_path, monkeypatch):
    """Run once by a lead, the job hands every node's step the pin, never the tree it was taken from."""
    spool, place = tmp_path / "spool", _place(repo, tmp_path)
    SpoolPaths(spool).ensure()
    files.write_json_atomic(SpoolPaths(spool).pool, {"at": time.time(), "stopping": False})
    slurm = {"account": "ACCOUNT", "partition": "PARTITION", "time": "01:00:00", "gpus_per_node": 4}
    env = RunEnvironment("slurm", "/venv/bin/python", spool=str(spool), slurm=slurm)
    (repo / "tensorrt_llm" / "__init__.py").write_text("WHERE = 'edited'\n")
    executor = select_command_executor(env, nodes=2, name_prefix="hop", tree=repo)
    queued = [_queued(monkeypatch, executor, spool, CommandSpec("pytest -q", timeout_s=30))]
    script, name = queued[0].read_text(), queued[0].stem
    [header] = [line for line in script.splitlines() if line.startswith("#spool:")]
    pin_file = next(token[4:] for token in header.split() if token.startswith("pin="))
    assert "tree=" not in header and "nodes=2" in header
    pin = pintree.read_pin(pin_file)
    assert _git(repo, "show", f"{pin['commit']}:tensorrt_llm/__init__.py") == "WHERE = 'edited'"
    steps = [line for line in script.splitlines() if line.startswith("srun ")]
    assert len(steps) == 2
    record = SpoolPaths(spool).tree_log(name)
    for step in steps:
        assert (
            f"/venv/bin/python {pintree.__file__} run --pin {pin_file} --job {name} "
            f"--record {record} -- bash "
        ) in step
    # The node's script no longer enters the tree that moves on: the pin's tree is its directory.
    inner = next((spool / "jobs").glob("*/node.sh")).read_text()
    assert "cd " not in inner and inner.rstrip().endswith("pytest -q")
    # The pin's tree is under the place the checkout names, on whichever node makes it.
    trees, _ = _trees(repo, tmp_path, root=place)
    with trees.at(trees.pinned(pin["commit"])) as tree:
        assert tree.parent == pintree.pins_root(place)


def test_a_checks_job_names_no_path_of_its_pins_tree_so_a_lead_that_has_none_runs_it(
    repo, tmp_path, monkeypatch
):
    """A job of several nodes is run by a lead with no tree of the pin: each step makes its own."""
    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    files.write_json_atomic(SpoolPaths(spool).pool, {"at": time.time(), "stopping": False})
    monkeypatch.setattr(spoolctl, "wait_for_job", lambda *a, **kw: {"rc": 0, "reason": "exit"})
    trees, _ = _trees(repo, tmp_path)
    pin = trees.pin()
    env = RunEnvironment("local", sys.executable, spool=str(spool))
    said = 'echo "ran in=$PWD pin=$MV2FLOW_PIN tree=${MV2FLOW_TREE:-none}"'
    scripts = {}
    with trees.at(pin) as tree:
        # The check's own runner, as a gate and the run make it, on the run's spool.
        check = CheckTree(pin.commit, tree, snapshots=trees)
        for nodes in (0, 2):
            executor = select_command_executor(
                env, gpus=1, nodes=nodes, once=nodes > 1, tree=tree, pin=pin.file
            )
            before = set((spool / "pending").glob("*.sh"))
            check.runner(shell_runner_for(executor), executor)(said, None, 9)
            (scripts[nodes],) = set((spool / "pending").glob("*.sh")) - before
    # The lead is another host: nothing of the pin is on its disk, and it makes nothing.
    pintree._discard(tree)
    node = [sys.executable, pintree.__file__, "run", "--pin", str(pin.file), "--job", "j"]
    lead = subprocess.run(
        [*node, "--steps", "--", "bash", str(scripts[2])],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert lead.returncode == 0, lead.stderr
    assert lead.stdout.strip() == f"ran in={tmp_path} pin={pin.file} tree=none"
    assert not tree.exists()
    # A job of one node: the node makes the pin's tree and the command runs inside it.
    one = subprocess.run(
        [*node, "--", "bash", str(scripts[0])], capture_output=True, text=True, cwd=tmp_path
    )
    assert one.returncode == 0, one.stderr
    assert one.stdout.strip() == f"ran in={tree} pin={pin.file} tree={tree}"
    for script in scripts.values():
        assert f" pin={pin.file}" in script.read_text() and str(tree) not in script.read_text()


def test_a_pin_taken_where_the_build_cannot_be_read_leaves_its_libraries_to_the_jobs_nodes(
    repo, tmp_path
):
    """The checkout's build is a link into an image: a host outside it pins, a node inside records."""
    image, outside = tmp_path / "image", tmp_path / "not-on-this-host"
    shutil.move(str(repo / "tensorrt_llm" / "libs"), str(outside))
    (repo / "tensorrt_llm" / "libs").symlink_to(image)
    (repo / ".gitignore").write_text("tensorrt_llm/libs\n")  # as the engine's own names it
    _git(repo, "commit", "-qam", "the build is linked")
    trees, said = _trees(repo, tmp_path)
    pin = trees.pin()
    # Where the link leads nowhere the pin records no library, and says which link it is.
    assert pin.build == {} and pintree.read_pin(pin.file)["unread"] == ["tensorrt_llm/libs"]
    assert [line for line in said if "cannot be read on this host" in line]
    outside.rename(image)
    record = tmp_path / "tree.jsonl"
    node = [sys.executable, pintree.__file__, "run", "--pin", str(pin.file), "--job", "j"]
    done = subprocess.run(
        [*node, "--record", str(record), "--", "bash", "-c", 'echo "libs=$MV2FLOW_TREE_LIBS"'],
        capture_output=True,
        text=True,
    )
    # On a node where it resolves, the job runs and the node records the libraries it found:
    # the identity its processes report, and the one every node of the job must agree on.
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"libs={LIB}" and "1 more recorded here" in done.stderr
    (entry,) = pintree.read_tree_records(record)
    assert entry["ok"] and entry["recorded_here"] == 1
    assert entry["build"] == pintree.record_digest({LIB: _sha(b"build-one!")})
    elsewhere = {**entry, "host": "n2", "build": pintree.record_digest({LIB: _sha(b"another")})}
    said = provenance.check_pinned([], [entry, elsewhere])
    assert said.startswith("the nodes of the job found different libraries for its pin")
    # A library the pin could have read is held to its record as before.
    ok, why, _ = pintree.verify({"build": {}, "unread": ["tensorrt_llm/other"]}, entry["tree"])
    assert not ok and f"{LIB} is not in the record" in why


def _queued(monkeypatch, executor, spool: Path, spec: CommandSpec) -> Path:
    """The script ``executor`` queues for ``spec``, as the file it left in the spool.

    Nobody watches the spool. The executor's wait for the job is answered at once, so the
    command is queued and returned from in this thread: nothing is polled against a clock.
    """
    files.write_json_atomic(SpoolPaths(spool).pool, {"at": time.time(), "stopping": False})
    monkeypatch.setattr(spoolctl, "wait_for_job", lambda *a, **kw: {"rc": 0, "reason": "exit"})
    before = set((spool / "pending").glob("*.sh"))
    executor.execute(spec)
    (new,) = set((spool / "pending").glob("*.sh")) - before
    return new


# ------------------------------------------- what a check reads, and the spec's block


def test_a_checks_spec_reads_every_input_of_the_checkout_from_the_pin(repo, tmp_path):
    trees, _ = _trees(repo, tmp_path)
    pin = trees.pin()
    with trees.at(pin) as tree:
        spec = {
            "trtllm_repo_path": str(repo),
            "deployments": [
                {
                    "name": "d",
                    "options": str(repo / "docs/page.md"),
                    "benchmarks": [
                        {"name": "b", "dataset_path": str(repo / "cpp/kernels/rmsnorm.cu")}
                    ],
                }
            ],
            "environment": {"python": str(repo / ".venv/bin/python")},
        }
        moved = CheckTree(pin.commit, tree, snapshots=trees).spec(spec, trees.tracked(pin.commit))
    block = moved["deployments"][0]
    assert moved["trtllm_repo_path"] == str(tree) and block["options"] == str(tree / "docs/page.md")
    assert block["benchmarks"][0]["dataset_path"] == str(tree / "cpp/kernels/rmsnorm.cu")
    assert moved["environment"]["python"] == str(repo / ".venv/bin/python"), "untracked: stays"


def test_a_tree_holds_an_lfs_file_as_its_pointer_and_its_making_fetches_nothing(repo, tmp_path):
    """Where git fills LFS files in and their store cannot be reached, a tree is still made."""
    # git-lfs in small, under a name of its own so that the host's git-lfs has no say: it
    # passes a pointer on only when told to skip, and here it could fetch nothing.
    fetch = tmp_path / "smudge"
    fetch.write_text('#!/bin/sh\n[ "$GIT_LFS_SKIP_SMUDGE" = 1 ] && exec cat\nexit 2\n')
    fetch.chmod(0o755)
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n"
    (repo / ".gitattributes").write_text("*.png filter=stand-in\n")
    (repo / "image.png").write_text(pointer)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "an image")
    _git(repo, "config", "filter.stand-in.smudge", str(fetch))
    _git(repo, "config", "filter.stand-in.required", "true")
    trees, _ = _trees(repo, tmp_path)
    assert (trees.open("task") / "image.png").read_text() == pointer  # a worker's tree
    with trees.at(trees.pin()) as tree:  # a pin's tree, as a check started by hand makes it
        assert (tree / "image.png").read_text() == pointer


def test_a_runs_first_jobs_pin_under_its_own_workspace_whoever_used_the_checkout_before(
    repo, tmp_path
):
    """A check by hand left its place in the checkout: the run that starts next names its own."""
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    by_hand = Integration(repo, tmp_path / "control", "mv2/gate", ENV)
    by_hand.pin()  # the check that measured an anchor, in a directory of its own
    assert str(tmp_path / "control") in pintree.pin_tree(str(repo), "probe")
    shutil.rmtree(tmp_path / "control")  # done with, and removed
    run = SimpleNamespace(task_data={"trtllm_repo_path": str(repo)}, workspace=tmp_path / "ws")
    bootstrap._state_trees(run)
    pin = pintree.pin_tree(str(repo), "preflight-cpu")
    assert Path(pin).parent == tmp_path / "ws" / "trees" / "pins"
    assert not (tmp_path / "control").exists()


def test_the_spec_accepts_a_trees_block_and_says_what_is_wrong_with_one(tmp_path):
    def problems(trees):
        block = {"kind": "local", "python": "/py", "trees": trees}
        errors: list[str] = []
        _validate_environment(block, errors)
        return [e for e in errors if "trees" in e]

    assert problems({"root": str(tmp_path)}) == []
    [said] = problems({"root": "relative/place"})
    assert said == "'environment.trees.root' must be an absolute path: where the run's trees go"
