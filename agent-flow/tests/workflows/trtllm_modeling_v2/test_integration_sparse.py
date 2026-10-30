# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse task worktrees on a real git repository: a cone on disk, the whole tree in the branch."""

from __future__ import annotations

import shlex
import subprocess
import threading
from pathlib import Path

import pytest

import agent_flow.workflows.trtllm_modeling_v2.run.integration as integration_mod
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.integration import (
    SPARSE_CONE,
    SPARSE_ON,
    Integration,
    cone,
    sparse_cone,
    worktree_block,
)

from .test_integration import _Stage

_ENV = {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}

#: Files of a small TensorRT-LLM-shaped checkout, and which the default cone leaves off disk.
IN_CONE = (
    "README.md",
    "setup.py",
    "tensorrt_llm/__init__.py",
    "tensorrt_llm/models/modeling.py",
    "tests/conftest.py",
    "tests/unittest/conftest.py",
    "tests/unittest/_torch/test_modeling.py",
    "tests/unittest/utils/util.py",
    "tests/test_common/hooks.py",
    "tests/integration/defs/conftest.py",
    "tests/integration/test_lists/l0.yml",
    "examples/llm-api/quickstart_advanced.py",
    "examples/visual_gen/cat.png",
)
OUTSIDE = (
    "cpp/CMakeLists.txt",
    "cpp/kernels/rmsnorm.cu",
    "3rdparty/lib/lib.h",
    "docs/guide/page.md",
    "docs/other/notes.md",
    "scripts/build.py",
    "tests/unittest/llmapi/test_llm_args.py",
)


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
    for path in IN_CONE + OUTSIDE:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(f"{path} 1\n")
    (repo / ".gitignore").write_text("tensorrt_llm/*.so\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "tensorrt_llm" / "libnative.so").write_text("elf")  # the native build
    return repo


def _on_disk(tree: Path) -> set[str]:
    return {
        str(p.relative_to(tree))
        for p in tree.rglob("*")
        if (p.is_file() or p.is_symlink()) and ".git" not in p.relative_to(tree).parts
    }


def _commit(tree: Path, path: str, text: str, message: str) -> None:
    (tree / path).write_text(text)
    _git(tree, "add", path)
    _git(tree, "commit", "-q", "-m", f"{message}\n\nSigned-off-by: T <t@example.com>")


def test_a_task_worktree_checks_out_the_cone_and_links_the_native_build(tmp_path):
    repo = _repo(tmp_path)
    tree = Integration(repo, tmp_path / "ws", "mv2/run", _ENV).open("k1")
    assert sparse_cone(tree) == SPARSE_CONE
    assert _on_disk(tree) == {*IN_CONE, ".gitignore", "tensorrt_llm/libnative.so"}
    assert (tree / "tensorrt_llm" / "libnative.so").resolve() == repo / "tensorrt_llm/libnative.so"
    # What is not on disk is still the branch's, and git does not read it as deleted.
    assert _git(tree, "status", "--porcelain", "--untracked-files=no") == ""
    assert set(_git(tree, "ls-files").split()) >= set(OUTSIDE)


def test_the_directories_a_task_touches_widen_its_cone(tmp_path):
    repo = _repo(tmp_path)
    tree = Integration(repo, tmp_path / "ws", "mv2/run", _ENV).open(
        "f1", ["docs/guide/page.md", "scripts/", "tests/unittest/llmapi/test_llm_args.py"]
    )
    on_disk = _on_disk(tree)
    assert {"docs/guide/page.md", "scripts/build.py", "tests/unittest/llmapi/test_llm_args.py"} <= (
        on_disk
    )
    assert "docs/other/notes.md" not in on_disk and "cpp/CMakeLists.txt" not in on_disk


@pytest.mark.parametrize(
    ("touches", "full"),
    [(["cpp/kernels/rmsnorm.cu"], False), (["3rdparty/lib/"], False), ([], True)],
    ids=["cpp", "3rdparty", "exclusive"],
)
def test_native_or_exclusive_work_gets_the_whole_tree(tmp_path, touches, full):
    repo = _repo(tmp_path)
    tree = Integration(repo, tmp_path / "ws", "mv2/run", _ENV).open("k1", touches, full=full)
    assert sparse_cone(tree) == ()
    assert set(IN_CONE + OUTSIDE) <= _on_disk(tree)


def test_a_commit_from_a_sparse_worktree_leaves_every_file_outside_the_cone_intact(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("k1")
    # The checkout moves on outside the cone while the task works, so integrating is a real rebase.
    (repo / "cpp" / "kernels" / "rmsnorm.cu").write_text("rmsnorm 2\n")
    (repo / "docs" / "new.md").write_text("new\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "native")
    expected = {p: (repo / p).read_text() for p in _git(repo, "ls-files").split()}
    (tree / "tensorrt_llm" / "models" / "modeling.py").write_text("modeling 2\n")
    _git(tree, "commit", "-q", "-am", "model\n\nSigned-off-by: T <t@example.com>")

    outcome = integration.integrate("k1")
    assert outcome.merged and outcome.commits == 1
    expected["tensorrt_llm/models/modeling.py"] = "modeling 2\n"
    assert set(_git(repo, "ls-files").split()) == set(expected)
    for path, text in expected.items():
        assert (repo / path).read_text() == text, path
        assert _git(repo, "show", f"HEAD:{path}") + "\n" == text, path
    assert _git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    integration.close("k1")
    assert not tree.exists()


def test_a_reopened_worktree_widens_to_what_its_task_touches_now(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.open("f1")
    _commit(tree, "tensorrt_llm/models/modeling.py", "modeling 2\n", "model")
    assert integration.open("f1", ["docs/guide/"]) == tree
    assert "docs/guide/page.md" in _on_disk(tree) and "docs/guide" in sparse_cone(tree)
    assert integration.open("f1", ["cpp/kernels/"]) == tree
    assert sparse_cone(tree) == () and "cpp/kernels/rmsnorm.cu" in _on_disk(tree)
    assert (tree / "tensorrt_llm" / "models" / "modeling.py").read_text() == "modeling 2\n"
    assert _git(tree, "status", "--porcelain", "--untracked-files=no") == ""


def test_a_worktree_whose_checkout_was_cut_short_is_made_again(tmp_path):
    """A run stopped mid-checkout leaves a registered tree with no index and missing files."""
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.tree("k1")
    _git(repo, "worktree", "add", "-q", "--no-checkout", "-b", "mv2/run/k1", str(tree), "HEAD")
    assert integration.open("k1") == tree
    assert _on_disk(tree) == {*IN_CONE, ".gitignore", "tensorrt_llm/libnative.so"}
    assert _git(tree, "status", "--porcelain", "--untracked-files=no") == ""


def test_a_whole_worktree_made_before_sparse_ones_is_kept_as_it_is(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    tree = integration.tree("k1")
    _git(repo, "worktree", "add", "-q", "-b", "mv2/run/k1", str(tree), "HEAD")
    (tree / "docs" / "guide" / "page.md").write_text("work in progress\n")
    assert integration.open("k1", ["tensorrt_llm/models/"]) == tree
    assert sparse_cone(tree) == () and set(OUTSIDE) <= _on_disk(tree)
    assert (tree / "docs" / "guide" / "page.md").read_text() == "work in progress\n"


def test_worktrees_opened_together_check_out_side_by_side(tmp_path, monkeypatch):
    """Only registering a worktree is serialised; every checkout runs while the others do."""
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    count = 4
    together = threading.Barrier(count, timeout=60)
    real = integration_mod.git_checked
    checkouts = []

    def checkout_waits_for_the_others(cwd, *args, check=True):
        if "reset" in args:
            checkouts.append(args)
            together.wait()  # breaks if one checkout held the repository lock
        return real(cwd, *args, check=check)

    monkeypatch.setattr(integration_mod, "git_checked", checkout_waits_for_the_others)
    trees, errors = {}, []

    def open_one(task_id):
        try:
            trees[task_id] = integration.open(task_id)
        except Exception as exc:  # noqa: BLE001 - reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=open_one, args=(f"k{i}",)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(checkouts) == count
    for task_id, tree in trees.items():
        assert sparse_cone(tree) == SPARSE_CONE
        assert _git(tree, "rev-parse", "--abbrev-ref", "HEAD") == f"mv2/run/{task_id}"
        assert "tensorrt_llm/models/modeling.py" in _on_disk(tree)


def test_a_sparse_worker_is_told_how_to_widen_its_checkout(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    sparse, whole = integration.open("k1"), integration.open("k2", ["cpp/"])
    widen = f"git -C {sparse} {' '.join(SPARSE_ON)} sparse-checkout add <dir>"
    for scratch in (False, True):
        block = worktree_block(sparse, repo, "/ws/bin/commit.sh", scratch=scratch)
        assert widen in block and "Keep the `-c` options" in block
        assert "`tensorrt_llm/`" in block
        assert "sparse-checkout" not in worktree_block(whole, repo, "c", scratch=scratch)
    # The command the worker is given is one that works.
    subprocess.run(shlex.split(widen.replace("<dir>", "docs/guide")), check=True)
    assert "docs/guide/page.md" in _on_disk(sparse) and "docs/guide" in sparse_cone(sparse)
    assert _git(sparse, "status", "--porcelain", "--untracked-files=no") == ""


def test_the_planned_stage_opens_a_task_worktree_with_its_touches(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    stage = _Stage(tmp_path / "ws", integration, lambda turn: "SCRATCH_DONE")
    task = g.Task(id="i1", kind="investigate", goal="why", touches=["docs/guide/page.md"])
    assert stage._work_task_turn(task) == "SCRATCH_DONE"
    tree = integration.tree("i1")
    assert "docs/guide" in sparse_cone(tree) and "sparse-checkout add" in stage.blocks[0]
    native = g.Task(id="n1", kind="investigate", goal="build", exclusive=True)
    stage._work_task_turn(native)
    assert sparse_cone(integration.tree("n1")) == ()


@pytest.mark.parametrize(
    ("touches", "added"),
    [
        (["tensorrt_llm/_torch/models/modeling_x.py"], ()),  # inside the package already
        (["docs/source/models/supported-models.md"], ("docs/source/models",)),
        (["docs/source/"], ("docs/source",)),
        (["docker/Makefile"], ("docker/Makefile",)),  # its parent's files come with it
        (["examples/models/core/llama3.1"], ()),
        (["scripts/gen_*.py"], ("scripts",)),
        (["setup.py", "*.md", "../outside", "/abs/path", "."], ()),
        (["docs/a/", "docs/a/b/c.md", "docs/"], ("docs",)),
    ],
)
def test_the_cone_holds_each_touched_directory_once(touches, added):
    assert cone(touches) == tuple(sorted(SPARSE_CONE + added))


def test_native_paths_and_exclusive_tasks_ask_for_the_whole_tree():
    assert cone(["cpp/tensorrt_llm/kernels/rmsnorm.cu"]) is None
    assert cone(["3rdparty/"]) is None and cone(["cpp"]) is None
    assert cone(["tensorrt_llm/"], full=True) is None
    assert cone(["cppfoo/x.py"]) == tuple(sorted(SPARSE_CONE + ("cppfoo",)))


def test_the_cone_holds_every_top_level_package_the_package_imports():
    """A sparse worktree must import: each repository package ``tensorrt_llm`` imports is in it."""
    import re as _re

    from agent_flow.workflows.trtllm_modeling_v2.run.integration import SPARSE_CONE

    root = Path(__file__).resolve().parents[4]
    package = root / "tensorrt_llm"
    if not package.is_dir():
        pytest.skip("this checkout has no tensorrt_llm/ beside agent-flow/")
    top = {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    pattern = _re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", _re.MULTILINE)
    imported = set()
    for source in package.rglob("*.py"):
        imported.update(pattern.findall(source.read_text(encoding="utf-8", errors="replace")))
    needed = sorted((top & imported) - {"tensorrt_llm"})
    assert needed, "expected tensorrt_llm to import at least one sibling package"
    assert [name for name in needed if name not in SPARSE_CONE] == []
