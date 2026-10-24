# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse task worktrees on a real git repository: a cone on disk, the whole tree in the branch."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run.integration import SPARSE_CONE, Integration, cone

from .test_integration import _ENV, _git

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


def test_a_task_worktree_checks_out_the_cone_and_links_the_native_build(tmp_path):
    repo = _repo(tmp_path)
    tree = Integration(repo, tmp_path / "ws", "mv2/run", _ENV).open("k1")
    assert _on_disk(tree) == {*IN_CONE, ".gitignore", "tensorrt_llm/libnative.so"}
    assert (tree / "tensorrt_llm" / "libnative.so").resolve() == repo / "tensorrt_llm/libnative.so"
    # What is not on disk is still the branch's, and git does not read it as deleted.
    assert _git(tree, "status", "--porcelain", "--untracked-files=no") == ""
    assert set(_git(tree, "ls-files").split()) >= set(OUTSIDE)


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


@pytest.mark.parametrize(
    ("touches", "added"),
    [
        (["tensorrt_llm/_torch/models/modeling_x.py"], ()),  # inside the package already
        (["docs/source/models/supported-models.md"], ("docs/source/models",)),
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
