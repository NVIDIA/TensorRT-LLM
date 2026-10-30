# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate worktrees are registered under the repo lock and checked out side by side."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import native_artifacts, open_candidate


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_candidates_open_in_parallel_with_their_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    for i in range(20):
        (repo / f"f{i}.py").write_text(f"X = {i}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    slugs = [f"cand-{i}" for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        trees = list(
            pool.map(
                lambda s: open_candidate(repo, tmp_path / "wt", s, "HEAD", prefix="tune/run"), slugs
            )
        )
    for slug, tree in zip(slugs, trees):
        assert (tree / "f19.py").read_text() == "X = 19\n"
        assert _git(tree, "status", "--porcelain") == ""
        assert _git(tree, "rev-parse", "--abbrev-ref", "HEAD") == f"tune/run/{slug}"


def test_a_half_created_worktree_is_rebuilt(tmp_path):
    """An orchestrator stopped mid `git worktree add` leaves the tree locked 'initializing' with no index.

    The next open rebuilds it instead of handing a tuner an empty checkout.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("A = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    tree = tmp_path / "wt" / "cand"
    _git(repo, "worktree", "add", "--no-checkout", "-b", "tune/run/cand", str(tree), "HEAD")
    gitdir = _git(tree, "rev-parse", "--absolute-git-dir")
    (Path(gitdir) / "locked").write_text("initializing")
    (Path(gitdir) / "index").unlink(missing_ok=True)
    assert open_candidate(repo, tmp_path / "wt", "cand", "HEAD", prefix="tune/run") == tree
    assert (tree / "a.py").read_text() == "A = 1\n" and _git(tree, "status", "--porcelain") == ""


def test_the_native_build_is_what_git_ignores_under_the_package(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text("tensorrt_llm/bindings/\ntensorrt_llm/*.so\n")
    (repo / "tensorrt_llm" / "__init__.py").write_text("")
    (repo / "tensorrt_llm" / "bindings").mkdir()
    (repo / "tensorrt_llm" / "bindings" / "__init__.pyi").write_text("")
    (repo / "tensorrt_llm" / "libnative.so").write_text("elf")
    assert native_artifacts(repo) == ["tensorrt_llm/bindings", "tensorrt_llm/libnative.so"]
