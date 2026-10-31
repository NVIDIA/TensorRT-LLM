# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The regression check: the whole declared list at the head, against its results at the base."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import test_results as tc
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import test_driver as drv
from agent_flow.workflows.trtllm_modeling_v2.run.execution import handoff as handoff_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **(env or {})},
    )
    return done.stdout.strip()


def _commit(repo: Path, base: str, changes: dict[str, str], message: str = "change") -> str:
    """A commit on ``base`` with ``changes``, made without touching the checkout."""
    index = repo / ".git" / f"index-{abs(hash(message)) % 10**8}"
    env = {"GIT_INDEX_FILE": str(index)}
    _git(repo, "read-tree", base, env=env)
    for path, content in changes.items():
        blob = subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=content,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
    tree = _git(repo, "write-tree", env=env)
    index.unlink(missing_ok=True)
    return _git(repo, "commit-tree", tree, "-p", base, "-m", message)


def _run(command: str) -> tuple[int, str]:
    # pytest-timeout is not a test dependency here.
    done = subprocess.run(
        command.replace(" --timeout=180", ""), shell=True, capture_output=True, text=True
    )
    return done.returncode, done.stdout + done.stderr


def _regression_repo(root: Path) -> tuple[Path, Path, str]:
    """A repository whose base commit's ``test_a`` and ``test_b`` are recorded, and the workspace."""
    repo, workspace = root / "repo", root / "ws"
    files = {
        "pkg/__init__.py": "",
        "pkg/mod.py": "def f():\n    return 1\n",
        "tests/test_a.py": "from pkg.mod import f\n\n\ndef test_f():\n    assert f() == 1\n",
        "tests/test_b.py": "def test_alone():\n    assert True\n",
        "tests/test_broken.py": "def test_broken():\n    assert False, 'broken before the run'\n",
    }
    for path, content in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    cache = workspace / tc.CACHE_DIRNAME
    _run(f"{sys.executable} {drv.__file__} run --cache {cache} --repo {repo} tests/test_a.py "
         "tests/test_b.py -- -q -p no:cacheprovider")  # fmt: skip
    # The last check cleared the base: the next one compares against it.
    handoff_mod.write(
        workspace / "tasks",
        {"task": "regression", "kind": "check", "cleared": base, "written_at": "1"},
    )
    return repo, workspace, base


def _check(repo: Path, workspace: Path, head: str, root: Path):
    ran = []

    @contextlib.contextmanager
    def tree_at(commit):
        tree = root / f"tree-{commit[:8]}-{len(ran)}"
        _git(repo, "worktree", "add", "-q", "--detach", str(tree), commit)
        try:
            yield tree
        finally:
            _git(repo, "worktree", "remove", "--force", str(tree))

    def run_job(command, tree):
        ran.append(command)
        return _run(command)

    result = regression.run_regression_check(
        workspace=workspace,
        repo=repo,
        head=head,
        tree_at=tree_at,
        run_job=run_job,
        python=sys.executable,
        name="regression-2",
        paths=["tests"],
    )
    return result, ran


def test_environment_change_cannot_reuse_a_pass_and_probe_failure_is_incomplete(
    tmp_path, monkeypatch
):
    repo, workspace, base = _regression_repo(tmp_path)
    monkeypatch.setenv("TRTLLM_TEST_CACHE_ENV", "A")
    first, _ = _check(repo, workspace, base, tmp_path)
    assert first.verdict == "CLEAN"
    a = tc.recorded(workspace / tc.CACHE_DIRNAME, base).env_digest
    monkeypatch.setenv("TRTLLM_TEST_CACHE_ENV", "B")
    second, ran = _check(repo, workspace, base, tmp_path)
    assert second.verdict == "CLEAN" and len(ran) == 2
    assert tc.recorded(workspace / tc.CACHE_DIRNAME, base).env_digest != a
    results, jobs, code, tail = regression.record_at(
        base,
        ["tests/test_a.py"],
        cache_dir=workspace / tc.CACHE_DIRNAME,
        out=tmp_path / "failed-probe",
        tree_at=lambda commit: contextlib.nullcontext(repo),
        run_job=lambda command, tree: (1, "unavailable"),
        python=sys.executable,
    )
    assert not results.env_digest and results.unrun(["tests/test_a.py"])
    assert jobs == 0 and code == 1 and "environment probe failed" in tail


def test_a_change_reaches_the_tests_that_import_it_through_other_files(tmp_path):
    """``catalog/_op.py`` reaches a test through the entry that imports it, or the helper that does."""
    repo = tmp_path / "repo"
    catalog, tests = regression.CATALOG_DIR, regression.CATALOG_TESTS_DIR
    dotted = catalog.replace("/", ".")
    files = {
        f"{catalog}/_op.py": "class OpWrapper:\n    pass\n",
        f"{catalog}/gemm/mm.py": "from .._op import OpWrapper\n",
        f"{catalog}/norm/rms.py": "X = 1\n",
        f"{tests}/_validating.py": f"from {dotted}._op import OpWrapper\n",
        f"{tests}/gemm/test_mm.py": f"from {dotted}.gemm import mm\n",
        f"{tests}/norm/test_rms.py": f"from _validating import validating\nfrom {dotted}.norm.rms import X\n",
        f"{tests}/norm/test_untouched.py": f"from {dotted}.norm.rms import X\n",
        # A rank matrix the test starts by file path, through a launcher it imports.
        f"{tests}/comm/_rank_job.py": "import subprocess\n",
        f"{tests}/comm/_mm_matrix.py": f"def body():\n    from {dotted}.gemm import mm\n",
        f"{tests}/comm/test_mm_matrix.py": "import _rank_job\n",
        f"{tests}/test_modeling_v2_claims.py": "",
    }
    for path, content in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    listed = sorted(p for p in files if "/test_" in p)
    head = _commit(repo, base, {f"{catalog}/_op.py": "class OpWrapper:\n    x = 1\n"})
    picked, why = regression.affected_files(repo, base, head, listed)
    assert picked == [
        f"{tests}/comm/test_mm_matrix.py",
        f"{tests}/gemm/test_mm.py",
        f"{tests}/norm/test_rms.py",
        f"{tests}/test_modeling_v2_claims.py",
    ]
    assert why == "4 of 5 test files reach what changed"
