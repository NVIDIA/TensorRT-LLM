# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The test-result cache: a change re-runs only the tests it can affect, and a regression is named."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import testcache as tc
from agent_flow.workflows.trtllm_modeling_v2.checks import testcache_driver as drv
from agent_flow.workflows.trtllm_modeling_v2.run import receipts as receipts_mod

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")

_FILES = {
    "pkg/__init__.py": "",
    "pkg/consts.py": "SCALE = 2\n",
    "pkg/ops.py": textwrap.dedent(
        """\
        from .consts import SCALE

        _CACHE = {}


        def f(x):
            return x * SCALE


        def g(x):
            return x + 1


        def _build(x):
            return x * 10


        def h(x):
            if x not in _CACHE:
                _CACHE[x] = _build(x)
            return _CACHE[x]
        """
    ),
    "tests/test_ops.py": textwrap.dedent(
        """\
        from pkg.ops import f, g, h


        def test_f():
            assert f(2) == 4


        def test_g():
            assert g(1) == 2


        def test_h1():
            assert h(1) == 10


        def test_h2():
            assert h(1) == 10
        """
    ),
    "tests/test_broken.py": "def test_broken():\n    assert False, 'broken before the run'\n",
}


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


def _drive(repo: Path, cache: Path, *items: str, cwd: Path | None = None) -> None:
    done = subprocess.run(
        [
            sys.executable,
            str(drv.__file__),
            "run",
            "--cache",
            str(cache),
            "--repo",
            str(cwd or repo),
        ]
        + list(items)
        + ["--", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """The repository, its base commit and the cache holding one recorded run of every test file."""
    root = tmp_path_factory.mktemp("testcache")
    repo, cache = root / "repo", root / "cache"
    for path, content in _FILES.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _drive(repo, cache, *sorted(p for p in _FILES if p.startswith("tests/")))
    return repo, cache, base


def _plan(recorded, changes: dict[str, str]) -> tuple[str, dict[str, tc.FilePlan]]:
    repo, cache, base = recorded
    head = _commit(repo, base, changes, message=json.dumps(sorted(changes)))
    report = tc.affected(tc.ResultCache(cache, repo), head, base=base, paths=["tests"])
    return head, {p.file: p for p in report.files}


def _ran(plans: dict[str, tc.FilePlan]) -> set[str]:
    """Every test id to run, a whole file as ``<file>::*``."""
    out = set()
    for plan in plans.values():
        out |= {f"{plan.file}::*"} if plan.whole else set(plan.run)
    return out


def _reused(plans: dict[str, tc.FilePlan]) -> set[str]:
    return {r.nodeid for p in plans.values() for r in p.reused}


def test_a_function_body_affects_the_tests_from_the_first_that_ran_it(recorded):
    """test_f ran before g first did, so its result holds; test_g's does not."""
    source = _FILES["pkg/ops.py"].replace("return x + 1", "return x + 2")
    _, plans = _plan(recorded, {"pkg/ops.py": source})
    assert _ran(plans) == {
        "tests/test_ops.py::test_g",
        "tests/test_ops.py::test_h1",  # later in the same process: they could read what g left
        "tests/test_ops.py::test_h2",
    }
    assert "tests/test_ops.py::test_f" in _reused(plans)
    assert (
        plans["tests/test_ops.py"].reasons["tests/test_ops.py::test_g"] == "pkg/ops.py: g changed"
    )


def test_compare_names_the_regression_and_reuses_the_rest(recorded, tmp_path):
    repo, cache, base = recorded
    source = _FILES["pkg/ops.py"].replace("return x + 1", "return x + 2")
    new = "def test_new():\n    assert False, 'new and failing'\n"
    head = _commit(
        repo, base, {"pkg/ops.py": source, "tests/test_new.py": new}, message="regress g"
    )
    tree = tmp_path / "head"
    _git(repo, "worktree", "add", "-q", "--detach", str(tree), head)
    try:
        selection = tc.affected(
            tc.ResultCache(cache, repo), head, base=base, paths=["tests"]
        ).selection()
        assert "tests/test_ops.py::test_f" not in selection
        assert "tests/test_new.py" in selection
        _drive(repo, cache, *selection, cwd=tree)
        env = tc.head_environment(tc.load_sessions(cache), head)
        report = tc.compare(
            tc.ResultCache(cache, repo, env_digest=env),
            base,
            head,
            paths=["tests/test_ops.py", "tests/test_new.py", "tests/test_broken.py"],
        )
    finally:
        _git(repo, "worktree", "remove", "--force", str(tree))
    assert report.verdict == "REGRESSED" and report.exit_code == tc.EXIT_REGRESSED
    assert [c.nodeid for c in report.regressions] == [
        "tests/test_new.py::test_new",
        "tests/test_ops.py::test_g",
    ]
    assert report.regressions[0].new and report.regressions[1].base.outcome == "passed"
    assert [c.nodeid for c in report.pre_existing] == ["tests/test_broken.py::test_broken"]
    assert report.env_checked and report.reused == 2 and report.ran_at_head == 4


def _regression_repo(root: Path) -> tuple[Path, Path, str]:
    """A repository whose base commit's tests are recorded in a cache, and that cache."""
    repo, cache = root / "repo", root / "ws" / tc.CACHE_DIRNAME
    files = {
        "pkg/__init__.py": "",
        "pkg/mod.py": "def f():\n    return 1\n",
        "tests/test_a.py": "from pkg.mod import f\n\n\ndef test_f():\n    assert f() == 1\n",
        "tests/test_b.py": "def test_alone():\n    assert True\n",
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
    _drive(repo, cache, "tests/test_a.py", "tests/test_b.py")
    # The last check cleared the base: the next one compares against it.
    receipts_mod.write(
        root / "ws" / "tasks",
        {"task": "regression", "kind": "check", "cleared": base, "written_at": "1"},
    )
    return repo, root / "ws", base


def _check(repo: Path, workspace: Path, head: str, root: Path):
    import contextlib

    from agent_flow.workflows.trtllm_modeling_v2.run import regression

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
        # pytest-timeout is not a test dependency here.
        done = subprocess.run(
            command.replace(" --timeout=180", ""), shell=True, capture_output=True, text=True
        )
        return done.returncode, done.stdout + done.stderr

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


def test_a_regression_check_runs_only_what_changed_and_names_what_regressed(tmp_path):
    repo, workspace, base = _regression_repo(tmp_path)
    head = _commit(repo, base, {"pkg/mod.py": "def f():\n    return 2\n"}, message="break f")
    result, ran = _check(repo, workspace, head, tmp_path)
    assert result.verdict == "REGRESSED" and not result.passed
    assert result.base == base and result.ran == 1 and result.reused == 1
    [failure] = result.failures
    assert failure.startswith("tests/test_a.py::test_f (passed at the base, fails now): ")
    assert "2 == 1" in failure
    assert len(ran) == 1 and "--selection" in ran[0]
    selection = (workspace / "regression" / "regression-2" / "head.txt").read_text()
    assert selection.split() == ["tests/test_a.py"]  # its only test: the whole file
