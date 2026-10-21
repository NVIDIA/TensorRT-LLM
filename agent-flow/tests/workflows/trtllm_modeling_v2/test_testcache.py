# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The test-result cache: a change re-runs only the tests it can affect, and every way a test reads its inputs is seen.

One small repository is recorded once at its base commit, one pytest
process per test file, by the real driver. Each change below is a commit on
top of it (made with git plumbing, so the checkout stays at the base) and is
read against that one recording: the unaffected results are reused, and a
change that would make a test fail always makes it affected -- through a
function body, a module-level constant, a data file, a memoized result, a
collection-time parametrization, a file a failed import would now find, a
path whose existence was asked, a native source, or a process it started.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

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
    "pkg/cells.py": "def cells():\n    return [1, 2]\n",
    "data/x.json": '{"v": 1}\n',
    "cpp/kernel.cu": "// kernel\n",
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
    "tests/test_inputs.py": textwrap.dedent(
        """\
        import json
        import os

        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


        def test_probe():
            assert not os.path.exists(os.path.join(ROOT, "pkg", "flag.txt"))


        def test_import():
            try:
                from pkg import fast
            except ImportError:
                fast = None
            assert fast is None


        def test_data():
            with open(os.path.join(ROOT, "data", "x.json")) as handle:
                assert json.load(handle)["v"] == 1
        """
    ),
    "tests/test_cells.py": textwrap.dedent(
        """\
        import pytest

        from pkg.cells import cells


        @pytest.mark.parametrize("c", cells())
        def test_cell(c):
            assert c < 3
        """
    ),
    "tests/test_proc.py": textwrap.dedent(
        """\
        import subprocess
        import sys


        def test_git():
            subprocess.run(["git", "--version"], check=True, capture_output=True)


        def test_python():
            subprocess.run([sys.executable, "-c", "print(1)"], check=True, capture_output=True)


        def test_after():
            assert True
        """
    ),
    "tests/test_pool.py": textwrap.dedent(
        """\
        import multiprocessing
        import os
        from concurrent.futures import ThreadPoolExecutor


        def test_threads():
            assert multiprocessing.cpu_count() >= 1
            with ThreadPoolExecutor(2) as pool:
                assert list(pool.map(abs, [-1])) == [1]


        def test_spawned_process():
            process = multiprocessing.get_context("spawn").Process(target=os.getpid)
            process.start()
            process.join()
            assert process.exitcode == 0
        """
    ),
    "tests/test_native.py": textwrap.dedent(
        """\
        import ctypes
        import os

        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


        def test_loads_the_library():
            ctypes.CDLL(os.path.join(ROOT, "lib", "fake.so"))
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


def _commit(repo: Path, base: str, changes: dict[str, str | None], message: str = "change") -> str:
    """A commit on ``base`` with ``changes`` (None deletes), made without touching the checkout."""
    index = repo / ".git" / f"index-{abs(hash(message)) % 10**8}"
    env = {"GIT_INDEX_FILE": str(index)}
    _git(repo, "read-tree", base, env=env)
    for path, content in changes.items():
        if content is None:
            _git(repo, "update-index", "--force-remove", path, env=env)
            continue
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
    import _ctypes

    (repo / "lib").mkdir()
    shutil.copy(_ctypes.__file__, repo / "lib" / "fake.so")  # a built library, never tracked
    (repo / ".gitignore").write_text("lib/\n__pycache__/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _drive(repo, cache, *sorted(p for p in _FILES if p.startswith("tests/")))
    return repo, cache, base


def _plan(recorded, changes: dict[str, str | None], **kw) -> tuple[str, dict[str, tc.FilePlan]]:
    repo, cache, base = recorded
    head = _commit(repo, base, changes, message=json.dumps(sorted(changes)))
    report = tc.affected(tc.ResultCache(cache, repo, **kw), head, base=base, paths=["tests"])
    return head, {p.file: p for p in report.files}


def _ran(plans: dict[str, tc.FilePlan]) -> set[str]:
    """Every test id to run, a whole file as ``<file>::*``."""
    out = set()
    for plan in plans.values():
        out |= {f"{plan.file}::*"} if plan.whole else set(plan.run)
    return out


def _reused(plans: dict[str, tc.FilePlan]) -> set[str]:
    return {r.nodeid for p in plans.values() for r in p.reused}


#: Opaque from the process each starts on: re-run at every commit but the one it ran at.
_OPAQUE = {
    "tests/test_proc.py::test_python",
    "tests/test_proc.py::test_after",
    "tests/test_pool.py::test_spawned_process",
}


# ------------------------------------------------------------------ recording
def test_a_session_records_each_test_with_what_it_ran_on(recorded):
    _, cache, base = recorded
    sessions = {next(iter(s.collected)): s for s in tc.load_sessions(cache) if s.collected}
    ops = sessions["tests/test_ops.py"]
    assert ops.commit == base and ops.clean is True
    assert [ops.tests[f"tests/test_ops.py::{t}"]["index"] for t in ("test_f", "test_g")] == [0, 1]
    assert ops.deps["loaded"]["pkg/consts.py"] == -1  # imported while collecting
    assert ops.deps["executed"]["pkg/ops.py"] == {"f": 0, "g": 1, "h": 2, "_build": 2}
    assert ops.opaque is None
    inputs = sessions["tests/test_inputs.py"]
    assert inputs.deps["read"]["data/x.json"] == 2
    assert inputs.deps["probed"]["pkg/flag.txt"] == 0
    assert inputs.deps["imports"]["pkg.fast"] == 1
    assert sessions["tests/test_proc.py"].opaque[0] == 1  # git is not a process that runs code
    # A spawned multiprocessing child starts without an audit event; threads start no process.
    assert sessions["tests/test_pool.py"].opaque == (
        1,
        "hands work to other processes (BaseProcess.start)",
    )
    assert "lib/fake.so" in sessions["tests/test_native.py"].deps["native"]
    broken = sessions["tests/test_broken.py"].tests["tests/test_broken.py::test_broken"]
    assert broken["outcome"] == "failed" and "broken before the run" in broken["message"]


def test_the_profile_hook_sees_what_sys_monitoring_sees(recorded, tmp_path):
    """The tracer of an interpreter without ``sys.monitoring`` records the same functions."""
    repo, _, _ = recorded
    cache = tmp_path / "cache"
    code = drv.main(
        [
            "run",
            "--cache",
            str(cache),
            "--repo",
            str(repo),
            "--tracer",
            "profile",
            "tests/test_ops.py",
            "--",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
    )
    assert code == 0
    (session,) = tc.load_sessions(cache)
    assert json.loads(session.path.read_text(encoding="utf-8"))["tracer"] == "profile"
    assert session.deps["executed"]["pkg/ops.py"] == {"f": 0, "g": 1, "h": 2, "_build": 2}


def test_nothing_selected_still_records_the_environment(tmp_path):
    cache = tmp_path / "cache"
    assert drv.main(["run", "--cache", str(cache), "--repo", str(tmp_path)]) == 0
    (record,) = (cache / drv.SESSIONS_DIRNAME).glob("*.json")
    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["tests"] == {} and data["env_digest"] == drv.env_digest(data["env"])


# --------------------------------------------------- what a change affects
def test_the_base_itself_reuses_every_result(recorded):
    repo, cache, base = recorded
    report = tc.affected(tc.ResultCache(cache, repo), base, paths=["tests"])
    assert report.selection() == []


def test_a_comment_or_moved_line_affects_nothing(recorded):
    source = _FILES["pkg/ops.py"].replace("def g(x):", "# g adds one\n\n\ndef g(x):")
    _, plans = _plan(recorded, {"pkg/ops.py": source, "README.md": "words\n"})
    assert _ran(plans) == _OPAQUE


def test_a_function_body_affects_the_tests_from_the_first_that_ran_it(recorded):
    """test_f ran before g first did, so its result holds; test_g's does not."""
    source = _FILES["pkg/ops.py"].replace("return x + 1", "return x + 2")
    _, plans = _plan(recorded, {"pkg/ops.py": source})
    assert _ran(plans) - _OPAQUE == {
        "tests/test_ops.py::test_g",
        "tests/test_ops.py::test_h1",  # later in the same process: they could read what g left
        "tests/test_ops.py::test_h2",
    }
    assert "tests/test_ops.py::test_f" in _reused(plans)
    assert (
        plans["tests/test_ops.py"].reasons["tests/test_ops.py::test_g"] == "pkg/ops.py: g changed"
    )


def test_a_module_level_constant_affects_every_test_that_loaded_it(recorded):
    """F reads SCALE through an import: no function of consts.py ever runs."""
    _, plans = _plan(recorded, {"pkg/consts.py": "SCALE = 3\n"})
    assert _ran(plans) - _OPAQUE == {"tests/test_ops.py::*"}
    assert (
        "code outside function bodies changed"
        in plans["tests/test_ops.py"].reasons["tests/test_ops.py"]
    )
    assert "tests/test_cells.py::test_cell[1]" in _reused(plans)


def test_a_memoized_result_affects_the_test_that_only_read_it_back(recorded):
    """test_h2 never calls _build: test_h1 left its result in the cache."""
    source = _FILES["pkg/ops.py"].replace("return x * 10", "return x * 11")
    _, plans = _plan(recorded, {"pkg/ops.py": source})
    assert _ran(plans) - _OPAQUE == {"tests/test_ops.py::test_h1", "tests/test_ops.py::test_h2"}
    assert {"tests/test_ops.py::test_f", "tests/test_ops.py::test_g"} <= _reused(plans)


def test_a_data_file_affects_the_test_that_read_it(recorded):
    _, plans = _plan(recorded, {"data/x.json": '{"v": 2}\n'})
    assert _ran(plans) - _OPAQUE == {"tests/test_inputs.py::test_data"}
    assert (
        "data/x.json changed (read as data)"
        in plans["tests/test_inputs.py"].reasons["tests/test_inputs.py::test_data"]
    )


def test_a_path_whose_existence_was_asked_affects_the_test_that_asked(recorded):
    _, plans = _plan(recorded, {"pkg/flag.txt": "on\n"})
    assert _ran(plans) - _OPAQUE == {"tests/test_inputs.py::*"}  # test_probe is the first
    assert plans["tests/test_inputs.py"].reasons["tests/test_inputs.py"] == "pkg/flag.txt appeared"


def test_a_file_a_failed_import_would_now_find_affects_the_test(recorded):
    _, plans = _plan(recorded, {"pkg/fast.py": "SPEED = 1\n"})
    assert _ran(plans) - _OPAQUE == {
        "tests/test_inputs.py::test_import",
        "tests/test_inputs.py::test_data",  # ran after it, in the same process
    }
    assert (
        "now finds a different file"
        in plans["tests/test_inputs.py"].reasons["tests/test_inputs.py::test_import"]
    )


def test_what_collection_ran_changes_the_test_ids_so_the_whole_file_runs(recorded):
    """A third cell would be a test id no record names: only running the file finds it."""
    _, plans = _plan(recorded, {"pkg/cells.py": "def cells():\n    return [1, 2, 3]\n"})
    plan = plans["tests/test_cells.py"]
    assert plan.whole and "cells changed" in plan.reasons["tests/test_cells.py"]
    assert "tests/test_ops.py::test_f" in _reused(plans)


def test_a_native_source_affects_the_tests_that_loaded_the_trees_library(recorded):
    _, plans = _plan(recorded, {"cpp/kernel.cu": "// kernel, faster\n"})
    assert _ran(plans) - _OPAQUE == {"tests/test_native.py::*"}
    assert (
        "native sources changed: cpp/kernel.cu"
        in plans["tests/test_native.py"].reasons["tests/test_native.py"]
    )
    assert "tests/test_ops.py::test_f" in _reused(plans)


def test_a_test_from_its_first_python_process_on_reruns_unless_at_its_own_commit(recorded):
    repo, cache, base = recorded
    _, plans = _plan(recorded, {"README.md": "unrelated\n"})
    assert {"tests/test_proc.py::test_python", "tests/test_proc.py::test_after"} <= _ran(plans)
    assert "tests/test_proc.py::test_git" in _reused(plans)
    assert (
        "opaque: starts" in plans["tests/test_proc.py"].reasons["tests/test_proc.py::test_python"]
    )
    at_base = tc.affected(tc.ResultCache(cache, repo), base, paths=["tests/test_proc.py"])
    assert at_base.selection() == []


def test_another_environment_holds_no_result(recorded):
    _, plans = _plan(recorded, {"README.md": "x\n"}, env_digest="another")
    assert all(p.whole for p in plans.values())


def test_full_runs_every_selected_file(recorded):
    repo, cache, base = recorded
    report = tc.affected(tc.ResultCache(cache, repo), base, paths=["tests/test_ops.py"], full=True)
    assert report.selection() == ["tests/test_ops.py"]


def test_a_file_never_recorded_runs_whole(recorded):
    _, plans = _plan(recorded, {"tests/test_new.py": "def test_new():\n    pass\n"})
    assert plans["tests/test_new.py"].whole
    assert plans["tests/test_new.py"].reasons["tests/test_new.py"] == "never recorded"


# ------------------------------------------------------------ python change
def test_a_body_change_names_its_unit_and_a_signature_change_is_the_skeleton():
    old = b"class C:\n    def m(self, a):\n        return a\n\n\ndef f():\n    return 1\n"
    body = old.replace(b"return a\n", b"return a + 1\n")
    assert tc.python_change(old, body) == tc.PyChange(False, frozenset({"C.m"}))
    signature = old.replace(b"def m(self, a)", b"def m(self, a, b=0)")
    assert tc.python_change(old, signature).skeleton
    constant = old + b"X = 1\n"
    assert tc.python_change(old, constant).skeleton


def test_the_body_of_a_compiled_function_is_the_skeleton():
    """A JIT reads the source or the bytecode; its body never starts as the Python it reads as."""
    old = b"import triton\n\n\n@triton.jit\ndef kernel(x):\n    return x\n"
    assert tc.python_change(old, old.replace(b"return x", b"return x + 1")).skeleton
    old = b"import torch\n\n\nclass M:\n    @torch.compile\n    def forward(self, x):\n        return x\n"
    assert tc.python_change(old, old.replace(b"return x", b"return -x")).skeleton


def test_a_code_objects_unit_is_its_outermost_function():
    assert tc.unit_of("C.m.<locals>.<lambda>") == "C.m"
    assert tc.unit_of("f.<locals>.inner.<locals>.<genexpr>") == "f"
    assert tc.unit_of("C.<lambda>") == "C"
    assert tc.unit_of("<lambda>") == ""


# ---------------------------------------------------------------- compare
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
        _drive(
            repo,
            cache,
            *[s for s in selection if s.startswith(("tests/test_ops.py", "tests/test_new.py"))],
            cwd=tree,
        )
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


def test_compare_is_clean_when_only_what_failed_before_fails(recorded, tmp_path):
    """Nothing re-runs; the head's environment, recorded on its own, is what the reuse is held to."""
    repo, cache, base = recorded
    head = _commit(repo, base, {"README.md": "docs\n"}, message="docs only")
    paths = ["tests/test_ops.py", "tests/test_broken.py"]
    report = tc.compare(tc.ResultCache(cache, repo), base, head, paths=paths)
    assert report.verdict == "INCOMPLETE" and not report.env_checked and report.reused == 5
    assert drv.main(["run", "--cache", str(cache), "--repo", str(tmp_path), "--commit", head]) == 0
    env = tc.head_environment(tc.load_sessions(cache), head)
    report = tc.compare(tc.ResultCache(cache, repo, env_digest=env), base, head, paths=paths)
    assert report.verdict == "CLEAN" and report.ran_at_head == 0
    assert [c.nodeid for c in report.pre_existing] == ["tests/test_broken.py::test_broken"]


def test_compare_is_incomplete_until_the_affected_tests_ran(recorded):
    repo, cache, base = recorded
    head = _commit(repo, base, {"pkg/consts.py": "SCALE = 5\n"}, message="scale 5")
    report = tc.compare(tc.ResultCache(cache, repo), base, head, paths=["tests/test_ops.py"])
    assert report.verdict == "INCOMPLETE" and report.exit_code == tc.EXIT_INCOMPLETE
    assert report.not_run == ["tests/test_ops.py"]


def test_head_results_say_whether_a_files_tests_pass_on_this_tree(recorded):
    repo, cache, base = recorded
    clean = tc.head_results(tc.open_cache(cache, repo, base), base, ["tests/test_ops.py"])
    assert clean.verdict == "CLEAN" and len(clean.passed) == 4
    source = _FILES["pkg/ops.py"].replace("return x + 1", "return x + 2")
    head = _commit(repo, base, {"pkg/ops.py": source}, message="g adds two")
    stale = tc.head_results(tc.open_cache(cache, repo, head), head, ["tests/test_ops.py"])
    assert stale.verdict == "INCOMPLETE" and "tests/test_ops.py::test_g" in stale.not_run
    assert "tests/test_ops.py::test_f" in stale.passed
    nothing = tc.HeadResults(base, [], ["t::a"], [], [])
    assert nothing.verdict == "INCOMPLETE" and "no test passed" in nothing.describe()


def test_a_process_runs_code_by_its_program_not_by_the_words_it_is_given():
    """`platform.architecture()` runs `file -b <the interpreter>`, which runs nothing."""
    assert not drv.starts_code(["file", "-b", "/usr/bin/python3.12"])
    assert not drv.starts_code(["git", "status"])
    assert not drv.starts_code(["sh", "-c", "ls /usr/bin/python3"])
    assert drv.starts_code(["/usr/bin/python3", "-c", "pass"])
    assert drv.starts_code(["/usr/bin/env", "FOO=1", "python", "x.py"])
    assert drv.starts_code(["bash", "-c", "git status && python -m x"])
    assert drv.starts_code(["sh", "-c", "X=1 trtllm-serve model"])
    assert drv.starts_code(["./run.py"])
    assert drv.starts_code_script("make -j8") and not drv.starts_code_script("grep python f")


# -------------------------------------------------------------------- CLI
def test_the_cli_writes_the_selection_and_prints_the_command(recorded, tmp_path, capsys):
    repo, cache, base = recorded
    source = _FILES["pkg/ops.py"].replace("return x + 1", "return x + 3")
    head = _commit(repo, base, {"pkg/ops.py": source}, message="cli g")
    out = tmp_path / "sel"
    code = tc.main(
        [
            "affected",
            "--cache",
            str(cache),
            "--repo",
            str(repo),
            "--head",
            head,
            "--base",
            base,
            "--paths",
            "tests/test_ops.py",
            "--out",
            str(out),
            "--python",
            "/opt/py",
        ]
    )
    text = capsys.readouterr().out
    assert code == 0
    lines = (out / "head.txt").read_text(encoding="utf-8").split()
    assert "tests/test_ops.py::test_g" in lines and "tests/test_ops.py::test_f" not in lines
    assert f"/opt/py {tc.DRIVER} run --cache {cache}" in text
    assert "--timeout=180" in text
    assert "reused: 1 result(s)" in text
    code = tc.main(
        [
            "compare",
            "--cache",
            str(cache),
            "--repo",
            str(repo),
            "--head",
            head,
            "--base",
            base,
            "--paths",
            "tests/test_ops.py",
            "--out",
            str(out),
        ]
    )
    assert code == tc.EXIT_INCOMPLETE
    assert "not run at the head: 3" in capsys.readouterr().out


def test_the_base_defaults_to_what_a_review_last_cleared(recorded, tmp_path):
    repo, _, base = recorded
    head = _commit(repo, base, {"README.md": "a\n"}, message="default base")
    workspace = tmp_path / "ws"
    tasks = workspace / "tasks"
    tasks.mkdir(parents=True)
    git = tc.Git(repo)
    receipts_mod.write(tasks, {"task": "k1", "kind": "kernel", "commits": [f"{head} a"]})
    assert tc.default_base(workspace, git, head) == base  # the parent of the first integration
    receipts_mod.write(
        tasks, {"task": "xmodel", "kind": "review", "cleared": head, "written_at": "z"}
    )
    assert tc.default_base(workspace, git, head) == head


# --------------------------------------------------------------- receipts
def test_a_regression_reviews_receipt_carries_the_commit_it_cleared(tmp_path):
    decision = tmp_path / "decision.yaml"
    decision.write_text(
        "decision: APPROVE\nsummary: nothing regressed\ncleared: 0123456789abcdef0123456789abcdef01234567\n",
        encoding="utf-8",
    )
    receipt = receipts_mod.build("xmodel", "review", decision_file=decision)
    assert receipt["cleared"] == "0123456789abcdef0123456789abcdef01234567"
    assert "- cleared: `0123456789abcdef0123456789abcdef01234567`" in receipts_mod.render([receipt])
    decision.write_text("decision: REJECT\ncleared: not-a-commit\n", encoding="utf-8")
    assert "cleared" not in receipts_mod.build("xmodel", "review", decision_file=decision)


# ------------------------------------------------------------------ prompt
def test_the_regression_reviewer_runs_only_what_the_cache_cannot_answer(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.assemble import AssembleStage

    class Stage(AssembleStage):
        workspace = tmp_path
        task_data = {"trtllm_repo_path": "/repo"}
        _preflight = SimpleNamespace(target_rel="models/fam/target")

        def _cli(self):
            return "trtllm-modeling-v2"

        def _report_instruction(self, role):
            return ""

    dispatch = Stage()._regression_review_prompt(["k1@1"], ("models/fam/target",))
    assert f"trtllm-modeling-v2 tests affected --workspace {tmp_path}" in dispatch
    assert "trtllm-modeling-v2 tests compare" in dispatch
    assert "Boot or test the base tree only for what fails at the head" in dispatch
    assert "`cleared: <head commit>`" in DEFAULT_PROMPTS.reviewer
