# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The regression check: what a batch of changes to shared code did to the package's tests.

No agent turn runs for it. It runs a declared list, every test under the
modeling_v2 package's test tree (:data:`PACKAGE_TESTS`), through the run's test
results (``checks/test_results.py``), which hold at the exact commit they ran at:

1. at the commit the checkout is at (the head), every test file with no result
   there yet, in a read-only snapshot of the head (:func:`record_at`);
2. at the commit the last check cleared, else the run's starting commit (the
   base), the tests that fail at the head and have no result there;
3. the results at the head against those at the base.

It runs only the test files the change can reach (:func:`affected_files`):
those it changed, those that import a module it changed (directly, or through
a helper or another module of the package that does), and the claims gates.
A change to code the whole package shares (outside the catalog's and the
targets' own directories, an ``__init__.py``, a ``conftest.py``) reaches every
file. With no base to compare with, every file runs.

A test that passed at the base and fails now, or a new one that fails, is a
regression, and so is a test with no result at the head: the check fails,
naming each. The head it found clean is its receipt's ``cleared``, the next
check's base, whose results this check has already recorded.
"""

from __future__ import annotations

import ast
import contextlib
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Sequence

from ...checks import test_results
from ...infra.compute.commands import CommandSpec
from ...infra.compute.executor import FEATURE, select_command_executor
from ...infra.compute.scheduler import parse_duration
from ...infra.git import git_output, read_files_at
from ...operators.coverage import CATALOG_DIR, CATALOG_TESTS_DIR, MODELING_V2_PACKAGE

#: The declared list a regression check runs: the package's source-level gates,
#: the catalog's entries and every target's own.
PACKAGE_TESTS: tuple[str, ...] = (CATALOG_TESTS_DIR,)
#: Where a check's selections go, under the workspace: ``<this>/<task id>/``.
SELECTIONS_DIRNAME = "regression"
#: How many failing tests a failure names before it counts the rest.
MAX_NAMED = 20
#: How long one test job may run.
JOB_TIMEOUT_S = 2 * 3600
#: Jobs :func:`record_at` runs at most: a job out of time keeps the files it
#: finished, and the next runs the rest.
MAX_JOBS = 3

#: Where a change reaches only the files that import it: the catalog's entries, the targets.
SCOPED_SOURCE_DIRS: tuple[str, ...] = (f"{CATALOG_DIR}/", f"{MODELING_V2_PACKAGE}/models/")
#: Test files every check runs: the package's source-level gates read every entry.
ALWAYS_RUN = re.compile(r"(?:^|/)test_modeling_v2_[a-z_]*claims[a-z_]*\.py$")

#: Runs one command line in a tree on the run's devices; returns its exit code and output tail.
RunJob = Callable[[str, Path], tuple[int, str]]
#: A read-only tree at one commit, held for the block.
TreeAt = Callable[[str], ContextManager[Path]]


@dataclass
class RegressionResult:
    """What a regression check found."""

    head: str
    base: str | None
    #: ``CLEAN``, ``REGRESSED`` or ``INCOMPLETE`` (``test_results.CompareReport.verdict``).
    verdict: str
    #: One line per failing or missing test; empty when clean.
    failures: list[str] = field(default_factory=list)
    #: Tests that did not fail at the head.
    held: int = 0
    #: Test jobs this check ran.
    jobs: int = 0
    #: Which test files it ran, and why (:func:`affected_files`).
    scope: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "CLEAN"

    def summary(self) -> str:
        base = (self.base or "none")[:10]
        return (
            f"{self.verdict} at {self.head[:10]} against {base}: {self.held} test(s) did not "
            f"fail at the head, {self.jobs} test job(s) run"
            + (f"; {self.scope}" if self.scope else "")
        )


def _job_timeout(environment: Any) -> int:
    """Respect the declared gate-session job budget without changing test scope."""
    if getattr(environment, "kind", "") != "session_spool":
        return JOB_TIMEOUT_S
    budgets = [
        parse_duration(str(session.scheduler["default_wall"]))
        for session in environment.sessions
        if session.role in ("gates", "any") and session.scheduler.get("default_wall")
    ]
    positive = [budget for budget in budgets if budget > 0]
    return int(min(JOB_TIMEOUT_S, max(positive))) if positive else JOB_TIMEOUT_S


def job_runner(workflow: Any, name: str) -> tuple[TreeAt, RunJob, str]:
    """``(tree_at, run_job, python)``: how ``workflow`` runs a test job, named ``name``.

    In a read-only snapshot of a commit when the run keeps them, else in the
    checkout; on the run's devices, with no agent turn.
    """
    repo = Path(str(workflow.task_data.get("trtllm_repo_path") or ""))
    snapshots = getattr(workflow, "_snapshots", None)
    env = workflow._gate_env()

    def tree_at(commit: str) -> ContextManager[Path]:
        return snapshots.lease(commit) if snapshots is not None else contextlib.nullcontext(repo)

    def run_job(command: str, tree: Path) -> tuple[int, str]:
        executor = select_command_executor(
            env, name_prefix=name, tree=tree, gpus="all", priority=FEATURE
        )
        done = executor.execute(
            CommandSpec(argv=command, cwd=str(tree), timeout_s=_job_timeout(env))
        )
        return done.returncode, (done.stdout + done.stderr)[-2000:]

    python = str((workflow.task_data.get("environment") or {}).get("python") or "python3")
    return tree_at, run_job, python


def record_at(
    commit: str,
    files: Sequence[str],
    *,
    cache_dir: Path,
    out: Path,
    tree_at: TreeAt,
    run_job: RunJob,
    python: str,
) -> tuple[test_results.Results, int, int, str]:
    """Probe the current environment, then run missing ``files`` at ``commit``.

    The probe uses one executor job; :data:`MAX_JOBS` test jobs run at most.
    Returns results, test jobs run, and the last job's exit code and output tail.
    A failed probe returns empty results, never historical results.
    """
    with tree_at(commit) as tree:
        command = f"{python} {shlex.quote(str(test_results.DRIVER))} --environment"
        code, tail = run_job(command, tree)
    match = re.search(r"^MV2FLOW_TEST_ENV ([0-9a-f]{16})$", tail, re.MULTILINE)
    if code or match is None:
        return test_results.Results(commit), 0, code, f"environment probe failed: {tail}"
    env_digest = match.group(1)
    results = test_results.recorded(cache_dir, commit, env_digest)
    unrun = results.unrun(files)
    jobs, code, tail = 0, 0, ""
    while unrun and jobs < MAX_JOBS:
        listed = test_results.write_selection(out / f"{commit[:12]}.txt", unrun)
        with tree_at(commit) as tree:
            code, tail = run_job(test_results.driver_command(python, cache_dir, tree, listed), tree)
        jobs += 1
        results = test_results.recorded(cache_dir, commit, env_digest)
        left = results.unrun(files)
        if len(left) >= len(unrun):
            break
        unrun = left
    return results, jobs, code, tail


def _module_names(path: str) -> list[str]:
    """How a test may name the module at ``path``: dotted from the root, and its last two parts."""
    stem = path[: -len(".py")] if path.endswith(".py") else path
    dotted = stem.replace("/", ".")
    parts = dotted.split(".")
    names = [dotted, stem]
    if len(parts) >= 2:
        names += [".".join(parts[-2:]), "/".join(parts[-2:])]
    return names


def _module_of(path: str) -> str:
    """The dotted module the ``.py`` file at ``path`` is, from the root (a package by its init)."""
    stem = path[: -len(".py")]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def _imported(path: str, text: str) -> set[str]:
    """The dotted names the file at ``path`` imports: each module, and each name from one.

    ``from a import b`` gives ``a`` and ``a.b`` (``b`` may be a module); a relative
    import resolves against the file's own package. A file that does not parse gives none.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    module = _module_of(path)
    package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = package.split(".") if package else []
            if node.level > len(parts) + 1:
                continue
            up = parts[: len(parts) - node.level + 1] if node.level else []
            source = ".".join([*up, *([node.module] if node.module else [])])
            if source:
                names.add(source)
                names.update(f"{source}.{a.name}" for a in node.names if a.name != "*")
    return names


def _importers(
    repo: str | Path, head: str, roots: Sequence[str], changed: Sequence[str]
) -> set[str] | None:
    """The ``.py`` files under ``roots`` at ``head`` that import one of ``changed``, directly or not.

    A file names a module by its full dotted path or by any trailing part of it (a test
    imports a helper beside it, on the path it puts first: ``from _validating import``);
    an over-match only runs a test more. ``None`` when the files cannot be read.
    """
    listing = git_output(repo, "ls-tree", "-r", "-z", "--name-only", head, "--", *roots)
    if listing is None:
        return None
    present = [p for p in listing.split("\0") if p.endswith(".py")]
    texts = read_files_at(repo, head, present)
    if texts is None:
        return None
    by_name: dict[str, set[str]] = {}
    for path in {*present, *(p for p in changed if p.endswith(".py"))}:
        parts = _module_of(path).split(".")
        for start in range(len(parts)):
            by_name.setdefault(".".join(parts[start:]), set()).add(path)
    importers: dict[str, set[str]] = {}
    for path, text in texts.items():
        for name in _imported(path, text):
            for target in by_name.get(name, ()):
                if target != path:
                    importers.setdefault(target, set()).add(path)
    reached, frontier = set(changed), list(changed)
    while frontier:
        for importer in importers.get(frontier.pop(), ()):
            if importer not in reached:
                reached.add(importer)
                frontier.append(importer)
    return reached


def affected_files(
    repo: str | Path,
    base: str | None,
    head: str,
    files: Sequence[str],
    roots: Sequence[str] = (MODELING_V2_PACKAGE, *PACKAGE_TESTS),
) -> tuple[list[str], str]:
    """The test files of ``files`` the change from ``base`` to ``head`` can reach, and why.

    Every file when there is no base, nothing changed since it, or the change
    touches code the whole package shares (:data:`SCOPED_SOURCE_DIRS`); else the test files it
    changed, those that import a module it changed, directly or through any other file under
    ``roots`` (:func:`_importers`), those beside a test helper that does (a test may start its
    helper by file path), those whose text names one (dotted, as a path, or by its last two
    parts: an over-match only runs a test more), and :data:`ALWAYS_RUN`.
    """
    if base is None:
        return list(files), "every test file (no base to compare with)"
    diff = git_output(repo, "diff", "--no-renames", "--name-only", base, head) or ""
    changed = [p for p in diff.splitlines() if p]
    if not changed:
        return list(files), "every test file (nothing changed since the base)"
    known = set(files)
    for path in changed:
        if path in known:
            continue
        name = path.rsplit("/", 1)[-1]
        scoped = path.endswith(".py") and any(path.startswith(d) for d in SCOPED_SOURCE_DIRS)
        if not scoped or name in ("__init__.py", "conftest.py"):
            return list(files), f"every test file ({path} is shared by the package)"
    reached = _importers(repo, head, roots, changed)
    if reached is None:
        return list(files), "every test file (the imports at the head could not be read)"
    # A helper a test starts by file path (a collective's rank matrix) reaches the tests beside it.
    helpers = {p.rpartition("/")[0] for p in reached if p not in known}
    chosen = {
        f for f in files if f in reached or f.rpartition("/")[0] in helpers or ALWAYS_RUN.search(f)
    }
    names = sorted({n for p in changed if p.endswith(".py") for n in _module_names(p)})
    if names and files:
        args = ["grep", "-l", "-F"]
        for name in names:
            args += ["-e", name]
        found = git_output(repo, *args, head, "--", *files) or ""
        prefix = f"{head}:"
        chosen |= {
            line[len(prefix) :] if line.startswith(prefix) else line
            for line in found.splitlines()
            if line
        } & known
    picked = sorted(chosen)
    return picked, f"{len(picked)} of {len(files)} test files reach what changed"


def _named(lines: Sequence[str]) -> list[str]:
    shown = list(lines[:MAX_NAMED])
    if len(lines) > MAX_NAMED:
        shown.append(f"and {len(lines) - MAX_NAMED} more")
    return shown


def _failure(comparison: test_results.Comparison) -> str:
    head = comparison.head
    why = f": {head.message.splitlines()[0][:200]}" if head.message else ""
    was = "new, fails" if comparison.new else "passed at the base, fails now"
    return f"{head.nodeid} ({was}){why}"


def run_regression_check(
    *,
    workspace: Path,
    repo: Path,
    head: str,
    tree_at: TreeAt,
    run_job: RunJob,
    python: str,
    name: str,
    paths: Sequence[str] = PACKAGE_TESTS,
) -> RegressionResult:
    """Run one regression check on ``head``; see the module docstring.

    ``repo`` holds both commits; ``tree_at`` gives a read-only tree at one,
    ``run_job`` runs a command line in it on the run's devices, and
    ``python`` is the tree's interpreter. ``name`` keeps this check's
    selections apart from another's.
    """
    cache_dir = Path(workspace) / test_results.CACHE_DIRNAME
    out = Path(workspace) / SELECTIONS_DIRNAME / name
    job = dict(cache_dir=cache_dir, out=out, tree_at=tree_at, run_job=run_job, python=python)
    base = test_results.default_base(workspace, repo, head)
    files, scope = affected_files(
        repo,
        base,
        head,
        test_results.list_test_files(repo, head, paths),
        roots=(MODELING_V2_PACKAGE, *paths),
    )
    selected = set(files)
    results, ran, code, tail = record_at(head, files, **job)
    if files and not results.env_digest:
        # No session at the head: nothing the jobs ran can be read.
        failure = f"the test job recorded nothing (exit {code}): {tail[-500:]}"
        return RegressionResult(head, base, "INCOMPLETE", [failure], 0, ran, scope)
    if base is None:
        # Nothing landed before the head to compare with: each failure is the head's own.
        view = test_results.head_results(results, files)
        failures = [f"{r.nodeid} (fails; no base to compare with)" for r in view.failing]
        failures += [f"{nodeid} (no result at the head)" for nodeid in view.not_run]
        verdict = "REGRESSED" if view.failing else "INCOMPLETE" if failures else "CLEAN"
        held = len(view.passed) + len(view.skipped)
        return RegressionResult(head, base, verdict, _named(failures), held, ran, scope)
    base_files = [f for f in test_results.list_test_files(repo, base, paths) if f in selected]

    def against_base() -> test_results.CompareReport:
        at_base = test_results.recorded(cache_dir, base, results.env_digest)
        return test_results.compare(results, at_base, files, base_files)

    report = against_base()
    if report.base_unknown:
        listed = test_results.write_selection(
            out / f"{base[:12]}.txt", [c.head.nodeid for c in report.base_unknown]
        )
        with tree_at(base) as tree:
            run_job(test_results.driver_command(python, cache_dir, tree, listed), tree)
        ran += 1
        report = against_base()
    failures = [_failure(c) for c in report.regressions]
    failures += [f"{nodeid} (no result at the head)" for nodeid in report.not_run]
    failures += [
        f"{c.head.nodeid} (fails at the head; no result at the base)" for c in report.base_unknown
    ]
    return RegressionResult(head, base, report.verdict, _named(failures), report.passed, ran, scope)


__all__ = [
    "JOB_TIMEOUT_S",
    "MAX_JOBS",
    "MAX_NAMED",
    "PACKAGE_TESTS",
    "SELECTIONS_DIRNAME",
    "RegressionResult",
    "affected_files",
    "job_runner",
    "record_at",
    "run_regression_check",
]
