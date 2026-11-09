# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the run's change did to the modeling_v2 package's tests, checked once, at delivery.

No agent turn runs for it. Of every test under the package's test tree
(:data:`PACKAGE_TESTS`) it runs, through the run's test results
(``checks/test_results.py``), which hold at the exact commit they ran at:

1. at the commit the checkout is at (the head), the test files the change
   reaches (:func:`affected_files`) and those the caller always wants, with no
   result there yet, in a read-only snapshot of the head (:func:`record_at`),
   less the tests that already fail at the base: none of those can regress;
2. at the run's starting commit (the base), the tests that fail at the head and
   have no result there;
3. the results at the head against those at the base.

What the change reaches is read by directory, as the run's own paths are laid
out (``run/graph.py``, ``own_paths``): a run changes its family and the tests
that mirror it, so a change reaches the family's tests, the tests it changed,
and the claims gates. A change to code other models run, which only an
operator's ``commit_policy.allowed_paths`` lets a run make, reaches every file.

A test that passed at the base and fails now, or a new one that fails, is a
regression, and so is a test with no result at the head: the check fails,
naming each.
"""

from __future__ import annotations

import contextlib
import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, ContextManager, Sequence

from ...checks import test_results
from ...infra.compute.commands import CommandSpec
from ...infra.compute.executor import FEATURE, select_command_executor
from ...infra.compute.scheduler import parse_duration
from ...infra.compute.sessions import GATE_ROLES
from ...infra.git import git_output
from ...operators.coverage import CATALOG_TESTS_DIR, MODELING_V2_PACKAGE

#: The declared list the delivery's package tests come from: the package's source-level gates,
#: the catalog's entries and every target's own.
PACKAGE_TESTS: tuple[str, ...] = (CATALOG_TESTS_DIR,)
#: Where a check's selections go, under the workspace: ``<this>/<task id>/``.
SELECTIONS_DIRNAME = "regression"
#: How many failing tests a failure names before it counts the rest.
MAX_NAMED = 20
#: How long one test job may run at most; less where the gate sessions' usable window is
#: shorter (:func:`job_timeout`).
JOB_TIMEOUT_S = 2 * 3600
#: How long the test jobs at one commit may run in all: as many jobs as that holds at the job
#: size (:func:`max_jobs`). A job out of time keeps the files it finished, and the next runs the
#: rest.
TEST_BUDGET_S = 3 * JOB_TIMEOUT_S
#: The environment probe's wall limit: it prints one line, and a short job starts in any allocation.
PROBE_TIMEOUT_S = 600
#: What a session's usable window keeps back, at least, from its time limit: the allocation's
#: start-up and the margin its scheduler wants past a job's wall limit.
WINDOW_SLACK_S = 600

#: Where each family's directory is; a change under one reaches its tests' directory.
MODELS_DIR = f"{MODELING_V2_PACKAGE}/models/"
#: The registries a new family adds a line to: the claims gates read them.
REGISTRIES = frozenset({f"{MODELS_DIR}__init__.py", f"{MODELING_V2_PACKAGE}/_router_index.py"})
#: Paths no modeling_v2 unit test imports or reads: a change there reaches no test file.
UNREACHED_PREFIXES: tuple[str, ...] = (
    "tests/integration/",
    "docs/",
    "examples/",
    "benchmarks/",
    "agent-flow/",
    ".github/",
)
UNREACHED_SUFFIXES: tuple[str, ...] = (".md", ".rst")
#: Test files every check runs: the package's source-level gates read every entry.
ALWAYS_RUN = re.compile(r"(?:^|/)test_modeling_v2_[a-z_]*claims[a-z_]*\.py$")

#: Runs one command line in a tree on the run's devices; returns its exit code and output tail.
#: Called ``(command, tree)``, or with ``timeout_s=`` for a job shorter than the run's job size,
#: which a runner may say as its ``timeout_s`` (:class:`SessionJobs`; :data:`JOB_TIMEOUT_S`
#: otherwise).
RunJob = Callable[..., tuple[int, str]]
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


def job_timeout(environment: Any) -> int:
    """How long one test job may run in ``environment``: what its gate sessions can run.

    :data:`JOB_TIMEOUT_S`, or less on a session spool: the longest usable window of a session
    gate jobs reach (those on ``environment.spool`` and the ones whose role takes them,
    :data:`sessions.GATE_ROLES`: the pool ``executor.select_command_executor`` sends them to).
    A session's window is its ``scheduler.default_wall`` and its time limit less its hand-over
    lead (at least :data:`WINDOW_SLACK_S`), the shorter of those it declares: its scheduler holds
    a job its allocation cannot finish for the successor, so a job longer than every window never
    starts. A session declaring neither is not counted.
    """
    if getattr(environment, "kind", "") != "session_spool":
        return JOB_TIMEOUT_S
    spool = Path(str(environment.spool or ""))
    windows = []
    for session in environment.sessions:
        if session.role not in GATE_ROLES and Path(str(session.spool)) != spool:
            continue
        bounds = []
        if session.scheduler.get("default_wall"):
            bounds.append(parse_duration(str(session.scheduler["default_wall"])))
        if session.time_limit_s:
            lead = max(session.handover_lead_s, WINDOW_SLACK_S)
            bounds.append(session.time_limit_s - lead)
        positive = [bound for bound in bounds if bound > 0]
        if positive:
            windows.append(min(positive))
    return int(min(JOB_TIMEOUT_S, max(windows))) if windows else JOB_TIMEOUT_S


def max_jobs(timeout_s: float) -> int:
    """How many test jobs of ``timeout_s`` :data:`TEST_BUDGET_S` holds; one at least."""
    return max(1, math.ceil(TEST_BUDGET_S / max(1.0, float(timeout_s))))


class SessionJobs:
    """A :data:`RunJob` on the run's devices: each job allowed :func:`job_timeout`, or less.

    A long wait on a job is said to ``note``.
    """

    def __init__(
        self, environment: Any, name: str, note: Callable[[str], None] | None = None
    ) -> None:
        self.environment = environment
        self.name = name
        self.note = note
        #: How long one job may run (:func:`job_timeout`).
        self.timeout_s = job_timeout(environment)

    def __call__(self, command: str, tree: Path, timeout_s: float | None = None) -> tuple[int, str]:
        executor = select_command_executor(
            self.environment,
            name_prefix=self.name,
            tree=tree,
            gpus="all",
            priority=FEATURE,
            note=self.note,
        )
        wall = self.timeout_s if timeout_s is None else min(self.timeout_s, timeout_s)
        done = executor.execute(CommandSpec(argv=command, cwd=str(tree), timeout_s=wall))
        return done.returncode, (done.stdout + done.stderr)[-2000:]


def job_runner(workflow: Any, name: str) -> tuple[TreeAt, RunJob, str]:
    """``(tree_at, run_job, python)``: how ``workflow`` runs a test job, named ``name``.

    In a read-only snapshot of a commit when the run keeps them, else in the
    checkout; on the run's devices, with no agent turn, each job sized to the
    gate sessions (:class:`SessionJobs`).
    """
    repo = Path(str(workflow.task_data.get("trtllm_repo_path") or ""))
    snapshots = getattr(workflow, "_snapshots", None)

    def tree_at(commit: str) -> ContextManager[Path]:
        return snapshots.lease(commit) if snapshots is not None else contextlib.nullcontext(repo)

    python = str((workflow.task_data.get("environment") or {}).get("python") or "python3")
    jobs = SessionJobs(
        workflow._gate_env(), name, note=lambda said: workflow.journal.event("waiting", said)
    )
    return tree_at, jobs, python


def record_at(
    commit: str,
    files: Sequence[str],
    *,
    cache_dir: Path,
    out: Path,
    tree_at: TreeAt,
    run_job: RunJob,
    python: str,
    leave_out: Callable[[str], Collection[str]] | None = None,
) -> tuple[test_results.Results, int, int, str]:
    """Probe the current environment, then run missing ``files`` at ``commit``.

    The probe is one short job (:data:`PROBE_TIMEOUT_S`); the test jobs, as many as
    :data:`TEST_BUDGET_S` holds at the runner's job size (:func:`_run_unrun`).
    ``leave_out`` names, given the probed environment's digest, the node ids not to run.
    Returns results, test jobs run, and the last job's exit code and output tail.
    A failed probe returns empty results, never historical results.
    """
    with tree_at(commit) as tree:
        command = f"{python} {shlex.quote(str(test_results.DRIVER))} --environment"
        code, tail = run_job(command, tree, timeout_s=PROBE_TIMEOUT_S)
    match = re.search(r"^MV2FLOW_TEST_ENV ([0-9a-f]{16})$", tail, re.MULTILINE)
    if code or match is None:
        return test_results.Results(commit), 0, code, f"environment probe failed: {tail}"
    return _run_unrun(
        commit,
        files,
        match.group(1),
        cache_dir=cache_dir,
        out=out,
        tree_at=tree_at,
        run_job=run_job,
        python=python,
        skip=leave_out(match.group(1)) if leave_out is not None else (),
    )


def _run_unrun(
    commit: str,
    items: Sequence[str],
    env_digest: str,
    *,
    cache_dir: Path,
    out: Path,
    tree_at: TreeAt,
    run_job: RunJob,
    python: str,
    skip: Collection[str] = (),
) -> tuple[test_results.Results, int, int, str]:
    """Run the ``items`` (test files, node ids) with no result at ``commit`` in ``env_digest``.

    A job at a time, each resuming from what the last recorded, while one makes progress and
    :func:`max_jobs` of the runner's job size are not spent. The node ids in ``skip`` are left
    out and not waited for. Returns what :func:`record_at` does.
    """
    allowed = max_jobs(getattr(run_job, "timeout_s", JOB_TIMEOUT_S))
    results = test_results.recorded(cache_dir, commit, env_digest)
    unrun = results.unrun(items, skip)
    skipped = (
        test_results.write_selection(out / f"{commit[:12]}.skip.txt", sorted(skip))
        if skip
        else None
    )
    jobs, code, tail = 0, 0, ""
    while unrun and jobs < allowed:
        listed = test_results.write_selection(out / f"{commit[:12]}.txt", unrun)
        with tree_at(commit) as tree:
            command = test_results.driver_command(python, cache_dir, tree, listed, skipped)
            code, tail = run_job(command, tree)
        jobs += 1
        results = test_results.recorded(cache_dir, commit, env_digest)
        left = results.unrun(items, skip)
        if len(left) >= len(unrun):
            break
        unrun = left
    return results, jobs, code, tail


def _unreached(path: str) -> bool:
    """Whether no modeling_v2 unit test imports or reads ``path`` (see :data:`UNREACHED_PREFIXES`)."""
    return path.startswith(UNREACHED_PREFIXES) or path.endswith(UNREACHED_SUFFIXES)


def affected_files(
    repo: str | Path, base: str | None, head: str, files: Sequence[str]
) -> tuple[list[str], str]:
    """The test files of ``files`` the change from ``base`` to ``head`` reaches, and why.

    A change under a family's directory reaches the test files under that
    family's tests' directory; a test file reaches itself, and another file of
    the test tree (a helper, a ``conftest.py``) every test file beside and below
    it; a registry (:data:`REGISTRIES`) and a path no modeling_v2 unit test reads
    (:func:`_unreached`, other test trees) reach none; :data:`ALWAYS_RUN` always
    runs. Any other change is code other models run and reaches every file, as
    does a check with no base to compare with, or nothing changed since it.
    """
    if base is None:
        return list(files), "every test file (no base to compare with)"
    diff = git_output(repo, "diff", "--no-renames", "--name-only", base, head) or ""
    if not diff.strip():
        return list(files), "every test file (nothing changed since the base)"
    known = set(files)
    chosen = {f for f in files if ALWAYS_RUN.search(f)}
    tests = tuple(f"{t}/" for t in PACKAGE_TESTS)
    for path in (p for p in diff.splitlines() if p):
        if path in known:
            chosen.add(path)
        elif path in REGISTRIES or _unreached(path):
            continue
        elif path.startswith(MODELS_DIR):
            family = path[len(MODELS_DIR) :].split("/", 1)[0]
            mirror = f"{CATALOG_TESTS_DIR}/models/{family}/"
            chosen |= {f for f in files if f.startswith(mirror)}
        elif path.startswith(tests):
            folder = path.rpartition("/")[0] + "/"
            chosen |= {f for f in files if f.startswith(folder)}
        elif path.startswith("tests/"):
            continue
        else:
            return list(files), f"every test file ({path} is code other models run)"
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
    always: Sequence[str] = (),
    base: str = "",
) -> RegressionResult:
    """Run one regression check on ``head``; see the module docstring.

    ``repo`` holds both commits; ``tree_at`` gives a read-only tree at one,
    ``run_job`` runs a command line in it on the run's devices, and
    ``python`` is the tree's interpreter. ``name`` keeps this check's
    selections apart from another's. ``always`` names test files or
    directories that run whatever the change reaches (the delivered tests).
    ``base`` is the commit to compare with; the run's starting commit
    (``test_results.default_base``) when not given.
    """
    cache_dir = Path(workspace) / test_results.CACHE_DIRNAME
    out = Path(workspace) / SELECTIONS_DIRNAME / name
    job = dict(cache_dir=cache_dir, out=out, tree_at=tree_at, run_job=run_job, python=python)
    base = base or test_results.default_base(workspace, repo, head)
    files, scope = affected_files(repo, base, head, test_results.list_test_files(repo, head, paths))
    wanted = test_results.list_test_files(repo, head, always) if always else []
    files = sorted({*files, *wanted})
    selected = set(files)

    def failing_at_base(env_digest: str) -> set[str]:
        """The tests of ``files`` that failed at the base in ``env_digest``: none can regress."""
        if base is None:
            return set()
        at_base = test_results.recorded(cache_dir, base, env_digest)
        return {
            nodeid
            for nodeid, entry in at_base.tests.items()
            if "::" in nodeid
            and nodeid.partition("::")[0] in selected
            and entry.get("outcome") in test_results.FAILING
        }

    results, ran, code, tail = record_at(head, files, leave_out=failing_at_base, **job)
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
    base_files = [
        f for f in test_results.list_test_files(repo, base, [*paths, *always]) if f in selected
    ]

    def against_base() -> test_results.CompareReport:
        at_base = test_results.recorded(cache_dir, base, results.env_digest)
        return test_results.compare(results, at_base, files, base_files)

    report = against_base()
    if report.base_unknown:
        unknown = [c.head.nodeid for c in report.base_unknown]
        ran += _run_unrun(base, unknown, results.env_digest, **job)[1]
        report = against_base()
    failures = [_failure(c) for c in report.regressions]
    failures += [f"{nodeid} (no result at the head)" for nodeid in report.not_run]
    failures += [
        f"{c.head.nodeid} (fails at the head; no result at the base)" for c in report.base_unknown
    ]
    if report.failing_at_base:
        left_out = f"{len(report.failing_at_base)} test(s) failing at the base not run"
        scope = f"{scope}; {left_out}" if scope else left_out
    return RegressionResult(head, base, report.verdict, _named(failures), report.passed, ran, scope)


__all__ = [
    "JOB_TIMEOUT_S",
    "MAX_NAMED",
    "PACKAGE_TESTS",
    "PROBE_TIMEOUT_S",
    "SELECTIONS_DIRNAME",
    "TEST_BUDGET_S",
    "RegressionResult",
    "SessionJobs",
    "affected_files",
    "job_runner",
    "job_timeout",
    "max_jobs",
    "record_at",
    "run_regression_check",
]
