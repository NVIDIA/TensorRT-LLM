# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The final tests: what the run's changes did to the tests, once, on the frozen candidate.

No job runs here before the final pass: a change to shared code that lands in the middle of
a run starts nothing. When the candidate is frozen (``GraphOrchestrator._final_pass``), one
selection of test files runs on it, through the run's test results
(``checks/test_results.py``), which hold at the exact commit they ran at:

1. at the candidate, every selected test file with no result there yet, in a read-only tree
   of it (:func:`record_at`), less the cases already known to fail at the base: none of
   those can regress, and some cost a job's whole wall to fail again;
2. at the base (the commit the run started from), the cases that fail at the candidate and
   have no result there;
3. the results at the candidate against those at the base: each failing case is reported
   with both.

What is selected (:func:`select`) is read off the change and the tasks, with no analysis of
imports: the target's own tests, the tests the tasks' rows and the delivery name, and the
test files the change itself changed. A change to shared code is covered by the tests of
the task that was given it; one that no such task was given has an unknown reach, and then
the package's collected suite (:data:`PACKAGE_TESTS`) runs, once.

A job states the devices its tests state, or the nodes when a row states whole nodes
(:class:`SessionJobs`), never a whole node by default. A case that passed at the base and
fails now, or a new one that fails, fails the final tests, and so does a selected test with
no result at the candidate. A case that fails at the base too is reported as such and fails
nothing.

Two things are **not verified**, and say so instead of passing: a selection of which every
test was skipped and none passed; and a test file a task declared that the candidate does
not hold. The run then closes ``partial`` with them listed. The base runs with the
candidate's libraries, which are the base's own: a run rebuilds no native code.

One kind of selected file is **collected and not run**: a test of the engine's accuracy
harness (``test_results.accuracy_tests``; told by the class it is built on, never by where
it is).
Each of its cases loads the model and scores a whole benchmark, which the deployment's
session has done through the endpoint: running it again is a second load for the same
number. The file is imported and its cases listed (a file that does not import or collect
fails the final tests), and it is reported as left to CI, with the benchmark a session of
this run measured; where no session measured a benchmark it names, that is "not measured
here" and not verified.

Results are kept per commit and per environment (``checks/test_results.py``). The
environment is the driver's own digest and the identity each job is given
(:data:`IDENTITY_VAR`: the native build by content and the runtime): results recorded with
other libraries are not this build's results.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, ContextManager, Mapping, Sequence

from ...checks import test_results
from ...infra.compute.commands import CommandSpec
from ...infra.compute.executor import FEATURE, select_command_executor
from ...infra.git import git_output
from ...operators.coverage import CATALOG_TESTS_DIR

#: The package's collected suite: what runs, once, when the reach of a change to shared code
#: is unknown.
PACKAGE_TESTS: tuple[str, ...] = (CATALOG_TESTS_DIR,)
#: Where a selection goes, under the workspace: ``<this>/<name>/``.
SELECTIONS_DIRNAME = "regression"
#: The name of the final tests: of their jobs, their selection and their row in the ledger.
FINAL_TESTS = "final-tests"
#: The verdict of final tests that failed nothing and do not certify (:attr:`TestsResult.unverified`).
NOT_VERIFIED = "NOT VERIFIED"
#: The variable a test job carries its identity in. The driver's environment digest takes
#: every variable of this prefix, so results are kept apart by it.
IDENTITY_VAR = "TRTLLM_TEST_IDENTITY"
#: Changed test files the final tests leave to CI: they need its models and its lists.
CI_TESTS_PREFIX = "tests/integration/"
#: The spec's key (under ``accuracy``) that has the engine's accuracy tests run in the
#: final tests, and what tells such a test (``checks/test_results.py``).
accuracy_tests = test_results.accuracy_tests
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

#: Paths no unit test imports or reads: a change there reaches no test file.
UNREACHED_PREFIXES: tuple[str, ...] = (
    "tests/integration/",
    "docs/",
    "examples/",
    "benchmarks/",
    "agent-flow/",
    ".github/",
)
UNREACHED_SUFFIXES: tuple[str, ...] = (".md", ".rst")

#: Runs one command line in a tree on the run's devices; returns its exit code and output tail.
#: Called ``(command, tree)``, or with ``timeout_s=`` for a job shorter than the run's job size,
#: which a runner may say as its ``timeout_s`` (:class:`SessionJobs`; :data:`JOB_TIMEOUT_S`
#: otherwise).
RunJob = Callable[..., tuple[int, str]]
#: A read-only tree at one commit, held for the block.
TreeAt = Callable[[str], ContextManager[Path]]


@dataclass
class TestsResult:
    """What the final tests found on one candidate."""

    #: Not a test class, whatever its name starts with.
    __test__ = False

    head: str
    base: str | None
    #: ``CLEAN``, ``REGRESSED`` or ``INCOMPLETE`` (``test_results.CompareReport.verdict``), or
    #: :data:`NOT_VERIFIED`: nothing failed, and :attr:`unverified` says what was not shown.
    verdict: str
    #: One line per case that fails the final tests, each with its result at the candidate
    #: and at the base; empty when clean.
    failures: list[str] = field(default_factory=list)
    #: One line per case that fails at the candidate and at the base too: reported, and it
    #: fails nothing.
    at_base: list[str] = field(default_factory=list)
    #: Tests that did not fail at the candidate.
    held: int = 0
    #: Test jobs run.
    jobs: int = 0
    #: Which test files ran, and why (:func:`select`).
    scope: str = ""
    #: The devices each job stated.
    gpus: int = 0
    #: The whole nodes each job stated, when a row states nodes; 0 for devices on one node.
    nodes: int = 0
    #: What the tests did not show though nothing failed, one line each: the third outcome.
    unverified: list[str] = field(default_factory=list)
    #: The engine's accuracy tests that were collected and not run, one line each, with
    #: the benchmark a session of this run measured: left to CI, and they fail nothing.
    left: list[str] = field(default_factory=list)
    #: The last line the last test job wrote, kept when the verdict is ``INCOMPLETE``: no
    #: case failed there, and what ended the job says where to look.
    job_said: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "CLEAN"

    @property
    def partial(self) -> bool:
        """Nothing failed, and the tests do not certify: the run closes ``partial``."""
        return self.verdict == NOT_VERIFIED

    def summary(self) -> str:
        base = (self.base or "none")[:10]
        size = f"{self.nodes} node(s)" if self.nodes > 1 else f"{self.gpus} device(s)"
        return (
            f"{self.verdict} at {self.head[:10]} against {base}: {self.held} test(s) did not "
            f"fail at the candidate, {len(self.at_base)} fail(s) at the base too, "
            f"{self.jobs} test job(s) run, each stating {size}"
            + (
                f"; {len(self.left)} accuracy test file(s) collected and left to CI"
                if self.left
                else ""
            )
            + (f"; not verified: {'; '.join(self.unverified)}" if self.unverified else "")
            + (f"; {self.scope}" if self.scope else "")
        )


def job_timeout(environment: Any) -> int:
    """How long one test job may run in ``environment``: what one holder can run.

    :data:`JOB_TIMEOUT_S`, or less where a holder has an end: a watcher starts a job only
    where its wall and the fit margin end before the allocation does, so a job as long as a
    whole holder never starts. The window is the holder's time less :data:`WINDOW_SLACK_S`.
    """
    limit = getattr(environment, "holder_time_s", None)
    if not limit:
        return JOB_TIMEOUT_S
    return int(min(JOB_TIMEOUT_S, max(60.0, float(limit) - WINDOW_SLACK_S)))


def max_jobs(timeout_s: float) -> int:
    """How many test jobs of ``timeout_s`` :data:`TEST_BUDGET_S` holds; one at least."""
    return max(1, math.ceil(TEST_BUDGET_S / max(1.0, float(timeout_s))))


def one_rank_env(environment: Any) -> dict[str, str]:
    """What a one-rank engine is given beside the run's own variables, as a session gives it.

    Read from the run's environment at one rank (``RunEnvironment.launch_env``), the one
    place that knows it: a one-rank engine has no launcher to create its worker, and the
    spawn the LLM API would use instead is forbidden in a container. A test that
    constructs an engine in a test job meets the same wall a session's engine would.
    Empty for an environment that says nothing of it.
    """
    try:
        one = dataclasses.replace(environment, world_size=1)
        return {key: value for key, value in one.launch_env.items() if key not in one.env}
    except (AttributeError, TypeError):
        return {}


class SessionJobs:
    """A :data:`RunJob` on the run's devices: each job allowed :func:`job_timeout`, or less.

    Each job states ``gpus`` devices: what the tests it runs state, never a node's count
    by default; or ``nodes`` whole nodes when a row's check states nodes, as that row's own
    check does. Each job is given what a session's one-rank engine is given
    (:func:`one_rank_env`), and ``identity`` (:data:`IDENTITY_VAR`). A long wait on a job
    is said to ``note``.
    """

    def __init__(
        self,
        environment: Any,
        name: str,
        note: Callable[[str], None] | None = None,
        *,
        gpus: int = 1,
        nodes: int = 0,
        identity: str = "",
    ) -> None:
        self.environment = environment
        self.name = name
        self.note = note
        #: The devices one job states.
        self.gpus = max(0, int(gpus))
        #: The whole nodes one job states; 0 or 1 for devices on one node.
        self.nodes = max(0, int(nodes))
        #: What the results of this job are kept under, beside the commit.
        self.identity = identity
        #: How long one job may run (:func:`job_timeout`).
        self.timeout_s = job_timeout(environment)

    def __call__(self, command: str, tree: Path, timeout_s: float | None = None) -> tuple[int, str]:
        executor = select_command_executor(
            self.environment,
            name_prefix=self.name,
            tree=tree,
            gpus=str(self.gpus),
            # Wider than one node: a job of whole nodes, whose script the lead runs once.
            nodes=self.nodes if self.nodes > 1 else 0,
            once=self.nodes > 1,
            priority=FEATURE,
            note=self.note,
        )
        wall = self.timeout_s if timeout_s is None else min(self.timeout_s, timeout_s)
        env = one_rank_env(self.environment)
        if self.identity:
            env[IDENTITY_VAR] = self.identity
        spec = CommandSpec(argv=command, cwd=str(tree), timeout_s=wall, env=env or None)
        done = executor.execute(spec)
        return done.returncode, (done.stdout + done.stderr)[-2000:]


def job_runner(
    workflow: Any, name: str, *, gpus: int = 1, nodes: int = 0, identity: str = ""
) -> tuple[TreeAt, RunJob, str]:
    """``(tree_at, run_job, python)``: how ``workflow`` runs a test job, named ``name``.

    In a read-only tree of a commit when the run keeps trees, else in the
    checkout; on the run's devices, with no agent turn, each job stating ``gpus``
    devices or ``nodes`` whole nodes and carrying ``identity`` (:class:`SessionJobs`).
    """
    repo = Path(str(workflow.task_data.get("trtllm_repo_path") or ""))
    snapshots = getattr(workflow, "_snapshots", None)

    def tree_at(commit: str) -> ContextManager[Path]:
        if snapshots is None:
            return contextlib.nullcontext(repo)
        return snapshots.at(snapshots.pinned(commit))

    python = str((workflow.task_data.get("environment") or {}).get("python") or "python3")
    jobs = SessionJobs(
        workflow._gate_env(),
        name,
        note=lambda said: workflow.journal.event("waiting", said),
        gpus=gpus,
        nodes=nodes,
        identity=identity,
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
    collect: Sequence[str] = (),
) -> tuple[test_results.Results, int, int, str]:
    """Probe the current environment, then run missing ``files`` at ``commit``.

    The probe is one short job (:data:`PROBE_TIMEOUT_S`); the test jobs, as many as
    :data:`TEST_BUDGET_S` holds at the runner's job size (:func:`_run_unrun`).
    ``leave_out`` names, given the probed environment's digest, the node ids not to run.
    ``collect`` are files to import and list, not run.
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
        collect=collect,
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
    collect: Sequence[str] = (),
) -> tuple[test_results.Results, int, int, str]:
    """Run the ``items`` (test files, node ids) with no result at ``commit`` in ``env_digest``.

    A job at a time, each resuming from what the last recorded, while one makes progress and
    :func:`max_jobs` of the runner's job size are not spent. The node ids in ``skip`` are left
    out and not waited for. The files in ``collect`` that no session collected yet are
    imported and listed in the same job, before anything runs, and none of their cases
    runs. Returns what :func:`record_at` does.
    """
    allowed = max_jobs(getattr(run_job, "timeout_s", JOB_TIMEOUT_S))
    results = test_results.recorded(cache_dir, commit, env_digest)
    unrun = results.unrun(items, skip)
    uncollected = [f for f in collect if results.test_ids(f) is None]
    skipped = (
        test_results.write_selection(out / f"{commit[:12]}.skip.txt", sorted(skip))
        if skip
        else None
    )
    stem = out / commit[:12]
    jobs, code, tail = 0, 0, ""
    while (unrun or uncollected) and jobs < allowed:
        with tree_at(commit) as tree:

            def line(listed: Path, **how: Any) -> str:
                return test_results.driver_command(python, cache_dir, tree, listed, **how)

            commands = []
            if uncollected:
                listed = test_results.write_selection(Path(f"{stem}.collect.txt"), uncollected)
                commands.append(line(listed, collect_only=True))
            if unrun:
                listed = test_results.write_selection(Path(f"{stem}.txt"), unrun)
                commands.append(line(listed, skip=skipped))
            # One job: each driver line records what it did, and the last one's status is the job's.
            command = commands[0] if len(commands) == 1 else f"( {' ; '.join(commands)} )"
            code, tail = run_job(command, tree)
        jobs += 1
        results = test_results.recorded(cache_dir, commit, env_digest)
        # A file that ran in a tree of unknown state is not run again for it: another job
        # would learn the same, and the result says why it does not count.
        left = [i for i in results.unrun(items, skip) if not results.said_of(i)]
        still = [f for f in collect if results.test_ids(f) is None]
        if len(left) >= len(unrun) and len(still) >= len(uncollected):
            break
        unrun, uncollected = left, still
    return results, jobs, code, tail


def measured_by_sessions(text: str, measured: Mapping[str, Sequence[str]]) -> str:
    """What the run's sessions measured of the benchmarks an accuracy test's ``text`` names.

    ``measured`` is each benchmark a session measured on the candidate, with the
    deployments. ``""`` when the file names none of them: nothing measured here stands
    for it.
    """
    lowered = text.lower()
    said = [
        f"the session measured `{benchmark}` on {', '.join(f'`{unit}`' for unit in units)}"
        for benchmark, units in sorted(measured.items())
        if units and benchmark.lower() in lowered
    ]
    return "; ".join(said)


def _unreached(path: str) -> bool:
    """Whether no unit test imports or reads ``path`` (see :data:`UNREACHED_PREFIXES`)."""
    return path.startswith(UNREACHED_PREFIXES) or path.endswith(UNREACHED_SUFFIXES)


def _inside(path: str, roots: Sequence[str]) -> bool:
    return any(path == r.rstrip("/") or path.startswith(r.rstrip("/") + "/") for r in roots if r)


def select(
    repo: str | Path,
    base: str | None,
    head: str,
    *,
    own: Sequence[str] = (),
    obligations: Sequence[str] = (),
    covered: Sequence[str] = (),
    package: Sequence[str] = PACKAGE_TESTS,
) -> tuple[list[str], str]:
    """The test files the final tests run at ``head``, and why.

    ``own`` are the run's own paths (its target, and the directory its tests mirror it in):
    every test file there is the target's. ``obligations`` are the test files and
    directories the tasks' rows and the delivery name. To those are added the test files
    the change from ``base`` to ``head`` itself changed. ``covered`` are the shared paths a
    task with tests of its own was given: its tests are its reach. A change to any other
    shared file has an unknown reach, as has a run with no base, and then every test file
    of ``package`` runs too: the collected suite, once. Nothing is inferred from imports.
    """
    chosen = set(test_results.list_test_files(repo, head, [*own, *obligations]))
    if base is None:
        files = sorted(chosen | set(test_results.list_test_files(repo, head, package)))
        return files, (
            f"{len(files)} test file(s): the package's collected suite, once (no base to "
            "read the change from)"
        )
    diff = git_output(repo, "diff", "--no-renames", "--name-only", base, head) or ""
    changed = [p for p in diff.splitlines() if p]
    shared = [p for p in changed if not _inside(p, own) and not _unreached(p)]
    tests = [p for p in shared if test_results.TEST_FILE.search(p)]
    # The repository's integration tests need CI's models and are run from its lists: a
    # changed one is not run here, and the selection says which.
    ci = [p for p in changed if p.startswith(CI_TESTS_PREFIX) and test_results.TEST_FILE.search(p)]
    left = (
        f"; {len(ci)} changed test file(s) under {CI_TESTS_PREFIX} are CI's to run and were "
        f"not run here: {', '.join(ci[:4])}{' ...' if len(ci) > 4 else ''}"
        if ci
        else ""
    )
    chosen |= set(test_results.list_test_files(repo, head, tests)) if tests else set()
    unknown = sorted(p for p in shared if p not in tests and not _inside(p, covered))
    if unknown:
        files = sorted(chosen | set(test_results.list_test_files(repo, head, package)))
        more = f" and {len(unknown) - 1} more" if len(unknown) > 1 else ""
        return files, (
            f"{len(files)} test file(s): the package's collected suite, once (the reach of "
            f"the change to {unknown[0]}{more} is unknown: no task with tests of its own "
            f"was given it){left}"
        )
    files = sorted(chosen)
    return files, (
        f"{len(files)} test file(s): the target's, those the tasks and the delivery name, "
        f"and {len(tests)} the change itself changed{left}"
    )


def missing_declared(repo: str | Path, head: str, declared: Sequence[str]) -> list[str]:
    """The test files a task declared that ``head`` does not hold.

    ``declared`` are write scopes' test files, by name. A declared test that is not in
    the candidate was not run: the selection says so, it does not drop it.
    """
    named = sorted({p for p in declared if test_results.TEST_FILE.search(p)})
    held = set(test_results.list_test_files(repo, head, named)) if named else set()
    return [p for p in named if p not in held]


def _named(lines: Sequence[str]) -> list[str]:
    shown = list(lines[:MAX_NAMED])
    if len(lines) > MAX_NAMED:
        shown.append(f"and {len(lines) - MAX_NAMED} more")
    return shown


def _why(result: test_results.Result) -> str:
    if not result.message:
        return ""
    said = f": {result.message.splitlines()[0][:200]}"
    if test_results.cut_at_default(result.message):
        # Not the test's own limit: the one the driver gives a case that states none.
        said += (
            f" (cut at {test_results.CASE_TIMEOUT_S} s, the limit a case gets when it states "
            "none of its own)"
        )
    return said


def _no_result(results: test_results.Results, item: str) -> str:
    """The line of a selected file or test with no result at the candidate, and why."""
    said = results.said_of(item)
    return f"{item} ({said})" if said else f"{item} (no result at the candidate)"


def _last_line(output: str) -> str:
    """The last line of a job's ``output`` that says anything."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _failure(comparison: test_results.Comparison) -> str:
    head = comparison.head
    was = (
        "fails at the candidate; the base has no such test"
        if comparison.new
        else "fails at the candidate; passed at the base"
    )
    return f"{head.nodeid} ({was}){_why(head)}"


def run_final_tests(
    *,
    workspace: Path,
    repo: Path,
    head: str,
    base: str | None,
    files: Sequence[str],
    tree_at: TreeAt,
    run_job: RunJob,
    python: str,
    scope: str = "",
    name: str = FINAL_TESTS,
    paths: Sequence[str] = PACKAGE_TESTS,
    missing: Sequence[str] = (),
    ci: Mapping[str, str] | None = None,
) -> TestsResult:
    """Run ``files`` on the candidate ``head`` and compare what fails with ``base``.

    ``repo`` holds both commits; ``tree_at`` gives a read-only tree at one, ``run_job``
    runs a command line in it on the run's devices, and ``python`` is the tree's
    interpreter. Every selected file with no result at ``head`` runs there, less the
    cases a result at ``base`` already says fail there: none of them can regress. A case
    that fails at ``head`` and has no result at ``base`` runs at ``base``, so that each
    failing case is reported with both results. ``name`` keeps this selection apart from
    another's.

    ``missing`` are declared test files the candidate does not hold. They, and a
    selection of which every test was skipped, make the result :data:`NOT_VERIFIED` when
    nothing failed.

    ``ci`` are the selected files that are the engine's accuracy tests
    (:func:`accuracy_tests`), each with what the run's sessions measured of it
    (:func:`measured_by_sessions`): they are collected and not run, and listed as left to
    CI (:attr:`TestsResult.left`); one with nothing measured here is not verified; one
    that does not collect fails.
    """
    cache_dir = Path(workspace) / test_results.CACHE_DIRNAME
    out = Path(workspace) / SELECTIONS_DIRNAME / name
    job = dict(cache_dir=cache_dir, out=out, tree_at=tree_at, run_job=run_job, python=python)
    gpus = int(getattr(run_job, "gpus", 0) or 0)
    nodes = int(getattr(run_job, "nodes", 0) or 0)
    ci = dict(ci or {})
    files = [f for f in files if f not in ci]
    selected = set(files)
    unverified = [
        f"{path} is declared by a task and is not in the candidate: it was not run"
        for path in missing
    ]

    def result(verdict: str, failures: Sequence[str], at_base: Sequence[str], held: int, ran: int):
        view = test_results.head_results(results, files)
        if files and view.skipped and not view.passed and not view.failing:
            unverified.append(
                f"every test that ran was skipped ({len(view.skipped)} skipped, none passed)"
            )
        # The accuracy tests left to CI: collected, not run. One that does not collect fails.
        left, broken, lost = [], [], []
        for path in sorted(ci):
            ids = results.test_ids(path)
            found = results.result(path)
            if ids is None:
                lost.append(f"{path} (no result at the candidate: it was not collected)")
            elif found is not None and found.outcome in test_results.FAILING:
                broken.append(f"{path} (does not collect at the candidate){_why(found)}")
            elif ci[path]:
                left.append(f"{path} ({len(ids)} case(s)): left to CI: {ci[path]}")
            else:
                unverified.append(
                    f"{path} ({len(ids)} case(s)): left to CI, not measured here: no session "
                    "of this run measured a benchmark it names"
                )
        if broken:
            verdict = "REGRESSED"
        elif lost and verdict == "CLEAN":
            verdict = "INCOMPLETE"
        if verdict == "CLEAN" and unverified:
            verdict = NOT_VERIFIED
        return TestsResult(
            head, base, verdict, _named([*failures, *broken, *lost]), _named(at_base), held,
            ran, scope, gpus, nodes, list(unverified), left,
            _last_line(tail) if verdict == "INCOMPLETE" else "",
        )  # fmt: skip

    def failing_at_base(env_digest: str) -> set[str]:
        """The selected cases a result at the base says fail there, in ``env_digest``."""
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

    results, ran, code, tail = record_at(
        head, files, leave_out=failing_at_base, collect=sorted(ci), **job
    )
    if (files or ci) and not results.env_digest:
        # No session at the candidate: nothing the jobs ran can be read.
        failure = f"the test job recorded nothing (exit {code}): {tail[-500:]}"
        return TestsResult(
            head, base, "INCOMPLETE", [failure], [], 0, ran, scope, gpus, nodes,
            job_said=_last_line(tail),
        )  # fmt: skip
    if base is None:
        # Nothing to compare with: each failure is the candidate's own.
        view = test_results.head_results(results, files)
        failures = [
            f"{r.nodeid} (fails at the candidate; no base to compare with){_why(r)}"
            for r in view.failing
        ]
        failures += [_no_result(results, nodeid) for nodeid in view.not_run]
        verdict = "REGRESSED" if view.failing else "INCOMPLETE" if failures else "CLEAN"
        return result(verdict, failures, [], len(view.passed) + len(view.skipped), ran)
    base_files = [f for f in test_results.list_test_files(repo, base, paths) if f in selected]
    base_files += [
        f for f in test_results.list_test_files(repo, base, files) if f not in base_files
    ]

    def against_base() -> test_results.CompareReport:
        at_base = test_results.recorded(cache_dir, base, results.env_digest)
        return test_results.compare(results, at_base, files, base_files)

    report = against_base()
    if report.base_unknown:
        # Each case that fails at the candidate is run at the base: both results are said.
        unknown = [c.head.nodeid for c in report.base_unknown]
        ran += _run_unrun(base, unknown, results.env_digest, **job)[1]
        report = against_base()
    failures = [_failure(c) for c in report.regressions]
    failures += [_no_result(results, nodeid) for nodeid in report.not_run]
    failures += [
        f"{c.head.nodeid} (fails at the candidate; no result at the base){_why(c.head)}"
        for c in report.base_unknown
    ]
    both = [
        f"{c.head.nodeid} (fails at the candidate and at the base){_why(c.head)}"
        for c in report.pre_existing
    ]
    both += [
        f"{nodeid} (fails at the base; not run on the candidate: it cannot regress)"
        for nodeid in report.failing_at_base
    ]
    return result(report.verdict, failures, both, report.passed, ran)


__all__ = [
    "FINAL_TESTS",
    "IDENTITY_VAR",
    "JOB_TIMEOUT_S",
    "MAX_NAMED",
    "NOT_VERIFIED",
    "PACKAGE_TESTS",
    "PROBE_TIMEOUT_S",
    "SELECTIONS_DIRNAME",
    "TEST_BUDGET_S",
    "SessionJobs",
    "TestsResult",
    "accuracy_tests",
    "job_runner",
    "job_timeout",
    "max_jobs",
    "measured_by_sessions",
    "missing_declared",
    "one_rank_env",
    "record_at",
    "run_final_tests",
    "select",
]
