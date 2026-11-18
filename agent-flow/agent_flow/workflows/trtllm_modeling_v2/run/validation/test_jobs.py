# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test jobs on the run's devices: how the delivered tests are run at a commit, with no agent turn.

The jobs run the test driver (``checks/harness/test_driver.py``) in a read-only tree of the
commit and record each result there (``checks/test_results.py``). A job out of time keeps the
files it finished, and the next runs the rest.
"""

from __future__ import annotations

import contextlib
import math
import re
import shlex
from pathlib import Path
from typing import Any, Callable, ContextManager, Sequence

from ...checks import test_results
from ...infra.compute.commands import CommandSpec
from ...infra.compute.executor import FEATURE, select_command_executor
from ...infra.compute.scheduler import parse_duration
from ...infra.compute.sessions import GATE_ROLES

#: Where the selections of test jobs go, under the workspace: ``<this>/<name>/``.
SELECTIONS_DIRNAME = "test_jobs"
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

#: Runs one command line in a tree on the run's devices; returns its exit code and output tail.
#: Called ``(command, tree)``, or with ``timeout_s=`` for a job shorter than the run's job size,
#: which a runner may say as its ``timeout_s`` (:class:`SessionJobs`; :data:`JOB_TIMEOUT_S`
#: otherwise).
RunJob = Callable[..., tuple[int, str]]
#: A read-only tree at one commit, held for the block.
TreeAt = Callable[[str], ContextManager[Path]]


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
) -> tuple[test_results.Results, int, int, str]:
    """Probe the current environment, then run missing ``files`` at ``commit``.

    The probe is one short job (:data:`PROBE_TIMEOUT_S`); the test jobs, as many as
    :data:`TEST_BUDGET_S` holds at the runner's job size (:func:`_run_unrun`).
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
) -> tuple[test_results.Results, int, int, str]:
    """Run the ``items`` (test files, node ids) with no result at ``commit`` in ``env_digest``.

    A job at a time, each resuming from what the last recorded, while one makes progress and
    :func:`max_jobs` of the runner's job size are not spent. Returns what :func:`record_at`
    does.
    """
    allowed = max_jobs(getattr(run_job, "timeout_s", JOB_TIMEOUT_S))
    results = test_results.recorded(cache_dir, commit, env_digest)
    unrun = results.unrun(items)
    jobs, code, tail = 0, 0, ""
    while unrun and jobs < allowed:
        listed = test_results.write_selection(out / f"{commit[:12]}.txt", unrun)
        with tree_at(commit) as tree:
            command = test_results.driver_command(python, cache_dir, tree, listed)
            code, tail = run_job(command, tree)
        jobs += 1
        results = test_results.recorded(cache_dir, commit, env_digest)
        left = results.unrun(items)
        if len(left) >= len(unrun):
            break
        unrun = left
    return results, jobs, code, tail
