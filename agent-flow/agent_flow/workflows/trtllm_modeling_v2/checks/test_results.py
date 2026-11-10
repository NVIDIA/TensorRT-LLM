# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest results grouped by exact commit and environment.

The driver writes sessions under ``<cache>/sessions/``. Only clean-tree
sessions are eligible for reuse; records never carry across commits.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Collection, Iterable, Sequence

from ..infra.git import git_output
from .harness.test_driver import RECORD_VERSION, SESSIONS_DIRNAME

#: The driver, run by path with the tree's python.
DRIVER = Path(__file__).resolve().parent / "harness" / "test_driver.py"
#: ``<workspace>/<this>`` is a run's cache.
CACHE_DIRNAME = "test_cache"
#: A pytest test file.
TEST_FILE = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")
FAILING = ("failed", "error")


@dataclass(frozen=True)
class Result:
    """How one test ended."""

    nodeid: str
    outcome: str
    message: str


@dataclass
class Session:
    """One recorded pytest process."""

    commit: str | None
    clean: bool | None
    env_digest: str
    tests: dict[str, dict]
    collected: dict[str, list[str]]
    ended: str = ""

    @classmethod
    def load(cls, path: Path) -> "Session | None":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or raw.get("version") != RECORD_VERSION:
            return None
        return cls(
            commit=raw.get("commit") or None,
            clean=raw.get("clean"),
            env_digest=str(raw.get("env_digest") or ""),
            tests={k: v for k, v in (raw.get("tests") or {}).items() if isinstance(v, dict)},
            collected=raw.get("collected") or {},
            ended=str(raw.get("ended") or raw.get("started") or ""),
        )


def load_sessions(cache: str | Path) -> list[Session]:
    """Every readable session record under ``cache``, oldest first."""
    paths = sorted((Path(cache) / SESSIONS_DIRNAME).glob("*.json"))
    return sorted((s for s in map(Session.load, paths) if s is not None), key=lambda s: s.ended)


@dataclass
class Results:
    """What the clean sessions at one commit recorded in one environment; the newest wins."""

    commit: str
    env_digest: str = ""
    tests: dict[str, dict] = field(default_factory=dict)
    collected: dict[str, list[str]] = field(default_factory=dict)

    def test_ids(self, file: str) -> list[str] | None:
        """The tests ``file`` holds here; ``None`` when no session collected it.

        A file that failed to collect, or skipped itself as a module, holds one test: the file
        itself, with that error or skip.
        """
        names = self.collected.get(file)
        if names is not None:
            return [f"{file}::{name}" if name else file for name in names]
        return [file] if file in self.tests else None

    def result(self, nodeid: str) -> Result | None:
        entry = self.tests.get(nodeid)
        if entry is None:
            return None
        return Result(nodeid, str(entry.get("outcome") or "error"), str(entry.get("message") or ""))

    def unrun(self, files: Iterable[str], settled: Collection[str] = ()) -> list[str]:
        """The ``files`` holding a test with no result here: what is still to run.

        A test in ``settled`` is not waited for: it is left out on purpose.
        """
        return [
            file
            for file in files
            if (ids := self.test_ids(file)) is None
            or any(i not in self.tests and i not in settled for i in ids)
        ]


def recorded(cache: str | Path, commit: str, env_digest: str = "") -> Results:
    """The results the clean sessions at ``commit`` recorded.

    In ``env_digest`` when given, else in the environment the newest of them ran in.
    """
    sessions = [s for s in load_sessions(cache) if s.commit == commit and s.clean]
    env = env_digest or next((s.env_digest for s in reversed(sessions) if s.env_digest), "")
    found = Results(commit, env)
    for session in sessions:
        if session.env_digest == env:
            found.tests.update(session.tests)
            found.collected.update(session.collected)
    return found


def list_test_files(repo: str | Path, commit: str, paths: Iterable[str]) -> list[str]:
    """The test files under ``paths`` at ``commit``; none where git cannot say."""
    listing = git_output(repo, "ls-tree", "-r", "-z", "--name-only", commit, "--", *paths) or ""
    return sorted(f for f in listing.split("\0") if f and TEST_FILE.search(f))


@dataclass
class HeadResults:
    """What the recorded results say of some tests at one commit, on their own."""

    head: str
    passed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failing: list[Result] = field(default_factory=list)
    #: Test ids with no result; a file no session collected, whole.
    not_run: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """``FAILING``, ``INCOMPLETE`` (something is still to run, or nothing passed), or ``CLEAN``."""
        if self.failing:
            return "FAILING"
        if self.not_run or not self.passed:
            return "INCOMPLETE"
        return "CLEAN"

    def describe(self, limit: int = 5) -> str:
        """One line: the verdict and what makes it so."""
        counts = (
            f"{len(self.passed)} passed, {len(self.skipped)} skipped, {len(self.failing)} failing, "
            f"{len(self.not_run)} not run at {self.head[:10]}"
        )
        if self.failing:
            named = "; ".join(
                f"{r.nodeid}: {r.outcome}: {r.message[:120]}" for r in self.failing[:limit]
            )
            return f"FAILING ({counts}): {named}"
        if self.not_run:
            return f"INCOMPLETE ({counts}): {', '.join(self.not_run[:limit])}"
        if not self.passed:
            return f"INCOMPLETE ({counts}): no test passed"
        return f"CLEAN ({counts})"


def head_results(results: Results, files: Iterable[str]) -> HeadResults:
    """Every test of ``files`` by its result in ``results``."""
    found = HeadResults(results.commit)
    for file in files:
        ids = results.test_ids(file)
        if ids is None:
            found.not_run.append(file)
            continue
        for nodeid in ids:
            result = results.result(nodeid)
            if result is None:
                found.not_run.append(nodeid)
            elif result.outcome in FAILING:
                found.failing.append(result)
            elif result.outcome == "passed":
                found.passed.append(nodeid)
            else:
                found.skipped.append(nodeid)
    return found


def write_selection(path: Path, lines: Sequence[str]) -> Path:
    """``lines``, one per line, at ``path``: what one driver job runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def driver_command(
    python: str,
    cache: str | Path,
    tree: str | Path,
    selection: str | Path,
    skip: str | Path | None = None,
) -> str:
    """The driver's command line over the tests ``selection`` lists, run in ``tree``.

    ``skip`` lists node ids to leave out of the files the selection names.
    """
    words = [str(DRIVER), "run", "--cache", str(cache), "--repo", str(tree)]
    words += ["--selection", str(selection)]
    if skip is not None:
        words += ["--skip", str(skip)]
    return (
        f"{python} {' '.join(shlex.quote(w) for w in words)} "
        "-- -p no:cacheprovider -rA --timeout=180"
    )


#: How long one test job may run at most.
JOB_TIMEOUT_S = 2 * 3600
#: How long the test jobs at one commit may run in all. A job out of time keeps the files
#: it finished, and the next runs the rest.
TEST_BUDGET_S = 3 * JOB_TIMEOUT_S
#: The environment probe's wall limit: it prints one line.
PROBE_TIMEOUT_S = 600
#: Runs one command line in a tree on the run's devices: ``(command, tree, timeout_s)``
#: to its exit code and output tail.
RunJob = Callable[[str, Path, int], "tuple[int, str]"]


def run_at(
    commit: str,
    files: Sequence[str],
    *,
    cache_dir: Path,
    out: Path,
    tree: Path,
    run_job: RunJob,
    python: str,
) -> tuple[Results, int, int, str]:
    """Probe the environment, then run the ``files`` with no result at ``commit`` in it.

    The probe is one short job. Then a job at a time, each resuming from what the last
    recorded, while one makes progress and :data:`TEST_BUDGET_S` is not spent. Returns the
    results, the test jobs run, and the last job's exit code and output tail. A failed
    probe returns empty results, never historical ones.
    """
    command = f"{python} {shlex.quote(str(DRIVER))} --environment"
    code, tail = run_job(command, tree, PROBE_TIMEOUT_S)
    match = re.search(r"^MV2FLOW_TEST_ENV ([0-9a-f]{16})$", tail, re.MULTILINE)
    if code or match is None:
        return Results(commit), 0, code, f"environment probe failed: {tail}"
    env_digest = match.group(1)
    results = recorded(cache_dir, commit, env_digest)
    unrun = results.unrun(files)
    jobs, allowed = 0, max(1, TEST_BUDGET_S // JOB_TIMEOUT_S)
    while unrun and jobs < allowed:
        listed = write_selection(out / f"{commit[:12]}.txt", unrun)
        code, tail = run_job(driver_command(python, cache_dir, tree, listed), tree, JOB_TIMEOUT_S)
        jobs += 1
        results = recorded(cache_dir, commit, env_digest)
        left = results.unrun(files)
        if len(left) >= len(unrun):
            break
        unrun = left
    return results, jobs, code, tail


__all__ = [
    "CACHE_DIRNAME",
    "DRIVER",
    "JOB_TIMEOUT_S",
    "HeadResults",
    "Result",
    "Results",
    "Session",
    "driver_command",
    "head_results",
    "list_test_files",
    "load_sessions",
    "recorded",
    "run_at",
    "write_selection",
]
