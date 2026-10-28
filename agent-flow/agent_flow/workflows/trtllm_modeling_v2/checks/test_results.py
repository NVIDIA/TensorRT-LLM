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
from typing import Iterable, Sequence

import yaml

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

        A file that failed to collect holds one test, the file itself: its error.
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

    def unrun(self, files: Iterable[str]) -> list[str]:
        """The ``files`` holding a test with no result here: what is still to run."""
        return [
            file
            for file in files
            if (ids := self.test_ids(file)) is None or any(i not in self.tests for i in ids)
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


@dataclass(frozen=True)
class Comparison:
    """A test that fails at the head, and how it ended at the base."""

    head: Result
    base: Result | None
    #: The base does not hold the test at all.
    new: bool = False


@dataclass
class CompareReport:
    """The tests of a declared list at the head, against the base."""

    regressions: list[Comparison] = field(default_factory=list)
    pre_existing: list[Comparison] = field(default_factory=list)
    base_unknown: list[Comparison] = field(default_factory=list)
    #: Test ids with no result at the head; a file no session collected, whole.
    not_run: list[str] = field(default_factory=list)
    passed: int = 0

    @property
    def verdict(self) -> str:
        """``REGRESSED``; ``INCOMPLETE``: a test unrun at the head, or of unknown standing; ``CLEAN``."""
        if self.regressions:
            return "REGRESSED"
        if self.base_unknown or self.not_run:
            return "INCOMPLETE"
        return "CLEAN"


def compare(
    head: Results, base: Results, files: Sequence[str], base_files: Iterable[str]
) -> CompareReport:
    """``head``'s results for the tests of ``files`` against ``base``'s.

    A test that fails at the head is a regression when it passed at the base or
    the base does not hold it (its file is not among ``base_files``, or the
    base's collection of that file has no such test); pre-existing when it
    failed there too; and of unknown standing when the base has no result for it.
    """
    at_base = set(base_files)
    report = CompareReport()
    for file in files:
        ids = head.test_ids(file)
        if ids is None:
            report.not_run.append(file)
            continue
        held = base.test_ids(file) if file in at_base else []
        for nodeid in ids:
            now = head.result(nodeid)
            if now is None:
                report.not_run.append(nodeid)
                continue
            if now.outcome not in FAILING:
                report.passed += 1
                continue
            then = base.result(nodeid)
            new = then is None and held is not None and nodeid not in held
            entry = Comparison(now, then, new)
            if new or (then is not None and then.outcome not in FAILING):
                report.regressions.append(entry)
            elif then is None:
                report.base_unknown.append(entry)
            else:
                report.pre_existing.append(entry)
    return report


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


def default_base(workspace: str | Path, repo: str | Path, head: str) -> str | None:
    """The commit a regression check last cleared, else the run's starting commit.

    The first is the newest ``cleared`` among the receipts in
    ``<workspace>/tasks/`` that is an ancestor of ``head``; the second, the
    parent of the oldest commit any receipt says its task integrated.
    """

    def resolve(rev: str) -> str | None:
        found = git_output(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
        return (found or "").strip() or None

    def ancestor(older: str, newer: str) -> bool:
        return git_output(repo, "merge-base", "--is-ancestor", older, newer) is not None

    receipts = []
    for path in sorted((Path(workspace) / "tasks").glob("*.receipt.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict):
            receipts.append(data)
    cleared = []
    for receipt in receipts:
        commit = resolve(str(receipt["cleared"])) if receipt.get("cleared") else None
        if commit and ancestor(commit, head):
            cleared.append((str(receipt.get("written_at") or ""), commit))
    if cleared:
        return max(cleared)[1]
    integrated = []
    for receipt in receipts:
        for line in receipt.get("commits") or ():
            words = str(line).split()
            commit = resolve(words[0]) if words else None
            if commit and ancestor(commit, head):
                integrated.append(commit)
    oldest = [c for c in integrated if all(ancestor(c, o) for o in integrated)]
    return resolve(f"{oldest[0]}^") if oldest else None


def write_selection(path: Path, lines: Sequence[str]) -> Path:
    """``lines``, one per line, at ``path``: what one driver job runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def driver_command(python: str, cache: str | Path, tree: str | Path, selection: str | Path) -> str:
    """The driver's command line over the tests ``selection`` lists, run in ``tree``."""
    words = [str(DRIVER), "run", "--cache", str(cache), "--repo", str(tree)]
    words += ["--selection", str(selection)]
    return (
        f"{python} {' '.join(shlex.quote(w) for w in words)} "
        "-- -p no:cacheprovider -rA --timeout=180"
    )


__all__ = [
    "CACHE_DIRNAME",
    "DRIVER",
    "CompareReport",
    "Comparison",
    "HeadResults",
    "Result",
    "Results",
    "Session",
    "compare",
    "default_base",
    "driver_command",
    "head_results",
    "list_test_files",
    "load_sessions",
    "recorded",
    "write_selection",
]
