# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One way a pytest selection of the tree is run and judged.

A deployment's native tests (a drafter's acceptance, the disaggregated-serving tests)
and the tests a target is delivered with all run the same way: one pytest process under
the tree's interpreter, from the directory pytest takes as the selection's root, writing
a JUnit report outside the tree. The report is what is judged, with the exit code: a
selection that reports nothing, or exits nonzero beside a clean report, did not pass.
"""

from __future__ import annotations

import re
import shlex
import xml.etree.ElementTree as ET  # nosec B405 - the report is written by the pytest this ran
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..infra.git import git_output

#: A pytest test file.
TEST_FILE = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")
#: What marks the directory pytest runs a test from, as pytest itself decides its root.
PYTEST_ROOT_MARKERS: tuple[str, ...] = ("pytest.ini",)
#: How long one selection may run at most.
TIMEOUT_S = 2 * 3600


def pytest_root(repo: str | Path, test_file: str) -> Path:
    """Where pytest runs ``test_file`` from: the nearest directory above it marked as a root.

    The tree's integration tests run from the directory holding their ``pytest.ini``, as
    CI runs them; without one, the repository root.
    """
    root = Path(repo)
    here = (root / test_file).parent
    while here != root and root in here.parents:
        if any((here / marker).is_file() for marker in PYTEST_ROOT_MARKERS):
            return here
        here = here.parent
    return root


def list_test_files(repo: str | Path, commit: str, paths: Sequence[str]) -> list[str]:
    """The test files under ``paths`` at ``commit``; none where git cannot say."""
    listing = git_output(repo, "ls-tree", "-r", "-z", "--name-only", commit, "--", *paths) or ""
    return sorted(f for f in listing.split("\0") if f and TEST_FILE.search(f))


def pytest_argv(
    repo: str | Path, nodes: Sequence[str], report: str | Path, extra: Sequence[str] = ()
) -> tuple[Path, list[str]]:
    """``(root, argv)``: the directory to run from and ``-m pytest ...`` over ``nodes``.

    ``nodes`` are repository-relative files or node ids; they run from the root of the
    first (:func:`pytest_root`), named relative to it.
    """
    root = pytest_root(repo, nodes[0].partition("::")[0])
    named = []
    for node in nodes:
        file, _, rest = node.partition("::")
        relative = (Path(repo) / file).relative_to(root).as_posix()
        named.append(relative + (f"::{rest}" if rest else ""))
    argv = ["-m", "pytest", *named, "-q", "-rs", "-p", "no:cacheprovider", f"--junitxml={report}"]
    return root, [*argv, *extra]


def pytest_command(
    python: str,
    repo: str | Path,
    nodes: Sequence[str],
    report: str | Path,
    extra: Sequence[str] = (),
) -> str:
    """One pytest process over ``nodes`` under ``python``, from the selection's root."""
    root, argv = pytest_argv(repo, nodes, report, extra)
    return f"cd {shlex.quote(str(root))} && {shlex.quote(python)} {shlex.join(argv)}"


@dataclass(frozen=True)
class Outcome:
    """What a JUnit report says of a selection."""

    passed: int = 0
    skipped: tuple[str, ...] = ()
    #: ``<test> <failure|error>: <first line of its message>``.
    failing: tuple[str, ...] = ()
    #: Why there is no report to read.
    unread: str = ""

    def verdict(self, *, skips_fail: bool = False) -> str:
        """``FAILING``, ``INCOMPLETE`` (no report, or nothing passed), or ``CLEAN``."""
        if self.failing or (skips_fail and self.skipped):
            return "FAILING"
        return "CLEAN" if self.passed and not self.unread else "INCOMPLETE"

    def describe(self, *, skips_fail: bool = False, limit: int = 5) -> str:
        """One line: the verdict and what makes it so."""
        counts = f"{self.passed} passed, {len(self.skipped)} skipped, {len(self.failing)} failing"
        verdict = self.verdict(skips_fail=skips_fail)
        if self.unread:
            return f"{verdict} ({self.unread})"
        problems = [*self.failing, *(f"{name} skipped" for name in self.skipped if skips_fail)]
        if problems:
            return f"{verdict} ({counts}): {'; '.join(problems[:limit])}"
        return f"{verdict} ({counts})" + ("" if self.passed else ": no test passed")


def read_report(report: str | Path) -> Outcome:
    """Every test case of the JUnit ``report``, by how it ended."""
    try:
        cases = list(ET.parse(report).getroot().iter("testcase"))  # nosec B314
    except (OSError, ET.ParseError) as exc:
        return Outcome(unread=f"no test report: {exc}")
    passed, skipped, failing = 0, [], []
    for case in cases:
        name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
        ended = next((c for c in case if c.tag in ("failure", "error", "skipped")), None)
        if ended is None:
            passed += 1
        elif ended.tag == "skipped":
            skipped.append(name)
        else:
            message = (ended.get("message") or ended.text or "").strip().splitlines()
            failing.append(f"{name} {ended.tag}: {message[0][:200] if message else ''}")
    return Outcome(passed=passed, skipped=tuple(skipped), failing=tuple(failing))


__all__ = [
    "PYTEST_ROOT_MARKERS",
    "TEST_FILE",
    "TIMEOUT_S",
    "Outcome",
    "list_test_files",
    "pytest_argv",
    "pytest_command",
    "pytest_root",
    "read_report",
]
