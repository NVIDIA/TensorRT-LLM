# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The regression check: what a batch of changes to shared code did to the package's tests.

No agent turn runs for it. It runs through the run's test-result cache
(``checks/testcache.py``):

1. the tests under the modeling_v2 package's test tree that no recorded result
   holds for at the commit the checkout is at (the head), against the commit
   the last check cleared, else the run's starting commit (the base);
2. those tests, as one job in a read-only snapshot of the head; with nothing to
   run, the same job records only the head's environment, which the reused
   results are read in;
3. the results at the head against those at the base. A test that fails at the
   head and has no result at the base runs there once.

A test that passed at the base and fails now, or a new one that fails, is a
regression, and so is a test that never produced a result at the head: the
check fails, naming each. The head it found clean is its receipt's
``cleared``, the next check's base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ContextManager, Sequence

from ..checks import testcache

#: The tests a regression check selects from: the package's source-level gates,
#: the catalog's entries and every target's own.
PACKAGE_TESTS: tuple[str, ...] = ("tests/unittest/_torch/modeling_v2",)
#: Where a check's selections go, under the workspace: ``<this>/<task id>/``.
SELECTIONS_DIRNAME = "regression"
#: How many failing tests a failure names before it counts the rest.
MAX_NAMED = 20

#: Runs one command line in a tree on the run's devices; returns its exit code and output tail.
RunJob = Callable[[str, Path], tuple[int, str]]
#: A read-only tree at one commit, held for the block.
TreeAt = Callable[[str], ContextManager[Path]]


@dataclass
class RegressionResult:
    """What a regression check found."""

    head: str
    base: str | None
    #: ``CLEAN``, ``REGRESSED`` or ``INCOMPLETE`` (``testcache.CompareReport.verdict``).
    verdict: str
    #: One line per failing or missing test; empty when clean.
    failures: list[str] = field(default_factory=list)
    ran: int = 0
    reused: int = 0

    @property
    def passed(self) -> bool:
        return self.verdict == "CLEAN"

    def summary(self) -> str:
        base = (self.base or "none")[:10]
        return (
            f"{self.verdict} at {self.head[:10]} against {base}: {self.ran} test(s) ran, "
            f"{self.reused} result(s) reused"
        )


def _named(lines: Sequence[str]) -> list[str]:
    shown = list(lines[:MAX_NAMED])
    if len(lines) > MAX_NAMED:
        shown.append(f"and {len(lines) - MAX_NAMED} more")
    return shown


def _failure(comparison: testcache.Comparison) -> str:
    head = comparison.head
    why = f": {head.message.splitlines()[0][:200]}" if head is not None and head.message else ""
    was = "new, fails" if comparison.new else "passed at the base, fails now"
    return f"{comparison.nodeid} ({was}){why}"


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
    cache_dir = Path(workspace) / testcache.CACHE_DIRNAME
    out = Path(workspace) / SELECTIONS_DIRNAME / name
    base = testcache.default_base(Path(workspace), testcache.Git(repo), head)
    with tree_at(head) as tree:
        cache = testcache.open_cache(cache_dir, repo, head)
        selection = testcache.affected(cache, head, base=base, paths=paths).selection()
        listed = testcache._write_selection(out / "head.txt", selection) if selection else None
        code, tail = run_job(testcache._command(python, cache_dir, str(tree), listed), tree)
    cache = testcache.open_cache(cache_dir, repo, head)
    if not testcache.head_environment(cache.sessions, head):
        # The job recorded no session at the head: nothing it ran can be read.
        return RegressionResult(
            head,
            base,
            "INCOMPLETE",
            [f"the test job recorded nothing (exit {code}): {tail[-500:]}"],
        )
    if base is None:
        # Nothing landed before the head to compare with: each failure is the head's own.
        results = testcache.head_results(cache, head, paths)
        failures = [f"{r.nodeid} (fails; no base to compare with)" for r in results.failing]
        failures += [f"{nodeid} (no result at the head)" for nodeid in results.not_run]
        verdict = "REGRESSED" if results.failing else "INCOMPLETE" if failures else "CLEAN"
        return RegressionResult(head, base, verdict, _named(failures), len(results.passed), 0)
    report = testcache.compare(cache, base, head, paths=paths)
    if report.base_unknown:
        unknown = [c.nodeid for c in report.base_unknown]
        listed = testcache._write_selection(out / "base.txt", unknown)
        with tree_at(base) as tree:
            run_job(testcache._command(python, cache_dir, str(tree), listed), tree)
        cache = testcache.open_cache(cache_dir, repo, head)
        report = testcache.compare(cache, base, head, paths=paths)
    failures = [_failure(c) for c in report.regressions]
    failures += [f"{nodeid} (no result at the head)" for nodeid in report.not_run]
    failures += [
        f"{c.nodeid} (fails at the head; no result at the base)" for c in report.base_unknown
    ]
    return RegressionResult(
        head, base, report.verdict, _named(failures), report.ran_at_head, report.reused
    )


__all__ = [
    "MAX_NAMED",
    "PACKAGE_TESTS",
    "SELECTIONS_DIRNAME",
    "RegressionResult",
    "run_regression_check",
]
