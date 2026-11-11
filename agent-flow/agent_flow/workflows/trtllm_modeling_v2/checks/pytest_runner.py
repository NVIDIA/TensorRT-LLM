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
import time
import xml.etree.ElementTree as ET  # nosec B405 - the report is written by the pytest this ran
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.compute.commands import ended_early
from ..infra.compute.environment import RunEnvironment, behind_launch
from ..infra.git import git_output
from ..spec.deployments import Deployment
from . import servers
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

#: Where a test keeps what it writes for after it: under the check's work directory, never
#: in the tree it runs in, which a check leaves as it found it.
ARTIFACTS_ENV = "MV2FLOW_GATE_ARTIFACTS"
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
    site: RunEnvironment | None = None,
) -> str:
    """One pytest process over ``nodes`` under ``python``, from the selection's root.

    Behind ``site``'s launch prefix when it states one: one-GPU work runs where the site
    says, not on the orchestrator's host.
    """
    root, argv = pytest_argv(repo, nodes, report, extra)
    program = f"{shlex.quote(python)} {shlex.join(argv)}"
    if site is not None and site.launch:
        program = behind_launch(site.launch, site.launch_env, program)
    return f"cd {shlex.quote(str(root))} && {program}"


@dataclass(frozen=True)
class Outcome:
    """What a JUnit report says of a selection."""

    passed: int = 0
    skipped: tuple[str, ...] = ()
    #: ``<test> <failure|error>: <first line of its message>``.
    failing: tuple[str, ...] = ()
    #: Why there is no report to read.
    unread: str = ""
    #: The test files asked for that the report has no case of: deselected, or not collected.
    unrun: tuple[str, ...] = ()

    def verdict(self, *, skips_fail: bool = False) -> str:
        """``FAILING``, ``INCOMPLETE`` (no report, a file unrun, or nothing passed), or ``CLEAN``."""
        if self.failing or (skips_fail and self.skipped):
            return "FAILING"
        return "CLEAN" if self.passed and not (self.unread or self.unrun) else "INCOMPLETE"

    def describe(self, *, skips_fail: bool = False, limit: int = 5) -> str:
        """One line: the verdict and what makes it so."""
        counts = f"{self.passed} passed, {len(self.skipped)} skipped, {len(self.failing)} failing"
        verdict = self.verdict(skips_fail=skips_fail)
        if self.unread:
            return f"{verdict} ({self.unread})"
        problems = [*self.failing, *(f"{name} skipped" for name in self.skipped if skips_fail)]
        problems += [f"no test of {file} ran" for file in self.unrun]
        if problems:
            return f"{verdict} ({counts}): {'; '.join(problems[:limit])}"
        return f"{verdict} ({counts})" + ("" if self.passed else ": no test passed")


def read_report(report: str | Path, files: Sequence[str] = ()) -> Outcome:
    """Every test case of the JUnit ``report``, by how it ended.

    ``files`` are the test files the selection asked for: one the report has no case of
    did not run (a configured deselection, a file not collected), and another file's
    passing does not stand in for it.
    """
    try:
        cases = list(ET.parse(report).getroot().iter("testcase"))  # nosec B314
    except (OSError, ET.ParseError) as exc:
        return Outcome(unread=f"no test report: {exc}")
    passed, skipped, failing = 0, [], []
    # A case names its module by its dotted path from pytest's root: the tail of the file's.
    heads = set()
    for case in cases:
        parts = (case.get("classname") or case.get("name") or "").split(".")
        heads.update(".".join(parts[:end]) for end in range(1, len(parts) + 1))
    unrun = [
        file
        for file in files
        if not any(
            f".{Path(file).with_suffix('').as_posix().replace('/', '.')}".endswith(f".{head}")
            for head in heads
        )
    ]
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
    return Outcome(passed, tuple(skipped), tuple(failing), unrun=tuple(unrun))


def run_tests(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    deployment: Deployment,
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
    name: str = "tests",
    missing: str = "the deployment names no test",
) -> GateResult:
    """Run each of ``deployment``'s native tests as one pytest job and judge the reports.

    Every job runs under the deployment's environment with the Modeling V2 switch on,
    behind the deployment's launch prefix when it has one. A test that skipped proves
    nothing and fails the check, as does a node id that selects nothing; the first job
    that does not pass ends it.
    """
    verdict = verdict_factory(inspect_tree(task_data))
    if not deployment.tests:
        return verdict(name=name, deployment=deployment.name, passed=False, detail=missing)
    timeout = timeout or deployment.timeout or TIMEOUT_S
    repo = task_data["trtllm_repo_path"]
    directory = gate_workdir(workdir)
    commands, said, tail, started = [], [], "", time.monotonic()

    def ran(**fields: Any) -> GateResult:
        return verdict(
            name=name,
            deployment=deployment.name,
            command="\n".join(commands),
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(tail.splitlines()[-40:]),
            **fields,
        )

    for index, test in enumerate(deployment.tests):
        report = directory / f"{name}.{deployment.name}.{index}.xml"
        report.unlink(missing_ok=True)
        # What the test keeps goes beside the report, outside the tree it runs in.
        kept = {ARTIFACTS_ENV: str(report.with_suffix(".artifacts"))}
        env = with_mode(
            replace(environment, env={**environment.env, **kept, **deployment.env}), MODE_REQUIRE
        )
        root, argv = pytest_argv(repo, [test], report)
        launch = deployment.server.launch.strip() if deployment.server is not None else ""
        if launch:
            # Behind the deployment's prefix, as its server and its benchmark run: the
            # prefix places the test (rank 0 runs it where the options ask for several),
            # and nothing of this host's devices goes with it.
            words = [*servers.launcher(deployment.server), shlex.quote(env.python)]
            run = behind_launch(launch, env.env, f"{' '.join(words)} {shlex.join(argv)}")
        else:
            run = env.python_command(argv, single_process=True)
        command = f"cd {shlex.quote(str(root))} && {run}"
        commands.append(command)
        result = runner(command, None, timeout)
        tail = result.stdout + result.stderr
        if result.infrastructure:
            return ran(**lost_session(result))
        if result.timed_out:
            return ran(passed=False, detail=f"{test}: {ended_early(result, timeout)}")
        if result.returncode != 0:
            return ran(passed=False, detail=f"{test}: pytest exited {result.returncode}")
        outcome = read_report(report)
        said.append(f"{test}: {outcome.describe(skips_fail=True)}")
        if outcome.verdict(skips_fail=True) != "CLEAN":
            return ran(passed=False, detail="; ".join(said))
    return ran(passed=True, detail="; ".join(said))
