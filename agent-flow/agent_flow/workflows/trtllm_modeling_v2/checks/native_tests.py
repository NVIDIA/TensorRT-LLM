# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tests as a check: tests of the tree, collected by node id, run where they say.

What a session does not replace. A deployment may name native tests that also judge it
(``tests``: a state-transfer test across unequal layouts, a feature's own test), and a
kernel row is certified by its entry's own test. Both are run here the way the engine's
CI runs them: ``pytest`` under the tree's interpreter, from the test's own pytest root,
with the Modeling V2 switch on, as one job with the devices the caller states. No model
is loaded unless a test itself loads one, and the flow deploys nothing for them.

The check passes when every collected test passed. A test that skipped (too few
devices, a missing model root) proves nothing and fails it, as does a node id that
selects nothing. With nothing to collect there is nothing to run, which is no failure
(:func:`nothing_to_run`).

One process of one node is all a test gets: the check starts no MPI world. A test that
asks for ranks of its own (:func:`needs_world`) would wait for them to the check's wall,
so it is not run, and the line says it is not verified, with the node id and the reason.
"""

from __future__ import annotations

import ast
import shlex
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.compute.commands import ended_early
from ..infra.compute.environment import RunEnvironment
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

#: The gate a deployment's named tests are recorded under.
GATE_NAME = "tests"
#: Where a test keeps what it writes for after it: under the check's work directory, never
#: in the read-only snapshot it runs in, which is pruned once the check ends.
ARTIFACTS_ENV = "MV2FLOW_GATE_ARTIFACTS"
#: Seconds the tests may take when the caller states none.
DEFAULT_TIMEOUT_S = 7200
#: What marks the directory pytest runs a test from, as pytest itself decides its root.
PYTEST_ROOT_MARKERS: tuple[str, ...] = ("pytest.ini",)
#: How a test of the engine asks for ranks of its own: an MPI pool of workers, made by
#: the engine's fixture or by the test itself.
WORLD_NAMES: tuple[str, ...] = ("mpi_pool_executor", "MPIPoolExecutor")


def needs_world(repo: str | Path, test: str) -> bool:
    """Whether the node id asks for ranks of its own, as the source of its test says.

    Read in the test's file, without importing it: the function the node id names (its
    decorators, its arguments with their annotations, its body), what the file defines
    under the name of each of its arguments (a fixture of the file, or one it took from
    another), the class it is in, and the file's own ``pytestmark``. An id that names a
    class or a whole file asks where any test of it does. Not seen: a pool made by a
    helper the test calls. A file that cannot be read here says nothing.
    """
    file, _, rest = test.partition("::")
    try:
        module = ast.parse((Path(repo) / file).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return False

    def defined(name: str) -> list[ast.AST]:
        return [
            node
            for node in module.body
            if getattr(node, "name", "") == name
            or (
                isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == name for t in node.targets)
            )
        ]

    scope: ast.AST = module
    said = defined("pytestmark")
    for name in (part.partition("[")[0] for part in rest.split("::") if part):
        named = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        inner = next(
            (n for n in getattr(scope, "body", ()) if isinstance(n, named) and n.name == name),
            None,
        )
        if inner is None:
            return False
        said += inner.decorator_list
        scope = inner
    for argument in getattr(getattr(scope, "args", None), "args", ()):
        said += defined(argument.arg)
    text = "\n".join(ast.unparse(node) for node in (*said, scope))
    return any(name in text for name in WORLD_NAMES)


def pytest_root(repo: str | Path, test_file: str) -> Path:
    """Where pytest runs ``test_file`` from: the nearest directory above it marked as a root.

    The tree's integration tests run from the directory holding their
    ``pytest.ini``, as CI runs them; without one, the repository root.
    """
    root = Path(repo)
    here = (root / test_file).parent
    while here != root and root in here.parents:
        if any((here / marker).is_file() for marker in PYTEST_ROOT_MARKERS):
            return here
        here = here.parent
    return root


def tests_command(
    environment: RunEnvironment,
    repo: str | Path,
    tests: Sequence[str],
    report: Path,
    env: Mapping[str, str] | None = None,
) -> str:
    """``tests`` (repository-relative node ids of one pytest root) under the tree's interpreter.

    Run from their pytest root, with a JUnit report: one process, which starts whatever
    ranks its tests need itself.
    """
    first = tests[0].partition("::")[0]
    root = pytest_root(repo, first)
    nodes = []
    for test in tests:
        file, _, rest = test.partition("::")
        node = (Path(repo) / file).relative_to(root).as_posix()
        nodes.append(node + (f"::{rest}" if rest else ""))
    # What a test keeps goes beside the report, outside the snapshot it runs in (pruned after).
    kept = {ARTIFACTS_ENV: str(report.parent / f"{report.stem}.artifacts")}
    launch = with_mode(
        replace(environment, env={**environment.env, **kept, **(env or {})}), MODE_REQUIRE
    )
    argv = ["-m", "pytest", *nodes, "-q", "-rs", "-p", "no:cacheprovider", f"--junitxml={report}"]
    return f"cd {shlex.quote(str(root))} && {launch.python_command(argv, single_process=True)}"


def judge_report(report: Path) -> tuple[bool, str]:
    """Whether every test the JUnit ``report`` holds passed, and what it says."""
    try:
        cases = ET.parse(report).getroot().iter("testcase")
    except (OSError, ET.ParseError) as exc:
        return False, f"no test report ({exc})"
    passed, problems = 0, []
    for case in cases:
        name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
        outcome = next(
            (child for child in case if child.tag in ("failure", "error", "skipped")), None
        )
        if outcome is None:
            passed += 1
            continue
        message = (outcome.get("message") or outcome.text or "").strip().splitlines()
        problems.append(f"{name} {outcome.tag}: {message[0][:200] if message else ''}")
    if problems:
        return False, "; ".join(problems)
    if not passed:
        return False, "the node id(s) selected no test"
    return True, f"{passed} test(s) passed"


def nothing_to_run(task_data: Mapping[str, Any], name: str, member: str | None, why: str):
    """The line of a check with nothing collected: said, and no failure."""
    return verdict_factory(inspect_tree(task_data))(
        name=name,
        member=member,
        passed=True,
        explained=True,
        detail=f"[NOTHING TO RUN] {why}",
    )


def run_native_tests(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    tests: Sequence[str],
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    name: str = GATE_NAME,
    member: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int | None = None,
) -> GateResult:
    """Run the collected ``tests`` of the tree as one job per pytest root, and judge their reports.

    ``runner`` submits each job with the devices the caller states. ``name`` is what the
    line is recorded under (:data:`GATE_NAME`, ``kernel:<row>``, ``cpu:<row>``).
    """
    if not tests:
        return nothing_to_run(task_data, name, member, "no native test is collected for it")
    _verdict = verdict_factory(inspect_tree(task_data))
    repo = task_data["trtllm_repo_path"]
    worlds = [test for test in tests if needs_world(repo, test)]
    tests = [test for test in tests if test not in worlds]
    # Not run, and said so: it would wait for its ranks until the check's wall.
    unverified = worlds and (
        f"{', '.join(worlds)}: not run. Its source asks for ranks of its own (an MPI pool: "
        f"`{'`, `'.join(WORLD_NAMES)}`), and this check runs a test in one process and starts "
        f"no MPI world: run it by hand under the engine's launcher, or take it out of `tests`"
    )
    timeout = int(timeout or DEFAULT_TIMEOUT_S)
    directory = gate_workdir(workdir)
    by_root: dict[Path, list[str]] = {}
    for test in tests:
        by_root.setdefault(pytest_root(repo, test.partition("::")[0]), []).append(test)
    started = time.monotonic()
    commands: list[str] = []
    said: list[str] = []
    tail = ""
    passed = True
    label = name.replace(":", ".").replace("/", ".")
    for index, group in enumerate(by_root.values()):
        report = directory / f"{label}.{member or 'tests'}.{index}.xml"
        report.unlink(missing_ok=True)
        command = tests_command(environment, repo, group, report, env)
        commands.append(command)
        result = runner(command, None, timeout)
        combined = result.stdout + result.stderr
        tail = "\n".join(combined.splitlines()[-40:])
        ran = dict(
            name=name,
            member=member,
            command="\n".join(commands),
            duration_s=time.monotonic() - started,
            stdout_tail=tail,
        )
        if result.infrastructure:
            return _verdict(**lost_session(result), **ran)
        if result.timed_out:
            return _verdict(passed=False, detail=ended_early(result, timeout), **ran)
        ok, what = judge_report(report)
        if result.returncode != 0 and ok:
            ok, what = False, f"pytest exited {result.returncode}"
        passed &= ok
        said.append(f"{', '.join(group)}: {what}")
        if not ok:
            break
    if unverified and passed:
        said.append(f"not verified: {unverified}")
    return _verdict(
        name=name,
        member=member,
        passed=passed,
        explained=True,
        detail="; ".join(said),
        unverified=str(unverified or "") if passed else "",
        command="\n".join(commands),
        duration_s=time.monotonic() - started,
        stdout_tail=tail,
    )


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "GATE_NAME",
    "judge_report",
    "needs_world",
    "nothing_to_run",
    "pytest_root",
    "run_native_tests",
    "tests_command",
]
