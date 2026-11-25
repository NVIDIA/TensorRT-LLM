# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A pytest node of the tree as a gate: run as the tree runs it, judged by its report.

Some things are proven the way the engine's CI proves them, by a test in the
tree: the target's generated e2e test (``checks/e2e.py``), the disaggregated
hop (``checks/disagg.py``) and the spec's own tests (``checks/user_tests.py``).
The node runs under the tree's interpreter from its pytest root and writes a
JUnit report.

Two multi-node launch modes are kept explicit:

- ``launch_world=True`` (with ``nodes == 1``): a test that constructs a whole-model
  ``LLM`` across ``environment.world_size`` ranks. At one rank it runs as a single
  scrubbed process; above one rank it runs under ``trtllm-llmapi-launch``
  (``mpirun`` on one node, a lead-launched ``srun --mpi=pmix`` step across
  ``environment.nodes`` nodes), so rank 0 runs pytest and writes one JUnit report
  while the other ranks serve as MPI workers.
- ``nodes > 1`` (with ``launch_world=False``): per-node gang pytest, where the
  spool runs one pytest process per node of one allocation, publishing
  ``MV2FLOW_JOB_NODE_RANK/HOSTS/NODES``; each node writes its own report, and all
  reports and process exit codes must pass. The flow deploys nothing of its own.

The gate passes when every test the node id selects passed. A test that
skipped (too few devices, a missing model root) proves nothing and fails it,
as does a node id that selects nothing.
"""

from __future__ import annotations

import re
import shlex
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..infra.compute.commands import ended_early
from ..infra.compute.environment import RunEnvironment
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

#: Where the test keeps what it writes for after it: under the gate's work directory, never
#: in the read-only snapshot it runs in, which is pruned once the gate ends.
ARTIFACTS_ENV = "MV2FLOW_GATE_ARTIFACTS"
#: What marks the directory pytest runs a test from, as pytest itself decides its root.
PYTEST_ROOT_MARKERS: tuple[str, ...] = ("pytest.ini",)
#: A line of pytest's short summary: a test that failed or errored, and why.
_SUMMARY = re.compile(r"^(?:FAILED|ERROR) \S+(?: - .+)?$", re.M)
#: How many of those lines a verdict quotes, and how much of each.
SUMMARY_LINES, SUMMARY_WIDTH = 3, 300


def run_token() -> str:
    """A name part no other run of the same check shares.

    A worker's ``trtllm-modeling-v2 gate`` and the run's own check of the same
    test write to the same directory: each run's report and results carry their
    own token, so neither reads what the other wrote.
    """
    return uuid.uuid4().hex[:12]


def failure_summary(output: str) -> str:
    """Pytest's own account of what failed, from its short summary; ``""`` when it printed none."""
    lines = list(dict.fromkeys(m.group(0)[:SUMMARY_WIDTH] for m in _SUMMARY.finditer(output)))
    return "; ".join(lines[:SUMMARY_LINES])


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


def node_command(
    environment: RunEnvironment,
    repo: str | Path,
    node: str,
    report: Path,
    *,
    env: Mapping[str, str] | None = None,
    nodes: int = 1,
    capture: bool = True,
    launch_world: bool = False,
) -> str:
    """``node`` under the tree's interpreter, from its pytest root, with a JUnit report.

    ``capture=False`` (``-s``) lets what the test's processes print reach the
    command's output as they print it: the resolve hook's record of where each
    imported the package under test (``run/workspace/snapshots.py``) among it.
    ``launch_world=True`` launches the engine's full MPI world via
    ``trtllm-llmapi-launch`` whenever ``environment.ntasks > 1`` (mutually
    exclusive with per-node gang ``nodes > 1``).
    """
    if launch_world and nodes > 1:
        raise ValueError(
            "launch_world launches one pytest across a world and cannot be combined with "
            "per-node gang pytest (nodes > 1)"
        )
    file, _, rest = node.partition("::")
    root = pytest_root(repo, file)
    selected = (Path(repo) / file).relative_to(root).as_posix() + (f"::{rest}" if rest else "")
    # What the test keeps goes beside the report, outside the snapshot it runs in (pruned after).
    kept = {ARTIFACTS_ENV: str(report.parent / f"{report.stem}.artifacts")}
    environment = with_mode(
        replace(environment, env={**environment.env, **kept, **(env or {})}), MODE_REQUIRE
    )
    # The short summary names skips (which fail the check) and failures with their reason.
    argv = ["-m", "pytest", selected, "-q", "-rsfE", "-p", "no:cacheprovider"]
    if not capture:
        argv.append("-s")
    if nodes == 1:
        argv.append(f"--junitxml={report}")
        single_process = not (launch_world and environment.ntasks > 1)
        return f"cd {shlex.quote(str(root))} && {environment.python_command(argv, single_process=single_process)}"
    report_arg = (
        shlex.quote(str(report.with_suffix("")))
        + '.node"${MV2FLOW_JOB_NODE_RANK:?missing gang node rank}".xml'
    )
    invocation = shlex.join(argv) + f" --junitxml={report_arg}"
    return (
        f'test "${{MV2FLOW_JOB_NODES:?missing gang node count}}" -eq {nodes} && '
        f"cd {shlex.quote(str(root))} && {environment.python_command(invocation, single_process=True)}"
    )


def judge_report(report: Path, passed_as: str = "passed") -> tuple[bool, str]:
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
        return False, "the node id selected no test"
    return True, f"{passed} test(s) {passed_as}"


def run_node(
    environment: RunEnvironment,
    *,
    name: str,
    node: str,
    task_data: Mapping[str, Any],
    workdir: str | Path,
    timeout: int,
    runner: ShellRunner = default_shell_runner,
    env: Mapping[str, str] | None = None,
    nodes: int = 1,
    unit: str | None = None,
    member: str | None = None,
    passed_as: str = "passed",
    what: str = "the test command",
    capture: bool = True,
    token: str | None = None,
    launch_world: bool = False,
) -> GateResult:
    """Run ``node`` as gate ``name`` and judge its report.

    ``passed_as`` says what its tests passing means, and ``what`` names the
    command in a failure. The report is this run's own (``token``, else
    :func:`run_token`).
    """
    _verdict = verdict_factory(inspect_tree(task_data))
    directory = gate_workdir(workdir)
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    token = token or run_token()
    report = directory / f"{stem}.{unit or 'unit'}.{member or 'member'}.{token}.xml"
    reports = (
        [report.with_name(f"{report.stem}.node{rank}.xml") for rank in range(nodes)]
        if nodes > 1
        else [report]
    )
    for item in reports:
        item.unlink(missing_ok=True)
    command = node_command(
        environment,
        task_data["trtllm_repo_path"],
        node,
        report,
        env=env,
        nodes=nodes,
        capture=capture,
        launch_world=launch_world,
    )
    started = time.monotonic()
    result = runner(command, None, timeout)
    combined = result.stdout + result.stderr
    ran = dict(
        name=name,
        member=member,
        command=command,
        duration_s=time.monotonic() - started,
        stdout_tail="\n".join(combined.splitlines()[-40:]),
    )
    if result.infrastructure:
        return _verdict(**lost_session(result), **ran)
    if result.timed_out:
        return _verdict(passed=False, detail=ended_early(result, timeout), **ran)
    if result.returncode != 0:
        said = failure_summary(combined)
        return _verdict(
            passed=False,
            detail=f"{what} exited {result.returncode}" + (f": {said}" if said else ""),
            **ran,
        )
    outcomes = [judge_report(item, passed_as) for item in reports]
    passed = all(ok for ok, _ in outcomes)
    detail = "; ".join(
        f"node{rank}: {said}" if nodes > 1 else said for rank, (_, said) in enumerate(outcomes)
    )
    return _verdict(passed=passed, detail=f"{node}: {detail}", **ran)


def combine(parts: list[GateResult], of: int, jobs: str = "pytest jobs") -> GateResult:
    """The result of a gate that ran ``parts`` of its ``of`` jobs in turn.

    It passes when every part passed; the rest is the last part's. Every part's
    command, time and output are kept, and the detail names each part.
    """
    last = parts[-1]
    return replace(
        last,
        passed=all(part.passed for part in parts),
        command="\n".join(part.command for part in parts),
        duration_s=sum(part.duration_s for part in parts),
        stdout_tail="\n".join("\n".join(part.stdout_tail for part in parts).splitlines()[-40:]),
        detail=f"{len(parts)}/{of} {jobs}: " + "; ".join(part.detail for part in parts),
    )


__all__ = [
    "ARTIFACTS_ENV",
    "PYTEST_ROOT_MARKERS",
    "combine",
    "failure_summary",
    "run_token",
    "judge_report",
    "node_command",
    "pytest_root",
    "run_node",
]
