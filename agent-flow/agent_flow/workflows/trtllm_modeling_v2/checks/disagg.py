# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The disagg gate: the tree's own disaggregated-serving accuracy test for this target.

Disaggregation is proven the way the engine's CI proves it: a test in the
tree that serves the model through ``trtllm-serve disaggregated`` (context and
generation workers, the KV hand-off between them) and scores a benchmark
across that hop. ``accuracy.disagg.test`` names the native test. The default
runs its one-host launcher from the tree's pytest root. With ``nodes > 1``,
the spool runs one native pytest process per node of one allocation, publishing
``MV2FLOW_JOB_NODE_RANK/HOSTS/NODES``. Each node writes its own report and all
reports and process exit codes must pass. The flow deploys no serving stack.

The gate passes when every test the node id selects passed. A test that
skipped (too few devices, a missing model root) proves nothing and fails it,
as does a node id that selects nothing.
"""

from __future__ import annotations

import shlex
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ..infra.compute.environment import RunEnvironment
from ..spec.task_schema import ACCURACY_FIELD, gpus_per_node, parse_parallel_segment
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

GATE_NAME = "disagg"
#: Where the test keeps what it writes for after it: under the gate's work directory, never
#: in the read-only snapshot it runs in, which is pruned once the gate ends.
ARTIFACTS_ENV = "MV2FLOW_GATE_ARTIFACTS"
#: Seconds the test may take: two engines to start and a benchmark to score.
DEFAULT_TIMEOUT_S = 7200
#: What marks the directory pytest runs a test from, as pytest itself decides its root.
PYTEST_ROOT_MARKERS: tuple[str, ...] = ("pytest.ini",)


@dataclass(frozen=True)
class DisaggSpec:
    """``accuracy.disagg``: the test that proves the hop, and what it runs with."""

    #: Repository-relative node id, e.g. ``tests/.../test_disaggregated_serving.py::TestX::test_y``.
    test: str
    #: Devices the test starts its engines on; default both engines at the member's
    #: width, at most one node's.
    devices: int | None = None
    timeout: int = DEFAULT_TIMEOUT_S
    #: Environment the test process runs under (``LLM_MODELS_ROOT``, ...).
    env: Mapping[str, str] | None = None
    #: The devices one node of the run holds (``task_schema.gpus_per_node``); None unknown.
    node_gpus: int | None = None
    #: Nodes running one native pytest process each; devices is per node.
    nodes: int = 1
    #: Explicit refinements run in separate jobs, each with its own timeout and node reports.
    tests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError("disagg nodes must be positive")

    @classmethod
    def from_task(cls, task_data: Mapping[str, Any]) -> "DisaggSpec | None":
        block = (task_data.get(ACCURACY_FIELD) or {}).get("disagg")
        if not isinstance(block, Mapping) or not block.get("test"):
            return None
        return cls(
            test=str(block["test"]),
            devices=int(block["devices"]) if block.get("devices") else None,
            timeout=int(block.get("timeout") or DEFAULT_TIMEOUT_S),
            env={str(k): str(v) for k, v in (block.get("env") or {}).items()},
            node_gpus=gpus_per_node(task_data),
            nodes=int(block.get("nodes", 1)),
            tests=tuple(block.get("tests") or ()),
        )

    def device_count(self, member: str | None) -> int | None:
        """The devices the test runs on: declared, else two engines at ``member``'s width.

        Devices are per node. Multi-node native tests place one full engine on
        each node; the default single-node test shares that node's devices.
        The native test must attest its actual engine mappings.
        """
        if self.devices:
            return self.devices
        width = parse_parallel_segment(member)[1] if member else None
        if not width:
            return None
        count = width if self.nodes > 1 else 2 * width
        return min(count, self.node_gpus) if self.node_gpus else count


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


def disagg_command(
    environment: RunEnvironment, repo: str | Path, spec: DisaggSpec, report: Path
) -> str:
    """``spec.test`` under the tree's interpreter, from its pytest root, with a JUnit report."""
    file, _, rest = spec.test.partition("::")
    root = pytest_root(repo, file)
    node = (Path(repo) / file).relative_to(root).as_posix() + (f"::{rest}" if rest else "")
    # What the test keeps goes beside the report, outside the snapshot it runs in (pruned after).
    kept = {ARTIFACTS_ENV: str(report.parent / f"{report.stem}.artifacts")}
    env = with_mode(
        replace(environment, env={**environment.env, **kept, **(spec.env or {})}), MODE_REQUIRE
    )
    argv = ["-m", "pytest", node, "-q", "-rs", "-p", "no:cacheprovider"]
    if spec.nodes == 1:
        argv.append(f"--junitxml={report}")
        return f"cd {shlex.quote(str(root))} && {env.python_command(argv, single_process=True)}"
    report_arg = (
        shlex.quote(str(report.with_suffix("")))
        + '.node"${MV2FLOW_JOB_NODE_RANK:?missing gang node rank}".xml'
    )
    invocation = shlex.join(argv) + f" --junitxml={report_arg}"
    return (
        f'test "${{MV2FLOW_JOB_NODES:?missing gang node count}}" -eq {spec.nodes} && '
        f"cd {shlex.quote(str(root))} && {env.python_command(invocation, single_process=True)}"
    )


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
        return False, "the node id selected no test"
    return True, f"{passed} test(s) passed through the disaggregated hop"


def run_disagg_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    unit: str | None = None,
    member: str | None = None,
    spec: DisaggSpec | None = None,
) -> GateResult:
    """Run the tree's disaggregated accuracy test for this target and judge its report."""
    _verdict = verdict_factory(inspect_tree(task_data))
    spec = spec or DisaggSpec.from_task(task_data)
    if spec is None:
        return _verdict(
            name=GATE_NAME,
            member=member,
            passed=False,
            detail="`accuracy.disagg.test` names no test: write the target's disaggregated "
            "accuracy test beside the tree's own (the tests that serve through "
            "`trtllm-serve disaggregated`) and name its node id there",
        )
    if spec.tests:
        parts = []
        for index, test in enumerate(spec.tests):
            part = run_disagg_gate(
                environment,
                task_data=task_data,
                workdir=workdir,
                runner=runner,
                unit=f"{unit or 'unit'}.part{index}",
                member=member,
                spec=replace(spec, test=test, tests=()),
            )
            parts.append(part)
            if not part.passed:
                break
        last = parts[-1]
        return replace(
            last,
            command="\n".join(part.command for part in parts),
            duration_s=sum(part.duration_s for part in parts),
            stdout_tail="\n".join("\n".join(part.stdout_tail for part in parts).splitlines()[-40:]),
            detail=f"{len(parts)}/{len(spec.tests)} native jobs: "
            + "; ".join(part.detail for part in parts),
        )
    directory = gate_workdir(workdir)
    report = directory / f"disagg.{unit or 'unit'}.{member or 'member'}.xml"
    reports = (
        [report.with_name(f"{report.stem}.node{rank}.xml") for rank in range(spec.nodes)]
        if spec.nodes > 1
        else [report]
    )
    for item in reports:
        item.unlink(missing_ok=True)
    command = disagg_command(environment, task_data["trtllm_repo_path"], spec, report)
    started = time.monotonic()
    result = runner(command, None, spec.timeout)
    combined = result.stdout + result.stderr
    ran = dict(
        name=GATE_NAME,
        member=member,
        command=command,
        duration_s=time.monotonic() - started,
        stdout_tail="\n".join(combined.splitlines()[-40:]),
    )
    if result.infrastructure:
        return _verdict(**lost_session(result), **ran)
    if result.timed_out:
        return _verdict(passed=False, detail=f"timed out after {spec.timeout}s", **ran)
    if result.returncode != 0:
        return _verdict(
            passed=False, detail=f"native disagg command exited {result.returncode}", **ran
        )
    outcomes = [judge_report(item) for item in reports]
    passed = all(ok for ok, _ in outcomes)
    detail = "; ".join(
        f"node{rank}: {said}" if spec.nodes > 1 else said for rank, (_, said) in enumerate(outcomes)
    )
    return _verdict(passed=passed, detail=f"{spec.test}: {detail}", **ran)


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "GATE_NAME",
    "DisaggSpec",
    "disagg_command",
    "judge_report",
    "pytest_root",
    "run_disagg_gate",
]
