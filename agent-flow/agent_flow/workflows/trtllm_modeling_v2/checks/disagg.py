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

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ..infra.compute.environment import RunEnvironment
from ..spec.task_schema import ACCURACY_FIELD, gpus_per_node, parse_parallel_segment
from . import pytest_node
from .check_runtime import ShellRunner, default_shell_runner
from .pytest_node import ARTIFACTS_ENV, PYTEST_ROOT_MARKERS, pytest_root
from .verdict import GateResult, inspect_tree, verdict_factory

GATE_NAME = "disagg"
#: Seconds the test may take: two engines to start and a benchmark to score.
DEFAULT_TIMEOUT_S = 7200
#: What a passing report says the tests did.
PASSED_AS = "passed through the disaggregated hop"


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


def disagg_command(
    environment: RunEnvironment, repo: str | Path, spec: DisaggSpec, report: Path
) -> str:
    """``spec.test`` under the tree's interpreter, from its pytest root, with a JUnit report."""
    return pytest_node.node_command(
        environment, repo, spec.test, report, env=spec.env, nodes=spec.nodes
    )


def judge_report(report: Path) -> tuple[bool, str]:
    """Whether every test the JUnit ``report`` holds passed, and what it says."""
    return pytest_node.judge_report(report, PASSED_AS)


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
        return pytest_node.combine(parts, len(spec.tests), "native jobs")
    return pytest_node.run_node(
        environment,
        name=GATE_NAME,
        node=spec.test,
        task_data=task_data,
        workdir=workdir,
        timeout=spec.timeout,
        runner=runner,
        env=spec.env,
        nodes=spec.nodes,
        unit=unit,
        member=member,
        passed_as=PASSED_AS,
        what="native disagg command",
    )


__all__ = [
    "ARTIFACTS_ENV",
    "DEFAULT_TIMEOUT_S",
    "PASSED_AS",
    "PYTEST_ROOT_MARKERS",
    "GATE_NAME",
    "DisaggSpec",
    "disagg_command",
    "judge_report",
    "pytest_root",
    "run_disagg_gate",
]
