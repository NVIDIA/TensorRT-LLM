# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The disagg gate: a benchmark scored across the hop of disaggregated serving.

``accuracy.disagg`` states one of two forms. **A deployment**: the tree's three files
(``config`` for ``trtllm-serve disaggregated -c``, the ``context`` and ``generation`` servers'
``--config``). The flow serves it, reading only each side's ``num_instances`` and a server's
``tensor_parallel_size`` x ``pipeline_parallel_size`` (:func:`plan_layout`), and holds the
unit's benchmark, run through the proxy (``harness/disagg_driver.py``) under the unit's
``accuracy.evaluator_kwargs``, to its statistical floor (``checks/eval_score.py``).

**A test** (``test``): a pytest the run writes beside the tree's own, run from the tree's
pytest root; with ``nodes > 1`` one process per node of one allocation
(``MV2FLOW_JOB_NODE_RANK/HOSTS/NODES``), and every node's report must pass. A test that
skipped, or a node id that selects nothing, fails it.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..infra.compute import steps
from ..infra.compute.commands import ended_early
from ..infra.compute.environment import LLMAPI_LAUNCH, RunEnvironment
from ..spec.task_schema import (
    ACCURACY_FIELD,
    DISAGG_FILES,
    DISAGG_SIDES,
    gpus_per_node,
    parse_parallel_segment,
    resolve_per_unit,
)
from . import check_config, pytest_node
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .eval_score import eval_arguments, judge_score
from .harness import disagg_driver, workload_io
from .pytest_node import ARTIFACTS_ENV, PYTEST_ROOT_MARKERS, pytest_root
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

GATE_NAME = "disagg"
#: Seconds the test may take: two engines to start and a benchmark to score.
DEFAULT_TIMEOUT_S = 7200
#: What a passing report says the tests did.
PASSED_AS = "passed through the disaggregated hop"
#: The client of a deployment, run under the measured tree's interpreter.
DISAGG_DRIVER = Path(__file__).resolve().parent / "harness" / "disagg_driver.py"
#: A side's role in its servers' names, and as ``trtllm-serve --server_role`` takes it.
ROLES: tuple[str, ...] = ("ctx", "gen")
SERVER_ROLES = dict(zip(ROLES, ("context", "generation")))
#: The engine's role variable: ``trtllm-serve`` sets it in rank 0 alone; every rank reads it.
ENGINE_ROLE_ENV = "TRTLLM_DISAGG_ROLE"


@dataclass(frozen=True)
class Side:
    """One side of a stated deployment: its servers' ``--config``, how many, and their ranks."""

    role: str
    options: str
    instances: int = 1
    ranks: int = 1


@dataclass(frozen=True)
class Slot:
    """Where one server runs: its first node, how many, and its devices on each."""

    node: int
    nodes: int
    devices: tuple[int, ...]


def plan_layout(
    servers: Sequence[tuple[str, int]], per_node: int | None
) -> tuple[int, dict[str, Slot]]:
    """``(nodes, layout)`` of ``(name, ranks)`` servers, one device a rank, in order.

    A server shares a node when it fits what is left, else starts a fresh one; one wider than a
    node takes whole nodes (``ValueError`` otherwise). Without ``per_node``, one node.
    """
    per_node = per_node or sum(ranks for _, ranks in servers)
    node, used = 0, 0
    layout: dict[str, Slot] = {}
    for name, ranks in servers:
        if used and ranks > per_node - used:
            node, used = node + 1, 0
        if ranks <= per_node:
            layout[name] = Slot(node, 1, tuple(range(used, used + ranks)))
            used += ranks
        elif ranks % per_node:
            raise ValueError(
                f"`{name}` is one world of {ranks} ranks, and a node has {per_node} devices: a "
                f"world wider than a node takes whole nodes (a multiple of {per_node} ranks)"
            )
        else:
            whole = ranks // per_node
            layout[name] = Slot(node, whole, tuple(range(per_node)))
            node, used = node + whole - 1, per_node
        if used == per_node:
            node, used = node + 1, 0
    return node + (1 if used else 0), layout


@dataclass(frozen=True)
class DisaggSpec:
    """``accuracy.disagg``: the deployment the flow serves, or the test that proves the hop."""

    #: Repository-relative node id, e.g. ``tests/.../test_disaggregated_serving.py::TestX::test_y``.
    test: str = ""
    #: The stated deployment: its context side, then its generation side. Empty with a test.
    sides: tuple[Side, ...] = ()
    #: The stated deployment's file for ``trtllm-serve disaggregated -c``.
    config: str = ""
    #: Devices the test starts its engines on, per node (:meth:`device_count`).
    devices: int | None = None
    timeout: int = DEFAULT_TIMEOUT_S
    #: Environment the servers or the test run under.
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
        if not isinstance(block, Mapping):
            return None
        config, *files = (block.get(key) for key in DISAGG_FILES)
        fleets = check_config.load_llm_api_options(config)
        sides = []
        for role, key, path in zip(ROLES, DISAGG_SIDES, files if config and all(files) else ()):
            own = check_config.load_llm_api_options(path)
            ranks = int(own.get("tensor_parallel_size") or 1)
            ranks *= int(own.get("pipeline_parallel_size") or 1)
            instances = int((fleets.get(f"{key}_servers") or {}).get("num_instances") or 1)
            sides.append(Side(role, str(path), instances, ranks))
        if not block.get("test") and not sides:
            return None
        return cls(
            test=str(block.get("test") or ""),
            sides=tuple(sides) if not block.get("test") else (),
            config=str(config or ""),
            devices=int(block["devices"]) if block.get("devices") else None,
            timeout=int(block.get("timeout") or DEFAULT_TIMEOUT_S),
            env={str(k): str(v) for k, v in (block.get("env") or {}).items()},
            node_gpus=gpus_per_node(task_data),
            nodes=int(block.get("nodes", 1)),
            tests=tuple(block.get("tests") or ()),
        )

    @property
    def served(self) -> bool:
        """Whether the spec states the deployment, which the flow serves itself."""
        return bool(self.sides)

    def servers(self) -> list[tuple[str, int]]:
        """Each server of the stated deployment, context first: its name and ranks."""
        return [(f"{s.role}{i}", s.ranks) for s in self.sides for i in range(s.instances)]

    def placement(self) -> tuple[int, dict[str, Slot]]:
        """``(nodes, layout)`` of the stated deployment (:func:`plan_layout`)."""
        return plan_layout(self.servers(), self.node_gpus)

    def job_nodes(self) -> int:
        """The nodes the gate's job holds: the deployment's, or the test's."""
        return self.placement()[0] if self.served else self.nodes

    def device_count(self, member: str | None) -> int | None:
        """The devices the gate's job asks one node for (``None``: whole nodes).

        A deployment on one node: a device a rank. A test: declared, else two engines at
        ``member``'s width on one node, one engine a node across nodes, at most a node's.
        """
        if self.served:
            return sum(ranks for _, ranks in self.servers()) if self.job_nodes() == 1 else None
        if self.devices:
            return self.devices
        width = parse_parallel_segment(member)[1] if member else None
        if not width:
            return None
        count = width if self.nodes > 1 else 2 * width
        return min(count, self.node_gpus) if self.node_gpus else count


def job_span(task_data: Mapping[str, Any], gate: str, engine_nodes: int) -> tuple[int, bool]:
    """``(nodes, lead)``: the nodes a check of ``gate`` holds, and whether the lead runs it.

    The lead runs what starts a world across nodes: an engine wider than one
    (``engine_nodes``) or a deployment's servers; a test is a process on each node.
    """
    if gate != GATE_NAME:
        return engine_nodes, engine_nodes > 1
    spec = DisaggSpec.from_task(task_data)
    nodes = spec.job_nodes() if spec else 1
    return nodes, bool(spec and spec.served and nodes > 1)


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


# ------------------------------------------------------------ a stated deployment


def run_served_gate(
    environment: RunEnvironment,
    spec: DisaggSpec,
    *,
    task_data: Mapping[str, Any],
    workdir: str | Path,
    runner: ShellRunner,
    unit: str | None,
    member: str | None,
) -> GateResult:
    """Serve the stated deployment; hold the unit's benchmark through its proxy to its floor.

    The job's script starts each server as one world (a step across nodes, an engine's own
    launch on one), then the client (``steps.servers_script``).
    """
    from .accuracy import resolve_model_key, resolve_reference

    _verdict = verdict_factory(inspect_tree(task_data))
    accuracy = dict(task_data.get(ACCURACY_FIELD) or {})
    benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or "gsm8k")
    protocol = dict(accuracy.get("evaluator_kwargs") or {})
    arguments = eval_arguments(protocol)
    common: dict[str, Any] = dict(name=GATE_NAME, member=member)
    try:  # before any load
        reference = resolve_reference(
            task_data, model_key=resolve_model_key(task_data, unit), unit=unit
        )
        nodes, layout = spec.placement()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _verdict(passed=False, detail=f"nothing was loaded: {exc}", **common)
    common["reference"] = reference
    model = str(task_data["checkpoint_path"])
    directory = gate_workdir(Path(workdir) / f"disagg.{unit or 'unit'}.{member or 'member'}")
    env = with_mode(replace(environment, env={**environment.env, **(spec.env or {})}), MODE_REQUIRE)
    worlds = []
    for side in spec.sides:
        # The side's own file, with the gate's mode over it.
        options = directory / f"options.{side.role}.yaml"
        own = {**check_config.load_llm_api_options(side.options), "modeling_v2": MODE_REQUIRE}
        options.write_text(yaml.safe_dump(own, sort_keys=True), encoding="utf-8")
        role = SERVER_ROLES[side.role]
        for index in range(side.instances):
            name = f"{side.role}{index}"
            words = [
                *([LLMAPI_LAUNCH] if nodes > 1 and side.ranks > 1 else []),  # a step's ranks
                *("trtllm-serve", model, "--config", str(options)),
                *("--host", "0.0.0.0", "--port", "0"),  # nosec B104
                *("--report_addr", str(directory / f"address.{name}"), "--server_role", role),
            ]
            slot = layout[name]
            worlds.append(
                steps.World(name, side.ranks, slot.node, slot.nodes, slot.devices,
                            shlex.join(words), {ENGINE_ROLE_ENV: role})
            )  # fmt: skip
    where = [f"--dataset_path={accuracy['dataset_path']}"] if accuracy.get("dataset_path") else []
    client = shlex.join(
        [
            # Offline, as for `trtllm-eval` (`eval_harness_command`).
            *("env", "HF_DATASETS_OFFLINE=1", "HF_HUB_OFFLINE=1", env.python, str(DISAGG_DRIVER)),
            *("--directory", str(directory), "--config", spec.config, "--model", model),
            *("--servers", ",".join(w.name for w in worlds), "--timeout", str(spec.timeout)),
            *(benchmark, *where, *arguments),
        ]
    )
    launch = env.step if nodes > 1 else env.world
    script = steps.servers_script(launch, worlds, directory=directory, nodes=nodes, client=client)
    command = f"bash {shlex.quote(str(script))}"
    timeout = spec.timeout + disagg_driver.STOP_GRACE_S * (len(worlds) + 1)
    started = time.monotonic()
    result = runner(command, None, timeout)
    combined = result.stdout + result.stderr
    ran = dict(
        command=command,
        duration_s=time.monotonic() - started,
        stdout_tail="\n".join(combined.splitlines()[-40:]),
        **common,
    )
    if result.infrastructure:
        return _verdict(**lost_session(result), **ran)
    if result.timed_out:
        return _verdict(passed=False, detail=ended_early(result, timeout), **ran)
    session = (workload_io.parse_marked(combined, disagg_driver.SESSION_MARKER) or [{}])[-1]
    why = session.get("error") or f"the client exited {result.returncode}"
    if not session.get("started"):
        return _verdict(passed=False, detail=f"the deployment did not start: {why}", **ran)
    served = f"{len(worlds)} server(s) on {nodes} node(s) and their proxy came up"
    if result.returncode != 0:
        return _verdict(passed=False, detail=f"{served}; {benchmark} through it: {why}", **ran)
    said = f"{served}; {benchmark} through the proxy measured"
    judged = judge_score(benchmark, reference, combined, evaluator_kwargs=protocol, said=said)
    return _verdict(**judged, **ran)


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
    """Score a benchmark across the hop: the stated deployment, or the tree's own test."""
    _verdict = verdict_factory(inspect_tree(task_data))
    spec = spec or DisaggSpec.from_task(task_data)
    environment = replace(environment, nodes=1)  # the gate places its own servers
    if spec is None:
        return _verdict(
            name=GATE_NAME,
            member=member,
            passed=False,
            detail="`accuracy.disagg` states nothing to run: state the deployment (`context` "
            "and `generation`), which the flow serves and scores itself, or write the target's "
            "disaggregated accuracy test beside the tree's own (the tests that serve through "
            "`trtllm-serve disaggregated`) and name its node id in `accuracy.disagg.test`",
        )
    if spec.served:
        return run_served_gate(
            environment,
            spec,
            task_data=task_data,
            workdir=workdir,
            runner=runner,
            unit=unit,
            member=member,
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
    "job_span",
    "judge_report",
    "plan_layout",
    "pytest_root",
    "run_disagg_gate",
]
