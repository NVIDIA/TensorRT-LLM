# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The disagg gate: a benchmark scored across the hop of disaggregated serving.

``accuracy.disagg`` states one of two forms. **A deployment**: the tree's three files
(``config`` for ``trtllm-serve disaggregated -c``, the ``context`` and ``generation`` servers'
``--config``). The flow serves it, reading only each side's ``num_instances`` and a server's
``tensor_parallel_size`` x ``pipeline_parallel_size`` (:func:`plan_layout`), and holds the
unit's benchmark through the proxy (``harness/disagg_driver.py``) to the accuracy gate's floor.

**A test** (``test``): a pytest the run writes beside the tree's own, run from the tree's
pytest root; with ``nodes > 1`` one process per node of one allocation
(``MV2FLOW_JOB_NODE_RANK/HOSTS/NODES``), and every node's report must pass. A test that
skipped, or a node id that selects nothing, fails it.
"""

from __future__ import annotations

import shlex
import time
import xml.etree.ElementTree as ET
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
from . import check_config
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .harness import disagg_driver, workload_io
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

GATE_NAME = "disagg"
#: Where the test keeps what it writes for after it: under the gate's work directory, never
#: in the read-only snapshot it runs in, which is pruned once the gate ends.
ARTIFACTS_ENV = "MV2FLOW_GATE_ARTIFACTS"
#: Seconds the test may take: two engines to start and a benchmark to score.
DEFAULT_TIMEOUT_S = 7200
#: What marks the directory pytest runs a test from, as pytest itself decides its root.
PYTEST_ROOT_MARKERS: tuple[str, ...] = ("pytest.ini",)
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
    from .accuracy import judge_score, resolve_model_key, resolve_reference

    _verdict = verdict_factory(inspect_tree(task_data))
    accuracy = dict(task_data.get(ACCURACY_FIELD) or {})
    benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or "gsm8k")
    arguments = [str(arg) for arg in resolve_per_unit(accuracy, "benchmark_args", unit) or ()]
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
    return _verdict(**judge_score(accuracy, unit, benchmark, reference, combined, said=said), **ran)


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
        return _verdict(passed=False, detail=ended_early(result, spec.timeout), **ran)
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
    "job_span",
    "judge_report",
    "plan_layout",
    "pytest_root",
    "run_disagg_gate",
]
