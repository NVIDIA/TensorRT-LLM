# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``trtllm-modeling-v2 gate <name>``: run one gate or witness exactly as the orchestrator does.

An assembler that "runs the witness exactly as the workflow runs it" used to
reconstruct the command, and passed where the orchestrator's own run of the
same witness failed: the reconstruction ran from another directory. This
calls the workflow's own gate functions, with the environment, runner and
flags resolved from the same task.yaml, so there is nothing to reconstruct.

It is also how a gate is replayed on an arbitrary TensorRT-LLM tree (a
historical commit, a worktree): ``--tree`` measures that tree instead of
``trtllm_repo_path`` -- every command runs from inside it with it first on
``PYTHONPATH`` -- and ``--results`` keeps the per-tree measurements that the
acceptance and performance gates compare a later tree against::

    trtllm-modeling-v2 gate scheduler-equivalence --task task.yaml \
        --unit dspark --member tep4 --tree .dev-worktrees/pre-fix

Exit status: 0 when the gate passed, 1 when it failed, 2 on a usage or spec
error. ``--json`` prints the verdict as JSON instead of its summary line.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from . import gates as gates_mod
from .environment import RunEnvironment, resolve_environment
from .execution import select_command_executor, shell_runner_for
from .task_schema import (
    TaskSchemaError,
    assembly_units,
    load_and_validate_task_yaml,
    members_for,
    parse_parallel_segment,
)

#: The gates and witnesses this command runs, by the name it takes.
GATE_NAMES: tuple[str, ...] = (
    "boot",
    "accuracy",
    "acceptance",
    "scheduler-equivalence",
    "launch",
    "witness",
)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 gate",
        description="Run one trtllm_modeling_v2 gate or witness standalone, as the orchestrator runs it.",
    )
    parser.add_argument("name", choices=GATE_NAMES, help="the gate or witness to run")
    parser.add_argument("--task", type=Path, required=True, help="the run's task.yaml")
    parser.add_argument(
        "--unit",
        default="core",
        help="the assembly unit whose configuration to gate (default core); a witness "
        "needs the unit whose feature it witnesses",
    )
    parser.add_argument(
        "--member",
        default=None,
        help="the parallel configuration to serve at (default: the unit's cheapest member)",
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=None,
        help="measure this TensorRT-LLM tree instead of task.yaml's trtllm_repo_path: commands "
        "run from inside it, with it first on PYTHONPATH",
    )
    parser.add_argument(
        "--python", default=None, help="the interpreter to run under (default: task.yaml's)"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("trtllm-modeling-v2-gates"),
        help="where the gate writes the files its job reads; on the shared filesystem when the "
        "job runs elsewhere (a spool session). Default ./trtllm-modeling-v2-gates",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="the results file the acceptance and performance gates compare against and append "
        "to (default <workdir>/gate_results.jsonl); share one across trees to compare them",
    )
    parser.add_argument(
        "--boot-yaml",
        type=Path,
        default=None,
        help="a workspace boot.yaml, used when task.yaml declares no accuracy.boot",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="comma-separated features the target implements (default: the unit's and every "
        "earlier unit's)",
    )
    parser.add_argument(
        "--devices", default=None, help="comma-separated device ids (default 0..world_size-1)"
    )
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="seconds, per leg where legged")
    parser.add_argument(
        "--mode",
        choices=(gates_mod.MODE_REQUIRE, gates_mod.MODE_OFF),
        default=gates_mod.MODE_REQUIRE,
        help="boot only: off runs the same command on the stock path, as a control",
    )
    parser.add_argument("--allow-dirty", action="store_true", help="measure a dirty target path")
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON")
    return parser.parse_args(argv)


def features_for(task_data: Mapping[str, Any], unit: str) -> list[str]:
    """The features a target implements once ``unit`` is assembled."""
    units = assembly_units(task_data)
    if unit not in units:
        return [str(f) for f in task_data.get("features") or ()]
    return [str(u) for u in units[1 : units.index(unit) + 1]]


def _with_tree(task_data: dict[str, Any], environment: RunEnvironment, tree: Path | None):
    """Point the task and the environment at ``tree``."""
    if tree is None:
        return task_data, environment
    root = str(tree.resolve())
    task_data = {**task_data, "trtllm_repo_path": root}
    inherited = environment.env.get("PYTHONPATH")
    path = f"{root}:{inherited}" if inherited else root
    return task_data, replace(environment, env={**environment.env, "PYTHONPATH": path})


def _in_tree(runner: Callable[..., Any], tree: Path | None) -> Callable[..., Any]:
    """Run every command from inside ``tree``.

    An MPI worker resolves ``tensorrt_llm`` from its working directory ahead
    of ``PYTHONPATH``, so a job started elsewhere measures another checkout on
    every rank but the first.
    """
    if tree is None:
        return runner
    prefix = f"cd {shlex.quote(str(tree.resolve()))} && "

    def run(command: str, env: Mapping[str, str] | None = None, timeout: int = 1800):
        return runner(prefix + command, env, timeout)

    return run


def run_gate(args: argparse.Namespace) -> gates_mod.GateResult:
    """Resolve everything from the task and run the named gate."""
    from . import launch_gate, step_check
    from . import witness as witness_mod
    from .preflight import resolve_model_key

    task_data = load_and_validate_task_yaml(args.task)
    unit = args.unit
    world = args.world_size
    member = args.member
    if member is None:
        members = members_for(task_data, unit, world or 1)
        member = gates_mod.order_members_by_cost(members)[0]
    if world is None:
        world = parse_parallel_segment(member)[1] or 1
    devices = (
        tuple(int(d) for d in args.devices.split(",")) if args.devices else tuple(range(world))
    )
    environment = resolve_environment(task_data, devices=devices, world_size=world, workload="gate")
    if args.python:
        environment = replace(environment, python=args.python)
    task_data, environment = _with_tree(task_data, environment, args.tree)
    executor = select_command_executor(environment, name_prefix=f"gate-{args.name}")
    runner = _in_tree(shell_runner_for(executor), args.tree)
    features = (
        [f for f in args.features.split(",") if f]
        if args.features is not None
        else features_for(task_data, unit)
    )
    workdir = args.workdir.resolve()
    records = gates_mod.GateRecords(args.results or workdir / gates_mod.RESULTS_FILENAME)
    timeout = {"timeout": args.timeout} if args.timeout else {}
    common = dict(task_data=task_data, runner=runner, unit=unit, member=member)

    if args.name == "boot":
        return gates_mod.run_boot_gate(
            environment,
            spec=gates_mod.resolve_boot_spec(task_data, unit, features, args.boot_yaml),
            features=features,
            workdir=workdir / unit,
            mode=args.mode,
            allow_dirty=args.allow_dirty or None,
            **timeout,
            **common,
        )
    if args.name == "accuracy":
        reference = gates_mod.resolve_reference(
            task_data, model_key=resolve_model_key(task_data, unit), unit=unit
        )
        return gates_mod.run_accuracy_gate(
            environment,
            reference=reference,
            allow_dirty=args.allow_dirty or None,
            **timeout,
            **common,
        )
    if args.name == "acceptance":
        return gates_mod.run_acceptance_gate(
            environment,
            features=features,
            workdir=workdir / unit,
            records=records,
            allow_dirty=args.allow_dirty or None,
            **timeout,
            **common,
        )
    if args.name == "scheduler-equivalence":
        return step_check.run_scheduler_equivalence(
            environment,
            features=features,
            workdir=workdir / unit / "scheduler_equivalence",
            allow_dirty=args.allow_dirty or None,
            **timeout,
            **common,
        )
    if args.name == "launch":
        return launch_gate.run_launch_gate(
            environment,
            reference=launch_gate.smoke_reference(
                task_data, unit, lambda scope: resolve_model_key(task_data, scope)
            ),
            features=features,
            workdir=workdir / unit / "launch",
            allow_dirty=args.allow_dirty or None,
            **timeout,
            **common,
        )
    if unit not in witness_mod.WITNESS_KINDS:
        raise TaskSchemaError(
            f"unit {unit!r} has no witness; the witnessed features are "
            f"{sorted(witness_mod.WITNESS_KINDS)}"
        )
    return witness_mod.run_declared_witness(
        environment,
        features=features,
        workdir=workdir / "witness" / unit,
        boot=gates_mod.resolve_boot_spec(task_data, unit, None, args.boot_yaml),
        **common,
    )


def invocation(task: str | Path, workdir: str | Path, unit: str) -> str:
    """The command an agent uses to rerun one of the orchestrator's gates by hand."""
    return (
        f"trtllm-modeling-v2 gate <{'|'.join(GATE_NAMES)}> --task {task} --unit {unit} "
        f"--member <member> --workdir {workdir}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run one gate; print its verdict; return 0 on a pass, 1 on a failure, 2 on an error."""
    args = _parse(list(sys.argv[1:] if argv is None else argv))
    try:
        result = run_gate(args)
    except (TaskSchemaError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(asdict(result), indent=1, default=str))
    else:
        print(result.summary())
        if not result.passed and result.stdout_tail.strip():
            print(result.stdout_tail)
    return 0 if result.passed else 1


__all__ = ["GATE_NAMES", "features_for", "invocation", "main", "run_gate"]
