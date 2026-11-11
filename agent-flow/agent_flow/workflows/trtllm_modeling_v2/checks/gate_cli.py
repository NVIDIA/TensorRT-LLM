# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``trtllm-modeling-v2 gate <name>``: run one check, the way the final certification runs it.

:func:`run_check` is the one place a check is resolved and run: the worker's
``gate`` command calls it on the worker's own worktree (``--tree``) while it
builds, and the certification (``run/certify.py``) calls it for every check
of the contract on the final commit. The environment, runner and flags come
from the same ``task.yaml`` either way::

    trtllm-modeling-v2 gate boot --task <workspace>/task.yaml --unit core --member tp4 \
        --tree <your worktree>

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
from typing import Any, Callable, Mapping, Sequence

from ..infra.compute.environment import resolve_environment
from ..infra.compute.executor import select_command_executor, shell_runner_for
from ..infra.files import read_json
from ..spec import shorthand
from ..spec import variants as variants_mod
from ..spec.task_schema import (
    TaskSchemaError,
    assembly_units,
    load_and_validate_task_yaml,
    members_for,
    parse_parallel_segment,
    widest_world_size,
)
from . import accuracy as accuracy_checks
from . import boot as boot_checks
from . import check_config, check_runtime, datasets, disagg, serve
from . import verdict as check_result
from .accuracy import resolve_model_key
from .harness import workload_io

#: The gates this command runs, by the name it takes.
GATE_NAMES: tuple[str, ...] = (
    "boot",
    "accuracy",
    "serve",
    "disagg",
)
#: The run's checkpoint file, beside the workspace's ``task.yaml``: it names the claimed devices.
_STATE_FILENAME = ".trtllm_modeling_v2_state.json"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 gate",
        description="Run one trtllm_modeling_v2 check standalone, as the certification runs it.",
    )
    parser.add_argument("name", choices=GATE_NAMES, help="the gate to run")
    parser.add_argument("--task", type=Path, required=True, help="the run's task.yaml")
    parser.add_argument(
        "--unit",
        default="core",
        help="the assembly unit or configuration variant to check (default core); disagg "
        "checks the disagg unit",
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
        help="the TensorRT-LLM tree to measure, as it is: your worktree. Commands run from "
        "inside it, with it first on PYTHONPATH (default: task.yaml's trtllm_repo_path)",
    )
    parser.add_argument(
        "--python", default=None, help="the interpreter to run under (default: task.yaml's)"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("trtllm-modeling-v2-gates"),
        help="where the gate writes its files. Default ./trtllm-modeling-v2-gates",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="comma-separated features the target implements (default: the unit's and every "
        "earlier unit's)",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated device ids (default: the devices the run claimed, else "
        "0..world_size-1)",
    )
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    parser.add_argument(
        "--mode",
        choices=(check_runtime.MODE_REQUIRE, check_runtime.MODE_OFF),
        default=check_runtime.MODE_REQUIRE,
        help="boot only: off runs the same command on the stock path, as a control",
    )
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON")
    return parser.parse_args(argv)


def features_for(task_data: Mapping[str, Any], unit: str) -> list[str]:
    """The features a target implements once ``unit`` is assembled.

    A configuration variant's target is its unit's.
    """
    if variants_mod.is_variant(task_data, unit):
        unit = variants_mod.base_unit(task_data, unit)
    units = assembly_units(task_data)
    if unit not in units:
        return [str(f) for f in task_data.get("features") or ()]
    return [str(u) for u in units[1 : units.index(unit) + 1]]


def gate_name(gate: str) -> str:
    """A contract check's gate (``accuracy[gsm8k]``) as the name this module runs (``accuracy``)."""
    return gate.partition("[")[0]


def claimed_devices(workspace: str | Path | None) -> tuple[int, ...]:
    """The devices the run in ``workspace`` claimed at preflight; none when it does not say."""
    if workspace is None:
        return ()
    state = read_json(Path(workspace) / _STATE_FILENAME) or {}
    try:
        return tuple(int(d) for d in state.get("devices") or ())
    except (TypeError, ValueError):
        return ()


class _ImportWatch:
    """A check's runner, noting where each engine the check started imported the package from.

    The engine driver says it once per engine (``workload_io.DRIVER_MARKER``). A
    check one of whose engines imported ``tensorrt_llm`` from anywhere but the
    tree it measures did not measure that tree: :meth:`judge` fails it, whatever
    it measured. A command that does not say is not judged.
    """

    def __init__(self, runner: Callable[..., Any], tree: Path) -> None:
        self.runner = runner
        self.tree = Path(tree).resolve()
        self.roots: set[str] = set()

    def __call__(self, command: str, env: Mapping[str, str] | None = None, timeout: int = 1800):
        result = self.runner(command, env, timeout)
        output = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
        for record in workload_io.parse_marked(output, workload_io.DRIVER_MARKER):
            origin = Path(str(record.get("tensorrt_llm") or ""))
            if origin.name:
                # .../<root>/tensorrt_llm/__init__.py -> <root>
                package = origin.parent if origin.name == "__init__.py" else origin
                self.roots.add(str(package.parent.resolve()))
        return result

    def judge(self, result: check_result.GateResult) -> check_result.GateResult:
        others = sorted(root for root in self.roots if Path(root) != self.tree)
        if not others:
            return result
        return replace(
            result,
            passed=False,
            explained=True,
            detail=f"its engines imported tensorrt_llm from {', '.join(others)}, not from the tree "
            f"it measures ({self.tree}); what it measured: {result.detail}",
        )


def run_check(
    task_data: Mapping[str, Any],
    gate: str,
    unit: str,
    member: str | None = None,
    *,
    workdir: str | Path,
    tree: str | Path | None = None,
    workspace: str | Path | None = None,
    runner: Callable[..., Any] | None = None,
    devices: Sequence[int] | None = None,
    world_size: int | None = None,
    python: str | None = None,
    timeout: int | None = None,
    features: Sequence[str] | None = None,
    mode: str = check_runtime.MODE_REQUIRE,
    anchor_score: float | None = None,
) -> check_result.GateResult:
    """Resolve everything from the task and run ``gate`` for ``unit`` at ``member`` on ``tree``.

    ``tree`` is the TensorRT-LLM tree measured as it is: the checkout
    (``trtllm_repo_path``) by default. Every command runs from inside it with
    it first on ``PYTHONPATH``. ``workspace`` is the run's: its staged datasets
    (``checks/datasets.py``) and claimed devices are used when it is given.
    ``runner`` replaces the environment's own, for tests. A variant whose
    accuracy anchor is relative is judged against ``anchor_score`` (its unit's
    score at the same member on the same tree); without one its unit's
    benchmark runs here first. A check whose engines imported the package from
    another tree fails (:class:`_ImportWatch`): it measured something else.
    """
    name = gate_name(gate)
    if name not in GATE_NAMES:
        raise TaskSchemaError(f"{gate!r} names no gate this workflow runs")
    declared = dict(task_data)
    spec = declared
    world = world_size
    workdir = Path(workdir).resolve()
    if variants_mod.is_variant(declared, unit):
        spec = shorthand.variant_spec(
            declared, unit, world_size=widest_world_size(declared, world or 1)
        )
    if member is None:
        member = shorthand.order_members_by_cost(members_for(spec, unit, world or 1))[0]
    if world is None:
        world = parse_parallel_segment(member)[1] or 1
    # The disaggregated-serving test starts two engines, so it asks for both.
    disagg_spec = disagg.DisaggSpec.from_task(spec) if name == "disagg" else None
    gpus = (disagg_spec.device_count(member) if disagg_spec else None) or world
    if devices is None:
        devices = claimed_devices(workspace)
    # A check takes as many of the run's devices as it needs, the first ones; never a device
    # the run did not claim. With no run behind it (a bare command line), the first ones.
    devices = tuple(devices)[:gpus] if devices else tuple(range(gpus))
    repo = Path(str(spec.get("trtllm_repo_path") or "."))
    tree = Path(tree).resolve() if tree is not None else repo.resolve()
    spec = {**spec, "trtllm_repo_path": str(tree)}
    declared = {**declared, "trtllm_repo_path": str(tree)}
    environment = resolve_environment(spec, devices=devices, world_size=world)
    unit_environment = resolve_environment(declared, devices=devices, world_size=world)
    if python:
        environment = replace(environment, python=python)
        unit_environment = replace(unit_environment, python=python)
    if runner is None:
        runner = shell_runner_for(select_command_executor(environment))
    runner = watch = _ImportWatch(check_runtime.in_tree(runner, tree), tree)
    stated = features is not None
    features = list(features) if stated else features_for(spec, unit)
    # What the check runs: the deployment the unit or variant is at this member.
    deployment = check_config.deployment_of(
        declared, unit, member, world, features if stated else None
    )
    spec = datasets.staged_spec(spec, gate if "[" in gate else name, workspace)
    bound = {"timeout": timeout} if timeout else {}
    common = dict(task_data=spec, runner=runner, unit=unit, member=member, deployment=deployment)

    def measure() -> check_result.GateResult:
        nonlocal anchor_score, disagg_spec
        if name == "boot":
            return boot_checks.run_boot_gate(
                environment,
                spec=boot_checks.resolve_boot_spec(spec, unit, features),
                features=features,
                workdir=workdir / unit,
                mode=mode,
                **bound,
                **common,
            )
        if name == "accuracy":
            relative = variants_mod.anchor_state(declared, unit) == variants_mod.ANCHOR_RELATIVE
            if relative and anchor_score is None:
                base = variants_mod.base_unit(declared, unit)
                try:
                    anchor = accuracy_checks.resolve_reference(
                        declared, model_key=resolve_model_key(declared, base), unit=base
                    )
                except (KeyError, FileNotFoundError, TypeError, ValueError):
                    anchor = 0.0  # only the score it measures is used, as the variant's anchor
                first = accuracy_checks.run_accuracy_gate(
                    unit_environment,
                    reference=anchor,
                    workdir=workdir / base,
                    **bound,
                    **{
                        **common,
                        "task_data": datasets.staged_spec(declared, gate, workspace),
                        "unit": base,
                        "deployment": check_config.deployment_of(declared, base, member, world),
                    },
                )
                if first.infrastructure or first.measured is None:
                    return replace(
                        first,
                        passed=False,
                        detail=f"{base}'s benchmark at {member} measured no score ({first.detail}), "
                        f"so {unit}, whose anchor is that score, has nothing to be judged against",
                    )
                anchor_score = first.measured
            if relative:
                return accuracy_checks.run_accuracy_gate(
                    environment,
                    reference=anchor_score,
                    workdir=workdir / unit,
                    relative=True,
                    **bound,
                    **common,
                )
            reference = accuracy_checks.resolve_reference(
                spec, model_key=resolve_model_key(spec, unit), unit=unit
            )
            return accuracy_checks.run_accuracy_gate(
                environment, reference=reference, workdir=workdir / unit, **bound, **common
            )
        if name == "serve":
            serve_spec = serve.ServeSpec.from_task(spec)
            return serve.run_serve_gate(
                environment,
                features=features,
                workdir=workdir / unit,
                boot=boot_checks.resolve_boot_spec(spec, unit, features),
                spec=replace(serve_spec, timeout=timeout) if timeout else serve_spec,
                **common,
            )
        if disagg_spec is not None and timeout:
            disagg_spec = replace(disagg_spec, timeout=timeout)
        common.pop("deployment")
        return disagg.run_disagg_gate(
            environment, workdir=workdir / unit, spec=disagg_spec, **common
        )

    return watch.judge(measure())


def run_gate(args: argparse.Namespace) -> check_result.GateResult:
    """The ``gate`` command: :func:`run_check` with what the command line names."""
    task = Path(args.task)
    return run_check(
        load_and_validate_task_yaml(task),
        args.name,
        args.unit,
        args.member,
        workdir=args.workdir,
        tree=args.tree,
        # The run's own task.yaml sits in its workspace, beside its datasets and its state.
        workspace=task.resolve().parent,
        devices=tuple(int(d) for d in args.devices.split(",")) if args.devices else None,
        world_size=args.world_size,
        python=args.python,
        timeout=args.timeout,
        features=[f for f in args.features.split(",") if f] if args.features is not None else None,
        mode=args.mode,
    )


#: The workflow's CLI as the run's agents call it: ``<workspace>/bin/<this>``.
CLI_NAME = "trtllm-modeling-v2"


def install_cli(workspace: str | Path) -> Path:
    """``<workspace>/bin/trtllm-modeling-v2``: this interpreter and package, callable by path.

    ``PYTHONSAFEPATH`` keeps the working directory off ``sys.path``, so a shell
    inside another agent-flow checkout still imports this package.
    """
    import agent_flow

    root = Path(agent_flow.__file__).resolve().parents[1]
    script = Path(workspace) / "bin" / CLI_NAME
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/bin/bash\n"
        "# The trtllm-modeling-v2 CLI under the orchestrator's interpreter and package.\n"
        f'export PYTHONPATH={shlex.quote(str(root))}"${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        "export PYTHONSAFEPATH=1\n"
        f"exec {shlex.quote(sys.executable)} -m agent_flow.workflows.trtllm_modeling_v2.cli "
        '"$@"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def invocation(task: str | Path, workdir: str | Path, unit: str, cli: str = CLI_NAME) -> str:
    """The command a worker runs one of its unit's checks with, on its own worktree."""
    return (
        f"{cli} gate <{'|'.join(GATE_NAMES)}> --task {task} --unit {unit} "
        f"--member <member> --workdir {workdir} --tree <your worktree>"
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


__all__ = [
    "GATE_NAMES",
    "claimed_devices",
    "features_for",
    "gate_name",
    "install_cli",
    "invocation",
    "main",
    "run_check",
    "run_gate",
]
