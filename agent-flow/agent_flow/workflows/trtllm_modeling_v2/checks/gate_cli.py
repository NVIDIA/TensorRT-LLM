# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``trtllm-modeling-v2 gate <name>``: run one gate exactly as the orchestrator does.

The workflow's own gate functions, with the environment, runner and flags
resolved from the same task.yaml. Like the orchestrator's checks, it measures
a snapshot of one commit of ``trtllm_repo_path`` (``run/workspace/snapshots.py``): the
commit the checkout is at, or ``--commit``. ``--tree`` measures another tree
as it is, and ``--results`` keeps the per-tree measurements the acceptance
gate compares a later tree against::

    trtllm-modeling-v2 gate acceptance --task task.yaml \
        --unit dspark --member tep4 --tree .dev-worktrees/pre-fix

Exit status: 0 when the gate passed, 1 when it failed, 2 on a usage or spec
error. ``--json`` prints the verdict as JSON instead of its summary line.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from ..infra.compute.environment import RunEnvironment, resolve_environment, scheduler_owns_devices
from ..infra.compute.executor import FEATURE, select_command_executor, shell_runner_for
from ..infra.git import resolve_commit
from ..spec import variants as variants_mod
from ..spec.task_schema import (
    TaskSchemaError,
    assembly_units,
    load_and_validate_task_yaml,
    members_for,
    parse_parallel_segment,
    widest_world_size,
)
from . import acceptance as acceptance_checks
from . import accuracy as accuracy_checks
from . import boot as boot_checks
from . import check_config, check_runtime, gate_records
from . import verdict as check_result

#: The gates this command runs, by the name it takes.
GATE_NAMES: tuple[str, ...] = (
    "boot",
    "accuracy",
    "acceptance",
    "serve",
    "disagg",
)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 gate",
        description="Run one trtllm_modeling_v2 gate standalone, as the orchestrator runs it.",
    )
    parser.add_argument("name", choices=GATE_NAMES, help="the gate to run")
    parser.add_argument("--task", type=Path, required=True, help="the run's task.yaml")
    parser.add_argument(
        "--unit",
        default="core",
        help="the assembly unit whose configuration to gate (default core); disagg "
        "gates the disagg unit",
    )
    parser.add_argument(
        "--member",
        default=None,
        help="the parallel configuration to serve at (default: the unit's cheapest member)",
    )
    tree = parser.add_mutually_exclusive_group()
    tree.add_argument(
        "--tree",
        type=Path,
        default=None,
        help="measure this TensorRT-LLM tree as it is, instead of a snapshot of a commit of "
        "task.yaml's trtllm_repo_path: commands run from inside it, with it first on PYTHONPATH",
    )
    tree.add_argument(
        "--commit",
        default=None,
        help="measure a snapshot of this commit of task.yaml's trtllm_repo_path (default: the "
        "commit the checkout is at)",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        help="where snapshots of commits are kept (default: snapshots/ beside --task, the "
        "run's own when --task is the workspace's task.yaml)",
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
        help="the results file the acceptance gate compares against and appends "
        "to (default <workdir>/gate_results.jsonl); share one across trees to compare them",
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
        help="comma-separated device ids (default 0..world_size-1 locally; under a scheduler, "
        "the devices it hands the job)",
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


@contextlib.contextmanager
def _snapshot(args: argparse.Namespace, task_data: Mapping[str, Any]) -> Iterator[Any]:
    """The snapshot this gate measures (a ``run.workspace.snapshots.CheckTree``), leased while it runs.

    ``None`` when the gate measures a tree as it is: ``--tree``'s, or a
    checkout that is not a git one, which has no commit to snapshot.
    """
    from ..run.workflow import gate_inputs
    from ..run.workspace import snapshots as snapshots_mod

    repo = Path(str(task_data.get("trtllm_repo_path") or ""))
    if args.tree is not None or not (repo / ".git").exists():
        if args.commit:
            raise TaskSchemaError(f"--commit {args.commit}: {repo} is not a git checkout")
        yield None
        return
    commit = resolve_commit(repo, args.commit or "HEAD")
    if commit is None:
        raise TaskSchemaError(f"{args.commit or 'HEAD'} names no commit of {repo}")
    snapshots = snapshots_mod.Snapshots(
        repo,
        args.snapshots or args.task.resolve().parent / snapshots_mod.SNAPSHOTS_DIRNAME,
        cone_paths=gate_inputs(task_data, repo),
    )
    with snapshots.lease(commit) as tree:
        yield snapshots_mod.CheckTree(commit=commit, tree=tree, snapshots=snapshots)


def run_gate(args: argparse.Namespace) -> check_result.GateResult:
    """Resolve everything from the task and run the named gate on the tree it measures."""
    from . import disagg

    task_data = load_and_validate_task_yaml(args.task)
    declared = task_data
    unit = args.unit
    world = args.world_size
    member = args.member
    if variants_mod.is_variant(task_data, unit):
        task_data = check_config.variant_spec(
            task_data,
            unit,
            world_size=widest_world_size(task_data, world or 1),
            options_dir=args.workdir.resolve() / variants_mod.OPTIONS_DIRNAME,
        )
    if member is None:
        members = members_for(task_data, unit, world or 1)
        member = check_config.order_members_by_cost(members)[0]
    if world is None:
        world = parse_parallel_segment(member)[1] or 1
    # The disaggregated-serving test starts two engines, so it asks for both, as the
    # orchestrator's run does.
    spec = disagg.DisaggSpec.from_task(task_data) if args.name == "disagg" else None
    gpus = (spec.device_count(member) if spec else None) or world
    if args.devices:
        devices = tuple(int(d) for d in args.devices.split(","))
    elif scheduler_owns_devices(str((task_data.get("environment") or {}).get("kind", "local"))):
        # The scheduler exports CUDA_VISIBLE_DEVICES per job; naming devices here would
        # mask the ones it assigned and stack concurrent jobs on the same cards.
        devices = ()
    else:
        devices = tuple(range(gpus))
    environment = resolve_environment(task_data, devices=devices, world_size=world)
    # The unit's own, without a variant's overlay: what a relative variant is judged against.
    unit_environment = resolve_environment(declared, devices=devices, world_size=world)
    if args.python:
        environment = replace(environment, python=args.python)
        unit_environment = replace(unit_environment, python=args.python)
    with _snapshot(args, task_data) as pin:
        tree = pin.tree if pin is not None else args.tree
        if pin is not None:
            tracked = pin.snapshots.tracked(pin.commit)
            task_data, declared = pin.spec(task_data, tracked), pin.spec(declared, tracked)
        elif tree is not None:
            task_data = {**task_data, "trtllm_repo_path": str(tree.resolve())}
            declared = {**declared, "trtllm_repo_path": str(tree.resolve())}
        # A worker's own gate run queues after the run's checks and before perf work.
        executor = select_command_executor(
            environment, name_prefix=f"gate-{args.name}", gpus=str(gpus), tree=tree,
            priority=FEATURE,
        )  # fmt: skip
        runner = shell_runner_for(executor)
        runner = pin.runner(runner) if pin is not None else check_runtime.in_tree(runner, tree)
        result = _run_named(
            args,
            task_data,
            environment,
            runner,
            unit=unit,
            member=member,
            declared=(declared, unit_environment),
        )
        return pin.stamp(result) if pin is not None else result


def _run_named(
    args: argparse.Namespace,
    task_data: dict[str, Any],
    environment: RunEnvironment,
    runner: Callable[..., Any],
    *,
    unit: str,
    member: str,
    declared: tuple[dict[str, Any], RunEnvironment] | None = None,
) -> check_result.GateResult:
    """Run the gate ``args.name`` names, with everything already resolved.

    ``declared`` is the spec as written and its environment, before a variant's
    overlay: a variant whose anchor is relative is judged against its unit's
    benchmark, run here first under them.
    """
    from ..checks.accuracy import resolve_model_key
    from . import disagg, serve

    features = (
        [f for f in args.features.split(",") if f]
        if args.features is not None
        else features_for(task_data, unit)
    )
    workdir = args.workdir.resolve()
    records = gate_records.GateRecords(args.results or workdir / gate_records.RESULTS_FILENAME)
    timeout = {"timeout": args.timeout} if args.timeout else {}
    common = dict(task_data=task_data, runner=runner, unit=unit, member=member)

    if args.name == "boot":
        return boot_checks.run_boot_gate(
            environment,
            spec=boot_checks.resolve_boot_spec(task_data, unit, features),
            features=features,
            workdir=workdir / unit,
            mode=args.mode,
            **timeout,
            **common,
        )
    spec, unit_environment = declared or (task_data, environment)
    if args.name == "accuracy" and (
        declared is not None
        and variants_mod.anchor_state(spec, unit) == variants_mod.ANCHOR_RELATIVE
    ):
        base = variants_mod.base_unit(spec, unit)
        try:
            anchor = accuracy_checks.resolve_reference(
                spec, model_key=resolve_model_key(spec, base), unit=base
            )
        except (KeyError, FileNotFoundError, TypeError, ValueError):
            anchor = 0.0  # only the score it measures is used, as the variant's anchor
        first = accuracy_checks.run_accuracy_gate(
            unit_environment,
            reference=anchor,
            workdir=workdir / base,
            **timeout,
            **{**common, "task_data": spec, "unit": base},
        )
        if first.infrastructure or first.measured is None:
            return replace(
                first,
                passed=False,
                detail=f"{base}'s benchmark at {member} measured no score ({first.detail}), so "
                f"{unit}, whose anchor is that score, has nothing to be judged against",
            )
        return accuracy_checks.run_accuracy_gate(
            environment,
            reference=first.measured,
            workdir=workdir / unit,
            relative=True,
            **timeout,
            **common,
        )
    if args.name == "accuracy":
        reference = accuracy_checks.resolve_reference(
            task_data, model_key=resolve_model_key(task_data, unit), unit=unit
        )
        return accuracy_checks.run_accuracy_gate(
            environment,
            reference=reference,
            workdir=workdir / unit,
            **timeout,
            **common,
        )
    if args.name == "acceptance":
        return acceptance_checks.run_acceptance_gate(
            environment,
            features=features,
            workdir=workdir / unit,
            records=records,
            **timeout,
            **common,
        )
    if args.name == "serve":
        spec = serve.ServeSpec.from_task(task_data)
        return serve.run_serve_gate(
            environment,
            features=features,
            workdir=workdir / unit,
            boot=boot_checks.resolve_boot_spec(task_data, unit, features),
            spec=replace(spec, timeout=args.timeout) if args.timeout else spec,
            **common,
        )
    spec = disagg.DisaggSpec.from_task(task_data)
    if spec is not None and args.timeout:
        spec = replace(spec, timeout=args.timeout)
    return disagg.run_disagg_gate(environment, workdir=workdir / unit, spec=spec, **common)


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
    """The command an agent uses to rerun one of the orchestrator's gates by hand."""
    return (
        f"{cli} gate <{'|'.join(GATE_NAMES)}> --task {task} --unit {unit} "
        f"--member <member> --workdir {workdir}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run one gate; print its verdict; return 0 on a pass, 1 on a failure, 2 on an error."""
    from ..run.workspace.snapshots import SnapshotError

    args = _parse(list(sys.argv[1:] if argv is None else argv))
    try:
        result = run_gate(args)
    except (TaskSchemaError, KeyError, FileNotFoundError, SnapshotError) as exc:
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
    "features_for",
    "invocation",
    "main",
    "run_gate",
]
