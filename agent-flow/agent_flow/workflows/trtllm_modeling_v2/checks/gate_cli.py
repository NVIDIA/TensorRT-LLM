# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``trtllm-modeling-v2 gate <name>``: run one gate or witness exactly as the orchestrator does.

A hand-reconstructed gate command can pass where the orchestrator's own run
fails, for instance because it runs from another directory. This calls the
workflow's own gate functions, with the environment, runner and flags
resolved from the same task.yaml, so there is nothing to reconstruct.

Like the orchestrator's checks, it measures a snapshot of one commit of
``trtllm_repo_path`` (``run/snapshots.py``), never the checkout itself, which
the run's integrations move: the commit the checkout is at, or ``--commit``.
The snapshots are the run's own when ``--task`` is the workspace's
``task.yaml``, so a commit a check already measured is not checked out again.
The verdict says which commit every command imported, and is void when one
imported anything else.

It is also how a gate is replayed on an arbitrary TensorRT-LLM tree (a
worktree with uncommitted work, a historical checkout): ``--tree`` measures
that tree as it is -- every command runs from inside it with it first on
``PYTHONPATH`` -- and ``--results`` keeps the per-tree measurements that the
acceptance gate compares a later tree against::

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

from ..infra.environment import RunEnvironment, resolve_environment, scheduler_owns_devices
from ..infra.execution import select_command_executor, shell_runner_for
from ..infra.vcs import resolve_commit
from ..spec import variants as variants_mod
from ..spec.task_schema import (
    TaskSchemaError,
    assembly_units,
    load_and_validate_task_yaml,
    members_for,
    parse_parallel_segment,
    widest_world_size,
)
from . import gates as gates_mod

#: The gates and witnesses this command runs, by the name it takes.
GATE_NAMES: tuple[str, ...] = (
    "boot",
    "accuracy",
    "acceptance",
    "envelope",
    "witness",
    "serve",
    "logits",
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
        "--devices",
        default=None,
        help="comma-separated device ids (default 0..world_size-1 locally; under a scheduler, "
        "the devices it hands the job)",
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
    """The features a target implements once ``unit`` is assembled.

    A configuration variant's target is its unit's.
    """
    if variants_mod.is_variant(task_data, unit):
        unit = variants_mod.base_unit(task_data, unit)
    units = assembly_units(task_data)
    if unit not in units:
        return [str(f) for f in task_data.get("features") or ()]
    return [str(u) for u in units[1 : units.index(unit) + 1]]


def in_tree(runner: Callable[..., Any], tree: Path | None) -> Callable[..., Any]:
    """Run every command from inside ``tree``, with it first on ``PYTHONPATH``.

    An MPI worker resolves ``tensorrt_llm`` from its working directory ahead
    of ``PYTHONPATH``, so a job started elsewhere measures another checkout on
    every rank but the first; a script's own directory, not the working
    directory, leads its ``sys.path``, so the tree goes on ``PYTHONPATH`` as
    well. It is put ahead of what the command inherits rather than in its
    place: a session's resolve hook is on it.
    """
    if tree is None:
        return runner
    root = shlex.quote(str(Path(tree).resolve()))
    prefix = f'cd {root} && export PYTHONPATH={root}"${{PYTHONPATH:+:$PYTHONPATH}}" && '

    def run(command: str, env: Mapping[str, str] | None = None, timeout: int = 1800):
        return runner(prefix + command, env, timeout)

    return run


@contextlib.contextmanager
def _snapshot(args: argparse.Namespace, task_data: Mapping[str, Any]) -> Iterator[Any]:
    """The snapshot this gate measures (a ``run.snapshots.CheckTree``), leased while it runs.

    ``None`` when the gate measures a tree as it is: ``--tree``'s, or a
    checkout that is not a git one, which has no commit to snapshot.
    """
    from ..run import snapshots as snapshots_mod
    from ..run.workflow import anchor_tables, gate_inputs

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
        cone_paths=gate_inputs(task_data, repo) | set(anchor_tables(task_data, repo)),
    )
    with snapshots.lease(commit) as tree:
        yield snapshots_mod.CheckTree(commit=commit, tree=tree, snapshots=snapshots)


def run_gate(args: argparse.Namespace) -> gates_mod.GateResult:
    """Resolve everything from the task and run the named gate on the tree it measures."""
    from . import witness as witness_mod

    task_data = load_and_validate_task_yaml(args.task)
    unit = args.unit
    world = args.world_size
    member = args.member
    if variants_mod.is_variant(task_data, unit):
        task_data = variants_mod.variant_spec(
            task_data,
            unit,
            world_size=widest_world_size(task_data, world or 1),
            options_dir=args.workdir.resolve() / variants_mod.OPTIONS_DIRNAME,
        )
    if member is None:
        members = members_for(task_data, unit, world or 1)
        member = gates_mod.order_members_by_cost(members)[0]
    if world is None:
        world = parse_parallel_segment(member)[1] or 1
    if args.devices:
        devices = tuple(int(d) for d in args.devices.split(","))
    elif scheduler_owns_devices(str((task_data.get("environment") or {}).get("kind", "local"))):
        # The scheduler exports CUDA_VISIBLE_DEVICES per job; naming devices here would
        # mask the ones it assigned and stack concurrent jobs on the same cards.
        devices = ()
    else:
        devices = tuple(range(world))
    environment = resolve_environment(task_data, devices=devices, world_size=world)
    if args.python:
        environment = replace(environment, python=args.python)
    with _snapshot(args, task_data) as pin:
        tree = pin.tree if pin is not None else args.tree
        if pin is not None:
            task_data = pin.spec(task_data, pin.snapshots.tracked(pin.commit))
        elif tree is not None:
            task_data = {**task_data, "trtllm_repo_path": str(tree.resolve())}
        # An equivalence witness runs both legs' engines in one job (leg B is two engines of
        # the member's width), so it asks for the node, as the orchestrator's run does.
        witness_kind = witness_mod.WITNESS_KINDS.get(unit) if args.name == "witness" else None
        gpus = "all" if witness_kind == "equivalence" else str(environment.world_size)
        executor = select_command_executor(
            environment, name_prefix=f"gate-{args.name}", gpus=gpus, tree=tree
        )
        runner = shell_runner_for(executor)
        runner = pin.runner(runner) if pin is not None else in_tree(runner, tree)
        result = _run_named(args, task_data, environment, runner, unit=unit, member=member)
        return pin.stamp(result) if pin is not None else result


def _run_named(
    args: argparse.Namespace,
    task_data: dict[str, Any],
    environment: RunEnvironment,
    runner: Callable[..., Any],
    *,
    unit: str,
    member: str,
) -> gates_mod.GateResult:
    """Run the gate ``args.name`` names, with everything already resolved."""
    from ..run.preflight import resolve_model_key
    from . import envelope_boot, logit_agreement, serve_gate
    from . import witness as witness_mod

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
            workdir=workdir / unit,
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
    if args.name == "envelope":
        return envelope_boot.run_envelope_boot(
            environment,
            features=features,
            workdir=workdir / "envelope",
            allow_dirty=args.allow_dirty or None,
            **common,
        )
    if args.name == "serve":
        spec = serve_gate.ServeSpec.from_task(task_data)
        return serve_gate.run_serve_gate(
            environment,
            features=features,
            workdir=workdir / unit,
            boot=gates_mod.resolve_boot_spec(task_data, unit, features, args.boot_yaml),
            spec=replace(spec, timeout=args.timeout) if args.timeout else spec,
            allow_dirty=args.allow_dirty or None,
            **common,
        )
    if args.name == "logits":
        spec = logit_agreement.LogitsSpec.from_task(task_data)
        return logit_agreement.run_logit_gate(
            environment,
            features=features,
            workdir=workdir / unit,
            spec=replace(spec, timeout=args.timeout) if args.timeout else spec,
            allow_dirty=args.allow_dirty or None,
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


#: The workflow's CLI as the run's agents call it: ``<workspace>/bin/<this>``.
CLI_NAME = "trtllm-modeling-v2"


def install_cli(workspace: str | Path) -> Path:
    """``<workspace>/bin/trtllm-modeling-v2``: this interpreter and package, callable by path.

    An agent's shell has neither the orchestrator's interpreter nor its package
    on ``PATH``, so the command its prompt names would not run there. Python
    puts the working directory ahead of ``PYTHONPATH`` for ``-m``, so a shell
    inside another agent-flow checkout would import that checkout's package;
    ``PYTHONSAFEPATH`` keeps the working directory off ``sys.path``.
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
    from ..run.snapshots import SnapshotError

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
    "in_tree",
    "invocation",
    "main",
    "run_gate",
]
