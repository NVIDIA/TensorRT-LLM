# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``trtllm-modeling-v2 gate <name>``: run one check, the way the run itself runs it.

:func:`run_check` is the one place a check is resolved and run: the worker's ``gate``
command calls it on the worker's own worktree (``--tree``) while it builds, and the run
calls it for the onboard check and the final pass (``run/report.py``). The deployment, its
options, the environment, the frozen anchors and the engine lock are the same either way::

    trtllm-modeling-v2 gate boot --task <workspace>/task.yaml --deployment tp4 \
        --tree <your worktree>

A deployment with a launch prefix of its own, and every disaggregated one, boots as a
server the check starts and asks through its endpoint; any other boots in process. An
aggregated deployment is scored by ``trtllm-eval`` behind its prefix, a disaggregated one
through its endpoint. The verdict says which path judged.

Exit status: 0 when the gate passed or measured a score nothing judges (``NO VERDICT``),
1 when it failed, 2 on a usage or spec error.
``--json`` prints the verdict as JSON instead of its summary line.
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
from ..spec import deployments as deployments_mod
from ..spec.deployments import AGGREGATED
from ..spec.task_schema import TaskSchemaError, load_and_validate_task_yaml
from . import accuracy as accuracy_checks
from . import anchors, check_runtime, disagg, endpoint, pytest_runner, serve, servers
from . import boot as boot_checks
from . import verdict as check_result

#: The gates this command runs, by the name it takes.
GATE_NAMES: tuple[str, ...] = ("boot", "accuracy", "serve", "tests")
IN_PROCESS = "in process"
THROUGH_ENDPOINT = "through the endpoint"
#: The run's checkpoint file, beside the workspace's ``task.yaml``: it names the claimed devices.
_STATE_FILENAME = ".trtllm_modeling_v2_state.json"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 gate",
        description="Run one trtllm_modeling_v2 check standalone, as the run itself runs it.",
    )
    parser.add_argument(
        "name", help=f"the gate to run: one of {', '.join(GATE_NAMES)}, or accuracy[<benchmark>]"
    )
    parser.add_argument("--task", type=Path, required=True, help="the run's task.yaml")
    parser.add_argument(
        "--deployment",
        default=None,
        help="the deployment to check, by its name in the spec (default: the first, core's)",
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
        "--devices",
        default=None,
        help="comma-separated device ids (default: the devices the run claimed, else the "
        "first ones the deployment's ranks need)",
    )
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    parser.add_argument(
        "--mode",
        choices=(check_runtime.MODE_REQUIRE, check_runtime.MODE_OFF),
        default=check_runtime.MODE_REQUIRE,
        help="boot only: off runs the same command on the stock path, as a control",
    )
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON")
    return parser.parse_args(argv)


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


def run_check(
    task_data: Mapping[str, Any],
    gate: str,
    deployment: str | None = None,
    *,
    workdir: str | Path,
    tree: str | Path | None = None,
    workspace: str | Path | None = None,
    runner: Callable[..., Any] | None = None,
    devices: Sequence[int] | None = None,
    python: str | None = None,
    timeout: int | None = None,
    mode: str = check_runtime.MODE_REQUIRE,
    relative_score: float | None = None,
    measure_base: bool = True,
    locked: bool = False,
) -> check_result.GateResult:
    """Run ``gate`` on the deployment named ``deployment`` (core's when unnamed), on ``tree``.

    ``tree`` is the TensorRT-LLM tree measured as it is: the checkout (``trtllm_repo_path``)
    by default. Every command runs from inside it with it first on ``PYTHONPATH``.
    ``workspace`` is the run's: its frozen anchors and claimed devices are used when it is
    given. ``runner`` replaces the environment's own, for tests. A benchmark whose anchor
    is another deployment's score is judged against ``relative_score``; without one that
    deployment's benchmark runs here first (``measure_base``), and when there is no such
    score this one's is reported with no verdict. ``locked``: the caller already holds the
    workspace's engine lock.
    """
    name = gate_name(gate)
    if name not in GATE_NAMES:
        raise TaskSchemaError(
            f"{gate!r} names no gate this workflow runs ({', '.join(GATE_NAMES)})"
        )
    found = deployments_mod.named(task_data, deployment)
    workdir = Path(workdir).resolve() / found.name
    repo = Path(str(task_data.get("trtllm_repo_path") or "."))
    tree = Path(tree).resolve() if tree is not None else repo.resolve()
    spec = {**task_data, "trtllm_repo_path": str(tree)}
    if python:
        spec["environment"] = {**(spec.get("environment") or {}), "python": python}
    runner_given, devices_given = runner, devices
    if devices is None:
        devices = claimed_devices(workspace)
    # A check takes as many of the run's devices as its ranks need, the first ones; never a
    # device the run did not claim. With no run behind it (a bare command line), the first ones.
    devices = tuple(devices)[: found.ranks] if devices else tuple(range(found.ranks))
    world = found.server.ranks if found.kind == AGGREGATED else 1
    environment = resolve_environment(spec, devices=devices, world_size=world)
    if runner is None:
        runner = shell_runner_for(select_command_executor(environment))
    runner = check_runtime.in_tree(runner, tree)
    bound = {"timeout": timeout} if timeout else {}
    common = dict(task_data=spec, deployment=found)
    served = found.kind != AGGREGATED or found.prefixed
    if workspace is None:
        workspace = workdir

    def accuracy() -> check_result.GateResult:
        nonlocal relative_score
        # `accuracy[<benchmark>]` names one of the deployment's; unnamed, its first.
        wanted = gate.partition("[")[2].rstrip("]")
        benchmark = found.benchmark(wanted) if wanted else next(iter(found.benchmarks), None)
        if benchmark is None:
            said = f"is not scored on `{wanted}`" if wanted else "names no benchmark"
            raise TaskSchemaError(f"deployment `{found.name}` {said}")
        held = anchors.frozen(workspace, found.name, benchmark.name) or anchors.resolve(
            spec, found, benchmark, tree
        )
        if held.kind == anchors.RELATIVE and relative_score is None and measure_base:
            first = run_check(
                task_data,
                f"accuracy[{benchmark.name}]",
                held.relative_to,
                workdir=workdir.parent,
                tree=tree,
                workspace=workspace,
                runner=runner_given,
                devices=devices_given,
                python=python,
                timeout=timeout,
                locked=True,
            )
            # With no score of that deployment, this one's is reported with no verdict.
            relative_score = None if first.infrastructure else first.measured
        judge = dict(anchor=held, relative_score=relative_score, runner=runner, **bound)
        if found.kind == AGGREGATED:
            result = accuracy_checks.run_accuracy_gate(
                environment, benchmark=benchmark, workdir=workdir, **judge, **common
            )
            return replace(result, path=IN_PROCESS)
        result = endpoint.run_endpoint_accuracy(
            found, benchmark, environment, task_data=spec, workspace=workspace, **judge
        )
        return replace(result, path=THROUGH_ENDPOINT)

    def measure() -> check_result.GateResult:
        if name == "accuracy":
            return accuracy()
        if name == "boot" and served:
            result = endpoint.run_endpoint_boot(
                found,
                task_data=spec,
                workspace=workspace,
                spec=boot_checks.resolve_boot_spec(spec, found),
                devices=devices,
                **bound,
            )
            return replace(result, path=THROUGH_ENDPOINT)
        if name == "boot":
            workload = boot_checks.resolve_boot_spec(spec, found)
            result = boot_checks.run_boot_gate(
                environment,
                workdir=workdir,
                spec=workload,
                runner=runner,
                mode=mode,
                **bound,
                **common,
            )
            return replace(result, path=IN_PROCESS)
        if name == "serve":
            if found.kind != AGGREGATED:
                raise TaskSchemaError(
                    f"`{found.name}` is disaggregated: its boot serves it; the serve gate "
                    f"starts one server"
                )
            serve_spec = serve.ServeSpec.from_task(spec)
            return serve.run_serve_gate(
                environment,
                workdir=workdir,
                runner=runner,
                spec=replace(serve_spec, timeout=timeout) if timeout else serve_spec,
                **common,
            )
        tests = disagg.run_disagg_gate if found.kind != AGGREGATED else pytest_runner.run_tests
        return tests(environment, workdir=workdir, runner=runner, **bound, **common)

    if locked:
        return measure()
    # One load at a time: the run's checks and a worker's own never share the devices.
    with servers.engine_lock(workspace):
        return measure()


def run_gate(args: argparse.Namespace) -> check_result.GateResult:
    """The ``gate`` command: :func:`run_check` with what the command line names."""
    task = Path(args.task)
    notices: list[str] = []
    try:
        task_data = load_and_validate_task_yaml(task, notices)
    finally:
        for notice in notices:
            print(f"note: {notice}", file=sys.stderr)
    return run_check(
        task_data,
        args.name,
        args.deployment,
        workdir=args.workdir,
        tree=args.tree,
        # The run's own task.yaml sits in its workspace, beside its anchors and its state.
        workspace=task.resolve().parent,
        devices=tuple(int(d) for d in args.devices.split(",")) if args.devices else None,
        python=args.python,
        timeout=args.timeout,
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


def invocation(task: str | Path, workdir: str | Path, deployment: str, cli: str = CLI_NAME) -> str:
    """The command a worker runs one of a deployment's checks with, on its own worktree."""
    return (
        f"{cli} gate <{'|'.join(GATE_NAMES)}> --task {task} --deployment {deployment} "
        f"--workdir {workdir} --tree <your worktree>"
    )


def main(argv: list[str] | None = None) -> int:
    """Run one gate; print its verdict; return 0 on a pass, 1 on a failure, 2 on an error."""
    args = _parse(list(sys.argv[1:] if argv is None else argv))
    try:
        result = run_gate(args)
    except (TaskSchemaError, KeyError, FileNotFoundError, anchors.AnchorError) as exc:
        # A KeyError's text is its argument's repr: say the argument.
        print(
            f"error: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(asdict(result), indent=1, default=str))
    else:
        print(result.summary())
        if not result.passed and result.stdout_tail.strip():
            print(result.stdout_tail)
    return 0 if result.passed else 1
