# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``trtllm-modeling-v2 gate <test>``: run one check of the run exactly as the orchestrator does.

``<test>`` is a test of the run (``spec/verification.py``: ``e2e``, ``boot``,
``serve``, ``serve_one_rank``, ``disagg`` or one of the spec's own) of the
feature ``--unit``, at ``--member``, with the environment, runner and flags
resolved from the same task.yaml. Like the orchestrator's checks, it measures
a snapshot of one commit of ``trtllm_repo_path`` (``run/workspace/snapshots.py``): the
commit the checkout is at, or ``--commit``. ``--tree`` measures another tree
as it is::

    trtllm-modeling-v2 gate e2e --task task.yaml \
        --unit dspark --member tep4 --tree .dev-worktrees/pre-fix

A test of datasets runs its case of the target's generated e2e test
(``checks/e2e.py``), which the tree measured must hold. Each benchmark is judged
against its references row; ``--reference <dataset>=<score>`` judges it against a
score instead, and a configuration variant whose anchor is relative is judged
against its unit's score, measured here first. ``--datasets generate`` runs only
the datasets named (the boot prompts, in a few minutes): a debugging run's cut of
the check, never its verdict.

Exit status: 0 when the check passed, 1 when it failed, 2 on a usage or spec
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
from typing import Any, Iterator, Mapping

from ..infra.compute.environment import resolve_environment, scheduler_owns_devices
from ..infra.compute.executor import FEATURE, select_command_executor, shell_runner_for
from ..infra.git import resolve_commit
from ..spec import variants as variants_mod
from ..spec import verification as verification_mod
from ..spec.task_schema import (
    TaskSchemaError,
    assembly_units,
    load_and_validate_task_yaml,
    parse_parallel_segment,
    widest_world_size,
)
from . import boot as boot_checks
from . import check_config, check_runtime, e2e, user_tests
from . import verdict as check_result


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 gate",
        description="Run one check of a trtllm_modeling_v2 run standalone, as the orchestrator "
        "runs it.",
    )
    parser.add_argument(
        "name",
        help="the test to run: e2e, boot, serve, serve_one_rank, disagg, or one of the spec's",
    )
    parser.add_argument("--task", type=Path, required=True, help="the run's task.yaml")
    parser.add_argument(
        "--unit",
        default="core",
        help="the feature (a unit or a variant) whose test to run (default core)",
    )
    parser.add_argument(
        "--member",
        default=None,
        help="the member to run at (default: the test's first)",
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
        help="where the check writes the files its job reads; on the shared filesystem when the "
        "job runs elsewhere (a spool session). Default ./trtllm-modeling-v2-gates",
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
        "--reference",
        action="append",
        default=[],
        metavar="DATASET=SCORE",
        help="judge a benchmark of a test of datasets against this score instead of its row",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="a test of datasets: run only these of its case, comma-separated (generate, "
        "gsm8k, ..., and its extra checks by function name), to debug what one of them shows",
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


def _references(values: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        name, sep, score = value.partition("=")
        try:
            out[name.strip()] = float(score)
        except ValueError:
            raise TaskSchemaError(f"--reference {value!r}: say <dataset>=<score>") from None
        if not sep or not name.strip():
            raise TaskSchemaError(f"--reference {value!r}: say <dataset>=<score>")
    return out


def _test(task_data: Mapping[str, Any], unit: str, name: str, world: int) -> verification_mod.Test:
    from ..run.validation import contract as contract_mod

    tests = contract_mod.run_tests(task_data, world)
    test = next((t for t in tests if t.feature == unit and t.name == name), None)
    if test is None:
        names = ", ".join(t.name for t in tests if t.feature == unit) or "none"
        raise TaskSchemaError(f"{unit} has no test {name!r} in this run (its tests: {names})")
    return test


def target_identity(task_data: Mapping[str, Any]) -> tuple[str, str]:
    """The run's ``(family, target_rel)``, as preflight resolves them."""
    from ..run.startup.preflight import PreflightError, resolve_target_identity

    config_path = Path(str(task_data["checkpoint_path"])) / "config.json"
    try:
        checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        checkpoint_config = {}
    try:
        return resolve_target_identity(task_data, checkpoint_config)
    except PreflightError as exc:
        raise TaskSchemaError(str(exc)) from exc


def run_gate(args: argparse.Namespace) -> check_result.GateResult:
    """Resolve everything from the task and run the named test's check on the tree it measures."""
    from . import disagg

    task_data = load_and_validate_task_yaml(args.task)
    declared = task_data
    unit = args.unit
    world = args.world_size or widest_world_size(task_data, 1)
    test = _test(task_data, unit, args.name, world)
    if args.timeout:
        test = replace(test, timeout=args.timeout)
    from ..run.validation import contract as contract_mod

    members = contract_mod.test_members(task_data, test, world)
    member = args.member or members[0]
    if variants_mod.is_variant(task_data, unit):
        task_data = check_config.variant_spec(
            task_data,
            unit,
            world_size=world,
            options_dir=args.workdir.resolve() / variants_mod.OPTIONS_DIRNAME,
        )
    size = parse_parallel_segment(member)[1] or 1
    spec = disagg.DisaggSpec.from_test(test, task_data)
    if spec is not None:
        gpus = spec.device_count(member) or size
    elif test.runner == verification_mod.RUN_PYTEST and not test.whole_model:
        gpus = test.devices or size
    else:
        gpus = size
    if args.devices:
        devices = tuple(int(d) for d in args.devices.split(","))
    elif scheduler_owns_devices(str((task_data.get("environment") or {}).get("kind", "local"))):
        # The scheduler exports CUDA_VISIBLE_DEVICES per job; naming devices here would
        # mask the ones it assigned and stack concurrent jobs on the same cards.
        devices = ()
    else:
        devices = tuple(range(gpus))
    environment = resolve_environment(task_data, devices=devices).at_width(size)
    nodes, lead = disagg.job_span(task_data, test, environment.nodes)
    if args.python:
        environment = replace(environment, python=args.python)
    features = (
        [f for f in args.features.split(",") if f]
        if args.features is not None
        else features_for(task_data, unit)
    )
    family, target_rel = target_identity(declared)
    with _snapshot(args, task_data) as pin:
        tree = pin.tree if pin is not None else args.tree
        if pin is not None:
            tracked = pin.snapshots.tracked(pin.commit)
            task_data, declared = pin.spec(task_data, tracked), pin.spec(declared, tracked)
        elif tree is not None:
            task_data = {**task_data, "trtllm_repo_path": str(tree.resolve())}
            declared = {**declared, "trtllm_repo_path": str(tree.resolve())}
        # A worker's own gate run queues with the run's other checks.
        executor = select_command_executor(
            environment, name_prefix=f"gate-{args.name}", gpus=str(gpus), tree=tree,
            priority=FEATURE, nodes=nodes, lead=lead,
        )  # fmt: skip
        runner = shell_runner_for(executor)
        runner = pin.runner(runner) if pin is not None else check_runtime.in_tree(runner, tree)
        run = Run(
            args=args, declared=declared, environment=environment, runner=runner,
            family=family, target_rel=target_rel, world=world,
        )  # fmt: skip
        result = run.check(test, task_data, member, features)
        return pin.stamp(result) if pin is not None else result


class Run:
    """What one standalone check runs with, resolved once."""

    def __init__(self, *, args, declared, environment, runner, family, target_rel, world) -> None:
        self.args = args
        self.declared = declared
        self.environment = environment
        self.runner = runner
        self.family = family
        self.target_rel = target_rel
        self.world = world

    def check(
        self,
        test: verification_mod.Test,
        task_data: Mapping[str, Any],
        member: str,
        features: list[str],
    ) -> check_result.GateResult:
        """Run ``test`` at ``member`` under ``task_data``; its verdict, named by the test."""
        from . import disagg, serve

        workdir = self.args.workdir.resolve() / test.feature
        unit = test.feature
        if test.runner == verification_mod.RUN_DATASETS:
            result = self._datasets(test, task_data, member, workdir)
        elif test.runner == verification_mod.RUN_SERVE:
            spec = serve.ServeSpec.from_test(test)
            result = serve.run_serve_gate(
                self.environment, task_data=task_data, runner=self.runner, unit=unit,
                member=member, features=features, workdir=workdir,
                boot=boot_checks.resolve_boot_spec(task_data, unit, features),
                spec=replace(spec, timeout=self.args.timeout) if self.args.timeout else spec,
            )  # fmt: skip
        elif test.runner == verification_mod.RUN_DISAGG:
            spec = disagg.DisaggSpec.from_test(test, task_data)
            if spec is not None and self.args.timeout:
                spec = replace(spec, timeout=self.args.timeout)
            result = disagg.run_disagg_gate(
                self.environment, workdir=workdir, spec=spec, test=test, task_data=task_data,
                runner=self.runner, unit=unit, member=member,
            )  # fmt: skip
        else:
            result = user_tests.run_user_test(
                self.environment, task_data=task_data, test=test, workdir=workdir,
                runner=self.runner, member=member, features=features,
            )  # fmt: skip
        return result if result.name == test.name else replace(result, name=test.name)

    def _datasets(
        self,
        test: verification_mod.Test,
        task_data: Mapping[str, Any],
        member: str,
        workdir: Path,
    ) -> check_result.GateResult:
        unit = test.feature
        repo = Path(str(task_data["trtllm_repo_path"]))
        path, _ = e2e.generated_paths(self.family, self.target_rel)
        if not (repo / path).is_file():
            raise TaskSchemaError(
                f"{repo} holds no {path}: the run writes the target's e2e test when it starts"
            )
        references = _references(self.args.reference)
        only = [d.strip() for d in (self.args.datasets or "").split(",") if d.strip()]
        runs = [*test.datasets, *test.extra_checks]
        unknown = [d for d in only if d not in runs]
        if unknown:
            raise TaskSchemaError(
                f"--datasets {','.join(unknown)}: {unit}'s {test.name} runs {runs}"
            )
        if (
            test.benchmarks
            and (not only or test.benchmarks[0] in only)
            and test.benchmarks[0] not in references
            and variants_mod.is_variant(self.declared, unit)
            and variants_mod.anchor_state(self.declared, unit) == variants_mod.ANCHOR_RELATIVE
        ):
            base = variants_mod.base_unit(self.declared, unit)
            first = self.check(
                _test(self.declared, base, verification_mod.E2E, self.world),
                self.declared,
                member,
                features_for(self.declared, base),
            )
            if first.infrastructure or first.measured is None:
                return replace(
                    first,
                    name=test.name,
                    passed=False,
                    detail=f"{base}'s benchmark at {member} measured no score ({first.detail}), "
                    f"so {unit}, whose anchor is that score, has nothing to be judged against",
                )
            references[test.benchmarks[0]] = first.measured
        node = e2e.node_id(self.family, self.target_rel, unit, test.name, member)
        result, _ = e2e.run_e2e(
            self.environment, task_data=task_data, test=test, member=member, node=node,
            workdir=workdir, runner=self.runner, references=references, datasets=only,
        )  # fmt: skip
        return result


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
    """The command an agent uses to rerun one of the orchestrator's checks by hand."""
    return f"{cli} gate <test> --task {task} --unit {unit} --member <member> --workdir {workdir}"


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
    "Run",
    "features_for",
    "invocation",
    "main",
    "run_gate",
    "target_identity",
]
