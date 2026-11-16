# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``trtllm-modeling-v2 gate <check>``: run one check exactly as the orchestrator does.

Three checks, each by its registry id:

``deployment:<name>``
    ``gate deployment --task task.yaml --deployment NAME``: one session of the
    deployment (``checks/session.py``): its servers loaded once, every check of it a
    client, one line each, then ``trtllm-eval`` in a load of its own for a benchmark the
    session's endpoint cannot carry. A worker's own loop on a deployment is this one
    command, and so one load.
``kernel:<row>``
    ``gate kernel --task task.yaml --row ROW``: the native tests of one row of the
    survey's ``vocabulary.yaml`` (its number, or its entry), on one device unless
    ``--gpus`` says more. No checkpoint is loaded.
``cpu:<row>``
    ``gate cpu --task task.yaml --row ROW --test FILE [--test FILE ...]``: a row's own
    tests, under the run's interpreter in a job that holds no device unless ``--gpus``
    says so. A row that names no test, or a test the tree does not have, fails: nothing
    could show its outcome.

A kernel or a cpu row whose tests need whole nodes says ``--nodes N``: one job of N nodes,
whose lead runs the tests once.

Run by hand, a check gets its devices as a run does. While a run of the task file's
workspace is going (it holds the workspace's lock), the check's job queues in that run's
spool and its pool serves it. With none going, the command builds the workspace's pool
itself and asks it for one holder of the job's nodes: on Slurm no longer than the job's wall
with the pool's margins and its lead, and counted against
``environment.slurm.max_node_hours``. It keeps the pool while the check runs and ends it
when the check is done. A spec of one host that states no spool runs the check in this
process.

Like the orchestrator's checks, each measures a snapshot of one commit of
``trtllm_repo_path`` (``run/workspace/snapshots.py``): the commit the checkout is at, or
``--commit``. ``--tree`` measures another tree as it is::

    trtllm-modeling-v2 gate deployment --task task.yaml \
        --deployment pd --tree .dev-worktrees/pre-fix

A deployment ``relative_to`` another is held to that one's score on the same tree, read
from the run's ledger: the orchestrator's session of the other deployment left it there,
or a worker's own ``gate deployment --deployment <the other>`` on that tree, which records
itself in a run (``run/evidence.py``). Outside a run nothing is on record: declare the
benchmark's ``anchor``.

Every line is printed, the check's own last. Exit status:

====== ========= ==============================================================
status outcome   means
====== ========= ==============================================================
0      pass      the check passed
1      fail      it failed, or was blocked before it could run
2                a usage or spec error; nothing was measured
3      partial   nothing failed, and what one of its checks is for could not be
                 observed ("not verified"): it does not fail, and it does not
                 certify. There is nothing to repair in the tree
====== ========= ==============================================================

``--json`` prints the lines as JSON instead, with the same outcome as ``status``
(``pass``, ``fail`` or ``partial``); ``passed`` is true only for ``pass``, and
``unverified`` says what a ``partial`` could not verify.
"""

from __future__ import annotations

import _thread
import argparse
import contextlib
import copy
import json
import math
import os
import shlex
import signal
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..infra.compute import pintree, spoolctl
from ..infra.compute import pool as pool_mod
from ..infra.compute.commands import CommandExecutor, CommandResult, CommandSpec
from ..infra.compute.environment import resolve_environment
from ..infra.compute.executor import FEATURE, select_command_executor, shell_runner_for
from ..infra.files import write_text_atomic
from ..infra.git import resolve_commit
from ..operators import coverage
from ..spec import deployments as deployments_mod
from ..spec.task_schema import TaskSchemaError, load_and_validate_task_yaml
from . import accuracy as accuracy_mod
from . import check_runtime, native_tests, session
from . import verdict as check_result

#: The checks this command runs, by the name it takes.
GATE_NAMES: tuple[str, ...] = ("deployment", "kernel", "cpu")
#: The workspace's record of what each check of the run measured (``run/validation/contract.py``).
STATUS_FILENAME = "contract_status.jsonl"


def registry_id(name: str, *, deployment: str = "", row: str = "") -> str:
    """``deployment:<name>``, ``kernel:<row>`` or ``cpu:<row>``: how a card names a check."""
    return f"{name}:{deployment if name == 'deployment' else row}"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 gate",
        description="Run one trtllm_modeling_v2 check standalone, as the orchestrator runs it.",
    )
    parser.add_argument("name", choices=GATE_NAMES, help="the check to run")
    parser.add_argument("--task", type=Path, required=True, help="the run's task.yaml")
    parser.add_argument(
        "--deployment",
        default=None,
        help="deployment: the deployment (default: the spec's first)",
    )
    parser.add_argument(
        "--row",
        default=None,
        help="kernel: a row of vocabulary.yaml, by its number or its entry "
        "(`norm/rmsnorm`, `ops/attn/qk_norm`)",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="kernel: the survey's vocabulary.yaml (default: beside --task)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="kernel: the devices its tests need (default 1); cpu: the same (default none)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="kernel, cpu: the whole nodes its tests need (default: one job on one node)",
    )
    parser.add_argument(
        "--test",
        action="append",
        default=[],
        metavar="FILE",
        help="cpu: a test file of the row, relative to the tree (repeat for more)",
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
        help="measure a pin of this commit of task.yaml's trtllm_repo_path (default: the "
        "commit the checkout is at)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("trtllm-modeling-v2-gates"),
        help="where the check writes the files its job reads; on the shared filesystem when the "
        "job runs elsewhere (a spool session). Default ./trtllm-modeling-v2-gates",
    )
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    parser.add_argument("--json", action="store_true", help="print the lines as JSON")
    args = parser.parse_args(argv)
    if args.name in ("kernel", "cpu") and not args.row:
        parser.error(f"gate {args.name} needs --row")
    return args


@contextlib.contextmanager
def _snapshot(args: argparse.Namespace, task_data: Mapping[str, Any]) -> Iterator[Any]:
    """The pinned tree this gate measures (a ``run.workspace.snapshots.CheckTree``), held while it runs.

    A commit of the checkout, or ``--tree`` pinned as it is now when it is a tree of that
    checkout. ``None`` when the gate measures a directory as it is: a ``--tree`` nothing can
    pin, or a checkout that is not a git one.
    """
    from ..infra.compute import pintree
    from ..run.workspace import integration as trees_mod
    from ..run.workspace import snapshots as snapshots_mod

    repo = Path(str(task_data.get("trtllm_repo_path") or ""))
    if not (repo / ".git").exists():
        if args.commit:
            raise TaskSchemaError(f"--commit {args.commit}: {repo} is not a git checkout")
        yield None
        return
    if args.tree is not None:
        try:
            commit = pintree.freeze(args.tree, f"gate-{args.name}")
        except pintree.PinError:
            commit = None
        if commit is None or resolve_commit(repo, commit) is None:
            yield None  # not a tree of this checkout: measured as it is
            return
    else:
        commit = resolve_commit(repo, args.commit or "HEAD")
        if commit is None:
            raise TaskSchemaError(f"{args.commit or 'HEAD'} names no commit of {repo}")
    trees = trees_mod.make(
        task_data,
        repo=repo,
        workspace=args.task.resolve().parent,
        prefix="mv2/gate",
    )
    with trees.at(trees.pinned(commit)) as tree:
        yield snapshots_mod.CheckTree(commit=commit, tree=tree, snapshots=trees)


def relative_scores(
    status: Path, deployment: deployments_mod.Deployment, tree_state: str, found: Any
) -> dict[str, float]:
    """The scores ``deployment``'s relative benchmarks are held to, read from the run's ledger.

    For each benchmark, the score the deployment named by ``relative_to`` last passed
    with on the tree this check measures (``tree_state``): a score of another tree is not
    this one's, and a later row of the same tree that did not pass takes it away. Empty
    where the ledger holds none; the benchmark is then blocked.
    """
    other = next((d for d in found if d.name == deployment.relative_to), None)
    if other is None or not status.is_file():
        return {}
    wanted = {f"accuracy[{benchmark.name}]": benchmark.name for benchmark in deployment.benchmarks}
    scores: dict[str, float] = {}
    for line in status.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        name = wanted.get(row.get("gate")) if isinstance(row, dict) else None
        if (
            name is None
            or row.get("unit") != other.name
            or row.get("member") != other.shape
            or row.get("tree_state") != tree_state
        ):
            continue
        if row.get("passed") and row.get("measured") is not None:
            scores[name] = float(row["measured"])
        else:
            scores.pop(name, None)
    return scores


def kernel_tests(
    task_data: Mapping[str, Any], row: str, vocabulary: Path
) -> tuple[coverage.QueueItem, list[str]]:
    """The row of ``vocabulary`` that ``row`` names, and its test files in the tree.

    ``row`` is the row's number or its entry. Raises ``KeyError`` naming the rows there
    are when it names none.
    """
    from ..run.startup.preflight import check_checkpoint, resolve_target_identity

    items = coverage.load_vocabulary(vocabulary)
    wanted = row.strip().strip("/")
    item = next((i for i in items if str(i.n) == wanted or i.target.strip("/") == wanted), None)
    if item is None:
        rows = ", ".join(f"{i.n} ({i.target})" for i in items)
        raise KeyError(f"{row!r} names no row of {vocabulary} ({rows or 'it has none'})")
    config = check_checkpoint(task_data["checkpoint_path"])
    _, target_rel = resolve_target_identity(task_data, config)
    paths = coverage.entry_paths(item, target_rel, task_data.get("trtllm_repo_path")) or ()
    # The module first, then its test files; a rank body beside a test is not collected.
    tests = [p for p in paths[1:] if Path(p).name.startswith("test_")]
    return item, tests


#: The lock a run's controller holds on its workspace (``cli.py``): while it is held, that
#: run keeps the workspace's pool.
LOCK_FILENAME = ".orchestrator.lock"
#: The lock of the checks run by hand on one workspace: they keep its pool one at a time.
TURN_FILENAME = ".check-by-hand.lock"


def one_job_spec(task_data: Mapping[str, Any], wall: float) -> dict[str, Any]:
    """The spec a check run by hand builds its pool from: one holder, no longer than it needs.

    On Slurm the holder is asked for ``wall`` (the walls of the check's jobs, one after the
    other) and the pool's lead, where the spec's ``time`` is longer: nothing is held past
    that, whatever becomes of this process, and that is what counts against
    ``max_node_hours``. A lead shorter than what a holder's start and the fit of a job take
    (the pool's own two margins) gives way to them: a shorter holder could not start the job.
    """
    from ..infra.compute import slurm as slurm_mod
    from ..infra.compute.scheduler import FIT_MARGIN_S

    spec = copy.deepcopy(dict(task_data))
    cfg = (spec.get("environment") or {}).get("slurm")
    if not isinstance(cfg, dict):
        return spec
    try:
        stated = slurm_mod.slurm_time_seconds(cfg["time"])
        lead = max(slurm_mod.lead_seconds(cfg), FIT_MARGIN_S + pool_mod.START_SLACK_S)
        need = wall + lead
    except (KeyError, ValueError):
        return spec  # the pool's own build says what the block lacks
    if need < stated:
        minutes = math.ceil(need / 60)
        cfg["time"] = f"{minutes // 60:02d}:{minutes % 60:02d}:00"
    return spec


class ByHand(CommandExecutor):
    """The spool's executor of a check run by hand, with the pool its jobs need.

    A job queued in a spool runs only where a holder's watcher takes it. A run's
    controller keeps its workspace's pool, and a check run while one is going is served by
    it. With none going (the workspace's lock is free) the first job queued here starts
    the pool as a run does (``infra/compute/pool.py``), from :func:`one_job_spec`, and
    keeps it until :meth:`close`, which ends what this command asked for. The spool and
    its records are the workspace's, so the bound of node-hours counts a check by hand
    with the run's own holders. Holders that were there before this command stay. Checks
    by hand on one workspace take turns: each holder is as long as one check's own jobs.
    """

    def __init__(
        self,
        inner: CommandExecutor,
        task_data: Mapping[str, Any],
        workspace: Path,
        later: Sequence[float],
    ) -> None:
        self.inner, self.task_data, self.workspace = inner, task_data, Path(workspace)
        #: The walls of the jobs this command queues after its first one, together.
        self.later = tuple(float(wall) for wall in later)
        self.lock: Any = None
        self.pool: Any = None
        self.keeper: Any = None
        #: Holders were open before this command: they are not its to end.
        self.found = False
        #: Why the pool can no longer serve this command, once it cannot.
        self.stopped = ""
        self.clean = False
        self.signal: Any = None

    @staticmethod
    def _say(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    def _start(self, wall: float) -> None:
        import fcntl

        turn = (self.workspace / TURN_FILENAME).open("a+")
        try:
            fcntl.flock(turn, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._say("pool: another check by hand keeps this workspace's pool; waiting for it")
            fcntl.flock(turn, fcntl.LOCK_EX)
        handle = (self.workspace / LOCK_FILENAME).open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            turn.close()
            self.lock = False
            self._say("pool: a run of this workspace is going; its pool serves this check")
            return
        self.lock = (handle, turn)
        whole = wall + sum(self.later)
        pool = pool_mod.build(
            one_job_spec(self.task_data, whole), workspace=self.workspace, note=self._say
        )
        if not pool.env.renews:
            self.close()  # watchers somebody else keeps serve the spool: nothing to keep
            return
        nodes = int(getattr(self.inner, "nodes", 0) or 1)
        seen = pool.observe()
        if seen is None:
            # Not knowing is not "none": a check that took a standing holder for its own
            # cancelled it when it was done.
            raise pool_mod.PoolError(
                "the scheduler did not say which holders this workspace has, so this check "
                "cannot tell a holder of its own from one that was there before it. Nothing "
                "was asked for and nothing was ended: run the check again"
            )
        self.found = any(holder.open for holder in seen)
        lasts = getattr(pool.env, "time_s", None)
        longest = max([wall, *self.later])
        if lasts and not self.found and longest > pool_mod.fit_wall_s(lasts):
            # Said before anything is asked for: the pool would hold a node for nothing.
            raise pool_mod.PoolError(
                f"a job of this check states a wall of {pool_mod.span(longest)}, and a holder "
                f"of this spec lasts {pool_mod.span(lasts)}: a job starts only where its wall "
                f"and {pool_mod.span(lasts - pool_mod.fit_wall_s(lasts))} end before the holder "
                f"does, so it could never start. No holder was asked for. Lengthen "
                f"`environment.slurm.time`, or shorten the wall (the deployment's `timeout`; "
                f"`accuracy.timeout` for a benchmark left to `trtllm-eval`)"
            )
        self._say(
            "pool: no run keeps this workspace's pool; "
            + (
                "holders that were there before this check serve it and stay "
                f"(`trtllm-modeling-v2 pool stop --workspace {self.workspace}` ends them)"
                if self.found
                else f"asking for a holder of {nodes} node(s)"
                + (f" for {pool_mod.span(lasts)}" if lasts else "")
                + f" (its jobs state {pool_mod.span(whole)}), given back when this check is done"
            )
        )
        self.pool = pool
        if threading.current_thread() is threading.main_thread():
            # A check that is told to stop still gives its holder back.
            self.signal = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        pool.start({nodes: wall})
        if pool.exhausted:
            raise pool_mod.PoolError(pool.exhausted)

        def cannot_serve(why: str) -> bool:
            # No holder is left and none may be asked for (the bound), or nothing more fits
            # this one host: the wait for the job would never end.
            self.stopped = why
            _thread.interrupt_main()
            return True

        def ready() -> dict[int, float]:
            # A job that runs has its holder: only what still waits asks for one.
            demand: dict[int, float] = {}
            for _, wide, limit in pool_mod.gpu_jobs(pool.paths.pending):
                demand[wide] = max(demand.get(wide, 0.0), limit)
            return demand

        self.keeper = pool_mod.Keeper(
            pool, ready, on_exhausted=cannot_serve, on_late=cannot_serve
        ).start()

    def execute(self, spec: CommandSpec) -> CommandResult:
        if self.lock is None:
            self._start(float(spec.timeout_s or 0.0))
        self.clean = False
        try:
            result = self.inner.execute(spec)
        except KeyboardInterrupt:
            if self.stopped:
                raise pool_mod.PoolError(self.stopped) from None
            raise
        self.clean = True
        return result

    def close(self) -> None:
        """End what this command started: its jobs if it was cut short, then its holders."""
        pool, self.pool = self.pool, None
        if pool is not None:
            mine = f"-{os.getpid()}-"
            for where in () if self.clean else (pool.paths.pending, pool.paths.running):
                for script in where.glob(f"*{mine}*.sh"):
                    spoolctl.cancel_job(pool.paths.root, script.stem)
            if self.keeper is not None:
                self.keeper.stop()
            if self.signal is not None:
                signal.signal(signal.SIGTERM, self.signal)
            if not self.found:
                local = pool.env.kind == pool_mod.KIND_LOCAL
                report = pool.end(wait_s=5.0 if local else pool_mod.END_WAIT_S)
                for record in pool.records() if local else ():
                    if not record.get("closed"):
                        pool.env.end(record)
                for rest in () if local else (*report.get("left", ()), *report.get("unknown", ())):
                    self._say(
                        f"pool: holder {rest['name']} (job {rest['job_id'] or '?'}) is "
                        f"{rest['state']} and may still be allocated: `trtllm-modeling-v2 pool "
                        f"status --workspace {self.workspace}`"
                    )
        for handle in self.lock or ():
            handle.close()
        self.lock = False


def run_gate(args: argparse.Namespace) -> tuple[list[check_result.GateResult], str]:
    """Resolve everything from the task and run the named check on the tree it measures.

    Returns its lines, the check's own last, and its registry id.
    """
    task_data = load_and_validate_task_yaml(args.task)
    found = deployments_mod.resolve(task_data)
    deployment = deployments_mod.named(task_data, args.deployment)
    workspace = args.task.resolve().parent
    if args.name == "deployment":
        check_id = registry_id("deployment", deployment=deployment.name)
        gpus = deployment.ranks
    elif args.name == "cpu":
        check_id, gpus = registry_id("cpu", row=args.row), int(args.gpus or 0)
    else:
        check_id, gpus = registry_id("kernel", row=args.row), int(args.gpus or 1)
    block = task_data.get("environment")
    if isinstance(block, dict) and (
        block.get("spool") or pool_mod.environment_kind(block) != pool_mod.KIND_LOCAL
    ):
        # As a run does: the spool is the spec's own, else the workspace's. One host that
        # names no spool runs the check in this process, on its own devices.
        pool_mod.resolve_spool(task_data, workspace)
    stated = task_data
    if (task_data.get("environment") or {}).get("spool"):
        # The spool's watcher exports CUDA_VISIBLE_DEVICES per job; naming devices here would
        # mask the ones it assigned and stack concurrent jobs on the same cards.
        devices = ()
    else:
        devices = tuple(range(gpus))
    environment = resolve_environment(task_data, devices=devices, world_size=max(gpus, 1))
    workdir = args.workdir.resolve()
    with _snapshot(args, task_data) as pin, contextlib.ExitStack() as closing:
        tree = pin.tree if pin is not None else args.tree
        if pin is not None:
            task_data = pin.spec(task_data, pin.snapshots.tracked(pin.commit))
        elif tree is not None:
            task_data = {**task_data, "trtllm_repo_path": str(tree.resolve())}
        # A worker's own check queues with the run's other checks, as one job with a
        # stated device count and wall time.
        nodes = 1
        if args.name == "deployment":
            try:
                # Wider than one node: a job of whole nodes, whose script the lead runs once.
                nodes = session.place(task_data, deployment).nodes
            except session.WiderThanOneNode:
                pass  # the session says why, before any load
        elif args.name in ("kernel", "cpu"):
            # A row states the nodes its tests need: the job has them, as the pool was asked.
            nodes = max(1, int(args.nodes or 1))
        # The job names the pin it measures, when it measures one.
        pin_file = pintree.pin_file_of(tree) if pin is not None and tree is not None else None
        executor = select_command_executor(
            environment, name_prefix=f"gate-{args.name}", gpus=str(gpus), tree=tree, pin=pin_file,
            priority=FEATURE, nodes=nodes if nodes > 1 else 0, once=nodes > 1,
            # A session's job states the ranks of each role instance, for the origin judge.
            ranks=session.stated_ranks(deployment) if args.name == "deployment" else "",
        )  # fmt: skip
        held = None
        if environment.spool:
            later: tuple[float, ...] = ()
            if args.name == "deployment":
                # A benchmark left for `trtllm-eval` is a job of its own, after the session.
                each = (task_data.get("accuracy") or {}).get("timeout")
                later = (float(each or accuracy_mod.ACCURACY_TIMEOUT_S),) * sum(
                    not session._through_endpoint(deployment, b) for b in deployment.benchmarks
                )
            held = ByHand(executor, stated, workspace, later)
        # The pinned runner is told the spool's own executor: it is what carries the pin.
        runner = shell_runner_for(held or executor)
        if pin is not None:
            runner = pin.runner(runner, executor)
        else:
            runner = check_runtime.in_tree(runner, tree)
        if held is not None:
            closing.callback(held.close)
        # The tree as its lines will name it: the pin's own description when it measures one.
        state = pin.description if pin is not None else check_result.inspect_tree(task_data)
        if args.name == "deployment":
            ran = session.run_deployment(
                deployment,
                environment,
                task_data=task_data,
                workdir=workdir,
                runner=runner,
                workspace=workspace if (workspace / "anchors.yaml").is_file() else None,
                relative_scores=relative_scores(
                    workspace / STATUS_FILENAME,
                    deployment,
                    state,
                    found,
                ),
                timeout=args.timeout,
            )
            lines = [*ran.lines, ran.result]
        elif args.name == "cpu":
            tests = [str(t) for t in args.test]
            missing = [t for t in tests if not (Path(task_data["trtllm_repo_path"]) / t).is_file()]
            if missing or not tests:
                lines = [
                    check_result.verdict_factory(check_result.inspect_tree(task_data))(
                        name=check_id,
                        passed=False,
                        detail=f"row {args.row} has no test in the tree it measures: "
                        f"{', '.join(missing) or 'none is named for it'}; nothing could show "
                        f"its outcome",
                    )
                ]
            else:
                lines = [
                    native_tests.run_native_tests(
                        environment,
                        task_data=task_data,
                        tests=tests,
                        workdir=workdir / "cpu",
                        runner=runner,
                        name=check_id,
                        timeout=args.timeout,
                    )
                ]
        else:
            vocabulary = args.vocabulary or workspace / coverage.VOCABULARY_FILENAME
            item, tests = kernel_tests(task_data, args.row, vocabulary)
            missing = [t for t in tests if not (Path(task_data["trtllm_repo_path"]) / t).is_file()]
            if missing or not tests:
                lines = [
                    check_result.verdict_factory(check_result.inspect_tree(task_data))(
                        name=check_id,
                        passed=False,
                        detail=f"row {item.n} ({item.target}) has no test in the tree it "
                        f"measures: {', '.join(missing) or 'none is named for it'}",
                    )
                ]
            else:
                lines = [
                    native_tests.run_native_tests(
                        environment,
                        task_data=task_data,
                        tests=tests,
                        workdir=workdir / "kernel",
                        runner=runner,
                        name=check_id,
                        timeout=args.timeout,
                    )
                ]
        return [pin.stamp(line) if pin is not None else line for line in lines], check_id


#: The workflow's CLI as the run's agents call it: ``<workspace>/bin/<this>``.
CLI_NAME = "trtllm-modeling-v2"


def install_cli(workspace: str | Path) -> Path:
    """``<workspace>/bin/trtllm-modeling-v2``: this interpreter and package, callable by path.

    ``PYTHONSAFEPATH`` keeps the working directory off ``sys.path``, so a shell
    inside another agent-flow checkout still imports this package. Every start writes it
    again, as a new file renamed over the old one: a process may be running the old one,
    and a file that is being executed cannot be opened for writing.
    """
    import agent_flow

    root = Path(agent_flow.__file__).resolve().parents[1]
    script = Path(workspace) / "bin" / CLI_NAME
    script.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        script,
        "#!/bin/bash\n"
        "# The trtllm-modeling-v2 CLI under the orchestrator's interpreter and package.\n"
        f'export PYTHONPATH={shlex.quote(str(root))}"${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        "export PYTHONSAFEPATH=1\n"
        f"exec {shlex.quote(sys.executable)} -m agent_flow.workflows.trtllm_modeling_v2.cli "
        '"$@"\n',
        mode=0o755,
    )
    script.chmod(0o755)
    return script


#: The outcome of a check, and the exit status that says it (2 is a usage or spec error).
PASS, FAIL, PARTIAL = "pass", "fail", "partial"
EXIT_STATUS = {PASS: 0, FAIL: 1, PARTIAL: 3}


def outcome(own: check_result.GateResult) -> str:
    """``pass``, ``fail`` or ``partial``: a check that failed nothing and could not verify something."""
    if not own.passed:
        return FAIL
    return PARTIAL if own.unverified else PASS


def main(argv: list[str] | None = None) -> int:
    """Run one check; print its lines; return its outcome's status (:data:`EXIT_STATUS`), 2 on an error.

    Only the verdict is printed on standard output; everything else goes to standard error.
    """
    from ..infra.compute.pintree import PinError
    from ..run.startup.preflight import PreflightError

    args = _parse(list(sys.argv[1:] if argv is None else argv))
    try:
        # Standard output is the verdict's alone (`--json > file` must be JSON): whatever
        # is said while the check runs (the pool's notices, a wait, a watcher's event) goes
        # to standard error.
        with contextlib.redirect_stdout(sys.stderr):
            lines, check_id = run_gate(args)
    except (
        TaskSchemaError,
        KeyError,
        FileNotFoundError,
        PinError,
        PreflightError,
        coverage.VocabularyError,
        pool_mod.PoolError,
    ) as exc:
        print(
            f"error: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}",
            file=sys.stderr,
        )
        return 2
    own = lines[-1]
    status = outcome(own)
    if args.json:
        payload = {
            "id": check_id,
            "status": status,
            # Whether the check certifies: a partial outcome did not fail and does not.
            "passed": status == PASS,
            "unverified": own.unverified if status == PARTIAL else "",
            "lines": [asdict(line) for line in lines],
        }
        print(json.dumps(payload, indent=1, default=str))
    else:
        print(f"check {check_id}")
        for line in lines:
            print(line.summary())
        if not own.passed and own.stdout_tail.strip():
            print(own.stdout_tail)
        if status == PARTIAL:
            print(
                f"partial (exit {EXIT_STATUS[PARTIAL]}): nothing failed, and this does not "
                f"certify: {own.unverified}"
            )
    return EXIT_STATUS[status]


__all__ = [
    "CLI_NAME",
    "EXIT_STATUS",
    "GATE_NAMES",
    "install_cli",
    "kernel_tests",
    "main",
    "outcome",
    "registry_id",
    "relative_scores",
    "run_gate",
]
