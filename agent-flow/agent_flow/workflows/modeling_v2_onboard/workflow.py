# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The planner/worker loop: the planner runs alongside the workers.

    plan   planner writes the first batch of plan.md (re-prompted until it
           validates): the tasks it can define before reading everything
    run    every task whose dependencies are DONE starts at once, up to
           max_parallel, each with its own worker in its own git worktree.
           While the planner says it is still planning, it gets its next
           turn as soon as one is applied and adds the tasks it has settled.
           Whenever workers finish, the planner is woken with their results
           and may, while the others keep running:
             - add tasks, or rewrite, rewire or remove tasks not yet started;
             - retry, rewrite or remove FAILED tasks;
             - cancel RUNNING tasks, which abandons them;
             - mark ``perf`` tasks ``side`` (not merged) or merge them again;
             - FINISH: let the final task start once its dependencies are DONE;
             - ABORT the run.
    wrap   the final task is DONE: side tasks still running get up to
           ``side_wait_minutes`` to finish, side tasks not started never start,
           and the planner may still chain a new landing task after the final one
    done   nothing is left to run, or the planner aborted

Cancelling a task abandons it: its worker's session is stopped and nothing
else is touched. Its worktree, branch, notes and GPU jobs stay as they are
(the jobs run to their end), it leaves the plan, and its id is never used
again; :data:`abandoned.md <Layout.abandoned_path>` lists what it left for a
human to clean up. A retried or rewritten ``FAILED`` task carries on from its
own branch; redoing work from scratch is a task with a new id.

A run that lands keeps every side task's branch as
``<landed branch>-unmerged/<id>`` and commits ``UNMERGED_OPTIMIZATIONS.md``
listing them, so a human can merge any of them later.

With ``primer`` on, an op primer session reads what every ``new_op`` and
``perf`` worker needs while the planner plans; those workers then start as
forks of it (waiting up to ``PRIMER_WAIT_MINUTES`` for it), and start afresh
from its written summary when it is not there.

When ``task.yaml`` asks for a GPU pool, its holder jobs are submitted before
the planner starts, kept submitted while the run goes on, and cancelled when it
ends; workers run their GPU commands in it.

A human steers a running run by leaving a message in its inbox
(``modeling-v2-onboard --workspace <ws> --feedback ...``): it wakes the planner,
which acts on it first. A message is consumed only by a turn that applied.

A ``FAILED`` task only holds back the tasks that depend on it; nothing pauses.
There is no turn limit. When nothing is running and nothing can start, the
planner gets a stalled turn; ``MAX_STALLED_TURNS`` in a row that change
nothing stop the run with an error. Task states are checkpointed in
``.state.json`` on every change and the running plan in ``.running_plan.md``,
so rerunning the same command after an interruption resumes: running tasks
restart on their branches and the planner is told.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypeVar

from agent_flow.console import print_message, print_rule
from agent_flow.git_worktree import GitWorktreeIsolation, node_id_to_slug
from agent_flow.jobs.kinds import KINDS
from agent_flow.jobs.registry import JobRegistry
from agent_flow.logger import get_logger
from agent_flow.orchestration import Node, NodeState

from .files import (
    HandoffError,
    Layout,
    PlannerDecision,
    TaskResult,
    parse_planner_decision,
    parse_task_result,
    read_handoff,
    read_inbox,
)
from .gpu_pool import GpuPool, PoolConfigError, pool_config
from .node_runner import Agent, AgentFactory, Primer, make_run_task, write_failure
from .plan import PRIORITIES, PlanError, PlanSpec, parse_plan
from .prompts import (
    PRIMED_TAGS,
    build_planner_system_prompt,
    build_primer_system_prompt,
    initial_plan_prompt,
    plan_fix_prompt,
    planner_turn_prompt,
    primer_prompt,
    primer_version,
)

__all__ = ["ModelingV2OnboardWorkflow", "RunOutcome", "WorkflowError"]

# How many times the planner is sent its validation errors within one turn.
MAX_PLAN_FIXES = 3
# How many stalled turns in a row the planner gets before the run stops.
MAX_STALLED_TURNS = 3
# How often the orchestrator looks for human messages in the inbox.
INBOX_POLL_SECONDS = 10
# How long a task that starts from the op primer waits for it to be ready.
PRIMER_WAIT_MINUTES = 20
# The file a landed run commits to list the side tasks it did not merge.
UNMERGED_FILE = "UNMERGED_OPTIMIZATIONS.md"

_STATE_VERSION = 2

T = TypeVar("T")


class WorkflowError(RuntimeError):
    """The run cannot start or continue as configured."""


class RunOutcome(str, Enum):
    """How a run ended."""

    SUCCEEDED = "succeeded"  # the final task landed
    ABORTED = "aborted"  # the planner judged the goal unreachable


@dataclass
class _State:
    phase: str = "plan"  # plan | run | done
    outcome: str | None = None
    turn: int = 0
    finish: bool = False
    # The planner is still writing the plan: it gets its next turn at once.
    planning: bool = False
    # The GPU pool's size the planner asked for; None keeps task.yaml's.
    gpu_nodes: int | None = None
    tasks: dict[str, NodeState] = field(default_factory=dict)
    # Tasks that finished since the planner's last turn: (id, status, summary).
    finished: list[tuple[str, str, str]] = field(default_factory=list)
    # Tasks cancelled, or removed after they had started: their ids are
    # never used again.
    abandoned: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        data = {"version": _STATE_VERSION, **asdict(self)}
        data["tasks"] = {tid: state.value for tid, state in self.tasks.items()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> _State:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _STATE_VERSION:
            raise WorkflowError(
                f"{path} has unsupported version {data.get('version')!r}; pass --clean to "
                "start over"
            )
        return cls(
            phase=data["phase"],
            outcome=data.get("outcome"),
            turn=data["turn"],
            finish=data["finish"],
            planning=data.get("planning", False),
            gpu_nodes=data.get("gpu_nodes"),
            tasks={tid: NodeState(s) for tid, s in data["tasks"].items()},
            finished=[tuple(item) for item in data.get("finished", [])],
            abandoned=data.get("abandoned", []),
        )


@dataclass
class _Change:
    """A validated planner turn, ready to apply."""

    decision: PlannerDecision
    plan: PlanSpec
    old_plan: PlanSpec
    # FAILED tasks run again, retried or rewritten: their last result
    # becomes an earlier attempt.
    rerun: set[str]
    # Tasks that leave the plan after they had started: cancelled, or
    # removed while FAILED.
    abandoned: set[str]
    removed: set[str]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


def _decision_problems(decision: PlannerDecision) -> list[str]:
    """What is wrong with a turn file on its own, whatever the task states."""
    if decision.planning and decision.decision == "FINISH":
        return ["`planning: true` with `FINISH`: decide FINISH once the plan is complete"]
    return []


def _session_dir(cwd: Path) -> Path:
    """Where Claude Code keeps the session files of agents run in ``cwd``."""
    home = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")).expanduser()
    return home / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def _result_summary(path: Path) -> str | None:
    try:
        return read_handoff(path, parse_task_result).summary
    except HandoffError:
        return None


class ModelingV2OnboardWorkflow:
    """Run the planner alongside parallel workers until the final task lands.

    Args:
        workspace: Where plan.md, results and all run state live. Must be
            outside ``repo_path``.
        task_path: The user's ``task.yaml``; copied into the workspace.
        task_data: Its validated contents.
        agent_factory: Builds the planner's and the workers' agents.
        max_parallel: The most workers that run at once.
        side_wait_minutes: How long side tasks still running when the final
            task is DONE may take to finish before they are stopped.
        primer: Whether ``new_op`` and ``perf`` workers start as forks of an
            op primer session; needs a backend that can fork sessions.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        task_path: Path,
        task_data: dict,
        agent_factory: AgentFactory,
        max_parallel: int = 8,
        side_wait_minutes: float = 60,
        primer: bool = False,
    ) -> None:
        self.layout = Layout(Path(workspace).absolute())
        self.task_path = Path(task_path)
        self.repo_path = Path(task_data["trtllm_repo_path"]).absolute()
        self.slurm = "slurm-environment" in task_data
        self.agent_factory = agent_factory
        self.max_parallel = max_parallel
        self.side_wait_minutes = side_wait_minutes
        self.primer = primer
        self.isolation = GitWorktreeIsolation(
            repo_path=self.repo_path, worktrees_root=self.layout.worktrees_dir
        )
        try:
            config = pool_config(task_data)
        except PoolConfigError as exc:
            raise WorkflowError(f"task.yaml: {exc}") from exc
        self.gpu_pool = None
        if config is not None:
            self.gpu_pool = GpuPool(
                directory=self.layout.gpu_pool_dir,
                config=config,
                workspace=self.layout.workspace,
                mounts=(self.layout.workspace, self.repo_path),
            )
        self._run_task = make_run_task(
            layout=self.layout,
            agent_factory=agent_factory,
            slurm=self.slurm,
            gpu_run=self.gpu_pool.command(sys.executable) if self.gpu_pool else None,
        )
        self._log = get_logger().console

    # ------------------------------------------------------------------ setup

    def _check_workspace(self) -> None:
        workspace = self.layout.workspace.resolve()
        repo = self.repo_path.resolve()
        if workspace == repo or repo in workspace.parents:
            raise WorkflowError(
                f"the workspace {workspace} is inside trtllm_repo_path {repo}; the final "
                "task commits everything untracked there, so put the workspace outside it"
            )

    def _check_checkout(self) -> None:
        """Refuse a checkout the final task's commit would sweep stray files into."""
        repo = self.repo_path
        status = _git(repo, "status", "--porcelain")
        if status.returncode != 0:
            raise WorkflowError(f"{repo} is not a git checkout: {status.stderr.strip()}")
        if status.stdout.strip():
            raise WorkflowError(
                f"{repo} has uncommitted or untracked files, which the final task would "
                "commit. Point trtllm_repo_path at a clean checkout (e.g. a dedicated "
                f"`git worktree add`):\n{status.stdout.rstrip()}"
            )

    def _init(self) -> _State:
        layout = self.layout
        layout.workspace.mkdir(parents=True, exist_ok=True)
        if layout.state_path.is_file():
            return _State.load(layout.state_path)
        if layout.plan_path.exists() or layout.running_plan_path.exists():
            raise WorkflowError(
                f"{layout.workspace} holds files from an earlier run but no .state.json; "
                "pass --clean to start over"
            )
        if self.task_path.resolve() != layout.task_path.resolve():
            shutil.copyfile(self.task_path, layout.task_path)
        for directory in (layout.results_dir, layout.turns_dir, layout.nodes_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if self.slurm:
            layout.test_commands_path.touch()
        state = _State()
        state.save(layout.state_path)
        return state

    def clean(self) -> None:
        """Delete this workflow's state, task worktrees and task branches."""
        layout = self.layout
        self._drop_task_branches()
        self._drop_sessions()
        if self.gpu_pool is not None:
            self.gpu_pool.release()
        shutil.rmtree(layout.gpu_pool_dir, ignore_errors=True)
        for path in (
            layout.state_path,
            layout.plan_path,
            layout.running_plan_path,
            layout.priorities_path,
            layout.test_commands_path,
            layout.primers_path,
            layout.abandoned_path,
        ):
            path.unlink(missing_ok=True)
        for directory in (
            layout.results_dir,
            layout.turns_dir,
            layout.nodes_dir,
            layout.worktrees_dir,
            layout.knowledge_path("op").parent,
        ):
            shutil.rmtree(directory, ignore_errors=True)

    def _drop_sessions(self) -> None:
        """Delete the session files of the run's primer and of its workers."""
        recorded = self._primers().get("op", {})
        session = recorded.get("session_id")
        for root in {self.repo_path, self.repo_path.resolve()}:
            if session:
                (_session_dir(root) / f"{session}.jsonl").unlink(missing_ok=True)
        for root in {self.layout.worktrees_dir, self.layout.worktrees_dir.resolve()}:
            prefix = _session_dir(root)
            if prefix.parent.is_dir():
                for directory in prefix.parent.glob(f"{prefix.name}-*"):
                    shutil.rmtree(directory, ignore_errors=True)

    def _primers(self) -> dict:
        try:
            return json.loads(self.layout.primers_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _drop_task_branches(self) -> None:
        repo, root = self.repo_path, self.layout.worktrees_dir.resolve()
        listing = _git(repo, "worktree", "list", "--porcelain").stdout
        for line in listing.splitlines():
            if not line.startswith("worktree "):
                continue
            path = Path(line[len("worktree ") :]).resolve()
            if root in path.parents:
                _git(repo, "worktree", "remove", "--force", str(path))
        _git(repo, "worktree", "prune")
        refs = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/node/")
        branches = refs.stdout.split()
        if branches:
            _git(repo, "branch", "-D", *branches)

    # ---------------------------------------------------------------- workers

    def _save(self) -> None:
        self._state.save(self.layout.state_path)

    def _set(self, task_id: str, state: NodeState) -> None:
        old = self._state.tasks.get(task_id)
        self._state.tasks[task_id] = state
        self._save()
        if old != state:
            before = old.value if old else "new"
            print_message(f"  task {task_id}: {before} → {state.value}", self._log)

    def _adopt(self, plan: PlanSpec) -> None:
        self._plan = plan
        self.layout.running_plan_path.write_text(plan.text, encoding="utf-8")
        priorities = self.layout.priorities_path
        tmp = priorities.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(plan.priorities), indent=1), encoding="utf-8")
        tmp.replace(priorities)

    def _startable(self, node: Node) -> bool:
        tasks = self._state.tasks
        if node.id in self._running:
            return False
        if tasks[node.id] == NodeState.RUNNING:
            # Started before the orchestrator restarted: resume it.
            return True
        if tasks[node.id] != NodeState.PENDING:
            return False
        if self._plan.is_landing(node.id) and not self._state.finish:
            return False
        if node.id in self._plan.side and self._landed():
            return False
        return all(tasks.get(dep) == NodeState.DONE for dep in node.depends_on)

    def _landed(self) -> bool:
        """Whether a landing task has committed to the checkout."""
        return any(
            self._plan.is_landing(node.id) and self._state.tasks[node.id] == NodeState.DONE
            for node in self._plan.graph.nodes
        )

    def _delivered(self) -> bool:
        """Whether the final task is DONE."""
        return self._state.tasks[self._plan.sink_id()] == NodeState.DONE

    def _ready(self) -> list[Node]:
        """Tasks that can start now but have no worker yet.

        Resumed tasks come first, then by priority class, then in plan order.
        """
        nodes = [n for n in self._plan.graph.nodes if self._startable(n)]
        return sorted(
            nodes,
            key=lambda n: (
                self._state.tasks[n.id] != NodeState.RUNNING,
                PRIORITIES.index(self._plan.priorities[n.id]),
            ),
        )

    def _dispatch(self) -> None:
        """Start ready tasks until every worker slot is taken."""
        if self._outcome is not None or self._crash is not None:
            return
        for node in self._ready():
            if len(self._running) >= self.max_parallel:
                break
            self._set(node.id, NodeState.RUNNING)
            self._running[node.id] = asyncio.create_task(self._task(node))

    async def _task(self, node: Node) -> None:
        """One task end to end; a side task stopped at the deadline ends FAILED."""
        try:
            result = await self._work(node)
        except asyncio.CancelledError:
            if node.id not in self._overdue:
                raise
            result = None
        except Exception as exc:
            # Worker failures arrive as FAILED results; an exception here is the
            # orchestrator's own git machinery failing, which stops the run.
            self._crash = exc
            self._wake.set()
            return
        if result is None:
            result = self._overdue_result(node.id)
        del self._running[node.id]
        state = NodeState.DONE if result.status == "DONE" else NodeState.FAILED
        self._set(node.id, state)
        self._state.finished.append((node.id, result.status, result.summary))
        self._save()
        self._wake.set()
        self._dispatch()

    async def _work(self, node: Node) -> TaskResult | None:
        """Worktree, merge of the dependencies, worker, commit.

        Returns ``None`` for a side task stopped at the deadline. What a
        stopped worker left in its worktree is still committed to its branch,
        so a resumed task starts from it. An abandoned task's worktree is left
        exactly as its worker left it.
        """
        plan = self._plan
        landing = plan.is_landing(node.id)
        dep_nodes = [plan.graph.by_id[dep] for dep in node.depends_on]
        path = await self._git(self.isolation.acquire, node)
        try:
            if dep_nodes:
                await self._git(self.isolation.prepare, node, dep_nodes, path)
            try:
                primer = await self._primer_for(node)
                result = await self._run_task(
                    node, path, plan.tags[node.id], landing, node.id in plan.side, primer
                )
            except asyncio.CancelledError:
                if node.id in self._abandoned:
                    raise
                if node.id not in self._overdue:
                    if node.isolation != "shared":
                        await self._git(self.isolation.commit, node, path)
                    raise
                result = None
            await self._git(self.isolation.commit, node, path)
        finally:
            if node.id not in self._abandoned:
                await self._git(self.isolation.release, node)
        return result

    async def _primer_for(self, node: Node) -> Primer | None:
        """The op primer, for a task that starts from it, once it is ready."""
        primer = self._op_primer
        if primer is None or not PRIMED_TAGS & set(self._plan.tags[node.id]):
            return None
        if not primer.done():
            try:
                await asyncio.wait_for(asyncio.shield(primer), PRIMER_WAIT_MINUTES * 60)
            except TimeoutError:
                return None
        return primer.result()

    async def _prime(self) -> Primer | None:
        """Build the op primer, or reuse the one an earlier session of the run built."""
        layout = self.layout
        version = primer_version()
        recorded = self._primers().get("op")
        if recorded and recorded.get("version") == version:
            return Primer(recorded["session_id"], self.repo_path, recorded["commit"])
        commit = _git(self.repo_path, "rev-parse", "HEAD").stdout.strip()
        summary = layout.knowledge_path("op")
        summary.parent.mkdir(parents=True, exist_ok=True)
        print_message("op primer: reading for the op workers", self._log)
        agent = self.agent_factory(
            "primer[op]", build_primer_system_prompt(), self.repo_path, "persistent"
        )
        try:
            await agent.aforward(
                primer_prompt(
                    repo=self.repo_path,
                    commit=commit,
                    task_yaml=layout.task_path,
                    test_commands=layout.test_commands_path,
                    summary=summary,
                )
            )
            session_id = getattr(agent, "session_id", None)
        except Exception as exc:
            print_message(
                f"[yellow]op primer failed ({exc!r}); op workers start afresh[/yellow]",
                self._log,
            )
            return None
        finally:
            await agent.__aexit__(None, None, None)
        if not session_id:
            print_message(
                "[yellow]op primer: the backend reported no session; op workers start "
                "afresh[/yellow]",
                self._log,
            )
            return None
        record = {"session_id": session_id, "commit": commit, "version": version}
        layout.primers_path.write_text(json.dumps({"op": record}, indent=1), encoding="utf-8")
        print_message(f"op primer: ready (session {session_id})", self._log)
        return Primer(session_id, self.repo_path, commit)

    def _overdue_result(self, task_id: str) -> TaskResult:
        """The result of a side task stopped at the deadline: its own, if it wrote one."""
        path = self.layout.result_path(task_id)
        try:
            return read_handoff(path, parse_task_result)
        except HandoffError:
            minutes = f"{self.side_wait_minutes:g}"
            return write_failure(
                path,
                f"The final task was DONE and this side task was still running "
                f"{minutes} minutes later, so the orchestrator stopped it. What it had "
                "changed is committed on its branch.",
                summary=f"stopped {minutes} minutes after the final task was DONE",
            )

    async def _stop_side_after(self, seconds: float) -> None:
        """Stop the side tasks still running ``seconds`` from now."""
        await asyncio.sleep(seconds)
        for task_id, task in list(self._running.items()):
            if task_id in self._plan.side:
                self._overdue.add(task_id)
                task.cancel()

    def _cancel_jobs(self, task_id: str) -> None:
        jobs_path = self.layout.jobs_path(task_id)
        if not jobs_path.is_file():
            return
        try:
            entries = JobRegistry(jobs_path).all()
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            # The worker wrote the file itself; its jobs are its own to stop.
            print_message(
                f"  task {task_id}: jobs.json not read ({exc}); its jobs are left", self._log
            )
            return
        for entry in entries:
            kind = KINDS.get(entry.kind)
            if kind is None:
                continue
            try:
                kind.cancel(entry.handle)
            except (OSError, KeyError, subprocess.SubprocessError):
                # A job that is already gone must not stop the run.
                continue

    async def _stop_workers(self) -> None:
        """Cancel every running worker; its state stays RUNNING for a resume."""
        tasks = list(self._running.values())
        self._running.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _git(self, operation: Callable[..., Awaitable[T]], *args: object) -> T:
        """Run one git-worktree operation alone and to completion.

        Operations on one repository run one at a time, and a cancelled task
        waits for its operation to finish: a ``git worktree add`` killed
        midway leaves a locked worktree that nothing can remove.
        """
        async with self._git_lock:
            op = asyncio.ensure_future(operation(*args))
            try:
                return await asyncio.shield(op)
            except asyncio.CancelledError:
                await asyncio.gather(op, return_exceptions=True)
                raise

    async def _discard(self, node: Node) -> None:
        await self._git(self.isolation.release, node)
        await self._git(self.isolation.reclaim, node)

    # ---------------------------------------------------------------- planner

    async def _converse(self, prompt: str, check: Callable[[], T]) -> T:
        """Send ``prompt``, then validation errors, until ``check`` passes.

        ``check`` validates the planner's files and applies them without
        awaiting, so the task states it validated against are the ones it
        changes. Raises the last :class:`PlanError` when the planner runs out
        of fixes.
        """
        await self._planner.aforward(prompt)
        for attempt in range(MAX_PLAN_FIXES + 1):
            try:
                return check()
            except PlanError as exc:
                problems = exc.problems
            except HandoffError as exc:
                problems = [str(exc)]
            if attempt == MAX_PLAN_FIXES:
                raise PlanError(problems)
            print_message(
                f"[yellow]planner output rejected: {len(problems)} problem(s); sending "
                "them back[/yellow]",
                self._log,
            )
            await self._planner.aforward(plan_fix_prompt(problems, plan=self.layout.plan_path))
        raise AssertionError("unreachable")

    def _read_plan_md(self) -> str:
        if not self.layout.plan_path.is_file():
            raise PlanError([f"{self.layout.plan_path} was not written"])
        return self.layout.plan_path.read_text(encoding="utf-8")

    async def _initial_plan(self) -> None:
        state = self._state
        turn_file = self.layout.turn_path(0)
        turn_file.unlink(missing_ok=True)

        def check() -> None:
            plan = parse_plan(self._read_plan_md())
            decision = read_handoff(turn_file, parse_planner_decision)
            problems = _decision_problems(decision)
            if decision.retry or decision.cancel:
                problems.append("`retry` and `cancel`: no task has run yet")
            if problems:
                raise PlanError(problems)
            self._adopt(plan)
            state.tasks = {node.id: NodeState.PENDING for node in plan.graph.nodes}
            state.planning = decision.planning
            state.finish = decision.decision == "FINISH"
            if decision.decision == "ABORT":
                self._outcome = RunOutcome.ABORTED
            state.phase = "run"
            self._save()

        print_rule("[bold magenta]Plan[/bold magenta]", self._log)
        await self._converse(
            initial_plan_prompt(
                task_yaml=self.layout.task_path,
                plan=self.layout.plan_path,
                repo=self.repo_path,
                turn_file=turn_file,
            ),
            check,
        )
        more = "; planning continues while they run" if state.planning else ""
        print_message(f"plan: {len(self._plan.graph.nodes)} tasks{more}", self._log)

    def _validate(self, turn_file: Path) -> _Change:
        """Check the planner's turn against the current task states."""
        decision = read_handoff(turn_file, parse_planner_decision)
        tasks = self._state.tasks
        problems = _decision_problems(decision)
        for tid in decision.cancel:
            if tasks.get(tid) != NodeState.RUNNING:
                state = tasks[tid].value if tid in tasks else "not in the plan"
                problems.append(f"`cancel`: task {tid!r} is not running ({state})")
            elif self._plan.graph.by_id[tid].isolation == "shared":
                problems.append(
                    f"`cancel`: task {tid!r} works in the checkout itself and cannot be "
                    "abandoned; let it finish, then add the follow-up work after it"
                )
        for tid in decision.retry:
            if tasks.get(tid) != NodeState.FAILED:
                state = tasks[tid].value if tid in tasks else "not in the plan"
                problems.append(f"`retry`: task {tid!r} is not FAILED ({state})")
        text = self._read_plan_md()
        plan = self._plan
        problems += self._gpu_nodes_problems(decision.gpu_nodes)
        if text != self._plan.text:
            cancelled = set(decision.cancel)
            effective = {
                tid: NodeState.PENDING if tid in cancelled else state
                for tid, state in tasks.items()
            }
            try:
                plan = parse_plan(text, previous=self._plan, states=effective)
            except PlanError as exc:
                problems += exc.problems
        elif (
            self._state.planning
            and decision.planning
            and not decision.retry
            and not decision.cancel
            and decision.gpu_nodes is None
        ):
            problems.append(
                "`planning: true` but this turn changes nothing: add the tasks you have "
                "settled, or write `planning: false` if the plan is complete"
            )
        for tid in decision.cancel:
            if tid in plan.graph.by_id:
                problems.append(
                    f"`cancel`: task {tid!r} is still in the plan. A cancelled task is "
                    "abandoned: remove it, and redo its work, if needed, as a task with a new id"
                )
        for tid in sorted(set(plan.graph.by_id) & set(self._state.abandoned)):
            problems.append(
                f"task {tid!r} was abandoned and its id cannot be used again; give the task "
                "a new id"
            )
        if problems:
            raise PlanError(problems)

        old = self._plan
        removed = set(old.graph.by_id) - set(plan.graph.by_id)
        rewritten_failed = {
            tid
            for tid in old.graph.by_id
            if tid in plan.graph.by_id
            and tasks.get(tid) == NodeState.FAILED
            and old.entry(tid) != plan.entry(tid)
        }
        abandoned = set(decision.cancel) | {
            tid for tid in removed if tasks.get(tid) == NodeState.FAILED
        }
        return _Change(
            decision=decision,
            plan=plan,
            old_plan=old,
            rerun=set(decision.retry) | rewritten_failed,
            abandoned=abandoned,
            removed=removed,
        )

    @property
    def _max_gpu_nodes(self) -> int:
        """The most nodes the planner may grow the GPU pool to."""
        assert self.gpu_pool is not None
        bound = self.gpu_pool.config.max_nodes or self.max_parallel
        return max(bound, self.gpu_pool.config.nodes)

    def _gpu_nodes_problems(self, gpu_nodes: int | None) -> list[str]:
        if gpu_nodes is None:
            return []
        if self.gpu_pool is None:
            return ["`gpu_nodes`: this run has no GPU pool (task.yaml sets no `gpu_pool`)"]
        if gpu_nodes < self.gpu_pool.nodes:
            return [
                f"`gpu_nodes`: {gpu_nodes} is fewer than the pool's {self.gpu_pool.nodes} "
                "nodes; the pool only grows"
            ]
        if gpu_nodes > self._max_gpu_nodes:
            return [
                f"`gpu_nodes`: {gpu_nodes} is more than the run's limit of {self._max_gpu_nodes}"
            ]
        return []

    def _apply(self, change: _Change) -> list[asyncio.Task[None]]:
        """Apply a validated turn to the run state; return the cancelled workers."""
        cancelled = []
        for tid in change.decision.cancel:
            self._abandoned.add(tid)
            task = self._running.pop(tid)
            task.cancel()
            cancelled.append(task)
        self._adopt(change.plan)
        tasks = self._state.tasks
        for tid in sorted(change.removed):
            tasks.pop(tid, None)
            if tid in change.abandoned:
                print_message(f"  task {tid}: abandoned", self._log)
            else:
                print_message(f"  task {tid}: removed", self._log)
        for tid in sorted(change.rerun):
            self._set(tid, NodeState.PENDING)
        if change.abandoned:
            self._abandoned |= change.abandoned
            self._state.abandoned += sorted(change.abandoned)
            self._write_abandoned()
        for node in change.plan.graph.nodes:
            if node.id not in tasks:
                self._set(node.id, NodeState.PENDING)
        self._state.planning = change.decision.planning
        gpu_nodes = change.decision.gpu_nodes
        if self.gpu_pool is not None and gpu_nodes is not None and gpu_nodes != self.gpu_pool.nodes:
            print_message(f"gpu pool: {self.gpu_pool.nodes} → {gpu_nodes} nodes", self._log)
            self.gpu_pool.nodes = self._state.gpu_nodes = gpu_nodes
        if change.decision.decision == "FINISH":
            self._state.finish = True
        if change.decision.decision == "ABORT":
            self._outcome = RunOutcome.ABORTED
        self._save()
        return cancelled

    async def _settle(self, change: _Change, cancelled: list[asyncio.Task[None]]) -> None:
        """Wait for the abandoned workers to stop; keep rerun tasks' last results.

        Nothing an abandoned task left is touched. A rerun task's last result
        becomes an earlier attempt for its next worker and the planner to read.
        """
        await asyncio.gather(*cancelled, return_exceptions=True)
        for tid in sorted(change.rerun):
            result = self.layout.result_path(tid)
            if result.is_file():
                attempt = len(self.layout.previous_attempts(tid)) + 1
                result.replace(self.layout.attempt_path(tid, attempt))

    def _write_abandoned(self) -> None:
        """List what every abandoned task left behind."""
        lines = [
            "# Abandoned tasks",
            "",
            "Each was cancelled, or removed after it had started. Nothing it left was "
            "touched: its worktree, branch, notes and GPU jobs (which ran to their end) "
            "are listed here for a human to clean up.",
            "",
        ]
        for tid in self._state.abandoned:
            worktree = self.layout.worktrees_dir / node_id_to_slug(tid)
            lines += [
                f"## `{tid}`",
                "",
                f"- branch: `{self.isolation.branch_prefix}{tid}` in `{self.repo_path}`",
                f"- worktree: `{worktree}`" + ("" if worktree.exists() else " (none)"),
                f"- notes and jobs: `{self.layout.node_dir(tid)}`",
                "",
            ]
        self.layout.abandoned_path.write_text("\n".join(lines), encoding="utf-8")

    def _describe(self, node: Node) -> str:
        tasks = self._state.tasks
        state = tasks[node.id]
        if state == NodeState.RUNNING:
            return "running" if node.id in self._running else "running — waiting for a slot"
        if state != NodeState.PENDING:
            return state.value
        waiting = [dep for dep in node.depends_on if tasks.get(dep) != NodeState.DONE]
        if waiting:
            return f"pending — waiting on {', '.join(waiting)}"
        if self._plan.is_landing(node.id) and not self._state.finish:
            return "pending — held until FINISH"
        if node.id in self._plan.side and self._landed():
            return "pending — side, will not start: the run has landed"
        return "pending — ready, waiting for a slot"

    async def _turn(self, *, stalled: bool, resumed: bool) -> None:
        """Wake the planner with what changed and apply its answer."""
        state = self._state
        finished = list(state.finished)
        feedback = read_inbox(self.layout)
        state.turn += 1
        self._save()
        turn_file = self.layout.turn_path(state.turn)
        turn_file.unlink(missing_ok=True)
        pool = None
        if self.gpu_pool is not None:
            pool = await asyncio.to_thread(self.gpu_pool.status)
            pool += f"\nThe pool may grow to {self._max_gpu_nodes} nodes (`gpu_nodes`)."
        prompt = planner_turn_prompt(
            turn_file=turn_file,
            plan=self.layout.plan_path,
            task_yaml=self.layout.task_path,
            results_dir=self.layout.results_dir,
            finished=[
                (tid, status, summary, self.layout.result_path(tid))
                for tid, status, summary in finished
            ],
            states=[(n.id, self._describe(n)) for n in self._plan.graph.nodes],
            running=len(self._running),
            ready=len(self._ready()),
            max_parallel=self.max_parallel,
            finish=state.finish,
            planning=state.planning,
            delivered=self._delivered(),
            side_wait_minutes=self.side_wait_minutes,
            stalled=stalled,
            resumed=resumed,
            rejected=self._rejected,
            feedback=[text for _, text in feedback],
            gpu_pool=pool,
        )
        self._rejected = []
        print_rule(f"[bold magenta]Planner turn {state.turn}[/bold magenta]", self._log)
        applied: list[tuple[_Change, list[asyncio.Task[None]]]] = []

        def check() -> None:
            change = self._validate(turn_file)
            applied.append((change, self._apply(change)))

        try:
            await self._converse(prompt, check)
        except PlanError as exc:
            # Keep running the current plan; the planner hears why next turn,
            # which waits for a worker to finish even while it was planning.
            self._rejected = exc.problems
            state.planning = False
            self._save()
            self.layout.plan_path.write_text(self._plan.text, encoding="utf-8")
            print_message("[yellow]planner turn discarded[/yellow]", self._log)
            return
        change, cancelled = applied[0]
        # Only now has the planner acted on these; a discarded turn sees them again.
        del state.finished[: len(finished)]
        self._save()
        self.layout.consumed_dir.mkdir(parents=True, exist_ok=True)
        for path, _ in feedback:
            path.replace(self.layout.consumed_dir / path.name)
        decision = change.decision
        print_message(f"planner: {decision.decision} — {decision.summary}", self._log)
        await self._settle(change, cancelled)
        self._dispatch()
        if self.gpu_pool is not None and decision.gpu_nodes is not None:
            note = await asyncio.to_thread(self.gpu_pool.top_up)
            if note:
                print_message(f"gpu pool: {note}", self._log)

    async def _watch_inbox(self) -> None:
        """Wake the planner when a human message arrives."""
        while True:
            if read_inbox(self.layout):
                self._wake.set()
            await asyncio.sleep(INBOX_POLL_SECONDS)

    def _idle(self) -> bool:
        return not self._running and not self._ready()

    async def _loop(self, *, resumed: bool) -> RunOutcome:
        stalled_turns = 0
        self._dispatch()
        while True:
            if self._crash is not None:
                raise self._crash
            if self._outcome is not None:
                return await self._finish(self._outcome)
            feedback = bool(read_inbox(self.layout))
            if self._delivered():
                if self._idle() and not feedback:
                    return await self._finish(RunOutcome.SUCCEEDED)
                if self._deadline is None:
                    seconds = self.side_wait_minutes * 60
                    self._deadline = asyncio.create_task(self._stop_side_after(seconds))
            finished = bool(self._state.finished)
            planning = self._state.planning
            stalled = self._idle() and not finished and not feedback and not planning
            if not (finished or feedback or stalled or resumed or planning):
                self._wake.clear()
                await self._wake.wait()
                continue
            stalled_turns = stalled_turns + 1 if stalled else 0
            if stalled_turns > MAX_STALLED_TURNS:
                raise WorkflowError(
                    f"nothing has run for {MAX_STALLED_TURNS} planner turns in a row; "
                    f"inspect {self.layout.plan_path} and the results, then rerun to resume"
                )
            await self._turn(stalled=stalled, resumed=resumed)
            resumed = False

    async def _finish(self, outcome: RunOutcome) -> RunOutcome:
        """Record the outcome; free task branches only when everything landed.

        After an ABORT, work the final task has not landed exists only on its
        ``node/<id>`` branch, so every branch is kept. After a landing, the
        side tasks' branches are kept and listed in :data:`UNMERGED_FILE`.
        """
        await self._stop_workers()
        if self._deadline is not None:
            self._deadline.cancel()
        if self.gpu_pool is not None:
            self._pool_keeper.cancel()
            await asyncio.to_thread(self.gpu_pool.release)
        for tid, state in self._state.tasks.items():
            if state != NodeState.DONE:
                self._cancel_jobs(tid)
        if outcome == RunOutcome.SUCCEEDED:
            for node in self._plan.graph.nodes:
                if node.id in self._plan.side:
                    await self._git(self.isolation.release, node)
                else:
                    await self._discard(node)
            self._list_unmerged(self._keep_side_branches())
        self._state.phase, self._state.outcome = "done", outcome.value
        self._save()
        print_rule(f"[bold]Run finished: {outcome.value}[/bold]", self._log)
        if outcome != RunOutcome.SUCCEEDED:
            print_message(
                "Task branches are kept for inspection: "
                "`git branch --list 'node/*'` in the TensorRT-LLM checkout.",
                self._log,
            )
        return outcome

    def _keep_side_branches(self) -> dict[str, str]:
        """Rename each side task's branch after the landed branch; return the new names."""
        repo = self.repo_path
        landed = _git(repo, "symbolic-ref", "--short", "-q", "HEAD").stdout.strip()
        prefix = f"{landed}-unmerged/" if landed else "unmerged/"
        kept = {}
        for task_id in sorted(self._plan.side):
            old, new = f"node/{task_id}", f"{prefix}{task_id}"
            if _git(repo, "branch", "-M", old, new).returncode == 0:
                kept[task_id] = new
        return kept

    def _list_unmerged(self, branches: dict[str, str]) -> None:
        """Commit :data:`UNMERGED_FILE`, describing every side task, to the checkout."""
        plan, tasks = self._plan, self._state.tasks
        side = [node for node in plan.graph.nodes if node.id in plan.side]
        if not side:
            return
        lines = [
            "# Performance work left unmerged",
            "",
            "The run that landed this branch also tried the changes below on side",
            "branches and did not merge them. Each branch starts from the code of the",
            "tasks it lists as its base. To take one, merge its branch and run the",
            "accuracy and performance gates again.",
        ]
        for node in side:
            state = tasks[node.id]
            path = self.layout.result_path(node.id)
            summary = "not started"
            if state in (NodeState.DONE, NodeState.FAILED):
                summary = f"{state.name}: " + (_result_summary(path) or "(no summary)")
            lines += [
                "",
                f"## {node.id}",
                "",
                f"- Branch: `{branches[node.id]}`" if node.id in branches else "- Branch: none",
                f"- Result: {summary}",
                f"- Base: {', '.join(node.depends_on) or 'the starting checkout'}",
            ]
            if path.is_file():
                lines.append(f"- Full report: `{path}`")
            section = plan.section(node.id)
            if section:
                lines += ["", section]
        repo = self.repo_path
        (repo / UNMERGED_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
        _git(repo, "add", UNMERGED_FILE)
        message = "[None][doc] List the performance work this run left unmerged"
        _git(repo, "commit", "-s", "--no-verify", "-m", message)

    async def _main(self) -> RunOutcome:
        self._running: dict[str, asyncio.Task[None]] = {}
        # Abandoned tasks; a worker of one leaves its worktree as it is.
        self._abandoned: set[str] = set(self._state.abandoned)
        # Concurrent worktree adds/removes, merges and commits in one
        # repository race on shared metadata under .git; see _git.
        self._git_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._outcome: RunOutcome | None = None
        self._crash: Exception | None = None
        self._rejected: list[str] = []
        # Side tasks stopped at the deadline, and the deadline's timer.
        self._overdue: set[str] = set()
        self._deadline: asyncio.Task[None] | None = None
        # The op primer reads while the planner plans.
        self._op_primer: asyncio.Task[Primer | None] | None = None
        if self.primer:
            self._op_primer = asyncio.create_task(self._prime())
        # GPUs are asked for first: holder jobs queue while the planner plans.
        self._pool_keeper: asyncio.Task[None] | None = None
        if self.gpu_pool is not None:
            if self._state.gpu_nodes is not None:
                self.gpu_pool.nodes = max(self.gpu_pool.nodes, self._state.gpu_nodes)
            self.gpu_pool.write()
            self._pool_keeper = asyncio.create_task(
                self.gpu_pool.keep(lambda note: print_message(note, self._log))
            )
        self._planner: Agent = self.agent_factory(
            "planner",
            build_planner_system_prompt(max_parallel=self.max_parallel),
            self.repo_path,
            "persistent",
        )
        try:
            resumed = self._state.phase == "run"
            if resumed:
                text = self.layout.running_plan_path.read_text(encoding="utf-8")
                self._adopt(parse_plan(text))
            else:
                await self._initial_plan()
            print_rule(f"[bold cyan]Run (up to {self.max_parallel} workers)[/bold cyan]", self._log)
            watcher = asyncio.create_task(self._watch_inbox())
            try:
                return await self._loop(resumed=resumed)
            finally:
                watcher.cancel()
        finally:
            await self._stop_workers()
            await self._planner.__aexit__(None, None, None)
            if self._op_primer is not None and not self._op_primer.done():
                self._op_primer.cancel()
                await asyncio.gather(self._op_primer, return_exceptions=True)
            if self._pool_keeper is not None:
                self._pool_keeper.cancel()
                if self._state.phase != "done":
                    print_message(
                        f"The GPU pool's jobs ({self.gpu_pool.job_name}) are kept for the "
                        f"resumed run; `scancel -n {self.gpu_pool.job_name}` releases them.",
                        self._log,
                    )

    def run(self) -> RunOutcome:
        """Run (or resume) the workflow to completion."""
        self._check_workspace()
        self._state = self._init()
        if self._state.phase == "done":
            print_message(
                f"[bold]This run already finished ({self._state.outcome}); pass --clean to "
                "start over.[/bold]",
                self._log,
            )
            return RunOutcome(self._state.outcome)
        self._check_checkout()
        return asyncio.run(self._main())
