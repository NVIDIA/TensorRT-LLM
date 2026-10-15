# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The planner/worker loop: the planner runs alongside the workers.

    plan   planner writes plan.md (re-prompted until it validates)
    run    every task whose dependencies are DONE starts at once, up to
           max_parallel, each with its own worker in its own git worktree.
           Whenever workers finish, the planner is woken with their results
           and may, while the others keep running:
             - add tasks, or rewrite, rewire or remove tasks not yet started;
             - retry, rewrite or remove FAILED tasks;
             - cancel RUNNING tasks;
             - FINISH: let the final task start once its dependencies are DONE;
             - ABORT the run.
    done   the final task is DONE, or the planner aborted

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
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypeVar

from agent_flow.console import print_message, print_rule
from agent_flow.git_worktree import GitWorktreeIsolation
from agent_flow.jobs.kinds import KINDS
from agent_flow.jobs.registry import JobRegistry
from agent_flow.logger import get_logger
from agent_flow.orchestration import Node, NodeState

from .files import (
    HandoffError,
    Layout,
    PlannerDecision,
    parse_planner_decision,
    read_handoff,
    read_inbox,
)
from .node_runner import Agent, AgentFactory, make_run_task
from .plan import PlanError, PlanSpec, parse_plan
from .prompts import (
    build_planner_system_prompt,
    initial_plan_prompt,
    plan_fix_prompt,
    planner_turn_prompt,
)

__all__ = ["ModelingV2OnboardWorkflow", "RunOutcome", "WorkflowError"]

# How many times the planner is sent its validation errors within one turn.
MAX_PLAN_FIXES = 3
# How many stalled turns in a row the planner gets before the run stops.
MAX_STALLED_TURNS = 3
# How often the orchestrator looks for human messages in the inbox.
INBOX_POLL_SECONDS = 10

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
    tasks: dict[str, NodeState] = field(default_factory=dict)
    # Tasks that finished since the planner's last turn: (id, status, summary).
    finished: list[tuple[str, str, str]] = field(default_factory=list)

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
            tasks={tid: NodeState(s) for tid, s in data["tasks"].items()},
            finished=[tuple(item) for item in data.get("finished", [])],
        )


@dataclass
class _Change:
    """A validated planner turn, ready to apply."""

    decision: PlannerDecision
    plan: PlanSpec
    old_plan: PlanSpec
    # Tasks whose worker, branch and jobs are dropped: cancelled, retried,
    # rewritten after FAILED, or removed.
    reset: set[str]
    removed: set[str]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


class ModelingV2OnboardWorkflow:
    """Run the planner alongside parallel workers until the final task lands.

    Args:
        workspace: Where plan.md, results and all run state live. Must be
            outside ``repo_path``.
        task_path: The user's ``task.yaml``; copied into the workspace.
        task_data: Its validated contents.
        agent_factory: Builds the planner's and the workers' agents.
        max_parallel: The most workers that run at once.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        task_path: Path,
        task_data: dict,
        agent_factory: AgentFactory,
        max_parallel: int = 8,
    ) -> None:
        self.layout = Layout(Path(workspace).absolute())
        self.task_path = Path(task_path)
        self.repo_path = Path(task_data["trtllm_repo_path"]).absolute()
        self.slurm = "slurm-environment" in task_data
        self.agent_factory = agent_factory
        self.max_parallel = max_parallel
        self.isolation = GitWorktreeIsolation(
            repo_path=self.repo_path, worktrees_root=self.layout.worktrees_dir
        )
        self._run_task = make_run_task(
            layout=self.layout, agent_factory=agent_factory, slurm=self.slurm
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
        for path in (
            layout.state_path,
            layout.plan_path,
            layout.running_plan_path,
            layout.priorities_path,
            layout.test_commands_path,
        ):
            path.unlink(missing_ok=True)
        for directory in (
            layout.results_dir,
            layout.turns_dir,
            layout.nodes_dir,
            layout.worktrees_dir,
        ):
            shutil.rmtree(directory, ignore_errors=True)

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
        if node.id in self._running or node.id in self._resetting:
            return False
        if tasks[node.id] == NodeState.RUNNING:
            # Started before the orchestrator restarted: resume it.
            return True
        if tasks[node.id] != NodeState.PENDING:
            return False
        if node.id == self._plan.sink_id() and not self._state.finish:
            return False
        return all(tasks.get(dep) == NodeState.DONE for dep in node.depends_on)

    def _ready(self) -> list[Node]:
        """Tasks that can start now but have no worker yet, resumed ones first."""
        nodes = [n for n in self._plan.graph.nodes if self._startable(n)]
        return sorted(nodes, key=lambda n: self._state.tasks[n.id] != NodeState.RUNNING)

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
        """One task end to end: worktree, merge of its dependencies, worker, commit."""
        tags = self._plan.tags[node.id]
        is_final = node.id == self._plan.sink_id()
        dep_nodes = [self._plan.graph.by_id[dep] for dep in node.depends_on]
        try:
            path = await self._git(self.isolation.acquire, node)
            try:
                if dep_nodes:
                    await self._git(self.isolation.prepare, node, dep_nodes, path)
                result = await self._run_task(node, path, tags, is_final)
                await self._git(self.isolation.commit, node, path)
            finally:
                await self._git(self.isolation.release, node)
        except Exception as exc:
            # Worker failures arrive as FAILED results; an exception here is the
            # orchestrator's own git machinery failing, which stops the run.
            self._crash = exc
            self._wake.set()
            return
        del self._running[node.id]
        state = NodeState.DONE if result.status == "DONE" else NodeState.FAILED
        self._set(node.id, state)
        self._state.finished.append((node.id, result.status, result.summary))
        self._save()
        if is_final and state == NodeState.DONE:
            self._outcome = RunOutcome.SUCCEEDED
        self._wake.set()
        self._dispatch()

    def _cancel_jobs(self, task_id: str) -> None:
        jobs_path = self.layout.jobs_path(task_id)
        if not jobs_path.is_file():
            return
        for entry in JobRegistry(jobs_path).all():
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
        def check() -> None:
            plan = parse_plan(self._read_plan_md())
            self._adopt(plan)
            self._state.tasks = {node.id: NodeState.PENDING for node in plan.graph.nodes}
            self._state.phase = "run"
            self._save()

        print_rule("[bold magenta]Plan[/bold magenta]", self._log)
        await self._converse(
            initial_plan_prompt(
                task_yaml=self.layout.task_path,
                plan=self.layout.plan_path,
                repo=self.repo_path,
            ),
            check,
        )
        print_message(f"plan: {len(self._plan.graph.nodes)} tasks", self._log)

    def _validate(self, turn_file: Path) -> _Change:
        """Check the planner's turn against the current task states."""
        decision = read_handoff(turn_file, parse_planner_decision)
        tasks = self._state.tasks
        problems: list[str] = []
        for tid in decision.cancel:
            if tasks.get(tid) != NodeState.RUNNING:
                state = tasks[tid].value if tid in tasks else "not in the plan"
                problems.append(f"`cancel`: task {tid!r} is not running ({state})")
        for tid in decision.retry:
            if tasks.get(tid) != NodeState.FAILED:
                state = tasks[tid].value if tid in tasks else "not in the plan"
                problems.append(f"`retry`: task {tid!r} is not FAILED ({state})")
        text = self._read_plan_md()
        plan = self._plan
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
        reset = set(decision.cancel) | set(decision.retry) | rewritten_failed | removed
        return _Change(decision=decision, plan=plan, old_plan=old, reset=reset, removed=removed)

    def _apply(self, change: _Change) -> list[asyncio.Task[None]]:
        """Apply a validated turn to the run state; return the cancelled workers."""
        cancelled = []
        for tid in change.decision.cancel:
            task = self._running.pop(tid)
            task.cancel()
            cancelled.append(task)
        self._resetting |= change.reset
        self._adopt(change.plan)
        tasks = self._state.tasks
        for tid in sorted(change.reset):
            if tid in change.removed:
                tasks.pop(tid, None)
                print_message(f"  task {tid}: removed", self._log)
            else:
                self._set(tid, NodeState.PENDING)
        for node in change.plan.graph.nodes:
            if node.id not in tasks:
                self._set(node.id, NodeState.PENDING)
        if change.decision.decision == "FINISH":
            self._state.finish = True
        if change.decision.decision == "ABORT":
            self._outcome = RunOutcome.ABORTED
        self._save()
        return cancelled

    async def _reset(self, change: _Change, cancelled: list[asyncio.Task[None]]) -> None:
        """Drop the reset tasks' workers, jobs, notes and branches.

        Each reset task's last result is kept as an earlier attempt for the
        next worker and the planner to read.
        """
        await asyncio.gather(*cancelled, return_exceptions=True)
        for tid in sorted(change.reset):
            node = change.plan.graph.by_id.get(tid) or change.old_plan.graph.by_id[tid]
            self._cancel_jobs(tid)
            result = self.layout.result_path(tid)
            if result.is_file():
                attempt = len(self.layout.previous_attempts(tid)) + 1
                result.replace(self.layout.attempt_path(tid, attempt))
            shutil.rmtree(self.layout.node_dir(tid), ignore_errors=True)
            await self._discard(node)
            self._resetting.discard(tid)

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
        if node.id == self._plan.sink_id() and not self._state.finish:
            return "pending — held until FINISH"
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
            stalled=stalled,
            resumed=resumed,
            rejected=self._rejected,
            feedback=[text for _, text in feedback],
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
            # Keep running the current plan; the planner hears why next turn.
            self._rejected = exc.problems
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
        await self._reset(change, cancelled)
        self._dispatch()

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
            finished = bool(self._state.finished)
            feedback = bool(read_inbox(self.layout))
            stalled = self._idle() and not finished and not feedback
            if not (finished or feedback or stalled or resumed):
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
        ``node/<id>`` branch, so every branch is kept.
        """
        await self._stop_workers()
        for tid, state in self._state.tasks.items():
            if state != NodeState.DONE:
                self._cancel_jobs(tid)
        if outcome == RunOutcome.SUCCEEDED:
            for node in self._plan.graph.nodes:
                await self._discard(node)
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

    async def _main(self) -> RunOutcome:
        self._running: dict[str, asyncio.Task[None]] = {}
        self._resetting: set[str] = set()
        # Concurrent worktree adds/removes, merges and commits in one
        # repository race on shared metadata under .git; see _git.
        self._git_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._outcome: RunOutcome | None = None
        self._crash: Exception | None = None
        self._rejected: list[str] = []
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
