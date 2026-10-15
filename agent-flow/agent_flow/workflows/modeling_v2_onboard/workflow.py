# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The planner/worker loop.

    plan     planner writes plan.md (re-prompted until it validates)
    build    NodeScheduler runs the plan's tasks in parallel, one worker per
             task, each in its own git worktree; pauses on the first FAILED
    replan   planner reads every result and decides:
               REVISED  reset the failed and changed tasks, back to build
               DONE     stop; the results already meet the completion criteria
               ABORT    stop; the goal cannot be reached
    done

There is no round limit: the run ends when every task is DONE or the planner
stops it. Every phase transition is checkpointed in ``.state.json`` and the
scheduler checkpoints task states in ``.graph_state.json``, so rerunning the
same command after an interruption resumes where it stopped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

import anyio

from agent_flow.console import print_message, print_rule
from agent_flow.git_worktree import GitWorktreeIsolation
from agent_flow.jobs.kinds import KINDS
from agent_flow.jobs.registry import JobRegistry
from agent_flow.logger import get_logger
from agent_flow.orchestration import GraphResult, GraphState, Node, NodeScheduler, NodeState
from agent_flow.orchestration.invalidation import invalidation_set

from .files import HandoffError, Layout, PlannerDecision, parse_planner_decision, read_handoff
from .node_runner import AgentFactory, make_run_node
from .plan import PlanError, PlanSpec, parse_plan
from .prompts import (
    build_planner_system_prompt,
    initial_plan_prompt,
    plan_fix_prompt,
    replan_prompt,
)

__all__ = ["ModelingV2OnboardWorkflow", "RunOutcome", "WorkflowError"]

# How many times the planner is sent its validation errors before the run stops.
MAX_PLAN_FIXES = 3

_STATE_VERSION = 1

T = TypeVar("T")


class WorkflowError(RuntimeError):
    """The run cannot start or continue as configured."""


class RunOutcome(str, Enum):
    """How a run ended."""

    SUCCEEDED = "succeeded"  # every task DONE
    PLANNER_DONE = "planner_done"  # the planner judged the criteria met
    ABORTED = "aborted"  # the planner judged the goal unreachable


@dataclass
class _State:
    phase: str = "plan"  # plan | build | replan | done
    round: int = 0
    outcome: str | None = None

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": _STATE_VERSION, **asdict(self)}), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> _State:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _STATE_VERSION:
            raise WorkflowError(f"{path} has unsupported version {data.get('version')!r}")
        return cls(phase=data["phase"], round=data["round"], outcome=data.get("outcome"))


class _KeepBranches:
    """Delegate to a worktree provider, but keep every task branch until the end.

    The scheduler reclaims a task's branch once its dependents are terminal —
    including a dependent that just FAILED. A failed task that the planner
    re-runs must start from its dependencies' code again, so their branches
    must outlive the build. The workflow reclaims explicitly instead: a reset
    task as soon as it is reset, and every task once the run is over.
    """

    def __init__(self, inner: GitWorktreeIsolation) -> None:
        self.inner = inner

    async def acquire(self, node: Node) -> Path:
        return await self.inner.acquire(node)

    async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object:
        return await self.inner.prepare(node, dep_nodes, cwd)

    async def commit(self, node: Node, cwd: Path) -> str | None:
        return await self.inner.commit(node, cwd)

    async def release(self, node: Node) -> None:
        await self.inner.release(node)

    async def reclaim(self, node: Node) -> None:
        return None

    async def discard(self, node: Node) -> None:
        """Drop ``node``'s worktree and branch so it re-runs from scratch."""
        await self.inner.release(node)
        await self.inner.reclaim(node)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


class ModelingV2OnboardWorkflow:
    """Plan, build in parallel, and replan on failure until the plan is done.

    Args:
        workspace: Where plan.md, results and all run state live. Must be
            outside ``repo_path``.
        task_path: The user's ``task.yaml``; copied into the workspace.
        task_data: Its validated contents.
        agent_factory: Builds the planner's and the workers' agents.
        max_parallel: The most tasks that run at once.
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
        self.isolation = _KeepBranches(
            GitWorktreeIsolation(repo_path=self.repo_path, worktrees_root=self.layout.worktrees_dir)
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
        if layout.plan_path.exists() or layout.graph_state_path.exists():
            raise WorkflowError(
                f"{layout.workspace} holds files from an earlier run but no .state.json; "
                "pass --clean to start over"
            )
        if self.task_path.resolve() != layout.task_path.resolve():
            shutil.copyfile(self.task_path, layout.task_path)
        for directory in (layout.results_dir, layout.decisions_dir, layout.nodes_dir):
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
            layout.graph_state_path,
            layout.plan_path,
            layout.built_plan_path,
            layout.test_commands_path,
        ):
            path.unlink(missing_ok=True)
        for directory in (
            layout.results_dir,
            layout.decisions_dir,
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

    # ---------------------------------------------------------------- planner

    async def _planner_session(self, prompt: str, check: Callable[[], T]) -> T:
        """Run one planner session, feeding validation errors back until ``check`` passes."""
        agent = self.agent_factory(
            "planner",
            build_planner_system_prompt(max_parallel=self.max_parallel),
            self.repo_path,
            "persistent",
        )
        try:
            await agent.aforward(prompt)
            for _ in range(MAX_PLAN_FIXES):
                try:
                    return check()
                except PlanError as exc:
                    problems = exc.problems
                except HandoffError as exc:
                    problems = [str(exc)]
                print_message(
                    f"[yellow]plan rejected: {len(problems)} problem(s); asking the planner "
                    "to fix them[/yellow]",
                    self._log,
                )
                await agent.aforward(plan_fix_prompt(problems, plan=self.layout.plan_path))
            return check()
        finally:
            await agent.__aexit__(None, None, None)

    def _read_plan(self, **kwargs) -> PlanSpec:
        if not self.layout.plan_path.is_file():
            raise PlanError([f"{self.layout.plan_path} was not written"])
        return parse_plan(self.layout.plan_path.read_text(encoding="utf-8"), **kwargs)

    def _plan(self) -> PlanSpec:
        prompt = initial_plan_prompt(
            task_yaml=self.layout.task_path, plan=self.layout.plan_path, repo=self.repo_path
        )
        return anyio.run(self._planner_session, prompt, self._read_plan)

    def _replan(
        self, round_index: int, built: PlanSpec, states: dict[str, NodeState]
    ) -> tuple[PlannerDecision, PlanSpec | None]:
        decision_path = self.layout.decision_path(round_index)
        decision_path.unlink(missing_ok=True)

        def by_state(state: NodeState) -> list[str]:
            return [tid for tid, s in states.items() if s == state and tid in built.graph.by_id]

        failed = by_state(NodeState.FAILED)
        prompt = replan_prompt(
            task_yaml=self.layout.task_path,
            plan=self.layout.plan_path,
            results_dir=self.layout.results_dir,
            failed=[(tid, self.layout.result_path(tid)) for tid in failed],
            interrupted=by_state(NodeState.INTERRUPTED),
            done=by_state(NodeState.DONE),
            decision=decision_path,
        )

        def check() -> tuple[PlannerDecision, PlanSpec | None]:
            decision = read_handoff(decision_path, parse_planner_decision)
            if decision.decision != "REVISED":
                return decision, None
            return decision, self._read_plan(previous=built, states=states)

        return anyio.run(self._planner_session, prompt, check)

    # ------------------------------------------------------------------ build

    def _build(self, plan: PlanSpec) -> GraphResult:
        self.layout.built_plan_path.write_text(plan.text, encoding="utf-8")

        def on_transition(node_id: str, old: NodeState, new: NodeState) -> None:
            print_message(f"  task {node_id}: {old.value} → {new.value}", self._log)

        scheduler = NodeScheduler(
            plan.graph,
            make_run_node(
                layout=self.layout,
                plan=plan,
                repo_path=self.repo_path,
                agent_factory=self.agent_factory,
                slurm=self.slurm,
            ),
            self.isolation,
            on_transition=on_transition,
            max_parallel=self.max_parallel,
            checkpoint_path=self.layout.graph_state_path,
            pause_on_failure=True,
        )
        return anyio.run(scheduler.run)

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
                # A job that is already gone must not stop the replan.
                continue

    def _reset(self, old: PlanSpec, new: PlanSpec, failed: set[str]) -> list[str]:
        """Reset every task the revision invalidates; return their ids.

        That is the failed tasks, the tasks whose graph entry changed or that
        were added, everything downstream of those, and the removed tasks. Each
        loses its detached jobs, notes, worktree and branch; its last result is
        kept as an earlier attempt for the next worker to read. ``DONE`` tasks
        and untouched interrupted tasks keep everything and resume.
        """
        graph_state = GraphState.load(self.layout.graph_state_path)
        removed = set(old.graph.by_id) - set(new.graph.by_id)
        frontier = failed | invalidation_set(old.graph, new.graph) | removed
        reset: list[str] = []
        for task_id in sorted(frontier):
            if graph_state.states.get(task_id) == NodeState.DONE:
                continue
            node = new.graph.by_id.get(task_id) or old.graph.by_id[task_id]
            self._cancel_jobs(task_id)
            result = self.layout.result_path(task_id)
            if result.is_file():
                attempt = len(self.layout.previous_attempts(task_id)) + 1
                result.replace(self.layout.attempt_path(task_id, attempt))
            shutil.rmtree(self.layout.node_dir(task_id), ignore_errors=True)
            anyio.run(self.isolation.discard, node)
            graph_state.worktrees.pop(task_id, None)
            if task_id in removed:
                graph_state.states.pop(task_id, None)
            else:
                graph_state.states[task_id] = NodeState.PENDING
            reset.append(task_id)
        graph_state.save(self.layout.graph_state_path)
        return reset

    # -------------------------------------------------------------------- run

    def _finish(self, state: _State, outcome: RunOutcome, plan: PlanSpec) -> RunOutcome:
        """Record the outcome; free task branches only when everything landed.

        When the planner stops the run early, work the final task has not
        landed exists only on its ``node/<id>`` branch, so every branch is kept.
        """
        states = GraphState.load(self.layout.graph_state_path).states
        for node in plan.graph.nodes:
            if states.get(node.id) != NodeState.DONE:
                self._cancel_jobs(node.id)
            if outcome == RunOutcome.SUCCEEDED:
                anyio.run(self.isolation.discard, node)
        state.phase, state.outcome = "done", outcome.value
        state.save(self.layout.state_path)
        print_rule(f"[bold]Run finished: {outcome.value}[/bold]", self._log)
        if outcome != RunOutcome.SUCCEEDED:
            print_message(
                "Task branches are kept for inspection: "
                "`git branch --list 'node/*'` in the TensorRT-LLM checkout.",
                self._log,
            )
        return outcome

    def run(self) -> RunOutcome:
        """Run (or resume) the workflow to completion."""
        self._check_workspace()
        state = self._init()
        layout = self.layout
        if state.phase == "done":
            print_message(
                f"[bold]This run already finished ({state.outcome}); pass --clean to start "
                "over.[/bold]",
                self._log,
            )
            return RunOutcome(state.outcome)
        self._check_checkout()

        if state.phase == "plan":
            print_rule("[bold magenta]Plan[/bold magenta]", self._log)
            self._plan()
            state.phase = "build"
            state.save(layout.state_path)

        while True:
            if state.phase == "build":
                plan = self._read_plan()
                print_rule(
                    f"[bold cyan]Build ({len(plan.graph.nodes)} tasks)[/bold cyan]", self._log
                )
                result = self._build(plan)
                if result.succeeded:
                    return self._finish(state, RunOutcome.SUCCEEDED, plan)
                state.phase = "replan"
                state.round += 1
                state.save(layout.state_path)

            built = parse_plan(layout.built_plan_path.read_text(encoding="utf-8"))
            states = GraphState.load(layout.graph_state_path).states
            print_rule(f"[bold magenta]Replan {state.round}[/bold magenta]", self._log)
            decision, revised = self._replan(state.round, built, states)
            print_message(f"planner: {decision.decision} — {decision.summary}", self._log)
            if decision.decision == "DONE":
                return self._finish(state, RunOutcome.PLANNER_DONE, built)
            if decision.decision == "ABORT":
                return self._finish(state, RunOutcome.ABORTED, built)
            failed = {tid for tid, s in states.items() if s == NodeState.FAILED}
            reset = self._reset(built, revised, failed)
            print_message(f"reset: {', '.join(reset) or 'none'}", self._log)
            state.phase = "build"
            state.save(layout.state_path)
