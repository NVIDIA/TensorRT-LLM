# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one task with one worker.

A worker is one agent session pinned to the task's worktree. It gets the base
worker prompt plus the block of each of the task's tags, runs the task in a
single turn, and ends by writing ``task_result_<id>.md``. The task's outcome is
that file's ``status``; a worker that leaves no valid file gets one follow-up
turn to write it, and the task is ``FAILED`` if it still does not.

A worker given a :class:`Primer` starts as a fork of the primer's session; if
that session cannot start, the worker starts afresh instead.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from agent_flow.orchestration import Node

from .files import HandoffError, Layout, TaskResult, parse_task_result, read_handoff
from .prompts import WorkerTask, build_worker_system_prompt, worker_retry_prompt, worker_turn_prompt

__all__ = ["Agent", "AgentFactory", "Primer", "RunTask", "make_run_task", "write_failure"]


class Agent(Protocol):
    """The part of :class:`~agent_flow.layers.AgentLayer` the workflow drives."""

    async def aforward(self, content: str) -> str: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...


class AgentFactory(Protocol):
    """Builds an agent; injected so tests can substitute scripted agents.

    ``fork_from`` starts the agent as a fork of that earlier session.
    """

    def __call__(
        self,
        name: str,
        system_prompt: str,
        cwd: Path,
        session_mode: str,
        fork_from: str | None = None,
    ) -> Agent: ...


@dataclass(frozen=True)
class Primer:
    """A briefing session that workers start as forks of.

    Attributes:
        session_id: The backend's id of the session.
        repo: The checkout it read.
        commit: The commit it read there.
    """

    session_id: str
    repo: Path
    commit: str


# ``(node, worktree, tags, is_final, side, primer) -> result``.
RunTask = Callable[
    [Node, Path, tuple[str, ...], bool, bool, "Primer | None"], Awaitable[TaskResult]
]


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def write_failure(
    path: Path, reason: str, summary: str = "the worker ended without a valid result file"
) -> TaskResult:
    """Record a task the worker could not report on, so the planner can see why."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nstatus: FAILED\nsummary: {summary}\n---\n\n"
        f"Written by the orchestrator, not by the worker.\n\n{reason}\n",
        encoding="utf-8",
    )
    return TaskResult(status="FAILED", summary=summary)


async def _run_worker(agent: Agent, prompt: str, result_path: Path) -> TaskResult:
    await agent.aforward(prompt)
    try:
        return read_handoff(result_path, parse_task_result)
    except HandoffError as exc:
        problem = str(exc)
    await agent.aforward(worker_retry_prompt(result_path, problem))
    try:
        return read_handoff(result_path, parse_task_result)
    except HandoffError as exc:
        return write_failure(result_path, str(exc))


def make_run_task(
    *,
    layout: Layout,
    agent_factory: AgentFactory,
    slurm: bool,
    gpu_run: str | None = None,
) -> RunTask:
    """Build the callback that runs one task with a fresh worker.

    Args:
        layout: The workspace layout.
        agent_factory: Builds each worker's agent.
        slurm: Whether ``task.yaml`` carries a ``slurm-environment``.
        gpu_run: The command prefix that runs a command in the run's GPU pool,
            when it has one.
    """

    async def run_task(
        node: Node,
        cwd: Path,
        tags: tuple[str, ...],
        is_final: bool,
        side: bool,
        primer: Primer | None = None,
    ) -> TaskResult:
        result_path = layout.result_path(node.id)
        # A result left by an interrupted turn of this same attempt is stale.
        result_path.unlink(missing_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        layout.node_dir(node.id).mkdir(parents=True, exist_ok=True)
        task = WorkerTask(
            task_id=node.id,
            tags=tags,
            worktree=Path(cwd),
            is_final=is_final,
            side=side,
            task_yaml=layout.task_path,
            plan=layout.plan_path,
            result=result_path,
            upstream_results=tuple((dep, layout.result_path(dep)) for dep in node.depends_on),
            notes_dir=layout.node_dir(node.id),
            jobs=layout.jobs_path(node.id),
            previous_attempts=tuple(layout.previous_attempts(node.id)),
            test_commands=layout.test_commands_path if slurm else None,
            gpu_run=gpu_run,
            knowledge=_existing(layout.knowledge_path("op")) if primer is None else None,
        )
        system_prompt = build_worker_system_prompt(tags, slurm=slurm, gpu_pool=gpu_run is not None)
        if primer is not None:
            agent = agent_factory(
                f"worker[{node.id}]",
                system_prompt,
                Path(cwd),
                "persistent",
                fork_from=primer.session_id,
            )
            primed = replace(task, primed_repo=primer.repo, primed_commit=primer.commit)
            try:
                return await _run_worker(agent, worker_turn_prompt(primed), result_path)
            except Exception:
                if result_path.is_file():
                    return write_failure(
                        result_path, "The worker's session raised after writing its result."
                    )
                # The fork did not start or broke: start afresh in the same worktree.
            finally:
                await agent.__aexit__(None, None, None)
            task = replace(task, knowledge=_existing(layout.knowledge_path("op")))
        agent = agent_factory(f"worker[{node.id}]", system_prompt, Path(cwd), "persistent")
        try:
            return await _run_worker(agent, worker_turn_prompt(task), result_path)
        except Exception as exc:
            # A crashed session fails this task, not the run: the planner decides.
            return write_failure(result_path, f"The worker's session raised: {exc!r}")
        finally:
            await agent.__aexit__(None, None, None)

    return run_task
