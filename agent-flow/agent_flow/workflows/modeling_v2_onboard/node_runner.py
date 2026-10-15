# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one task with one worker: the scheduler's ``run_node`` callback.

A worker is one agent session pinned to the task's worktree. It gets the base
worker prompt plus the block of each of the task's tags, runs the task in a
single turn, and ends by writing ``task_result_<id>.md``. The task's outcome is
that file's ``status``; a worker that leaves no valid file gets one follow-up
turn to write it, and the task is ``FAILED`` if it still does not.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from agent_flow.orchestration import Node, NodeOutcome, NodeState, RunNode

from .files import HandoffError, Layout, TaskResult, parse_task_result, read_handoff
from .plan import PlanSpec
from .prompts import WorkerTask, build_worker_system_prompt, worker_retry_prompt, worker_turn_prompt

__all__ = ["Agent", "AgentFactory", "make_run_node"]


class Agent(Protocol):
    """The part of :class:`~agent_flow.layers.AgentLayer` the workflow drives."""

    async def aforward(self, content: str) -> str: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...


# ``(name, system_prompt, cwd, session_mode) -> Agent``. Injected so tests can
# substitute scripted agents for real backends.
AgentFactory = Callable[[str, str, Path, str], Agent]


def _write_failure(path: Path, reason: str) -> TaskResult:
    """Record a task the worker could not report on, so the planner can see why."""
    summary = "the worker ended without a valid result file"
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
        return _write_failure(result_path, str(exc))


def make_run_node(
    *,
    layout: Layout,
    plan: PlanSpec,
    repo_path: Path,
    agent_factory: AgentFactory,
    slurm: bool,
) -> RunNode:
    """Build the scheduler callback that runs each task of ``plan``.

    Args:
        layout: The workspace layout.
        plan: The plan being built; it supplies each task's tags.
        repo_path: The TensorRT-LLM checkout the final task runs in.
        agent_factory: Builds each worker's agent.
        slurm: Whether ``task.yaml`` carries a ``slurm-environment``.
    """
    final_id = plan.sink_id()

    async def run_node(node: Node, cwd: Path) -> NodeOutcome:
        tags = plan.tags[node.id]
        result_path = layout.result_path(node.id)
        # A result left by an interrupted turn of this same attempt is stale.
        result_path.unlink(missing_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        layout.node_dir(node.id).mkdir(parents=True, exist_ok=True)
        task = WorkerTask(
            task_id=node.id,
            tags=tags,
            worktree=Path(cwd),
            is_final=node.id == final_id,
            task_yaml=layout.task_path,
            plan=layout.plan_path,
            result=result_path,
            upstream_results=tuple((dep, layout.result_path(dep)) for dep in node.depends_on),
            notes_dir=layout.node_dir(node.id),
            jobs=layout.jobs_path(node.id),
            previous_attempts=tuple(layout.previous_attempts(node.id)),
            test_commands=layout.test_commands_path if slurm else None,
        )
        agent = agent_factory(
            f"worker[{node.id}]",
            build_worker_system_prompt(tags, slurm=slurm),
            Path(cwd),
            "persistent",
        )
        info: dict[str, Any] = {"node_id": node.id, "tags": list(tags)}
        try:
            result = await _run_worker(agent, worker_turn_prompt(task), result_path)
        except Exception as exc:
            # A crashed session fails this task, not the run: the planner decides.
            result = _write_failure(result_path, f"The worker's session raised: {exc!r}")
        finally:
            await agent.__aexit__(None, None, None)
        info["summary"] = result.summary
        state = NodeState.DONE if result.status == "DONE" else NodeState.FAILED
        return NodeOutcome(terminal_state=state, info=info)

    return run_node
