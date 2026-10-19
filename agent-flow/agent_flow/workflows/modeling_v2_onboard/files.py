# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The files agents hand back to the orchestrator, and where they live.

A worker ends its task by writing ``task_results/task_result_<id>.md``: a YAML
front matter block whose ``status`` the orchestrator branches on, followed by a
free-form markdown report for the planner and for downstream workers. The
planner ends every turn by writing ``planner_turns/turn_<n>.yaml``. Both are
plain files so the claude-code and codex backends behave identically.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent_flow.git_worktree import node_id_to_slug

__all__ = [
    "PLANNER_DECISIONS",
    "TASK_STATUSES",
    "HandoffError",
    "PlannerDecision",
    "TaskResult",
    "Layout",
    "parse_planner_decision",
    "parse_task_result",
    "post_feedback",
    "read_handoff",
    "read_inbox",
]

TASK_STATUSES: tuple[str, ...] = ("DONE", "FAILED")
PLANNER_DECISIONS: tuple[str, ...] = ("CONTINUE", "FINISH", "ABORT")


class HandoffError(ValueError):
    """A handoff file is missing or does not follow its format."""


@dataclass(frozen=True)
class TaskResult:
    """The orchestrator-relevant part of ``task_result_<id>.md``."""

    status: str
    summary: str


@dataclass(frozen=True)
class PlannerDecision:
    """What a planner turn asks of the orchestrator, besides any plan.md edit.

    Attributes:
        decision: ``CONTINUE``; ``FINISH`` to let the final task start once its
            dependencies are done; ``ABORT`` to stop the run.
        summary: What the planner changed and why.
        retry: ``FAILED`` tasks to run again unchanged.
        cancel: ``RUNNING`` tasks to stop; they start over unless the plan
            removes or rewrites them.
        planning: The plan is not complete yet: the planner gets its next
            turn as soon as this one is applied, while the workers run.
        gpu_nodes: Grow the GPU pool to this many nodes; ``None`` keeps it.
    """

    decision: str
    summary: str
    retry: tuple[str, ...] = ()
    cancel: tuple[str, ...] = ()
    planning: bool = False
    gpu_nodes: int | None = None


@dataclass(frozen=True)
class Layout:
    """Every path the workflow keeps under its workspace."""

    workspace: Path

    @property
    def task_path(self) -> Path:
        return self.workspace / "task.yaml"

    @property
    def plan_path(self) -> Path:
        return self.workspace / "plan.md"

    @property
    def running_plan_path(self) -> Path:
        """The last plan.md that validated: the plan the orchestrator runs."""
        return self.workspace / ".running_plan.md"

    @property
    def state_path(self) -> Path:
        return self.workspace / ".state.json"

    @property
    def primers_path(self) -> Path:
        """The primer sessions this run built, by kind."""
        return self.workspace / "primers.json"

    def knowledge_path(self, kind: str) -> Path:
        """The summary a primer of ``kind`` wrote for workers not forked from it."""
        return self.workspace / "knowledge" / f"{kind}.md"

    @property
    def gpu_pool_dir(self) -> Path:
        """The run's GPU pool: its holder script, locks, queue and logs."""
        return self.workspace / "gpu_pool"

    @property
    def priorities_path(self) -> Path:
        """``{task id: priority class}`` of the running plan, for the GPU pool."""
        return self.workspace / "gpu_priorities.json"

    @property
    def results_dir(self) -> Path:
        return self.workspace / "task_results"

    @property
    def turns_dir(self) -> Path:
        return self.workspace / "planner_turns"

    @property
    def nodes_dir(self) -> Path:
        return self.workspace / "nodes"

    @property
    def inbox_dir(self) -> Path:
        """Human messages waiting for the planner, one file each."""
        return self.workspace / "inbox"

    @property
    def consumed_dir(self) -> Path:
        """Human messages a planner turn has acted on."""
        return self.inbox_dir / "consumed"

    @property
    def abandoned_path(self) -> Path:
        """What every abandoned task left behind, for a human to clean up."""
        return self.workspace / "abandoned.md"

    @property
    def worktrees_dir(self) -> Path:
        return self.workspace / "worktrees"

    @property
    def test_commands_path(self) -> Path:
        return self.workspace / "test_commands.md"

    def result_path(self, task_id: str) -> Path:
        return self.results_dir / f"task_result_{node_id_to_slug(task_id)}.md"

    def attempt_path(self, task_id: str, attempt: int) -> Path:
        """Where a superseded result is kept once its task is retried."""
        return self.results_dir / f"task_result_{node_id_to_slug(task_id)}.attempt{attempt}.md"

    def previous_attempts(self, task_id: str) -> list[Path]:
        slug = node_id_to_slug(task_id)
        found = self.results_dir.glob(f"task_result_{slug}.attempt*.md")
        return sorted(found, key=lambda p: int(p.stem.rsplit("attempt", 1)[1]))

    def node_dir(self, task_id: str) -> Path:
        """A task's private directory: its notes and its ``jobs.json``."""
        return self.nodes_dir / node_id_to_slug(task_id)

    def jobs_path(self, task_id: str) -> Path:
        return self.node_dir(task_id) / "jobs.json"

    def turn_path(self, turn: int) -> Path:
        return self.turns_dir / f"turn_{turn}.yaml"


def _front_matter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise HandoffError("the file must start with a `---` front matter line")
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:i])
    raise HandoffError("the front matter has no closing `---` line")


def _mapping(text: str) -> dict:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HandoffError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HandoffError("expected a YAML mapping")
    return data


def _choice(data: dict, key: str, allowed: tuple[str, ...]) -> str:
    value = str(data.get(key, "")).strip().upper()
    if value not in allowed:
        raise HandoffError(f"`{key}` must be one of {', '.join(allowed)}, got {data.get(key)!r}")
    return value


def parse_task_result(text: str) -> TaskResult:
    """Parse a worker's ``task_result_<id>.md``.

    Raises:
        HandoffError: When the front matter is missing or its ``status`` is not
            one of :data:`TASK_STATUSES`.
    """
    data = _mapping(_front_matter(text))
    return TaskResult(
        status=_choice(data, "status", TASK_STATUSES),
        summary=str(data.get("summary", "")).strip(),
    )


def _ids(data: dict, key: str) -> tuple[str, ...]:
    value = data.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise HandoffError(f"`{key}` must be a list of task ids, got {value!r}")
    return tuple(dict.fromkeys(value))


def parse_planner_decision(text: str) -> PlannerDecision:
    """Parse a planner's ``turn_<n>.yaml``.

    Raises:
        HandoffError: When ``decision`` is not one of :data:`PLANNER_DECISIONS`
            or ``retry``/``cancel`` is not a list of ids, ``planning`` is
            not a boolean or ``gpu_nodes`` not a positive integer.
    """
    data = _mapping(text)
    planning = data.get("planning", False)
    if not isinstance(planning, bool):
        raise HandoffError(f"`planning` must be true or false, got {planning!r}")
    gpu_nodes = data.get("gpu_nodes")
    if gpu_nodes is not None and (
        isinstance(gpu_nodes, bool) or not isinstance(gpu_nodes, int) or gpu_nodes < 1
    ):
        raise HandoffError(f"`gpu_nodes` must be a positive integer, got {gpu_nodes!r}")
    return PlannerDecision(
        decision=_choice(data, "decision", PLANNER_DECISIONS),
        summary=str(data.get("summary", "")).strip(),
        retry=_ids(data, "retry"),
        cancel=_ids(data, "cancel"),
        planning=planning,
        gpu_nodes=gpu_nodes,
    )


def read_handoff(path: Path, parse):
    """Read and parse ``path``, raising :class:`HandoffError` when it is absent."""
    if not path.is_file():
        raise HandoffError(f"{path} was not written")
    return parse(path.read_text(encoding="utf-8"))


def post_feedback(workspace: Path, text: str) -> Path:
    """Leave a human message for the planner of the run in ``workspace``.

    The running orchestrator picks it up within seconds and hands it to the
    planner's next turn. Returns the message file.
    """
    if not text.strip():
        raise ValueError("the message is empty")
    inbox = Layout(Path(workspace).absolute()).inbox_dir
    inbox.mkdir(parents=True, exist_ok=True)
    now = time.time_ns()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now / 1e9))
    # Names sort by time and never collide, so no message overwrites another.
    path = inbox / f"{stamp}.{now % 10**9:09d}-{os.getpid()}.md"
    tmp = inbox / f".{path.name}.tmp"
    with open(tmp, "x", encoding="utf-8") as handle:
        handle.write(text.strip() + "\n")
    tmp.rename(path)
    return path


def read_inbox(layout: Layout) -> list[tuple[Path, str]]:
    """Every waiting human message, oldest first."""
    if not layout.inbox_dir.is_dir():
        return []
    return [
        (path, path.read_text(encoding="utf-8")) for path in sorted(layout.inbox_dir.glob("*.md"))
    ]
