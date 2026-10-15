# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The files agents hand back to the orchestrator, and where they live.

A worker ends its task by writing ``task_results/task_result_<id>.md``: a YAML
front matter block whose ``status`` the orchestrator branches on, followed by a
free-form markdown report for the planner and for downstream workers. A planner
ends a replan by writing ``planner_decisions/round_<n>.yaml``. Both are plain
files so the claude-code and codex backends behave identically.
"""

from __future__ import annotations

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
    "read_handoff",
]

TASK_STATUSES: tuple[str, ...] = ("DONE", "FAILED")
PLANNER_DECISIONS: tuple[str, ...] = ("REVISED", "DONE", "ABORT")


class HandoffError(ValueError):
    """A handoff file is missing or does not follow its format."""


@dataclass(frozen=True)
class TaskResult:
    """The orchestrator-relevant part of ``task_result_<id>.md``."""

    status: str
    summary: str


@dataclass(frozen=True)
class PlannerDecision:
    """A planner's verdict at the end of a replan."""

    decision: str
    summary: str


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
    def built_plan_path(self) -> Path:
        """The plan.md the last build ran, which a replan is diffed against."""
        return self.workspace / ".built_plan.md"

    @property
    def state_path(self) -> Path:
        return self.workspace / ".state.json"

    @property
    def graph_state_path(self) -> Path:
        return self.workspace / ".graph_state.json"

    @property
    def results_dir(self) -> Path:
        return self.workspace / "task_results"

    @property
    def decisions_dir(self) -> Path:
        return self.workspace / "planner_decisions"

    @property
    def nodes_dir(self) -> Path:
        return self.workspace / "nodes"

    @property
    def worktrees_dir(self) -> Path:
        return self.workspace / "worktrees"

    @property
    def test_commands_path(self) -> Path:
        return self.workspace / "test_commands.md"

    def result_path(self, task_id: str) -> Path:
        return self.results_dir / f"task_result_{node_id_to_slug(task_id)}.md"

    def attempt_path(self, task_id: str, attempt: int) -> Path:
        """Where a superseded result is kept once its task is reset."""
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

    def decision_path(self, round_index: int) -> Path:
        return self.decisions_dir / f"round_{round_index}.yaml"


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


def parse_planner_decision(text: str) -> PlannerDecision:
    """Parse a planner's ``round_<n>.yaml``.

    Raises:
        HandoffError: When ``decision`` is not one of :data:`PLANNER_DECISIONS`.
    """
    data = _mapping(text)
    return PlannerDecision(
        decision=_choice(data, "decision", PLANNER_DECISIONS),
        summary=str(data.get("summary", "")).strip(),
    )


def read_handoff(path: Path, parse):
    """Read and parse ``path``, raising :class:`HandoffError` when it is absent."""
    if not path.is_file():
        raise HandoffError(f"{path} was not written")
    return parse(path.read_text(encoding="utf-8"))
