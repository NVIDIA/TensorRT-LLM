# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-neutral tools used by agent-flow workflows."""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class WorkflowTool:
    """A workflow-owned tool that either backend can expose."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], Awaitable[str]]
    required_before_stop: bool = False


def workflow_tool(
    name: str,
    description: str,
    input_schema: Mapping[str, Any],
    *,
    required_before_stop: bool = False,
) -> Callable[
    [Callable[[Mapping[str, Any]], Awaitable[str]]],
    WorkflowTool,
]:
    """Decorate a handler as an SDK-neutral workflow tool."""

    def decorate(handler: Callable[[Mapping[str, Any]], Awaitable[str]]) -> WorkflowTool:
        return WorkflowTool(name, description, input_schema, handler, required_before_stop)

    return decorate


class ToolCompletion:
    """Track required workflow tools completed in the current turn."""

    def __init__(self, required: set[str], state_path: Path | None = None) -> None:
        self.required = required
        self.completed: set[str] = set()
        self.state_path = state_path
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.completed.clear()
            self._write()

    def mark(self, name: str) -> None:
        with self._lock:
            self.completed.add(name)
            self._write()

    def missing(self) -> set[str]:
        with self._lock:
            return self.required - self.completed

    def _write(self) -> None:
        if self.state_path is not None:
            self.state_path.write_text(
                json.dumps(
                    {"required": sorted(self.required), "completed": sorted(self.completed)}
                ),
                encoding="utf-8",
            )


def required_tools(tools: list[Any] | None) -> set[str]:
    """Return the workflow tools that must finish before a turn stops."""
    return {
        tool.name
        for tool in tools or []
        if isinstance(tool, WorkflowTool) and tool.required_before_stop
    }


def run_stop_hook(state_path: Path) -> None:
    """Implement the Codex command Stop hook for a completion state file."""
    hook_input = json.load(sys.stdin)
    if hook_input.get("stop_hook_active"):
        print("{}")
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    missing = set(state["required"]) - set(state["completed"])
    if missing:
        names = ", ".join(f"`{name}`" for name in sorted(missing))
        print(json.dumps({"decision": "block", "reason": f"Call {names} before stopping."}))
    else:
        print("{}")


if __name__ == "__main__":
    run_stop_hook(Path(sys.argv[1]))
