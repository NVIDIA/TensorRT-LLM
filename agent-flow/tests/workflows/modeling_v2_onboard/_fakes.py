# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted agents that stand in for real backends."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

_RESULT_RE = re.compile(r"result file (?:to )?`([^`]+)`")


def result_path_in(prompt: str) -> Path:
    """The result file a worker prompt asks for."""
    match = _RESULT_RE.search(prompt)
    assert match, prompt
    return Path(match.group(1))


def write_result(path: Path, status: str, summary: str = "ok") -> None:
    path.write_text(f"---\nstatus: {status}\nsummary: {summary}\n---\n\nreport\n")


@dataclass
class FakeAgent:
    name: str
    system_prompt: str
    cwd: Path
    session_mode: str
    script: Callable[[FakeAgent, str], Awaitable[None]]
    fork_from: str | None = None
    prompts: list[str] = field(default_factory=list)
    closed: bool = False
    session_id: str | None = None

    async def aforward(self, content: str) -> str:
        self.prompts.append(content)
        await self.script(self, content)
        return ""

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.closed = True


@dataclass
class FakeFactory:
    """Build :class:`FakeAgent`s, routing planner and worker turns to scripts."""

    planner: Callable[[FakeAgent, str], Awaitable[None]] | None = None
    worker: Callable[[FakeAgent, str], Awaitable[None]] | None = None
    primer: Callable[[FakeAgent, str], Awaitable[None]] | None = None
    agents: list[FakeAgent] = field(default_factory=list)

    def __call__(
        self,
        name: str,
        system_prompt: str,
        cwd: Path,
        session_mode: str,
        fork_from: str | None = None,
    ) -> FakeAgent:
        if name == "planner":
            script = self.planner
        elif name.startswith("primer["):
            script = self.primer
        else:
            script = self.worker
        assert script is not None, name
        agent = FakeAgent(name, system_prompt, Path(cwd), session_mode, script, fork_from)
        self.agents.append(agent)
        return agent

    def workers(self, task_id: str) -> list[FakeAgent]:
        return [a for a in self.agents if a.name == f"worker[{task_id}]"]
