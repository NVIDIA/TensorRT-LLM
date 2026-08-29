from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from ..types import (
    AgentTextEvent,
    CompactBoundaryEvent,
    RateLimitWarningEvent,
    ServerToolCallEvent,
    SessionInitEvent,
    ThinkingEvent,
    ToolCallEvent,
    UsageInfo,
)


@dataclass
class ResultEvent:
    text: str
    usage: UsageInfo | None = None
    is_error: bool = False
    errors: list[str] = field(default_factory=list)
    permission_denials: list[Any] = field(default_factory=list)


BackendEvent = (
    ToolCallEvent
    | ServerToolCallEvent
    | ThinkingEvent
    | AgentTextEvent
    | SessionInitEvent
    | RateLimitWarningEvent
    | CompactBoundaryEvent
    | ResultEvent
)


class BackendClient(ABC):
    @abstractmethod
    def send_message(self, message: str) -> AsyncIterator[BackendEvent]:
        raise NotImplementedError

    async def get_context_usage(self) -> UsageInfo | None:
        """Pre-input context footprint for this client's session, if known.

        Returns the context usage (tokens / window / percentage) the
        backend reports *before any user message is sent* — i.e. the
        baseline cost of the system prompt, tools, and memory. Backends
        that cannot report this before a turn return ``None``; the default
        does so. ``ClaudeCodeClient`` overrides it with the SDK's local
        ``/context`` control request.
        """
        return None

    async def list_available_skills(self) -> list[str] | None:
        """Skill names this client's session can invoke, if knowable.

        Answered **without sending a turn**: both concrete backends
        already hold the answer by the time the client is connected —
        Claude Code from the CLI's initialize response, Codex from the
        ``skills_list`` RPC its client issues at creation — so a caller
        that only wants to know what is installed never has to pay for a
        model call to find out.

        ``None`` means *the backend could not say*, which is not the same
        as an empty list: callers must not read it as "no skills are
        installed". The default returns ``None``.
        """
        return None


class Backend(ABC):
    async def __aenter__(self) -> "Backend":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def version(self) -> str:
        """Short, human-readable version string for this backend.

        Used by ``print_agent_started`` to log which Claude Code or Codex
        build is actually driving the agent. The default is an empty
        string (no version surfaced); concrete backends override.
        """
        return ""

    def reasoning_effort(self) -> str:
        """Reasoning-effort setting the backend will pass to the model.

        Used by ``print_agent_started`` to surface the effort tier
        alongside the backend and model. Returning an empty string omits
        the field from the rendered panel.
        """
        return ""

    @abstractmethod
    @asynccontextmanager
    async def create_client(
        self,
        system_prompt: str,
        model: str,
        tools: list | None = None,
        hooks: dict | None = None,
        disallowed_tools: list[str] | None = None,
        extra_mcp_servers: dict[str, Any] | None = None,
        cwd: Path | None = None,
    ) -> AsyncIterator[BackendClient]:
        yield
