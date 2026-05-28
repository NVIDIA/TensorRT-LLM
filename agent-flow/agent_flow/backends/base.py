from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ..types import (AgentTextEvent, CompactBoundaryEvent,
                     RateLimitWarningEvent, ServerToolCallEvent,
                     SessionInitEvent, ThinkingEvent, ToolCallEvent, UsageInfo)


@dataclass
class ResultEvent:
    text: str
    usage: UsageInfo | None = None
    is_error: bool = False
    errors: list[str] = field(default_factory=list)
    permission_denials: list[Any] = field(default_factory=list)


BackendEvent = (ToolCallEvent | ServerToolCallEvent | ThinkingEvent
                | AgentTextEvent | SessionInitEvent | RateLimitWarningEvent
                | CompactBoundaryEvent | ResultEvent)


class BackendClient(ABC):

    @abstractmethod
    def send_message(self, message: str) -> AsyncIterator[BackendEvent]:
        raise NotImplementedError


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
    ) -> AsyncIterator[BackendClient]:
        yield
