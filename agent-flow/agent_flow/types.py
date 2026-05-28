from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    content: str
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageInfo:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    context_percentage: float | None = None


@dataclass
class AgentResponse:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: UsageInfo | None = None


@dataclass
class ToolCallEvent:
    name: str
    input: dict[str, Any]
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None
    agent_label: str | None = None


@dataclass
class ServerToolCallEvent:
    """Built-in server-side tool invocation (web_search, web_fetch, ...).

    The Anthropic API runs these on the model's behalf, so the call
    appears in the assistant stream but the harness never has to feed
    a result back. Surfaced as its own event (rather than ``ToolCallEvent``)
    so consumers can treat it differently — typically as informational.
    """
    name: str
    input: dict[str, Any]
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None
    agent_label: str | None = None


@dataclass
class ThinkingEvent:
    """Extended-thinking output emitted by the model.

    Surfaced separately from ``AgentTextEvent`` so a UI can collapse
    or dim it without affecting the user-facing message stream.
    """
    text: str
    parent_tool_use_id: str | None = None
    agent_label: str | None = None


@dataclass
class AgentTextEvent:
    text: str
    parent_tool_use_id: str | None = None
    agent_label: str | None = None


@dataclass
class SessionInitEvent:
    """Skills, plugins, and subagents loaded into a fresh session.

    Backends emit this once per session, before any agent text or tool
    calls, so callers can see *what was available* to the model — not
    just what it ended up using. Lists are passed through verbatim from
    the underlying SDK; alias entries (same skill exposed under multiple
    plugin prefixes) are intentionally preserved so the surfaced view
    matches what the model itself sees.
    """
    skills: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)


@dataclass
class RateLimitWarningEvent:
    """Rate-limit status transition reported by the backend.

    Emitted when the upstream service flips between ``allowed`` /
    ``allowed_warning`` / ``rejected``. Consumers can use this to warn
    users before a hard cap or to back off gracefully.
    """
    status: str
    rate_limit_type: str | None = None
    resets_at: int | None = None
    utilization: float | None = None


@dataclass
class CompactBoundaryEvent:
    """Conversation was auto-compacted by the backend.

    Token counts in subsequent ``ResultEvent.usage`` reflect the
    post-compaction context, so prior usage figures should not be
    compared directly across this boundary.
    """
    trigger: str | None = None
    pre_tokens: int | None = None
