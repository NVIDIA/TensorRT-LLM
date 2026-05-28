from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

BackendKind = Literal["claude-code", "codex"]
SessionMode = Literal["stateless", "persistent"]

CLAUDE_CODE_DEFAULT_MODEL = os.environ.get("CLAUDE_CODE_DEFAULT_MODEL",
                                           "claude-opus-4-7[1m]")
CODEX_DEFAULT_MODEL = os.environ.get("CODEX_DEFAULT_MODEL", "gpt-5.5")


@dataclass(frozen=True)
class BackendConfig:
    kind: BackendKind
    model: str
    tools: list[Any] | None = None
    # Backend-specific hook configuration. Currently only the ``claude-code``
    # backend consumes this — it is forwarded verbatim to the Claude Agent
    # SDK's ``ClaudeAgentOptions(hooks=...)`` (a dict keyed by event name,
    # values are lists of ``HookMatcher``). Other backends ignore it.
    hooks: dict[str, Any] | None = None
    # Extra MCP servers to make available to the agent, keyed by the
    # server name the model sees (tools become ``mcp__<name>__<tool>``).
    # Currently only the ``claude-code`` backend consumes this — values
    # are forwarded verbatim into ``ClaudeAgentOptions.mcp_servers``
    # alongside the in-process ``agent-tools`` server built from
    # ``tools``. Typical use: wiring an HTTP MCP server (Glean, etc.)
    # into a single agent's session via
    # ``{"Glean": {"type": "http", "url": "..."}}``. Other backends
    # accept and ignore the field.
    extra_mcp_servers: dict[str, Any] | None = None


@dataclass(frozen=True)
class SessionConfig:
    mode: SessionMode = "stateless"


@dataclass(frozen=True)
class HumanRequestOption:
    """Internal: a single choice rendered when the agent calls ``ask_human``."""

    label: str
    description: str = ""


@dataclass(frozen=True)
class HumanRequest:
    """Internal payload passed from the ``ask_human`` MCP tool to the renderer."""

    layer_name: str
    prompt: str
    options: tuple[HumanRequestOption, ...] = ()
    # Short chip-style label (≤12 chars) the agent can attach to the
    # question, mirroring Claude Code's ``AskUserQuestion`` ``header``
    # field. Empty string means "no chip"; the renderer suppresses it.
    header: str = ""


@dataclass(frozen=True)
class AgentLayerConfig:
    backend: BackendConfig
    session: SessionConfig = field(default_factory=SessionConfig)
    system_prompt: str | None = None
    name: str | None = None
    print_activity: bool = True
    # When True, the layer registers an ``ask_human`` MCP tool the agent
    # can call mid-turn to ask the human a question, and disables Claude
    # Code's built-in ``AskUserQuestion`` so the agent's questions reach
    # the human via stdin instead of being silently auto-defaulted.
    human_input_enabled: bool = False
