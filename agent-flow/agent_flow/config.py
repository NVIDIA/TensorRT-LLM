from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

BackendKind = Literal["claude-code", "codex"]
SessionMode = Literal["stateless", "persistent"]

CLAUDE_CODE_DEFAULT_MODEL = os.environ.get("CLAUDE_CODE_DEFAULT_MODEL", "claude-opus-5")
CODEX_DEFAULT_MODEL = os.environ.get("CODEX_DEFAULT_MODEL", "gpt-6-astra")


@dataclass(frozen=True)
class BackendConfig:
    kind: BackendKind
    model: str
    # Provider reasoning tier. ``None`` retains the backend's historical
    # maximum-effort default.
    reasoning_effort: str | None = None
    disabled_skills: tuple[str, ...] = ()
    tools: list[Any] | None = None
    # Native SDK hook configuration. Claude accepts HookMatcher callbacks.
    # Codex requires hooks to be configured and trusted in its native config;
    # per-client callback injection is rejected. Use AgentLayerConfig.required_tools
    # for the shared bounded tool-call policy on either backend.
    hooks: dict[str, Any] | None = None
    # Additional per-session MCP servers, in Claude-style STDIO/HTTP form.
    # Codex translates these into thread configuration without writing global
    # config. Unsupported transport/auth options fail during validation.
    extra_mcp_servers: dict[str, Any] | None = None
    # Working directory the backend runs the agent in. Forwarded to the
    # SDK as the session ``cwd`` (Claude Code's
    # ``ClaudeAgentOptions.cwd`` / Codex's ``ThreadStartParams.cwd``).
    # ``None`` keeps the launching process's ``Path.cwd()``. Set this when
    # an agent must operate on a repo other than the one the orchestrator
    # runs from — e.g. cwd-bound slash commands like ``/code-review`` that
    # diff the session cwd's git HEAD and must target the task's repo,
    # not the framework repo.
    cwd: Path | None = None


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
    # Tool names this layer must not be able to call, on top of whatever
    # the layer bans for its own reasons.
    #
    # Needed because the Claude Code backend runs with
    # ``permission_mode="bypassPermissions"`` and the sandbox off — the
    # right default for a layer whose whole job is to edit a checkout and
    # run benchmarks, and the wrong one for a layer whose INPUT is
    # untrusted. A layer that reads text a stranger pasted into a web form
    # and only has to produce YAML should not also be able to run ``Bash``
    # on the host: the paste would be a path to arbitrary execution.
    #
    # Names are the tool names the model sees (``Bash``, ``Write``,
    # ``WebFetch``, ...). Empty means "no extra bans", which is the
    # historical behaviour for every existing layer.
    disallowed_tools: tuple[str, ...] = ()
    # Every listed tool must be attempted by the root agent during one
    # layer invocation. Missing calls trigger one corrective SDK turn in
    # the same session; failure after that raises RequiredToolCallError.
    # Calls from previous invocations or subagents do not satisfy this.
    required_tools: tuple[str, ...] = ()
    # Called once per backend event as ``on_activity(kind, event)``, where
    # ``kind`` is a short tag (``tool``, ``text``, ``thinking``, ...).
    #
    # Separate from ``print_activity``, which writes to a console. A layer
    # driven by something with no console — a web request waiting on a
    # spinner, a queue worker — needs the same events somewhere it can
    # forward them from, and turning printing on would not give it that.
    #
    # Exceptions raised by the callback are swallowed: it is an observer,
    # and an observer must not be able to fail the run it is watching.
    on_activity: Any | None = None
