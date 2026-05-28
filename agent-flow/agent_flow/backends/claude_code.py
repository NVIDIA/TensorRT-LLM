from __future__ import annotations

import shutil
import subprocess
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, AsyncIterator

import claude_agent_sdk
from claude_agent_sdk import (ClaudeAgentOptions, ClaudeSDKClient,
                              PermissionResultAllow, create_sdk_mcp_server)
from claude_agent_sdk.types import (AssistantMessage, RateLimitEvent,
                                    ResultMessage, ServerToolUseBlock,
                                    SystemMessage, SystemPromptPreset,
                                    TextBlock, ThinkingBlock, ToolsPreset,
                                    ToolUseBlock)

from ..types import (AgentTextEvent, CompactBoundaryEvent,
                     RateLimitWarningEvent, ServerToolCallEvent,
                     SessionInitEvent, ThinkingEvent, ToolCallEvent, UsageInfo)
from .base import Backend, BackendClient, BackendEvent, ResultEvent


def _usage_from_result_message(sdk_message: Any) -> UsageInfo:
    raw = sdk_message.usage or {}

    def _get(*keys: str) -> int | None:
        for key in keys:
            value = raw.get(key)
            if value is not None:
                return value
        return None

    input_tokens = _get("input_tokens")
    output_tokens = _get("output_tokens")
    cache_creation = _get("cache_creation_input_tokens", "cache_creation")
    cache_read = _get("cache_read_input_tokens", "cache_read")

    total = None
    parts = [
        t for t in (input_tokens, output_tokens, cache_creation, cache_read)
        if t is not None
    ]
    if parts:
        total = sum(parts)

    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
        total_tokens=total,
        cost_usd=getattr(sdk_message, "total_cost_usd", None),
        num_turns=getattr(sdk_message, "num_turns", None),
        duration_ms=getattr(sdk_message, "duration_ms", None),
    )


async def _fetch_context_usage(sdk_client: Any) -> dict[str, Any] | None:
    getter = getattr(sdk_client, "get_context_usage", None)
    if getter is None:
        return None
    try:
        return await getter()
    except Exception:
        return None


def _apply_context_usage(usage: UsageInfo,
                         context: dict[str, Any] | None) -> UsageInfo:
    if not context:
        return usage
    usage.context_tokens = context.get("totalTokens")
    usage.context_window = context.get("maxTokens")
    percentage = context.get("percentage")
    if percentage is None and usage.context_tokens is not None and usage.context_window:
        percentage = 100.0 * usage.context_tokens / usage.context_window
    usage.context_percentage = percentage
    return usage


# Tool names that Claude Code uses to spawn a subagent. The CLI surfaces
# the tool as ``Agent``; older transcripts and the public docs sometimes
# call it ``Task``. We accept either, and fall back to a heuristic on
# ``subagent_type`` so future renames don't silently break labelling.
_SUBAGENT_SPAWN_TOOL_NAMES = frozenset({"Agent", "Task"})


def _is_subagent_spawn(block: ToolUseBlock) -> bool:
    if block.name in _SUBAGENT_SPAWN_TOOL_NAMES:
        return True
    return (isinstance(block.input, dict)
            and isinstance(block.input.get("subagent_type"), str))


def _session_init_event_from_data(data: dict[str, Any]) -> SessionInitEvent:
    """Build a ``SessionInitEvent`` from an ``init`` SystemMessage payload.

    The SDK ships ``skills``/``agents`` as ``list[str]`` and ``plugins``
    as a list of dicts (``{"name", "path", "source"}``); we keep just
    the ``name`` field of each plugin so the surfaced view stays compact
    and stable across SDK versions.
    """
    skills_raw = data.get("skills") or []
    agents_raw = data.get("agents") or []
    plugins_raw = data.get("plugins") or []
    skills = [s for s in skills_raw if isinstance(s, str)]
    agents = [a for a in agents_raw if isinstance(a, str)]
    plugins = [
        p["name"] for p in plugins_raw
        if isinstance(p, dict) and isinstance(p.get("name"), str)
    ]
    return SessionInitEvent(skills=skills, plugins=plugins, agents=agents)


def _compact_boundary_event_from_data(
        data: dict[str, Any]) -> CompactBoundaryEvent:
    """Build a ``CompactBoundaryEvent`` from a ``compact_boundary`` payload.

    The SDK doesn't model the payload as a typed subclass, so we read
    fields defensively. ``trigger`` typically reports why compaction
    fired (e.g. ``"auto"``); ``pre_tokens`` carries the token count
    immediately before compaction when reported.
    """
    trigger = data.get("trigger")
    if not isinstance(trigger, str):
        trigger = None
    pre_tokens = data.get("pre_tokens")
    if not isinstance(pre_tokens, int):
        pre_tokens = None
    return CompactBoundaryEvent(trigger=trigger, pre_tokens=pre_tokens)


def _rate_limit_warning_from_event(
        event: RateLimitEvent) -> RateLimitWarningEvent:
    info = event.rate_limit_info
    return RateLimitWarningEvent(
        status=info.status,
        rate_limit_type=info.rate_limit_type,
        resets_at=info.resets_at,
        utilization=info.utilization,
    )


def _subagent_label_from_task_input(tool_input: dict[str, Any]) -> str:
    """Pick a human-readable label for a subagent-spawning tool call.

    Prefer ``subagent_type`` (e.g. ``"Explore"``) — stripping any plugin
    namespace prefix so ``"trtllm-agent-toolkit:exec-compile-specialist"``
    renders as ``"exec-compile-specialist"``. Fall back to the short
    ``description`` field, then to a generic ``"subagent"``.
    """
    if not isinstance(tool_input, dict):
        return "subagent"
    subagent_type = tool_input.get("subagent_type")
    if isinstance(subagent_type, str) and subagent_type.strip():
        return subagent_type.strip().rsplit(":", 1)[-1]
    description = tool_input.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return "subagent"


class ClaudeCodeClient(BackendClient):

    def __init__(self, sdk_client) -> None:
        self._client = sdk_client
        # Maps a subagent-spawning ToolUseBlock id (Agent/Task) to a
        # human-readable label (subagent_type or description). Child
        # messages reference it via ``parent_tool_use_id`` so we can
        # render them under that label.
        self._subagent_labels: dict[str, str] = {}

    def _resolve_label(self, parent_id: str | None) -> str | None:
        if parent_id is None:
            return None
        return self._subagent_labels.get(parent_id, "subagent")

    async def send_message(self, message: str) -> AsyncIterator[BackendEvent]:
        got_result = False
        pending_result: ResultEvent | None = None
        try:
            await self._client.query(message)
            async for sdk_message in self._client.receive_response():
                if isinstance(sdk_message, SystemMessage):
                    if sdk_message.subtype == "init":
                        yield _session_init_event_from_data(sdk_message.data)
                    elif sdk_message.subtype == "compact_boundary":
                        yield _compact_boundary_event_from_data(
                            sdk_message.data)
                elif isinstance(sdk_message, RateLimitEvent):
                    yield _rate_limit_warning_from_event(sdk_message)
                elif isinstance(sdk_message, AssistantMessage):
                    if sdk_message.error is not None:
                        raise RuntimeError(
                            f"Claude Code turn failed: {sdk_message.error}")
                    parent_id = sdk_message.parent_tool_use_id
                    label = self._resolve_label(parent_id)
                    for block in sdk_message.content:
                        if isinstance(block, ToolUseBlock):
                            if _is_subagent_spawn(block):
                                self._subagent_labels[block.id] = (
                                    _subagent_label_from_task_input(
                                        block.input))
                            yield ToolCallEvent(
                                name=block.name,
                                input=block.input,
                                tool_use_id=block.id,
                                parent_tool_use_id=parent_id,
                                agent_label=label,
                            )
                        elif isinstance(block, ServerToolUseBlock):
                            yield ServerToolCallEvent(
                                name=block.name,
                                input=block.input,
                                tool_use_id=block.id,
                                parent_tool_use_id=parent_id,
                                agent_label=label,
                            )
                        elif isinstance(block, ThinkingBlock):
                            text = (block.thinking or "").strip()
                            if text:
                                yield ThinkingEvent(
                                    text=text,
                                    parent_tool_use_id=parent_id,
                                    agent_label=label,
                                )
                        elif isinstance(block, TextBlock):
                            text = (block.text or "").strip()
                            if text:
                                yield AgentTextEvent(
                                    text=text,
                                    parent_tool_use_id=parent_id,
                                    agent_label=label,
                                )
                elif isinstance(sdk_message, ResultMessage):
                    got_result = True
                    pending_result = ResultEvent(
                        text=sdk_message.result or "",
                        usage=_usage_from_result_message(sdk_message),
                        is_error=bool(sdk_message.is_error),
                        errors=list(sdk_message.errors or []),
                        permission_denials=list(sdk_message.permission_denials
                                                or []),
                    )
        except Exception:
            if not got_result:
                raise

        if pending_result is not None:
            context = await _fetch_context_usage(self._client)
            if pending_result.usage is not None:
                _apply_context_usage(pending_result.usage, context)
            yield pending_result


_CLI_VERSION_TIMEOUT_S = 5.0
_VERSION_CACHE: str | None = None


def _claude_sdk_version() -> str:
    """Version of the Python ``claude-agent-sdk`` package, or ``""``."""
    try:
        return _pkg_version("claude-agent-sdk")
    except PackageNotFoundError:
        return ""


def _find_claude_cli() -> str | None:
    """Locate the ``claude`` CLI binary the SDK would actually launch.

    Mirrors the SDK's lookup order — bundled binary first, then ``PATH``.
    Returns ``None`` when nothing is found so the caller can simply omit
    the CLI version from the rendered string.
    """
    bundled = (Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude")
    if bundled.is_file():
        return str(bundled)
    return shutil.which("claude")


def _claude_cli_version() -> str:
    """Run ``claude --version`` and return just the version token.

    Returns ``""`` when the binary is missing, fails to execute, or its
    output cannot be parsed. The actual stdout looks like
    ``"2.1.123 (Claude Code)"``.
    """
    cli_path = _find_claude_cli()
    if cli_path is None:
        return ""
    try:
        out = subprocess.run([cli_path, "--version"],
                             capture_output=True,
                             text=True,
                             timeout=_CLI_VERSION_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    parts = out.stdout.strip().split()
    return parts[0] if parts else ""


def _format_version(cli: str, sdk: str) -> str:
    if cli and sdk:
        return f"cli {cli} · sdk {sdk}"
    if cli:
        return f"cli {cli}"
    if sdk:
        return f"sdk {sdk}"
    return ""


def _claude_backend_version() -> str:
    """Cached ``cli X · sdk Y`` string for the Claude Code backend."""
    global _VERSION_CACHE
    if _VERSION_CACHE is None:
        _VERSION_CACHE = _format_version(_claude_cli_version(),
                                         _claude_sdk_version())
    return _VERSION_CACHE


_REASONING_EFFORT = "max"


class ClaudeCodeBackend(Backend):

    def version(self) -> str:
        return _claude_backend_version()

    def reasoning_effort(self) -> str:
        return _REASONING_EFFORT

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
        # External MCP server configs (e.g. ``{"Glean": {"type": "http",
        # "url": ...}}``) are layered alongside the in-process
        # ``agent-tools`` server built from ``tools``. The keys are the
        # server names the model sees and must therefore not collide with
        # ``agent-tools``.
        mcp_servers: dict[str, object] = dict(extra_mcp_servers or {})
        if "agent-tools" in mcp_servers:
            raise ValueError(
                "extra_mcp_servers must not contain a key named "
                "'agent-tools'; that name is reserved for the in-process "
                "MCP server that exposes the BackendConfig.tools list.")
        if tools:
            mcp_servers["agent-tools"] = create_sdk_mcp_server(
                name="agent-tools",
                tools=tools,
            )

        async def _approve_tool(tool_name, tool_input, context):
            return PermissionResultAllow()

        options = ClaudeAgentOptions(
            tools=ToolsPreset(type="tools_preset", preset="claude_code"),
            system_prompt=SystemPromptPreset(
                type="preset",
                preset="claude_code",
                append=system_prompt,
            ),
            mcp_servers=mcp_servers,
            model=model,
            effort=_REASONING_EFFORT,
            cwd=Path.cwd(),
            sandbox={"enabled": False},
            permission_mode="bypassPermissions",
            can_use_tool=_approve_tool,
            hooks=hooks,
            disallowed_tools=list(disallowed_tools or []),
        )

        async with ClaudeSDKClient(options=options) as sdk_client:
            yield ClaudeCodeClient(sdk_client)
