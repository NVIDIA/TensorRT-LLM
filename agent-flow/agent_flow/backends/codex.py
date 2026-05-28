from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from ..types import (AgentTextEvent, CompactBoundaryEvent,
                     RateLimitWarningEvent, ServerToolCallEvent,
                     SessionInitEvent, ThinkingEvent, ToolCallEvent, UsageInfo)
from .base import Backend, BackendClient, BackendEvent, ResultEvent

# Dispatch table keyed by thread id. Populated when a CodexClient is created
# with dynamic tools, drained when it is torn down. The AppServerClient's
# approval handler reads this to route ``item/tool/call`` server requests to
# the right handler without having to know about threads.
_TOOL_HANDLERS: dict[str, dict[str, Callable[[dict[str, Any]],
                                             Awaitable[dict[str, Any]]]]] = {}


def _module_or_none(name: str) -> Any | None:
    module = sys.modules.get(name)
    if module is not None:
        return module
    if name in sys.modules:
        return None
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


def _module(name: str) -> Any:
    module = sys.modules.get(name)
    if module is not None:
        return module
    return importlib.import_module(name)


def _symbol(module_name: str, symbol_name: str) -> Any:
    return getattr(_module(module_name), symbol_name)


def _resolve_codex_bin() -> str:
    override = os.environ.get("CODEX_BIN")
    if override:
        if not Path(override).is_file():
            raise FileNotFoundError(
                f"CODEX_BIN={override!r} does not point to a file.")
        return override

    codex_cli_bin = _module_or_none("codex_cli_bin")
    bundled_codex_path = getattr(codex_cli_bin, "bundled_codex_path",
                                 None) if codex_cli_bin is not None else None
    if bundled_codex_path is not None:
        return str(bundled_codex_path())

    system_bin = shutil.which("codex")
    if system_bin is not None:
        return system_bin

    raise FileNotFoundError(
        "Could not locate a Codex CLI binary. Install the pinned runtime "
        "(the 'openai-codex-cli-bin' package shipped with "
        "'openai-codex-app-server-sdk'), install the Codex CLI on PATH, or "
        "set CODEX_BIN to the binary path.")


def _usage_from_token_breakdown(breakdown: Any) -> UsageInfo:
    return UsageInfo(
        input_tokens=getattr(breakdown, "input_tokens", None),
        output_tokens=getattr(breakdown, "output_tokens", None),
        cache_read_tokens=getattr(breakdown, "cached_input_tokens", None),
        total_tokens=getattr(breakdown, "total_tokens", None),
    )


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if not isinstance(value, dict) and hasattr(value, name):
            return getattr(value, name)
    return default


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(by_alias=True, exclude_none=True)
        except TypeError:
            return model_dump()

    if hasattr(value, "__dict__"):
        return {
            k: _plain(v)
            for k, v in vars(value).items()
            if not k.startswith("_") and v is not None
        }

    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    return str(value)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        value = enum_value
    return str(value)


def _root(item: Any) -> Any:
    return item.root if hasattr(item, "root") else item


def _item_id(item: Any, root: Any | None = None) -> str | None:
    root = root if root is not None else _root(item)
    value = _get(root, "id", default=_get(item, "id"))
    return value if isinstance(value, str) else None


_THREAD_ITEM_KINDS = {
    "AgentMessageThreadItem": "agentMessage",
    "CollabAgentToolCallThreadItem": "collabAgentToolCall",
    "CollabToolCallThreadItem": "collabAgentToolCall",
    "CommandExecutionThreadItem": "commandExecution",
    "ContextCompactionThreadItem": "contextCompaction",
    "CompactedThreadItem": "compacted",
    "DynamicToolCallThreadItem": "dynamicToolCall",
    "FileChangeThreadItem": "fileChange",
    "ImageViewThreadItem": "imageView",
    "McpToolCallThreadItem": "mcpToolCall",
    "PlanThreadItem": "plan",
    "ReasoningThreadItem": "reasoning",
    "WebSearchThreadItem": "webSearch",
}


def _thread_item_kind(root: Any) -> str:
    raw = _get(root, "type", "kind")
    if raw is not None:
        return _as_text(raw)
    return _THREAD_ITEM_KINDS.get(root.__class__.__name__, "")


def _notification_kind(payload: Any) -> str:
    raw = _get(payload, "method", "type", "kind")
    if raw is not None:
        return _as_text(raw)

    name = payload.__class__.__name__
    if name == "ItemCompletedNotification":
        return "item/completed"
    if name == "ItemStartedNotification":
        return "item/started"
    if name == "ThreadTokenUsageUpdatedNotification":
        return "thread/tokenUsage/updated"
    if name == "TurnCompletedNotification":
        return "turn/completed"
    if "AgentMessage" in name and "Delta" in name:
        return "item/agentMessage/delta"
    if "Reasoning" in name and "Delta" in name:
        return "item/reasoning/delta"
    if name in {"WarningNotification", "ConfigWarningNotification"}:
        return "warning"
    if name == "ErrorNotification":
        return "error"
    return name


def _arguments(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reasoning_text(root: Any) -> str:
    parts: list[str] = []
    for field in ("summary", "content", "text"):
        value = _get(root, field)
        if isinstance(value, (list, tuple)):
            parts.extend(
                str(part).strip() for part in value if str(part).strip())
        elif isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts).strip()


def _collab_tool_input(root: Any) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for source, target in (
        ("prompt", "prompt"),
        ("model", "model"),
        ("reasoning_effort", "reasoning_effort"),
        ("reasoningEffort", "reasoning_effort"),
        ("sender_thread_id", "sender_thread_id"),
        ("senderThreadId", "sender_thread_id"),
        ("receiver_thread_ids", "receiver_thread_ids"),
        ("receiverThreadIds", "receiver_thread_ids"),
        ("agents_states", "agents_states"),
        ("agentsStates", "agents_states"),
        ("status", "status"),
    ):
        value = _get(root, source)
        if value is not None:
            body[target] = _plain(value)
    return body


def _tool_event_from_item(root: Any, item: Any) -> BackendEvent | None:
    kind = _thread_item_kind(root)
    tool_use_id = _item_id(item, root)
    if kind == "commandExecution":
        command = _get(root, "command", default="")
        return ToolCallEvent(name="Bash",
                             input={"command": command},
                             tool_use_id=tool_use_id)

    if kind in {"mcpToolCall", "dynamicToolCall"}:
        name = _get(root, "tool", default="tool")
        return ToolCallEvent(
            name=_as_text(name),
            input=_arguments(_get(root, "arguments")),
            tool_use_id=tool_use_id,
        )

    if kind in {"collabAgentToolCall", "collabToolCall"}:
        name = _get(root, "tool", default="collab_agent")
        return ToolCallEvent(
            name=_as_text(name),
            input=_collab_tool_input(root),
            tool_use_id=tool_use_id,
        )

    if kind == "fileChange":
        body = {"changes": _plain(_get(root, "changes", default=[]))}
        status = _get(root, "status")
        if status is not None:
            body["status"] = _as_text(status)
        return ToolCallEvent(name="FileChange",
                             input=body,
                             tool_use_id=tool_use_id)

    if kind == "webSearch":
        action = _get(root, "action")
        if action is not None:
            body = _plain(action)
            if not isinstance(body, dict):
                body = {"action": body}
        else:
            body = {}
        query = _get(root, "query")
        if isinstance(query, str):
            body.setdefault("query", query)
        return ServerToolCallEvent(name="web_search",
                                   input=body,
                                   tool_use_id=tool_use_id)

    if kind == "imageView":
        path = _get(root, "path")
        body = {"path": path} if isinstance(path, str) else {}
        return ServerToolCallEvent(name="view_image",
                                   input=body,
                                   tool_use_id=tool_use_id)

    return None


def _status_is(root: Any, *statuses: str) -> bool:
    status = _get(root, "status")
    return _as_text(status) in statuses


def _collect_item_errors(root: Any, item: Any, errors: list[str],
                         permission_denials: list[Any]) -> None:
    kind = _thread_item_kind(root)
    if kind not in {
            "collabAgentToolCall", "collabToolCall", "commandExecution",
            "dynamicToolCall", "fileChange", "mcpToolCall"
    }:
        return

    if _status_is(root, "declined"):
        permission_denials.append({
            "kind": kind,
            "id": _item_id(item, root),
            "item": _plain(root),
        })
        return

    if _status_is(root, "failed"):
        error = _get(root, "error")
        if error is None:
            errors.append(f"{kind} failed")
        else:
            errors.append(_as_text(_plain(error)))


def _compact_boundary_from_item(root: Any) -> CompactBoundaryEvent:
    trigger = _get(root, "trigger")
    if not isinstance(trigger, str):
        trigger = None
    pre_tokens = _get(root, "pre_tokens", "preTokens")
    if not isinstance(pre_tokens, int):
        pre_tokens = None
    return CompactBoundaryEvent(trigger=trigger, pre_tokens=pre_tokens)


def _rate_limit_warning_from_error(error: Any) -> RateLimitWarningEvent | None:
    info = _get(error, "codex_error_info", "codexErrorInfo")
    message = _as_text(_get(error, "message", default=error))
    haystack = f"{_as_text(info)} {message}".lower()
    if "usagelimit" not in haystack and "rate limit" not in haystack:
        return None
    return RateLimitWarningEvent(status="rejected",
                                 rate_limit_type=_as_text(info) or None)


def _is_final_answer_phase(phase: Any, final_answer: Any) -> bool:
    value = _as_text(phase)
    return value in {"final_answer", "finalAnswer"
                     } or value == _as_text(final_answer)


def _extract_final_response(items: list[object]) -> str:
    MessagePhase = _symbol("codex_app_server.generated.v2_all", "MessagePhase")

    last_unknown: str | None = None
    for item in reversed(items):
        root = _root(item)
        if _thread_item_kind(root) == "agentMessage":
            if _is_final_answer_phase(_get(root, "phase"),
                                      MessagePhase.final_answer):
                return root.text
            if _get(root, "phase") is None and last_unknown is None:
                last_unknown = root.text
    return last_unknown or ""


def _turn_completed_successfully(turn: Any) -> bool:
    status = _get(turn, "status")
    return status is None or _as_text(status) == "completed"


def _turn_failure_message(turn: Any) -> str:
    error = _get(turn, "error")
    if error is not None:
        return _as_text(_get(error, "message", default=error))
    status = _as_text(_get(turn, "status", default="unknown"))
    return f"turn ended with status {status!r}"


class CodexClient(BackendClient):

    def __init__(self,
                 thread,
                 session_init: SessionInitEvent | None = None) -> None:
        self._thread = thread
        self._session_init = session_init
        self._session_init_emitted = False

    async def send_message(self, message: str) -> AsyncIterator[BackendEvent]:
        TextInput = _symbol("codex_app_server", "TextInput")

        turn = await self._thread.turn(TextInput(message))
        items: list[object] = []
        latest_usage: UsageInfo | None = None
        errors: list[str] = []
        permission_denials: list[Any] = []
        emitted_tool_items: set[str] = set()
        emitted_compaction_items: set[str] = set()

        if self._session_init is not None and not self._session_init_emitted:
            self._session_init_emitted = True
            yield self._session_init

        async for event in turn.stream():
            payload = event.payload
            kind = _notification_kind(payload)
            if kind in {"item/started", "item/completed"}:
                item = _get(payload, "item")
                if item is None:
                    continue
                if kind == "item/completed":
                    items.append(item)
                root = _root(item)
                item_id = _item_id(item, root)

                tool_event = _tool_event_from_item(root, item)
                if tool_event is not None and item_id not in emitted_tool_items:
                    if item_id is not None:
                        emitted_tool_items.add(item_id)
                    yield tool_event

                if _thread_item_kind(root) in {
                        "contextCompaction", "compacted"
                } and item_id not in emitted_compaction_items:
                    if item_id is not None:
                        emitted_compaction_items.add(item_id)
                    yield _compact_boundary_from_item(root)

                if kind != "item/completed":
                    continue

                _collect_item_errors(root, item, errors, permission_denials)
                if _thread_item_kind(root) == "reasoning":
                    text = _reasoning_text(root)
                    if text:
                        yield ThinkingEvent(text=text)
                elif _thread_item_kind(root) == "agentMessage":
                    text = (root.text or "").strip()
                    if text:
                        yield AgentTextEvent(text=text)
            elif kind == "item/agentMessage/delta":
                continue
            elif kind == "item/reasoning/delta":
                continue
            elif kind == "thread/tokenUsage/updated":
                total = getattr(payload.token_usage, "total", None)
                if total is not None:
                    latest_usage = _usage_from_token_breakdown(total)
                    window = getattr(payload.token_usage,
                                     "model_context_window", None)
                    if window:
                        latest_usage.context_window = window
                        latest_usage.context_tokens = getattr(
                            total, "total_tokens", None)
                        if latest_usage.context_tokens is not None:
                            latest_usage.context_percentage = (
                                100.0 * latest_usage.context_tokens / window)
            elif kind == "warning":
                warning = _rate_limit_warning_from_error(
                    _get(payload, "message", default=payload))
                if warning is not None:
                    yield warning
            elif kind == "error":
                warning = _rate_limit_warning_from_error(
                    _get(payload, "error", default=payload))
                if warning is not None:
                    yield warning
            elif kind == "turn/completed" and payload.turn.id == turn.id:
                turn_error = getattr(payload.turn, "error", None)
                if turn_error is not None:
                    warning = _rate_limit_warning_from_error(turn_error)
                    if warning is not None:
                        yield warning
                    message = getattr(turn_error, "message",
                                      None) or str(turn_error)
                    raise RuntimeError(f"Codex turn failed: {message}")
                if not _turn_completed_successfully(payload.turn):
                    raise RuntimeError(
                        f"Codex turn failed: {_turn_failure_message(payload.turn)}"
                    )
                yield ResultEvent(
                    text=_extract_final_response(items),
                    usage=latest_usage,
                    is_error=bool(errors),
                    errors=errors,
                    permission_denials=permission_denials,
                )


def _dynamic_tool_spec(tool: Any) -> dict[str, Any]:
    """Serialize an ``SdkMcpTool`` into a Codex ``DynamicToolSpec`` dict."""
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
        "deferLoading": False,
    }


def _mcp_to_codex_content(result: Any) -> tuple[list[dict[str, Any]], bool]:
    """Translate an MCP tool result to Codex ``DynamicToolCallResponse`` parts.

    Tools decorated with ``claude_agent_sdk.tool`` return ``{"content": [{type:
    "text", text: "..."}], "is_error"?: bool}``. Codex expects content items
    shaped as ``{type: "inputText", text: "..."}`` plus a top-level ``success``
    flag.
    """
    items: list[dict[str, Any]] = []
    success = True
    if isinstance(result, dict):
        for item in result.get("content") or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "text":
                items.append({
                    "type": "inputText",
                    "text": item.get("text", "")
                })
            elif kind == "image":
                url = item.get("data") or item.get("image_url") or ""
                items.append({"type": "inputImage", "imageUrl": url})
        if result.get("is_error"):
            success = False
    if not items:
        items = [{"type": "inputText", "text": ""}]
    return items, success


def _handle_dynamic_tool_call(params: dict[str, Any]) -> dict[str, Any]:
    thread_id = params.get("threadId") or ""
    tool_name = params.get("tool") or ""
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    handlers = _TOOL_HANDLERS.get(thread_id, {})
    handler = handlers.get(tool_name)
    if handler is None:
        return {
            "contentItems": [{
                "type":
                "inputText",
                "text": (f"Tool {tool_name!r} is not registered for this "
                         f"thread."),
            }],
            "success":
            False,
        }

    try:
        result = asyncio.run(handler(arguments))
    except Exception as exc:  # noqa: BLE001 — surface any tool error to model
        return {
            "contentItems": [{
                "type": "inputText",
                "text": f"Tool {tool_name!r} failed: {exc}",
            }],
            "success":
            False,
        }

    items, success = _mcp_to_codex_content(result)
    return {"contentItems": items, "success": success}


async def _call_optional_client_method(client: Any, names: tuple[str, ...],
                                       payload: dict[str, Any]) -> Any:
    for name in names:
        method = getattr(client, name, None)
        if method is None:
            continue
        result = method(payload)
        if inspect.isawaitable(result):
            return await result
        return result
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _extract_codex_skill_names(response: Any) -> list[str]:
    names: list[str] = []
    for entry in _as_list(_get(response, "data", default=response)):
        for skill in _as_list(_get(entry, "skills")):
            if _get(skill, "enabled") is False:
                continue
            name = _get(skill, "name")
            if isinstance(name, str):
                names.append(name)
    return names


def _extract_codex_plugin_names(response: Any) -> list[str]:
    names: list[str] = []

    def walk(value: Any, depth: int = 0) -> None:
        if value is None or depth > 4:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item, depth + 1)
            return

        name = _get(value, "name", "plugin_name", "pluginName")
        if isinstance(name, str):
            names.append(name)

        for key in ("plugins", "items", "entries", "data"):
            child = _get(value, key)
            if child is not None:
                walk(child, depth + 1)

    walk(response)
    return names


async def _build_session_init_event(client: Any,
                                    cwd: Path) -> SessionInitEvent | None:
    try:
        skills_response = await _call_optional_client_method(
            client,
            ("skills_list", "skill_list"),
            {
                "cwds": [str(cwd)],
                "forceReload": False
            },
        )
        plugins_response = await _call_optional_client_method(
            client, ("plugin_list", "plugins_list"), {})
    except Exception:
        return None

    event = SessionInitEvent(
        skills=_extract_codex_skill_names(skills_response),
        plugins=_extract_codex_plugin_names(plugins_response),
        agents=[],
    )
    if event.skills or event.plugins or event.agents:
        return event
    return None


_CLI_VERSION_TIMEOUT_S = 5.0
_VERSION_CACHE: str | None = None


def _codex_sdk_version() -> str:
    """Version of the Python Codex app-server SDK package, or ``""``."""
    for package in (
            "openai-codex-app-server-sdk",
            "codex-app-server-sdk",
    ):
        try:
            return _pkg_version(package)
        except PackageNotFoundError:
            continue
    return ""


def _codex_cli_version() -> str:
    """Run ``codex --version`` and return just the version token.

    Returns ``""`` when the binary is missing, fails to execute, or its
    output cannot be parsed. The actual stdout looks like
    ``"codex-cli 0.116.0-alpha.1"``.
    """
    try:
        cli_path = _resolve_codex_bin()
    except FileNotFoundError:
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
    return parts[-1] if parts else ""


def _format_version(cli: str, sdk: str) -> str:
    if cli and sdk:
        return f"cli {cli} · sdk {sdk}"
    if cli:
        return f"cli {cli}"
    if sdk:
        return f"sdk {sdk}"
    return ""


def _codex_backend_version() -> str:
    """Cached ``cli X · sdk Y`` string for the Codex backend."""
    global _VERSION_CACHE
    if _VERSION_CACHE is None:
        _VERSION_CACHE = _format_version(_codex_cli_version(),
                                         _codex_sdk_version())
    return _VERSION_CACHE


_REASONING_EFFORT = "xhigh"

_SDK_PATCHED = False


def _relax_service_tier_on_module(module: Any) -> int:
    """Rewrite ``ServiceTier`` references to ``str`` on every pydantic model
    in ``module``. Returns the number of models that were rebuilt.
    """
    OldEnum = getattr(module, "ServiceTier", None)
    if OldEnum is None:
        return 0

    import types
    from typing import Union, get_args, get_origin

    from pydantic import BaseModel

    def _replace(annotation: Any) -> Any:
        if annotation is OldEnum:
            return str
        origin = get_origin(annotation)
        if origin is None:
            return annotation
        args = get_args(annotation)
        new_args = tuple(_replace(a) for a in args)
        if new_args == args:
            return annotation
        # ``X | Y`` produces ``types.UnionType``; ``Union[X, Y]`` produces
        # ``typing.Union``. Both must be rebuilt as a typing.Union so that
        # pydantic can re-derive the schema.
        if origin is Union or origin is types.UnionType:
            return Union[new_args]  # type: ignore[valid-type]
        try:
            return origin[new_args]
        except TypeError:
            return annotation

    affected: list[type] = []
    for name in dir(module):
        cls = getattr(module, name, None)
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            continue
        changed = False
        for finfo in cls.model_fields.values():
            new_ann = _replace(finfo.annotation)
            if new_ann is not finfo.annotation:
                finfo.annotation = new_ann
                changed = True
        if changed:
            affected.append(cls)

    for cls in affected:
        try:
            cls.model_rebuild(force=True)
        except Exception:
            pass
    return len(affected)


def _patch_codex_sdk_service_tier() -> None:
    """Relax the SDK's ``ServiceTier`` enum to ``str`` on response models.

    The rust protocol (``codex-rs/app-server-protocol/src/protocol/v2/thread.rs``)
    declares ``service_tier: Option<String>`` on responses, but the Python SDK
    was generated with a strict ``ServiceTier`` enum that only contains
    ``fast`` and ``flex``. The server returns ``"priority"`` (the request value
    for the Fast tier per ``ServiceTier::request_value`` in
    ``codex-rs/protocol/src/config_types.rs``), which the strict enum rejects.
    """
    global _SDK_PATCHED
    if _SDK_PATCHED:
        return
    try:
        v2_all = importlib.import_module("codex_app_server.generated.v2_all")
    except ImportError:
        return
    _relax_service_tier_on_module(v2_all)
    _SDK_PATCHED = True


class CodexBackend(Backend):

    def __init__(self) -> None:
        self._codex = None
        self._default_approval_handler: Callable[..., dict[str,
                                                           Any]] | None = None

    def version(self) -> str:
        return _codex_backend_version()

    def reasoning_effort(self) -> str:
        return _REASONING_EFFORT

    async def __aenter__(self) -> "CodexBackend":
        _patch_codex_sdk_service_tier()
        AsyncCodex = _symbol("codex_app_server", "AsyncCodex")
        AppServerConfig = _symbol("codex_app_server.client", "AppServerConfig")

        codex_bin = _resolve_codex_bin()

        self._codex = AsyncCodex(config=AppServerConfig(
            codex_bin=codex_bin,
            experimental_api=True,
        ))
        await self._codex.__aenter__()
        self._install_server_request_handler()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._codex is not None:
            await self._codex.__aexit__(None, None, None)
            self._codex = None
            self._default_approval_handler = None

    def _install_server_request_handler(self) -> None:
        """Route ``item/tool/call`` requests to ``_handle_dynamic_tool_call``
        and auto-accept every approval request the app-server sends.

        Combined with ``approval_policy=never`` and
        ``sandbox=danger_full_access`` in ``create_client``, the agent never
        prompts the user: any ``*/requestApproval`` JSON-RPC request is
        answered with ``{"decision": "accept"}`` regardless of which item
        kind raised it (commandExecution, fileChange, or anything new the
        SDK adds later).
        """
        sync_client = self._codex._client._sync
        existing = sync_client._approval_handler
        self._default_approval_handler = existing

        def _handler(method: str,
                     params: dict[str, Any] | None) -> dict[str, Any]:
            if method == "item/tool/call":
                return _handle_dynamic_tool_call(params or {})
            if method.endswith("/requestApproval"):
                return {"decision": "accept"}
            return existing(method, params)

        sync_client._approval_handler = _handler

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
        # ``disallowed_tools`` and ``extra_mcp_servers`` are currently
        # Claude-Code-only concepts (they map onto
        # ``ClaudeAgentOptions.disallowed_tools`` and
        # ``ClaudeAgentOptions.mcp_servers``). The Codex backend has no
        # analogous tool-filtering / external-MCP-server hooks, so we
        # accept and ignore them to keep ``Backend.create_client``
        # uniform.
        if self._codex is None:
            raise RuntimeError(
                "CodexBackend must be entered before creating clients.")

        AsyncThread = _symbol("codex_app_server.api", "AsyncThread")
        _params_dict = _symbol("codex_app_server.client", "_params_dict")
        AskForApproval = _symbol("codex_app_server.generated.v2_all",
                                 "AskForApproval")
        AskForApprovalValue = _symbol("codex_app_server.generated.v2_all",
                                      "AskForApprovalValue")
        SandboxMode = _symbol("codex_app_server.generated.v2_all",
                              "SandboxMode")
        ThreadStartParams = _symbol("codex_app_server.generated.v2_all",
                                    "ThreadStartParams")

        params = ThreadStartParams(
            model=model,
            developer_instructions=system_prompt or None,
            config={
                "model_reasoning_effort": _REASONING_EFFORT,
                "model_context_window": 1000000,
            },
            cwd=str(Path.cwd()),
            sandbox=SandboxMode.danger_full_access,
            approval_policy=AskForApproval(root=AskForApprovalValue.never),
        )
        start_payload = _params_dict(params)
        if tools:
            start_payload["dynamicTools"] = [
                _dynamic_tool_spec(t) for t in tools
            ]

        await self._codex._ensure_initialized()
        session_init = await _build_session_init_event(self._codex._client,
                                                       Path.cwd())
        started = await self._codex._client.thread_start(start_payload)
        thread_id = started.thread.id

        if tools:
            _TOOL_HANDLERS[thread_id] = {t.name: t.handler for t in tools}
        try:
            yield CodexClient(AsyncThread(self._codex, thread_id),
                              session_init=session_init)
        finally:
            _TOOL_HANDLERS.pop(thread_id, None)
