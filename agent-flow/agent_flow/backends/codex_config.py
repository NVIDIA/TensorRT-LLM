"""Translate session options to the configuration supported by Codex 0.154.0.

These overrides belong to ``thread/start.config``. They never edit the user's
Codex configuration. Unsupported options fail at registration rather than being
silently ignored, especially when they describe tool restrictions.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_MCP_COMMON = {
    "enabled",
    "required",
    "enabled_tools",
    "disabled_tools",
    "startup_timeout_sec",
    "startup_timeout_ms",
    "tool_timeout_sec",
    "supports_parallel_tool_calls",
    "default_tools_approval_mode",
    "tools",
    "omit_tools_from",
    "environment_id",
    "name",
}
_MCP_STDIO = {"command", "args", "env", "env_vars", "cwd"}
_MCP_HTTP = {
    "url",
    "http_headers",
    "env_http_headers",
    "http_headers_helper",
    "bearer_token_env_var",
    "auth",
    "oauth",
    "oauth_resource",
    "scopes",
}
_SHELL_TOOLS = {"Bash", "shell", "shell_command", "exec_command", "write_stdin"}
_APPROVAL_MODES = {"auto", "prompt", "writes", "approve"}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return copy.deepcopy(dict(value))


def _string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")


def _strings(value: Any, label: str) -> None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TypeError(f"{label} must be a list of strings")


def _string_map(value: Any, label: str) -> None:
    values = _mapping(value, label)
    if not all(isinstance(v, str) for v in values.values()):
        raise TypeError(f"{label} must contain string values")


def _nonnegative(value: Any, label: str, *, integer: bool = False) -> None:
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < 0:
        kind = "integer" if integer else "number"
        raise ValueError(f"{label} must be a finite nonnegative {kind}")


def _unknown_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported {label} fields: {', '.join(sorted(unknown))}")


def _exact_name(value: str, label: str, *, allow_all: bool = False) -> None:
    _string(value, label)
    if allow_all and value == "*":
        return
    if any(c in value for c in "*?()"):
        raise ValueError(f"{label} requires an exact tool name" + (" or '*'" if allow_all else ""))


def _mcp_server(name: str, value: Any, cwd: Path | None) -> dict[str, Any]:
    _string(name, "MCP server name")
    if name == "agent-tools":
        raise ValueError("MCP server name 'agent-tools' is reserved for framework tools")
    server = _mapping(value, f"MCP server {name!r}")
    transport = server.pop("type", "stdio" if "command" in server else "http")
    if transport not in {"stdio", "http", "streamable-http"}:
        raise ValueError(
            f"Codex does not support MCP transport {transport!r}; use STDIO or streamable HTTP"
        )
    if "headers" in server:
        if "http_headers" in server:
            raise ValueError(f"MCP server {name!r} specifies both headers and http_headers")
        server["http_headers"] = server.pop("headers")
    stdio = transport == "stdio"
    _unknown_keys(server, _MCP_COMMON | (_MCP_STDIO if stdio else _MCP_HTTP), "MCP server")
    _string(server.get("command" if stdio else "url"), "MCP command" if stdio else "MCP URL")
    if stdio:
        if "cwd" in server:
            path = Path(server["cwd"])
            server["cwd"] = str(path if path.is_absolute() else (cwd or Path.cwd()) / path)
        elif cwd is not None:
            server["cwd"] = str(cwd.absolute())
    else:
        url = urlsplit(server["url"])
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("An HTTP MCP server URL must use http:// or https:// with a host")

    for key in {
        "args",
        "enabled_tools",
        "disabled_tools",
        "scopes",
        "omit_tools_from",
    } & server.keys():
        _strings(server[key], f"MCP {key}")
    for key in {"env", "http_headers", "env_http_headers"} & server.keys():
        _string_map(server[key], f"MCP {key}")
    for key in {"enabled", "required", "supports_parallel_tool_calls"} & server.keys():
        if not isinstance(server[key], bool):
            raise TypeError(f"MCP {key} must be a boolean")
    for key in {"tool_timeout_sec", "startup_timeout_sec", "startup_timeout_ms"} & server.keys():
        _nonnegative(server[key], f"MCP {key}", integer=key.endswith("_ms"))
    if "startup_timeout_sec" in server and "startup_timeout_ms" in server:
        raise ValueError("Specify only one MCP startup timeout unit")
    for key in {
        "bearer_token_env_var",
        "http_headers_helper",
        "oauth_resource",
        "environment_id",
        "name",
    } & server.keys():
        _string(server[key], f"MCP {key}")
    if "auth" in server and server["auth"] not in {"oauth", "chatgpt"}:
        raise ValueError("MCP auth must be 'oauth' or 'chatgpt'")
    if "default_tools_approval_mode" in server:
        if server["default_tools_approval_mode"] not in _APPROVAL_MODES:
            raise ValueError("Unsupported MCP default_tools_approval_mode")
    if "omit_tools_from" in server and set(server["omit_tools_from"]) - {
        "direct",
        "deferred",
        "code_mode",
    }:
        raise ValueError("Unsupported MCP omit_tools_from surface")
    if "env_vars" in server:
        if not isinstance(server["env_vars"], list):
            raise TypeError("MCP env_vars must be a list")
        for variable in server["env_vars"]:
            if isinstance(variable, str):
                _string(variable, "MCP environment variable")
            else:
                variable = _mapping(variable, "MCP environment variable")
                _unknown_keys(variable, {"name", "source"}, "MCP environment variable")
                _string(variable.get("name"), "MCP environment variable name")
                if "source" in variable:
                    _string(variable["source"], "MCP environment variable source")
    if "oauth" in server:
        oauth = _mapping(server["oauth"], "MCP OAuth settings")
        _unknown_keys(oauth, {"client_id", "callback_port", "callback_url"}, "MCP OAuth")
        for key in {"client_id", "callback_url"} & oauth.keys():
            _string(oauth[key], f"MCP OAuth {key}")
        if "callback_port" in oauth:
            _nonnegative(oauth["callback_port"], "MCP OAuth callback_port", integer=True)
            if oauth["callback_port"] > 65535:
                raise ValueError("MCP OAuth callback_port exceeds 65535")
    if "tools" in server:
        for tool_name, settings in _mapping(server["tools"], "MCP tools").items():
            _string(tool_name, "MCP tool name")
            settings = _mapping(settings, "MCP tool settings")
            _unknown_keys(settings, {"approval_mode", "output_token_limit"}, "MCP tool")
            if "approval_mode" in settings and settings["approval_mode"] not in _APPROVAL_MODES:
                raise ValueError("Unsupported MCP tool approval_mode")
            if "output_token_limit" in settings:
                _nonnegative(settings["output_token_limit"], "MCP output_token_limit", integer=True)
                if not settings["output_token_limit"]:
                    raise ValueError("MCP output_token_limit must be positive")
    return server


def _reject_unverifiable_hooks(hooks: dict[str, Any]) -> None:
    for groups in hooks.values():
        if isinstance(groups, list) and any(not isinstance(group, Mapping) for group in groups):
            raise TypeError(
                "Codex cannot execute Claude SDK HookMatcher callbacks. Use required_tools "
                "for shared required-tool enforcement."
            )
    raise ValueError(
        "Codex cannot verify trust of per-client native hooks. Configure and trust native hooks "
        "through Codex before starting agent-flow, or use required_tools for shared required-tool "
        "enforcement. Untrusted session hooks would be silently skipped by Codex."
    )


def build_session_config(
    *,
    extra_mcp_servers: dict[str, Any] | None = None,
    hooks: dict[str, Any] | None = None,
    disallowed_tools: list[str] | None = None,
    tools: list[Any] | None = None,
    cwd: Path | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Build per-thread Codex config and remove prohibited framework tools.

    Restrictions accept framework tool names, ``mcp__SERVER__TOOL`` names,
    ``apps.APP_ID.TOOL`` names, and built-ins with native disable controls.
    ``mcp__SERVER__*`` and ``apps.APP_ID.*`` disable an entire server/app.
    Shell tool aliases disable the native shell family. A tool restriction is
    not a filesystem policy: use a sandbox to prohibit writes through all tools.
    Native hooks must be configured and trusted through Codex itself; this
    adapter cannot verify trust of per-client hooks and rejects them.
    """
    config: dict[str, Any] = {}
    servers = {
        name: _mcp_server(name, value, cwd)
        for name, value in _mapping(extra_mcp_servers or {}, "MCP servers").items()
    }
    if servers:
        config["mcp_servers"] = servers
    if hooks:
        _reject_unverifiable_hooks(_mapping(hooks, "Codex hooks"))
    allowed = list(tools or [])
    registered_names = {t.name for t in allowed}
    if disallowed_tools is not None:
        _strings(disallowed_tools, "disallowed_tools")
    for name in disallowed_tools or []:
        _string(name, "Disallowed tool")
        dynamic_name = name.removeprefix("mcp__agent-tools__")
        if (
            name in registered_names
            or dynamic_name in registered_names
            or name.startswith("mcp__agent-tools__")
        ):
            _exact_name(dynamic_name, "Framework tool restriction", allow_all=True)
            allowed = [t for t in allowed if t.name not in {dynamic_name, name}]
            if dynamic_name == "*":
                allowed = []
            continue
        if name in _SHELL_TOOLS:
            config.setdefault("features", {})["shell_tool"] = False
        elif name in {"AskUserQuestion", "request_user_input"}:
            config.setdefault("tools", {})["experimental_request_user_input"] = {"enabled": False}
        elif name in {"WebSearch", "WebFetch", "web_search"}:
            config["web_search"] = "disabled"
        elif name in {"view_image", "image_generation"}:
            config.setdefault("features", {})[name] = False
        elif name in {"update_plan", "TodoWrite"}:
            config.setdefault("tools", {})["update_plan"] = {"enabled": False}
        elif name.startswith("mcp__") and name.count("__") >= 2:
            server_name, tool_name = name[5:].split("__", 1)
            _exact_name(server_name, "Disallowed MCP server")
            _exact_name(tool_name, "Disallowed MCP tool", allow_all=True)
            if server_name == "codex_apps":
                raise ValueError("Restrict Codex app tools using apps.APP_ID.TOOL or apps.APP_ID.*")
            server = config.setdefault("mcp_servers", {}).setdefault(server_name, {})
            if tool_name == "*":
                server["enabled"] = False
            else:
                disabled = server.setdefault("disabled_tools", [])
                if tool_name not in disabled:
                    disabled.append(tool_name)
        elif name.startswith("apps.") and len(name.split(".", 2)) == 3:
            _, app_id, tool_name = name.split(".", 2)
            _exact_name(app_id, "Disallowed app ID")
            _exact_name(tool_name, "Disallowed app tool", allow_all=True)
            if app_id == "_default":
                raise ValueError("App restrictions require an explicit app ID, not _default")
            app = config.setdefault("apps", {}).setdefault(app_id, {})
            if tool_name == "*":
                app["enabled"] = False
            else:
                app.setdefault("tools", {})[tool_name] = {"enabled": False}
        else:
            raise ValueError(
                f"Codex cannot enforce disallowed tool {name!r}. Use a registered framework tool "
                "name, mcp__SERVER__TOOL, apps.APP_ID.TOOL, or a supported native tool control."
            )
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("Codex session configuration must be JSON serializable") from exc
    return config, allowed
