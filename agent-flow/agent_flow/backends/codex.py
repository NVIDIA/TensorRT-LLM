from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, AsyncIterator

from openai_codex.client import CodexConfig
from openai_codex.generated.v2_all import (
    AskForApproval,
    AskForApprovalValue,
    FunctionDynamicToolSpec,
    SandboxMode,
    ThreadStartParams,
)

from ..tools import normalize_tool
from ..types import SessionInitEvent
from .base import Backend, BackendClient
from .codex_config import build_session_config
from .codex_events import CodexClient
from .codex_transport import CodexTransport


def _resolve_codex_bin() -> str:
    override = os.environ.get("CODEX_BIN")
    if override:
        if not Path(override).is_file():
            raise FileNotFoundError(f"CODEX_BIN={override!r} does not point to a file.")
        return override
    try:
        from codex_cli_bin import bundled_codex_path
    except ImportError:
        system_bin = shutil.which("codex")
        if system_bin:
            return system_bin
        raise FileNotFoundError("Install the pinned Codex CLI runtime or set CODEX_BIN.") from None
    return str(bundled_codex_path())


def _dynamic_tool_spec(tool: Any) -> dict[str, Any]:
    definition = normalize_tool(tool)
    return FunctionDynamicToolSpec(
        type="function",
        name=definition.name,
        description=definition.description,
        input_schema=definition.input_schema,
        defer_loading=False,
    ).model_dump(by_alias=True, mode="json", exclude_none=True)


async def _session_init(transport: CodexTransport, cwd: Path) -> SessionInitEvent | None:
    async def optional(method, params):
        try:
            return await asyncio.wait_for(transport.request(method, params), timeout=5)
        except Exception:
            return None

    skills_response, plugins_response = await asyncio.gather(
        optional("skills/list", {"cwds": [str(cwd)], "forceReload": False}),
        optional("plugin/list", {}),
    )
    skills = []
    if skills_response is not None:
        for entry in skills_response.get("data", []):
            for skill in entry.get("skills", []):
                if skill.get("enabled") is not False and isinstance(skill.get("name"), str):
                    skills.append(skill["name"])
    plugins = []
    if plugins_response is not None:
        for marketplace in plugins_response.get("marketplaces", []):
            for plugin in marketplace.get("plugins", []):
                if plugin.get("installed") and plugin.get("enabled") is not False:
                    if isinstance(plugin.get("name"), str):
                        plugins.append(plugin["name"])
    if skills_response is None and not plugins:
        return None
    # The protocol has no resolved available-agent inventory API. Keep this
    # unknown instead of claiming every file under .codex/agents is loaded.
    return SessionInitEvent(skills=skills, plugins=plugins)


_CLI_VERSION_TIMEOUT_S = 5.0
_VERSION_CACHE: str | None = None


def _codex_sdk_version() -> str:
    """Version of the Python ``openai-codex`` SDK package, or ``""``."""
    try:
        return _pkg_version("openai-codex")
    except PackageNotFoundError:
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
        out = subprocess.run(
            [cli_path, "--version"], capture_output=True, text=True, timeout=_CLI_VERSION_TIMEOUT_S
        )
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
        _VERSION_CACHE = _format_version(_codex_cli_version(), _codex_sdk_version())
    return _VERSION_CACHE


_REASONING_EFFORT = "max"


def _disabled_skills_override(names: tuple[str, ...]) -> tuple[str, ...]:
    if not names:
        return ()
    entries = ", ".join(f"{{name={json.dumps(name)}, enabled=false}}" for name in names)
    return (f"skills.config=[{entries}]",)


class CodexBackend(Backend):
    def __init__(
        self,
        reasoning_effort: str | None = None,
        disabled_skills: tuple[str, ...] = (),
    ) -> None:
        self._transport: CodexTransport | None = None
        self._reasoning_effort = reasoning_effort or _REASONING_EFFORT
        self._disabled_skills = disabled_skills

    def version(self) -> str:
        return _codex_backend_version()

    def reasoning_effort(self) -> str:
        return self._reasoning_effort

    async def __aenter__(self) -> "CodexBackend":
        self._transport = CodexTransport(
            CodexConfig(
                codex_bin=_resolve_codex_bin(),
                experimental_api=True,
                config_overrides=_disabled_skills_override(self._disabled_skills),
            )
        )
        await self._transport.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._transport is not None:
            await self._transport.close()
            self._transport = None

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
        if self._transport is None:
            raise RuntimeError("CodexBackend must be entered before creating clients.")
        session_cwd = (cwd or Path.cwd()).resolve()
        config, definitions = build_session_config(
            extra_mcp_servers=extra_mcp_servers,
            hooks=hooks,
            disallowed_tools=disallowed_tools,
            tools=tools,
            cwd=session_cwd,
        )
        mcp_overrides = config.get("mcp_servers", {})
        if any("disabled_tools" in server for server in mcp_overrides.values()):
            # Codex replaces arrays when merging thread overrides. Preserve
            # exclusions inherited from the native config for this session's cwd.
            # A failed read must prevent starting a thread with weakened bans.
            native_config = await asyncio.wait_for(
                self._transport.request(
                    "config/read", {"cwd": str(session_cwd), "includeLayers": True}
                ),
                timeout=5,
            )
            inherited_configs = [native_config["config"]]
            # Starting an unrestricted session can trust a project and activate
            # its config. Conservatively retain those exclusions as well.
            inherited_configs.extend(
                layer["config"]
                for layer in native_config.get("layers") or []
                if layer["name"]["type"] == "project" and layer.get("disabledReason") is not None
            )
            for name, server in mcp_overrides.items():
                if "disabled_tools" not in server:
                    continue
                inherited = []
                for inherited_config in inherited_configs:
                    native_servers = inherited_config.get("mcp_servers") or {}
                    excluded = native_servers.get(name, {}).get("disabled_tools")
                    if excluded is None:
                        continue
                    if not isinstance(excluded, list) or not all(
                        isinstance(tool_name, str) for tool_name in excluded
                    ):
                        raise ValueError(
                            f"Native MCP disabled_tools for {name!r} must be a list of strings"
                        )
                    inherited.extend(excluded)
                server["disabled_tools"] = list(
                    dict.fromkeys([*inherited, *server["disabled_tools"]])
                )
        config.update(
            model_reasoning_effort=self._reasoning_effort,
            model_context_window=1000000,
        )
        params = ThreadStartParams(
            model=model,
            developer_instructions=system_prompt or None,
            config=config,
            cwd=str(session_cwd),
            sandbox=SandboxMode.danger_full_access,
            approval_policy=AskForApproval(root=AskForApprovalValue.never),
        )
        payload = params.model_dump(by_alias=True, mode="json", exclude_none=True)
        if definitions:
            payload["dynamicTools"] = [_dynamic_tool_spec(tool) for tool in definitions]
        session_init = await _session_init(self._transport, session_cwd)
        started = await self._transport.request("thread/start", payload)
        thread_id = started["thread"]["id"]
        self._transport.register_tools(thread_id, definitions)
        try:
            yield CodexClient(self._transport, thread_id, session_init=session_init)
        finally:
            self._transport.unregister_tools(thread_id)
            # Unload the thread, while retaining the saved session just as the
            # native runtime does. No global MCP config is changed.
            try:
                await asyncio.wait_for(
                    self._transport.request("thread/unsubscribe", {"threadId": thread_id}),
                    timeout=5,
                )
            except Exception:
                pass
