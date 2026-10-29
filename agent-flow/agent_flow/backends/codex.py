# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import functools
import json
import os
import random
import shutil
import subprocess
from contextlib import asynccontextmanager
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, AsyncIterator

from openai_codex.client import CodexConfig
from openai_codex.errors import TransportClosedError
from openai_codex.generated.v2_all import (
    AskForApproval,
    AskForApprovalValue,
    FunctionDynamicToolSpec,
    SandboxMode,
    ThreadForkParams,
    ThreadStartParams,
)

from ..tools import normalize_tool
from ..types import SessionInitEvent
from .base import Backend, BackendClient
from .codex_config import build_session_config
from .codex_events import CodexClient
from .codex_transport import CodexTransport


def _resolve_codex_bin() -> str:
    """The ``codex`` binary a Codex backend launches.

    ``$CODEX_BIN`` names one outright. Otherwise the pinned runtime, unless
    the ``codex`` on ``PATH`` is newer: a login offers a new model only to a
    client that knows it, and the operator's own CLI is the one kept current.
    """
    override = os.environ.get("CODEX_BIN")
    if override:
        if not Path(override).is_file():
            raise FileNotFoundError(f"CODEX_BIN={override!r} does not point to a file.")
        return override
    system_bin = shutil.which("codex")
    try:
        from codex_cli_bin import bundled_codex_path
    except ImportError:
        if system_bin:
            return system_bin
        raise FileNotFoundError("Install the pinned Codex CLI runtime or set CODEX_BIN.") from None
    bundled = str(bundled_codex_path())
    if system_bin and _newer(system_bin, bundled):
        return system_bin
    return bundled


@functools.lru_cache(maxsize=None)
def _newer(candidate: str, than: str) -> bool:
    """Whether ``candidate``'s ``codex --version`` is newer than ``than``'s; False when unreadable."""
    mine = _version_tuple(candidate)
    return bool(mine) and mine > _version_tuple(than)


def _version_tuple(cli_path: str) -> tuple[int, ...]:
    """``codex-cli 0.162.1`` as ``(0, 162, 1)``; empty when it cannot be read."""
    try:
        out = subprocess.run(
            [cli_path, "--version"], capture_output=True, text=True, timeout=_CLI_VERSION_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    words = out.stdout.split() if out.returncode == 0 else []
    try:
        return tuple(int(part) for part in words[-1].split("-")[0].split(".")) if words else ()
    except ValueError:
        return ()


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

#: Most to least open. A session asks for the first its site's requirements
#: allow: Codex narrows a forbidden request by itself, and silently.
_SANDBOXES = (SandboxMode.danger_full_access, SandboxMode.workspace_write, SandboxMode.read_only)
_APPROVALS = (
    AskForApprovalValue.never,
    AskForApprovalValue.on_request,
    AskForApprovalValue.untrusted,
)
#: ``thread/start``'s sandbox policy types, named as a configuration names its modes.
_SANDBOX_NAMES = {
    "dangerFullAccess": SandboxMode.danger_full_access.value,
    "workspaceWrite": SandboxMode.workspace_write.value,
    "readOnly": SandboxMode.read_only.value,
    "externalSandbox": "external-sandbox",
}
#: App servers started together can collide on Codex's shared state database,
#: and the one that loses exits at once; a moment later it starts.
_START_ATTEMPTS = 3
_RESTART_WAIT_S = (1.0, 3.0)


async def _permissions(transport: CodexTransport) -> tuple[SandboxMode, AskForApprovalValue]:
    """The most open sandbox and approval policy the site's Codex requirements allow.

    Requirements come from ``/etc/codex/requirements.toml``, MDM or the cloud;
    a runtime that cannot read them gets the most open of each.
    """
    try:
        response = await asyncio.wait_for(
            transport.request("configRequirements/read", {}), timeout=5
        )
    except Exception:
        response = {}
    requirements = response.get("requirements") or {}
    sandboxes = requirements.get("allowedSandboxModes")
    approvals = requirements.get("allowedApprovalPolicies")
    sandbox = next((m for m in _SANDBOXES if sandboxes is None or m.value in sandboxes), None)
    approval = next((a for a in _APPROVALS if approvals is None or a.value in approvals), None)
    return sandbox or _SANDBOXES[0], approval or _APPROVALS[0]


def _sandbox_name(policy: Any) -> str | None:
    kind = policy.get("type") if isinstance(policy, dict) else None
    return _SANDBOX_NAMES.get(kind, kind)


def _approval_name(policy: Any) -> str | None:
    return policy if policy is None or isinstance(policy, str) else "granular"


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


def _disabled_skills_override(names: tuple[str, ...]) -> tuple[str, ...]:
    if not names:
        return ()
    entries = ", ".join(f"{{name={json.dumps(name)}, enabled=false}}" for name in names)
    return (f"skills.config=[{entries}]",)


#: The feature switches that keep a session's user-installed plugins and apps out of it.
_NO_PLUGINS_OVERRIDES: tuple[str, ...] = ("features.plugins=false", "features.apps=false")


def _config_overrides(disabled_skills: tuple[str, ...], plugins: bool) -> tuple[str, ...]:
    """The app server's ``-c`` overrides: skills turned off, and plugins unless kept."""
    return _disabled_skills_override(disabled_skills) + (() if plugins else _NO_PLUGINS_OVERRIDES)


class CodexBackend(Backend):
    def __init__(
        self,
        reasoning_effort: str | None = None,
        disabled_skills: tuple[str, ...] = (),
        writable_roots: tuple[str, ...] = (),
        plugins: bool = True,
    ) -> None:
        self._transport: CodexTransport | None = None
        self._reasoning_effort = reasoning_effort
        self._disabled_skills = disabled_skills
        self._writable_roots = writable_roots
        self._plugins = plugins
        self._sandbox, self._approval = _SANDBOXES[0], _APPROVALS[0]

    def version(self) -> str:
        return _codex_backend_version()

    def reasoning_effort(self) -> str:
        return self._reasoning_effort or ""

    async def __aenter__(self) -> "CodexBackend":
        for attempt in range(1, _START_ATTEMPTS + 1):
            transport = CodexTransport(
                CodexConfig(
                    codex_bin=_resolve_codex_bin(),
                    experimental_api=True,
                    config_overrides=_config_overrides(self._disabled_skills, self._plugins),
                )
            )
            try:
                await transport.start()
                break
            except TransportClosedError:
                if attempt == _START_ATTEMPTS:
                    raise
                await asyncio.sleep(random.uniform(*_RESTART_WAIT_S))
        self._transport = transport
        self._sandbox, self._approval = await _permissions(transport)
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
        fork_from: str | None = None,
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
        if self._reasoning_effort is not None:
            config["model_reasoning_effort"] = self._reasoning_effort
        if self._sandbox is SandboxMode.workspace_write and self._writable_roots:
            # Besides the session's cwd. Anything else waits on an approval,
            # which the transport grants.
            config["sandbox_workspace_write"] = {
                "writable_roots": [str(Path(root).resolve()) for root in self._writable_roots]
            }
        options = dict(
            # An empty model leaves the choice to Codex's own configuration.
            model=model or None,
            developer_instructions=system_prompt or None,
            config=config,
            cwd=str(session_cwd),
            sandbox=self._sandbox,
            approval_policy=AskForApproval(root=self._approval),
        )
        if fork_from is not None:
            if definitions:
                raise ValueError(
                    "thread/fork cannot introduce dynamic tools; use native MCP tools "
                    "or start a fresh session with these tools"
                )
            params = ThreadForkParams(thread_id=fork_from, exclude_turns=True, **options)
        else:
            params = ThreadStartParams(**options)
        payload = params.model_dump(by_alias=True, mode="json", exclude_none=True)
        if definitions:
            payload["dynamicTools"] = [_dynamic_tool_spec(tool) for tool in definitions]
        session_init = await _session_init(self._transport, session_cwd)
        started = await self._transport.request(
            "thread/fork" if fork_from is not None else "thread/start", payload
        )
        thread_id = started["thread"]["id"]
        session_init = replace(
            session_init or SessionInitEvent(),
            backend="codex",
            model=started.get("model"),
            reasoning_effort=started.get("reasoningEffort"),
            session_id=thread_id,
            sandbox=_sandbox_name(started.get("sandbox")),
            approval_policy=_approval_name(started.get("approvalPolicy")),
        )
        self._transport.register_tools(thread_id, definitions)
        try:
            yield CodexClient(
                self._transport,
                thread_id,
                session_init=session_init,
                inherited_history=fork_from is not None,
            )
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
