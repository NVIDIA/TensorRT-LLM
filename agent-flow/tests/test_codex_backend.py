from __future__ import annotations

import copy
from pathlib import Path

import pytest
from openai_codex.generated.v2_all import (
    DynamicToolSpec,
    FunctionDynamicToolSpec,
    ThreadStartParams,
)
from pydantic import TypeAdapter

from agent_flow.backends import codex as codex_module
from agent_flow.backends.codex import CodexBackend, _dynamic_tool_spec
from agent_flow.tools import tool


class FakeTransport:
    """Record backend requests without launching a Codex process."""

    def __init__(self, config=None):
        self.config = config
        self.started = False
        self.closed = False
        self.requests = []
        self.registered = {}
        self.unregistered = []
        self.thread_count = 0

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def request(self, method, params):
        self.requests.append((method, copy.deepcopy(params)))
        if method == "thread/start":
            self.thread_count += 1
            return {"thread": {"id": f"thread-{self.thread_count}"}}
        if method == "skills/list":
            return {
                "data": [
                    {"skills": [{"name": "available"}, {"name": "disabled", "enabled": False}]}
                ]
            }
        if method == "plugin/list":
            return {"marketplaces": [{"plugins": [{"name": "installed", "installed": True}]}]}
        return {}

    def register_tools(self, thread_id, tools):
        self.registered[thread_id] = list(tools)

    def unregister_tools(self, thread_id):
        self.unregistered.append(thread_id)
        self.registered.pop(thread_id, None)


@pytest.fixture
async def backend(monkeypatch):
    monkeypatch.setattr(codex_module, "CodexTransport", FakeTransport)
    monkeypatch.setattr(codex_module, "_resolve_codex_bin", lambda: "/unused/codex")
    async with CodexBackend() as instance:
        yield instance


@tool("read", "Read data", {"path": str}, annotations={"readOnlyHint": True})
async def read_tool(args):
    return {"content": [{"type": "text", "text": args["path"]}]}


@tool("write", "Write data", {"path": str})
async def write_tool(args):
    return {"content": [{"type": "text", "text": "written"}]}


def test_dynamic_tool_spec_validates_against_pinned_sdk_discriminated_union():
    payload = _dynamic_tool_spec(read_tool)
    function = FunctionDynamicToolSpec.model_validate(payload)
    union = TypeAdapter(DynamicToolSpec).validate_python(payload)
    assert function.type == union.root.type == "function"
    assert payload["inputSchema"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    assert payload["deferLoading"] is False


async def test_backend_start_and_close_use_configured_runtime(monkeypatch):
    monkeypatch.setattr(codex_module, "CodexTransport", FakeTransport)
    monkeypatch.setattr(codex_module, "_resolve_codex_bin", lambda: "/configured/codex")
    instance = CodexBackend()
    async with instance:
        transport = instance._transport
        assert transport.started
        assert transport.config.codex_bin == "/configured/codex"
        assert transport.config.experimental_api is True
    assert transport.closed
    assert instance._transport is None


async def test_session_options_use_developer_instructions_and_preserve_native_defaults(
    backend, tmp_path
):
    transport = backend._transport
    async with backend.create_client(
        "Project instructions", "gpt-5.4", tools=[read_tool], cwd=tmp_path
    ) as client:
        payload = next(params for method, params in transport.requests if method == "thread/start")
        validated = ThreadStartParams.model_validate(payload)
        assert validated.model == "gpt-5.4"
        assert payload["cwd"] == str(tmp_path.resolve())
        assert payload["developerInstructions"] == "Project instructions"
        assert "baseInstructions" not in payload
        assert payload["approvalPolicy"] == "never"
        assert payload["sandbox"] == "danger-full-access"
        assert payload["config"]["model_reasoning_effort"] == backend.reasoning_effort()
        assert payload["dynamicTools"] == [_dynamic_tool_spec(read_tool)]
        assert transport.registered == {"thread-1": [read_tool]}
        assert await client.list_available_skills() == ["available"]
    assert transport.registered == {}
    assert transport.unregistered == ["thread-1"]
    assert ("thread/unsubscribe", {"threadId": "thread-1"}) in transport.requests


async def test_empty_prompt_does_not_replace_runtime_instructions(backend, tmp_path):
    async with backend.create_client("", "gpt-5.4", cwd=tmp_path):
        payload = next(
            params for method, params in backend._transport.requests if method == "thread/start"
        )
        assert "developerInstructions" not in payload
        assert "baseInstructions" not in payload
        assert "dynamicTools" not in payload


async def test_tool_restrictions_affect_both_advertised_tools_and_dispatch_registration(
    backend, tmp_path
):
    async with backend.create_client(
        "",
        "gpt-5.4",
        tools=[read_tool, write_tool],
        disallowed_tools=["mcp__agent-tools__write", "Bash", "WebSearch"],
        cwd=tmp_path,
    ):
        transport = backend._transport
        payload = next(params for method, params in transport.requests if method == "thread/start")
        assert [entry["name"] for entry in payload["dynamicTools"]] == ["read"]
        assert [entry.name for entry in transport.registered["thread-1"]] == ["read"]
        assert payload["config"]["features"]["shell_tool"] is False
        assert payload["config"]["web_search"] == "disabled"


async def test_concurrent_sessions_keep_mcp_overrides_local_and_do_not_mutate_inputs(
    backend, tmp_path
):
    first_servers = {
        "docs": {"type": "stdio", "command": "python", "args": ["first.py"], "env": {"TEAM": "one"}}
    }
    second_servers = {
        "docs": {"type": "http", "url": "https://example.com/mcp", "headers": {"X-Team": "two"}}
    }
    originals = copy.deepcopy((first_servers, second_servers))
    async with backend.create_client(
        "first", "gpt-5.4", extra_mcp_servers=first_servers, cwd=tmp_path
    ):
        async with backend.create_client(
            "second", "gpt-5.4", extra_mcp_servers=second_servers, cwd=tmp_path
        ):
            requests = [
                params for method, params in backend._transport.requests if method == "thread/start"
            ]
            first, second = (params["config"]["mcp_servers"]["docs"] for params in requests)
            assert first == {
                "command": "python",
                "args": ["first.py"],
                "env": {"TEAM": "one"},
                "cwd": str(tmp_path),
            }
            assert second == {"url": "https://example.com/mcp", "http_headers": {"X-Team": "two"}}
            assert set(backend._transport.registered) == {"thread-1", "thread-2"}
        assert set(backend._transport.registered) == {"thread-1"}
    assert (first_servers, second_servers) == originals
    assert not any(method.startswith("config/") for method, _ in backend._transport.requests)


async def test_session_exception_unregisters_tools_and_unsubscribes(backend, tmp_path):
    with pytest.raises(ValueError, match="application failure"):
        async with backend.create_client("", "gpt-5.4", tools=[read_tool], cwd=tmp_path):
            raise ValueError("application failure")
    assert backend._transport.registered == {}
    assert ("thread/unsubscribe", {"threadId": "thread-1"}) in backend._transport.requests


@pytest.mark.parametrize(
    "options, message",
    [
        ({"disallowed_tools": ["arbitrary-native-tool"]}, "cannot enforce"),
        (
            {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}},
            "cannot verify trust",
        ),
        (
            {"extra_mcp_servers": {"remote": {"type": "sse", "url": "https://example.com/sse"}}},
            "MCP transport",
        ),
    ],
)
async def test_unsupported_options_fail_before_any_session_rpc(backend, tmp_path, options, message):
    with pytest.raises((TypeError, ValueError), match=message):
        async with backend.create_client("", "gpt-5.4", cwd=tmp_path, **options):
            pytest.fail("invalid options must not create a client")
    assert backend._transport.requests == []


async def test_client_requires_entered_backend():
    with pytest.raises(RuntimeError, match="must be entered"):
        async with CodexBackend().create_client("", "gpt-5.4", cwd=Path.cwd()):
            pytest.fail("an unopened backend must not create a client")
