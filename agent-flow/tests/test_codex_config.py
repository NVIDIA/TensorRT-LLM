import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.backends.codex_config import build_session_config


def test_empty_configuration_does_not_set_native_defaults():
    assert build_session_config() == ({}, [])
    assert build_session_config(hooks={}) == ({}, [])


def test_stdio_mcp_uses_session_cwd_and_keeps_environment():
    servers = {
        "files": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "files"],
            "env": {"ROOT": "/workspace"},
            "env_vars": ["TOKEN", {"name": "HOST_TOKEN"}],
            "startup_timeout_sec": 15,
            "tool_timeout_sec": 30,
        }
    }
    original = copy.deepcopy(servers)
    config, _ = build_session_config(extra_mcp_servers=servers, cwd=Path("/workspace"))
    assert config["mcp_servers"]["files"] == {
        "command": "python",
        "args": ["-m", "files"],
        "env": {"ROOT": "/workspace"},
        "env_vars": ["TOKEN", {"name": "HOST_TOKEN"}],
        "startup_timeout_sec": 15,
        "tool_timeout_sec": 30,
        "cwd": "/workspace",
    }
    assert servers == original


def test_stdio_explicit_relative_cwd_resolves_against_session_cwd():
    config, _ = build_session_config(
        extra_mcp_servers={"files": {"command": "files", "cwd": "child"}},
        cwd=Path("/workspace"),
    )
    assert config["mcp_servers"]["files"]["cwd"] == "/workspace/child"


def test_http_mcp_translates_headers_and_preserves_auth():
    server = {
        "type": "http",
        "url": "https://example.test/mcp",
        "headers": {"X-Client": "agent-flow"},
        "env_http_headers": {"X-Token": "TOKEN"},
        "bearer_token_env_var": "API_TOKEN",
        "auth": "oauth",
        "oauth": {"client_id": "client", "callback_port": 8123},
        "scopes": ["read"],
        "enabled_tools": ["read"],
        "tools": {"read": {"approval_mode": "auto", "output_token_limit": 500}},
    }
    config, _ = build_session_config(extra_mcp_servers={"docs": server})
    expected = {k: v for k, v in server.items() if k not in {"headers", "type"}}
    expected["http_headers"] = server["headers"]
    assert config["mcp_servers"]["docs"] == expected


@pytest.mark.parametrize(
    "server, message",
    [
        ({"type": "sse", "url": "https://example.test"}, "transport"),
        ({"type": "sdk", "instance": object()}, "transport"),
        ({"command": "server", "url": "https://example.test"}, "Unsupported"),
        ({"url": "file:///tmp/server"}, "http://"),
        ({"url": "https://example.test", "headers": {}, "http_headers": {}}, "both headers"),
        ({"command": "server", "env": {"TOKEN": 7}}, "string values"),
        ({"command": "server", "args": "argument"}, "list of strings"),
        ({"command": "server", "invented_option": True}, "Unsupported"),
        ({"command": "server", "enabled": "false"}, "boolean"),
        ({"command": "server", "startup_timeout_sec": float("nan")}, "finite"),
        ({"command": "server", "startup_timeout_sec": -1}, "nonnegative"),
        ({"command": "server", "startup_timeout_sec": 1, "startup_timeout_ms": 1}, "one MCP"),
        ({"url": "https://example.test", "auth": "none"}, "auth"),
        ({"url": "https://example.test", "oauth": {"callback_port": 70000}}, "65535"),
        ({"command": "server", "env_vars": [{"name": "TOKEN", "typo": "value"}]}, "Unsupported"),
        ({"command": "server", "tools": {"x": {"output_token_limit": 0}}}, "positive"),
        ({"command": "server", "default_tools_approval_mode": "never"}, "approval_mode"),
    ],
)
def test_invalid_mcp_options_fail_instead_of_being_dropped(server, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_session_config(extra_mcp_servers={"external": server})


def test_framework_server_name_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        build_session_config(extra_mcp_servers={"agent-tools": {"command": "server"}})


def test_dynamic_restrictions_accept_bare_and_claude_namespaced_names():
    one, two, three = [SimpleNamespace(name=name) for name in ["one", "two", "three"]]
    tools = [one, two, three]
    config, allowed = build_session_config(
        tools=tools, disallowed_tools=["one", "mcp__agent-tools__two", "one"]
    )
    assert config == {}
    assert allowed == [three]
    assert tools == [one, two, three]


def test_framework_namespace_wildcard_removes_all_dynamic_tools():
    assert build_session_config(
        tools=[SimpleNamespace(name="one")], disallowed_tools=["mcp__agent-tools__*"]
    ) == ({}, [])


def test_mcp_restrictions_merge_without_mutating_original_config():
    servers = {"files": {"command": "files", "disabled_tools": ["remove"]}}
    original = copy.deepcopy(servers)
    config, _ = build_session_config(
        extra_mcp_servers=servers,
        disallowed_tools=["mcp__files__write", "mcp__files__remove", "mcp__other__*"],
    )
    assert config["mcp_servers"]["files"]["disabled_tools"] == ["remove", "write"]
    assert config["mcp_servers"]["other"] == {"enabled": False}
    assert servers == original


@pytest.mark.parametrize("name", ["Bash", "shell", "shell_command", "exec_command", "write_stdin"])
def test_shell_aliases_disable_native_shell_family(name):
    assert build_session_config(disallowed_tools=[name])[0] == {"features": {"shell_tool": False}}


def test_builtin_controls_match_pinned_native_configuration():
    config, _ = build_session_config(
        disallowed_tools=[
            "AskUserQuestion",
            "WebFetch",
            "view_image",
            "image_generation",
            "TodoWrite",
        ]
    )
    assert config == {
        "tools": {
            "experimental_request_user_input": {"enabled": False},
            "update_plan": {"enabled": False},
        },
        "web_search": "disabled",
        "features": {"view_image": False, "image_generation": False},
    }


def test_app_controls_require_explicit_app_ids():
    config, _ = build_session_config(disallowed_tools=["apps.calendar.create", "apps.mail.*"])
    assert config == {
        "apps": {
            "calendar": {"tools": {"create": {"enabled": False}}},
            "mail": {"enabled": False},
        }
    }
    with pytest.raises(ValueError, match="APP_ID"):
        build_session_config(disallowed_tools=["mcp__codex_apps__create"])


@pytest.mark.parametrize("name", ["Write", "Edit", "apply_patch", "Bash(rm:*)", "unknown"])
def test_restrictions_without_native_enforcement_fail_explicitly(name):
    with pytest.raises(ValueError, match="cannot enforce"):
        build_session_config(disallowed_tools=[name])


def test_mcp_patterns_are_not_misrepresented_as_exact_restrictions():
    with pytest.raises(ValueError, match="exact tool name"):
        build_session_config(disallowed_tools=["mcp__files__write*"])


@pytest.mark.parametrize(
    "name", ["mcp__agent-tools__write*", "apps.calendar.create*", "mcp__*__write", "apps.*.create"]
)
def test_other_restriction_patterns_fail_explicitly(name):
    with pytest.raises(ValueError, match="exact tool name"):
        build_session_config(disallowed_tools=[name])


def test_default_app_settings_cannot_be_used_as_a_global_app_ban():
    with pytest.raises(ValueError, match="explicit app ID"):
        build_session_config(disallowed_tools=["apps._default.*"])


def test_claude_python_callbacks_require_shared_policy():
    with pytest.raises(TypeError, match="HookMatcher.*required_tools"):
        build_session_config(hooks={"Stop": [SimpleNamespace(hooks=[lambda: None])]})


def test_native_hooks_fail_if_session_trust_cannot_be_verified():
    with pytest.raises(ValueError, match="cannot verify trust"):
        build_session_config(
            hooks={
                "Stop": [{"hooks": [{"type": "command", "command": "validate-stop"}]}],
            }
        )


def test_embedded_native_trust_claims_do_not_bypass_trust_validation():
    with pytest.raises(ValueError, match="cannot verify trust"):
        build_session_config(hooks={"state": {"key": {"trusted_hash": "arbitrary"}}})
