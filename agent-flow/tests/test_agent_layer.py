from __future__ import annotations

from unittest.mock import patch

import anyio

from agent_flow import (AgentLayer, AgentLayerConfig, AgentRequest,
                        BackendConfig, SessionConfig, ToolCallEvent, UsageInfo)
from agent_flow.config import HumanRequest, HumanRequestOption

from .helpers import FakeBackend


def _config(*,
            session_mode: str = "stateless",
            name: str = "agent",
            human_input_enabled: bool = False,
            print_activity: bool = True) -> AgentLayerConfig:
    return AgentLayerConfig(
        name=name,
        system_prompt="You are helpful.",
        backend=BackendConfig(kind="claude-code", model="test-model"),
        session=SessionConfig(mode=session_mode),
        print_activity=print_activity,
        human_input_enabled=human_input_enabled,
    )


def test_stateless_layer_creates_fresh_client_per_call():
    backend = FakeBackend([{"text": "one"}, {"text": "two"}])
    layer = AgentLayer(_config())

    with patch("agent_flow.layers.create_backend", return_value=backend):
        first = layer("hello")
        second = layer("again")

    assert backend.create_client_calls == 2
    assert backend.enter_count == 2
    assert backend.exit_count == 2
    assert first == "one"
    assert second == "two"
    assert backend.clients[0] is not backend.clients[1]


def test_persistent_layer_reuses_client_across_calls():
    backend = FakeBackend([{"text": "first"}, {"text": "second"}])

    with patch("agent_flow.layers.create_backend", return_value=backend):
        with AgentLayer(_config(session_mode="persistent")) as layer:
            first = layer("hello")
            second = layer("again")

    assert backend.enter_count == 1
    assert backend.exit_count == 1
    assert backend.create_client_calls == 1
    assert backend.client_exit_count == 1
    assert first == "first"
    assert second == "first"
    assert backend.clients[0].send_count == 2


def test_one_forward_performs_one_backend_send_and_appends_response():
    backend = FakeBackend([{
        "text":
        "done",
        "tool_calls": [ToolCallEvent(name="Bash", input={"command": "ls"})],
    }])
    layer = AgentLayer(_config())

    with patch("agent_flow.layers.create_backend", return_value=backend):
        result = layer("run")

    assert backend.create_client_calls == 1
    assert backend.clients[0].send_count == 1
    assert result == "done"


def test_layer_prints_live_agent_activity(capsys):
    backend = FakeBackend([{
        "text":
        "done",
        "tool_calls": [ToolCallEvent(name="Bash", input={"command": "ls"})],
    }])
    layer = AgentLayer(_config(name="planner"))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer("run")

    captured = capsys.readouterr()
    assert "PLANNER" in captured.out
    assert "claude-code" in captured.out
    assert "test-model" in captured.out
    assert "Bash" in captured.out
    assert "completed" in captured.out


def test_layer_started_panel_includes_backend_version(capsys):
    """The started panel surfaces whichever ``cli/sdk`` string the
    backend reports. Uses a ``FakeBackend`` subclass whose ``version()``
    returns a fixed string so tests do not spawn the real CLI."""

    class VersionedBackend(FakeBackend):

        def version(self) -> str:
            return "cli 9.9.9 · sdk 0.0.1"

    backend = VersionedBackend([{"text": "done"}])
    layer = AgentLayer(_config(name="planner"))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer("run")

    captured = capsys.readouterr()
    assert "version" in captured.out
    assert "cli 9.9.9" in captured.out
    assert "sdk 0.0.1" in captured.out


def test_layer_started_panel_omits_version_when_backend_returns_empty(capsys):
    """``FakeBackend`` inherits the empty default — the started panel
    should silently skip the version segment instead of emitting
    ``version`` with a blank value."""
    backend = FakeBackend([{"text": "done"}])
    layer = AgentLayer(_config(name="planner"))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer("run")

    captured = capsys.readouterr()
    assert "started" in captured.out
    assert "claude-code" in captured.out
    assert "version" not in captured.out


def test_layer_can_disable_activity_printing(capsys):
    backend = FakeBackend([{"text": "done"}])
    layer = AgentLayer(
        AgentLayerConfig(
            name="quiet-agent",
            system_prompt="You are helpful.",
            backend=BackendConfig(kind="claude-code", model="test-model"),
            print_activity=False,
        ))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer("run")

    captured = capsys.readouterr()
    assert captured.out == ""


def test_completion_panel_includes_usage_and_cost(capsys):
    usage = UsageInfo(
        input_tokens=123,
        output_tokens=45,
        cache_read_tokens=10,
        total_tokens=178,
        cost_usd=0.0123,
        num_turns=3,
        duration_ms=2500,
        context_tokens=50000,
        context_window=200000,
        context_percentage=25.0,
    )
    backend = FakeBackend([{"text": "done", "usage": usage}])
    layer = AgentLayer(_config(name="planner"))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer("run")

    out = capsys.readouterr().out
    assert "tokens" in out
    assert "123" in out
    assert "178" in out
    assert "$0.0123" in out
    assert "turns" in out
    assert "2.50s" in out
    assert "context" in out
    assert "25.0%" in out
    assert "50,000" in out
    assert "200,000" in out


def test_completion_panel_without_usage_does_not_error(capsys):
    backend = FakeBackend([{"text": "done"}])
    layer = AgentLayer(_config(name="planner"))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        result = layer("run")

    assert result == "done"
    out = capsys.readouterr().out
    assert "completed" in out
    assert "tokens" not in out
    assert "cost" not in out


def test_human_input_disabled_by_default_runs_once():
    backend = FakeBackend([{"text": "done"}])
    layer = AgentLayer(_config(print_activity=False))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        result = layer("hello")

    assert result == "done"
    assert backend.clients[0].send_count == 1
    # Default config does not register the ask_human tool.
    assert backend.clients[0].tools is None


def test_no_post_turn_prompt_after_agent_completes_even_when_enabled():
    """Enabling HITL does NOT inject a forced post-turn human-input step.

    The agent is the only thing that can ask the human, via the
    ``ask_human`` MCP tool. When the agent finishes its turn without
    calling that tool, the layer returns immediately — no auto-injected
    "refine?" panel, no stdin read, no follow-up agent turn. This
    enforces the principle: human input only when the agent is working
    and asks for it.
    """
    backend = FakeBackend([{"text": "done"}])
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    # Patch the stdin reader so a stray prompt would be obvious — this
    # test asserts it never runs.
    with patch("agent_flow.layers.create_backend", return_value=backend), \
            patch("agent_flow.layers._read_stdin",
                  side_effect=AssertionError("stdin must not be read")):
        result = layer("hello")

    assert result == "done"
    assert backend.clients[0].send_count == 1
    assert backend.clients[0].messages == ["hello"]


def test_no_post_turn_prompt_in_persistent_session_either():
    """Persistent sessions follow the same no-auto-prompt contract."""
    backend = FakeBackend([{"text": "first"}, {"text": "second"}])

    with patch("agent_flow.layers.create_backend", return_value=backend), \
            patch("agent_flow.layers._read_stdin",
                  side_effect=AssertionError("stdin must not be read")):
        with AgentLayer(
                _config(
                    session_mode="persistent",
                    print_activity=False,
                    human_input_enabled=True,
                )) as layer:
            first = layer("hello")
            second = layer("again")

    assert first == "first"
    assert second == "first"  # FakeBackend persistent client replays first
    assert backend.create_client_calls == 1
    assert backend.clients[0].send_count == 2


def test_ask_human_tool_registered_when_enabled():
    backend = FakeBackend([{"text": "ok"}])
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer("hello")

    tools = backend.clients[0].tools or []
    names = [getattr(t, "name", None) for t in tools]
    assert "ask_human" in names


def test_ask_human_tool_reads_reply_from_stdin():
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    tool = layer._build_ask_human_tool()
    assert tool.name == "ask_human"

    with patch("agent_flow.layers._read_stdin", return_value="blue"):
        result = anyio.run(tool.handler, {"question": "favorite color?"})

    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "blue"


def test_ask_human_tool_empty_reply_returns_no_response_marker():
    """Whitespace-only / empty reply is normalized so the agent gets a
    deterministic placeholder rather than an empty tool result."""
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    tool = layer._build_ask_human_tool()
    with patch("agent_flow.layers._read_stdin", return_value="   "):
        result = anyio.run(tool.handler, {"question": "anything?"})

    assert result["content"][0]["text"] == "(no response from human)"


def test_disallowed_tools_blocks_ask_user_question_when_hitl_enabled():
    backend = FakeBackend([{"text": "ok"}])
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer("hello")

    assert backend.clients[0].disallowed_tools == ["AskUserQuestion"]


def test_disallowed_tools_skipped_when_hitl_disabled():
    backend = FakeBackend([{"text": "ok"}])
    layer = AgentLayer(_config(print_activity=False))

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer("hello")

    assert backend.clients[0].disallowed_tools is None


def test_ask_human_tool_resolves_numeric_choice_to_label():
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    tool = layer._build_ask_human_tool()
    with patch("agent_flow.layers._read_stdin", return_value="2"):
        result = anyio.run(
            tool.handler, {
                "question":
                "Pick a color",
                "options": [
                    {
                        "label": "red",
                        "description": "warm"
                    },
                    {
                        "label": "blue",
                        "description": "cool"
                    },
                ],
            })

    assert result["content"][0]["text"] == "blue"


def test_ask_human_tool_schema_advertises_header_and_required_description():
    """Schema should mirror AskUserQuestion's chip+options shape.

    ``header`` is optional and ``description`` becomes required per
    option — the partial alignment with Claude Code's built-in
    ``AskUserQuestion`` schema. ``questions[]``, ``multiSelect``, and
    ``preview`` are intentionally absent (see the docstring on
    ``_build_ask_human_tool``).
    """
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    tool = layer._build_ask_human_tool()
    schema = tool.input_schema

    # Top-level shape: question + optional header + optional options
    assert schema["required"] == ["question"]
    assert "header" in schema["properties"]
    assert "options" in schema["properties"]
    # Per-option: both label AND description required
    item_schema = schema["properties"]["options"]["items"]
    assert set(item_schema["required"]) == {"label", "description"}
    # Confirm we did NOT pull in the bits we deliberately skipped.
    assert "questions" not in schema["properties"]
    assert "multiSelect" not in schema["properties"]
    assert "preview" not in item_schema.get("properties", {})


def test_ask_human_tool_propagates_header_into_request(capsys):
    """When the agent supplies ``header``, the panel title carries it."""
    layer = AgentLayer(_config(name="planner", human_input_enabled=True))

    tool = layer._build_ask_human_tool()
    with patch("agent_flow.layers._read_stdin", return_value="ok"):
        anyio.run(
            tool.handler, {
                "question":
                "Pick a stack",
                "header":
                "Stack",
                "options": [
                    {
                        "label": "Go",
                        "description": "compiled, simpler"
                    },
                    {
                        "label": "Rust",
                        "description": "compiled, stricter"
                    },
                ],
            })

    out = capsys.readouterr().out
    assert "ask_human" in out
    assert "Stack" in out  # header rendered in panel title


def test_ask_human_tool_returns_free_form_when_no_options():
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    tool = layer._build_ask_human_tool()
    with patch("agent_flow.layers._read_stdin",
               return_value="free-form answer"):
        result = anyio.run(tool.handler, {"question": "anything?"})

    assert result["content"][0]["text"] == "free-form answer"


def test_default_reader_resolves_numeric_choice_to_label():
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    options = (
        HumanRequestOption(label="alpha"),
        HumanRequestOption(label="beta"),
        HumanRequestOption(label="gamma"),
    )

    captured: list = []

    def fake_stdin(prompt: str = "") -> str:
        captured.append(prompt)
        return "2"

    request = HumanRequest(layer_name="agent",
                           prompt="Pick one",
                           options=options)

    with patch("agent_flow.layers._read_stdin", side_effect=fake_stdin):
        reply = anyio.run(layer._dispatch_human_request, request)

    assert reply == "beta"
    assert captured == ["> "]


def test_default_reader_falls_back_to_free_text_when_input_not_a_number():
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    request = HumanRequest(
        layer_name="agent",
        prompt="Pick one",
        options=(HumanRequestOption(label="alpha"),
                 HumanRequestOption(label="beta")),
    )

    with patch("agent_flow.layers._read_stdin", return_value="something else"):
        reply = anyio.run(layer._dispatch_human_request, request)

    assert reply == "something else"


def test_default_reader_matches_label_case_insensitively():
    """Typing the option label (any casing) returns the canonical label.

    Numeric and label-text replies must both resolve to the same canonical
    answer so the agent receives a stable string regardless of how the
    human phrased their pick.
    """
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    request = HumanRequest(
        layer_name="agent",
        prompt="Pick one",
        options=(HumanRequestOption(label="Approve"),
                 HumanRequestOption(label="Reject")),
    )

    with patch("agent_flow.layers._read_stdin", return_value="  approve "):
        reply = anyio.run(layer._dispatch_human_request, request)

    assert reply == "Approve"


def test_default_reader_free_form_overrides_options_for_unmatched_input():
    """Free-form text wins when it doesn't match any option.

    Confirms the "options-or-free-form" contract: the human is never
    forced to pick from the canned list when none of the choices captures
    what they want to say.
    """
    layer = AgentLayer(_config(print_activity=False, human_input_enabled=True))

    request = HumanRequest(
        layer_name="agent",
        prompt="What now?",
        options=(HumanRequestOption(label="Approve"),
                 HumanRequestOption(label="Reject")),
    )

    custom = "Approve, but only after we sync with legal."
    with patch("agent_flow.layers._read_stdin", return_value=custom):
        reply = anyio.run(layer._dispatch_human_request, request)

    assert reply == custom


def test_prompt_builder_can_replace_content():
    backend = FakeBackend([{"text": "rewritten"}])

    def prompt_builder(content):
        return AgentRequest(
            content="normalized prompt",
            system_prompt="custom",
            metadata={"source": "builder"},
        )

    layer = AgentLayer(_config(), prompt_builder=prompt_builder)

    with patch("agent_flow.layers.create_backend", return_value=backend):
        result = layer("ignored")

    assert result == "rewritten"
    assert backend.clients[0].messages == ["normalized prompt"]
    assert backend.clients[0].system_prompt == "custom"
