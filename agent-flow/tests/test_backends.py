from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
import claude_agent_sdk
import jsonschema
import pytest
from claude_agent_sdk import types as sdk_types
from claude_agent_sdk.types import (
    AssistantMessage,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from agent_flow import AgentLayer, AgentLayerConfig
from agent_flow.backends import claude_code as cc_mod
from agent_flow.backends import codex as codex_mod
from agent_flow.backends import create_backend
from agent_flow.backends.base import ResultEvent
from agent_flow.backends.claude_code import ClaudeCodeBackend, ClaudeCodeClient
from agent_flow.backends.codex import CodexBackend
from agent_flow.config import BackendConfig
from agent_flow.types import (
    AgentTextEvent,
    CompactBoundaryEvent,
    RateLimitWarningEvent,
    ServerToolCallEvent,
    SessionInitEvent,
    ThinkingEvent,
    ToolCallEvent,
)


class ServerToolUseBlock:
    def __init__(self, id: str, name: str, input: dict[str, Any]):
        self.id = id
        self.name = name
        self.input = input


ServerToolUseBlock = getattr(sdk_types, "ServerToolUseBlock", ServerToolUseBlock)


def _make_assistant_message(content, parent_tool_use_id: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        model="test-model",
        parent_tool_use_id=parent_tool_use_id,
    )


def _make_result_message(result: str, **kwargs) -> ResultMessage:
    defaults = dict(
        subtype="success",
        duration_ms=0,
        duration_api_ms=0,
        is_error=False,
        num_turns=0,
        session_id="test-session",
        result=result,
    )
    defaults.update(kwargs)
    return ResultMessage(**defaults)


def _make_tool_use_block(name: str, input: dict, id: str = "tool-1") -> ToolUseBlock:
    return ToolUseBlock(id=id, name=name, input=input)


class TestCreateBackend:
    def test_factory_returns_expected_backend_types(self):
        assert create_backend("claude-code").__class__.__name__ == "ClaudeCodeBackend"
        assert create_backend("codex").__class__.__name__ == "CodexBackend"

    def test_factory_rejects_unknown_backends(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("unknown")

    def test_factory_passes_disabled_skills(self):
        config = BackendConfig(
            kind="codex",
            model="gpt-6-astra",
            disabled_skills=("perf-optimization-casebook",),
        )
        assert create_backend(config)._disabled_skills == ("perf-optimization-casebook",)


class TestClaudeBackend:
    async def test_client_maps_tool_use_and_result_messages(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_assistant_message([_make_tool_use_block("Bash", {"command": "ls"})])
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hello")]

        sdk_client.query.assert_awaited_once_with("hello")
        assert len(events) == 2
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].name == "Bash"
        assert events[0].input == {"command": "ls"}
        assert events[0].tool_use_id == "tool-1"
        assert events[0].parent_tool_use_id is None
        assert events[0].agent_label is None
        assert isinstance(events[1], ResultEvent)
        assert events[1].text == "done"

    async def test_client_emits_session_init_event_from_init_system_message(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield SystemMessage(
                subtype="init",
                data={
                    "skills": [
                        "update-config",
                        "trtllm-agent-toolkit:perf-analysis",
                        12345,  # non-string entries are filtered
                    ],
                    "agents": ["Explore", "Plan"],
                    "plugins": [
                        {
                            "name": "code-review",
                            "path": "/x/code-review",
                            "source": "code-review@official",
                        },
                        {
                            "name": "trtllm-agent-toolkit",
                            "path": "/x/trtllm",
                            "source": "trtllm@x",
                        },
                        {"path": "/x/no-name"},  # missing name -> filtered
                    ],
                },
            )
            yield _make_assistant_message([_make_tool_use_block("Bash", {"command": "ls"})])
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]

        # The session-init event lands first, before any tool/result.
        assert isinstance(events[0], SessionInitEvent)
        assert events[0].skills == [
            "update-config",
            "trtllm-agent-toolkit:perf-analysis",
        ]
        assert events[0].agents == ["Explore", "Plan"]
        assert events[0].plugins == ["code-review", "trtllm-agent-toolkit"]
        # Subsequent events are unchanged.
        assert isinstance(events[1], ToolCallEvent)
        assert isinstance(events[-1], ResultEvent)

    async def test_client_ignores_non_init_system_messages(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield SystemMessage(subtype="other", data={"skills": ["should"]})
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        assert not any(isinstance(e, SessionInitEvent) for e in events)

    async def test_client_tags_subagent_events_with_task_label(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            # Main agent spawns an Explore subagent via the Task tool.
            yield _make_assistant_message(
                [
                    _make_tool_use_block(
                        "Task",
                        {
                            "subagent_type": "Explore",
                            "description": "search repo",
                            "prompt": "find foo",
                        },
                        id="task-42",
                    )
                ]
            )
            # Subagent executes a Bash call; SDK marks it with the parent
            # tool_use id of the Task call above.
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "rg foo"}, id="tool-9")],
                parent_tool_use_id="task-42",
            )
            # Subagent emits a text message before returning.
            yield _make_assistant_message(
                [TextBlock(text="found it")],
                parent_tool_use_id="task-42",
            )
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("go")]

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        text_events = [e for e in events if isinstance(e, AgentTextEvent)]

        # The Task call itself is a main-agent event (no parent), but the
        # backend records its label so child events can resolve it.
        assert tool_events[0].name == "Task"
        assert tool_events[0].parent_tool_use_id is None
        assert tool_events[0].agent_label is None

        # The Bash call inside the subagent inherits the Task label.
        assert tool_events[1].name == "Bash"
        assert tool_events[1].parent_tool_use_id == "task-42"
        assert tool_events[1].agent_label == "Explore"

        # The subagent's text message is also tagged.
        assert len(text_events) == 1
        assert text_events[0].parent_tool_use_id == "task-42"
        assert text_events[0].agent_label == "Explore"

    async def test_client_falls_back_to_description_when_no_subagent_type(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_assistant_message(
                [
                    _make_tool_use_block(
                        "Task",
                        {"description": "fix bug"},
                        id="task-1",
                    )
                ]
            )
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "ls"}, id="t-2")],
                parent_tool_use_id="task-1",
            )
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("go")]
        sub_event = next(e for e in events if isinstance(e, ToolCallEvent) and e.name == "Bash")
        assert sub_event.agent_label == "fix bug"

    async def test_client_uses_generic_label_for_unrecognized_parent(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            # Bash call references a parent the client never observed.
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "ls"}, id="t-1")],
                parent_tool_use_id="phantom-task",
            )
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("go")]
        bash_event = next(e for e in events if isinstance(e, ToolCallEvent) and e.name == "Bash")
        assert bash_event.parent_tool_use_id == "phantom-task"
        assert bash_event.agent_label == "subagent"

    async def test_client_recognizes_agent_tool_and_strips_plugin_prefix(self):
        # The Claude Code CLI surfaces the subagent-spawning tool as
        # ``Agent`` (not ``Task``), and ``subagent_type`` may include a
        # plugin namespace prefix that should be hidden from the label.

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_assistant_message(
                [
                    _make_tool_use_block(
                        "Agent",
                        {
                            "subagent_type": "trtllm-agent-toolkit:exec-compile-specialist",
                            "description": "Compile TRT-LLM",
                            "prompt": "compile it",
                        },
                        id="agent-77",
                    )
                ]
            )
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "ls"}, id="t-3")],
                parent_tool_use_id="agent-77",
            )
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("go")]
        sub_event = next(e for e in events if isinstance(e, ToolCallEvent) and e.name == "Bash")
        assert sub_event.parent_tool_use_id == "agent-77"
        assert sub_event.agent_label == "exec-compile-specialist"

    async def test_client_extracts_usage_and_cost_from_result_message(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()
        sdk_client.get_context_usage = AsyncMock(
            return_value={
                "totalTokens": 50000,
                "maxTokens": 200000,
                "percentage": 25.0,
            }
        )

        async def receive_messages():
            yield _make_result_message(
                "done",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                },
                total_cost_usd=0.0123,
                num_turns=3,
                duration_ms=1500,
            )

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]

        assert len(events) == 1
        result = events[0]
        assert isinstance(result, ResultEvent)
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_creation_tokens == 20
        assert result.usage.cache_read_tokens == 30
        assert result.usage.total_tokens == 200
        assert result.usage.cost_usd == 0.0123
        assert result.usage.num_turns == 3
        assert result.usage.duration_ms == 1500
        assert result.usage.context_tokens == 50000
        assert result.usage.context_window == 200000
        assert result.usage.context_percentage == 25.0

    async def test_client_survives_context_usage_errors(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()
        sdk_client.get_context_usage = AsyncMock(side_effect=RuntimeError("boom"))

        async def receive_messages():
            yield _make_result_message("done", usage={"input_tokens": 1})

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]

        assert len(events) == 1
        assert events[0].usage is not None
        assert events[0].usage.context_tokens is None
        assert events[0].usage.context_percentage is None

    async def test_client_lists_skills_from_server_info_without_a_turn(self):
        # The whole point of sourcing skills from ``get_server_info``: it
        # is a control request, so nothing is queried and no turn is spent.
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()
        sdk_client.get_server_info = AsyncMock(
            return_value={
                "commands": [
                    {"name": "trtllm-agent-toolkit:internal-perf-sol-analysis"},
                    {"name": "perf-analyze", "description": "(project) ..."},
                    {"name": "clear"},
                ],
                "agents": [{"name": "Explore"}],
            }
        )
        client = ClaudeCodeClient(sdk_client)

        skills = await client.list_available_skills()

        assert skills == [
            "trtllm-agent-toolkit:internal-perf-sol-analysis",
            "perf-analyze",
            "clear",
        ]
        sdk_client.query.assert_not_awaited()

    async def test_client_skips_command_entries_without_a_name(self):
        sdk_client = MagicMock()
        sdk_client.get_server_info = AsyncMock(
            return_value={"commands": [{"name": "kept"}, {"description": "no name"}, "junk"]}
        )
        client = ClaudeCodeClient(sdk_client)

        assert await client.list_available_skills() == ["kept"]

    @pytest.mark.parametrize(
        "server_info",
        [
            None,
            "not-a-dict",
            {"agents": []},
            {"commands": "not-a-list"},
        ],
    )
    async def test_client_reports_no_skill_list_when_server_info_is_unusable(self, server_info):
        # ``None`` means "we did not learn what is installed" — callers
        # must not read an empty list as evidence of an empty environment.
        sdk_client = MagicMock()
        sdk_client.get_server_info = AsyncMock(return_value=server_info)
        client = ClaudeCodeClient(sdk_client)

        assert await client.list_available_skills() is None

    async def test_client_reports_no_skill_list_when_server_info_raises(self):
        sdk_client = MagicMock()
        sdk_client.get_server_info = AsyncMock(side_effect=RuntimeError("boom"))
        client = ClaudeCodeClient(sdk_client)

        assert await client.list_available_skills() is None

    async def test_client_reports_no_skill_list_when_sdk_lacks_server_info(self):
        # An older SDK without the control request degrades to "cannot
        # say" rather than raising on the workflow's launch path.
        sdk_client = MagicMock(spec=["query", "receive_messages"])
        client = ClaudeCodeClient(sdk_client)

        assert await client.list_available_skills() is None

    async def test_client_emits_thinking_event_from_thinking_block(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_assistant_message(
                [
                    ThinkingBlock(thinking="  reasoning step  ", signature="sig"),
                    TextBlock(text="answer"),
                ]
            )
            yield _make_result_message("answer")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]

        thinking = [e for e in events if isinstance(e, ThinkingEvent)]
        assert len(thinking) == 1
        # Whitespace must be stripped before surfacing.
        assert thinking[0].text == "reasoning step"
        # Empty thinking blocks are dropped — verify the loop doesn't choke.
        text_events = [e for e in events if isinstance(e, AgentTextEvent)]
        assert text_events and text_events[0].text == "answer"

    async def test_client_drops_empty_thinking_blocks(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_assistant_message([ThinkingBlock(thinking="   ", signature="sig")])
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        assert not any(isinstance(e, ThinkingEvent) for e in events)

    async def test_client_emits_server_tool_call_event(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_assistant_message(
                [
                    ServerToolUseBlock(
                        id="srv-1",
                        name="web_search",
                        input={"query": "claude code"},
                    ),
                ]
            )
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]

        server_calls = [e for e in events if isinstance(e, ServerToolCallEvent)]
        assert len(server_calls) == 1
        assert server_calls[0].name == "web_search"
        assert server_calls[0].tool_use_id == "srv-1"
        # Server tool calls are not subagent spawns; nothing should be
        # routed under a fake parent label.
        assert server_calls[0].parent_tool_use_id is None
        assert server_calls[0].agent_label is None

    async def test_client_emits_rate_limit_warning_event(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield RateLimitEvent(
                rate_limit_info=RateLimitInfo(
                    status="allowed_warning",
                    rate_limit_type="five_hour",
                    resets_at=1700000000,
                    utilization=0.85,
                ),
                uuid="rl-1",
                session_id="sess-1",
            )
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        warnings = [e for e in events if isinstance(e, RateLimitWarningEvent)]
        assert len(warnings) == 1
        assert warnings[0].status == "allowed_warning"
        assert warnings[0].rate_limit_type == "five_hour"
        assert warnings[0].resets_at == 1700000000
        assert warnings[0].utilization == 0.85

    async def test_client_emits_compact_boundary_event(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield SystemMessage(
                subtype="compact_boundary",
                data={
                    "trigger": "auto",
                    "pre_tokens": 150000,
                },
            )
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        boundaries = [e for e in events if isinstance(e, CompactBoundaryEvent)]
        assert len(boundaries) == 1
        assert boundaries[0].trigger == "auto"
        assert boundaries[0].pre_tokens == 150000

    async def test_client_compact_boundary_handles_missing_fields(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield SystemMessage(subtype="compact_boundary", data={})
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        boundary = next(e for e in events if isinstance(e, CompactBoundaryEvent))
        assert boundary.trigger is None
        assert boundary.pre_tokens is None

    async def test_client_raises_on_assistant_message_error(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield AssistantMessage(
                content=[],
                model="test-model",
                error="billing_error",
            )

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        with pytest.raises(RuntimeError, match="Claude Code turn failed: billing_error"):
            async for _ in client.send_message("hi"):
                pass

    async def test_client_swallows_assistant_error_after_result(self):
        # An AssistantMessage.error AFTER a ResultMessage should not poison
        # the run — the existing got_result guard covers this; verify it
        # still holds with the new error path in place.

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_result_message("done")
            yield AssistantMessage(
                content=[],
                model="test-model",
                error="rate_limit",
            )

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        assert any(isinstance(e, ResultEvent) for e in events)

    async def test_client_error_message_names_the_adjacent_sdk_fields(self):
        # ``AssistantMessage.error`` is frequently the bare string
        # "unknown", which tells an operator nothing. The raised message
        # must carry the surrounding fields so a dead campaign stage is
        # diagnosable from the run log alone.

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield AssistantMessage(
                content=[TextBlock(text="partial answer")],
                model="claude-test",
                error="unknown",
                stop_reason="max_tokens",
                usage={"input_tokens": 12},
                uuid="msg-uuid",
                session_id="sess-1",
            )

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        with pytest.raises(RuntimeError) as excinfo:
            async for _ in client.send_message("hi"):
                pass

        message = str(excinfo.value)
        assert "Claude Code turn failed: unknown" in message
        assert "model='claude-test'" in message
        assert "stop_reason='max_tokens'" in message
        assert "uuid='msg-uuid'" in message
        assert "session_id='sess-1'" in message
        assert "input_tokens" in message
        assert "content_blocks=['TextBlock']" in message
        assert "partial answer" in message

    def test_assistant_error_detail_truncates_partial_text(self):
        # A failed turn can carry a large partial answer; the run log
        # gets a bounded excerpt, not the whole thing.
        message = AssistantMessage(
            content=[TextBlock(text="x" * 1000)],
            model="claude-test",
            error="unknown",
        )
        detail = cc_mod._assistant_error_detail(message)
        assert "x" * 400 in detail
        assert "x" * 401 not in detail

    def test_assistant_error_detail_reports_block_kinds_without_text(self):
        # A turn that died mid-tool-call has no text to quote, but the
        # block kinds still say how far it got.
        message = AssistantMessage(
            content=[_make_tool_use_block("Bash", {"command": "ls"})],
            model="claude-test",
            error="unknown",
        )
        detail = cc_mod._assistant_error_detail(message)
        assert "content_blocks=['ToolUseBlock']" in detail
        assert "partial_text" not in detail

    def test_assistant_error_detail_empty_when_no_context(self):
        # Nothing to add beyond the bare error string, so no trailing
        # parenthetical is appended at all.
        assert cc_mod._assistant_error_detail(SimpleNamespace()) == ""

    async def test_client_threads_error_fields_through_result_event(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_result_message(
                "",
                is_error=True,
                errors=["transient upstream"],
                permission_denials=[{"tool": "Bash"}],
            )

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        result = next(e for e in events if isinstance(e, ResultEvent))
        assert result.is_error is True
        assert result.errors == ["transient upstream"]
        assert result.permission_denials == [{"tool": "Bash"}]

    async def test_client_result_event_defaults_when_no_errors(self):
        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_messages():
            yield _make_result_message("done")

        sdk_client.receive_messages = receive_messages
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        result = next(e for e in events if isinstance(e, ResultEvent))
        assert result.is_error is False
        assert result.errors == []
        assert result.permission_denials == []


# A recursive explicit JSON Schema: the form ``normalize_input_schema`` points users
# to when a TypedDict refers to itself.
_RECURSIVE_NODE_SCHEMA = {
    "$ref": "#/$defs/node",
    "$defs": {
        "node": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            },
            "required": ["name"],
        }
    },
}


def _session_state(state: str) -> SystemMessage:
    return SystemMessage(subtype="session_state_changed", data={"state": state})


def _task_started(task_id: str, task_type: str) -> TaskStartedMessage:
    return TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id=task_id,
        task_type=task_type,
        description="bg",
        uuid="u",
        session_id="s",
    )


def _task_notification(task_id: str) -> TaskNotificationMessage:
    return TaskNotificationMessage(
        subtype="task_notification",
        data={},
        task_id=task_id,
        status="completed",
        output_file="/tmp/out",
        summary="done",
        uuid="u",
        session_id="s",
    )


def _background_run(tool_name: str, task_type: str, *, idle_before_result: bool = False):
    """The stream the CLI sends when the model backgrounds work in one turn.

    The first turn launches the task and ends with a result while the task is
    still running. Its completion wakes the session for a second turn whose
    result carries the real answer; only then does the CLI report "idle".
    """
    final_turn = [
        _make_assistant_message([TextBlock(text="background work merged")]),
        _make_result_message("final answer", num_turns=1),
    ]
    tail = [_session_state("idle")]
    if idle_before_result:
        final_turn, tail = final_turn[:1] + tail, final_turn[1:]
    return [
        _session_state("running"),
        _make_assistant_message(
            [_make_tool_use_block(tool_name, {"run_in_background": True}, id="bg-1")]
        ),
        _task_started("t1", task_type),
        _make_result_message("launched, waiting", num_turns=1),
        _task_notification("t1"),
        *final_turn,
        *tail,
    ]


_UNTIL_STOPPED = object()


class FakeBackgroundSdkClient:
    """``ClaudeSDKClient`` stand-in that replays one background run.

    ``_UNTIL_STOPPED`` in the script blocks the stream until ``stop_task`` is
    called. The stream stays open after the script ends, as the CLI's does;
    ``closed_after`` records how much of it had been consumed when the
    session was torn down.
    """

    def __init__(self, messages):
        self.messages = messages
        self.consumed = 0
        self.closed_after: int | None = None
        self.stopped: list[str] = []
        self._stop = anyio.Event()

    async def query(self, message):
        pass

    async def stop_task(self, task_id):
        self.stopped.append(task_id)
        self._stop.set()

    async def receive_messages(self):
        while self.consumed < len(self.messages):
            message = self.messages[self.consumed]
            if message is _UNTIL_STOPPED:
                await self._stop.wait()
                self.consumed += 1
                continue
            self.consumed += 1
            yield message
        await anyio.sleep_forever()


class TestClaudeBackgroundWork:
    @pytest.fixture(autouse=True)
    def _short_settle(self, monkeypatch):
        monkeypatch.setattr(cc_mod, "_SETTLE_S", 0.05)

    @pytest.mark.parametrize(
        ("tool_name", "task_type", "idle_before_result"),
        [("Agent", "local_agent", False), ("Workflow", "local_workflow", True)],
    )
    async def test_send_message_waits_for_background_task(
        self, tool_name, task_type, idle_before_result
    ):
        sdk_client = FakeBackgroundSdkClient(
            _background_run(tool_name, task_type, idle_before_result=idle_before_result)
        )
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("optimize")]

        assert sdk_client.consumed == len(sdk_client.messages)
        results = [e for e in events if isinstance(e, ResultEvent)]
        assert len(results) == 1
        assert results[0].text == "final answer"
        assert results[0].usage.num_turns == 2
        texts = [e.text for e in events if isinstance(e, AgentTextEvent)]
        assert texts == ["background work merged"]

    async def test_send_message_waits_for_shell_a_subagent_left_running(self):
        # Recorded from CLI 2.1.286: the subagent backgrounds a shell and
        # finishes, the session goes "idle", then the shell's completion wakes
        # the subagent and the session again for the real answer.
        sdk_client = FakeBackgroundSdkClient(
            [
                _session_state("running"),
                _make_assistant_message([_make_tool_use_block("Agent", {}, id="bg-1")]),
                _task_started("agent", "local_agent"),
                _make_result_message("LAUNCHED"),
                _task_started("shell", "local_bash"),
                _task_notification("agent"),
                _make_assistant_message([TextBlock(text="still waiting")]),
                _make_result_message("waiting for the shell"),
                _session_state("idle"),
                _task_notification("shell"),
                _task_started("agent", "local_agent"),
                _task_notification("agent"),
                _session_state("running"),
                _make_assistant_message([TextBlock(text="FINAL: BG_OK")]),
                _make_result_message("FINAL: BG_OK"),
                _session_state("idle"),
            ]
        )
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("optimize")]

        assert sdk_client.consumed == len(sdk_client.messages)
        assert [e.text for e in events if isinstance(e, ResultEvent)] == ["FINAL: BG_OK"]

    async def test_send_message_stops_tasks_left_running_past_the_ceiling(self, monkeypatch):
        monkeypatch.setattr(cc_mod, "_BACKGROUND_WAIT_CEILING_S", 0.01)
        sdk_client = FakeBackgroundSdkClient(
            [
                _session_state("running"),
                _make_assistant_message([_make_tool_use_block("Bash", {}, id="srv")]),
                _task_started("server", "local_bash"),
                _make_result_message("server is up"),
                _session_state("idle"),
                _UNTIL_STOPPED,
                TaskUpdatedMessage(
                    subtype="task_updated",
                    data={},
                    task_id="server",
                    patch={"status": "killed"},
                    status="killed",
                ),
            ]
        )
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("serve")]

        assert sdk_client.stopped == ["server"]
        assert sdk_client.consumed == len(sdk_client.messages)
        assert [e.text for e in events if isinstance(e, ResultEvent)] == ["server is up"]

    async def test_send_message_stops_leftover_waiters_once_the_role_delivered(self):
        # The opt-010 case: the role called its deliverable tool and ended its
        # turn, leaving a shell waiter whose pattern never appears. It is
        # stopped at once rather than after the ceiling.
        sdk_client = FakeBackgroundSdkClient(
            [
                _session_state("running"),
                _make_assistant_message([_make_tool_use_block("Bash", {}, id="w")]),
                _task_started("waiter", "local_bash"),
                _make_assistant_message(
                    [_make_tool_use_block("mcp__agent-tools__append_optimizer_progress", {})]
                ),
                _make_result_message("done"),
                _session_state("idle"),
                _UNTIL_STOPPED,
                _task_notification("waiter"),
            ]
        )
        client = ClaudeCodeClient(sdk_client, required_tools=("append_optimizer_progress",))

        with anyio.fail_after(5):
            events = [event async for event in client.send_message("optimize")]

        assert sdk_client.stopped == ["waiter"]
        assert sdk_client.consumed == len(sdk_client.messages)
        assert [e.text for e in events if isinstance(e, ResultEvent)] == ["done"]

    async def test_send_message_keeps_waiting_on_workflow_after_delivery(self):
        # Delivery stops only shells and monitors; a workflow still running is
        # pending work whose result the parent has to handle.
        sdk_client = FakeBackgroundSdkClient(
            [
                _session_state("running"),
                _make_assistant_message([_make_tool_use_block("Workflow", {}, id="wf")]),
                _task_started("wf", "local_workflow"),
                _task_started("waiter", "local_bash"),
                _make_assistant_message([_make_tool_use_block("append_qa_progress", {})]),
                _make_result_message("launched"),
                _session_state("idle"),
                _task_notification("wf"),
                _session_state("running"),
                _make_assistant_message([TextBlock(text="merged")]),
                _make_result_message("final"),
                _session_state("idle"),
                _UNTIL_STOPPED,
                _task_notification("waiter"),
            ]
        )
        client = ClaudeCodeClient(sdk_client, required_tools=("append_qa_progress",))

        with anyio.fail_after(5):
            events = [event async for event in client.send_message("qa")]

        assert sdk_client.stopped == ["waiter"]
        assert [e.text for e in events if isinstance(e, ResultEvent)] == ["final"]

    async def test_send_message_waits_on_waiters_before_delivery(self, monkeypatch):
        # Before the deliverable tool is called a waiter may be how the role
        # gets woken, so it is only stopped at the ceiling.
        monkeypatch.setattr(cc_mod, "_BACKGROUND_WAIT_CEILING_S", 0.2)
        sdk_client = FakeBackgroundSdkClient(
            [
                _session_state("running"),
                _make_assistant_message([_make_tool_use_block("Monitor", {}, id="m")]),
                _task_started("monitor", "monitor"),
                _make_result_message("waiting for the job"),
                _session_state("idle"),
                _UNTIL_STOPPED,
                _task_notification("monitor"),
            ]
        )
        client = ClaudeCodeClient(sdk_client, required_tools=("append_optimizer_progress",))

        start = anyio.current_time()
        events = [event async for event in client.send_message("optimize")]

        assert anyio.current_time() - start >= 0.2
        assert sdk_client.stopped == ["monitor"]
        assert [e.text for e in events if isinstance(e, ResultEvent)] == ["waiting for the job"]

    async def test_subagent_delivery_does_not_count_for_the_role(self):
        sdk_client = FakeBackgroundSdkClient([])
        client = ClaudeCodeClient(sdk_client, required_tools=("append_optimizer_progress",))
        sdk_client.messages = [
            _session_state("running"),
            _make_assistant_message([_make_tool_use_block("Agent", {}, id="sub")]),
            _make_assistant_message(
                [_make_tool_use_block("append_optimizer_progress", {})], parent_tool_use_id="sub"
            ),
            _task_started("waiter", "local_bash"),
            _make_result_message("waiting"),
            _session_state("idle"),
        ]

        with anyio.move_on_after(0.3):
            [event async for event in client.send_message("optimize")]

        assert sdk_client.stopped == []

    async def test_send_message_raises_on_a_continuation_error(self):
        # A result that only ends the first turn must not mask a failure in
        # the turn its background work wakes up.
        sdk_client = FakeBackgroundSdkClient(
            [
                _session_state("running"),
                _make_assistant_message([_make_tool_use_block("Agent", {}, id="bg-1")]),
                _task_started("t1", "local_agent"),
                _make_result_message("launched, waiting"),
                _task_notification("t1"),
                AssistantMessage(
                    content=[], model="test-model", parent_tool_use_id=None, error="billing_error"
                ),
            ]
        )
        client = ClaudeCodeClient(sdk_client)

        with pytest.raises(RuntimeError, match="billing_error"):
            [event async for event in client.send_message("optimize")]

    async def test_send_message_ends_at_result_without_session_state(self):
        sdk_client = FakeBackgroundSdkClient([_make_result_message("done")])
        client = ClaudeCodeClient(sdk_client)
        # Without session state nothing says the run is over but the result.

        events = [event async for event in client.send_message("hi")]

        assert [e.text for e in events if isinstance(e, ResultEvent)] == ["done"]

    @pytest.mark.parametrize(
        ("tool_name", "task_type"), [("Agent", "local_agent"), ("Workflow", "local_workflow")]
    )
    def test_layer_keeps_session_until_background_task_finishes(
        self, monkeypatch, tool_name, task_type
    ):
        sdk_client = FakeBackgroundSdkClient(_background_run(tool_name, task_type))

        class FakeSdkClientContext:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return sdk_client

            async def __aexit__(self, *args):
                sdk_client.closed_after = sdk_client.consumed

        monkeypatch.setattr(cc_mod, "ClaudeSDKClient", FakeSdkClientContext)
        layer = AgentLayer(
            AgentLayerConfig(
                name="analyzer",
                system_prompt="You are helpful.",
                backend=BackendConfig(kind="claude-code", model="test-model"),
                print_activity=False,
            )
        )

        assert layer("optimize") == "final answer"
        assert sdk_client.closed_after == len(sdk_client.messages)


class TestClaudeBackendCreateClient:
    def test_reasoning_effort_override(self):
        assert ClaudeCodeBackend(reasoning_effort="medium").reasoning_effort() == "medium"

    async def test_framework_tools_keep_annotations_and_independent_handlers(self):
        from agent_flow.tools import tool

        @tool("resource", "Read a resource", {"path": str}, annotations={"readOnlyHint": True})
        async def resource(args):
            return {"content": [{"type": "resource", "resource": {"text": args["path"]}}]}

        @tool("image", "Read an image", {}, annotations={"maxResultSizeChars": 1024})
        async def image(args):
            return {"content": [{"type": "image", "data": "YWJj", "mimeType": "image/png"}]}

        resource_sdk, image_sdk = cc_mod._sdk_tools([resource, image])
        assert resource_sdk.input_schema["properties"]["path"] == {"type": "string"}
        assert resource_sdk.annotations.model_dump(by_alias=True)["readOnlyHint"] is True
        assert image_sdk.annotations.maxResultSizeChars == 1024
        assert (await resource_sdk.handler({"path": "document"}))["content"] == [
            {"type": "text", "text": "document"}
        ]
        assert (await image_sdk.handler({}))["content"] == [
            {"type": "image", "data": "YWJj", "mimeType": "image/png"}
        ]

    @pytest.mark.parametrize(
        "schema, accepted, rejected",
        [
            (
                {"type": "object", "additionalProperties": {"type": "string"}},
                {"key": "value"},
                {"key": 1},
            ),
            (
                _RECURSIVE_NODE_SCHEMA,
                {"name": "root", "children": [{"name": "leaf"}]},
                {"name": "root", "children": [{"children": []}]},
            ),
        ],
    )
    def test_explicit_json_schema_survives_sdk_schema_builder(self, schema, accepted, rejected):
        from agent_flow.tools import tool

        @tool("explicit", "Explicit JSON Schema", schema)
        async def explicit(args):
            return {"content": []}

        [sdk_tool] = cc_mod._sdk_tools([explicit])
        # ``_build_input_schema`` is the pinned SDK's wire-schema builder. A dict without a
        # string ``type`` and a ``properties`` key is re-read as Python shorthand, so the
        # schema's own keywords would be advertised, and then validated on every call, as
        # required parameters. The framework must hand it a shape it passes through as is.
        advertised = claude_agent_sdk._build_input_schema(sdk_tool)
        assert advertised is sdk_tool.input_schema
        jsonschema.validate(instance=accepted, schema=advertised)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=rejected, schema=advertised)

    async def _capture_options(self, monkeypatch, backend=None, **kwargs):
        # Stand-in for ``ClaudeSDKClient`` that just records the options
        # ``create_client`` would have launched the real SDK with.

        captured: dict[str, Any] = {}

        class FakeSdkClient:
            def __init__(self, options):
                captured["options"] = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        monkeypatch.setattr(cc_mod, "ClaudeSDKClient", FakeSdkClient)
        # Skip MCP server creation; ``tools=None`` means it isn't called,
        # but we patch defensively in case that changes.
        monkeypatch.setattr(cc_mod, "create_sdk_mcp_server", lambda **_: object())

        backend = backend or ClaudeCodeBackend()
        async with backend.create_client(system_prompt="hi", model="claude-test", **kwargs):
            pass
        return captured["options"]

    async def test_create_client_denies_disabled_skills(self, monkeypatch):
        options = await self._capture_options(
            monkeypatch,
            backend=ClaudeCodeBackend(
                disabled_skills=("perf-optimization-casebook",),
            ),
        )
        assert options.disallowed_tools == ["Skill(perf-optimization-casebook)"]

    async def test_create_client_disables_bash_sandbox(self, monkeypatch):
        # ``danger_full_access`` analogue for Claude Code: the bash
        # sandbox is fully off so commands can touch anything on disk
        # or the network without restrictions.
        options = await self._capture_options(monkeypatch)
        assert options.sandbox == {"enabled": False}

    async def test_create_client_hands_required_tools_to_the_client(self, monkeypatch):
        class FakeSdkClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        monkeypatch.setattr(cc_mod, "ClaudeSDKClient", FakeSdkClient)
        async with ClaudeCodeBackend().create_client(
            system_prompt="hi", model="claude-test", required_tools=("append_qa_progress",)
        ) as client:
            assert client._required_tools == ("append_qa_progress",)

    async def test_create_client_asks_cli_for_session_state(self, monkeypatch):
        options = await self._capture_options(monkeypatch)
        assert options.env["CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS"] == "1"

    async def test_create_client_uses_bypass_permission_mode(self, monkeypatch):
        # ``bypassPermissions`` is the SDK setting that suppresses every
        # permission prompt the CLI would otherwise raise.
        options = await self._capture_options(monkeypatch)
        assert options.permission_mode == "bypassPermissions"

    async def test_create_client_omits_shadowed_permission_callback(self, monkeypatch):
        # The pinned SDK bypasses this callback in bypassPermissions mode.
        options = await self._capture_options(monkeypatch)
        assert options.can_use_tool is None

    async def test_create_client_no_extra_mcp_servers_by_default(self, monkeypatch):
        # Without ``extra_mcp_servers``, ``mcp_servers`` stays empty
        # (the in-process ``agent-tools`` server is only added when
        # ``tools`` is provided).
        options = await self._capture_options(monkeypatch)
        assert options.mcp_servers == {}

    async def test_create_client_extra_mcp_servers_merged_alongside_agent_tools(self, monkeypatch):
        # ``extra_mcp_servers`` values are passed verbatim into
        # ``ClaudeAgentOptions.mcp_servers``; the in-process
        # ``agent-tools`` server is layered on top when ``tools`` is set.
        options = await self._capture_options(
            monkeypatch,
            tools=[
                SimpleNamespace(
                    name="test",
                    description="test",
                    input_schema={"type": "object", "properties": {}},
                    handler=AsyncMock(),
                )
            ],
            extra_mcp_servers={
                "knowledge-base": {"type": "http", "url": "https://example.test/mcp"},
            },
        )
        assert set(options.mcp_servers.keys()) == {"knowledge-base", "agent-tools"}
        # The external knowledge-base entry passes through verbatim.
        assert options.mcp_servers["knowledge-base"] == {
            "type": "http",
            "url": "https://example.test/mcp",
        }
        # ``agent-tools`` is the in-process SDK MCP server, an opaque
        # object (the helper patches ``create_sdk_mcp_server`` to
        # return ``object()``); it must not be a dict-shaped server
        # config — that would mean the external config silently
        # overwrote the in-process server.
        assert not isinstance(options.mcp_servers["agent-tools"], dict)

    async def test_create_client_extra_mcp_servers_without_tools(self, monkeypatch):
        # With only ``extra_mcp_servers`` and no ``tools``, ``mcp_servers``
        # carries just the external entries — no implicit ``agent-tools``.
        options = await self._capture_options(
            monkeypatch,
            extra_mcp_servers={
                "knowledge-base": {"type": "http", "url": "https://example.test/mcp"},
            },
        )
        assert set(options.mcp_servers.keys()) == {"knowledge-base"}

    async def test_create_client_rejects_reserved_agent_tools_key(self, monkeypatch):
        # The ``agent-tools`` key is reserved for the in-process MCP
        # server built from ``tools``; user-supplied values under that
        # name would silently get overwritten, so we surface it as an
        # explicit error.
        backend = ClaudeCodeBackend()
        with pytest.raises(ValueError, match="agent-tools"):
            async with backend.create_client(
                system_prompt="hi",
                model="claude-test",
                extra_mcp_servers={"agent-tools": {"type": "http", "url": "x"}},
            ):
                pass


class TestResolveCodexBin:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        fake = tmp_path / "codex"
        fake.write_text("#!/bin/sh\n")
        monkeypatch.setenv("CODEX_BIN", str(fake))
        monkeypatch.setitem(
            sys.modules,
            "codex_cli_bin",
            SimpleNamespace(bundled_codex_path=lambda: Path("/nope/should-not-be-used")),
        )

        assert codex_mod._resolve_codex_bin() == str(fake)

    def test_env_override_missing_file_raises(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing"
        monkeypatch.setenv("CODEX_BIN", str(missing))

        with pytest.raises(FileNotFoundError, match="CODEX_BIN"):
            codex_mod._resolve_codex_bin()

    def test_prefers_bundled_pinned_runtime(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        bundled = tmp_path / "bundled-codex"
        bundled.write_text("#!/bin/sh\n")
        monkeypatch.setitem(
            sys.modules, "codex_cli_bin", SimpleNamespace(bundled_codex_path=lambda: bundled)
        )
        monkeypatch.setattr(codex_mod.shutil, "which", lambda _: "/some/system/codex")

        assert codex_mod._resolve_codex_bin() == str(bundled)

    def test_falls_back_to_path_when_bundle_missing(self, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        sys.modules.pop("codex_cli_bin", None)
        # Make the import fail even if the package is installed for real.
        monkeypatch.setitem(sys.modules, "codex_cli_bin", None)
        monkeypatch.setattr(codex_mod.shutil, "which", lambda _: "/usr/bin/codex")

        assert codex_mod._resolve_codex_bin() == "/usr/bin/codex"
        sys.modules.pop("codex_cli_bin", None)

    def test_raises_when_nothing_found(self, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        monkeypatch.setitem(sys.modules, "codex_cli_bin", None)
        monkeypatch.setattr(codex_mod.shutil, "which", lambda _: None)

        with pytest.raises(FileNotFoundError, match="Codex CLI"):
            codex_mod._resolve_codex_bin()
        sys.modules.pop("codex_cli_bin", None)


class TestClaudeBackendVersion:
    def _reset_cache(self, monkeypatch):
        monkeypatch.setattr(cc_mod, "_VERSION_CACHE", None)

    def test_version_combines_cli_and_sdk(self, monkeypatch):
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(cc_mod, "_claude_cli_version", lambda: "2.1.123")
        monkeypatch.setattr(cc_mod, "_claude_sdk_version", lambda: "0.1.65")

        assert ClaudeCodeBackend().version() == "cli 2.1.123 · sdk 0.1.65"

    def test_version_handles_missing_cli(self, monkeypatch):
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(cc_mod, "_claude_cli_version", lambda: "")
        monkeypatch.setattr(cc_mod, "_claude_sdk_version", lambda: "0.1.65")

        assert ClaudeCodeBackend().version() == "sdk 0.1.65"

    def test_version_returns_empty_when_nothing_resolves(self, monkeypatch):
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(cc_mod, "_claude_cli_version", lambda: "")
        monkeypatch.setattr(cc_mod, "_claude_sdk_version", lambda: "")

        assert ClaudeCodeBackend().version() == ""

    def test_cli_version_parses_first_token_of_stdout(self, monkeypatch):
        # The Claude CLI prints e.g. "2.1.123 (Claude Code)" — only the
        # leading token is the version.

        monkeypatch.setattr(cc_mod, "_find_claude_cli", lambda: "/bin/claude")

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stdout="2.1.123 (Claude Code)\n", stderr="")

        monkeypatch.setattr(cc_mod.subprocess, "run", fake_run)
        assert cc_mod._claude_cli_version() == "2.1.123"

    def test_cli_version_returns_empty_on_failure(self, monkeypatch):
        monkeypatch.setattr(cc_mod, "_find_claude_cli", lambda: "/bin/claude")

        def fake_run(cmd, **kwargs):
            raise OSError("nope")

        monkeypatch.setattr(cc_mod.subprocess, "run", fake_run)
        assert cc_mod._claude_cli_version() == ""

    def test_cli_version_returns_empty_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(cc_mod, "_find_claude_cli", lambda: None)
        # Should never reach subprocess; explode if it does.
        monkeypatch.setattr(
            cc_mod.subprocess,
            "run",
            lambda *a, **k: pytest.fail("subprocess.run should not be called"),
        )
        assert cc_mod._claude_cli_version() == ""

    def test_version_is_cached_across_calls(self, monkeypatch):
        self._reset_cache(monkeypatch)
        calls: list[int] = []

        def cli_version():
            calls.append(1)
            return "2.1.123"

        monkeypatch.setattr(cc_mod, "_claude_cli_version", cli_version)
        monkeypatch.setattr(cc_mod, "_claude_sdk_version", lambda: "0.1.65")

        first = ClaudeCodeBackend().version()
        second = ClaudeCodeBackend().version()
        assert first == second == "cli 2.1.123 · sdk 0.1.65"
        # Two backend instances, but only one CLI invocation.
        assert calls == [1]


class TestCodexBackendVersion:
    def _reset_cache(self, monkeypatch):
        monkeypatch.setattr(codex_mod, "_VERSION_CACHE", None)

    def test_version_combines_cli_and_sdk(self, monkeypatch):
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(codex_mod, "_codex_cli_version", lambda: "0.116.0-alpha.1")
        monkeypatch.setattr(codex_mod, "_codex_sdk_version", lambda: "0.2.0")

        assert CodexBackend().version() == "cli 0.116.0-alpha.1 · sdk 0.2.0"

    def test_sdk_version_reads_openai_codex_distribution(self, monkeypatch):
        seen: list[str] = []

        def fake_pkg_version(name):
            seen.append(name)
            if name == "openai-codex":
                return "0.1.0b3"
            raise codex_mod.PackageNotFoundError(name)

        monkeypatch.setattr(codex_mod, "_pkg_version", fake_pkg_version)

        assert codex_mod._codex_sdk_version() == "0.1.0b3"
        assert seen == ["openai-codex"]

    def test_sdk_version_empty_when_distribution_missing(self, monkeypatch):
        def fake_pkg_version(name):
            raise codex_mod.PackageNotFoundError(name)

        monkeypatch.setattr(codex_mod, "_pkg_version", fake_pkg_version)

        assert codex_mod._codex_sdk_version() == ""

    def test_cli_version_parses_last_token_of_stdout(self, monkeypatch):
        # The codex CLI prints "codex-cli 0.116.0-alpha.1" — the trailing
        # token is the version.

        monkeypatch.setattr(codex_mod, "_resolve_codex_bin", lambda: "/bin/codex")

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(
                returncode=0, stdout="codex-cli 0.116.0-alpha.1\n", stderr="some warning"
            )

        monkeypatch.setattr(codex_mod.subprocess, "run", fake_run)
        assert codex_mod._codex_cli_version() == "0.116.0-alpha.1"

    def test_cli_version_returns_empty_when_binary_unresolvable(self, monkeypatch):
        def boom():
            raise FileNotFoundError("no codex")

        monkeypatch.setattr(codex_mod, "_resolve_codex_bin", boom)
        assert codex_mod._codex_cli_version() == ""

    def test_cli_version_returns_empty_on_nonzero_exit(self, monkeypatch):
        monkeypatch.setattr(codex_mod, "_resolve_codex_bin", lambda: "/bin/codex")
        monkeypatch.setattr(
            codex_mod.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        )
        assert codex_mod._codex_cli_version() == ""


class TestSystemPromptSpill:
    def test_short_prompt_passes_through_unchanged(self, tmp_path):
        assert cc_mod.spill_system_prompt("hello", tmp_path, threshold=1024) == "hello"
        assert not (tmp_path / ".agent-flow").exists()

    def test_long_prompt_is_written_to_a_file_and_replaced_by_a_pointer(self, tmp_path):
        prompt = "x" * 5000
        pointer = cc_mod.spill_system_prompt(prompt, tmp_path, threshold=1024)
        files = list((tmp_path / ".agent-flow").glob("system-prompt-*.md"))
        assert len(files) == 1
        assert files[0].read_text() == prompt
        assert str(files[0]) in pointer
        assert "Read tool" in pointer
        assert len(pointer.encode()) < 1024

    def test_spill_files_are_content_addressed(self, tmp_path):
        prompt = "x" * 5000
        cc_mod.spill_system_prompt(prompt, tmp_path, threshold=1024)
        # Same content -> same file, no duplicate; different content -> new file.
        cc_mod.spill_system_prompt(prompt, tmp_path, threshold=1024)
        cc_mod.spill_system_prompt(prompt + "y", tmp_path, threshold=1024)
        assert len(list((tmp_path / ".agent-flow").glob("system-prompt-*.md"))) == 2
