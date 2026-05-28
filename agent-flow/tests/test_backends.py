from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from claude_agent_sdk import types as sdk_types
from claude_agent_sdk.types import (AssistantMessage, RateLimitEvent,
                                    RateLimitInfo, ResultMessage, SystemMessage,
                                    TextBlock, ThinkingBlock, ToolUseBlock)

from agent_flow.backends import claude_code as cc_mod
from agent_flow.backends import codex as codex_mod
from agent_flow.backends import create_backend
from agent_flow.backends.base import ResultEvent
from agent_flow.backends.claude_code import ClaudeCodeBackend, ClaudeCodeClient
from agent_flow.backends.codex import (CodexBackend, CodexClient,
                                       _dynamic_tool_spec,
                                       _extract_final_response,
                                       _mcp_to_codex_content)
from agent_flow.types import (AgentTextEvent, CompactBoundaryEvent,
                              RateLimitWarningEvent, ServerToolCallEvent,
                              SessionInitEvent, ThinkingEvent, ToolCallEvent)


class ServerToolUseBlock:

    def __init__(self, id: str, name: str, input: dict[str, Any]):
        self.id = id
        self.name = name
        self.input = input


ServerToolUseBlock = getattr(sdk_types, "ServerToolUseBlock",
                             ServerToolUseBlock)


def _make_assistant_message(content,
                            parent_tool_use_id: str | None = None
                            ) -> AssistantMessage:
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


def _make_tool_use_block(name: str,
                         input: dict,
                         id: str = "tool-1") -> ToolUseBlock:
    return ToolUseBlock(id=id, name=name, input=input)


def _install_codex_sdk_modules():
    package = ModuleType("codex_app_server")
    client_module = ModuleType("codex_app_server.client")
    api_module = ModuleType("codex_app_server.api")
    generated = ModuleType("codex_app_server.generated.v2_all")

    class TextInput:

        def __init__(self, text):
            self.text = text

    class AsyncCodex:

        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class AppServerConfig:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AgentMessageThreadItem:

        def __init__(self, text, phase=None):
            self.text = text
            self.phase = phase

    class AskForApproval:

        def __init__(self, root):
            self.root = root

    class AskForApprovalValue:
        untrusted = "untrusted"
        never = "never"

    class CommandExecutionThreadItem:

        def __init__(self, command, id="cmd-1", status="completed"):
            self.id = id
            self.command = command
            self.status = status

    class CollabAgentToolCallThreadItem:

        def __init__(self,
                     tool,
                     id="collab-1",
                     status="completed",
                     prompt=None,
                     model=None,
                     reasoning_effort=None,
                     sender_thread_id="sender-1",
                     receiver_thread_ids=None,
                     agents_states=None):
            self.id = id
            self.tool = tool
            self.status = status
            self.prompt = prompt
            self.model = model
            self.reasoning_effort = reasoning_effort
            self.sender_thread_id = sender_thread_id
            self.receiver_thread_ids = receiver_thread_ids or []
            self.agents_states = agents_states or {}

    class ContextCompactionThreadItem:

        def __init__(self, id="compact-1", trigger=None, pre_tokens=None):
            self.id = id
            self.trigger = trigger
            self.pre_tokens = pre_tokens

    class DynamicToolCallThreadItem:

        def __init__(self, tool, arguments, id="dyn-1", status="completed"):
            self.id = id
            self.tool = tool
            self.arguments = arguments
            self.status = status

    class FileChangeThreadItem:

        def __init__(self,
                     changes,
                     id="file-1",
                     status="completed",
                     error=None):
            self.id = id
            self.changes = changes
            self.status = status
            self.error = error

    class ItemCompletedNotification:

        def __init__(self, item):
            self.item = item

    class ItemStartedNotification:

        def __init__(self, item):
            self.item = item

    class McpToolCallThreadItem:

        def __init__(self,
                     tool,
                     arguments,
                     id="mcp-1",
                     status="completed",
                     error=None):
            self.id = id
            self.tool = tool
            self.arguments = arguments
            self.status = status
            self.error = error

    class ReasoningThreadItem:

        def __init__(self, summary=None, content=None, id="reason-1"):
            self.id = id
            self.summary = summary or []
            self.content = content or []

    class WebSearchThreadItem:

        def __init__(self, query, action=None, id="web-1"):
            self.id = id
            self.query = query
            self.action = action

    class MessagePhase:
        final_answer = "final_answer"

    class SandboxMode:
        workspace_write = "workspace_write"
        danger_full_access = "danger_full_access"

    class ThreadItem:
        pass

    class TurnCompletedNotification:

        def __init__(self, turn):
            self.turn = turn

    class TurnStatus:
        completed = "completed"
        failed = "failed"
        in_progress = "inProgress"
        interrupted = "interrupted"

    class TokenUsageBreakdown:

        def __init__(self,
                     input_tokens=0,
                     output_tokens=0,
                     cached_input_tokens=0,
                     total_tokens=0,
                     reasoning_output_tokens=0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.cached_input_tokens = cached_input_tokens
            self.total_tokens = total_tokens
            self.reasoning_output_tokens = reasoning_output_tokens

    class ThreadTokenUsage:

        def __init__(self, last, total, model_context_window=None):
            self.last = last
            self.total = total
            self.model_context_window = model_context_window

    class ThreadTokenUsageUpdatedNotification:

        def __init__(self, thread_id, token_usage, turn_id):
            self.thread_id = thread_id
            self.token_usage = token_usage
            self.turn_id = turn_id

    class ThreadStartParams:

        def __init__(self,
                     model=None,
                     base_instructions=None,
                     developer_instructions=None,
                     config=None,
                     cwd=None,
                     sandbox=None,
                     approval_policy=None,
                     **kwargs):
            self.model = model
            self.base_instructions = base_instructions
            self.developer_instructions = developer_instructions
            self.config = config
            self.cwd = cwd
            self.sandbox = sandbox
            self.approval_policy = approval_policy

    def _params_dict(params):
        out = {}
        if params.model is not None:
            out["model"] = params.model
        if params.base_instructions is not None:
            out["baseInstructions"] = params.base_instructions
        if params.developer_instructions is not None:
            out["developerInstructions"] = params.developer_instructions
        if params.config is not None:
            out["config"] = params.config
        if params.cwd is not None:
            out["cwd"] = params.cwd
        if params.sandbox is not None:
            out["sandbox"] = params.sandbox
        if params.approval_policy is not None:
            out["approvalPolicy"] = params.approval_policy
        return out

    class AsyncThread:

        def __init__(self, codex, thread_id):
            self.codex = codex
            self.thread_id = thread_id

    package.TextInput = TextInput
    package.AsyncCodex = AsyncCodex
    client_module.AppServerConfig = AppServerConfig
    client_module._params_dict = _params_dict
    api_module.AsyncThread = AsyncThread
    generated.AgentMessageThreadItem = AgentMessageThreadItem
    generated.AskForApproval = AskForApproval
    generated.AskForApprovalValue = AskForApprovalValue
    generated.CollabAgentToolCallThreadItem = CollabAgentToolCallThreadItem
    generated.CommandExecutionThreadItem = CommandExecutionThreadItem
    generated.ContextCompactionThreadItem = ContextCompactionThreadItem
    generated.DynamicToolCallThreadItem = DynamicToolCallThreadItem
    generated.FileChangeThreadItem = FileChangeThreadItem
    generated.ItemCompletedNotification = ItemCompletedNotification
    generated.ItemStartedNotification = ItemStartedNotification
    generated.McpToolCallThreadItem = McpToolCallThreadItem
    generated.MessagePhase = MessagePhase
    generated.ReasoningThreadItem = ReasoningThreadItem
    generated.SandboxMode = SandboxMode
    generated.ThreadItem = ThreadItem
    generated.TurnCompletedNotification = TurnCompletedNotification
    generated.WebSearchThreadItem = WebSearchThreadItem
    generated.TokenUsageBreakdown = TokenUsageBreakdown
    generated.ThreadTokenUsage = ThreadTokenUsage
    generated.ThreadTokenUsageUpdatedNotification = (
        ThreadTokenUsageUpdatedNotification)
    generated.ThreadStartParams = ThreadStartParams
    generated.TurnStatus = TurnStatus

    sys.modules["codex_app_server"] = package
    sys.modules["codex_app_server.client"] = client_module
    sys.modules["codex_app_server.api"] = api_module
    sys.modules["codex_app_server.generated.v2_all"] = generated

    return {
        "TextInput": TextInput,
        "AgentMessageThreadItem": AgentMessageThreadItem,
        "CollabAgentToolCallThreadItem": CollabAgentToolCallThreadItem,
        "CommandExecutionThreadItem": CommandExecutionThreadItem,
        "ContextCompactionThreadItem": ContextCompactionThreadItem,
        "DynamicToolCallThreadItem": DynamicToolCallThreadItem,
        "FileChangeThreadItem": FileChangeThreadItem,
        "ItemCompletedNotification": ItemCompletedNotification,
        "ItemStartedNotification": ItemStartedNotification,
        "McpToolCallThreadItem": McpToolCallThreadItem,
        "MessagePhase": MessagePhase,
        "ReasoningThreadItem": ReasoningThreadItem,
        "TurnCompletedNotification": TurnCompletedNotification,
        "WebSearchThreadItem": WebSearchThreadItem,
        "TokenUsageBreakdown": TokenUsageBreakdown,
        "ThreadTokenUsage": ThreadTokenUsage,
        "ThreadTokenUsageUpdatedNotification":
        ThreadTokenUsageUpdatedNotification,
        "TurnStatus": TurnStatus,
    }


class TestCreateBackend:

    def test_factory_returns_expected_backend_types(self):
        assert create_backend(
            "claude-code").__class__.__name__ == "ClaudeCodeBackend"
        assert create_backend("codex").__class__.__name__ == "CodexBackend"

    def test_factory_rejects_unknown_backends(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("unknown")


class TestClaudeBackend:

    async def test_client_maps_tool_use_and_result_messages(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "ls"})])
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
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

    async def test_client_emits_session_init_event_from_init_system_message(
            self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
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
                        {
                            "path": "/x/no-name"
                        },  # missing name -> filtered
                    ],
                },
            )
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "ls"})])
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
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

        async def receive_response():
            yield SystemMessage(subtype="other", data={"skills": ["should"]})
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        assert not any(isinstance(e, SessionInitEvent) for e in events)

    async def test_client_tags_subagent_events_with_task_label(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            # Main agent spawns an Explore subagent via the Task tool.
            yield _make_assistant_message([
                _make_tool_use_block(
                    "Task",
                    {
                        "subagent_type": "Explore",
                        "description": "search repo",
                        "prompt": "find foo",
                    },
                    id="task-42",
                )
            ])
            # Subagent executes a Bash call; SDK marks it with the parent
            # tool_use id of the Task call above.
            yield _make_assistant_message(
                [
                    _make_tool_use_block("Bash", {"command": "rg foo"},
                                         id="tool-9")
                ],
                parent_tool_use_id="task-42",
            )
            # Subagent emits a text message before returning.
            yield _make_assistant_message(
                [TextBlock(text="found it")],
                parent_tool_use_id="task-42",
            )
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
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

        async def receive_response():
            yield _make_assistant_message([
                _make_tool_use_block(
                    "Task",
                    {"description": "fix bug"},
                    id="task-1",
                )
            ])
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "ls"}, id="t-2")],
                parent_tool_use_id="task-1",
            )
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("go")]
        sub_event = next(e for e in events
                         if isinstance(e, ToolCallEvent) and e.name == "Bash")
        assert sub_event.agent_label == "fix bug"

    async def test_client_uses_generic_label_for_unrecognized_parent(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            # Bash call references a parent the client never observed.
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "ls"}, id="t-1")],
                parent_tool_use_id="phantom-task",
            )
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("go")]
        bash_event = next(e for e in events
                          if isinstance(e, ToolCallEvent) and e.name == "Bash")
        assert bash_event.parent_tool_use_id == "phantom-task"
        assert bash_event.agent_label == "subagent"

    async def test_client_recognizes_agent_tool_and_strips_plugin_prefix(self):
        # The Claude Code CLI surfaces the subagent-spawning tool as
        # ``Agent`` (not ``Task``), and ``subagent_type`` may include a
        # plugin namespace prefix that should be hidden from the label.

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield _make_assistant_message([
                _make_tool_use_block(
                    "Agent",
                    {
                        "subagent_type":
                        "trtllm-agent-toolkit:exec-compile-specialist",
                        "description": "Compile TRT-LLM",
                        "prompt": "compile it",
                    },
                    id="agent-77",
                )
            ])
            yield _make_assistant_message(
                [_make_tool_use_block("Bash", {"command": "ls"}, id="t-3")],
                parent_tool_use_id="agent-77",
            )
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("go")]
        sub_event = next(e for e in events
                         if isinstance(e, ToolCallEvent) and e.name == "Bash")
        assert sub_event.parent_tool_use_id == "agent-77"
        assert sub_event.agent_label == "exec-compile-specialist"

    async def test_client_extracts_usage_and_cost_from_result_message(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()
        sdk_client.get_context_usage = AsyncMock(return_value={
            "totalTokens": 50000,
            "maxTokens": 200000,
            "percentage": 25.0,
        })

        async def receive_response():
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

        sdk_client.receive_response = receive_response
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
        sdk_client.get_context_usage = AsyncMock(
            side_effect=RuntimeError("boom"))

        async def receive_response():
            yield _make_result_message("done", usage={"input_tokens": 1})

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]

        assert len(events) == 1
        assert events[0].usage is not None
        assert events[0].usage.context_tokens is None
        assert events[0].usage.context_percentage is None

    async def test_client_emits_thinking_event_from_thinking_block(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield _make_assistant_message([
                ThinkingBlock(thinking="  reasoning step  ", signature="sig"),
                TextBlock(text="answer"),
            ])
            yield _make_result_message("answer")

        sdk_client.receive_response = receive_response
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

        async def receive_response():
            yield _make_assistant_message(
                [ThinkingBlock(thinking="   ", signature="sig")])
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        assert not any(isinstance(e, ThinkingEvent) for e in events)

    async def test_client_emits_server_tool_call_event(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield _make_assistant_message([
                ServerToolUseBlock(
                    id="srv-1",
                    name="web_search",
                    input={"query": "claude code"},
                ),
            ])
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
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

        async def receive_response():
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

        sdk_client.receive_response = receive_response
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

        async def receive_response():
            yield SystemMessage(
                subtype="compact_boundary",
                data={
                    "trigger": "auto",
                    "pre_tokens": 150000,
                },
            )
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        boundaries = [e for e in events if isinstance(e, CompactBoundaryEvent)]
        assert len(boundaries) == 1
        assert boundaries[0].trigger == "auto"
        assert boundaries[0].pre_tokens == 150000

    async def test_client_compact_boundary_handles_missing_fields(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield SystemMessage(subtype="compact_boundary", data={})
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        boundary = next(e for e in events
                        if isinstance(e, CompactBoundaryEvent))
        assert boundary.trigger is None
        assert boundary.pre_tokens is None

    async def test_client_raises_on_assistant_message_error(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield AssistantMessage(
                content=[],
                model="test-model",
                error="billing_error",
            )

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        with pytest.raises(RuntimeError,
                           match="Claude Code turn failed: billing_error"):
            async for _ in client.send_message("hi"):
                pass

    async def test_client_swallows_assistant_error_after_result(self):
        # An AssistantMessage.error AFTER a ResultMessage should not poison
        # the run — the existing got_result guard covers this; verify it
        # still holds with the new error path in place.

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield _make_result_message("done")
            yield AssistantMessage(
                content=[],
                model="test-model",
                error="rate_limit",
            )

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        assert any(isinstance(e, ResultEvent) for e in events)

    async def test_client_threads_error_fields_through_result_event(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield _make_result_message(
                "",
                is_error=True,
                errors=["transient upstream"],
                permission_denials=[{
                    "tool": "Bash"
                }],
            )

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        result = next(e for e in events if isinstance(e, ResultEvent))
        assert result.is_error is True
        assert result.errors == ["transient upstream"]
        assert result.permission_denials == [{"tool": "Bash"}]

    async def test_client_result_event_defaults_when_no_errors(self):

        sdk_client = MagicMock()
        sdk_client.query = AsyncMock()

        async def receive_response():
            yield _make_result_message("done")

        sdk_client.receive_response = receive_response
        client = ClaudeCodeClient(sdk_client)

        events = [event async for event in client.send_message("hi")]
        result = next(e for e in events if isinstance(e, ResultEvent))
        assert result.is_error is False
        assert result.errors == []
        assert result.permission_denials == []


class TestClaudeBackendCreateClient:

    async def _capture_options(self, monkeypatch, **kwargs):
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
        monkeypatch.setattr(cc_mod, "create_sdk_mcp_server",
                            lambda **_: object())

        backend = ClaudeCodeBackend()
        async with backend.create_client(system_prompt="hi",
                                         model="claude-test",
                                         **kwargs):
            pass
        return captured["options"]

    async def test_create_client_disables_bash_sandbox(self, monkeypatch):
        # ``danger_full_access`` analogue for Claude Code: the bash
        # sandbox is fully off so commands can touch anything on disk
        # or the network without restrictions.
        options = await self._capture_options(monkeypatch)
        assert options.sandbox == {"enabled": False}

    async def test_create_client_uses_bypass_permission_mode(self, monkeypatch):
        # ``bypassPermissions`` is the SDK setting that suppresses every
        # permission prompt the CLI would otherwise raise.
        options = await self._capture_options(monkeypatch)
        assert options.permission_mode == "bypassPermissions"

    async def test_create_client_can_use_tool_always_allows(self, monkeypatch):
        # Defense-in-depth: even if a tool ends up being routed through
        # the ``can_use_tool`` hook, it must approve every call.
        options = await self._capture_options(monkeypatch)
        result = await options.can_use_tool("Bash", {"command": "rm -rf /"},
                                            None)
        assert isinstance(result, cc_mod.PermissionResultAllow)

    async def test_create_client_no_extra_mcp_servers_by_default(
            self, monkeypatch):
        # Without ``extra_mcp_servers``, ``mcp_servers`` stays empty
        # (the in-process ``agent-tools`` server is only added when
        # ``tools`` is provided).
        options = await self._capture_options(monkeypatch)
        assert options.mcp_servers == {}

    async def test_create_client_extra_mcp_servers_merged_alongside_agent_tools(
            self, monkeypatch):
        # ``extra_mcp_servers`` values are passed verbatim into
        # ``ClaudeAgentOptions.mcp_servers``; the in-process
        # ``agent-tools`` server is layered on top when ``tools`` is set.
        options = await self._capture_options(
            monkeypatch,
            tools=[object()],
            extra_mcp_servers={
                "Glean": {
                    "type": "http",
                    "url": "https://example.test/mcp"
                },
            },
        )
        assert set(options.mcp_servers.keys()) == {"Glean", "agent-tools"}
        # The external Glean entry passes through verbatim.
        assert options.mcp_servers["Glean"] == {
            "type": "http",
            "url": "https://example.test/mcp",
        }
        # ``agent-tools`` is the in-process SDK MCP server, an opaque
        # object (the helper patches ``create_sdk_mcp_server`` to
        # return ``object()``); it must not be a dict-shaped server
        # config — that would mean the external config silently
        # overwrote the in-process server.
        assert not isinstance(options.mcp_servers["agent-tools"], dict)

    async def test_create_client_extra_mcp_servers_without_tools(
            self, monkeypatch):
        # With only ``extra_mcp_servers`` and no ``tools``, ``mcp_servers``
        # carries just the external entries — no implicit ``agent-tools``.
        options = await self._capture_options(
            monkeypatch,
            extra_mcp_servers={
                "Glean": {
                    "type": "http",
                    "url": "https://example.test/mcp"
                },
            },
        )
        assert set(options.mcp_servers.keys()) == {"Glean"}

    async def test_create_client_rejects_reserved_agent_tools_key(
            self, monkeypatch):
        # The ``agent-tools`` key is reserved for the in-process MCP
        # server built from ``tools``; user-supplied values under that
        # name would silently get overwritten, so we surface it as an
        # explicit error.
        backend = ClaudeCodeBackend()
        with pytest.raises(ValueError, match="agent-tools"):
            async with backend.create_client(
                    system_prompt="hi",
                    model="claude-test",
                    extra_mcp_servers={
                        "agent-tools": {
                            "type": "http",
                            "url": "x"
                        }
                    },
            ):
                pass


class TestCodexBackend:

    async def test_client_maps_command_and_final_response(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            cmd_item = sdk["CommandExecutionThreadItem"]("echo hello")
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=cmd_item)))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hello")]

        thread.turn.assert_awaited_once()
        assert events[0] == ToolCallEvent(name="Bash",
                                          input={"command": "echo hello"},
                                          tool_use_id="cmd-1")
        assert isinstance(events[1], ResultEvent)

    async def test_client_captures_token_usage_from_notifications(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            total = sdk["TokenUsageBreakdown"](
                input_tokens=100,
                output_tokens=50,
                cached_input_tokens=20,
                total_tokens=170,
            )
            usage = sdk["ThreadTokenUsage"](
                last=total,
                total=total,
                model_context_window=400000,
            )
            yield SimpleNamespace(
                payload=sdk["ThreadTokenUsageUpdatedNotification"](
                    thread_id="t-1",
                    token_usage=usage,
                    turn_id="turn-1",
                ))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hi")]

        assert len(events) == 1
        result = events[0]
        assert isinstance(result, ResultEvent)
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_read_tokens == 20
        assert result.usage.total_tokens == 170
        assert result.usage.context_tokens == 170
        assert result.usage.context_window == 400000
        assert result.usage.context_percentage == pytest.approx(0.0425)

    async def test_client_emits_final_answer_as_agent_text_event(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            draft = sdk["AgentMessageThreadItem"]("draft")
            final = sdk["AgentMessageThreadItem"](
                "final answer", phase=sdk["MessagePhase"].final_answer)
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=draft)))
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=final)))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hi")]

        text_events = [e for e in events if isinstance(e, AgentTextEvent)]
        assert [e.text for e in text_events] == ["draft", "final answer"]
        result_events = [e for e in events if isinstance(e, ResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].text == "final answer"

    async def test_client_ignores_token_deltas_and_emits_completed_message(
            self):
        sdk = _install_codex_sdk_modules()

        class AgentMessageDeltaNotification:

            def __init__(self, itemId, delta):
                self.itemId = itemId
                self.delta = delta

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            yield SimpleNamespace(
                payload=AgentMessageDeltaNotification("msg-1", "A"))
            yield SimpleNamespace(
                payload=AgentMessageDeltaNotification("msg-1", "B"))
            final = sdk["AgentMessageThreadItem"](
                "AB", phase=sdk["MessagePhase"].final_answer)
            final.id = "msg-1"
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=final)))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hi")]

        text_events = [e for e in events if isinstance(e, AgentTextEvent)]
        assert [e.text for e in text_events] == ["AB"]

    async def test_client_maps_codex_reasoning_to_thinking_event(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            reasoning = sdk["ReasoningThreadItem"](summary=["checking tests"],
                                                   content=["raw detail"])
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=reasoning)))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hi")]

        thinking = [e for e in events if isinstance(e, ThinkingEvent)]
        assert len(thinking) == 1
        assert thinking[0].text == "checking tests\nraw detail"

    async def test_client_maps_codex_web_search_to_server_tool_event(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            search = sdk["WebSearchThreadItem"](
                query="codex app-server",
                action={
                    "type": "search",
                    "query": "codex app-server"
                },
                id="web-7",
            )
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=search)))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hi")]

        server_calls = [e for e in events if isinstance(e, ServerToolCallEvent)]
        assert len(server_calls) == 1
        assert server_calls[0].name == "web_search"
        assert server_calls[0].tool_use_id == "web-7"
        assert server_calls[0].input == {
            "type": "search",
            "query": "codex app-server",
        }

    async def test_client_maps_codex_collab_agent_tool_calls_and_failures(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            collab = sdk["CollabAgentToolCallThreadItem"](
                tool="spawnAgent",
                id="collab-7",
                status="failed",
                prompt="inspect repo",
                model="gpt-test",
                reasoning_effort="high",
                receiver_thread_ids=["child-1"],
            )
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=collab)))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hi")]

        tool_call = next(e for e in events if isinstance(e, ToolCallEvent))
        assert tool_call.name == "spawnAgent"
        assert tool_call.tool_use_id == "collab-7"
        assert tool_call.input["prompt"] == "inspect repo"
        assert tool_call.input["model"] == "gpt-test"
        assert tool_call.input["reasoning_effort"] == "high"
        assert tool_call.input["receiver_thread_ids"] == ["child-1"]

        result = next(e for e in events if isinstance(e, ResultEvent))
        assert result.is_error is True
        assert result.errors == ["collabAgentToolCall failed"]

    async def test_client_emits_compact_boundary_for_context_compaction_item(
            self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            compact = sdk["ContextCompactionThreadItem"](trigger="auto",
                                                         pre_tokens=150000)
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=compact)))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hi")]

        boundaries = [e for e in events if isinstance(e, CompactBoundaryEvent)]
        assert len(boundaries) == 1
        assert boundaries[0].trigger == "auto"
        assert boundaries[0].pre_tokens == 150000

    async def test_client_maps_codex_file_changes_and_denials(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            file_change = sdk["FileChangeThreadItem"](
                changes=[{
                    "path": "app.py",
                    "kind": "update"
                }],
                status="declined",
                id="file-9",
            )
            yield SimpleNamespace(
                payload=sdk["ItemCompletedNotification"](SimpleNamespace(
                    root=file_change)))
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1")))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = [event async for event in client.send_message("hi")]

        file_event = next(e for e in events if isinstance(e, ToolCallEvent))
        assert file_event.name == "FileChange"
        assert file_event.tool_use_id == "file-9"
        assert file_event.input == {
            "changes": [{
                "path": "app.py",
                "kind": "update"
            }],
            "status": "declined",
        }
        result = next(e for e in events if isinstance(e, ResultEvent))
        assert result.permission_denials[0]["kind"] == "fileChange"
        assert result.permission_denials[0]["id"] == "file-9"

    async def test_client_emits_rate_limit_warning_before_turn_error(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            error = SimpleNamespace(
                message="usage limit reached",
                codex_error_info="UsageLimitExceeded",
            )
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1",
                    error=error,
                )))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)
        events = []
        with pytest.raises(RuntimeError, match="usage limit reached"):
            async for event in client.send_message("hi"):
                events.append(event)

        warnings = [e for e in events if isinstance(e, RateLimitWarningEvent)]
        assert len(warnings) == 1
        assert warnings[0].status == "rejected"
        assert warnings[0].rate_limit_type == "UsageLimitExceeded"

    async def test_client_raises_when_turn_completes_with_error(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1",
                    error=SimpleNamespace(message="model does not exist"),
                )))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)

        with pytest.raises(RuntimeError,
                           match="Codex turn failed: model does not exist"):
            async for _ in client.send_message("hi"):
                pass

    async def test_client_raises_when_turn_completes_interrupted(self):
        sdk = _install_codex_sdk_modules()

        thread = MagicMock()
        turn = MagicMock()
        turn.id = "turn-1"

        async def stream():
            yield SimpleNamespace(
                payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                    id="turn-1",
                    status=sdk["TurnStatus"].interrupted,
                )))

        turn.stream = stream
        thread.turn = AsyncMock(return_value=turn)

        client = CodexClient(thread)

        with pytest.raises(RuntimeError, match="status 'interrupted'"):
            async for _ in client.send_message("hi"):
                pass

    async def test_enter_passes_resolved_binary_to_app_server_config(
            self, monkeypatch):
        _install_codex_sdk_modules()

        monkeypatch.setattr(codex_mod, "_resolve_codex_bin",
                            lambda: "/bin/codex")
        monkeypatch.setattr(CodexBackend, "_install_server_request_handler",
                            lambda self: None)

        backend = CodexBackend()
        async with backend:
            config = backend._codex.config

        assert config.kwargs["codex_bin"] == "/bin/codex"
        assert config.kwargs["experimental_api"] is True
        assert "launch_args_override" not in config.kwargs

    def test_extract_final_response_prefers_final_answer_phase(self):
        sdk = _install_codex_sdk_modules()

        items = [
            sdk["AgentMessageThreadItem"]("draft"),
            sdk["AgentMessageThreadItem"](
                "final", phase=sdk["MessagePhase"].final_answer),
        ]

        assert _extract_final_response(items) == "final"

    async def _run_create_client(self, system_prompt: str) -> dict[str, Any]:
        _install_codex_sdk_modules()

        backend = CodexBackend()
        captured: dict[str, Any] = {}

        async def fake_thread_start(payload):
            captured["payload"] = payload
            return SimpleNamespace(thread=SimpleNamespace(id="t-1"))

        backend._codex = SimpleNamespace(
            _ensure_initialized=AsyncMock(),
            _client=SimpleNamespace(thread_start=fake_thread_start),
        )

        async with backend.create_client(system_prompt=system_prompt,
                                         model="gpt-5.4"):
            pass

        return captured["payload"]

    async def test_create_client_uses_codex_default_system_prompt(self):
        # Always defer to Codex's bundled base prompt — never override it
        # via base_instructions. An empty user prompt must not become an
        # empty developer_instructions either, otherwise Codex forwards an
        # empty instructions payload to the Responses API and the request
        # fails with 400 "Instructions are required".
        payload = await self._run_create_client(system_prompt="")
        assert "baseInstructions" not in payload
        assert "developerInstructions" not in payload

    async def test_create_client_appends_user_prompt_via_developer_message(
            self):
        # User-provided prompts ride on developer_instructions so they
        # layer on top of Codex's default base prompt, mirroring how the
        # Claude Code backend appends to its preset.
        payload = await self._run_create_client(system_prompt="be helpful")
        assert "baseInstructions" not in payload
        assert payload["developerInstructions"] == "be helpful"

    async def test_create_client_overrides_model_context_window_to_1m(self):
        # The bundled codex CLI's models.json caps gpt-5.5 at 272k; without
        # this override codex auto-compacts at ~258k even though the API
        # serves the full 1M context.
        payload = await self._run_create_client(system_prompt="hi")
        assert payload["config"]["model_context_window"] == 1000000

    async def test_create_client_bypasses_all_approvals_in_thread_start(self):
        # The Codex backend runs in fully autonomous mode: no sandbox
        # restrictions and the model never asks for approval. The mock
        # captures the params that would be sent to ``thread/start``.
        payload = await self._run_create_client(system_prompt="hi")
        assert payload["sandbox"] == "danger_full_access"
        assert payload["approvalPolicy"].root == "never"

    async def test_create_client_accepts_extra_mcp_servers_as_noop(self):
        # ``extra_mcp_servers`` is currently a Claude-Code-only concept;
        # the Codex backend must accept the kwarg without erroring and
        # without leaking it into ``thread/start``.
        _install_codex_sdk_modules()
        backend = CodexBackend()
        captured: dict[str, Any] = {}

        async def fake_thread_start(payload):
            captured["payload"] = payload
            return SimpleNamespace(thread=SimpleNamespace(id="t-1"))

        backend._codex = SimpleNamespace(
            _ensure_initialized=AsyncMock(),
            _client=SimpleNamespace(thread_start=fake_thread_start),
        )

        async with backend.create_client(
                system_prompt="hi",
                model="gpt-5.4",
                extra_mcp_servers={
                    "Glean": {
                        "type": "http",
                        "url": "https://example.test/mcp",
                    },
                },
        ):
            pass

        # Codex thread/start payload must not carry MCP-server config.
        payload = captured["payload"]
        assert "mcp_servers" not in payload
        assert "mcpServers" not in payload

    async def test_create_client_emits_codex_session_init_snapshot(self):
        sdk = _install_codex_sdk_modules()

        backend = CodexBackend()

        async def fake_thread_start(payload):
            return SimpleNamespace(thread=SimpleNamespace(id="t-1"))

        async def fake_skills_list(payload):
            return SimpleNamespace(data=[
                SimpleNamespace(skills=[
                    SimpleNamespace(name="skill-a", enabled=True),
                    SimpleNamespace(name="skill-disabled", enabled=False),
                ])
            ])

        async def fake_plugin_list(payload):
            return SimpleNamespace(plugins=[
                SimpleNamespace(name="plugin-a"),
                SimpleNamespace(name="plugin-b"),
            ])

        backend._codex = SimpleNamespace(
            _ensure_initialized=AsyncMock(),
            _client=SimpleNamespace(
                skills_list=fake_skills_list,
                plugin_list=fake_plugin_list,
                thread_start=fake_thread_start,
            ),
        )

        async with backend.create_client(system_prompt="hi",
                                         model="gpt-5.4") as client:
            thread = client._thread
            turn = MagicMock()
            turn.id = "turn-1"

            async def stream():
                yield SimpleNamespace(
                    payload=sdk["TurnCompletedNotification"](SimpleNamespace(
                        id="turn-1")))

            turn.stream = stream
            thread.turn = AsyncMock(return_value=turn)
            events = [event async for event in client.send_message("hi")]

        init = next(e for e in events if isinstance(e, SessionInitEvent))
        assert init.skills == ["skill-a"]
        assert init.plugins == ["plugin-a", "plugin-b"]


class TestCodexApprovalBypass:

    def _make_backend_with_handler(self):
        # Wire a CodexBackend up to a fake AsyncCodex whose sync client
        # exposes ``_approval_handler`` so we can exercise the override
        # ``_install_server_request_handler`` installs.

        captured_methods: list[tuple[str, dict[str, Any] | None]] = []

        def existing_handler(method, params):
            captured_methods.append((method, params))
            return {"sentinel": "fallthrough"}

        sync_client = SimpleNamespace(_approval_handler=existing_handler)
        backend = CodexBackend()
        backend._codex = SimpleNamespace(_client=SimpleNamespace(
            _sync=sync_client))
        backend._install_server_request_handler()
        return backend, sync_client, captured_methods

    def test_install_handler_auto_accepts_command_approval(self):
        _, sync_client, captured = self._make_backend_with_handler()
        result = sync_client._approval_handler(
            "item/commandExecution/requestApproval",
            {"command": "rm -rf /tmp/everything"},
        )
        assert result == {"decision": "accept"}
        # Auto-accept must short-circuit the SDK's default handler.
        assert captured == []

    def test_install_handler_auto_accepts_file_change_approval(self):
        _, sync_client, captured = self._make_backend_with_handler()
        result = sync_client._approval_handler(
            "item/fileChange/requestApproval",
            {"changes": [{
                "path": "x",
                "kind": "delete"
            }]},
        )
        assert result == {"decision": "accept"}
        assert captured == []

    def test_install_handler_auto_accepts_unknown_future_approval(self):
        # Future SDK versions could add e.g. ``item/applyPatch/requestApproval``
        # — the bypass is intentionally generic on the ``/requestApproval``
        # suffix so new approval kinds are auto-accepted without code changes.
        _, sync_client, captured = self._make_backend_with_handler()
        result = sync_client._approval_handler(
            "item/applyPatch/requestApproval",
            {"diff": "..."},
        )
        assert result == {"decision": "accept"}
        assert captured == []

    def test_install_handler_routes_dynamic_tool_call(self, monkeypatch):
        _, sync_client, captured = self._make_backend_with_handler()

        captured_tool_params: list[dict[str, Any]] = []

        def fake_handle(params):
            captured_tool_params.append(params)
            return {"contentItems": [], "success": True}

        monkeypatch.setattr(codex_mod, "_handle_dynamic_tool_call", fake_handle)
        result = sync_client._approval_handler(
            "item/tool/call",
            {
                "threadId": "t-1",
                "tool": "x"
            },
        )
        assert result == {"contentItems": [], "success": True}
        assert captured_tool_params == [{"threadId": "t-1", "tool": "x"}]
        # Dynamic-tool dispatch must not also fall through to the SDK
        # default handler.
        assert captured == []

    def test_install_handler_falls_through_for_unrelated_methods(self):
        _, sync_client, captured = self._make_backend_with_handler()
        result = sync_client._approval_handler("some/other/method", {"k": "v"})
        # Unknown non-approval methods defer to the SDK's default handler.
        assert result == {"sentinel": "fallthrough"}
        assert captured == [("some/other/method", {"k": "v"})]


class TestCodexDynamicTools:

    def test_dynamic_tool_spec_serializes_sdk_tool(self):

        tool = SimpleNamespace(
            name="do_thing",
            description="Does a thing.",
            input_schema={
                "type": "object",
                "properties": {}
            },
        )

        assert _dynamic_tool_spec(tool) == {
            "name": "do_thing",
            "description": "Does a thing.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            },
            "deferLoading": False,
        }

    def test_mcp_to_codex_content_maps_text_and_errors(self):

        items, success = _mcp_to_codex_content({
            "content": [{
                "type": "text",
                "text": "ok"
            }],
        })
        assert items == [{"type": "inputText", "text": "ok"}]
        assert success is True

        items, success = _mcp_to_codex_content({
            "content": [{
                "type": "text",
                "text": "boom"
            }],
            "is_error":
            True,
        })
        assert items == [{"type": "inputText", "text": "boom"}]
        assert success is False

    def test_mcp_to_codex_content_emits_placeholder_when_empty(self):

        items, success = _mcp_to_codex_content({"content": []})
        assert items == [{"type": "inputText", "text": ""}]
        assert success is True

    def test_handle_dynamic_tool_call_dispatches_to_registered_handler(self):

        calls: list[dict[str, Any]] = []

        async def handler(args):
            calls.append(args)
            return {"content": [{"type": "text", "text": f"got {args['x']}"}]}

        codex_mod._TOOL_HANDLERS["t-1"] = {"my_tool": handler}
        try:
            response = codex_mod._handle_dynamic_tool_call({
                "threadId": "t-1",
                "turnId": "turn-1",
                "callId": "call-1",
                "tool": "my_tool",
                "arguments": {
                    "x": 7
                },
            })
        finally:
            codex_mod._TOOL_HANDLERS.pop("t-1", None)

        assert calls == [{"x": 7}]
        assert response == {
            "contentItems": [{
                "type": "inputText",
                "text": "got 7"
            }],
            "success": True,
        }

    def test_handle_dynamic_tool_call_reports_unknown_tool(self):

        response = codex_mod._handle_dynamic_tool_call({
            "threadId": "nobody",
            "tool": "ghost",
            "arguments": {},
        })

        assert response["success"] is False
        assert "not registered" in response["contentItems"][0]["text"]

    def test_handle_dynamic_tool_call_captures_handler_exceptions(self):

        async def handler(_args):
            raise RuntimeError("explode")

        codex_mod._TOOL_HANDLERS["t-1"] = {"broken": handler}
        try:
            response = codex_mod._handle_dynamic_tool_call({
                "threadId": "t-1",
                "tool": "broken",
                "arguments": {},
            })
        finally:
            codex_mod._TOOL_HANDLERS.pop("t-1", None)

        assert response["success"] is False
        assert "explode" in response["contentItems"][0]["text"]


class TestResolveCodexBin:

    def test_env_override_wins(self, tmp_path, monkeypatch):

        fake = tmp_path / "codex"
        fake.write_text("#!/bin/sh\n")
        monkeypatch.setenv("CODEX_BIN", str(fake))
        monkeypatch.setitem(
            sys.modules, "codex_cli_bin",
            SimpleNamespace(
                bundled_codex_path=lambda: Path("/nope/should-not-be-used")))

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
        monkeypatch.setitem(sys.modules, "codex_cli_bin",
                            SimpleNamespace(bundled_codex_path=lambda: bundled))
        monkeypatch.setattr(codex_mod.shutil, "which",
                            lambda _: "/some/system/codex")

        assert codex_mod._resolve_codex_bin() == str(bundled)

    def test_falls_back_to_path_when_bundle_missing(self, monkeypatch):

        monkeypatch.delenv("CODEX_BIN", raising=False)
        sys.modules.pop("codex_cli_bin", None)
        # Make the import fail even if the package is installed for real.
        monkeypatch.setitem(sys.modules, "codex_cli_bin", None)
        monkeypatch.setattr(codex_mod.shutil, "which",
                            lambda _: "/usr/bin/codex")

        assert codex_mod._resolve_codex_bin() == "/usr/bin/codex"
        sys.modules.pop("codex_cli_bin", None)

    def test_raises_when_nothing_found(self, monkeypatch):

        monkeypatch.delenv("CODEX_BIN", raising=False)
        monkeypatch.setitem(sys.modules, "codex_cli_bin", None)
        monkeypatch.setattr(codex_mod.shutil, "which", lambda _: None)

        with pytest.raises(FileNotFoundError, match="Codex CLI"):
            codex_mod._resolve_codex_bin()
        sys.modules.pop("codex_cli_bin", None)


class TestRelaxServiceTier:
    """``_relax_service_tier_on_module`` rewrites ``ServiceTier`` references
    to ``str`` so the SDK accepts new wire values like ``"priority"`` that
    the generated enum hasn't been regenerated to include yet.
    """

    def _build_fake_sdk_module(self) -> ModuleType:
        from enum import Enum

        from pydantic import BaseModel, ConfigDict, Field
        from typing_extensions import Annotated

        module = ModuleType("fake_codex_sdk")

        class ServiceTier(Enum):
            fast = "fast"
            flex = "flex"

        class ThreadStartResponse(BaseModel):
            model_config = ConfigDict(populate_by_name=True)
            model: str
            service_tier: Annotated[ServiceTier | None,
                                    Field(alias="serviceTier")] = None

        class TurnEvent(BaseModel):
            model_config = ConfigDict(populate_by_name=True)
            service_tier: ServiceTier | None = None

        class Unrelated(BaseModel):
            value: str

        module.ServiceTier = ServiceTier
        module.ThreadStartResponse = ThreadStartResponse
        module.TurnEvent = TurnEvent
        module.Unrelated = Unrelated
        return module

    def test_priority_value_rejected_before_patch(self):
        module = self._build_fake_sdk_module()
        with pytest.raises(Exception):
            module.ThreadStartResponse.model_validate({
                "model": "m",
                "serviceTier": "priority"
            })

    def test_relax_accepts_arbitrary_string(self):
        module = self._build_fake_sdk_module()
        affected = codex_mod._relax_service_tier_on_module(module)
        # Two models referenced ServiceTier; Unrelated did not.
        assert affected == 2

        ok = module.ThreadStartResponse.model_validate({
            "model": "m",
            "serviceTier": "priority"
        })
        assert ok.service_tier == "priority"

        ok2 = module.TurnEvent.model_validate({"service_tier": "anything"})
        assert ok2.service_tier == "anything"

    def test_no_servicetier_attr_returns_zero(self):
        module = ModuleType("empty")
        assert codex_mod._relax_service_tier_on_module(module) == 0


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
            return SimpleNamespace(returncode=0,
                                   stdout="2.1.123 (Claude Code)\n",
                                   stderr="")

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
            cc_mod.subprocess, "run",
            lambda *a, **k: pytest.fail("subprocess.run should not be called"))
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
        monkeypatch.setattr(codex_mod, "_codex_cli_version",
                            lambda: "0.116.0-alpha.1")
        monkeypatch.setattr(codex_mod, "_codex_sdk_version", lambda: "0.2.0")

        assert CodexBackend().version() == "cli 0.116.0-alpha.1 · sdk 0.2.0"

    def test_sdk_version_prefers_current_distribution_name(self, monkeypatch):

        seen: list[str] = []

        def fake_pkg_version(name):
            seen.append(name)
            if name == "openai-codex-app-server-sdk":
                return "0.116.0a1"
            raise codex_mod.PackageNotFoundError(name)

        monkeypatch.setattr(codex_mod, "_pkg_version", fake_pkg_version)

        assert codex_mod._codex_sdk_version() == "0.116.0a1"
        assert seen == ["openai-codex-app-server-sdk"]

    def test_cli_version_parses_last_token_of_stdout(self, monkeypatch):
        # The codex CLI prints "codex-cli 0.116.0-alpha.1" — the trailing
        # token is the version.

        monkeypatch.setattr(codex_mod, "_resolve_codex_bin",
                            lambda: "/bin/codex")

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0,
                                   stdout="codex-cli 0.116.0-alpha.1\n",
                                   stderr="some warning")

        monkeypatch.setattr(codex_mod.subprocess, "run", fake_run)
        assert codex_mod._codex_cli_version() == "0.116.0-alpha.1"

    def test_cli_version_returns_empty_when_binary_unresolvable(
            self, monkeypatch):

        def boom():
            raise FileNotFoundError("no codex")

        monkeypatch.setattr(codex_mod, "_resolve_codex_bin", boom)
        assert codex_mod._codex_cli_version() == ""

    def test_cli_version_returns_empty_on_nonzero_exit(self, monkeypatch):

        monkeypatch.setattr(codex_mod, "_resolve_codex_bin",
                            lambda: "/bin/codex")
        monkeypatch.setattr(
            codex_mod.subprocess, "run", lambda *a, **k: SimpleNamespace(
                returncode=1, stdout="", stderr="boom"))
        assert codex_mod._codex_cli_version() == ""


@pytest.fixture(autouse=True)
def cleanup_fake_sdk_modules():
    yield
    for name in [
            "codex_app_server",
            "codex_app_server.api",
            "codex_app_server.client",
            "codex_app_server.generated.v2_all",
    ]:
        sys.modules.pop(name, None)
