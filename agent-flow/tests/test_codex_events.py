from __future__ import annotations

import pytest
from openai_codex.generated.v2_all import (
    AccountRateLimitsUpdatedNotification,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)
from openai_codex.models import Notification, UnknownNotification

from agent_flow.backends.base import ResultEvent
from agent_flow.backends.codex_events import CodexClient
from agent_flow.types import (
    AgentTextEvent,
    CompactBoundaryEvent,
    RateLimitWarningEvent,
    ServerToolCallEvent,
    SessionInitEvent,
    ThinkingEvent,
    ToolCallEvent,
)


def notice(method, **payload):
    models = {
        "thread/tokenUsage/updated": ThreadTokenUsageUpdatedNotification,
        "account/rateLimits/updated": AccountRateLimitsUpdatedNotification,
        "turn/completed": TurnCompletedNotification,
    }
    model = models.get(method)
    return Notification(
        method, model.model_validate(payload) if model is not None else UnknownNotification(payload)
    )


def item(kind, item_id, *, thread="main", method="item/completed", **fields):
    return notice(
        method,
        threadId=thread,
        turnId="turn",
        item={"id": item_id, "type": kind, **fields},
    )


def completed(*, thread="main", **fields):
    return notice(
        "turn/completed",
        threadId=thread,
        turn={"id": "turn", "status": "completed", "items": [], "itemsView": "full", **fields},
    )


def tokens(total, last=None, *, thread="main"):
    def breakdown(multiplier):
        return {
            "inputTokens": 80 * multiplier,
            "outputTokens": 20 * multiplier,
            "cachedInputTokens": 30 * multiplier,
            "cacheWriteInputTokens": 10 * multiplier,
            "totalTokens": 100 * multiplier,
            "reasoningOutputTokens": 5 * multiplier,
        }

    return notice(
        "thread/tokenUsage/updated",
        threadId=thread,
        turnId="turn",
        tokenUsage={
            "total": breakdown(total),
            "last": breakdown(last if last is not None else total),
            "modelContextWindow": 1000,
        },
    )


def started(thread, parent, nickname=None, role=None):
    return notice(
        "thread/started",
        thread={
            "id": thread,
            "parentThreadId": parent,
            "agentNickname": nickname,
            "agentRole": role,
        },
    )


class Transport:
    def __init__(self, *turns, responses=None):
        self.turns = list(turns)
        self.responses = responses or {}
        self.requests = []
        self.messages = []

    async def stream_turn(self, thread_id, message):
        self.messages.append((thread_id, message))
        for event in self.turns.pop(0):
            yield event

    async def request(self, method, params):
        self.requests.append((method, params))
        value = self.responses.get(method, {})
        if isinstance(value, Exception):
            raise value
        return value


async def collect(client, message="hello"):
    return [event async for event in client.send_message(message)]


async def test_persistent_usage_is_per_send_context_is_last_and_cache_write_is_mapped():
    transport = Transport(
        [tokens(2, 1), completed(durationMs=1234)],
        [tokens(3, 1), tokens(5, 2), tokens(900, thread="unrelated"), completed(durationMs=90)],
    )
    client = CodexClient(transport, "main")
    assert await client.get_context_usage() is None
    assert transport.messages == []
    first = (await collect(client))[-1].usage
    second = (await collect(client))[-1].usage
    assert (first.total_tokens, second.total_tokens) == (200, 300)
    assert second.input_tokens == 240
    assert second.output_tokens == 60
    assert second.cache_read_tokens == 90
    assert second.cache_creation_tokens == 30
    assert (first.context_tokens, second.context_tokens) == (100, 200)
    assert second.context_percentage == 20
    assert second.context_window == 1000
    assert (first.duration_ms, second.duration_ms) == (1234, 90)
    assert (first.num_turns, second.num_turns) == (1, 1)
    context = await client.get_context_usage()
    context.context_tokens = 99
    assert (await client.get_context_usage()).context_tokens == 200


async def test_token_counter_reset_never_reports_negative_usage():
    client = CodexClient(Transport([tokens(4), completed()], [tokens(1), completed()]), "main")
    await collect(client)
    assert (await collect(client))[-1].usage.total_tokens == 100


async def test_session_inventory_is_emitted_once_without_an_extra_model_turn():
    init = SessionInitEvent(skills=["review"], plugins=["plugin"], agents=["explorer"])
    transport = Transport([completed()], [completed()])
    client = CodexClient(transport, "main", init)
    assert await client.list_available_skills() == ["review"]
    assert transport.messages == []
    assert (await collect(client))[0] is init
    assert not any(isinstance(event, SessionInitEvent) for event in await collect(client))


async def test_tools_text_reasoning_and_compaction_are_deduplicated():
    client = CodexClient(
        Transport(
            [
                item("commandExecution", "bash", method="item/started", command="ls"),
                item("commandExecution", "bash", command="ls", status="completed"),
                item("dynamicToolCall", "dynamic", tool="update_status", arguments={"ok": True}),
                item("mcpToolCall", "mcp", server="kb", tool="lookup", arguments={"term": "x"}),
                item("fileChange", "file", changes=[{"path": "x.py"}], status="completed"),
                item("webSearch", "web", query="query", action={"type": "search"}),
                item("imageView", "img", path="/tmp/image.png"),
                item("contextCompaction", "compact", method="item/started"),
                item("contextCompaction", "compact"),
                item("reasoning", "reason", summary=["thinking"], content=["detail"]),
                notice("item/agentMessage/delta", threadId="main", delta="ignored"),
                item("agentMessage", "comment", text="working", phase="commentary"),
                item("agentMessage", "final", text="answer", phase="final_answer"),
                completed(),
            ]
        ),
        "main",
    )
    events = await collect(client)
    assert [e.name for e in events if isinstance(e, ToolCallEvent)] == [
        "Bash",
        "update_status",
        "mcp__kb__lookup",
        "FileChange",
    ]
    assert [e.name for e in events if isinstance(e, ServerToolCallEvent)] == [
        "web_search",
        "view_image",
    ]
    assert len([e for e in events if isinstance(e, CompactBoundaryEvent)]) == 1
    assert [e.text for e in events if isinstance(e, ThinkingEvent)] == ["thinking\ndetail"]
    assert [e.text for e in events if isinstance(e, AgentTextEvent)] == ["working", "answer"]
    assert events[-1].text == "answer"


async def test_recovered_tool_failure_does_not_fail_successful_turn():
    events = await collect(
        CodexClient(
            Transport(
                [
                    item("commandExecution", "bad", command="false", status="failed"),
                    item("fileChange", "denied", changes=[], status="declined"),
                    item("agentMessage", "answer", text="recovered"),
                    completed(),
                ]
            ),
            "main",
        )
    )
    result = events[-1]
    assert not result.is_error
    assert result.errors == []
    assert result.permission_denials[0]["id"] == "denied"
    assert result.text == "recovered"


async def test_child_before_parent_race_preserves_labels_and_spawn_attribution():
    events = await collect(
        CodexClient(
            Transport(
                [
                    # A child can emit before its parent's spawn call completes.
                    item("agentMessage", "child-text", thread="child", text="found it"),
                    started("grandchild", "child", "Ada", "worker"),
                    item("reasoning", "grand-reason", thread="grandchild", summary=["checking"]),
                    started("child", "main", "Grace", "explorer"),
                    item("commandExecution", "child-tool", thread="child", command="rg foo"),
                    item(
                        "collabAgentToolCall",
                        "nested-spawn",
                        thread="child",
                        tool="spawnAgent",
                        senderThreadId="child",
                        receiverThreadIds=["grandchild"],
                    ),
                    item(
                        "collabAgentToolCall",
                        "spawn",
                        tool="spawnAgent",
                        senderThreadId="main",
                        receiverThreadIds=["child"],
                    ),
                    tokens(80, thread="child"),
                    completed(thread="child"),
                    started("unrelated", "elsewhere", "Other", "worker"),
                    item("agentMessage", "other", thread="unrelated", text="invisible"),
                    completed(thread="unrelated"),
                    tokens(1),
                    item("agentMessage", "main-final", text="done", phase="final_answer"),
                    completed(),
                ]
            ),
            "main",
        )
    )
    child_text = next(e for e in events if isinstance(e, AgentTextEvent) and e.text == "found it")
    assert (child_text.parent_tool_use_id, child_text.agent_label) == ("spawn", "Grace (explorer)")
    child_tool = next(e for e in events if isinstance(e, ToolCallEvent) and e.name == "Bash")
    assert (child_tool.parent_tool_use_id, child_tool.agent_label) == ("spawn", "Grace (explorer)")
    grandchild = next(e for e in events if isinstance(e, ThinkingEvent))
    assert (grandchild.parent_tool_use_id, grandchild.agent_label) == (
        "nested-spawn",
        "Ada (worker)",
    )
    assert not any(isinstance(e, AgentTextEvent) and e.text == "invisible" for e in events)
    results = [e for e in events if isinstance(e, ResultEvent)]
    assert len(results) == 1
    assert results[0].text == "done"
    assert results[0].usage.total_tokens == 100


async def test_child_without_spawn_event_has_label_but_no_invented_invocation_id():
    events = await collect(
        CodexClient(
            Transport(
                [
                    started("child", "main", role="worker"),
                    item("agentMessage", "text", thread="child", text="child"),
                    completed(),
                ]
            ),
            "main",
        )
    )
    text = next(e for e in events if isinstance(e, AgentTextEvent))
    assert text.agent_label == "worker"
    assert text.parent_tool_use_id is None


async def test_structured_rate_limits_preserve_sparse_metadata_and_report_thresholds():
    initial = {
        "rateLimits": {
            "limitId": "codex",
            "primary": {"usedPercent": 10, "resetsAt": 100},
            "secondary": {"usedPercent": 85, "resetsAt": 200},
        }
    }
    update = notice(
        "account/rateLimits/updated",
        rateLimits={"limitId": "codex", "primary": {"usedPercent": 100, "resetsAt": 110}},
    )
    events = await collect(
        CodexClient(
            Transport(
                [update, update, completed()], responses={"account/rateLimits/read": initial}
            ),
            "main",
        )
    )
    limits = [e for e in events if isinstance(e, RateLimitWarningEvent)]
    assert [(e.rate_limit_type, e.status, e.utilization, e.resets_at) for e in limits] == [
        ("codex/primary", "allowed", 0.1, 100),
        ("codex/secondary", "allowed_warning", 0.85, 200),
        ("codex/primary", "rejected", 1.0, 110),
    ]


async def test_native_spend_control_limit_reached_needs_no_text_matching():
    events = await collect(
        CodexClient(
            Transport(
                [
                    notice(
                        "account/rateLimits/updated",
                        rateLimits={"rateLimitReachedType": "workspace_member_credits_depleted"},
                    ),
                    completed(),
                ]
            ),
            "main",
        )
    )
    rate = next(e for e in events if isinstance(e, RateLimitWarningEvent))
    assert rate.status == "rejected"
    assert rate.rate_limit_type == "workspace_member_credits_depleted"


async def test_sparse_window_update_preserves_previously_reported_reset_time():
    events = await collect(
        CodexClient(
            Transport(
                [
                    notice(
                        "account/rateLimits/updated",
                        rateLimits={"primary": {"usedPercent": 90}},
                    ),
                    completed(),
                ],
                responses={
                    "account/rateLimits/read": {
                        "rateLimits": {"primary": {"usedPercent": 10, "resetsAt": 123}}
                    }
                },
            ),
            "main",
        )
    )
    rates = [e for e in events if isinstance(e, RateLimitWarningEvent)]
    assert rates[-1].status == "allowed_warning"
    assert rates[-1].utilization == 0.9
    assert rates[-1].resets_at == 123


async def test_nullable_sparse_reached_state_requires_full_snapshot_to_confirm_recovery():
    class RecoveryTransport(Transport):
        async def request(self, method, params):
            if method == "account/rateLimits/read":
                recovered = bool(self.requests)
                self.requests.append((method, params))
                return {
                    "rateLimits": {
                        "limitId": "codex",
                        "primary": {"usedPercent": 10, "resetsAt": 100},
                        "rateLimitReachedType": None if recovered else "rate_limit_reached",
                    }
                }
            return await super().request(method, params)

    events = await collect(
        CodexClient(
            RecoveryTransport(
                [
                    notice(
                        "account/rateLimits/updated",
                        rateLimits={"limitId": "codex", "rateLimitReachedType": None},
                    ),
                    completed(),
                ]
            ),
            "main",
        )
    )
    rates = [e for e in events if isinstance(e, RateLimitWarningEvent)]
    assert [e.status for e in rates] == ["rejected", "allowed"]


async def test_sparse_null_does_not_clear_reached_when_snapshot_cannot_be_refetched():
    class UnavailableRecoveryTransport(Transport):
        async def request(self, method, params):
            if self.requests:
                raise RuntimeError("temporarily unavailable")
            return await super().request(method, params)

    events = await collect(
        CodexClient(
            UnavailableRecoveryTransport(
                [
                    notice(
                        "account/rateLimits/updated",
                        rateLimits={"spendControlReached": None},
                    ),
                    completed(),
                ],
                responses={
                    "account/rateLimits/read": {"rateLimits": {"spendControlReached": True}}
                },
            ),
            "main",
        )
    )
    assert [e.status for e in events if isinstance(e, RateLimitWarningEvent)] == ["rejected"]


async def test_another_turn_on_same_thread_cannot_finish_active_turn():
    events = await collect(
        CodexClient(
            Transport(
                [
                    notice("turn/started", threadId="main", turn={"id": "active"}),
                    tokens(900),
                    completed(),
                    completed(id="active"),
                ]
            ),
            "main",
        )
    )
    assert len(events) == 1
    assert events[0].usage.total_tokens is None


@pytest.mark.parametrize("estimate", [0, 1_250_000, None])
async def test_cost_is_optional_cumulative_thread_estimate_separate_from_turn_cost(estimate):
    transport = Transport(
        [completed()],
        responses={
            "account/usage/read": {
                "threadUsage": {"threadId": "main", "estimatedUsageUsdMicros": estimate}
            }
        },
    )
    result = (await collect(CodexClient(transport, "main")))[-1]
    assert result.usage.estimated_thread_cost_usd == (
        estimate / 1_000_000 if estimate is not None else None
    )
    assert result.usage.cost_usd is None
    assert ("account/usage/read", {"threadId": "main"}) in transport.requests


async def test_unavailable_account_apis_do_not_fail_completed_turn():
    events = await collect(
        CodexClient(
            Transport(
                [completed()],
                responses={
                    "account/usage/read": RuntimeError("API key billing"),
                    "account/rateLimits/read": RuntimeError("API key billing"),
                },
            ),
            "main",
        )
    )
    assert events[-1].usage.estimated_thread_cost_usd is None


async def test_failure_keeps_native_details_http_status_ids_and_partial_output():
    client = CodexClient(
        Transport(
            [
                item("agentMessage", "partial", text="I reached this step", phase="commentary"),
                notice(
                    "error",
                    threadId="main",
                    error={"message": "retry failed", "additionalDetails": "attempt 3"},
                ),
                completed(
                    status="failed",
                    error={
                        "message": "upstream failure",
                        "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 503}},
                        "additionalDetails": "request-123",
                    },
                ),
            ]
        ),
        "main",
    )
    with pytest.raises(RuntimeError) as exc:
        await collect(client)
    for detail in ("upstream failure", "503", "request-123", "I reached this step", "attempt 3"):
        assert detail in str(exc.value)
    assert '"threadId": "main"' in str(exc.value)
    assert '"turnId": "turn"' in str(exc.value)


async def test_stream_ending_without_root_completion_cannot_report_success():
    with pytest.raises(RuntimeError, match="before the main turn completed"):
        await collect(CodexClient(Transport([completed(thread="child")]), "main"))
