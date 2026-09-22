"""Translate the pinned Codex app-server protocol into shared backend events."""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing
from dataclasses import replace
from typing import Any, AsyncIterator, Protocol

from ..types import (
    AgentTextEvent,
    CompactBoundaryEvent,
    RateLimitWarningEvent,
    ServerToolCallEvent,
    SessionInitEvent,
    ThinkingEvent,
    ToolCallEvent,
    UsageInfo,
)
from .base import BackendClient, BackendEvent, ResultEvent


class CodexTransport(Protocol):
    def stream_turn(self, thread_id: str, message: str) -> AsyncIterator[Any]: ...

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...


def _payload(notification: Any) -> dict[str, Any]:
    """Normalize SDK models once, including forward-compatible notifications."""
    payload = notification.payload
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "params"):
        return payload.params
    return payload.model_dump(by_alias=True, mode="json", exclude_unset=True)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


_TOKEN_FIELDS = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "cachedInputTokens": "cache_read_tokens",
    "cacheWriteInputTokens": "cache_creation_tokens",
    "totalTokens": "total_tokens",
}


def _usage(token_usage: dict[str, Any], baseline: dict[str, int]) -> UsageInfo:
    # Thread totals are cumulative; ResultEvent describes this send_message call.
    fields: dict[str, Any] = {}
    for source, target in _TOKEN_FIELDS.items():
        current = _integer(token_usage.get("total", {}).get(source))
        if current is not None:
            previous = baseline.get(source, 0)
            fields[target] = current - previous if current >= previous else current
    context = _integer(token_usage.get("last", {}).get("totalTokens"))
    window = _integer(token_usage.get("modelContextWindow"))
    return UsageInfo(
        **fields,
        context_tokens=context,
        context_window=window,
        context_percentage=100.0 * context / window if context is not None and window else None,
    )


def _tool_event(item: dict[str, Any]) -> ToolCallEvent | ServerToolCallEvent | None:
    kind = item.get("type")
    item_id = item.get("id")
    if kind == "commandExecution":
        return ToolCallEvent("Bash", {"command": item.get("command", "")}, item_id)
    if kind in {"dynamicToolCall", "mcpToolCall"}:
        arguments = item.get("arguments")
        name = item["tool"]
        if kind == "mcpToolCall" and item.get("server"):
            name = f"mcp__{item['server']}__{name}"
        return ToolCallEvent(name, arguments if isinstance(arguments, dict) else {}, item_id)
    if kind == "collabAgentToolCall":
        fields = {
            "prompt": "prompt",
            "model": "model",
            "reasoningEffort": "reasoning_effort",
            "senderThreadId": "sender_thread_id",
            "receiverThreadIds": "receiver_thread_ids",
            "agentsStates": "agents_states",
            "status": "status",
        }
        return ToolCallEvent(
            item["tool"],
            {
                target: item[source]
                for source, target in fields.items()
                if item.get(source) is not None
            },
            item_id,
        )
    if kind == "fileChange":
        body = {"changes": item.get("changes", [])}
        if item.get("status") is not None:
            body["status"] = item["status"]
        return ToolCallEvent("FileChange", body, item_id)
    if kind == "webSearch":
        action = item.get("action")
        body = dict(action) if isinstance(action, dict) else {}
        if item.get("query") is not None:
            body.setdefault("query", item["query"])
        return ServerToolCallEvent("web_search", body, item_id)
    if kind == "imageView":
        return ServerToolCallEvent("view_image", {"path": item.get("path")}, item_id)
    return None


def _reasoning_text(item: dict[str, Any]) -> str:
    parts = []
    for field in ("summary", "content"):
        value = item.get(field, [])
        if isinstance(value, str):
            value = [value]
        parts.extend(part.strip() for part in value if isinstance(part, str) and part.strip())
    return "\n".join(parts)


def _rate_limit_events(snapshot: dict[str, Any]) -> list[RateLimitWarningEvent]:
    """Use an explicit local 80% warning threshold with native reset/limit data."""
    prefix = snapshot.get("limitId") or snapshot.get("limitName") or "codex"
    reached = snapshot.get("rateLimitReachedType")
    rejected = reached is not None or snapshot.get("spendControlReached") is True
    events = []
    for name in ("primary", "secondary", "individualLimit"):
        window = snapshot.get(name)
        if not isinstance(window, dict):
            continue
        percent = window.get("usedPercent")
        if name == "individualLimit" and _integer(window.get("remainingPercent")) is not None:
            percent = 100 - window["remainingPercent"]
        if _integer(percent) is None:
            continue
        utilization = percent / 100
        status = (
            "rejected"
            if rejected or utilization >= 1
            else "allowed_warning"
            if utilization >= 0.8
            else "allowed"
        )
        events.append(
            RateLimitWarningEvent(
                status, f"{prefix}/{name}", _integer(window.get("resetsAt")), utilization
            )
        )
    if rejected and not events:
        events.append(RateLimitWarningEvent("rejected", reached or f"{prefix}/spend_control"))
    return events


def _merge_available(previous: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(previous)
    for name, value in update.items():
        if isinstance(value, dict):
            merged[name] = _merge_available(previous.get(name) or {}, value)
        elif value is not None:
            merged[name] = value
    return merged


class CodexClient(BackendClient):
    def __init__(
        self,
        transport: CodexTransport,
        thread_id: str,
        session_init: SessionInitEvent | None = None,
    ) -> None:
        self._transport = transport
        self._thread_id = thread_id
        self._session_init = session_init
        self._initialized = False
        self._totals: dict[str, int] = {}
        self._context: UsageInfo | None = None
        self._parents: dict[str, str] = {}
        self._labels: dict[str, str] = {}
        self._parent_tools: dict[str, str] = {}
        self._rate_limits: dict[str, dict[str, Any]] = {}
        self._last_rate_events: dict[str | None, RateLimitWarningEvent] = {}

    async def list_available_skills(self) -> list[str] | None:
        return list(self._session_init.skills) if self._session_init is not None else None

    async def get_context_usage(self) -> UsageInfo | None:
        # The protocol has no pre-turn context query. Never send a probe turn.
        return replace(self._context) if self._context is not None else None

    async def _optional_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(self._transport.request(method, params), timeout=5)
        except Exception:
            # Account APIs are unavailable for some billing routes, including API keys.
            return {}

    def _rate_updates(self, payload: dict[str, Any], *, complete: bool = False):
        snapshots = dict(payload.get("rateLimitsByLimitId") or {})
        main = payload.get("rateLimits")
        if isinstance(main, dict):
            snapshots[main.get("limitId") or "codex"] = main
        for key, update in snapshots.items():
            previous = {} if complete else self._rate_limits.get(key, {})
            # Rolling updates explicitly use null for unavailable metadata.
            snapshot = _merge_available(previous, update)
            snapshot.setdefault("limitId", key)
            self._rate_limits[key] = snapshot
            for event in _rate_limit_events(snapshot):
                if self._last_rate_events.get(event.rate_limit_type) != event:
                    self._last_rate_events[event.rate_limit_type] = event
                    yield event

    def _rate_recovery_needs_snapshot(self, payload: dict[str, Any]) -> bool:
        update = payload.get("rateLimits") or {}
        previous = self._rate_limits.get(update.get("limitId") or "codex", {})
        return any(
            previous.get(field) and field in update and update[field] is None
            for field in ("rateLimitReachedType", "spendControlReached")
        )

    def _belongs_to_session(self, thread_id: str) -> bool:
        seen: set[str] = set()
        while thread_id != self._thread_id:
            if thread_id in seen or thread_id not in self._parents:
                return False
            seen.add(thread_id)
            thread_id = self._parents[thread_id]
        return True

    def _register_thread(self, thread: dict[str, Any]) -> None:
        thread_id = thread["id"]
        if thread.get("parentThreadId"):
            self._parents[thread_id] = thread["parentThreadId"]
        nickname, role = thread.get("agentNickname"), thread.get("agentRole")
        if nickname or role:
            self._labels[thread_id] = (
                f"{nickname} ({role})" if nickname and role else nickname or role
            )

    def _attribute(self, event: BackendEvent, thread_id: str) -> BackendEvent:
        if thread_id != self._thread_id:
            event.parent_tool_use_id = self._parent_tools.get(thread_id)
            event.agent_label = self._labels.get(thread_id, thread_id)
        return event

    async def send_message(self, message: str) -> AsyncIterator[BackendEvent]:
        if not self._initialized:
            self._initialized = True
            if self._session_init is not None:
                yield self._session_init
            initial_limits = await self._optional_request("account/rateLimits/read", {})
            for event in self._rate_updates(initial_limits, complete=True):
                yield event

        baseline = dict(self._totals)
        latest_usage: UsageInfo | None = None
        final_text = ""
        fallback_text = ""
        partial_text: list[str] = []
        permission_denials: list[Any] = []
        diagnostics: list[dict[str, Any]] = []
        main_turn_id: str | None = None
        emitted: set[tuple[str, str, str]] = set()
        pending: list[tuple[str, BackendEvent]] = []

        def flush_pending(*, final: bool = False):
            remaining = []
            for thread_id, event in pending:
                if self._belongs_to_session(thread_id) and (
                    final or thread_id in self._parent_tools
                ):
                    yield self._attribute(event, thread_id)
                else:
                    remaining.append((thread_id, event))
            pending[:] = remaining

        async with aclosing(self._transport.stream_turn(self._thread_id, message)) as stream:
            async for notification in stream:
                method, payload = notification.method, _payload(notification)
                thread_id = payload.get("threadId")
                if method == "account/rateLimits/updated":
                    recovery = self._rate_recovery_needs_snapshot(payload)
                    for event in self._rate_updates(payload):
                        yield event
                    if recovery:
                        # Null in a sparse update is unavailable, not proof of recovery.
                        snapshot = await self._optional_request("account/rateLimits/read", {})
                        for event in self._rate_updates(snapshot, complete=True):
                            yield event
                    continue
                if method == "thread/started":
                    self._register_thread(payload["thread"])
                    for event in flush_pending():
                        yield event
                    continue
                if method == "turn/started" and thread_id == self._thread_id:
                    main_turn_id = payload["turn"]["id"]
                    continue
                if thread_id == self._thread_id and main_turn_id is not None:
                    event_turn_id = (
                        payload.get("turn", {}).get("id")
                        if method == "turn/completed"
                        else payload.get("turnId")
                    )
                    if event_turn_id is not None and event_turn_id != main_turn_id:
                        continue
                if method in {"item/started", "item/completed"}:
                    item = payload["item"]
                    kind = item.get("type")
                    if kind == "collabAgentToolCall":
                        for receiver in item.get("receiverThreadIds", []):
                            self._parents.setdefault(
                                receiver, item.get("senderThreadId", thread_id)
                            )
                            if item.get("tool") == "spawnAgent":
                                self._parent_tools[receiver] = item["id"]
                        for event in flush_pending():
                            yield event

                    events: list[BackendEvent] = []
                    tool = _tool_event(item)
                    if tool is not None:
                        events.append(tool)
                    if kind == "contextCompaction" and thread_id == self._thread_id:
                        events.append(
                            CompactBoundaryEvent(
                                item.get("trigger"), _integer(item.get("preTokens"))
                            )
                        )
                    if method == "item/completed":
                        if item.get("status") == "declined" and self._belongs_to_session(thread_id):
                            permission_denials.append(
                                {"kind": kind, "id": item.get("id"), "item": item}
                            )
                        # A failed command/tool can be recovered by the agent. Only turn failure
                        # makes the request fail; permission denials remain separately visible.
                        if kind == "reasoning":
                            text = _reasoning_text(item)
                            if text:
                                events.append(ThinkingEvent(text))
                        elif kind == "agentMessage":
                            text = (item.get("text") or "").strip()
                            if text:
                                events.append(AgentTextEvent(text))
                                if thread_id == self._thread_id:
                                    partial_text.append(text)
                                    if item.get("phase") == "final_answer":
                                        final_text = text
                                    elif item.get("phase") is None:
                                        fallback_text = text
                    for event in events:
                        key = (thread_id, item.get("id"), type(event).__name__)
                        if key in emitted:
                            continue
                        emitted.add(key)
                        if thread_id == self._thread_id:
                            yield event
                        else:
                            pending.append((thread_id, event))
                    for event in flush_pending():
                        yield event
                elif method == "thread/tokenUsage/updated" and thread_id == self._thread_id:
                    token_usage = payload["tokenUsage"]
                    latest_usage = _usage(token_usage, baseline)
                    self._totals = {
                        key: value
                        for key, value in token_usage.get("total", {}).items()
                        if key in _TOKEN_FIELDS and _integer(value) is not None
                    }
                    self._context = UsageInfo(
                        context_tokens=latest_usage.context_tokens,
                        context_window=latest_usage.context_window,
                        context_percentage=latest_usage.context_percentage,
                    )
                elif method == "error" and thread_id == self._thread_id:
                    diagnostics.append(payload)
                elif method == "turn/completed" and thread_id == self._thread_id:
                    turn = payload["turn"]
                    for event in flush_pending(final=True):
                        yield event
                    if turn.get("error") is not None or turn.get("status") != "completed":
                        error = turn.get("error") or {
                            "message": f"Turn ended: {turn.get('status')}"
                        }
                        info = error.get("codexErrorInfo")
                        if info in ("usageLimitExceeded", "rateLimitExceeded"):
                            yield RateLimitWarningEvent("rejected", info)
                        detail = {
                            "threadId": self._thread_id,
                            "turnId": turn.get("id"),
                            "error": error,
                            "diagnostics": diagnostics,
                            "partialText": partial_text,
                        }
                        raise RuntimeError(
                            f"Codex turn failed: {json.dumps(detail, ensure_ascii=False)}"
                        )
                    usage = latest_usage or UsageInfo()
                    usage.num_turns = 1
                    usage.duration_ms = _integer(turn.get("durationMs"))
                    cost = await self._optional_request(
                        "account/usage/read", {"threadId": self._thread_id}
                    )
                    thread_usage = cost.get("threadUsage") or {}
                    estimate = _integer(thread_usage.get("estimatedUsageUsdMicros"))
                    if thread_usage.get("threadId") == self._thread_id and estimate is not None:
                        usage.estimated_thread_cost_usd = estimate / 1_000_000
                    yield ResultEvent(
                        text=final_text or fallback_text,
                        usage=usage,
                        permission_denials=permission_denials,
                    )
                    return
        raise RuntimeError("Codex notification stream ended before the main turn completed")
