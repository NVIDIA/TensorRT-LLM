from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
from collections.abc import Callable
from typing import Any

import pytest
from openai_codex.client import CodexConfig
from openai_codex.generated.v2_all import ThreadStartedNotification
from openai_codex.models import Notification, UnknownNotification

from agent_flow.backends import codex_transport as transport_module
from agent_flow.backends.base import ResultEvent
from agent_flow.backends.codex_events import CodexClient
from agent_flow.backends.codex_transport import CodexTransport, notification_data
from agent_flow.tools import tool
from agent_flow.types import AgentTextEvent, RateLimitWarningEvent


class FakeSdkClient(transport_module._ObservedSdkClient):
    """Use real SDK notification parsing with an in-memory request/reader seam."""

    def __init__(self, config, approval_handler, observer):
        super().__init__(config, approval_handler, observer)
        self.server_request = approval_handler
        self.notifications: queue.Queue[Notification | BaseException] = queue.Queue()
        self.turn_requests: queue.Queue[dict[str, Any]] = queue.Queue()
        self.requests: list[tuple[str, dict]] = []
        self.on_request: Callable[[str, dict], dict | None] | None = None
        self.started = False
        self.initialized = False
        self.closed = False

    def start(self):
        self.started = True

    def initialize(self):
        self.initialized = True

    def close(self):
        self.closed = True
        self.notifications.put(EOFError("fake SDK closed"))

    def next_notification(self):
        event = self.notifications.get(timeout=5)
        if isinstance(event, BaseException):
            raise event
        return event

    def emit(self, method: str, params: dict):
        event = self._coerce_notification(method, params)
        self.notifications.put(event)
        return event

    def request(self, method, params, *, response_model):
        self.requests.append((method, params))
        if method == "turn/start":
            self.turn_requests.put(params)
        response = self.on_request(method, params) if self.on_request is not None else None
        if response is None:
            response = (
                {"turn": {"id": f"turn-{params['threadId']}"}} if method == "turn/start" else {}
            )
        return response_model.model_validate(response)

    async def next_turn(self):
        return await asyncio.to_thread(self.turn_requests.get, True, 2)


@pytest.fixture
async def transport(monkeypatch):
    monkeypatch.setattr(transport_module, "_ObservedSdkClient", FakeSdkClient)
    instance = CodexTransport(CodexConfig(codex_bin="/unused/codex"))
    await instance.start()
    try:
        yield instance
    finally:
        await instance.close()


def _turn(thread_id: str, status: str = "completed"):
    return {
        "threadId": thread_id,
        "turn": {"id": f"turn-{thread_id}", "items": [], "itemsView": "full", "status": status},
    }


def _thread(thread_id: str, parent: str):
    return {
        "thread": {
            "id": thread_id,
            "parentThreadId": parent,
            "sessionId": "session-one",
            "cliVersion": "0.154.0",
            "createdAt": 0,
            "updatedAt": 0,
            "cwd": "/tmp",
            "ephemeral": False,
            "historyMode": "legacy",
            "modelProvider": "openai",
            "preview": "",
            "source": "appServer",
            "status": {"type": "idle"},
            "turns": [],
        }
    }


def _text(thread_id: str, text: str):
    return {
        "threadId": thread_id,
        "turnId": f"turn-{thread_id}",
        "completedAtMs": 100,
        "item": {
            "id": f"message-{thread_id}",
            "type": "agentMessage",
            "text": text,
            "phase": "final_answer",
        },
    }


async def _collect(stream):
    return [event async for event in stream]


def test_unknown_sdk_notifications_expose_original_params():
    params = {"threadId": "child", "newField": {"value": 2}}
    event = Notification("future/notification", UnknownNotification(params=params))
    assert notification_data(event) == params


async def test_completion_before_start_response_and_global_events_are_retained(transport):
    sdk = transport._client

    def complete_before_response(method, params):
        if method != "turn/start":
            return None
        sdk.emit("thread/started", _thread("child", "root"))
        sdk.emit("account/rateLimits/updated", {"rateLimits": {"primary": {"usedPercent": 90}}})
        sdk.emit("item/completed", _text("child", "child result"))
        sdk.emit("turn/completed", _turn("root"))
        return {"turn": {"id": "turn-root"}}

    sdk.on_request = complete_before_response
    events = await asyncio.wait_for(_collect(transport.stream_turn("root", "hello")), timeout=2)
    assert [event.method for event in events] == [
        "thread/started",
        "account/rateLimits/updated",
        "item/completed",
        "turn/completed",
    ]
    assert notification_data(events[0])["thread"]["parentThreadId"] == "root"
    assert isinstance(events[0].payload, ThreadStartedNotification)
    assert not any(method == "turn/interrupt" for method, _ in sdk.requests)


async def test_concurrent_clients_filter_other_roots_and_keep_account_updates(transport):
    sdk = transport._client
    clients = {thread: CodexClient(transport, thread) for thread in ("alpha", "beta")}
    tasks = {
        thread: asyncio.create_task(_collect(client.send_message("work")))
        for thread, client in clients.items()
    }
    try:
        started = [await sdk.next_turn(), await sdk.next_turn()]
        assert {params["threadId"] for params in started} == {"alpha", "beta"}
        for thread in clients:
            sdk.emit("turn/started", _turn(thread, "inProgress"))
        sdk.emit("account/rateLimits/updated", {"rateLimits": {"primary": {"usedPercent": 90}}})
        sdk.emit("item/completed", _text("beta", "beta only"))
        sdk.emit("item/completed", _text("alpha", "alpha only"))
        sdk.emit("turn/completed", _turn("alpha"))
        sdk.emit("turn/completed", _turn("beta"))
        results = await asyncio.wait_for(asyncio.gather(*tasks.values()), timeout=2)
        for thread, events in zip(tasks, results):
            assert [event.text for event in events if isinstance(event, AgentTextEvent)] == [
                f"{thread} only"
            ]
            assert [event.text for event in events if isinstance(event, ResultEvent)] == [
                f"{thread} only"
            ]
            warnings = [event for event in events if isinstance(event, RateLimitWarningEvent)]
            assert len(warnings) == 1
            assert warnings[0].utilization == 0.9
    finally:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


async def test_sdk_eof_wakes_waiting_turn_and_subsequent_requests_fail(transport):
    task = asyncio.create_task(_collect(transport.stream_turn("root", "work")))
    await transport._client.next_turn()
    transport._client.notifications.put(EOFError("reader reached EOF"))
    with pytest.raises(EOFError, match="reader reached EOF"):
        await asyncio.wait_for(task, timeout=2)
    with pytest.raises(RuntimeError, match="transport failed"):
        await transport.request("thread/read", {"threadId": "root"})


async def test_cancelled_turn_interrupts_and_releases_the_session(transport):
    sdk = transport._client
    observed = asyncio.Event()

    async def consume():
        async for _event in transport.stream_turn("root", "work"):
            observed.set()

    task = asyncio.create_task(consume())
    await sdk.next_turn()
    sdk.emit("turn/started", _turn("root", "inProgress"))
    await asyncio.wait_for(observed.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ("turn/interrupt", {"threadId": "root", "turnId": "turn-root"}) in sdk.requests
    assert not transport._queues

    retry = asyncio.create_task(_collect(transport.stream_turn("root", "retry")))
    await sdk.next_turn()
    sdk.emit("turn/completed", _turn("root"))
    await asyncio.wait_for(retry, timeout=2)


async def test_cancellation_while_start_response_is_pending_interrupts_created_turn(transport):
    release_response = threading.Event()

    def wait_before_response(method, params):
        if method == "turn/start":
            if not release_response.wait(timeout=2):
                raise TimeoutError("test did not release the turn/start response")
        return None

    transport._client.on_request = wait_before_response
    task = asyncio.create_task(_collect(transport.stream_turn("root", "work")))
    try:
        await transport._client.next_turn()
        task.cancel()
        release_response.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert (
            "turn/interrupt",
            {"threadId": "root", "turnId": "turn-root"},
        ) in transport._client.requests
        assert not transport._queues
    finally:
        release_response.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_same_session_rejects_concurrent_turn_without_disturbing_first(transport):
    first = asyncio.create_task(_collect(transport.stream_turn("root", "first")))
    await transport._client.next_turn()
    with pytest.raises(RuntimeError, match="Concurrent turns"):
        await _collect(transport.stream_turn("root", "second"))
    transport._client.emit("turn/completed", _turn("root"))
    assert await asyncio.wait_for(first, timeout=2)


async def test_consecutive_client_turns_release_transport_before_returning(transport):
    def complete(method, params):
        if method == "turn/start":
            transport._client.emit("item/completed", _text("root", "done"))
            transport._client.emit("turn/completed", _turn("root"))

    transport._client.on_request = complete
    client = CodexClient(transport, "root")
    for prompt in ("initial", "correct missing required calls"):
        events = await _collect(client.send_message(prompt))
        assert isinstance(events[-1], ResultEvent)
        assert not transport._active_threads
        assert not transport._queues


async def test_dynamic_tool_runs_on_owner_loop_with_registration_context_and_child_ancestry(
    transport,
):
    current_workflow = contextvars.ContextVar("test_codex_workflow", default="unset")
    owner_loop = asyncio.get_running_loop()
    observed = []

    @tool("read", "Read data", {"query": str})
    async def read(args):
        observed.append((asyncio.get_running_loop(), current_workflow.get(), args))
        await asyncio.sleep(0)
        return {"content": [{"type": "text", "text": "result"}]}

    token = current_workflow.set("workflow-one")
    transport.register_tools("root", [read])
    current_workflow.reset(token)

    def request_from_reader():
        transport._client.emit(
            "thread/started", {"thread": {"id": "child", "parentThreadId": "root"}}
        )
        transport._client.emit(
            "thread/started", {"thread": {"id": "grandchild", "parentThreadId": "child"}}
        )
        return transport._client.server_request(
            "item/tool/call",
            {"threadId": "grandchild", "tool": "read", "arguments": {"query": "value"}},
        )

    response = await asyncio.wait_for(asyncio.to_thread(request_from_reader), timeout=2)
    assert response == {"contentItems": [{"type": "inputText", "text": "result"}], "success": True}
    assert observed == [(owner_loop, "workflow-one", {"query": "value"})]

    transport.unregister_tools("root")
    unavailable = await asyncio.to_thread(
        transport._client.server_request,
        "item/tool/call",
        {"threadId": "grandchild", "tool": "read", "arguments": {}},
    )
    assert unavailable["success"] is False
    assert "not registered" in unavailable["contentItems"][0]["text"]


@pytest.mark.parametrize(
    "scenario", ["unknown", "wrong_arguments", "handler_error", "unsupported_result"]
)
async def test_dynamic_tool_failures_return_explicit_unsuccessful_results(transport, scenario):
    @tool("work", "Do work", {})
    async def work(args):
        if scenario == "handler_error":
            raise ValueError("backend unavailable")
        return {"content": [{"type": "audio", "data": "YWJj", "mimeType": "audio/wav"}]}

    transport.register_tools("root", [work])
    params = {
        "threadId": "root",
        "tool": "missing" if scenario == "unknown" else "work",
        "arguments": [] if scenario == "wrong_arguments" else {},
    }
    response = await asyncio.wait_for(
        asyncio.to_thread(transport._client.server_request, "item/tool/call", params), timeout=2
    )
    assert response["success"] is False
    expected = {
        "unknown": "not registered",
        "wrong_arguments": "requires an object",
        "handler_error": "backend unavailable",
        "unsupported_result": "Unsupported tool result",
    }[scenario]
    assert expected in response["contentItems"][0]["text"]
