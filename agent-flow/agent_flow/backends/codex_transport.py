"""Codex SDK transport and per-session tool dispatch.

The SDK's public turn streams omit account notifications and descendant turns.
One isolated override observes decoded notifications before the SDK routes them;
all process management and JSON-RPC requests still use the pinned SDK. Register
observers before starting a turn so even an immediate completion is retained.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any, AsyncIterator

from openai_codex.client import CodexClient as SdkClient
from openai_codex.client import CodexConfig
from openai_codex.models import Notification
from pydantic import RootModel

from ..tools import ToolDefinition, normalize_tool, tool_result_to_codex


class _Response(RootModel[dict[str, Any]]):
    pass


def notification_data(event: Notification) -> dict[str, Any]:
    """Normalize typed and unknown SDK notifications at the transport boundary."""
    payload = event.payload
    if hasattr(payload, "params"):
        return payload.params
    if hasattr(payload, "model_dump"):
        return payload.model_dump(by_alias=True, mode="json", exclude_unset=True)
    if isinstance(payload, dict):
        return payload
    return {}


class _ObservedSdkClient(SdkClient):
    def __init__(self, config, approval_handler, observer) -> None:
        super().__init__(config=config, approval_handler=approval_handler)
        self._observer = observer

    def _coerce_notification(self, method: str, params: object) -> Notification:
        event = super()._coerce_notification(method, params)
        self._observer(event)
        return event


@dataclass
class _ToolBinding:
    tools: dict[str, ToolDefinition]
    context: contextvars.Context


class CodexTransport:
    def __init__(self, config: CodexConfig) -> None:
        self._loop = asyncio.get_running_loop()
        self._client = _ObservedSdkClient(config, self._server_request, self._receive)
        self._bindings: dict[str, _ToolBinding] = {}
        self._parents: dict[str, str] = {}
        self._threads: dict[str, Notification] = {}
        self._queues: set[asyncio.Queue] = set()
        self._active_threads: set[str] = set()
        self._monitor: asyncio.Task | None = None
        self._closed = False
        self._failure: BaseException | None = None

    async def start(self) -> None:
        try:
            await asyncio.to_thread(self._client.start)
            await asyncio.to_thread(self._client.initialize)
        except BaseException:
            await self.close()
            raise
        self._monitor = asyncio.create_task(self._drain_global_notifications())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._client.close)
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
        self._broadcast(RuntimeError("Codex transport closed."))
        self._bindings.clear()
        self._parents.clear()
        self._threads.clear()

    async def _drain_global_notifications(self) -> None:
        # The observer already delivered these events. Drain the SDK's global
        # queue to bound memory and to propagate transport failures to waiters.
        try:
            while True:
                await asyncio.to_thread(self._client.next_notification)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
            self._broadcast(exc)

    def _receive(self, event: Notification) -> None:
        # Called on the SDK reader thread. Record ancestry before a subsequent
        # dynamic-tool request, which may arrive before the event loop wakes.
        data = notification_data(event)
        if event.method == "thread/started":
            thread = data.get("thread", {})
            thread_id = thread.get("id")
            parent_id = thread.get("parentThreadId")
            if isinstance(thread_id, str):
                self._threads[thread_id] = event
                if isinstance(parent_id, str):
                    self._parents[thread_id] = parent_id
        if not self._closed:
            self._loop.call_soon_threadsafe(self._broadcast, event)

    def _broadcast(self, event: Notification | BaseException) -> None:
        for queue in tuple(self._queues):
            queue.put_nowait(event)

    def register_tools(self, thread_id: str, tools: list) -> None:
        definitions = [normalize_tool(tool) for tool in tools]
        self._bindings[thread_id] = _ToolBinding(
            {tool.name: tool for tool in definitions}, contextvars.copy_context()
        )

    def unregister_tools(self, thread_id: str) -> None:
        self._bindings.pop(thread_id, None)

    def _binding(self, thread_id: str) -> _ToolBinding | None:
        seen: set[str] = set()
        while thread_id and thread_id not in seen:
            seen.add(thread_id)
            if thread_id in self._bindings:
                return self._bindings[thread_id]
            thread_id = self._parents.get(thread_id, "")
        return None

    def _server_request(self, method: str, params: dict | None) -> dict:
        if method.endswith("/requestApproval"):
            if method == "item/permissions/requestApproval":
                return {"permissions": (params or {}).get("permissions", {}), "scope": "turn"}
            return {"decision": "accept"}
        if method != "item/tool/call":
            # Interactive input belongs to the shared ask_human tool. Do not
            # invent an answer to other kinds of server-initiated requests.
            return {}
        params = params or {}
        tool_name = params.get("tool", "")
        binding = self._binding(params.get("threadId", ""))
        tool = binding.tools.get(tool_name) if binding else None
        if tool is None:
            return self._tool_error(f"Tool {tool_name!r} is not registered for this thread.")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            return self._tool_error(f"Tool {tool_name!r} requires an object argument.")

        async def invoke() -> dict:
            return tool_result_to_codex(await tool.handler(arguments))

        try:
            future = binding.context.copy().run(
                asyncio.run_coroutine_threadsafe, invoke(), self._loop
            )
            return future.result()
        except Exception as exc:
            return self._tool_error(f"Tool {tool_name!r} failed: {exc}")

    @staticmethod
    def _tool_error(message: str) -> dict:
        return {"contentItems": [{"type": "inputText", "text": message}], "success": False}

    async def request(self, method: str, params: dict | None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Codex transport is closed.")
        if self._failure is not None:
            raise RuntimeError("Codex transport failed.") from self._failure
        response = await asyncio.to_thread(
            self._client.request, method, params, response_model=_Response
        )
        return response.root

    async def stream_turn(self, thread_id: str, message: str) -> AsyncIterator[Notification]:
        if thread_id in self._active_threads:
            raise RuntimeError("Concurrent turns on the same Codex session are not supported.")
        self._active_threads.add(thread_id)
        queue: asyncio.Queue = asyncio.Queue()
        self._queues.add(queue)
        completed = False
        turn_id = None
        try:
            # Rehydrate lineage for descendants surviving a previous turn.
            for event in tuple(self._threads.values()):
                queue.put_nowait(event)
            start_request = asyncio.create_task(
                self.request(
                    "turn/start",
                    {"threadId": thread_id, "input": [{"type": "text", "text": message}]},
                )
            )
            try:
                started = await asyncio.shield(start_request)
            except asyncio.CancelledError:
                # Cancelling a Python waiter does not cancel a JSON-RPC request.
                # Recover the newly created turn so the finally block can stop it.
                try:
                    started = await asyncio.shield(start_request)
                    turn_id = started["turn"]["id"]
                except Exception:
                    pass
                raise
            turn_id = started["turn"]["id"]
            while True:
                event = await queue.get()
                if isinstance(event, BaseException):
                    raise event
                if event.method == "turn/completed":
                    data = notification_data(event)
                    if data.get("turn", {}).get("id") == turn_id:
                        completed = True
                yield event
                if completed:
                    break
        finally:
            self._queues.discard(queue)
            self._active_threads.discard(thread_id)
            if turn_id is not None and not completed and not self._closed:
                try:
                    await asyncio.wait_for(
                        self.request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id}),
                        timeout=5,
                    )
                except Exception:
                    pass
