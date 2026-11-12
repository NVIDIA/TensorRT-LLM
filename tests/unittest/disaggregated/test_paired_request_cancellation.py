# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import AsyncExitStack
from unittest import mock

import aiohttp
import msgspec
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from starlette.requests import Request
from starlette.types import Message

from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.llmapi.disagg_utils import DisaggServerConfig, ServerRole
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import CompletionRequest
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.responses_utils import UCompletionResponseOrGenerator
from tensorrt_llm.serve.router import CoordinatorDelegatingRouter, LoadBalancingRouter, Router

pytestmark = pytest.mark.cpu_only


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved_before_cancel", [False, True])
async def test_ctx_failure_releases_pending_gen_placement(
    monkeypatch: pytest.MonkeyPatch, reserved_before_cancel: bool
) -> None:
    """Cancellation must release a late GEN placement before reservation expiry."""
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", mock.Mock())
    config = DisaggServerConfig(server_configs=[], schedule_style="generation_first")
    coordinator = DisaggCoordinatorService(config, client_factory=mock.Mock())
    coordinator._gen_router = LoadBalancingRouter(ServerRole.GENERATION, servers=["gen:8000"])
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=mock.Mock())
    coordinator.get_disagg_request_id = mock.AsyncMock(return_value=42)
    service._check_gen_only_disagg = mock.AsyncMock(return_value=False)
    service._ctx_router = mock.AsyncMock(spec=Router)
    service._ctx_router.get_next_server.return_value = ("ctx:8000", {})
    selection_paused, release_selection = asyncio.Event(), asyncio.Event()
    selection_completed = asyncio.Event()
    gen_cancel_requested = asyncio.Event()
    finished_ids = []
    gen_tasks = []
    ctx_error = RuntimeError("CTX failed during GEN selection")

    async def select(http_request: web.Request) -> web.Response:
        body = msgspec.msgpack.decode(await http_request.read())

        async def reserve() -> tuple[str, dict, int]:
            return await coordinator.select(
                body["role"], body["routing_key"], body["req_id"], body["exclude_server"]
            )

        if reserved_before_cancel:
            selected = await reserve()
        selection_paused.set()
        await release_selection.wait()
        if not reserved_before_cancel:
            selected = await reserve()
        server, info, request_id = selected
        selection_completed.set()
        return web.Response(
            body=msgspec.msgpack.encode({"server": server, "info": info, "req_id": request_id})
        )

    async def finish(http_request: web.Request) -> web.Response:
        body = msgspec.msgpack.decode(await http_request.read())
        finished_ids.append((body["req_id"], body["success"]))
        await coordinator.finish(body["role"], body["req_id"], body["success"])
        return web.Response(body=msgspec.msgpack.encode({}))

    async def fail_ctx(*args: object, **kwargs: object) -> None:
        await selection_paused.wait()
        raise ctx_error

    service._ctx_client = mock.Mock(send_request=mock.AsyncMock(side_effect=fail_ctx))
    app = web.Application()
    app.router.add_post("/select", select)
    app.router.add_post("/finish", finish)
    async with TestServer(app) as endpoint:
        router = CoordinatorDelegatingRouter(
            str(endpoint.make_url("/")).rstrip("/"),
            LoadBalancingRouter(ServerRole.GENERATION, servers=["gen:8000"]),
            "generation",
        )
        get_next_server = router.get_next_server

        async def track_gen(*args: object, **kwargs: object) -> tuple[str, dict]:
            task = asyncio.current_task()
            gen_tasks.append(task)
            cancel = task.cancel

            def cancel_gen(msg: str | None = None) -> bool:
                cancelled = cancel(msg)
                gen_cancel_requested.set()
                return cancelled

            monkeypatch.setattr(task, "cancel", cancel_gen)
            return await get_next_server(*args, **kwargs)

        monkeypatch.setattr(router, "get_next_server", track_gen)
        session = mock.Mock(spec=aiohttp.ClientSession)
        service._gen_client = OpenAIHttpClient(router, ServerRole.GENERATION, session=session)
        pending = asyncio.create_task(
            service._send_disagg_request(CompletionRequest(model="m", prompt="hello"))
        )
        try:
            await asyncio.wait_for(selection_paused.wait(), timeout=3)
            await asyncio.wait_for(gen_cancel_requested.wait(), timeout=3)
            assert not gen_tasks[0].done()
            assert not finished_ids
            assert coordinator.gen_router._server_state["gen:8000"]._num_active_requests == int(
                reserved_before_cancel
            )
            release_selection.set()
            with pytest.raises(RuntimeError) as raised:
                await asyncio.wait_for(asyncio.shield(pending), timeout=3)
            assert raised.value is ctx_error
            await asyncio.wait_for(router._finish_queue.join(), timeout=3)
            assert finished_ids == [(42, False)]
            assert coordinator._reservation_tasks == {}
            assert coordinator.gen_router._server_state["gen:8000"]._num_active_requests == 0
            assert gen_tasks[0].cancelled()
            session.post.assert_not_called()
        finally:
            release_selection.set()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            if selection_paused.is_set():
                await asyncio.wait_for(selection_completed.wait(), timeout=3)
            await router.close()
            # Cleanup is deliberately after the release/load assertions.
            await coordinator.finish("generation", 42, success=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream", "failed_role", "sse_error"),
    [
        (False, ServerRole.CONTEXT, False),
        (True, ServerRole.CONTEXT, False),
        (False, ServerRole.GENERATION, False),
        (True, ServerRole.GENERATION, False),
        (True, ServerRole.GENERATION, True),
    ],
)
async def test_http_failure_reaches_paired_worker_abort(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    failed_role: ServerRole,
    sse_error: bool,
) -> None:
    """Exercise HTTP disconnect through the worker's real abort entry points.

    Model execution is replaced by an engine spy. This checks cancellation
    delivery, not backend quiescence or physical resource retirement.
    """
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    if stream and failed_role == ServerRole.GENERATION:
        # A rejected streaming GEN must cancel CTX without replaying GEN,
        # even when the deployment allows retries.
        monkeypatch.delenv("TRTLLM_DISAGG_NO_RETRY", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", mock.Mock())
    sibling_started = asyncio.Event()
    engine_cancelled = asyncio.Event()
    sibling_finished = asyncio.Event()
    engine = mock.Mock()
    engine.can_enqueue_requests.return_value = True
    engine.cancel_request.side_effect = lambda request_id: engine_cancelled.set()
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = True
    worker.engine = engine
    worker._client_id_to_request_id = {42: 101}
    generation_request = GenerationRequest(
        [1], SamplingParams(), streaming=stream and failed_role == ServerRole.CONTEXT
    )
    generation_request.id = 42
    promise = GenerationResult(generation_request, executor=worker)
    server = object.__new__(OpenAIServer)
    watchers: list[asyncio.Task[None]] = []
    error_message = (
        "context failed before transfer"
        if failed_role == ServerRole.CONTEXT
        else context_length_exceeded_message(2048, 4096)
    )
    error_status = 500 if failed_role == ServerRole.CONTEXT else 400
    error_event = (
        b"data: "
        + msgspec.json.encode(
            {
                "error": {
                    "message": "generation executor failed",
                    "type": "server_error",
                    "code": None,
                    "param": None,
                }
            }
        )
        + b"\n\n"
    )
    done_event = b"data: [DONE]\n\n"
    failed_requests = 0

    async def failure_handler(http_request: web.Request) -> web.StreamResponse:
        """Reject only after the paired worker has accepted its request."""
        nonlocal failed_requests
        failed_requests += 1
        await http_request.read()
        await asyncio.wait_for(sibling_started.wait(), timeout=5)
        if sse_error:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(http_request)
            await response.write(error_event)
            # The service must cancel CTX on the error event, not at GEN EOF.
            await asyncio.wait_for(engine_cancelled.wait(), timeout=5)
            await response.write(done_event)
            await response.write_eof()
            return response
        if failed_role == ServerRole.CONTEXT:
            return web.Response(status=error_status, text=error_message)
        error_response = OpenAIServer.create_error_response(error_message)
        return web.Response(
            status=error_status, body=error_response.body, content_type="application/json"
        )

    async def pending_handler(http_request: web.Request) -> web.Response:
        """Bridge the actual HTTP socket state to Starlette's disconnect API."""
        body = msgspec.msgpack.decode(await http_request.read())
        pairing_key = "ctx_request_id" if failed_role == ServerRole.CONTEXT else "disagg_request_id"
        assert body["disaggregated_params"][pairing_key] == 42
        transport = http_request.transport
        assert transport is not None

        async def receive() -> Message:
            """Report disconnect only when the client closes the real socket."""
            if transport.is_closing():
                return {"type": "http.disconnect"}
            return {"type": "http.request", "body": b"", "more_body": True}

        raw_request = Request({"type": "http", "headers": []}, receive=receive)
        watcher = asyncio.create_task(server.await_disconnected(raw_request, promise))
        watchers.append(watcher)
        sibling_started.set()
        try:
            await asyncio.shield(watcher)
            return web.Response(status=499)
        finally:
            sibling_finished.set()

    config = DisaggServerConfig(server_configs=[], schedule_style="generation_first")
    coordinator = DisaggCoordinatorService(config, client_factory=mock.Mock())
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=mock.Mock())
    coordinator.get_disagg_request_id = mock.AsyncMock(return_value=42)
    service._check_gen_only_disagg = mock.AsyncMock(return_value=False)
    request_task: asyncio.Task[UCompletionResponseOrGenerator] | None = None
    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(aiohttp.ClientSession())
        for role, router_attr, client_attr in (
            (ServerRole.CONTEXT, "_ctx_router", "_ctx_client"),
            (ServerRole.GENERATION, "_gen_router", "_gen_client"),
        ):
            app = web.Application()
            app.router.add_post(
                "/v1/completions", failure_handler if role == failed_role else pending_handler
            )
            endpoint = await stack.enter_async_context(TestServer(app))
            router = mock.AsyncMock(spec=Router)
            router.get_next_server.return_value = (
                f"{endpoint.host}:{endpoint.port}",
                {"server_info": {}},
            )
            setattr(service, router_attr, router)
            setattr(
                service,
                client_attr,
                OpenAIHttpClient(
                    router, role, timeout_secs=10, retry_interval_sec=0, session=session
                ),
            )
        try:
            request_task = asyncio.create_task(
                service._send_disagg_request(
                    CompletionRequest(model="test-model", prompt="hello", stream=stream)
                )
            )
            if sse_error:
                response = await asyncio.wait_for(asyncio.shield(request_task), timeout=3)

                async def read_stream() -> bytes:
                    return b"".join([chunk async for chunk in response])

                assert await asyncio.wait_for(read_stream(), timeout=3) == error_event + done_event
            else:
                with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                    await asyncio.wait_for(asyncio.shield(request_task), timeout=3)
                assert exc_info.value.status == error_status
                assert error_message in exc_info.value.message
                if failed_role == ServerRole.GENERATION:
                    disagg_server = object.__new__(OpenAIDisaggServer)
                    disagg_server._perf_metrics_collector = mock.Mock()
                    response = disagg_server._handle_exception(exc_info.value)
                    assert response.status_code == 400
                    assert msgspec.json.decode(response.body)["code"] == "context_length_exceeded"

            await asyncio.wait_for(engine_cancelled.wait(), timeout=3)
            await asyncio.wait_for(sibling_finished.wait(), timeout=3)
            engine.cancel_request.assert_called_once_with(101)
            assert failed_requests == 1
            assert promise.aborted()
            assert not promise.finished
        finally:
            # Cancellation proof above must precede any teardown that closes HTTP.
            if request_task is not None:
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
            for watcher in watchers:
                watcher.cancel()
            await asyncio.gather(*watchers, return_exceptions=True)
