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
from tensorrt_llm.serve.router import Router

pytestmark = pytest.mark.cpu_only


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

    async def failure_handler(http_request: web.Request) -> web.StreamResponse:
        """Reject only after the paired worker has accepted its request."""
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
                OpenAIHttpClient(router, role, timeout_secs=10, session=session),
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
