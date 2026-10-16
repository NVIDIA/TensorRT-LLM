# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import AsyncExitStack
from typing import Literal
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
from tensorrt_llm.llmapi.disagg_utils import DisaggServerConfig, ServerRole
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DisaggregatedParams,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.responses_utils import UCompletionResponseOrGenerator
from tensorrt_llm.serve.router import Router

pytestmark = pytest.mark.cpu_only


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("context_outcome", ["error", "cancelled"])
async def test_context_termination_reaches_paired_worker_abort(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    context_outcome: Literal["error", "cancelled"],
) -> None:
    """Exercise HTTP disconnect through the worker's real abort entry points.

    Model execution is replaced by an engine spy. This checks cancellation
    delivery, not backend quiescence or physical resource retirement.
    """
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", mock.Mock())
    gen_started = asyncio.Event()
    engine_cancelled = asyncio.Event()
    gen_finished = asyncio.Event()
    engine = mock.Mock()
    engine.can_enqueue_requests.return_value = True
    engine.cancel_request.side_effect = lambda request_id: engine_cancelled.set()
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = True
    worker.engine = engine
    worker._client_id_to_request_id = {42: 101}
    generation_request = GenerationRequest([1], SamplingParams(), streaming=stream)
    generation_request.id = 42
    promise = GenerationResult(generation_request, executor=worker)
    server = object.__new__(OpenAIServer)
    watchers: list[asyncio.Task[None]] = []

    async def context_handler(http_request: web.Request) -> web.Response:
        """Return the CTX outcome only after GEN has accepted its request."""
        await http_request.read()
        await asyncio.wait_for(gen_started.wait(), timeout=5)
        if context_outcome == "error":
            return web.Response(status=500, text="context failed before transfer")
        response = CompletionResponse(
            model="test-model",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=0, total_tokens=1),
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text="",
                    finish_reason="cancelled",
                    disaggregated_params=DisaggregatedParams(
                        request_type="context_only", disagg_request_id=42, ctx_request_id=42
                    ),
                )
            ],
        )
        return web.json_response(response.model_dump(mode="json"))

    async def generation_handler(http_request: web.Request) -> web.Response:
        """Bridge the actual HTTP socket state to Starlette's disconnect API."""
        body = msgspec.msgpack.decode(await http_request.read())
        assert body["disaggregated_params"]["ctx_request_id"] == 42
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
        gen_started.set()
        try:
            await watcher
            return web.Response(status=499)
        finally:
            gen_finished.set()

    config = DisaggServerConfig(server_configs=[], schedule_style="generation_first")
    coordinator = DisaggCoordinatorService(config, client_factory=mock.Mock())
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=mock.Mock())
    coordinator.get_disagg_request_id = mock.AsyncMock(return_value=42)
    service._check_gen_only_disagg = mock.AsyncMock(return_value=False)
    request_task: asyncio.Task[UCompletionResponseOrGenerator] | None = None
    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(aiohttp.ClientSession())
        for role, handler, router_attr, client_attr in (
            (ServerRole.CONTEXT, context_handler, "_ctx_router", "_ctx_client"),
            (ServerRole.GENERATION, generation_handler, "_gen_router", "_gen_client"),
        ):
            app = web.Application()
            app.router.add_post("/v1/completions", handler)
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
            if context_outcome == "error":
                with pytest.raises(aiohttp.ClientResponseError, match="context failed"):
                    await asyncio.wait_for(asyncio.shield(request_task), timeout=3)
            else:
                response = await asyncio.wait_for(asyncio.shield(request_task), timeout=3)
                if stream:
                    assert [chunk async for chunk in response] == [b"data: [DONE]\n\n"]
                else:
                    assert response.choices[0].finish_reason == "cancelled"

            await asyncio.wait_for(engine_cancelled.wait(), timeout=3)
            await asyncio.wait_for(gen_finished.wait(), timeout=3)
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
