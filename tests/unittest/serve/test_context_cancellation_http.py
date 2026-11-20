# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import aiohttp
import anyio
import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import HTTPException, Request
from starlette.requests import ClientDisconnect
from starlette.types import Message
from test_common.serve.app_factory import AggConfig, ServeTestEnv

from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.executor.rpc import RPCError, RPCTimeout
from tensorrt_llm.llmapi.disagg_utils import CtxGenServerConfig, DisaggServerConfig, ServerRole
from tensorrt_llm.serve.disagg_auth import (
    CONTEXT_CANCELLATION_AUTH_HEADER,
    build_context_cancellation_headers,
    context_cancellation_reservation_payload,
    context_cancellation_submission_payload,
    validate_context_cancellation_auth,
)
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer, _ClientDisconnected
from tensorrt_llm.serve.openai_disagg_service import (
    OpenAIDisaggregatedService,
    _ContextCancellation,
)
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ContextCancellationRequest,
    ContextCancellationReservationRequest,
    DisaggregatedParams,
    UsageInfo,
    to_llm_disaggregated_params,
)
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.output_length_predictor import OutputLengthPriority
from tensorrt_llm.serve.responses_utils import ClosingStreamingResponse
from tensorrt_llm.serve.router import Router

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def context_worker(serve_env: ServeTestEnv) -> OpenAIServer:
    """Build a real CTX HTTP server with isolated process state and fake execution."""
    worker = serve_env.build_openai_server(
        AggConfig(role=ServerRole.CONTEXT, env=(("TRTLLM_DISAGG_NO_RETRY", "1"),))
    )
    worker._internal_disagg_auth_key = "secret"
    worker.generator.args.cache_transceiver_config = {"backend": "DEFAULT"}
    return worker


def _raw_request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
        }
    )


def _response(finish_reason: str = "length") -> CompletionResponse:
    return CompletionResponse(
        model="model",
        prompt_token_ids=[1, 2],
        usage=UsageInfo(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        choices=[
            CompletionResponseChoice(
                index=0,
                text="",
                finish_reason=finish_reason,
                disaggregated_params=DisaggregatedParams(
                    request_type="context_only",
                    ctx_request_id=7,
                    disagg_request_id=42,
                    schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
                ),
            )
        ],
    )


def _service(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True
) -> OpenAIDisaggregatedService:
    monkeypatch.setenv("TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT", str(int(enabled)))
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    config = DisaggServerConfig(
        server_configs=[], schedule_style="context_first", internal_request_auth_key="secret"
    )
    coordinator = SimpleNamespace(
        ctx_router=AsyncMock(spec=Router),
        gen_router=AsyncMock(spec=Router),
        get_disagg_request_id=AsyncMock(return_value=42),
    )
    service = OpenAIDisaggregatedService(config, coordinator, Mock())
    service._check_conditional_disagg = AsyncMock(return_value=(None, True))
    service._check_gen_only_disagg = AsyncMock(return_value=False)
    service._ctx_router.get_next_server.return_value = ("ctx:8000", {})
    service._gen_router.get_next_server.return_value = ("gen:8001", {})
    service._ctx_client = SimpleNamespace(
        reserve_context_cancellation=AsyncMock(return_value="worker-token"),
        cancel_context_cancellation=AsyncMock(return_value="queued"),
        send_request=AsyncMock(return_value=_response()),
    )
    service._gen_client = SimpleNamespace(send_request=AsyncMock(return_value=_response("stop")))
    return service


def test_cancellation_auth_is_required_and_bound_to_action_and_identity() -> None:
    payload = {"token": "opaque"}
    headers = build_context_cancellation_headers("secret", "cancel", payload)
    validate_context_cancellation_auth("secret", "cancel", payload, headers)
    for key, action, body, supplied in [
        (None, "cancel", payload, headers),
        ("secret", "reserve", payload, headers),
        ("secret", "cancel", {"token": "other"}, headers),
        ("secret", "cancel", payload, {}),
    ]:
        with pytest.raises(ValueError):
            validate_context_cancellation_auth(key, action, body, supplied)


def test_submission_token_reaches_executor_only_with_authenticated_ctx_identity(
    context_worker: OpenAIServer,
) -> None:
    request = CompletionRequest(
        model="model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(
            request_type="context_only",
            disagg_request_id=42,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            context_cancellation_token="opaque",
        ),
    )
    server = context_worker
    headers = build_context_cancellation_headers(
        "secret", "submit", context_cancellation_submission_payload(request)
    )
    server._validate_internal_disagg_request(request, _raw_request(headers))
    assert (
        to_llm_disaggregated_params(request.disaggregated_params).context_cancellation_token
        == "opaque"
    )
    request.disaggregated_params.disagg_request_id = 43
    with pytest.raises(ValueError, match="authentication"):
        server._validate_internal_disagg_request(request, _raw_request(headers))
    request.disaggregated_params.request_type = "generation_only"
    with pytest.raises(ValueError, match="context-first"):
        server._validate_internal_disagg_request(request, _raw_request(headers))


@pytest.mark.asyncio
async def test_control_round_trip_uses_worker_rpc_and_reports_only_queue_receipt(
    monkeypatch: pytest.MonkeyPatch,
    context_worker: OpenAIServer,
) -> None:
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", Mock())
    worker = context_worker
    executor = worker.generator._executor
    executor.reserve_context_cancellation = Mock(return_value="opaque")
    executor.cancel_context_cancellation = Mock(return_value="queued")

    async def control(request: web.Request) -> web.Response:
        """Forward signed traffic through the factory-built app's actual routes."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
        ) as asgi:
            response = await asgi.post(
                request.path, json=await request.json(), headers=dict(request.headers)
            )
        assert response.status_code == (200 if request.match_info["action"] == "reserve" else 202)
        return web.Response(
            body=response.content, status=response.status_code, content_type="application/json"
        )

    app = web.Application()
    app.router.add_post("/internal/context-cancellation/{action}", control)
    async with TestServer(app) as http, aiohttp.ClientSession() as session:
        client = OpenAIHttpClient(
            Mock(spec=Router),
            ServerRole.CONTEXT,
            session=session,
            internal_disagg_auth_key="secret",
        )
        server = str(http.make_url("/"))
        assert await client.reserve_context_cancellation(server, 42) == "opaque"
        assert await client.cancel_context_cancellation(server, "opaque") == "queued"
    executor.reserve_context_cancellation.assert_called_once()
    request_id, nonce, expiry = executor.reserve_context_cancellation.call_args.args
    assert request_id == 42 and len(nonce) == 32 and isinstance(expiry, int)
    executor.cancel_context_cancellation.assert_called_once_with("opaque")
    with pytest.raises(HTTPException) as error:
        await worker.cancel_context_cancellation(
            ContextCancellationRequest(token="opaque"), _raw_request({})
        )
    assert error.value.status_code == 403
    assert executor.cancel_context_cancellation.call_count == 1


@pytest.mark.asyncio
async def test_cancel_during_reservation_keeps_late_token_and_rejects_ctx_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()

    async def reserve(*args: object) -> str:
        started.set()
        await release.wait()
        return "late-token"

    service._ctx_client.reserve_context_cancellation.side_effect = reserve
    task = asyncio.create_task(
        service._send_disagg_request_ctx_first(CompletionRequest(model="model", prompt="hello"))
    )
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    service._ctx_client.send_request.assert_not_awaited()
    service._ctx_client.cancel_context_cancellation.assert_awaited_once_with(
        "ctx:8000", "late-token"
    )
    service._ctx_router.finish_request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_ctx_response", [False, True])
async def test_cancel_before_gen_dispatch_cancels_ctx_even_after_http_result(
    monkeypatch: pytest.MonkeyPatch,
    after_ctx_response: bool,
) -> None:
    service = _service(monkeypatch)
    paused = asyncio.Event()

    async def block(*args: object, **kwargs: object) -> None:
        paused.set()
        await asyncio.Future()

    blocked = (
        service._gen_router.get_next_server
        if after_ctx_response
        else service._ctx_client.send_request
    )
    blocked.side_effect = block
    task = asyncio.create_task(
        service._send_disagg_request_ctx_first(CompletionRequest(model="model", prompt="hello"))
    )
    await asyncio.wait_for(paused.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    service._ctx_client.cancel_context_cancellation.assert_awaited_once_with(
        "ctx:8000", "worker-token"
    )
    service._gen_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_releases_pre_reserved_gen_without_creating_cleanup_gen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(monkeypatch)
    service._check_conditional_disagg.return_value = ("gen:8001", True)
    paused = asyncio.Event()

    async def context(*args: object, **kwargs: object) -> None:
        paused.set()
        await asyncio.Future()

    service._ctx_client.send_request.side_effect = context
    request = CompletionRequest(model="model", prompt="hello")
    task = asyncio.create_task(service._send_disagg_request_ctx_first(request))
    await asyncio.wait_for(paused.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    service._gen_router.finish_request.assert_awaited_once()
    assert service._gen_router.finish_request.call_args.args[0] is request
    assert service._gen_router.finish_request.call_args.kwargs["req_id"] == 42
    service._gen_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses_api", [False, True])
async def test_close_before_first_gen_chunk_closes_source_and_cancels_ctx(
    responses_api: bool,
) -> None:
    closed = asyncio.Event()

    async def source() -> AsyncGenerator[None, None]:
        try:
            yield None
            pytest.fail("No GEN request should start")
        finally:
            closed.set()

    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(return_value="queued"))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"
    gen = source()
    await gen.__anext__()
    stream = cancellation.stream(gen, responses_api)
    await stream.__anext__()
    await stream.aclose()
    assert closed.is_set()
    client.cancel_context_cancellation.assert_awaited_once_with("ctx:8000", "opaque")


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_after", [0, 1, None], ids=["before_read", "after_frame", "complete"])
async def test_tracked_context_stream_closes_source_and_preserves_learning(
    monkeypatch: pytest.MonkeyPatch, stop_after: int | None
) -> None:
    """Tracking cannot hide the primed HTTP source or learn an interrupted reply."""
    service = _service(monkeypatch)
    coordinator = service._coordinator
    coordinator.predict_output_length = AsyncMock(return_value=(20.0, 7))
    coordinator.record_output_lengths = AsyncMock()
    tracker = OutputLengthPriority(coordinator, "ridge")
    service._output_length_priority = tracker
    frame = b'data: {"choices":[{"text":"hello"}]}\n\n'
    usage = b'data: {"choices":[],"usage":{"completion_tokens":3}}\n\n'
    done = b"data: [DONE]\n\n"
    read, closed = [], []

    async def source() -> AsyncGenerator[bytes | None, None]:
        try:
            yield None
            for chunk in (frame, usage, done):
                read.append(chunk)
                yield chunk
        finally:
            closed.append(True)

    original = source()
    await original.__anext__()
    service._gen_client.send_request.return_value = original
    stream = await service._send_disagg_request_ctx_first(
        CompletionRequest(model="model", prompt="hello", stream=True)
    )
    try:
        if stop_after is None:
            assert [chunk async for chunk in stream] == [frame, done]
            assert read == [frame, usage, done]
        else:
            if stop_after:
                assert await stream.__anext__() == frame
            assert read == ([frame] if stop_after else [])
            await stream.aclose()
        assert closed == [True]
        assert service._ctx_client.cancel_context_cancellation.await_count == int(
            stop_after is not None
        )
    finally:
        await stream.aclose()
        await original.aclose()
        if tracker._flush_task is not None:
            await asyncio.wait_for(tracker._flush_task, 2)
        await asyncio.wait_for(asyncio.gather(*tracker._tasks), 2)
    coordinator.predict_output_length.assert_awaited_once_with(None, 2.0)
    coordinator.record_output_lengths.assert_awaited_once_with(
        [(None, 2.0, 3.0 if stop_after is None else None, 7)]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_healthy_handoff_and_legacy_do_not_cancel(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    service = _service(monkeypatch, enabled=enabled)
    if enabled:
        params = service._ctx_client.send_request.return_value.choices[0].disaggregated_params
        params.context_cancellation_token = "worker-token"
    result = await service._send_disagg_request_ctx_first(
        CompletionRequest(model="model", prompt="hello")
    )
    assert result.choices[0].finish_reason == "stop"
    service._ctx_client.cancel_context_cancellation.assert_not_awaited()
    assert service._ctx_client.reserve_context_cancellation.await_count == int(enabled)
    ctx_send = service._ctx_client.send_request.await_args
    ctx_params = ctx_send.args[0].disaggregated_params
    assert ctx_send.kwargs["server"] == "ctx:8000"
    assert ctx_params.disagg_request_id == ctx_send.kwargs["req_id"] == 42
    assert ctx_params.context_cancellation_token == ("worker-token" if enabled else None)
    if enabled:
        service._ctx_client.reserve_context_cancellation.assert_awaited_once_with(
            ctx_send.kwargs["server"], ctx_params.disagg_request_id
        )
    gen_request = service._gen_client.send_request.call_args.args[0]
    assert gen_request.disaggregated_params.context_cancellation_token is None


@pytest.mark.asyncio
async def test_prelude_disconnect_cancels_service_while_ctx_is_inflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(monkeypatch)
    paused, disconnected = asyncio.Event(), asyncio.Event()

    async def context(*args: object, **kwargs: object) -> None:
        paused.set()
        await asyncio.Future()

    async def receive() -> Message:
        return {"type": "http.disconnect"} if disconnected.is_set() else {"type": "http.request"}

    service._ctx_client.send_request.side_effect = context
    raw = Request({"type": "http", "headers": []}, receive)
    task = asyncio.create_task(
        OpenAIDisaggServer._await_context_request(
            service._send_disagg_request_ctx_first,
            CompletionRequest(model="model", prompt="hello"),
            None,
            raw,
        )
    )
    await asyncio.wait_for(paused.wait(), 2)
    disconnected.set()
    with pytest.raises(_ClientDisconnected):
        await asyncio.wait_for(task, 2)
    service._ctx_client.cancel_context_cancellation.assert_awaited_once()
    service._gen_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["response", "error", "cancel"])
async def test_prelude_stops_watcher_when_disconnect_probe_consumes_cancellation(
    outcome: str,
) -> None:
    """A probe's cancellation scope cannot keep response cleanup waiting for disconnect."""
    probing, consumed, disconnect = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_probe = asyncio.Event()

    async def is_disconnected() -> bool:
        if probing.is_set():
            return disconnect.is_set()
        probing.set()
        try:
            await release_probe.wait()
        except asyncio.CancelledError:
            consumed.set()
        return False

    response = _response()

    async def entry(*args: object) -> CompletionResponse:
        await probing.wait()
        if outcome == "cancel":
            await asyncio.Future()
        if outcome == "error":
            raise RuntimeError("service failed")
        return response

    task = asyncio.create_task(
        OpenAIDisaggServer._await_context_request(
            entry,
            CompletionRequest(model="model", prompt="hello"),
            None,
            SimpleNamespace(is_disconnected=is_disconnected),
        )
    )
    try:
        await asyncio.wait_for(probing.wait(), 2)
        if outcome == "cancel":
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=2)
        assert task in done, "Watcher cleanup must not require a client disconnect"
        assert consumed.is_set()
        assert not release_probe.is_set()
        if outcome == "cancel":
            assert task.cancelled()
        elif outcome == "error":
            with pytest.raises(RuntimeError, match="service failed"):
                task.result()
        else:
            assert task.result() is response
    finally:
        # Release the probe even if the bounded completion assertion fails.
        release_probe.set()
        disconnect.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("unconsumed_stream", [False, True])
async def test_prelude_cancelled_scope_waits_for_cleanup_without_spinning(
    monkeypatch: pytest.MonkeyPatch, unconsumed_stream: bool
) -> None:
    """Level cancellation and direct task cancellation cannot abandon CTX delivery."""
    ready, delivering, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    scopes = []
    original_shield = asyncio.shield
    shields = 0

    def bounded_shield(awaitable):
        """Fail a repeated-cancellation spin without blocking the test event loop."""
        nonlocal shields
        shields += 1
        assert shields <= 64, "Prelude cleanup must not spin in a cancelled scope"
        return original_shield(awaitable)

    monkeypatch.setattr(asyncio, "shield", bounded_shield)

    async def deliver(*args: object) -> str:
        """Hold real cancellation delivery until the observer permits its receipt."""
        delivering.set()
        await release.wait()
        return "queued"

    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(side_effect=deliver))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"

    async def source() -> AsyncGenerator[bytes, None]:
        """An unconsumed GEN stream must never dispatch model work."""
        pytest.fail("GEN must not start")
        yield b""

    async def entry(*args: object):
        """Expose either an in-flight service or a primed, unconsumed response."""
        if unconsumed_stream:
            stream = cancellation.stream(source(), False)
            await stream.__anext__()
            ready.set()
            return stream
        ready.set()
        try:
            await asyncio.Future()
        finally:
            await cancellation.cancel()

    async def is_disconnected() -> bool:
        """The stream variant races disconnect with its completed prelude."""
        await ready.wait()
        return unconsumed_stream

    async def run() -> None:
        """Mirror the request's ASGI cancellation scope."""
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            with pytest.raises(asyncio.CancelledError):
                await OpenAIDisaggServer._await_context_request(
                    entry,
                    CompletionRequest(model="model", prompt="hello"),
                    None,
                    SimpleNamespace(is_disconnected=is_disconnected),
                )

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(ready.wait(), 1)
        if not unconsumed_stream:
            scopes[0].cancel()
        await asyncio.wait_for(delivering.wait(), 1)
        scopes[0].cancel()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "The cancellation receipt remains pending"
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    assert cancellation._token is None
    client.cancel_context_cancellation.assert_awaited_once_with("ctx:8000", "opaque")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses_api", "frame", "cancelled"),
    [
        (False, b"data: [DONE]\n\n", False),
        (False, b'data: {"choices":[{"text":"partial"}]}\n\n', True),
        (False, b'data: {"error":{"message":"failed"}}\n\n', True),
        (True, b'event: response.completed\ndata: {"type":"response.completed"}\n\n', False),
        (True, b'event: response.incomplete\ndata: {"type":"response.incomplete"}\n\n', False),
        (
            True,
            b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta"}\n\n',
            True,
        ),
        (True, b'event: response.failed\ndata: {"type":"response.failed"}\n\n', True),
    ],
    ids=[
        "done",
        "clean_eof",
        "error",
        "responses_done",
        "responses_incomplete",
        "responses_eof",
        "responses_error",
    ],
)
async def test_stream_completion_requires_a_terminal_event(
    responses_api: bool, frame: bytes, cancelled: bool
) -> None:
    """Clean EOF is not completion; protocol terminals and errors remain distinct."""
    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(return_value="queued"))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"

    chunks = (frame[: len(frame) // 2], frame[len(frame) // 2 :])

    async def source() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    stream = cancellation.stream(source(), responses_api)
    await stream.__anext__()
    assert [chunk async for chunk in stream] == list(chunks)
    assert client.cancel_context_cancellation.await_count == int(cancelled)


@pytest.mark.asyncio
async def test_disconnect_winning_completed_prelude_closes_unconsumed_stream() -> None:
    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(return_value="queued"))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"

    async def source() -> AsyncGenerator[bytes, None]:
        pytest.fail("GEN must not start")
        yield b""

    async def entry(*args: object) -> AsyncGenerator[str | bytes | None, None]:
        stream = cancellation.stream(source(), False)
        await stream.__anext__()
        return stream

    raw = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
    with pytest.raises(_ClientDisconnected):
        await OpenAIDisaggServer._await_context_request(
            entry, CompletionRequest(model="model", prompt="hello"), None, raw
        )
    client.cancel_context_cancellation.assert_awaited_once_with("ctx:8000", "opaque")


@pytest.mark.parametrize("missing", ["auth", "no_retry"])
def test_protected_coordinator_rejects_missing_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    monkeypatch.setenv("TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT", "1")
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "0" if missing == "no_retry" else "1")
    config = DisaggServerConfig(
        server_configs=[],
        schedule_style="context_first",
        internal_request_auth_key=None if missing == "auth" else "secret",
    )
    coordinator = SimpleNamespace(ctx_router=Mock(), gen_router=Mock())
    with pytest.raises(ValueError, match="requires"):
        OpenAIDisaggregatedService(config, coordinator, Mock())


@pytest.mark.asyncio
async def test_context_eos_needs_no_generation_and_keeps_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(monkeypatch)
    service._ctx_client.send_request.return_value = _response("stop")
    result = await service._send_disagg_request_ctx_first(
        CompletionRequest(model="model", prompt="hello")
    )
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].disaggregated_params is None
    service._gen_client.send_request.assert_not_awaited()
    service._ctx_client.cancel_context_cancellation.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_delivery_is_not_claimed_as_worker_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(monkeypatch)
    original = RuntimeError("CTX failed")
    service._ctx_client.send_request.side_effect = original
    service._ctx_client.cancel_context_cancellation.side_effect = asyncio.TimeoutError()
    with pytest.raises(RuntimeError, match="CTX failed"):
        await service._send_disagg_request_ctx_first(
            CompletionRequest(model="model", prompt="hello")
        )
    assert service._ctx_client.cancel_context_cancellation.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status"),
    [
        (RPCError("worker failed"), 503),
        (RPCTimeout("unknown outcome"), 503),
        (RPCError("worker rejection", cause=ValueError("invalid token")), 409),
        (RPCError("worker rejection", cause=RuntimeError("closed admission")), 409),
        (RuntimeError("closed admission"), 409),
    ],
    ids=["rpc_failure", "timeout", "wrapped_invalid", "wrapped_closed", "direct_closed"],
)
@pytest.mark.parametrize("action", ["reserve", "cancel"])
async def test_control_rpc_error_is_unavailable_not_a_cancel_receipt(
    error: Exception, status: int, action: str, context_worker: OpenAIServer
) -> None:
    worker = context_worker
    worker.generator._executor.reserve_context_cancellation = Mock(side_effect=error)
    worker.generator._executor.cancel_context_cancellation = Mock(side_effect=error)
    request = (
        ContextCancellationReservationRequest(**context_cancellation_reservation_payload(42))
        if action == "reserve"
        else ContextCancellationRequest(token="opaque")
    )
    method = (
        worker.reserve_context_cancellation
        if action == "reserve"
        else worker.cancel_context_cancellation
    )
    headers = build_context_cancellation_headers("secret", action, request.model_dump())
    with pytest.raises(HTTPException) as failure:
        await method(request, _raw_request(headers))
    assert failure.value.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize("send_disconnect", [False, True])
async def test_adapted_response_closes_protected_source_before_first_read_or_failed_send(
    send_disconnect: bool,
) -> None:
    closed = asyncio.Event()

    async def source() -> AsyncGenerator[None, None]:
        try:
            yield None
            await asyncio.Future()
        finally:
            closed.set()

    async def adapted() -> AsyncGenerator[str, None]:
        yield "event: message_start\n\ndata: {}\n\n"

    original = source()
    await original.__anext__()
    stream = OpenAIDisaggServer._close_adapted_context_stream(original, adapted())
    await stream.__anext__()
    if not send_disconnect:
        await stream.aclose()
    else:
        response = ClosingStreamingResponse(stream)

        async def send(message: Message) -> None:
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send)
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failures", "attempts", "accepted"),
    [
        ([asyncio.TimeoutError(), "queued"], 2, True),
        ([aiohttp.ClientConnectionError("lost"), "not_submitted"], 2, True),
        ([aiohttp.ClientResponseError(None, (), status=503), "queued"], 2, True),
        ([aiohttp.ClientResponseError(None, (), status=409)], 1, False),
        ([ValueError("invalid receipt")], 1, False),
        ([asyncio.TimeoutError(), asyncio.TimeoutError()], 2, False),
        (
            [aiohttp.ClientPayloadError("truncated"), aiohttp.ClientPayloadError("truncated")],
            2,
            False,
        ),
    ],
    ids=[
        "timeout_retry",
        "connection_retry",
        "unavailable_retry",
        "rejected",
        "invalid_receipt",
        "exhausted",
        "truncated_replies_exhausted",
    ],
)
async def test_context_cancel_redelivery_is_bounded_and_idempotent(
    failures: list, attempts: int, accepted: bool
) -> None:
    """Duplicate callers share one retry budget and clear identity only on receipt."""
    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(side_effect=failures))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"
    await asyncio.gather(cancellation.cancel(), cancellation.cancel())
    await cancellation.cancel()
    assert client.cancel_context_cancellation.await_count == attempts
    assert all(
        call.args == ("ctx:8000", "opaque")
        for call in client.cancel_context_cancellation.await_args_list
    )
    assert cancellation._token == (None if accepted else "opaque")


@pytest.mark.asyncio
async def test_pending_cancel_delivery_keeps_one_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pending attempt consumes the original budget, including repeat callers."""
    monkeypatch.setattr(_ContextCancellation, "_DELIVERY_BUDGET_SECONDS", 0.05)
    monkeypatch.setattr(_ContextCancellation, "_ATTEMPT_TIMEOUT_SECONDS", 5.0)
    stopped = asyncio.Event()
    release = asyncio.Event()

    async def pending(*args: object) -> str:
        """Remain pending until the real delivery timeout cancels the attempt."""
        try:
            await release.wait()
            return "queued"
        finally:
            stopped.set()

    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(side_effect=pending))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"
    # A separate hang guard detects an unbounded attempt; it is not a release SLO.
    callers = []
    try:
        for _ in range(2):
            task = asyncio.create_task(cancellation.cancel())
            callers.append(task)
            done, pending_tasks = await asyncio.wait([task], timeout=1.0)
            assert task in done and not pending_tasks, "control delivery exceeded its budget"
            task.result()
        assert stopped.is_set()
        assert cancellation._delivery.done()
        assert cancellation._token == "opaque"
        client.cancel_context_cancellation.assert_awaited_once_with("ctx:8000", "opaque")
    finally:
        release.set()
        await asyncio.gather(*callers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_scope", [False, True])
async def test_cancel_callers_wait_for_shared_receipt(cancel_scope: bool) -> None:
    """Both callers await one delivery, even inside a cancelled ASGI scope."""
    started, release = asyncio.Event(), asyncio.Event()

    async def deliver(*args: object) -> str:
        """Expose delivery readiness without completing it."""
        started.set()
        await release.wait()
        return "queued"

    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(side_effect=deliver))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"

    async def cancel() -> None:
        """Mirror level-triggered ASGI cancellation when requested."""
        with anyio.CancelScope() as scope:
            if cancel_scope:
                scope.cancel()
            await cancellation.cancel()

    first = asyncio.create_task(cancel())
    await asyncio.wait_for(started.wait(), 1)
    second = asyncio.create_task(cancel())
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 1)
    assert cancellation._token is None
    client.cancel_context_cancellation.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reserve", "cancel", "submit", "retry_enabled_submit"])
@pytest.mark.parametrize("status", [301, 307, 308])
async def test_signed_context_posts_reject_redirect_without_replay(
    monkeypatch: pytest.MonkeyPatch, action: str, status: int
) -> None:
    """Signed control and submission bodies never follow worker redirects."""
    retry_enabled = action == "retry_enabled_submit"
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "0" if retry_enabled else "1")
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", Mock())
    received = []

    async def redirect(request: web.Request) -> web.Response:
        """Record every request, including any wrongly followed redirect."""
        received.append(request.path)
        return web.Response(status=status, headers={"Location": "/unexpected-target"})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", redirect)
    async with TestServer(app) as http, aiohttp.ClientSession() as session:
        client = OpenAIHttpClient(
            AsyncMock(spec=Router),
            ServerRole.CONTEXT,
            session=session,
            internal_disagg_auth_key="secret",
        )
        server = str(http.make_url("/"))
        with pytest.raises(aiohttp.ClientResponseError) as failure:
            if action == "reserve":
                await client.reserve_context_cancellation(server, 42)
            elif action == "cancel":
                await client.cancel_context_cancellation(server, "opaque")
            else:
                request = CompletionRequest(
                    model="model",
                    prompt=[1, 2],
                    disaggregated_params=DisaggregatedParams(
                        request_type="context_only",
                        disagg_request_id=42,
                        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
                        context_cancellation_token=None if retry_enabled else "opaque",
                    ),
                )
                await client.send_request(request, server=server, req_id=42)
        assert failure.value.status == status
    assert len(received) == 1
    assert received[0] != "/unexpected-target"


@pytest.mark.asyncio
async def test_responses_failure_preserves_exception_and_cancels_once() -> None:
    """The protected wrapper keeps the real guard's single terminal error frame."""
    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(return_value="queued"))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"
    partial = 'event: response.output_text.delta\ndata: {"type":"response.output_text.delta"}\n\n'
    original = RuntimeError("worker stream failed")
    closed = []

    async def source() -> AsyncGenerator[str, None]:
        try:
            yield partial
            raise original
        finally:
            closed.append(True)

    stream = cancellation.stream(source(), True)
    await stream.__anext__()
    frames = []
    with pytest.raises(RuntimeError) as failure:
        async for frame in stream:
            frames.append(frame)
    assert failure.value is original
    assert frames[0] == partial
    assert len(frames) == 2 and frames[1].startswith(b"event: error\n")
    assert closed == [True]
    client.cancel_context_cancellation.assert_awaited_once_with("ctx:8000", "opaque")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schedule_style", DisaggScheduleStyle.GENERATION_FIRST),
        ("disagg_request_id", None),
        ("disagg_request_id", -1),
    ],
    ids=["gen_first", "missing_id", "negative_id"],
)
def test_submission_identity_rejects_unqualified_fields(field: str, value: object) -> None:
    """Each context-first identity guard rejects before signing a submission."""
    request = CompletionRequest(
        model="model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(
            request_type="context_only",
            disagg_request_id=42,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            context_cancellation_token="opaque",
        ),
    )
    setattr(request.disaggregated_params, field, value)
    with pytest.raises(ValueError, match="context-first"):
        context_cancellation_submission_payload(request)


@pytest.mark.parametrize(
    ("prompt", "accepted"),
    [([], False), (["a"], False), ([[1, 2]], False), ("hello", True), ([1, 2, 3], True)],
    ids=["empty", "text_batch", "token_batch", "text", "tokens"],
)
def test_submission_token_covers_exactly_one_prompt(
    prompt: str | list, accepted: bool, context_worker: OpenAIServer
) -> None:
    """One reservation cannot admit several prompts with the same transfer identity."""
    request = CompletionRequest(
        model="model",
        prompt=prompt,
        disaggregated_params=DisaggregatedParams(
            request_type="context_only",
            disagg_request_id=42,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            context_cancellation_token="opaque",
        ),
    )
    headers = build_context_cancellation_headers(
        "secret", "submit", context_cancellation_submission_payload(request)
    )
    if accepted:
        context_worker._validate_internal_disagg_request(request, _raw_request(headers))
    else:
        with pytest.raises(ValueError, match="one prompt"):
            context_worker._validate_internal_disagg_request(request, _raw_request(headers))


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reserve", "cancel"])
async def test_control_rejects_missing_auth_before_worker_call(
    action: str, context_worker: OpenAIServer
) -> None:
    """Both control routes reject missing credentials before RPC."""
    request = (
        ContextCancellationReservationRequest(**context_cancellation_reservation_payload(42))
        if action == "reserve"
        else ContextCancellationRequest(token="opaque")
    )
    executor = context_worker.generator._executor
    callback = Mock()
    setattr(executor, f"{action}_context_cancellation", callback)
    method = getattr(context_worker, f"{action}_context_cancellation")
    with pytest.raises(HTTPException) as failure:
        await method(request, _raw_request({}))
    assert failure.value.status_code == 403
    callback.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(("receipt", "http_status"), [("not_submitted", 202), ("done", 503)])
async def test_cancel_handler_validates_routing_receipt(
    receipt: str, http_status: int, context_worker: OpenAIServer
) -> None:
    """Only the known routing receipt may be acknowledged as accepted."""
    context_worker.generator._executor.cancel_context_cancellation = Mock(return_value=receipt)
    request = ContextCancellationRequest(token="opaque")
    headers = build_context_cancellation_headers("secret", "cancel", request.model_dump())
    if http_status == 202:
        response = await context_worker.cancel_context_cancellation(request, _raw_request(headers))
        assert response.status_code == 202
        assert response.body == b'{"status":"not_submitted"}'
    else:
        with pytest.raises(HTTPException) as failure:
            await context_worker.cancel_context_cancellation(request, _raw_request(headers))
        assert failure.value.status_code == http_status


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["authorization_id", "authorization_expires_at_ms"])
async def test_reserve_authorization_fields_are_signed_before_worker_rpc(
    context_worker: OpenAIServer, field: str
) -> None:
    """A changed nonce or expiry cannot renew a previously signed authorization."""
    payload = context_cancellation_reservation_payload(42)
    headers = build_context_cancellation_headers("secret", "reserve", payload)
    if field == "authorization_id":
        payload[field] = ("1" if payload[field][0] == "0" else "0") + payload[field][1:]
    else:
        payload[field] += 1
    callback = Mock()
    context_worker.generator._executor.reserve_context_cancellation = callback
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=context_worker.app), base_url="http://worker"
    ) as client:
        response = await client.post(
            "/internal/context-cancellation/reserve", json=payload, headers=headers
        )
    assert response.status_code == 403
    callback.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "no_retry"),
    [(ServerRole.GENERATION, "1"), (ServerRole.CONTEXT, "0")],
    ids=["generation_role", "retry_enabled"],
)
async def test_token_client_rejects_unqualified_submission(
    monkeypatch: pytest.MonkeyPatch, role: ServerRole, no_retry: str
) -> None:
    """The real submission path must fail before HTTP for an unsafe client."""
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", no_retry)
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", Mock())
    session = Mock()
    client = OpenAIHttpClient(
        AsyncMock(spec=Router), role, session=session, internal_disagg_auth_key="secret"
    )
    request = CompletionRequest(
        model="model",
        prompt=[1, 2],
        disaggregated_params=DisaggregatedParams(
            request_type="context_only",
            disagg_request_id=42,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            context_cancellation_token="opaque",
        ),
    )
    with pytest.raises(ValueError, match="no-retry context client"):
        await client.send_request(request, server="ctx:8000", req_id=42)
    session.post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "body"),
    [
        ("reserve", {}),
        ("cancel", {"status": "done"}),
        ("reserve", []),
        ("reserve", None),
        ("cancel", []),
        ("cancel", None),
    ],
)
async def test_client_rejects_malformed_control_response(
    monkeypatch: pytest.MonkeyPatch, action: str, body: object
) -> None:
    """Malformed control responses cannot establish a token or accepted cancellation."""
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", Mock())
    client = OpenAIHttpClient(
        Mock(spec=Router), ServerRole.CONTEXT, session=Mock(), internal_disagg_auth_key="secret"
    )
    response = SimpleNamespace(status=200, json=AsyncMock(return_value=body))
    response_context = AsyncMock()
    response_context.__aenter__.return_value = response
    client._session.post.return_value = response_context
    with pytest.raises(ValueError, match="reservation|status|non-object"):
        if action == "reserve":
            await client.reserve_context_cancellation("ctx:8000", 42)
        else:
            await client.cancel_context_cancellation("ctx:8000", "opaque")


@pytest.mark.asyncio
@pytest.mark.parametrize("authenticated", [True, False])
async def test_real_client_submission_auth_reaches_executor_input(
    monkeypatch: pytest.MonkeyPatch, context_worker: OpenAIServer, authenticated: bool
) -> None:
    """Real client serialization and worker handler retain the executor-bound token."""
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", Mock())
    worker = context_worker
    worker._extract_metrics = AsyncMock()
    worker.await_disconnected = AsyncMock()

    async def submit(incoming: web.Request) -> web.Response:
        """Forward the client bytes and protocol headers through the actual worker route."""
        headers = {
            key: value
            for key, value in incoming.headers.items()
            if authenticated or key.lower() != CONTEXT_CANCELLATION_AUTH_HEADER
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
        ) as asgi:
            result = await asgi.post(incoming.path, content=await incoming.read(), headers=headers)
        if not authenticated:
            assert result.status_code == 400
            assert "Invalid context cancellation authentication" in result.text
        return web.Response(
            body=result.content, status=result.status_code, content_type="application/json"
        )

    app = web.Application()
    app.router.add_post("/v1/completions", submit)
    request = CompletionRequest(
        model="model",
        prompt=[1, 2],
        disaggregated_params=DisaggregatedParams(
            request_type="context_only",
            disagg_request_id=42,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            context_cancellation_token="opaque",
        ),
    )
    async with TestServer(app) as http, aiohttp.ClientSession() as session:
        client = OpenAIHttpClient(
            AsyncMock(spec=Router),
            ServerRole.CONTEXT,
            session=session,
            internal_disagg_auth_key="secret",
        )
        if authenticated:
            await client.send_request(request, server=str(http.make_url("/")), req_id=42)
        else:
            with pytest.raises(aiohttp.ClientResponseError) as failure:
                await client.send_request(request, server=str(http.make_url("/")), req_id=42)
            assert failure.value.status == 400
            assert "Invalid context cancellation authentication" in failure.value.message
    assert len(worker.generator.generate_calls) == int(authenticated)
    if authenticated:
        params = worker.generator.generate_calls[0].kwargs["disaggregated_params"]
        assert params.context_cancellation_token == "opaque"
        assert params.disagg_request_id == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [True, False, 1, None])
async def test_server_info_reports_actual_context_cancellation_capability(
    context_worker: OpenAIServer, supported: object
) -> None:
    """Server metadata reports the live executor query, not an environment guess."""
    context_worker.generator.args.cache_transceiver_config = None
    context_worker.generator._disaggregated_params = {}
    context_worker.generator._startup_metrics = {}
    context_worker.generator._executor.get_kv_cache_capacity = Mock(return_value={})
    capability = Mock(return_value=supported)
    context_worker.generator._executor.supports_logical_context_cancellation = capability
    content = json.loads((await context_worker.get_server_info()).body)
    assert content["supports_logical_context_cancellation"] is (supported is True)
    capability.assert_called_once_with()


@pytest.mark.asyncio
async def test_server_info_reports_legacy_executor_without_capability(
    context_worker: OpenAIServer,
) -> None:
    """Missing executor support remains an explicit false capability."""
    context_worker.generator.args.cache_transceiver_config = None
    context_worker.generator._disaggregated_params = {}
    context_worker.generator._startup_metrics = {}
    context_worker.generator._executor.get_kv_cache_capacity = Mock(return_value={})
    content = json.loads((await context_worker.get_server_info()).body)
    assert content["supports_logical_context_cancellation"] is False


@pytest.mark.asyncio
async def test_server_info_survives_capability_rpc_timeout_and_recovers(
    context_worker: OpenAIServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protected startup retries the real handler's unavailable probe, not a false capability."""
    monkeypatch.setenv("TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT", "1")
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    context_worker.generator.args.cache_transceiver_config = None
    context_worker.generator._disaggregated_params = {}
    context_worker.generator._startup_metrics = {}
    context_worker.generator._executor.get_kv_cache_capacity = Mock(return_value={})
    capability = Mock(side_effect=[RPCTimeout("unavailable"), True])
    context_worker.generator._executor.supports_logical_context_cancellation = capability
    config = DisaggServerConfig(
        server_configs=[CtxGenServerConfig(type="ctx", hostname="ctx", port=8000)],
        schedule_style="context_first",
        internal_request_auth_key="secret",
    )
    owner = DisaggCoordinatorService(
        config, Mock(), health_check_interval_secs=0, server_start_timeout_secs=1
    )
    replies = []

    async def server_info(server: str, timeout: float | None) -> dict:
        """Pass the real worker's JSON metadata through the protected preparation gate."""
        assert server == "ctx:8000"
        response = await context_worker.get_server_info()
        assert response.status_code == 200
        content = json.loads(response.body)
        replies.append(content)
        return content

    owner.ctx_router._fetch_server_info = AsyncMock(side_effect=server_info)
    try:
        await owner._prepare_context_servers()
        first, second = replies
        assert first["supports_logical_context_cancellation"] is None
        assert second["supports_logical_context_cancellation"] is True
        assert first["disaggregated_params"] == second["disaggregated_params"] == {}
        assert owner.ctx_router.prepared_servers == {"ctx:8000"}
        assert capability.call_count == 2
    finally:
        await owner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reserve", "cancel"])
@pytest.mark.parametrize("attached_frontend", [False, True])
async def test_control_with_missing_real_proxy_rpc_returns_unavailable(
    context_worker: OpenAIServer, action: str, attached_frontend: bool
) -> None:
    """The real launcher/attached proxy distinguishes missing RPC from rejection."""
    from tensorrt_llm.executor.proxy import GenerationExecutorFrontendProxy, GenerationExecutorProxy

    proxy_type = GenerationExecutorFrontendProxy if attached_frontend else GenerationExecutorProxy
    method = getattr(proxy_type, f"{action}_context_cancellation")
    proxy = SimpleNamespace(rpc_client=None)
    setattr(
        context_worker.generator._executor,
        f"{action}_context_cancellation",
        lambda *args: method(proxy, *args),
    )
    request = (
        ContextCancellationReservationRequest(**context_cancellation_reservation_payload(42))
        if action == "reserve"
        else ContextCancellationRequest(token="opaque")
    )
    headers = build_context_cancellation_headers("secret", action, request.model_dump())
    handler = getattr(context_worker, f"{action}_context_cancellation")
    with pytest.raises(HTTPException) as failure:
        await handler(request, _raw_request(headers))
    assert failure.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connection", "timeout", "rpc_unavailable"])
async def test_reserve_redelivery_reuses_recorded_authorization_after_lost_reply(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """A lost reserve reply uses one ledger slot and never submits model work."""
    from tensorrt_llm.executor.context_cancellation import ContextCancellationReservations

    owner = ContextCancellationReservations()
    deliveries = []
    tokens = []
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", Mock())
    client = OpenAIHttpClient(
        Mock(spec=Router), ServerRole.CONTEXT, session=Mock(), internal_disagg_auth_key="secret"
    )

    async def deliver(server: str, action: str, payload: dict) -> dict:
        """Lose only the first reply after the real worker records admission."""
        deliveries.append((server, action, dict(payload)))
        token = owner.reserve(
            payload["disagg_request_id"],
            payload["authorization_id"],
            payload["authorization_expires_at_ms"],
        )
        tokens.append(token)
        if len(deliveries) == 1:
            if failure == "timeout":
                raise asyncio.TimeoutError
            if failure == "rpc_unavailable":
                raise aiohttp.ClientResponseError(None, (), status=503)
            raise aiohttp.ClientConnectionError("reply lost")
        return {"token": token}

    client._context_cancellation_control = AsyncMock(side_effect=deliver)
    token = await client.reserve_context_cancellation("ctx:8000", 42)
    assert deliveries[0] == deliveries[1]
    assert tokens == [token, token]
    assert list(owner._pending) == [token]


@pytest.mark.asyncio
async def test_truncated_control_replies_redeliver_only_the_original_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial 200 response can hide an accepted reserve or cancel."""
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", Mock())
    deliveries = []

    async def control(request: web.Request) -> web.Response:
        action = request.match_info["action"]
        deliveries.append((action, await request.json()))
        if len(deliveries) in (1, 3):
            response = web.Response(
                body=b"{", content_type="application/json", headers={"Content-Length": "32"}
            )
            response.force_close()
            return response
        return web.json_response(
            {"token": "opaque"} if action == "reserve" else {"status": "queued"}
        )

    app = web.Application()
    app.router.add_post("/internal/context-cancellation/{action:reserve|cancel}", control)
    async with TestServer(app) as http, aiohttp.ClientSession() as session:
        client = OpenAIHttpClient(
            Mock(spec=Router),
            ServerRole.CONTEXT,
            session=session,
            internal_disagg_auth_key="secret",
        )
        cancellation = _ContextCancellation(client)
        assert await cancellation.reserve(str(http.make_url("/")), 42) == "opaque"
        await cancellation.cancel()

    assert [action for action, _ in deliveries] == ["reserve", "reserve", "cancel", "cancel"]
    assert deliveries[0] == deliveries[1]
    assert deliveries[2][1] == deliveries[3][1] == {"token": "opaque"}
    assert cancellation._token is None


@pytest.mark.asyncio
async def test_prelude_cancellation_during_successful_stream_watcher_join() -> None:
    """Caller cancellation during watcher cleanup closes the unhanded stream."""
    watching, joining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_probe = asyncio.Event()
    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(return_value="queued"))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"
    streams = []

    async def source() -> AsyncGenerator[bytes, None]:
        """No GEN request may be dispatched from an unconsumed response."""
        pytest.fail("GEN must not start")
        yield b""

    async def entry(*args: object) -> AsyncGenerator:
        """Prime the real cleanup wrapper before the prelude returns it."""
        await watching.wait()
        stream = cancellation.stream(source(), False)
        await stream.__anext__()
        streams.append(stream)
        return stream

    async def is_disconnected() -> bool:
        """Hold watcher teardown after the response has become ready."""
        watching.set()
        try:
            await release_probe.wait()
        finally:
            joining.set()
            await release.wait()
        return False

    task = asyncio.create_task(
        OpenAIDisaggServer._await_context_request(
            entry,
            CompletionRequest(model="model", prompt="hello"),
            None,
            SimpleNamespace(is_disconnected=is_disconnected),
        )
    )
    try:
        await asyncio.wait_for(joining.wait(), 1)
        assert not release_probe.is_set()
        assert streams and not task.done()
        task.cancel()
        await asyncio.sleep(0)
    finally:
        release_probe.set()
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        closed_before_cleanup = [stream.ag_frame is None for stream in streams]
        for stream in streams:
            await stream.aclose()
    assert task.cancelled()
    assert closed_before_cleanup == [True]
    client.cancel_context_cancellation.assert_awaited_once_with("ctx:8000", "opaque")


@pytest.mark.parametrize("rank,supported", [(0, True), (0, False), (0, 1), (1, True)])
def test_worker_capability_uses_running_engine_and_rank(rank: int, supported: object) -> None:
    """Only the qualified rank-zero executor advertises independent retirement."""
    from tensorrt_llm.executor.base_worker import BaseWorker

    engine = SimpleNamespace(supports_logical_context_cancellation=Mock(return_value=supported))
    worker = SimpleNamespace(engine=engine, rank=rank)
    assert BaseWorker.supports_logical_context_cancellation(worker) is (
        rank == 0 and supported is True
    )
    worker.engine = SimpleNamespace()
    assert BaseWorker.supports_logical_context_cancellation(worker) is False


@pytest.mark.parametrize("attached_frontend", [False, True])
def test_executor_proxy_capability_queries_shared_owner(attached_frontend: bool) -> None:
    """Launcher and attached frontends query the same authenticated worker owner."""
    from tensorrt_llm.executor.proxy import GenerationExecutorFrontendProxy, GenerationExecutorProxy

    proxy_type = GenerationExecutorFrontendProxy if attached_frontend else GenerationExecutorProxy
    rpc = Mock()
    rpc.supports_logical_context_cancellation.return_value.remote.return_value = True
    proxy = SimpleNamespace(rpc_client=rpc)
    assert proxy_type.supports_logical_context_cancellation(proxy) is True
    rpc.supports_logical_context_cancellation.return_value.remote.assert_called_once_with(
        timeout=5.0
    )
    proxy.rpc_client = None
    assert proxy_type.supports_logical_context_cancellation(proxy) is False
