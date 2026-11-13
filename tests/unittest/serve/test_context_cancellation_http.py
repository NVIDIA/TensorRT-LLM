# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import HTTPException, Request
from starlette.requests import ClientDisconnect
from starlette.types import Message
from test_common.serve.app_factory import AggConfig, ServeTestEnv

from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.executor.rpc import RPCError, RPCTimeout
from tensorrt_llm.llmapi.disagg_utils import DisaggServerConfig, ServerRole
from tensorrt_llm.serve.disagg_auth import (
    build_context_cancellation_headers,
    context_cancellation_submission_payload,
    validate_context_cancellation_auth,
)
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
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
        action = request.match_info["action"]
        model = (
            ContextCancellationReservationRequest
            if action == "reserve"
            else ContextCancellationRequest
        )
        method = (
            worker.reserve_context_cancellation
            if action == "reserve"
            else worker.cancel_context_cancellation
        )
        response = await method(model(**await request.json()), _raw_request(dict(request.headers)))
        return web.Response(
            body=response.body, status=response.status_code, content_type="application/json"
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
    executor.reserve_context_cancellation.assert_called_once_with(42)
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
    task = asyncio.create_task(
        service._send_disagg_request_ctx_first(CompletionRequest(model="model", prompt="hello"))
    )
    await asyncio.wait_for(paused.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    service._gen_router.finish_request.assert_awaited_once()
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
@pytest.mark.parametrize("enabled", [False, True])
async def test_healthy_handoff_and_legacy_do_not_cancel(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    service = _service(monkeypatch, enabled=enabled)
    result = await service._send_disagg_request_ctx_first(
        CompletionRequest(model="model", prompt="hello")
    )
    assert result.choices[0].finish_reason == "stop"
    service._ctx_client.cancel_context_cancellation.assert_not_awaited()
    assert service._ctx_client.reserve_context_cancellation.await_count == int(enabled)
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
    with pytest.raises(asyncio.CancelledError):
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

    async def is_disconnected() -> bool:
        if probing.is_set():
            return disconnect.is_set()
        probing.set()
        try:
            await asyncio.Future()
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
        if outcome == "cancel":
            assert task.cancelled()
        elif outcome == "error":
            with pytest.raises(RuntimeError, match="service failed"):
                task.result()
        else:
            assert task.result() is response
    finally:
        # Release the probe even if the bounded completion assertion fails.
        disconnect.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_event", [False, True])
async def test_stream_completion_preserves_success_but_error_cancels_ctx(error_event: bool) -> None:
    client = SimpleNamespace(cancel_context_cancellation=AsyncMock(return_value="queued"))
    cancellation = _ContextCancellation(client)
    cancellation._server, cancellation._token = "ctx:8000", "opaque"
    frame = b'data: {"error":{"message":"failed"}}\n\n' if error_event else b"data: [DONE]\n\n"

    async def source() -> AsyncGenerator[bytes, None]:
        yield frame

    stream = cancellation.stream(source(), False)
    await stream.__anext__()
    assert [chunk async for chunk in stream] == [frame]
    assert client.cancel_context_cancellation.await_count == int(error_event)


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
    with pytest.raises(asyncio.CancelledError):
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
    service._ctx_client.cancel_context_cancellation.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RPCError("worker failed"), RPCTimeout("unknown outcome")])
@pytest.mark.parametrize("action", ["reserve", "cancel"])
async def test_control_rpc_error_is_unavailable_not_a_cancel_receipt(
    error: RPCError, action: str, context_worker: OpenAIServer
) -> None:
    worker = context_worker
    worker.generator._executor.reserve_context_cancellation = Mock(side_effect=error)
    worker.generator._executor.cancel_context_cancellation = Mock(side_effect=error)
    request = (
        ContextCancellationReservationRequest(disagg_request_id=42)
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
    assert failure.value.status_code == 503


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
