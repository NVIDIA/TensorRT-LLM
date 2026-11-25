# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response delivery and cleanup while a regular RPC send is in flight."""

import asyncio
import gc
import threading
from types import SimpleNamespace
from typing import NoReturn
from unittest.mock import AsyncMock

import pytest

from tensorrt_llm.executor.rpc.rpc_client import RPCClient
from tensorrt_llm.executor.rpc.rpc_common import (
    RPCCancelled,
    RPCError,
    RPCParams,
    RPCRequest,
    RPCResponse,
    RPCTimeout,
    get_unique_ipc_addr,
)
from tensorrt_llm.executor.rpc.rpc_server import RPCServer

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def client() -> RPCClient:
    rpc_client = RPCClient.__new__(RPCClient)
    rpc_client._closed = False
    rpc_client._server_stopped = False
    rpc_client._timeout = None
    rpc_client._call_lock = threading.Lock()
    rpc_client._pending_futures = {}
    rpc_client._sending_tasks = set()
    rpc_client._client_socket = SimpleNamespace(put_async=AsyncMock())
    return rpc_client


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_error", [False, True])
async def test_reply_before_send_returns(client: RPCClient, reply_error: bool) -> None:
    error = RPCError("remote failure") if reply_error else None

    async def send(request: RPCRequest) -> None:
        client._handle_regular_response(RPCResponse(request.request_id, result=42, error=error))
        # Let the response callback run before the transport send completes.
        await asyncio.sleep(0)

    client._client_socket.put_async.side_effect = send
    call = client._call_async("method")
    if reply_error:
        with pytest.raises(RPCError, match="remote failure"):
            await asyncio.wait_for(call, timeout=1)
    else:
        assert await asyncio.wait_for(call, timeout=1) == 42
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_send_failure_removes_registration(client: RPCClient) -> None:
    registered = []

    async def send(request: RPCRequest) -> None:
        registered.append(client._pending_futures.get(request.request_id))
        raise ConnectionError("send failed")

    client._client_socket.put_async.side_effect = send
    with pytest.raises(ConnectionError, match="send failed"):
        await client._call_async("method")
    assert registered[0] is not None
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_cancellation_during_send_removes_registration(client: RPCClient) -> None:
    sending = asyncio.Event()
    registered = []

    async def send(request: RPCRequest) -> None:
        registered.append(client._pending_futures.get(request.request_id))
        sending.set()
        await asyncio.Event().wait()

    client._client_socket.put_async.side_effect = send
    task = asyncio.create_task(client._call_async("method"))
    try:
        await asyncio.wait_for(sending.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert registered[0] is not None
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_fire_and_forget_does_not_register_response(client: RPCClient) -> None:
    async def send(request: RPCRequest) -> None:
        assert not request.need_response
        assert not client._pending_futures

    client._client_socket.put_async.side_effect = send
    assert await client._call_async("method", __rpc_params=RPCParams(need_response=False)) is None
    client._client_socket.put_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_closed_client_does_not_send(client: RPCClient) -> None:
    client._closed = True
    with pytest.raises(RPCCancelled):
        await client._call_async("method")
    client._client_socket.put_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_transport_reply_during_send(monkeypatch: pytest.MonkeyPatch) -> None:
    server = RPCServer(SimpleNamespace(answer=lambda: 42))
    server.bind(get_unique_ipc_addr())
    server.start()
    try:
        with RPCClient(server.address, hmac_key=server.hmac_key) as rpc_client:
            received = threading.Event()
            send = rpc_client._client_socket.put_async
            handle = rpc_client._handle_regular_response

            def handle_response(response: RPCResponse) -> None:
                handle(response)
                received.set()

            async def send_until_reply(request: RPCRequest) -> None:
                await send(request)
                assert await asyncio.to_thread(received.wait, 2)

            monkeypatch.setattr(rpc_client, "_handle_regular_response", handle_response)
            monkeypatch.setattr(rpc_client._client_socket, "put_async", send_until_reply)
            result = asyncio.run_coroutine_threadsafe(
                rpc_client._call_async("answer"), rpc_client._loop
            )
            assert await asyncio.wait_for(asyncio.wrap_future(result), timeout=3) == 42
            assert not rpc_client._pending_futures
    finally:
        server.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("send_error", [ConnectionError, TimeoutError, RPCCancelled])
async def test_send_error_propagates_unchanged(
    client: RPCClient, send_error: type[Exception]
) -> None:
    error = send_error("send failed")
    client._client_socket.put_async.side_effect = error
    with pytest.raises(send_error) as raised:
        await client._call_async("method")
    assert raised.value is error
    assert not client._server_stopped
    assert not client._pending_futures
    assert not client._sending_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_error", [False, True])
@pytest.mark.parametrize("cancel_send", [False, True])
async def test_early_send_exit_settles_response_future(
    client: RPCClient, reply_error: bool, cancel_send: bool
) -> None:
    loop = asyncio.get_running_loop()
    errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    def handle_error(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        errors.append(context)

    async def send(request: RPCRequest) -> None:
        if reply_error:
            client._handle_regular_response(
                RPCResponse(request.request_id, result=None, error=RPCError("early remote error"))
            )
            await asyncio.sleep(0)
        if cancel_send:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)
        raise ConnectionError("send failed")

    loop.set_exception_handler(handle_error)
    client._client_socket.put_async.side_effect = send
    try:
        with pytest.raises(asyncio.CancelledError if cancel_send else ConnectionError):
            await client._call_async("method")
        # Force disposal of failed futures while the exception handler is installed.
        gc.collect()
        await asyncio.sleep(0)
        assert not errors
        assert not client._pending_futures
        assert not client._sending_tasks
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_cancelled_send_cancels_pending_response(client: RPCClient) -> None:
    registered: list[asyncio.Future] = []

    async def send(request: RPCRequest) -> None:
        registered.append(client._pending_futures[request.request_id][0])
        raise asyncio.CancelledError

    client._client_socket.put_async.side_effect = send
    with pytest.raises(asyncio.CancelledError):
        await client._call_async("method")
    assert registered[0].cancelled()
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_fire_and_forget_checks_close_after_send(client: RPCClient) -> None:
    async def send(request: RPCRequest) -> None:
        client._closed = True

    client._client_socket.put_async.side_effect = send
    with pytest.raises(RPCCancelled, match="RPC client closed"):
        await client._call_async("method", __rpc_params=RPCParams(need_response=False))
    assert not client._pending_futures
    assert not client._sending_tasks


@pytest.mark.asyncio
async def test_response_timeout_is_translated(client: RPCClient) -> None:
    with pytest.raises(RPCTimeout, match="timed out"):
        await client._call_async("method", __rpc_params=RPCParams(timeout=0.01))
    assert not client._pending_futures


@pytest.mark.asyncio
@pytest.mark.parametrize("need_response", [False, True])
async def test_close_drains_blocked_send(
    monkeypatch: pytest.MonkeyPatch, need_response: bool
) -> None:
    server = RPCServer(SimpleNamespace())
    server.bind(get_unique_ipc_addr())
    server.start()
    rpc_client = RPCClient(server.address, hmac_key=server.hmac_key)
    sending = threading.Event()
    cleaned_up = threading.Event()

    async def send(request: RPCRequest) -> None:
        assert asyncio.get_running_loop() is rpc_client._loop
        sending.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    monkeypatch.setattr(rpc_client._client_socket, "put_async", send)
    result = asyncio.run_coroutine_threadsafe(
        rpc_client._call_async("method", __rpc_params=RPCParams(need_response=need_response)),
        rpc_client._loop,
    )
    try:
        assert await asyncio.to_thread(sending.wait, 2)
        await asyncio.to_thread(rpc_client.close)
        assert cleaned_up.is_set()
        assert not rpc_client._pending_futures
        assert not rpc_client._sending_tasks
        assert not rpc_client._loop.is_running()
        with pytest.raises(RPCCancelled, match="RPC client closed"):
            await asyncio.wait_for(asyncio.wrap_future(result), timeout=1)
    finally:
        rpc_client.close()
        server.shutdown()


@pytest.mark.asyncio
async def test_pickle_timeout_is_not_a_response_timeout() -> None:
    error = TimeoutError("pickle reducer failed")

    class Unserializable:
        def __reduce__(self) -> NoReturn:
            raise error

    with RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key") as rpc_client:

        async def call() -> None:
            with pytest.raises(TimeoutError) as raised:
                await rpc_client._call_async("method", Unserializable())
            assert raised.value is error
            assert not rpc_client._pending_futures
            assert not rpc_client._sending_tasks

        result = asyncio.run_coroutine_threadsafe(call(), rpc_client._loop)
        await asyncio.wait_for(asyncio.wrap_future(result), timeout=2)
