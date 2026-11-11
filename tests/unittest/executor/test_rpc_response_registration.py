# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response delivery and cleanup while a regular RPC send is in flight."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tensorrt_llm.executor.rpc.rpc_client import RPCClient
from tensorrt_llm.executor.rpc.rpc_common import (
    RPCCancelled,
    RPCError,
    RPCParams,
    RPCResponse,
    get_unique_ipc_addr,
)
from tensorrt_llm.executor.rpc.rpc_server import RPCServer

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def client():
    rpc_client = RPCClient.__new__(RPCClient)
    rpc_client._closed = False
    rpc_client._server_stopped = False
    rpc_client._timeout = None
    rpc_client._call_lock = threading.Lock()
    rpc_client._pending_futures = {}
    rpc_client._client_socket = SimpleNamespace(put_async=AsyncMock())
    return rpc_client


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_error", [False, True])
async def test_reply_before_send_returns(client, reply_error):
    error = RPCError("remote failure") if reply_error else None

    async def send(request):
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
async def test_send_failure_removes_registration(client):
    registered = []

    async def send(request):
        registered.append(client._pending_futures.get(request.request_id))
        raise ConnectionError("send failed")

    client._client_socket.put_async.side_effect = send
    with pytest.raises(ConnectionError, match="send failed"):
        await client._call_async("method")
    assert registered[0] is not None
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_cancellation_during_send_removes_registration(client):
    sending = asyncio.Event()
    registered = []

    async def send(request):
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
async def test_fire_and_forget_does_not_register_response(client):
    async def send(request):
        assert not request.need_response
        assert not client._pending_futures

    client._client_socket.put_async.side_effect = send
    assert await client._call_async("method", __rpc_params=RPCParams(need_response=False)) is None
    client._client_socket.put_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_closed_client_does_not_send(client):
    client._closed = True
    with pytest.raises(RPCCancelled):
        await client._call_async("method")
    client._client_socket.put_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_transport_reply_during_send(monkeypatch):
    server = RPCServer(SimpleNamespace(answer=lambda: 42))
    server.bind(get_unique_ipc_addr())
    server.start()
    try:
        with RPCClient(server.address, hmac_key=server.hmac_key) as rpc_client:
            received = threading.Event()
            send = rpc_client._client_socket.put_async
            handle = rpc_client._handle_regular_response

            def handle_response(response):
                handle(response)
                received.set()

            async def send_until_reply(request):
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
