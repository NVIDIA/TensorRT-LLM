# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response delivery and cleanup while a regular RPC send is in flight."""

import asyncio
import concurrent.futures
import gc
import threading
import time
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import NoReturn
from unittest.mock import AsyncMock, MagicMock, Mock

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
    rpc_client._closing = False
    rpc_client._loop = None
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
        assert rpc_client._loop.is_closed()
        assert not rpc_client._loop_thread.is_alive()
        assert not asyncio.all_tasks(rpc_client._loop)
        with pytest.raises(RPCCancelled, match="RPC client closed"):
            await asyncio.wait_for(asyncio.wrap_future(result), timeout=1)
    finally:
        rpc_client.close()
        server.shutdown()


def test_close_releases_only_owned_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    rpc_client = RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key")
    socket_threads: list[threading.Thread] = []
    close_socket = rpc_client._client_socket.close

    def record_close() -> None:
        socket_threads.append(threading.current_thread())
        close_socket()

    monkeypatch.setattr(rpc_client._client_socket, "close", record_close)
    try:
        assert not rpc_client._executor._threads
        rpc_client.close()
        assert rpc_client._closed
        assert not rpc_client._loop_thread.is_alive()
        assert rpc_client._loop.is_closed()
        assert not asyncio.all_tasks(rpc_client._loop)
        assert socket_threads == [rpc_client._loop_thread]
        assert not rpc_client._executor._threads
        rpc_client.close()
        assert len(socket_threads) == 1
    finally:
        rpc_client.close()


@pytest.mark.parametrize("failure_site", ["socket", "reader"])
def test_initialization_failure_cleans_started_loop(
    monkeypatch: pytest.MonkeyPatch, failure_site: str
) -> None:
    error = RuntimeError("startup failed")
    captured: list[RPCClient] = []
    ensure_loop = RPCClient._ensure_event_loop
    start_reader = RPCClient._start_response_reader_eagerly

    def fail() -> NoReturn:
        raise error

    def start_and_inject(rpc_client: RPCClient) -> None:
        ensure_loop(rpc_client)
        captured.append(rpc_client)
        if failure_site == "socket":
            monkeypatch.setattr(rpc_client._client_socket, "setup_lazily", fail)
        else:

            def start_then_fail() -> NoReturn:
                start_reader(rpc_client)
                raise error

            monkeypatch.setattr(rpc_client, "_start_response_reader_eagerly", start_then_fail)

    monkeypatch.setattr(RPCClient, "_ensure_event_loop", start_and_inject)
    try:
        with pytest.raises(RuntimeError) as raised:
            RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key")
        assert raised.value is error
        assert len(captured) == 1
        assert captured[0]._closed
        assert captured[0]._loop.is_closed()
        assert not captured[0]._loop_thread.is_alive()
        assert not captured[0]._executor._threads
    finally:
        for rpc_client in captured:
            rpc_client.close()


def test_cleanup_error_surfaces_and_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    rpc_client = RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key")
    close_socket = rpc_client._client_socket.close
    attempts: list[threading.Thread] = []
    error = ConnectionError("socket close failed")

    def fail_once() -> None:
        attempts.append(threading.current_thread())
        if len(attempts) == 1:
            raise error
        close_socket()

    monkeypatch.setattr(rpc_client._client_socket, "close", fail_once)
    try:
        with pytest.raises(ConnectionError) as raised:
            rpc_client.close()
        assert raised.value is error
        assert not rpc_client._closed
        assert rpc_client._closing
        rpc_client.close()
        assert rpc_client._closed
        assert rpc_client._loop.is_closed()
        assert not rpc_client._loop_thread.is_alive()
        assert attempts == [rpc_client._loop_thread, rpc_client._loop_thread]
    finally:
        rpc_client.close()


def test_concurrent_close_confirms_one_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    rpc_client = RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key")
    start = threading.Barrier(3)
    close_socket = rpc_client._client_socket.close
    socket_threads: list[threading.Thread] = []

    def record_close() -> None:
        socket_threads.append(threading.current_thread())
        close_socket()

    def close_client() -> None:
        start.wait(timeout=3)
        rpc_client.close()
        assert not rpc_client._loop_thread.is_alive()
        assert rpc_client._loop.is_closed()

    monkeypatch.setattr(rpc_client._client_socket, "close", record_close)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as callers:
            closing = [callers.submit(close_client) for _ in range(2)]
            start.wait(timeout=3)
            for future in closing:
                future.result(timeout=3)
        assert rpc_client._closed
        assert socket_threads == [rpc_client._loop_thread]
    finally:
        rpc_client.close()


@pytest.mark.parametrize("blocked_owner", ["loop", "pool"])
def test_close_timeout_is_bounded_and_retriable(
    monkeypatch: pytest.MonkeyPatch, blocked_owner: str
) -> None:
    monkeypatch.setattr(RPCClient, "_CLOSE_TIMEOUT_SECONDS", 0.05, raising=False)
    rpc_client = RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key", num_workers=1)
    entered, release = threading.Event(), threading.Event()

    def block_owner() -> None:
        entered.set()
        assert release.wait(10)

    if blocked_owner == "loop":
        rpc_client._loop.call_soon_threadsafe(block_owner)
    else:
        rpc_client._executor.submit(block_owner)
    try:
        assert entered.wait(2)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="still closing"):
            rpc_client.close()
        assert time.monotonic() - started < 1
        assert not rpc_client._closed
        assert rpc_client._closing
        with pytest.raises(RPCCancelled, match="RPC client closed"):
            rpc_client.method().remote()
        with pytest.raises(RPCCancelled, match="RPC client closed"):
            rpc_client.method().remote_future()
        with pytest.raises(TimeoutError, match="still closing"):
            rpc_client.close()
        assert not rpc_client._closed
    finally:
        release.set()
        monkeypatch.setattr(RPCClient, "_CLOSE_TIMEOUT_SECONDS", 3.0)
        rpc_client.close()
    assert rpc_client._closed
    assert rpc_client._loop.is_closed()
    assert not rpc_client._loop_thread.is_alive()
    assert all(not worker.is_alive() for worker in rpc_client._executor._threads)


@pytest.mark.parametrize("callback_owner", ["loop", "pool"])
def test_close_from_owned_callback_requires_external_confirmation(callback_owner: str) -> None:
    rpc_client = RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key")
    completed = threading.Event()
    errors: list[Exception] = []

    def close_in_callback() -> None:
        try:
            rpc_client.close()
        except RuntimeError as error:
            errors.append(error)
        finally:
            completed.set()

    try:
        if callback_owner == "loop":
            rpc_client._loop.call_soon_threadsafe(close_in_callback)
        else:
            rpc_client._executor.submit(close_in_callback)
        assert completed.wait(2)
        assert len(errors) == 1
        assert "external thread" in str(errors[0])
        assert not rpc_client._closed
        rpc_client.close()
        assert rpc_client._closed
        assert rpc_client._loop.is_closed()
        assert not rpc_client._loop_thread.is_alive()
        assert all(not worker.is_alive() for worker in rpc_client._executor._threads)
    finally:
        rpc_client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_external_loop_call_and_inline_close(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    rpc_client = RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key")
    caller_loop = asyncio.get_running_loop()
    errors: list[dict[str, object]] = []
    previous_handler = caller_loop.get_exception_handler()
    caller_loop.set_exception_handler(lambda loop, context: errors.append(context))
    sending, cleaned = threading.Event(), threading.Event()
    transport_loops: list[asyncio.AbstractEventLoop] = []

    async def blocked_send(request: RPCRequest) -> None:
        transport_loops.append(asyncio.get_running_loop())
        sending.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(rpc_client._client_socket, "put_async", blocked_send)
    stream = rpc_client.method().remote_streaming() if streaming else None
    call = stream.__anext__() if stream is not None else rpc_client.method().remote_async()
    result = asyncio.create_task(call)
    try:
        assert await asyncio.to_thread(sending.wait, 2)
        # close must not wait for callbacks on the caller's currently occupied loop.
        rpc_client.close()
        assert cleaned.is_set()
        assert transport_loops == [rpc_client._loop]
        assert not rpc_client._loop_thread.is_alive()
        assert rpc_client._loop.is_closed()
        assert caller_loop.is_running()
        assert not caller_loop.is_closed()
        with pytest.raises(RPCCancelled, match="RPC client closed"):
            await asyncio.wait_for(result, 1)
        assert not rpc_client._streaming_queues
        gc.collect()
        await asyncio.sleep(0)
        assert not errors
    finally:
        rpc_client.close()
        if not result.done():
            result.cancel()
            await asyncio.gather(result, return_exceptions=True)
        caller_loop.set_exception_handler(previous_handler)


@pytest.mark.parametrize("already_closing", [False, True])
def test_close_during_finalization_does_not_queue_or_wait(
    monkeypatch: pytest.MonkeyPatch, already_closing: bool
) -> None:
    rpc_client = RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key")
    close_future = Mock()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(rpc_client, "_is_finalizing", lambda: True)
            schedule = Mock(side_effect=AssertionError("must not queue cleanup"))
            patch.setattr(asyncio, "run_coroutine_threadsafe", schedule)
            patch.setattr(rpc_client, "_call_lock", MagicMock())
            patch.setattr(rpc_client, "_closing", already_closing)
            patch.setattr(rpc_client, "_close_future", close_future if already_closing else None)
            with pytest.raises(RuntimeError, match="interpreter finalization"):
                rpc_client.close()
            assert not rpc_client._closed
            schedule.assert_not_called()
            close_future.result.assert_not_called()
            assert not rpc_client._call_lock.mock_calls
    finally:
        rpc_client.close()
    assert rpc_client._closed
    assert not rpc_client._loop_thread.is_alive()


def test_already_closed_client_succeeds_during_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rpc_client = RPCClient(get_unique_ipc_addr(), hmac_key=b"test-key")
    rpc_client.close()
    monkeypatch.setattr(rpc_client, "_is_finalizing", lambda: True)
    rpc_client.close()
    assert rpc_client._closed
    assert not rpc_client._loop_thread.is_alive()


@pytest.mark.parametrize("close_state", ["_closing", "_closed"])
def test_sync_call_cancels_when_cleanup_stops_loop_before_completion(
    client: RPCClient, monkeypatch: pytest.MonkeyPatch, close_state: str
) -> None:
    stopped = SimpleNamespace(is_running=lambda: False)
    result = Mock()

    def timeout(*, timeout: float) -> NoReturn:
        # The first wait straddles the transition to closing and a stopped loop.
        setattr(client, close_state, True)
        raise concurrent.futures.TimeoutError

    result.result.side_effect = timeout
    result.done.side_effect = [False, AssertionError("must not poll stopped loop again")]

    def schedule(coroutine: Coroutine, loop: asyncio.AbstractEventLoop) -> Mock:
        coroutine.close()
        return result

    client._loop = stopped
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", schedule)
    with pytest.raises(RPCCancelled, match="RPC client closed"):
        client._call_sync("method")
    result.cancel.assert_called_once()
    result.result.assert_called_once()


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
