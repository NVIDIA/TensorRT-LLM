# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary async RPCs must share the response reader's ZeroMQ socket thread."""

import asyncio
import concurrent.futures
import gc
import threading
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.rpc import RPCCancelled, RPCClient, RPCError, RPCServer, RPCTimeout
from tensorrt_llm.executor.rpc import rpc_client as rpc_client_module
from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr

pytestmark = pytest.mark.cpu_only


class _App:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.saved = None

    def echo(self, value):
        return value

    def fail(self):
        raise ValueError("remote failure")

    def save(self, value):
        self.saved = value

    async def hold(self):
        self.started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        return "released"


@pytest.fixture
def rpc(monkeypatch):
    monkeypatch.setenv("TLLM_LLMAPI_ZMQ_DEBUG", "1")
    app = _App()
    server = RPCServer(app)
    server.bind(get_unique_ipc_addr())
    server.start()
    try:
        with RPCClient(server.address, hmac_key=server.hmac_key) as client:
            # Establish the socket's owner using the synchronous API, which
            # already sends and receives on the client's dedicated loop.
            assert client.echo("ready").remote() == "ready"
            try:
                yield client, app
            finally:
                app.release.set()
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_async_unary_result_and_remote_error(rpc):
    client, _ = rpc
    assert await asyncio.wait_for(client.echo(42).remote_async(), 2) == 42
    with pytest.raises(RPCError, match="remote failure"):
        await asyncio.wait_for(client.fail().remote_async(), 2)
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_async_fire_and_forget(rpc):
    client, app = rpc
    assert await client.save(42).remote_async(need_response=False) is None
    assert await client.echo("barrier").remote_async() == "barrier"
    assert app.saved == 42
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_async_timeout(rpc):
    client, _ = rpc
    with pytest.raises(RPCTimeout):
        await asyncio.wait_for(client.hold().remote_async(timeout=0.05), 2)
    assert not client._pending_futures


async def _owner_barrier(client):
    async def settle():
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(settle(), client._loop))


@pytest.mark.asyncio
async def test_async_cancellation_cleans_up_on_owner_loop(rpc):
    client, app = rpc
    task = asyncio.create_task(client.hold().remote_async())
    try:
        assert await asyncio.to_thread(app.started.wait, 2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        app.release.set()
    assert task.cancelled()

    await _owner_barrier(client)
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_client_close_settles_async_caller(rpc):
    client, app = rpc
    task = asyncio.create_task(client.hold().remote_async())
    try:
        assert await asyncio.to_thread(app.started.wait, 2)
        client.close()
        with pytest.raises(RPCCancelled):
            await asyncio.wait_for(task, 2)
    finally:
        app.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_closed_client_rejects_async_call(rpc):
    client, _ = rpc
    client.close()
    with pytest.raises(RPCCancelled):
        await asyncio.wait_for(client.echo(42).remote_async(), 2)


@pytest.mark.asyncio
async def test_cancellation_racing_remote_error_retrieves_future_exception(rpc, monkeypatch):
    client, _ = rpc
    loop = asyncio.get_running_loop()
    errors = []
    original_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: errors.append(context))

    async def cancel_after_response(futures, **kwargs):
        ready = await asyncio.wait(futures, **kwargs)
        assert ready[0]
        asyncio.current_task().cancel()
        await asyncio.sleep(0)
        return ready

    proxy = SimpleNamespace(**vars(asyncio))
    proxy.wait = cancel_after_response
    monkeypatch.setattr(rpc_client_module, "asyncio", proxy)
    try:
        with pytest.raises(asyncio.CancelledError):
            await client.fail().remote_async()
        await _owner_barrier(client)
        gc.collect()
        await asyncio.sleep(0)
        assert not errors
        assert not client._pending_futures
    finally:
        loop.set_exception_handler(original_handler)


@pytest.mark.asyncio
async def test_async_call_from_client_owner_loop(rpc):
    client, _ = rpc
    result = asyncio.run_coroutine_threadsafe(client.echo(42).remote_async(), client._loop)
    assert await asyncio.wait_for(asyncio.wrap_future(result), 2) == 42


@pytest.mark.asyncio
async def test_cancellation_before_owner_dispatch(rpc):
    client, _ = rpc
    blocked = threading.Event()
    release = threading.Event()

    def block_owner():
        blocked.set()
        release.wait(3)

    client._loop.call_soon_threadsafe(block_owner)
    task = None
    try:
        assert await asyncio.to_thread(blocked.wait, 2)
        task = asyncio.create_task(client.echo(42).remote_async())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    await _owner_barrier(client)
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_timeout_while_owner_loop_is_busy(rpc):
    client, _ = rpc
    blocked = threading.Event()
    release = threading.Event()

    def block_owner():
        blocked.set()
        release.wait(3)

    client._loop.call_soon_threadsafe(block_owner)
    try:
        assert await asyncio.to_thread(blocked.wait, 2)
        with pytest.raises(RPCTimeout):
            await asyncio.wait_for(client.echo(42).remote_async(timeout=0.05), 1)
        assert not release.is_set()
    finally:
        release.set()
    await _owner_barrier(client)
    assert not client._pending_futures


def test_async_calls_from_multiple_caller_loops(rpc):
    client, _ = rpc

    def caller(offset):
        async def run():
            values = list(range(offset, offset + 25))
            results = await asyncio.gather(*[client.echo(value).remote_async() for value in values])
            assert results == values

        asyncio.run(run())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as callers:
        results = [callers.submit(caller, offset) for offset in (0, 25, 50, 75)]
        for result in results:
            result.result(timeout=10)
    assert not client._pending_futures
