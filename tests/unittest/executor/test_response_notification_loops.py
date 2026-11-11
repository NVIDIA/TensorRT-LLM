# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched responses must wake each request on its own asyncio loop."""

import asyncio
import queue
import threading
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.executor import base_worker
from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.executor.utils import ErrorResponse
from tensorrt_llm.llmapi.utils import AsyncQueue, _SyncQueue

pytestmark = pytest.mark.cpu_only


def _select_when_idle(select, idle, timeout=None):
    if timeout is None:
        idle.set()
    return select(timeout)


@pytest.fixture
def consumers(request):
    """Real loops, with no timeout timer that could accidentally wake a stuck get."""
    owners = []
    errors = []
    try:
        for _ in range(2):
            loop = asyncio.new_event_loop()
            loop.set_debug(request.param)
            loop.set_exception_handler(lambda loop, context: errors.append(context))
            idle = threading.Event()
            loop._selector.select = partial(_select_when_idle, loop._selector.select, idle)
            thread = threading.Thread(target=loop.run_forever, daemon=True)
            thread.start()

            async def prepare():
                result_queue = AsyncQueue()
                waiter = asyncio.create_task(result_queue.get())
                await asyncio.sleep(0)
                return result_queue, waiter

            result_queue, waiter = asyncio.run_coroutine_threadsafe(prepare(), loop).result(2)

            async def consume(task):
                return await task

            result = asyncio.run_coroutine_threadsafe(consume(waiter), loop)
            owners.append(
                SimpleNamespace(
                    loop=loop, thread=thread, queue=result_queue, result=result, idle=idle
                )
            )
        # Prove both loops reached a blocking select after their consumers
        # parked. A runnable setup callback could otherwise hide a missed wake.
        for owner in owners:
            cleared = threading.Event()

            def clear_idle(owner=owner, cleared=cleared):
                owner.idle.clear()
                cleared.set()

            owner.loop.call_soon_threadsafe(clear_idle)
            assert cleared.wait(2)
            assert owner.idle.wait(2)
        yield owners, errors
    finally:
        for owner in owners:
            owner.result.cancel()

            async def settle():
                await asyncio.sleep(0)

            asyncio.run_coroutine_threadsafe(settle(), owner.loop).result(2)
            owner.loop.call_soon_threadsafe(owner.loop.stop)
            owner.thread.join(2)
            assert not owner.thread.is_alive()
            owner.loop.close()


def _dispatch(dispatcher, results, responses):
    if dispatcher == "proxy":
        transport = queue.Queue()
        transport.put(responses)
        proxy = SimpleNamespace(_results=results, result_queue=transport)
        assert GenerationExecutorProxy.dispatch_result_task(proxy)
    elif dispatcher == "rpc":
        RpcExecutorMixin.handle_responses(SimpleNamespace(_results=results), responses)
    else:
        worker = SimpleNamespace(
            return_queue=lambda client_id: results[client_id].queue,
            _pop_result=lambda client_id: results.pop(client_id),
            _refused_client_ids=set(),
        )
        AwaitResponseHelper.handle_for_worker(SimpleNamespace(worker=worker), responses)


@pytest.mark.parametrize("consumers", [False, True], indirect=True)
@pytest.mark.parametrize("dispatcher", ["proxy", "rpc", "worker"])
@pytest.mark.parametrize("success", [False, True])
def test_batched_responses_wake_all_request_loops(consumers, dispatcher, success, monkeypatch):
    owners, errors = consumers
    if success:
        responses = [
            SimpleNamespace(
                client_id=index, has_error=lambda: False, result=SimpleNamespace(is_final=True)
            )
            for index in range(len(owners))
        ]
        monkeypatch.setattr(base_worker, "_maybe_wrap_response", lambda worker, response: response)
    else:
        responses = [
            ErrorResponse(client_id=index, error_msg="request failed", request_id=index)
            for index in range(len(owners))
        ]
    results = {
        index: SimpleNamespace(queue=owner.queue.sync_q) for index, owner in enumerate(owners)
    }
    _dispatch(dispatcher, results, responses)
    for owner, expected in zip(owners, responses):
        # A concurrent Future timeout does not wake the consumer's loop.
        assert owner.result.result(timeout=2) is expected
    assert not errors
    assert not results


@pytest.mark.parametrize("consumers", [True], indirect=True)
@pytest.mark.parametrize("dispatcher", ["proxy", "rpc", "worker"])
def test_stopped_request_loop_does_not_strand_other_requests(consumers, dispatcher):
    owners, errors = consumers
    stopped_loop = asyncio.new_event_loop()

    async def prepare():
        return AsyncQueue()

    stopped_queue = stopped_loop.run_until_complete(prepare())
    stopped_loop.close()
    all_queues = [stopped_queue.sync_q] + [owner.queue.sync_q for owner in owners]
    responses = [
        ErrorResponse(client_id=index, error_msg="request failed", request_id=index)
        for index in range(len(all_queues))
    ]
    results = {index: SimpleNamespace(queue=q) for index, q in enumerate(all_queues)}
    _dispatch(dispatcher, results, responses)
    for owner, expected in zip(owners, responses[1:]):
        assert owner.result.result(timeout=2) is expected
    assert not errors
    assert not results


@pytest.mark.parametrize("consumers", [True], indirect=True)
def test_engine_error_broadcast_wakes_each_request_loop(consumers):
    owners, errors = consumers
    delivered = threading.Event()
    results = {
        index: SimpleNamespace(queue=owner.queue.sync_q) for index, owner in enumerate(owners)
    }
    helper = AwaitResponseHelper.__new__(AwaitResponseHelper)
    helper.handler_kind = AwaitResponseHelper.HandlerKind.single_process_worker
    helper.worker = SimpleNamespace(
        _results=results,
        return_queue=lambda client_id: results[client_id].queue,
        _pop_result=lambda client_id: results.pop(client_id),
        engine=SimpleNamespace(_event_loop_error_delivered=delivered),
    )
    assert not helper._broadcast_event_loop_error(RuntimeError("engine failed"))
    for index, owner in enumerate(owners):
        response = owner.result.result(timeout=2)
        assert response.client_id == index
        assert "engine failed" in response.error_msg
    assert delivered.is_set()
    assert not errors
    assert not results


@pytest.mark.parametrize("dispatcher", ["proxy", "rpc", "worker"])
def test_shared_loop_notifications_stay_batched(dispatcher):
    loop = Mock()
    loop.is_running.return_value = True
    queues = [SimpleNamespace(sync_q=_SyncQueue(Mock(), loop)) for _ in range(8)]
    responses = [
        ErrorResponse(client_id=index, error_msg="request failed", request_id=index)
        for index in range(len(queues))
    ]
    results = {index: SimpleNamespace(queue=q.sync_q) for index, q in enumerate(queues)}
    _dispatch(dispatcher, results, responses)
    loop.call_soon_threadsafe.assert_called_once()
