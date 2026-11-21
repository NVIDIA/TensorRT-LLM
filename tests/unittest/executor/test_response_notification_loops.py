# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched responses must wake each request on its own asyncio loop."""

import asyncio
import queue
import threading
from contextlib import ExitStack
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


def _stop_loop(loop, thread):
    async def cancel_pending():
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    try:
        asyncio.run_coroutine_threadsafe(cancel_pending(), loop).result(2)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        assert not thread.is_alive()
        loop.close()


@pytest.fixture
def consumers(request):
    """Real loops, with no timeout timer that could accidentally wake a stuck get."""
    owners = []
    errors = []
    with ExitStack() as cleanup:
        for _ in range(2):
            loop = asyncio.new_event_loop()
            loop.set_debug(request.param)
            loop.set_exception_handler(lambda loop, context: errors.append(context))
            idle = threading.Event()
            loop._selector.select = partial(_select_when_idle, loop._selector.select, idle)
            thread = threading.Thread(target=loop.run_forever, daemon=True)
            thread.start()
            cleanup.callback(_stop_loop, loop, thread)

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


def _broadcast_crash(queues):
    delivered = threading.Event()
    results = {index: SimpleNamespace(queue=q) for index, q in enumerate(queues)}
    helper = AwaitResponseHelper.__new__(AwaitResponseHelper)
    helper.handler_kind = AwaitResponseHelper.HandlerKind.single_process_worker
    helper.worker = SimpleNamespace(
        _results=results,
        return_queue=lambda client_id: results[client_id].queue,
        _pop_result=lambda client_id: results.pop(client_id),
        engine=SimpleNamespace(_event_loop_error_delivered=delivered),
    )
    assert not helper._broadcast_event_loop_error(RuntimeError("engine failed"))
    assert not results
    return delivered


def _closed_sync_queue():
    loop = asyncio.new_event_loop()
    try:
        return _SyncQueue(Mock(), loop)
    finally:
        loop.close()


@pytest.mark.parametrize("consumers", [True], indirect=True)
@pytest.mark.parametrize("closed_first", [False, True])
def test_engine_error_broadcast_wakes_each_request_loop(consumers, closed_first):
    owners, errors = consumers
    queues = [owner.queue.sync_q for owner in owners]
    if closed_first:
        queues.insert(0, _closed_sync_queue())
    delivered = _broadcast_crash(queues)
    for index, owner in enumerate(owners, start=int(closed_first)):
        response = owner.result.result(timeout=2)
        assert response.client_id == index
        assert "engine failed" in response.error_msg
    assert delivered.is_set()
    assert not errors


def test_engine_error_delivery_gate_stays_clear_without_a_live_loop():
    delivered = _broadcast_crash([_closed_sync_queue()])
    assert not delivered.is_set()


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
    callback, notified_queues = loop.call_soon_threadsafe.call_args.args
    assert callback is _SyncQueue._notify_many
    assert set(notified_queues) == {q.sync_q for q in queues}
