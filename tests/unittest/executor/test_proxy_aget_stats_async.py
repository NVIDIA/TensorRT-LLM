# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``aget_stats`` awaits the stats RPC instead of blocking the event loop (IPC and RPC proxies).

Only for a model that opts in (``RuntimeHooks.async_stats_rpc``); every other model keeps the
blocking RPC. The awaited path returns the same records in the same order as the blocking path and
keeps the executor's ``IterationResult`` contract; the event loop keeps running while the RPC is in
flight.
"""

import asyncio
import concurrent.futures
import json
import threading
import time

import pytest

from tensorrt_llm.executor import proxy as proxy_mod
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.result import IterationResult
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy

pytestmark = pytest.mark.cpu_only

_RECORDS = [{"iter": i, "numActiveRequests": i % 3} for i in range(5)]


class _RpcGate:
    """Holds a fake RPC until the event loop releases it, or until ``timeout`` passes."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.started = threading.Event()
        self.release = threading.Event()
        self.released_by_loop = None

    def hold(self):
        self.started.set()
        self.released_by_loop = self.release.wait(self.timeout)


class _FakeCall:
    def __init__(self, pool, delay, records, gate=None):
        self._pool, self._delay, self._records, self._gate = pool, delay, records, gate

    def _run(self):
        if self._gate is None:
            time.sleep(self._delay)
        else:
            self._gate.hold()
        return [json.dumps(r) for r in self._records]

    def remote(self, timeout=None):
        return self._run()

    def remote_future(self, timeout=None):
        return self._pool.submit(self._run)


class _OptInCall:
    def __init__(self, opted_in):
        self._opted_in = opted_in

    def remote(self, timeout=None):
        if isinstance(self._opted_in, Exception):
            raise self._opted_in
        return self._opted_in


class _FakeRpcClient:
    def __init__(self, pool, delay, records, gate=None, opted_in=True):
        self._pool, self._delay, self._records, self._gate = pool, delay, records, gate
        self.opted_in = opted_in
        self.opt_in_queries = 0

    def async_stats_rpc_opted_in(self):
        self.opt_in_queries += 1
        return _OptInCall(self.opted_in)

    def fetch_stats_wait_async(self, timeout=None):
        return _FakeCall(self._pool, self._delay, self._records, self._gate)


@pytest.fixture
def pool():
    p = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    yield p
    p.shutdown(wait=True)


def _bare(cls, client):
    executor = cls.__new__(cls)
    executor._is_llm_executor = True
    executor._iter_stats_result = None
    executor._iter_kv_events_result = None
    executor.rpc_client = client
    return executor


async def _collect(executor, timeout):
    return [stat async for stat in executor.aget_stats(timeout=timeout)]


@pytest.mark.parametrize("cls", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("async_rpc", [False, True])
def test_event_loop_runs_during_rpc(cls, async_rpc, pool, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_ASYNC_STATS_RPC", async_rpc)
    # The RPC returns only once an event-loop task releases it. A blocking RPC holds the loop, so
    # no task can release it and it runs out its (short) hold; the outcome does not depend on
    # timing. An awaited RPC is released as soon as the loop runs; the long bound only catches a
    # loop that never does.
    gate = _RpcGate(timeout=30.0 if async_rpc else 0.3)

    async def main():
        async def releaser():
            while not gate.started.is_set():
                await asyncio.sleep(0.001)
            gate.release.set()

        task = asyncio.create_task(releaser())
        executor = _bare(cls, _FakeRpcClient(pool, 0.0, _RECORDS, gate))
        records = await _collect(executor, 0.05)
        await task
        return records

    records = asyncio.run(main())
    assert records == _RECORDS
    assert gate.released_by_loop is async_rpc


@pytest.mark.parametrize("cls", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("async_rpc", [False, True])
def test_the_iteration_result_contract_is_kept(cls, async_rpc, pool, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_ASYNC_STATS_RPC", async_rpc)

    async def main():
        executor = _bare(cls, _FakeRpcClient(pool, 0.05, _RECORDS))
        first = executor.aget_stats(timeout=0.05)
        assert isinstance(first, IterationResult)
        assert first._timeout == 0.05
        records = [stat async for stat in first]
        # The same object serves the next call, which drains a fresh fetch once it is undone.
        first.mark_undone()
        second = executor.aget_stats(timeout=0.05)
        assert second is first
        return records, [stat async for stat in second]

    first_records, second_records = asyncio.run(main())
    assert first_records == _RECORDS
    assert second_records == _RECORDS


@pytest.mark.parametrize("cls", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
def test_get_results_waits_for_the_pending_fetch(cls, pool, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_ASYNC_STATS_RPC", True)

    async def main():
        executor = _bare(cls, _FakeRpcClient(pool, 0.05, _RECORDS))
        return executor.aget_stats(timeout=0.05).get_results()

    assert asyncio.run(main()) == _RECORDS


def test_a_failed_fetch_queues_nothing():
    async def main():
        result = IterationResult()
        future = concurrent.futures.Future()
        future.set_exception(RuntimeError("rpc failed"))
        result.add_pending_records(future)
        result.set_timeout(0.01)
        return [stat async for stat in result]

    assert asyncio.run(main()) == []


def test_a_cancelled_read_leaves_its_pending_fetches_to_the_next_read():
    async def main():
        result = IterationResult()
        result.set_timeout(0.01)
        first, second = concurrent.futures.Future(), concurrent.futures.Future()
        result.add_pending_records(first)
        result.add_pending_records(second)
        reader = asyncio.ensure_future(result.__anext__())
        await asyncio.sleep(0.01)  # the reader waits on the first fetch
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        assert not first.cancelled()
        first.set_result([json.dumps({"iter": 1})])
        second.set_result([json.dumps({"iter": 2})])
        return [stat async for stat in result]

    assert asyncio.run(main()) == [{"iter": 1}, {"iter": 2}]


@pytest.mark.parametrize("cls", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("opted_in", [False, None, RuntimeError("no such method")])
def test_a_model_without_the_opt_in_keeps_the_blocking_rpc(cls, opted_in, pool, monkeypatch):
    """Without RuntimeHooks.async_stats_rpc, or with a worker that cannot answer, the RPC blocks.

    The worker is asked once.
    """
    monkeypatch.setattr(proxy_mod, "_ASYNC_STATS_RPC", True)
    gate = _RpcGate(timeout=0.3)

    async def main():
        async def releaser():
            while not gate.started.is_set():
                await asyncio.sleep(0.001)
            gate.release.set()

        task = asyncio.create_task(releaser())
        client = _FakeRpcClient(pool, 0.0, _RECORDS, gate, opted_in=opted_in)
        executor = _bare(cls, client)
        records = await _collect(executor, 0.05)
        await task
        released_by_loop = gate.released_by_loop
        executor.aget_stats(timeout=0.05)
        return records, released_by_loop, client.opt_in_queries

    records, released_by_loop, queries = asyncio.run(main())
    assert records == _RECORDS
    assert released_by_loop is False
    assert queries == 1


def test_the_kill_switch_skips_the_opt_in_query(pool, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_ASYNC_STATS_RPC", False)
    client = _FakeRpcClient(pool, 0.0, _RECORDS)
    executor = _bare(GenerationExecutorProxy, client)
    assert proxy_mod._async_stats_rpc(executor) is False
    assert client.opt_in_queries == 0


@pytest.mark.parametrize("hooks, expected", [(None, False), (False, False), (True, True)])
def test_the_worker_reports_the_model_opt_in(hooks, expected):
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
    from tensorrt_llm.executor.base_worker import BaseWorker

    model = SimpleNamespace()
    if hooks is not None:
        model.runtime_hooks = RuntimeHooks(async_stats_rpc=hooks)
    worker = BaseWorker.__new__(BaseWorker)
    worker.engine = SimpleNamespace(model_engine=SimpleNamespace(model=model))
    assert worker.async_stats_rpc_opted_in() is expected
    worker.engine = SimpleNamespace()
    assert worker.async_stats_rpc_opted_in() is False


class _CountingClient(_FakeRpcClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def fetch_stats_wait_async(self, timeout=None):
        self.calls += 1
        return super().fetch_stats_wait_async(timeout)


@pytest.mark.parametrize("cls", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
def test_at_most_one_stats_rpc_is_in_flight(cls, pool, monkeypatch):
    """A second call while the first RPC runs starts none, so records keep the worker's order."""
    monkeypatch.setattr(proxy_mod, "_ASYNC_STATS_RPC", True)
    gate = _RpcGate(timeout=30.0)

    async def main():
        client = _CountingClient(pool, 0.0, _RECORDS, gate)
        executor = _bare(cls, client)
        first = executor.aget_stats(timeout=0.05)
        second = executor.aget_stats(timeout=0.05)
        assert second is first
        assert client.calls == 1 and len(first._pending_records) == 1
        gate.release.set()
        return [stat async for stat in first]

    assert asyncio.run(main()) == _RECORDS


def test_pending_records_are_read_in_drain_order():
    async def main():
        result = IterationResult()
        result.set_timeout(0.01)
        first, second = concurrent.futures.Future(), concurrent.futures.Future()
        result.add_pending_records(first)
        result.add_pending_records(second)
        assert result.has_inflight_records()
        second.set_result([json.dumps({"iter": 2})])
        first.set_result([json.dumps({"iter": 1})])
        assert not result.has_inflight_records()
        return [stat async for stat in result]

    assert asyncio.run(main()) == [{"iter": 1}, {"iter": 2}]
