# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process ownership and asynchronous delivery of iteration observations."""

import asyncio
import json
import multiprocessing
import os
from concurrent.futures import Future
from multiprocessing.connection import Connection
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor import iteration_stats
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.iteration_stats import (
    IterationStatsFrame,
    IterationStatsSnapshot,
    RankStatsSnapshot,
)
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.result import IterationResult, _StatsBatchFetcher
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin

pytestmark = pytest.mark.cpu_only


def _frame() -> IterationStatsFrame:
    return IterationStatsFrame(
        stats=IterationStatsSnapshot(iter=17, num_active_requests=3),
        rank_payloads=(
            RankStatsSnapshot(0, 1, 12, 0, 1, 64, 0, 0),
            RankStatsSnapshot(1, 0, 0, 0, 1, 128, 0, 0),
        ),
    )


class _StatsWorker(RpcWorkerMixin):
    def __init__(self) -> None:
        self.rank = 0

    def fetch_stats(self) -> list:
        return [_frame()]

    @staticmethod
    def _stats_serializer(stats: object) -> str:
        raise AssertionError("Report assembly ran in the worker")


def _send_worker_stats(connection: Connection) -> None:
    try:
        stats = asyncio.run(_StatsWorker().fetch_stats_async())
        connection.send((os.getpid(), stats))
    finally:
        connection.close()


class _StatsRpc:
    def __init__(self, future: Future) -> None:
        self.future = future
        self.timeouts: list[float] = []
        self.submissions = 0

    def fetch_stats_wait_async(self, timeout: float) -> "_StatsRpc":
        self.timeouts.append(timeout)
        return self

    def remote(self) -> list:
        raise AssertionError("Statistics retrieval used a blocking RPC")

    def remote_future(self) -> Future:
        self.submissions += 1
        return self.future


def _proxy(proxy_type: type, future: Future) -> SimpleNamespace:
    proxy = SimpleNamespace(
        rpc_client=_StatsRpc(future),
        _stats_fetcher=_StatsBatchFetcher(),
        _iter_stats_result=IterationResult(),
        _maybe_initialize_iteration_results=lambda: None,
    )
    proxy.get_stats = lambda timeout: proxy_type.get_stats(proxy, timeout)
    return proxy


async def _collect(result: IterationResult) -> list[dict]:
    return [row async for row in result]


def test_worker_observations_cross_process_before_rank_report_assembly(monkeypatch) -> None:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_send_worker_stats, args=(sender,))
    worker.start()
    sender.close()
    try:
        assert receiver.poll(30), "Statistics worker did not return an observation"
        worker_pid, stats = receiver.recv()
        worker.join(timeout=10)
        assert worker.exitcode == 0
    finally:
        receiver.close()
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)

    assert worker_pid != os.getpid()
    assert len(stats) == 1
    assert isinstance(stats[0], IterationStatsFrame)
    assembly_pids = []
    materialize = iteration_stats._materialize_frame

    def record_assembly(frame: IterationStatsFrame) -> list[dict]:
        assembly_pids.append(os.getpid())
        return materialize(frame)

    monkeypatch.setattr(iteration_stats, "_materialize_frame", record_assembly)
    future = Future()
    future.set_result(stats)
    proxy = _proxy(GenerationExecutorProxy, future)
    reports = proxy.get_stats(timeout=0)
    assert assembly_pids == [os.getpid()]
    assert [row["attentionDpRank"] for row in reports] == [0, 1]
    assert [row["iter"] for row in reports] == [17, 17]
    assert [row["inflightBatchingStats"]["numCtxTokens"] for row in reports] == [12, 0]


@pytest.mark.parametrize("method", ["fetch_stats_async", "fetch_stats_wait_async"])
def test_worker_rpc_keeps_owned_frames_unexpanded(method: str) -> None:
    stats = asyncio.run(getattr(_StatsWorker(), method)(timeout=0))
    assert len(stats) == 1
    assert isinstance(stats[0], IterationStatsFrame)
    assert len(stats[0].rank_payloads) == 2


@pytest.mark.parametrize("proxy_type", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
def test_async_stats_fetch_yields_event_loop_and_normalizes_once(proxy_type: type) -> None:
    async def check() -> None:
        future = Future()
        proxy = _proxy(proxy_type, future)
        result = proxy_type.aget_stats(proxy, timeout=0.25)
        assert proxy.rpc_client.submissions == 0
        consumer = asyncio.create_task(_collect(result))
        await asyncio.sleep(0)
        assert proxy.rpc_client.submissions == 1
        assert not consumer.done()
        future.set_result([json.dumps({"iter": 1}), {"iter": 2}, _frame()])
        rows = await asyncio.wait_for(consumer, timeout=1)
        assert [row["iter"] for row in rows] == [1, 2, 17, 17]
        assert proxy.rpc_client.timeouts == [0.25]
        assert await _collect(result) == []

    asyncio.run(check())


@pytest.mark.parametrize("proxy_type", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
def test_cancelled_async_fetch_retains_batch_for_next_consumer(proxy_type: type) -> None:
    async def check() -> None:
        future = Future()
        proxy = _proxy(proxy_type, future)
        consumer = asyncio.create_task(_collect(proxy_type.aget_stats(proxy, timeout=1)))
        await asyncio.sleep(0)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert not future.cancelled()
        future.set_result([{"iter": 42}])
        rows = await _collect(proxy_type.aget_stats(proxy, timeout=1))
        assert rows == [{"iter": 42}]
        assert proxy.rpc_client.submissions == 1

    asyncio.run(check())


@pytest.mark.parametrize("proxy_type", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("stop", ["break", "cancel"])
@pytest.mark.parametrize("next_consumer", ["async", "sync", "sync_result"])
def test_partial_async_iteration_retains_rows_for_next_consumer(
    proxy_type: type, stop: str, next_consumer: str
) -> None:
    async def check() -> None:
        future = Future()
        future.set_result([{"iter": 1}, {"iter": 2}, {"iter": 3}])
        proxy = _proxy(proxy_type, future)
        first_result = proxy_type.aget_stats(proxy, timeout=0)
        first_row_consumed = asyncio.Event()

        async def consume_first() -> None:
            async for row in first_result:
                assert row == {"iter": 1}
                first_row_consumed.set()
                if stop == "break":
                    break
                await asyncio.Event().wait()

        first_consumer = asyncio.create_task(consume_first())
        await asyncio.wait_for(first_row_consumed.wait(), timeout=1)
        if stop == "cancel":
            first_consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_consumer
        else:
            await first_consumer

        if next_consumer == "async":
            rows = await _collect(proxy_type.aget_stats(proxy, timeout=0))
        elif next_consumer == "sync":
            rows = proxy.get_stats(timeout=0)
        else:
            rows = proxy_type.aget_stats(proxy, timeout=0).get_results()

        assert rows == [{"iter": 2}, {"iter": 3}]
        assert proxy.rpc_client.submissions == 1
        assert await _collect(first_result) == []

    asyncio.run(check())


def test_concurrent_consumers_claim_destructive_batch_once() -> None:
    async def check() -> None:
        future = Future()
        proxy = _proxy(GenerationExecutorProxy, future)
        consumers = [
            asyncio.create_task(_collect(GenerationExecutorProxy.aget_stats(proxy, timeout=1)))
            for _ in range(2)
        ]
        await asyncio.sleep(0)
        assert proxy.rpc_client.submissions == 1
        future.set_result([{"iter": 9}])
        batches = await asyncio.gather(*consumers)
        assert [row for batch in batches for row in batch] == [{"iter": 9}]

    asyncio.run(check())


@pytest.mark.parametrize("proxy_type", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
def test_async_result_remains_synchronously_consumable(proxy_type: type) -> None:
    future = Future()
    future.set_result([{"iter": 7}])
    proxy = _proxy(proxy_type, future)
    result = proxy_type.aget_stats(proxy, timeout=0)
    assert result.get_results() == [{"iter": 7}]
    assert result.get_results() == []
    assert proxy.rpc_client.submissions == 1


@pytest.mark.parametrize("proxy_type", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("payload", ["{", "[]", "null"])
def test_malformed_stats_are_visible_to_sync_and_async_consumers(
    proxy_type: type, payload: str
) -> None:
    future = Future()
    future.set_result([payload])
    proxy = _proxy(proxy_type, future)
    with pytest.raises((json.JSONDecodeError, TypeError)):
        proxy.get_stats(timeout=0)

    async def check() -> None:
        with pytest.raises((json.JSONDecodeError, TypeError)):
            await _collect(proxy_type.aget_stats(proxy, timeout=0))

    asyncio.run(check())


def test_failed_rpc_is_observable_and_can_be_retried() -> None:
    async def check() -> None:
        future = Future()
        future.set_exception(RuntimeError("Statistics worker failed"))
        proxy = _proxy(GenerationExecutorProxy, future)
        with pytest.raises(RuntimeError, match="Statistics worker failed"):
            await _collect(GenerationExecutorProxy.aget_stats(proxy, timeout=0))
        proxy.rpc_client.future = Future()
        proxy.rpc_client.future.set_result([{"iter": 11}])
        assert proxy.get_stats(timeout=0) == [{"iter": 11}]
        assert proxy.rpc_client.submissions == 2

    asyncio.run(check())


def test_singular_serializer_rejects_rank_fanout() -> None:
    with pytest.raises(ValueError, match="batch materialization"):
        BaseWorker._stats_serializer(_frame())
