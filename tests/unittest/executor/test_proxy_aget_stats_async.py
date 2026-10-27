# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``aget_stats`` awaits the stats RPC instead of blocking the event loop.

trtllm-serve's stats collector calls ``get_stats_async`` after every finished
request on the same event loop that sends every response and token. The RPC
waits up to ``timeout`` for a stats record, so a blocking ``.remote()`` there
stalls the server. These tests pin, for the IPC and the RPC proxy:

* the awaited path returns the same records in the same order as the blocking
  path (``TRTLLM_M31_DISABLE_A4_ASYNC_STATS=1``);
* the event loop keeps running while the awaited RPC is in flight, and does not
  run while the blocking one is.
"""

import asyncio
import concurrent.futures
import json
import time

import pytest

from tensorrt_llm.executor import proxy as proxy_mod
from tensorrt_llm.executor import rpc_proxy as rpc_proxy_mod
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy

_RECORDS = [{"iter": i, "numActiveRequests": i % 3} for i in range(5)]


class _FakeCall:
    def __init__(self, pool, delay, records):
        self._pool, self._delay, self._records = pool, delay, records

    def _run(self):
        time.sleep(self._delay)
        return [json.dumps(r) for r in self._records]

    def remote(self):
        return self._run()

    def remote_future(self):
        return self._pool.submit(self._run)


class _FakeRpcClient:
    def __init__(self, pool, delay, records):
        self._pool, self._delay, self._records = pool, delay, records

    def fetch_stats_wait_async(self, timeout=None):
        return _FakeCall(self._pool, self._delay, self._records)


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
def test_same_records_same_order(cls, pool, monkeypatch):
    got = {}
    for async_rpc in (False, True):
        monkeypatch.setattr(proxy_mod, "_ASYNC_STATS_RPC", async_rpc)
        monkeypatch.setattr(rpc_proxy_mod, "_ASYNC_STATS_RPC", async_rpc)
        executor = _bare(cls, _FakeRpcClient(pool, 0.0, _RECORDS))
        got[async_rpc] = asyncio.run(_collect(executor, 0.05))
    assert got[False] == _RECORDS
    assert got[True] == got[False]


@pytest.mark.parametrize("cls", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("async_rpc", [False, True])
def test_event_loop_runs_during_rpc(cls, async_rpc, pool, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_ASYNC_STATS_RPC", async_rpc)
    monkeypatch.setattr(rpc_proxy_mod, "_ASYNC_STATS_RPC", async_rpc)
    delay = 0.3

    async def main():
        ticks = []
        stop = asyncio.Event()

        async def ticker():
            while not stop.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.005)

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.02)
        executor = _bare(cls, _FakeRpcClient(pool, delay, _RECORDS))
        t0 = time.monotonic()
        records = await _collect(executor, 0.05)
        stop.set()
        await task
        during = [t for t in ticks if t0 < t < t0 + delay]
        return records, len(during)

    records, ticks_during_rpc = asyncio.run(main())
    assert records == _RECORDS
    if async_rpc:
        # ~60 ticks of 5 ms fit in the 300 ms RPC; leave a wide margin.
        assert ticks_during_rpc >= 20
    else:
        assert ticks_during_rpc <= 1
