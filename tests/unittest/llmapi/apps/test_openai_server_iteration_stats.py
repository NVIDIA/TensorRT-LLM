# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from tensorrt_llm.executor.rpc import RPCCancelled
from tensorrt_llm.serve.openai_server import OpenAIServer

# The CPU-* CI stages run pytest with -m 'cpu_only'; without the marker every
# test here is deselected.
pytestmark = pytest.mark.cpu_only

_METRICS_STATS_THREAD = "test_metrics_stats"
# How long a gated stats read waits for the test to open its gate. Only a broken
# handler, one that blocks the event loop on the read, ever hits it.
_GATE_TIMEOUT_S = 5


class _FakeGenerator:
    def __init__(
        self,
        stat_batches: list[list[dict]],
        iter_stats_max_iterations: int | None = None,
        enable_iter_perf_stats: bool = True,
        stats_gate: threading.Event | None = None,
        stats_error: Exception | None = None,
    ):
        self.args = SimpleNamespace(
            iter_stats_max_iterations=iter_stats_max_iterations,
            enable_iter_perf_stats=enable_iter_perf_stats,
            return_perf_metrics=False,
        )
        self._executor = SimpleNamespace(resource_governor_queue=None)
        self._stat_batches = deque(stat_batches)
        self._stats_gate = stats_gate
        self._stats_error = stats_error
        self.stats_timeouts = []
        self.stats_threads = []

    def get_stats(self, timeout: float | None) -> list[dict]:
        # Blocks like the executor's stats RPC, which waits for stats to arrive;
        # a gated read waits until the test opens the gate.
        self.stats_timeouts.append(timeout)
        self.stats_threads.append(threading.current_thread().name)
        if self._stats_gate is not None:
            assert self._stats_gate.wait(_GATE_TIMEOUT_S), "the stats gate was never opened"
        if self._stats_error is not None:
            raise self._stats_error
        return list(self._stat_batches.popleft()) if self._stat_batches else []

    def get_stats_async(self, timeout: float | None):
        self.stats_timeouts.append(timeout)

        async def _stats_iter():
            if self._stat_batches:
                for stat in self._stat_batches.popleft():
                    yield stat

        return _stats_iter()


class _FakeMetricsCollector:
    def __init__(self):
        self.logged_stats = []

    def log_iteration_stats(self, iteration_stats: dict) -> None:
        self.logged_stats.append(iteration_stats)


def _make_server(
    stat_batches: list[list[dict]],
    *,
    with_stats_buffer: bool = True,
    is_visual_gen: bool = False,
    iter_stats_max_iterations: int | None = None,
    enable_iter_perf_stats: bool = True,
    stats_gate: threading.Event | None = None,
    stats_error: Exception | None = None,
) -> OpenAIServer:
    server = object.__new__(OpenAIServer)
    server.generator = _FakeGenerator(
        stat_batches,
        iter_stats_max_iterations,
        enable_iter_perf_stats,
        stats_gate,
        stats_error,
    )
    server.metrics_collector = _FakeMetricsCollector()
    server._is_visual_gen = is_visual_gen
    max_buffer_size = OpenAIServer._iteration_stats_buffer_maxlen(
        server.generator.args.iter_stats_max_iterations
    )
    server._iteration_stats_buffer = deque(maxlen=max_buffer_size) if with_stats_buffer else None
    # The loop's default executor; a test that checks the dedicated metrics
    # thread installs (and shuts down) its own.
    server._metrics_stats_executor = None
    server._metrics_stats_read = None
    server._metrics_stats_read_waiting = False
    server._metrics_stats_read_consumed = True
    return server


def _response_content(response):
    return json.loads(response.body.decode("utf-8"))


async def _run_ready_tasks():
    """Let every task that is ready to run take its next step."""
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_metrics_endpoint_reads_background_buffer():
    stats = [{"iter": 1}, {"iter": 2}]
    server = _make_server([])
    server._iteration_stats_buffer.extend(stats)

    response = await server.get_iteration_stats()

    assert _response_content(response) == stats
    assert server.generator.stats_timeouts == []

    response = await server.get_iteration_stats()
    assert _response_content(response) == []


@pytest.mark.asyncio
async def test_metrics_endpoint_reads_unbounded_background_buffer():
    stats = [{"iter": idx} for idx in range(3)]
    server = _make_server([], iter_stats_max_iterations=-1)
    assert server._iteration_stats_buffer.maxlen is None
    server._iteration_stats_buffer.extend(stats)

    response = await server.get_iteration_stats()

    assert _response_content(response) == stats


@pytest.mark.asyncio
async def test_metrics_endpoint_reads_queue_without_background_buffer():
    stats = [{"iter": 5}, {"iter": 6}]
    server = _make_server([stats], with_stats_buffer=False)

    response = await server.get_iteration_stats()

    assert _response_content(response) == stats
    assert server.generator.stats_timeouts == [2]
    assert server.metrics_collector.logged_stats == []


@pytest.mark.asyncio
async def test_metrics_endpoint_does_not_block_the_event_loop():
    """Responses keep streaming while /metrics waits for stats."""
    gate = threading.Event()
    server = _make_server([[{"iter": 7}]], with_stats_buffer=False, stats_gate=gate)

    scrape = asyncio.create_task(server.get_iteration_stats())
    await _run_ready_tasks()

    # The loop got back to this test while the read is still blocked. A read on
    # the loop would have held it until the read gave up.
    assert not scrape.done()
    gate.set()
    assert _response_content(await scrape) == [{"iter": 7}]


@pytest.mark.asyncio
async def test_metrics_stats_read_runs_on_the_metrics_thread():
    """The waiting read stays off the default executor, which request handling uses."""
    server = _make_server([[{"iter": 8}]], with_stats_buffer=False)
    server._metrics_stats_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=_METRICS_STATS_THREAD
    )
    try:
        response = await server.get_iteration_stats()
    finally:
        server._metrics_stats_executor.shutdown(wait=True)

    assert _response_content(response) == [{"iter": 8}]
    assert len(server.generator.stats_threads) == 1
    assert server.generator.stats_threads[0].startswith(_METRICS_STATS_THREAD)


@pytest.mark.asyncio
async def test_sustained_scraping_keeps_one_stats_read_in_flight():
    """Scrapes that arrive during a read return at once instead of queueing.

    Queued reads on an idle server each wait the full timeout, so scraping
    faster than one read per timeout would grow the backlog without bound.
    """
    gate = threading.Event()
    server = _make_server([[{"iter": 8}], [{"iter": 9}]], with_stats_buffer=False, stats_gate=gate)

    first = asyncio.create_task(server.get_iteration_stats())
    await _run_ready_tasks()

    # While the first read is blocked, more scrapes return without reading.
    extra = await asyncio.gather(*(server.get_iteration_stats() for _ in range(4)))
    assert [_response_content(r) for r in extra] == [[]] * 4
    assert not first.done()

    gate.set()
    assert _response_content(await first) == [{"iter": 8}]
    assert len(server.generator.stats_timeouts) == 1

    # Once the read finishes, the next scrape reads again.
    response = await server.get_iteration_stats()
    assert _response_content(response) == [{"iter": 9}]
    assert len(server.generator.stats_timeouts) == 2


@pytest.mark.asyncio
async def test_stats_read_for_a_cancelled_scrape_goes_to_the_next_scrape():
    """A scrape that goes away mid-read leaves what the read drained for the next one.

    The read empties the engine's stats queue, so dropping its result would
    lose those iterations for good.
    """
    gate = threading.Event()
    server = _make_server([[{"iter": 8}], [{"iter": 9}]], with_stats_buffer=False, stats_gate=gate)

    abandoned = asyncio.create_task(server.get_iteration_stats())
    await _run_ready_tasks()
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned

    # The read runs to completion although its scrape went away.
    gate.set()
    await server._metrics_stats_read

    response = await server.get_iteration_stats()
    assert _response_content(response) == [{"iter": 8}]
    assert len(server.generator.stats_timeouts) == 1

    response = await server.get_iteration_stats()
    assert _response_content(response) == [{"iter": 9}]
    assert len(server.generator.stats_timeouts) == 2


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="asyncio.Task(eager_start=...) requires Python 3.12+"
)
@pytest.mark.asyncio
async def test_finished_read_stays_with_its_scrape_until_the_scrape_resumes():
    """A scrape arriving after a read finished, but before its scrape resumed, does not replace it.

    Starting a new read there would lose what the first read drained if its
    scrape is then cancelled before resuming.
    """
    gate = threading.Event()
    server = _make_server([[{"iter": 8}], [{"iter": 9}]], with_stats_buffer=False, stats_gate=gate)

    first = asyncio.create_task(server.get_iteration_stats())
    await _run_ready_tasks()
    loop = asyncio.get_running_loop()
    racing = []

    def on_read_done(_):
        # Runs once the read has finished and before the first scrape resumes:
        # another scrape arrives, then the first scrape is cancelled.
        racing.append(asyncio.Task(server.get_iteration_stats(), loop=loop, eager_start=True))
        first.cancel()

    server._metrics_stats_read.add_done_callback(on_read_done)
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert _response_content(await racing[0]) == []

    response = await server.get_iteration_stats()
    assert _response_content(response) == [{"iter": 8}]
    assert len(server.generator.stats_timeouts) == 1


@pytest.mark.asyncio
async def test_metrics_endpoint_reports_no_stats_when_executor_unreachable():
    server = _make_server(
        [],
        with_stats_buffer=False,
        stats_error=RPCCancelled("Server is shutting down, request cancelled"),
    )

    response = await server.get_iteration_stats()

    assert response.status_code == 200
    assert _response_content(response) == []
    assert server.generator.stats_timeouts == [2]


@pytest.mark.asyncio
async def test_metrics_endpoint_does_not_poll_when_iteration_stats_disabled():
    server = _make_server([], with_stats_buffer=False, enable_iter_perf_stats=False)

    for _ in range(2):
        response = await server.get_iteration_stats()
        assert _response_content(response) == []

    assert server.generator.stats_timeouts == []


@pytest.mark.parametrize(
    "enable_iter_perf_stats,with_stats_buffer",
    [(False, False), (True, True)],
)
@pytest.mark.asyncio
async def test_metrics_route_does_not_poll_iteration_stats_queue(
    enable_iter_perf_stats: bool, with_stats_buffer: bool
):
    server = _make_server(
        [],
        with_stats_buffer=with_stats_buffer,
        enable_iter_perf_stats=enable_iter_perf_stats,
    )
    server.app = FastAPI()
    server.use_harmony = False
    # The optional control routes are off by default, as in OpenAIServer.__init__.
    server._enable_runtime_control_endpoints = False
    server._enable_rl_control_endpoints = False
    server.register_routes()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == []
    assert server.generator.stats_timeouts == []


@pytest.mark.asyncio
async def test_metrics_endpoint_reads_visual_gen_stats():
    stats = [
        {
            "iter": 7,
            "numQueuedRequests": 2,
            "numActiveRequests": 1,
        }
    ]
    server = _make_server([stats], is_visual_gen=True)

    response = await server.get_iteration_stats()

    assert _response_content(response) == stats
    assert server.generator.stats_timeouts == [None]
    assert server.metrics_collector.logged_stats == []
