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
import time
from collections import deque
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from tensorrt_llm.executor.rpc import RPCCancelled
from tensorrt_llm.serve.openai_server import OpenAIServer

# The CPU-* CI stages run pytest with -m 'cpu_only'; without the marker every
# test here is deselected.
pytestmark = pytest.mark.cpu_only


class _FakeGenerator:
    def __init__(
        self,
        stat_batches: list[list[dict]],
        iter_stats_max_iterations: int | None = None,
        enable_iter_perf_stats: bool = True,
        stats_delay: float = 0.0,
        stats_error: Exception | None = None,
    ):
        self.args = SimpleNamespace(
            iter_stats_max_iterations=iter_stats_max_iterations,
            enable_iter_perf_stats=enable_iter_perf_stats,
            return_perf_metrics=False,
        )
        self._executor = SimpleNamespace(resource_governor_queue=None)
        self._stat_batches = deque(stat_batches)
        self._stats_delay = stats_delay
        self._stats_error = stats_error
        self.stats_timeouts = []

    def get_stats(self, timeout: float | None) -> list[dict]:
        # Blocks like the executor's stats RPC, which waits for stats to arrive.
        self.stats_timeouts.append(timeout)
        time.sleep(self._stats_delay)
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
    stats_delay: float = 0.0,
    stats_error: Exception | None = None,
) -> OpenAIServer:
    server = object.__new__(OpenAIServer)
    server.generator = _FakeGenerator(
        stat_batches,
        iter_stats_max_iterations,
        enable_iter_perf_stats,
        stats_delay,
        stats_error,
    )
    server.metrics_collector = _FakeMetricsCollector()
    server._is_visual_gen = is_visual_gen
    max_buffer_size = OpenAIServer._iteration_stats_buffer_maxlen(
        server.generator.args.iter_stats_max_iterations
    )
    server._iteration_stats_buffer = deque(maxlen=max_buffer_size) if with_stats_buffer else None
    return server


def _response_content(response):
    return json.loads(response.body.decode("utf-8"))


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
    stats = [{"iter": 7}]
    server = _make_server([stats], with_stats_buffer=False, stats_delay=0.5)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    try:
        response = await server.get_iteration_stats()
    finally:
        ticker_task.cancel()

    assert _response_content(response) == stats
    # About 50 ticks fit in the 0.5 s wait; none would if it held the loop.
    assert ticks >= 10


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
