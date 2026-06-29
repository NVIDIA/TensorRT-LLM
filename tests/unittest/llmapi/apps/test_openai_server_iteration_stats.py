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
import json
from collections import deque
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_server import OpenAIServer


class _FakeGenerator:
    def __init__(self, stat_batches: list[list[dict]]):
        self.args = SimpleNamespace(iter_stats_max_iterations=None)
        self._stat_batches = deque(stat_batches)
        self.stats_timeouts = []

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
) -> OpenAIServer:
    server = object.__new__(OpenAIServer)
    server.generator = _FakeGenerator(stat_batches)
    server.metrics_collector = _FakeMetricsCollector()
    server._is_visual_gen = is_visual_gen
    max_buffer_size = server.generator.args.iter_stats_max_iterations or 1000
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
async def test_metrics_endpoint_reads_queue_without_background_buffer():
    stats = [{"iter": 5}, {"iter": 6}]
    server = _make_server([stats], with_stats_buffer=False)

    response = await server.get_iteration_stats()

    assert _response_content(response) == stats
    assert server.generator.stats_timeouts == [2]
    assert server.metrics_collector.logged_stats == []


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
