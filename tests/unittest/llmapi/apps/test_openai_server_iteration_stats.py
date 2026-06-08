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
from collections import deque
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_server import OpenAIServer


class _FakeGenerator:
    def __init__(self, stat_batches: list[list[dict]]):
        self.args = SimpleNamespace(iter_stats_max_iterations=None)
        self._stat_batches = deque(stat_batches)

    def get_stats_async(self, _timeout: float):
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


def _make_server(stat_batches: list[list[dict]]) -> OpenAIServer:
    server = object.__new__(OpenAIServer)
    server.generator = _FakeGenerator(stat_batches)
    server.metrics_collector = _FakeMetricsCollector()
    server._iteration_stats_lock = asyncio.Lock()
    server._iteration_stats_buffer = deque(maxlen=server._get_iteration_stats_buffer_maxlen())
    return server


def _response_content(response):
    return json.loads(response.body.decode("utf-8"))


@pytest.mark.asyncio
async def test_background_iteration_stats_drain_preserves_metrics_endpoint():
    stats = [{"iter": 1}, {"iter": 2}]
    server = _make_server([stats, []])

    await server._drain_iteration_stats_to_sinks(timeout=0.5)
    response = await server.get_iteration_stats()

    assert _response_content(response) == stats
    assert server.metrics_collector.logged_stats == stats


@pytest.mark.asyncio
async def test_metrics_endpoint_drain_logs_prometheus_metrics():
    stats = [{"iter": 3}, {"iter": 4}]
    server = _make_server([stats])

    response = await server.get_iteration_stats()

    assert _response_content(response) == stats
    assert server.metrics_collector.logged_stats == stats

    response = await server.get_iteration_stats()
    assert _response_content(response) == []
