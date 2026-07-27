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

import pytest

from tensorrt_llm.serve import perf_metrics


class _DummyMetric:
    def inc(self):
        pass

    def observe(self, _value):
        pass


class _BlockingClient:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active_collectors = 0
        self.max_active_collectors = 0
        self.calls = 0

    async def collect_metrics(self):
        self.calls += 1
        self.active_collectors += 1
        self.max_active_collectors = max(self.max_active_collectors, self.active_collectors)
        self.entered.set()
        await self.release.wait()
        self.active_collectors -= 1
        return {}


@pytest.mark.asyncio
async def test_disagg_perf_metrics_collection_is_serialized(monkeypatch):
    monkeypatch.setattr(perf_metrics, "instance_metric", lambda _definition: _DummyMetric())
    collector = perf_metrics.DisaggPerfMetricsCollector(max_requests=8)
    client = _BlockingClient()
    collector.add_client(client)

    first_task = asyncio.create_task(collector.get_perf_metrics())
    await client.entered.wait()

    second_task = asyncio.create_task(collector.get_perf_metrics())
    await asyncio.sleep(0)

    assert client.calls == 1
    assert client.max_active_collectors == 1

    client.release.set()
    assert await first_task == []
    assert await second_task == []
    assert client.calls == 2
    assert client.max_active_collectors == 1
