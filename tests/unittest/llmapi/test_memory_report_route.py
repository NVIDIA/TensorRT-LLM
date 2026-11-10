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
"""``trtllm-serve``'s ``GET /memory_report``.

The route exists only in the monitor's full mode with the runtime-control
endpoints enabled, and takes their authentication.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm import LLM
from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.runtime_control_auth import (
    RuntimeControlReplayCache,
    build_runtime_control_auth_headers,
)

pytestmark = pytest.mark.cpu_only

_KEY = "test-runtime-control-key"


def _server(monkeypatch, *, monitor="1", runtime_control=True, asynchronous=False):
    if monitor is None:
        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    else:
        monkeypatch.setenv("TLLM_MEM_MONITOR", monitor)
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    server._enable_runtime_control_endpoints = runtime_control
    server._runtime_control_api_key = _KEY if runtime_control else None
    server._runtime_control_replay_cache = RuntimeControlReplayCache()
    server.generator = object.__new__(AsyncLLM if asynchronous else LLM)
    factory = AsyncMock if asynchronous else MagicMock
    server.generator.get_memory_report = factory(return_value=["rank 0 report"])
    server._register_memory_report_route()
    return server


def _get(client):
    headers = build_runtime_control_auth_headers(_KEY, "GET", "/memory_report", b"")
    return client.get("/memory_report", headers=headers)


def test_returns_each_workers_report(monkeypatch):
    for asynchronous in (False, True):
        server = _server(monkeypatch, asynchronous=asynchronous)
        with TestClient(server.app) as client:
            response = _get(client)
        assert response.status_code == 200, f"asynchronous={asynchronous}"
        assert response.json() == {"reports": ["rank 0 report"]}


def test_requires_authentication(monkeypatch):
    server = _server(monkeypatch)
    with TestClient(server.app) as client:
        assert client.get("/memory_report").status_code == 401


def test_exists_only_in_full_mode_with_runtime_control(monkeypatch):
    for monitor, runtime_control in [(None, True), ("0", True), ("1", False)]:
        server = _server(monkeypatch, monitor=monitor, runtime_control=runtime_control)
        with TestClient(server.app) as client:
            assert _get(client).status_code == 404, f"{monitor=} {runtime_control=}"

    # The frontend is rank 0 and serves every rank's report, so a rank
    # filter that leaves rank 0 out does not remove the route.
    monkeypatch.setenv("TLLM_MEM_MONITOR_RANKS", "1")
    server = _server(monkeypatch)
    with TestClient(server.app) as client:
        assert _get(client).status_code == 200


def test_failure_is_an_internal_error(monkeypatch):
    server = _server(monkeypatch)
    server.generator.get_memory_report.side_effect = RuntimeError("worker gone")
    with TestClient(server.app) as client:
        response = _get(client)
    assert response.status_code == 500
    assert "worker gone" in response.json()["error"]["message"]


def test_concurrent_requests_share_one_gather(monkeypatch):
    server = _server(monkeypatch)
    calls = []
    release = threading.Event()

    def slow_report():
        calls.append(1)
        release.wait(5)
        return ["rank 0 report"]

    server.generator.get_memory_report = slow_report

    async def two_requests():
        first = asyncio.ensure_future(server.memory_report())
        second = asyncio.ensure_future(server.memory_report())
        await asyncio.sleep(0.1)
        release.set()
        return await asyncio.gather(first, second)

    responses = asyncio.run(two_requests())
    assert [r.status_code for r in responses] == [200, 200]
    assert len(calls) == 1, "one gather served both requests"
