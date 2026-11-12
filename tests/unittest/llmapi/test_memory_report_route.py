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

The route exists in the monitor's full mode when a runtime-control key is
set, and takes the runtime-control request signing. Sleep mode and the
runtime-control endpoints are not needed.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm import LLM
from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.llmapi.llm_args import MemoryMonitorConfig
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.runtime_control_auth import (
    RuntimeControlReplayCache,
    build_runtime_control_auth_headers,
)

pytestmark = pytest.mark.cpu_only

_KEY = "test-runtime-control-key"
_KEY_ENV = "TRTLLM_RUNTIME_CONTROL_API_KEY"


@pytest.fixture(autouse=True)
def _clear_monitor_configuration():
    yield
    from tensorrt_llm._torch.memory.monitor import configure

    configure(None)


def _server(
    monkeypatch,
    *,
    monitor="1",
    key="env",
    asynchronous=False,
    config=None,
    frontends=1,
):
    """A server with the runtime-control key from ``key``.

    ``"env"`` sets ``TRTLLM_RUNTIME_CONTROL_API_KEY``, ``"server"`` passes it
    as the server's runtime-control key, and ``None`` sets it nowhere.
    """
    if monitor is None:
        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    else:
        monkeypatch.setenv("TLLM_MEM_MONITOR", monitor)
    if key == "env":
        monkeypatch.setenv(_KEY_ENV, _KEY)
    else:
        monkeypatch.delenv(_KEY_ENV, raising=False)
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    # Sleep mode, and with it the runtime-control endpoints, stays off.
    server._enable_runtime_control_endpoints = False
    server._runtime_control_api_key = _KEY if key == "server" else None
    server._runtime_control_replay_cache = RuntimeControlReplayCache()
    server.generator = object.__new__(AsyncLLM if asynchronous else LLM)
    server.generator.args = SimpleNamespace(
        memory_monitor_config=config, num_serve_frontends=frontends
    )
    factory = AsyncMock if asynchronous else MagicMock
    server.generator.get_memory_report = factory(return_value=["rank 0 report"])
    server._register_memory_report_route()
    return server


def _get(client):
    headers = build_runtime_control_auth_headers(_KEY, "GET", "/memory_report", b"")
    return client.get("/memory_report", headers=headers)


def test_returns_each_workers_report(monkeypatch):
    for asynchronous in (False, True):
        for key in ("env", "server"):
            server = _server(monkeypatch, asynchronous=asynchronous, key=key)
            with TestClient(server.app) as client:
                response = _get(client)
            assert response.status_code == 200, f"{asynchronous=} {key=}"
            assert response.json() == {"reports": ["rank 0 report"]}


def test_full_mode_from_the_configuration(monkeypatch):
    server = _server(monkeypatch, monitor=None, config=MemoryMonitorConfig(mode="full"))
    with TestClient(server.app) as client:
        assert _get(client).status_code == 200


def test_requires_authentication(monkeypatch):
    server = _server(monkeypatch)
    with TestClient(server.app) as client:
        assert client.get("/memory_report").status_code == 401
        headers = build_runtime_control_auth_headers("wrong-key", "GET", "/memory_report", b"")
        assert client.get("/memory_report", headers=headers).status_code == 401


def test_not_served_without_full_mode(monkeypatch):
    for monitor in (None, "oom", "0"):
        server = _server(monkeypatch, monitor=monitor)
        with TestClient(server.app) as client:
            assert _get(client).status_code == 404, f"{monitor=}"


def test_not_served_without_a_key_and_says_why(monkeypatch):
    logged = []
    monkeypatch.setattr(openai_server.logger, "info", logged.append)
    server = _server(monkeypatch, key=None)
    with TestClient(server.app) as client:
        assert _get(client).status_code == 404
    assert any("/memory_report is not served" in m and _KEY_ENV in m for m in logged), logged


def test_not_served_by_several_frontends(monkeypatch):
    logged = []
    monkeypatch.setattr(openai_server.logger, "info", logged.append)
    server = _server(monkeypatch, frontends=2)
    with TestClient(server.app) as client:
        assert _get(client).status_code == 404
    assert any("num_serve_frontends" in m for m in logged), logged


def test_a_rank_filter_does_not_remove_the_route(monkeypatch):
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
