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

The route exists only while the GPU memory monitor is on.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm import LLM
from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


def _server(monkeypatch, *, monitor: bool, asynchronous: bool = False):
    if monitor:
        monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    else:
        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    server.generator = object.__new__(AsyncLLM if asynchronous else LLM)
    factory = AsyncMock if asynchronous else MagicMock
    server.generator.get_memory_report = factory(return_value=["rank 0 report"])
    server._register_memory_report_route()
    return server


@pytest.mark.parametrize("asynchronous", [False, True])
def test_returns_each_workers_report(monkeypatch, asynchronous):
    server = _server(monkeypatch, monitor=True, asynchronous=asynchronous)
    with TestClient(server.app) as client:
        response = client.get("/memory_report")
    assert response.status_code == 200
    assert response.json() == {"reports": ["rank 0 report"]}


def test_absent_when_the_monitor_is_off(monkeypatch):
    # Off costs nothing and exposes nothing; sleep mode is not involved.
    server = _server(monkeypatch, monitor=False)
    with TestClient(server.app) as client:
        assert client.get("/memory_report").status_code == 404


def test_failure_is_an_internal_error(monkeypatch):
    server = _server(monkeypatch, monitor=True)
    server.generator.get_memory_report.side_effect = RuntimeError("worker gone")
    with TestClient(server.app) as client:
        response = client.get("/memory_report")
    assert response.status_code == 500
    assert "worker gone" in response.json()["error"]["message"]
