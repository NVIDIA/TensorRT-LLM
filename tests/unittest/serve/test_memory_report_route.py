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
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_common.serve.app_factory import AggConfig

from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve.runtime_control_auth import build_runtime_control_auth_headers

pytestmark = pytest.mark.cpu_only

# The key the factory gives a server built with AggConfig(memory_report=True).
_SERVER_KEY = "test-key"
_ENV_KEY = "test-runtime-control-key"
_KEY_ENV = "TRTLLM_RUNTIME_CONTROL_API_KEY"


def _env(*extra, mode="1", key=_ENV_KEY):
    """``AggConfig.env`` that sets the monitor mode and the key through the environment."""
    env = [] if mode is None else [("TLLM_MEM_MONITOR", mode)]
    if key is not None:
        env.append((_KEY_ENV, key))
    return tuple(env) + extra


def _build(serve_env, config):
    server = serve_env.build_openai_server(config)
    factory = AsyncMock if config.rl_control else MagicMock
    server.generator.get_memory_report = factory(return_value=["rank 0 report"])
    return server, serve_env.client(server.app, lifespan=False)


def _get(client, key):
    headers = build_runtime_control_auth_headers(key, "GET", "/memory_report", b"")
    return client.get("/memory_report", headers=headers)


# rl_control=True builds an AsyncLLM generator; otherwise an LLM.
@pytest.mark.parametrize("asynchronous", [False, True], ids=["LLM", "AsyncLLM"])
@pytest.mark.parametrize(
    "from_config, key",
    [(False, _ENV_KEY), (True, _SERVER_KEY)],
    ids=["env-mode-and-key", "config-mode-and-server-key"],
)
def test_returns_each_workers_report(serve_env, asynchronous, from_config, key):
    config = AggConfig(
        env=() if from_config else _env(), memory_report=from_config, rl_control=asynchronous
    )
    _, client = _build(serve_env, config)
    response = _get(client, key)
    assert response.status_code == 200
    assert response.json() == {"reports": ["rank 0 report"]}


def test_requires_authentication(serve_env):
    _, client = _build(serve_env, AggConfig(memory_report=True))
    assert client.get("/memory_report").status_code == 401
    assert _get(client, "wrong-key").status_code == 401


@pytest.mark.parametrize("mode", [None, "oom", "0"])
def test_not_served_without_full_mode(serve_env, mode):
    _, client = _build(serve_env, AggConfig(env=_env(mode=mode)))
    assert _get(client, _ENV_KEY).status_code == 404


def test_not_served_without_a_key_and_says_why(serve_env, monkeypatch):
    logged = []
    monkeypatch.setattr(openai_server.logger, "info", logged.append)
    _, client = _build(serve_env, AggConfig(env=_env(key=None)))
    assert _get(client, _ENV_KEY).status_code == 404
    assert any("/memory_report is not served" in m and _KEY_ENV in m for m in logged), logged


def test_not_served_by_several_frontends(serve_env, monkeypatch):
    logged = []
    monkeypatch.setattr(openai_server.logger, "info", logged.append)
    _, client = _build(serve_env, AggConfig(memory_report=True, num_serve_frontends=2))
    assert _get(client, _SERVER_KEY).status_code == 404
    assert any("num_serve_frontends" in m for m in logged), logged


def test_a_rank_filter_does_not_remove_the_route(serve_env):
    # The frontend is rank 0 and serves every rank's report, so a rank filter
    # that leaves rank 0 out does not remove the route.
    _, client = _build(serve_env, AggConfig(env=_env(("TLLM_MEM_MONITOR_RANKS", "1"))))
    assert _get(client, _ENV_KEY).status_code == 200


def test_failure_is_an_internal_error(serve_env):
    server, client = _build(serve_env, AggConfig(memory_report=True))
    server.generator.get_memory_report.side_effect = RuntimeError("worker gone")
    response = _get(client, _SERVER_KEY)
    assert response.status_code == 500
    assert "worker gone" in response.json()["error"]["message"]


def test_concurrent_requests_share_one_gather(serve_env):
    server, _ = _build(serve_env, AggConfig(memory_report=True))
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
