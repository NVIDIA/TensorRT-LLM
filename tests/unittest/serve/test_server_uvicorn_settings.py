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
"""The uvicorn settings each server hands to ``serve_uvicorn`` when it is called.

The two servers differ on purpose: the aggregated server logs at ``"info"``, keeps
connections alive for ``TIMEOUT_KEEP_ALIVE`` seconds and schedules the worker
registration next to the server; the disaggregated frontend logs at the TRT-LLM
logger's level, uses its configured keep-alive and schedules nothing. Both servers are
built through the real constructors; ``create_uvicorn_server`` is replaced where
``serve_uvicorn`` looks it up, so no port is bound.

The last test pins what ``build_http_app`` must carry for each app: the aggregated
server registers its API routes through the msgpack-decoding ``_MsgspecRoute`` and
the disaggregated frontend through ``KvHintsRoute``, and both read
``TRTLLM_TRUST_KV_HINTS`` into ``app.state.trust_kv_hints``.
"""

import asyncio
import os
import socket
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute
from test_common.serve.app_factory import AggConfig, DisaggConfig

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base import app_builder
from tensorrt_llm.serve._base.msgspec_route import _MsgspecRoute
from tensorrt_llm.serve.kv_hints import KvHintsRoute
from tensorrt_llm.serve.openai_server import TIMEOUT_KEEP_ALIVE

pytestmark = pytest.mark.cpu_only


class _FakeUvicorn:
    """Records the ``uvicorn.Config`` and the tasks ``serve_uvicorn`` creates."""

    def __init__(self, monkeypatch):
        self.configs = []
        self.serve_sockets = []
        self.task_names = []
        self._tasks = []
        self.server = SimpleNamespace(serve=self._serve, started=True, should_exit=False)
        monkeypatch.setattr(app_builder, "create_uvicorn_server", self._create)
        monkeypatch.setattr(app_builder, "asyncio", SimpleNamespace(create_task=self._create_task))

    def _create(self, config):
        self.configs.append(config)
        return self.server

    def _create_task(self, coroutine):
        self.task_names.append(coroutine.cr_code.co_name)
        task = asyncio.create_task(coroutine)
        self._tasks.append(task)
        return task

    async def _serve(self, sockets=None):
        self.serve_sockets.append(sockets)
        # A task started next to the server runs while the server serves.
        if self._tasks:
            await asyncio.gather(*self._tasks)


def test_aggregated_server_uvicorn_settings(serve_env, monkeypatch):
    server = serve_env.build_openai_server(AggConfig())
    fake = _FakeUvicorn(monkeypatch)
    with socket.socket() as sock:
        asyncio.run(server("127.0.0.1", 8123, sockets=[sock]))
        assert fake.serve_sockets == [[sock]]
    (config,) = fake.configs
    assert config.app is server.app
    assert (config.host, config.port) == ("127.0.0.1", 8123)
    assert config.log_level == "info", "the aggregated server runs uvicorn at log level info"
    assert config.timeout_keep_alive == TIMEOUT_KEEP_ALIVE, (
        "the aggregated server keeps connections alive for TIMEOUT_KEEP_ALIVE seconds"
    )
    assert fake.task_names == ["_register_after_serving"], (
        "the aggregated server schedules _register_after_serving next to the uvicorn server"
    )
    assert (server.binding_addr, server.host, server.port) == (
        "http://127.0.0.1:8123",
        "127.0.0.1",
        8123,
    )


def test_disaggregated_server_uvicorn_settings(serve_env, monkeypatch):
    server = serve_env.build_disagg_server(DisaggConfig())
    fake = _FakeUvicorn(monkeypatch)
    asyncio.run(server("127.0.0.1", 8123))
    assert fake.serve_sockets == [None]
    (config,) = fake.configs
    assert config.app is server.app
    assert (config.host, config.port) == ("127.0.0.1", 8123)
    assert config.log_level == logger.level, (
        "the disaggregated frontend runs uvicorn at the TRT-LLM logger's level, not at info"
    )
    assert config.timeout_keep_alive == server._config.server_keep_alive_timeout, (
        "the disaggregated frontend uses the configured server_keep_alive_timeout"
    )
    assert fake.task_names == [], "the disaggregated frontend schedules no post-start task"


@pytest.mark.parametrize("trust", [False, True], ids=["untrusted-ingress", "trusted-ingress"])
def test_route_class_and_kv_hint_policy_survive_the_assembly(serve_env, trust):
    """Each app keeps its route class and its KV-hint trust flag through build_http_app.

    The aggregated server decodes msgpack bodies through ``_MsgspecRoute`` (which also
    applies the KV-hint policy); the disaggregated frontend applies the policy through
    ``KvHintsRoute``. A plain ``APIRoute`` on either app would let ``kv_hints`` through
    from an untrusted ingress. ``TRTLLM_TRUST_KV_HINTS=1`` turns the trust on; the
    environment variable is restored at the serve environment's exit.
    """
    if trust:
        os.environ["TRTLLM_TRUST_KV_HINTS"] = "1"
    else:
        os.environ.pop("TRTLLM_TRUST_KV_HINTS", None)
    aggregated = serve_env.build_openai_server(AggConfig()).app
    disaggregated = serve_env.build_disagg_server(DisaggConfig()).app
    assert aggregated.router.route_class is _MsgspecRoute
    assert disaggregated.router.route_class is KvHintsRoute
    for name, app, route_class in (
        ("aggregated", aggregated, _MsgspecRoute),
        ("disaggregated", disaggregated, KvHintsRoute),
    ):
        api_routes = [route for route in app.routes if isinstance(route, APIRoute)]
        assert api_routes, f"the {name} app registers API routes"
        assert {type(route) for route in api_routes} == {route_class}, (
            f"every API route of the {name} app must be a {route_class.__name__}: "
            f"{sorted({type(route).__name__ for route in api_routes})}"
        )
        assert app.state.trust_kv_hints is trust, (
            f"the {name} app reads TRTLLM_TRUST_KV_HINTS into app.state.trust_kv_hints"
        )
