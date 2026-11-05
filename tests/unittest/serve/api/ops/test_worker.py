# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker endpoints through ``WorkerServing`` and its router.

Pins the collaborators the serving object shares with the composition root
(the clock the middlewares read, the chat serving object, the render
resources built once, the energy monitor the lifespan creates), what the
routes answer on the LLM roles built by the real composition root, and that
the other roles do not serve them.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig
from test_common.serve.routes import effective_routes

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.api.ops.worker import api_router
from tensorrt_llm.serve.api.ops.worker.serving import WorkerServing

# threadleak is off: the tests create TestClients inside the test body and the
# fake server's lazy worker-pool threads are alive when the leak check runs; the
# serve_env fixture closes them in its teardown, which runs after that check
# (precedent: tests/unittest/serve/api/ops/system/test_system_endpoints.py).
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REQUIRED = (
    "model",
    "generator",
    "adjusted_steady_clock",
    "shared_clock_offset",
    "chat_serving",
    "render_resources",
)
WORKER_PATHS = (
    "/health_generate",
    "/v1/data_transceiver_state",
    "/energy_metrics",
    "/steady_clock_offset",
    "/kv_cache_events",
    "/server_info",
)
WORKER_ROUTER_MODULE = "tensorrt_llm.serve.api.ops.worker.api_router"
LLM_ROLES = [None, ServerRole.CONTEXT, ServerRole.GENERATION]


def _dependencies():
    return {
        "model": "fake-model",
        "generator": SimpleNamespace(disaggregated_params=None, args=None),
        "adjusted_steady_clock": AdjustedSteadyClock(time_source=lambda: 100.0),
        "shared_clock_offset": None,
        "chat_serving": SimpleNamespace(openai_chat=AsyncMock()),
        "render_resources": SimpleNamespace(fingerprint=lambda: {"digest": "d"}),
    }


def test_constructor_requires_each_dependency():
    dependencies = _dependencies()
    # The control: with every dependency present, construction succeeds.
    WorkerServing(**dependencies)
    for name in REQUIRED:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            WorkerServing(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"WorkerServing must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )


def test_attach_router_requires_the_serving_object():
    with pytest.raises(RuntimeError, match=api_router.APP_STATE_KEY):
        api_router.attach_router(FastAPI())


def test_the_root_shares_its_objects_with_the_worker_serving(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    serving = getattr(server.app.state, api_router.APP_STATE_KEY)
    assert serving is server._worker_serving
    # The clock set_steady_clock_offset mutates is the one the middlewares and
    # the perf-metrics records read.
    assert serving._adjusted_steady_clock is server._adjusted_steady_clock
    assert serving._shared_clock_offset is server._shared_clock_offset
    # /health_generate drives the same ChatServing the chat router serves.
    assert serving._chat_serving is server._chat_serving
    # /server_info fingerprints the resources the root built once, and the
    # render routes read the same object.
    assert serving._render_resources is server._render_resources
    assert server.app.state.render_resources is server._render_resources


@pytest.mark.parametrize("role", LLM_ROLES, ids=["llm", "context", "generation"])
def test_routes_come_from_the_worker_router(role, serve_env):
    server = serve_env.build_openai_server(AggConfig(role=role))
    by_path = {path: [] for path in WORKER_PATHS}
    for path, route in effective_routes(server.app):
        if path in by_path:
            by_path[path].append(route)
    for path, routes in by_path.items():
        assert routes, f"{path} must be registered on the LLM roles"
        for route in routes:
            assert route.endpoint.__module__ == WORKER_ROUTER_MODULE, path
    # GET and POST /steady_clock_offset are two routes; every other path is one.
    assert len(by_path["/steady_clock_offset"]) == 2
    assert all(
        len(routes) == 1 for path, routes in by_path.items() if path != "/steady_clock_offset"
    )


def test_steady_clock_offset_moves_the_shared_clock(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app, lifespan=False)

    response = client.post("/steady_clock_offset", json={"offset": -7.25})

    assert response.status_code == 200
    # The offset is applied to the composition root's clock (the middlewares
    # stamp requests with it), not to a copy.
    assert server._adjusted_steady_clock.to_reference_time(0.0) == -7.25


def test_get_steady_clock_offset_samples_the_global_clock(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app, lifespan=False)
    global_clock = Mock(side_effect=[10.0, 10.2])

    # The handler moved: it looks the clock up in its own module.
    with patch(
        "tensorrt_llm.serve.api.ops.worker.serving.get_global_steady_clock_now_in_seconds",
        global_clock,
    ):
        response = client.get("/steady_clock_offset")

    assert response.status_code == 200
    assert response.json() == {"receive_ts": 10.0, "transmit_ts": 10.2}


def test_server_info_reports_the_fingerprint_of_the_injected_resources(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app, lifespan=False)

    response = client.get("/server_info")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["render_fingerprint"] == server._render_resources.fingerprint()
    assert body["max_batch_size"] == server.generator.args.max_batch_size
    assert body["disaggregated_params"] == {}
    assert body["startup_metrics"] == {}


def test_data_transceiver_state_is_404_without_state(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app, lifespan=False)

    response = client.get("/v1/data_transceiver_state")

    assert response.status_code == 404
    assert response.json() == {"error": "No transceiver state available"}


def test_energy_metrics_is_503_without_a_monitor(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app, lifespan=False)

    response = client.get("/energy_metrics")

    assert response.status_code == 503
    assert response.json() == {"error": "Energy monitoring is not available"}


class _FakeEnergyMonitor:
    def __init__(self, world_size):
        self.world_size = world_size

    def get_current_energy(self):
        return 12.5


def test_the_lifespan_hands_the_energy_monitor_to_the_worker_serving(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.generator.args.enable_energy_metrics = True

    # The lifespan creates the monitor (patched where the lifespan looks it
    # up: the composition root's module) and must hand it over before the
    # first request.
    with patch("tensorrt_llm.serve.openai_server.EnergyMonitor", _FakeEnergyMonitor):
        client = serve_env.client(server.app)
        response = client.get("/energy_metrics")

    assert response.status_code == 200, response.text
    assert response.json()["total_energy_j"] == 12.5
    assert isinstance(server.energy_monitor, _FakeEnergyMonitor)
    assert server._worker_serving.energy_monitor is server.energy_monitor


@pytest.mark.parametrize(
    "role", [ServerRole.MM_ENCODER, ServerRole.EMBEDDING], ids=["mm_encoder", "embedding"]
)
def test_worker_routes_are_not_served_on_other_roles(role, serve_env):
    server = serve_env.build_openai_server(AggConfig(role=role))
    assert server._worker_serving is None
    client = serve_env.client(server.app, lifespan=False)
    assert client.get("/server_info").status_code == 404
    assert client.get("/health_generate").status_code == 404
