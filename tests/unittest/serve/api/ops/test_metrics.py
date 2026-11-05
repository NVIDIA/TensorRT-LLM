# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GET /metrics`` through ``MetricsServing`` and its router, and the Prometheus mount.

Pins that the JSON route keeps winning over the Prometheus-text one when both
exist, that the composition root's collector loop feeds the serving object's
buffer, which roles serve ``/metrics``, and the collaborators the serving
object shares with the root.
"""

import asyncio
import dataclasses
import time
from collections import deque
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig
from test_common.serve.routes import effective_routes

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve._base.routing import ROUTERS, RoleGroup, ServerCapabilities
from tensorrt_llm.serve.api.ops.metrics import api_router
from tensorrt_llm.serve.api.ops.metrics.serving import MetricsServing

# threadleak is off: the tests create TestClients inside the test body and the
# fake server's lazy worker-pool threads are alive when the leak check runs; the
# serve_env fixture closes them in its teardown, which runs after that check
# (precedent: tests/unittest/serve/api/ops/system/test_system_endpoints.py).
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REQUIRED = (
    "model",
    "generator",
    "is_visual_gen",
    "metrics_collector",
    "iteration_stats_wakeup_event",
    "iteration_stats_poll_interval",
)
METRICS_ROUTER_MODULE = "tensorrt_llm.serve.api.ops.metrics.api_router"
JSON_FIRST = (
    "/metrics must stay the JSON route: in ROUTERS (tensorrt_llm/serve/_base/routing.py), "
    "tensorrt_llm.serve.api.ops.metrics.api_router must come before "
    "tensorrt_llm.serve.api.ops.metrics.prometheus, which must stay last"
)


def _dependencies():
    return {
        "model": "fake-model",
        "generator": SimpleNamespace(args=SimpleNamespace(enable_iter_perf_stats=False)),
        "is_visual_gen": False,
        "metrics_collector": None,
        "iteration_stats_wakeup_event": asyncio.Event(),
        "iteration_stats_poll_interval": None,
    }


def test_constructor_requires_each_dependency():
    dependencies = _dependencies()
    # The control: with every dependency present, construction succeeds.
    MetricsServing(**dependencies)._metrics_stats_executor.shutdown(wait=False)
    for name in REQUIRED:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            MetricsServing(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"MetricsServing must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )


def test_attach_router_requires_the_serving_object():
    with pytest.raises(RuntimeError, match=api_router.APP_STATE_KEY):
        api_router.attach_router(FastAPI())


def test_the_root_shares_its_objects_with_the_metrics_serving(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    serving = getattr(server.app.state, api_router.APP_STATE_KEY)
    assert serving is server._metrics_serving
    # The event the generate handlers set to wake the collector is the root's.
    assert serving._iteration_stats_wakeup_event is server._iteration_stats_wakeup_event
    assert serving.metrics_collector is server.metrics_collector
    assert serving._iteration_stats_poll_interval is server._iteration_stats_poll_interval
    assert serving._is_visual_gen is False


@pytest.mark.parametrize(
    "role",
    [None, ServerRole.CONTEXT, ServerRole.GENERATION, ServerRole.MM_ENCODER],
    ids=["llm", "context", "generation", "mm_encoder"],
)
def test_metrics_comes_from_the_metrics_router(role, serve_env):
    server = serve_env.build_openai_server(AggConfig(role=role))
    routes = [route for path, route in effective_routes(server.app) if path == "/metrics"]
    assert len(routes) == 1, "/metrics must be registered exactly once without perf metrics"
    assert routes[0].endpoint.__module__ == METRICS_ROUTER_MODULE
    client = serve_env.client(server.app, lifespan=False)

    response = client.get("/metrics")

    # Iteration stats are off in the fake arguments: no stats, no wait.
    assert response.status_code == 200
    assert response.json() == []


def test_metrics_is_not_served_on_the_embedding_role(serve_env):
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.EMBEDDING))
    assert not any(path == "/metrics" for path, _ in effective_routes(server.app))
    # The collector machinery still exists: the lifespan starts it for every role.
    assert server._metrics_serving is not None


def test_json_metrics_wins_with_perf_metrics_on(serve_env):
    server = serve_env.build_openai_server(
        AggConfig(return_perf_metrics=True, env=(("ENABLE_METRICS", "true"),))
    )
    metrics_routes = [route for path, route in effective_routes(server.app) if path == "/metrics"]
    assert len(metrics_routes) == 2, JSON_FIRST
    assert metrics_routes[0].endpoint.__module__ == METRICS_ROUTER_MODULE, JSON_FIRST
    client = serve_env.client(server.app, lifespan=False)

    response = client.get("/metrics")

    assert response.status_code == 200, JSON_FIRST
    assert response.headers["content-type"].startswith("application/json"), JSON_FIRST
    assert response.json() == []
    exposed = client.get("/prometheus/metrics")
    assert exposed.status_code == 200
    assert exposed.headers["content-type"].startswith("text/plain")


def test_the_collector_loop_feeds_the_serving_objects_buffer(serve_env):
    server = serve_env.build_openai_server(AggConfig(return_perf_metrics=True))
    server.generator.args.enable_iter_perf_stats = True
    stat = {"numActiveRequests": 1, "numQueuedRequests": 0}
    batches = deque([[stat]])
    server.generator.get_stats = lambda timeout: list(batches.popleft()) if batches else []

    # The lifespan starts the collector loop on the serving object's buffer.
    client = serve_env.client(server.app)

    assert server._iteration_stats_collector_task is not None
    assert server._metrics_serving._iteration_stats_buffer is not None
    deadline = time.monotonic() + 5
    body = client.get("/metrics").json()
    while not body and time.monotonic() < deadline:
        time.sleep(0.05)
        body = client.get("/metrics").json()
    assert body == [stat]
    # Served from the buffer, which a scrape clears.
    assert client.get("/metrics").json() == []


@pytest.mark.parametrize(
    "config, metrics_routes",
    [
        (
            AggConfig(
                role=ServerRole.MM_ENCODER,
                return_perf_metrics=True,
                env=(("ENABLE_METRICS", "true"),),
            ),
            1,
        ),
        (AggConfig(role=ServerRole.EMBEDDING, return_perf_metrics=True), 0),
    ],
    ids=["mm_encoder_perf_metrics_enable_metrics_env", "embedding_perf_metrics"],
)
def test_no_prometheus_mount_on_the_other_roles(config, metrics_routes, serve_env):
    # Perf metrics on a non-LLM role: the collector and the perf-metrics
    # middleware exist, but the Prometheus mount is LLM-only today, so
    # /prometheus/metrics, the second GET /metrics and the instrumentator's
    # middleware are absent. Adding the mount to a role is a route change:
    # update the router-table entry of tensorrt_llm.serve.api.ops.metrics.prometheus
    # in tensorrt_llm/serve/_base/routing.py and perf_metrics in
    # OpenAIServer._server_capabilities(), regenerate the route snapshot and
    # change this test on purpose.
    server = serve_env.build_openai_server(config)
    paths = [path for path, _ in effective_routes(server.app)]
    assert "/prometheus/metrics" not in paths, paths
    assert paths.count("/metrics") == metrics_routes, paths
    middleware = [middleware.cls.__name__ for middleware in server.app.user_middleware]
    assert "PrometheusInstrumentatorMiddleware" not in middleware, middleware
    # The mount is gated twice, and both gates must hold: the capability the
    # root computes (LLM role group only) and the role set of the router-table
    # entry (the test below).
    assert server._server_capabilities().perf_metrics is False, (
        "perf_metrics is an LLM-role-group capability: OpenAIServer._server_capabilities() "
        "must leave it False on the other roles"
    )


def test_the_prometheus_mount_is_an_llm_only_entry_and_the_last_one():
    spec = next(
        spec for spec in ROUTERS if spec.module == "tensorrt_llm.serve.api.ops.metrics.prometheus"
    )
    assert spec.roles == frozenset({RoleGroup.LLM}), (
        "the Prometheus mount is attached to the LLM role group only today; widening its "
        "router-table entry is a route change: regenerate the route snapshot and change this "
        "test on purpose"
    )
    assert spec is ROUTERS[-1], JSON_FIRST
    llm = ServerCapabilities.for_role(None)
    assert spec.when(llm) is False
    assert spec.when(dataclasses.replace(llm, perf_metrics=True)) is True
