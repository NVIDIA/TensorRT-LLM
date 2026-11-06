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
"""Middleware-order snapshot of the aggregated and disaggregated apps.

Starlette's ``add_middleware`` inserts at position 0, so the last one added is
outermost. ``ServerArrivalTimeMiddleware`` assigns ``scope['state'] = {}`` and
``PerfMetricsMiddleware`` only ``setdefault``s into it, so their relative order
is load-bearing: swapping them wipes every perf-metrics record. VisualGen is not
listed: it takes the default code path without perf metrics, and no VisualGen
path can change middleware registration.
"""

import pytest
from test_common.serve.app_factory import AggConfig, CoordinatorConfig, DisaggConfig
from test_common.serve.middleware import middleware_snapshot

from tensorrt_llm.llmapi.disagg_utils import ServerRole

pytestmark = pytest.mark.cpu_only

_ARRIVAL_WITH_CLOCK = {"class": "ServerArrivalTimeMiddleware", "kwargs": ["adjusted_clock"]}
_ARRIVAL_PLAIN = {"class": "ServerArrivalTimeMiddleware", "kwargs": []}


def _perf(kwargs, expose_headers):
    return {
        "class": "PerfMetricsMiddleware",
        "kwargs": sorted(kwargs),
        "expose_headers": expose_headers,
    }


# Outermost first, matching middleware_snapshot.
EXPECTED = {
    "llm": (AggConfig(), [_ARRIVAL_WITH_CLOCK]),
    "mm_encoder": (AggConfig(role=ServerRole.MM_ENCODER), [_ARRIVAL_WITH_CLOCK]),
    "embedding": (AggConfig(role=ServerRole.EMBEDDING), [_ARRIVAL_WITH_CLOCK]),
    "llm_perf_metrics": (
        AggConfig(return_perf_metrics=True),
        [
            _ARRIVAL_WITH_CLOCK,
            _perf(["adjusted_clock", "expose_headers", "writer"], True),
        ],
    ),
    "llm_perf_metrics_output_dir_only": (
        AggConfig(perf_metrics_output_dir=True),
        [
            _ARRIVAL_WITH_CLOCK,
            _perf(["adjusted_clock", "expose_headers", "writer"], False),
        ],
    ),
    "llm_perf_metrics_enable_metrics_env": (
        AggConfig(return_perf_metrics=True, env=(("ENABLE_METRICS", "true"),)),
        [
            _ARRIVAL_WITH_CLOCK,
            _perf(["adjusted_clock", "expose_headers", "writer"], True),
            {"class": "PrometheusInstrumentatorMiddleware"},
        ],
    ),
    "disagg_local": (DisaggConfig(), [_ARRIVAL_PLAIN]),
    "disagg_fleet": (DisaggConfig(fleet=True), [_ARRIVAL_PLAIN]),
    "disagg_local_perf_metrics": (
        DisaggConfig(return_perf_metrics=True),
        [_ARRIVAL_PLAIN, _perf(["expose_headers", "writer"], True)],
    ),
    "disagg_fleet_perf_metrics": (
        DisaggConfig(fleet=True, return_perf_metrics=True),
        [_ARRIVAL_PLAIN, _perf(["expose_headers", "writer"], True)],
    ),
    "coordinator": (CoordinatorConfig(), []),
}


def _build(serve_env, config):
    if isinstance(config, AggConfig):
        return serve_env.build_openai_server(config).app
    if isinstance(config, DisaggConfig):
        return serve_env.build_disagg_server(config).app
    return serve_env.build_coordinator_server(config).app


def _failure(name, expected, got):
    return (
        f"Middleware stack of '{name}' changed: expected {expected}, got {got}. "
        "Starlette's add_middleware inserts at position 0, so the last one added is "
        "outermost. ServerArrivalTimeMiddleware assigns scope['state'] = {} and must stay "
        "outside PerfMetricsMiddleware, or every perf-metrics record is wiped. Agg passes "
        "adjusted_clock to both, disagg to neither. If the change is intended, update "
        "EXPECTED in tests/unittest/serve/test_middleware_order.py and say why in the PR "
        "body."
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_middleware_order(name, serve_env):
    config, expected = EXPECTED[name]
    snapshot = middleware_snapshot(_build(serve_env, config))
    assert snapshot == expected, _failure(name, expected, snapshot)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_arrival_time_stays_outside_perf_metrics(name, serve_env):
    """When both are present, ServerArrivalTimeMiddleware is outermost."""
    config, _ = EXPECTED[name]
    classes = [entry["class"] for entry in middleware_snapshot(_build(serve_env, config))]
    if "PerfMetricsMiddleware" not in classes or "ServerArrivalTimeMiddleware" not in classes:
        return
    assert classes.index("ServerArrivalTimeMiddleware") < classes.index("PerfMetricsMiddleware"), (
        _failure(name, "ServerArrivalTimeMiddleware before PerfMetricsMiddleware", classes)
    )


def test_user_middleware_is_outermost(serve_env):
    """``--middleware`` classes are applied after construction, so they end up outermost."""
    from tensorrt_llm.commands.serve import _apply_fastapi_middlewares

    server = serve_env.build_openai_server(AggConfig())
    _apply_fastapi_middlewares(server.app, ["test_common.serve.fakes.UserProbeMiddleware"])
    snapshot = middleware_snapshot(server.app)
    expected = [{"class": "UserProbeMiddleware"}, _ARRIVAL_WITH_CLOCK]
    assert snapshot == expected, _failure("llm_user_middleware", expected, snapshot)
