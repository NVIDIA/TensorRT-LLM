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
"""``register_middlewares`` reproduces each app's middleware stack exactly.

The aggregated server passes ``adjusted_clock`` to both middlewares, the
disaggregated server to neither; ``--middleware`` entries are applied after the
server is built and stay outermost.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from starlette.testclient import TestClient
from test_common.serve.fakes import UserProbeMiddleware

from tensorrt_llm.serve.middleware.arrival_time import ServerArrivalTimeMiddleware
from tensorrt_llm.serve.middleware.perf_metrics import PerfMetricsMiddleware
from tensorrt_llm.serve.middleware.register import register_middlewares
from tensorrt_llm.serve.middleware.user_middleware import _apply_fastapi_middlewares
from tensorrt_llm.serve.perf_metrics import RETURN_METRICS_HEADER, SERVER_TIMING_HEADER

pytestmark = pytest.mark.cpu_only


def _record(status="complete"):
    # Copy of the record helper of tests/unittest/llmapi/apps/test_request_metrics.py:
    # arrival 1.0 and first_scheduled 1.01 give Server-Timing server_queue;dur=10.000000.
    return {
        "schema_version": 1,
        "request_id": "42",
        "status": status,
        "phases": {
            "server": {
                "timing_metrics": {
                    "arrival_time": 1.0,
                    "first_scheduled_time": 1.01,
                    "first_token_time": 1.02,
                    "last_token_time": 1.05,
                    "kv_cache_transfer_start": None,
                    "kv_cache_transfer_end": None,
                },
                "time_breakdown_metrics": {
                    "step_metrics": [
                        {
                            "iter": 3,
                            "forward_start_time": 2.0,
                            "forward_end_time": 2.002,
                            "sample_start_time": 2.002,
                            "sample_end_time": 2.003,
                            "gpu_forward_time": 1.5,
                            "gpu_sample_time": 0.5,
                        }
                    ],
                    "ctx_chunk_metrics": [
                        {
                            "forward_start_time": 1.0,
                            "forward_end_time": 1.004,
                            "sample_start_time": 1.004,
                            "sample_end_time": 1.005,
                            "gpu_forward_time": 3.0,
                            "gpu_sample_time": 0.25,
                        }
                    ],
                },
            }
        },
    }


def test_aggregated_stack_is_unchanged():
    app = FastAPI()
    w = SimpleNamespace(submit=Mock())
    c = SimpleNamespace(now=lambda: 123.25)
    register_middlewares(
        app,
        collect_perf_metrics=True,
        expose_perf_metrics=True,
        perf_metrics_writer=w,
        adjusted_clock=c,
    )
    assert [(m.cls, m.kwargs) for m in app.user_middleware] == [
        (ServerArrivalTimeMiddleware, {"adjusted_clock": c}),
        (PerfMetricsMiddleware, {"expose_headers": True, "writer": w, "adjusted_clock": c}),
    ], (
        "register_middlewares changed the aggregated stack; it must equal the two "
        "add_middleware calls OpenAIServer made before, in the same order and with the same "
        "keyword arguments"
    )


def test_disaggregated_stack_passes_no_clock():
    app = FastAPI()
    w = SimpleNamespace(submit=Mock())
    register_middlewares(
        app,
        collect_perf_metrics=True,
        expose_perf_metrics=True,
        perf_metrics_writer=w,
    )
    assert [(m.cls, m.kwargs) for m in app.user_middleware] == [
        (ServerArrivalTimeMiddleware, {}),
        (PerfMetricsMiddleware, {"expose_headers": True, "writer": w}),
    ], "the disaggregated app passes no adjusted_clock to either middleware"


def test_no_perf_metrics_installs_only_the_arrival_time_middleware():
    app = FastAPI()
    register_middlewares(
        app,
        collect_perf_metrics=False,
        expose_perf_metrics=False,
        perf_metrics_writer=None,
    )
    assert [m.cls for m in app.user_middleware] == [ServerArrivalTimeMiddleware]


def test_a_request_trace_sits_between_the_arrival_stamp_and_the_perf_metrics():
    from tensorrt_llm.serve.middleware.request_trace import (
        RequestTrace,
        RequestTraceMiddleware,
        RequestTraceWriter,
    )

    trace = RequestTrace(RequestTraceWriter(None))
    w = SimpleNamespace(submit=Mock())
    c = SimpleNamespace(now=lambda: 123.25)
    for adjusted_clock, arrival_kwargs, perf_kwargs in (
        (c, {"adjusted_clock": c}, {"expose_headers": True, "writer": w, "adjusted_clock": c}),
        (None, {}, {"expose_headers": True, "writer": w}),
    ):
        app = FastAPI()
        register_middlewares(
            app,
            collect_perf_metrics=True,
            expose_perf_metrics=True,
            perf_metrics_writer=w,
            adjusted_clock=adjusted_clock,
            request_trace=trace,
        )
        assert [(m.cls, m.kwargs) for m in app.user_middleware] == [
            (ServerArrivalTimeMiddleware, arrival_kwargs),
            (RequestTraceMiddleware, {"trace": trace}),
            (PerfMetricsMiddleware, perf_kwargs),
        ], (
            "the trace runs after the arrival stamp, which its request line records, and "
            "wraps PerfMetricsMiddleware, whose metrics event its response line records"
        )


def test_user_middleware_is_outermost():
    app = FastAPI()
    w = SimpleNamespace(submit=Mock())
    c = SimpleNamespace(now=lambda: 123.25)
    register_middlewares(
        app,
        collect_perf_metrics=True,
        expose_perf_metrics=True,
        perf_metrics_writer=w,
        adjusted_clock=c,
    )
    _apply_fastapi_middlewares(app, ["test_common.serve.fakes.UserProbeMiddleware"])
    assert [m.cls for m in app.user_middleware] == [
        UserProbeMiddleware,
        ServerArrivalTimeMiddleware,
        PerfMetricsMiddleware,
    ], "--middleware entries must wrap the built-in middleware"


def test_records_reach_the_writer_and_headers():
    app = FastAPI()
    writer = SimpleNamespace(submit=Mock())
    record = _record()
    register_middlewares(
        app,
        collect_perf_metrics=True,
        expose_perf_metrics=True,
        perf_metrics_writer=writer,
    )

    @app.get("/probe")
    async def probe(request: Request) -> dict:
        request.state.perf_metrics_records.append(record)
        return {}

    with TestClient(app) as client:
        response = client.get("/probe", headers={RETURN_METRICS_HEADER: "1"})
    assert response.status_code == 200
    assert "server_queue;dur=10.000000" in response.headers.get(SERVER_TIMING_HEADER, ""), (
        "PerfMetricsMiddleware no longer sees the records handlers append to "
        "request.state.perf_metrics_records; check that both middlewares share scope['state'] "
        "and that their order is unchanged"
    )
    writer.submit.assert_called_once_with(record)


def test_arrival_time_is_set():
    app = FastAPI()
    register_middlewares(
        app,
        collect_perf_metrics=False,
        expose_perf_metrics=False,
        perf_metrics_writer=None,
        adjusted_clock=SimpleNamespace(now=lambda: 123.25),
    )

    @app.get("/probe")
    async def probe(request: Request) -> dict:
        return {"server_arrival_time": request.state.server_arrival_time}

    with TestClient(app) as client:
        response = client.get("/probe")
    assert response.json() == {"server_arrival_time": 123.25}


_SETDEFAULT_FAILURE = (
    "ServerArrivalTimeMiddleware replaced scope['state']; it must use setdefault, so state "
    "set by outer middleware or the lifespan survives"
)


class _OuterStateMiddleware:
    """Pure ASGI middleware that writes into the request state before the built-ins run."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})["outer"] = 1
        await self.app(scope, receive, send)


def test_state_from_an_outer_middleware_survives():
    app = FastAPI()
    register_middlewares(
        app,
        collect_perf_metrics=False,
        expose_perf_metrics=False,
        perf_metrics_writer=None,
    )
    app.add_middleware(_OuterStateMiddleware)

    @app.get("/probe")
    async def probe(request: Request) -> dict:
        return {
            "outer": request.state.outer,
            "has_arrival_time": hasattr(request.state, "server_arrival_time"),
        }

    with TestClient(app) as client:
        response = client.get("/probe")
    assert response.status_code == 200 and response.json() == {
        "outer": 1,
        "has_arrival_time": True,
    }, _SETDEFAULT_FAILURE


def test_lifespan_state_survives():
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        yield {"from_lifespan": 1}

    app = FastAPI(lifespan=lifespan)
    register_middlewares(
        app,
        collect_perf_metrics=False,
        expose_perf_metrics=False,
        perf_metrics_writer=None,
    )

    @app.get("/probe")
    async def probe(request: Request) -> dict:
        return {"from_lifespan": request.state.from_lifespan}

    with TestClient(app) as client:
        response = client.get("/probe")
    assert response.status_code == 200 and response.json() == {"from_lifespan": 1}, (
        _SETDEFAULT_FAILURE
    )
