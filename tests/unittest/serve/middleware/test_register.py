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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from starlette.testclient import TestClient
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from test_common.serve.fakes import UserProbeMiddleware

from tensorrt_llm._utils import AdjustedSteadyClock
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


@pytest.mark.parametrize("has_observer", [False, True])
def test_aggregated_stack_is_unchanged(has_observer):
    app = FastAPI()
    w = SimpleNamespace(submit=Mock())
    c = SimpleNamespace(now=lambda: 123.25)
    observer = Mock() if has_observer else None
    register_middlewares(
        app,
        collect_perf_metrics=True,
        expose_perf_metrics=True,
        perf_metrics_writer=w,
        adjusted_clock=c,
        input_processing_observer=observer,
    )
    assert [(m.cls, m.kwargs) for m in app.user_middleware] == [
        (ServerArrivalTimeMiddleware, {"adjusted_clock": c}),
        (
            PerfMetricsMiddleware,
            {
                "expose_headers": True,
                "writer": w,
                "adjusted_clock": c,
                "input_processing_observer": observer,
            },
        ),
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing_state",
    [None, {}, {"from_outer": 1, "from_lifespan": 2, "server_arrival_time": -1}],
    ids=["absent", "empty", "populated"],
)
async def test_arrival_replaces_http_state_with_a_fresh_timestamp_dict(
    existing_state: dict[str, int] | None,
) -> None:
    scope: Scope = {"type": "http"}
    original = dict(existing_state) if existing_state is not None else None
    if existing_state is not None:
        scope["state"] = existing_state
    seen = []

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    async def app(current: Scope, current_receive: Receive, current_send: Send) -> None:
        assert current is scope and current_receive is receive and current_send is send
        seen.append(current["state"])

    middleware = ServerArrivalTimeMiddleware(
        app, adjusted_clock=AdjustedSteadyClock(time_source=lambda: 123.25)
    )
    await middleware(scope, receive, send)
    assert seen == [{"server_arrival_time": 123.25}]
    assert seen[0] is not existing_state
    assert existing_state == original


def test_outer_and_lifespan_state_do_not_reach_the_handler() -> None:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, int]]:
        yield {"from_lifespan": 1}

    incoming = []

    class OuterStateMiddleware:
        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                scope["state"]["from_outer"] = 2
                incoming.append(scope["state"])
            await self.app(scope, receive, send)

    app = FastAPI(lifespan=lifespan)
    register_middlewares(
        app,
        collect_perf_metrics=False,
        adjusted_clock=AdjustedSteadyClock(time_source=lambda: 123.25),
    )
    app.add_middleware(OuterStateMiddleware)

    @app.get("/probe")
    async def probe(request: Request) -> dict:
        return dict(request.scope["state"])

    with TestClient(app) as client:
        response = client.get("/probe")
    assert response.status_code == 200
    assert response.json() == {"server_arrival_time": 123.25}
    assert incoming == [{"from_lifespan": 1, "from_outer": 2}]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/probe", "/health_generate"])
@pytest.mark.parametrize("fails", [False, True])
async def test_input_observer_is_request_local_and_skips_liveness(path, fails):
    from tensorrt_llm.metrics.perf_utils import _input_processing_observer

    app = FastAPI()
    writer = SimpleNamespace(submit=Mock())
    observer = Mock()
    outer_observer = Mock()
    register_middlewares(
        app,
        collect_perf_metrics=True,
        expose_perf_metrics=True,
        perf_metrics_writer=writer,
        adjusted_clock=SimpleNamespace(now=lambda: 123.25),
        input_processing_observer=observer,
    )
    seen = []

    @app.get(path)
    async def probe(request: Request) -> dict:
        current = _input_processing_observer.get()
        assert current is not None and current is not outer_observer
        current("processor", 0.125, None)
        seen.append(dict(request.state.input_processing_metrics))
        assert request.state.perf_metrics_records == []
        if fails:
            raise ValueError("input failed")
        return {}

    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "query_string": b"",
        "headers": [(RETURN_METRICS_HEADER.encode(), b"1")],
    }
    token = _input_processing_observer.set(outer_observer)
    try:
        if fails:
            with pytest.raises(ValueError, match="input failed"):
                await app(scope, receive, send)
        else:
            await app(scope, receive, send)
        assert _input_processing_observer.get() is outer_observer
    finally:
        _input_processing_observer.reset(token)
    assert seen == [{"processor": 0.125}]
    if path == "/health_generate":
        observer.assert_not_called()
    else:
        observer.assert_called_once_with("processor", 0.125, None)
    outer_observer.assert_not_called()
    writer.submit.assert_not_called()
    starts = [message for message in messages if message["type"] == "http.response.start"]
    assert len(starts) == 1
    assert not any(
        name.lower() == SERVER_TIMING_HEADER.lower().encode() for name, _ in starts[0]["headers"]
    )
