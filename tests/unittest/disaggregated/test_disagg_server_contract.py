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
"""What a subclass of ``OpenAIDisaggServer`` outside this repository relies on.

NeMo-RL's ``OpenAIDisaggServerAdaptor`` subclasses ``OpenAIDisaggServer``. It
constructs the server as ``(config, req_timeout_secs=..., coordinator_url=None)``
plus one keyword argument of its own, adds a pure-ASGI middleware to
``self.app`` right after ``super().__init__()`` (so the app must exist with
every route registered when ``__init__`` returns), replaces ``__call__(host,
port, sockets=None)`` to keep a handle on its own uvicorn server, overrides
``_wrap_entry_point`` (swapping the request type and re-wrapping the returned
handler) and ``_handle_exception`` (answering a 4xx upstream error itself,
counting it on ``_perf_metrics_collector.http_exceptions``, delegating the
rest to ``super()``). These tests build a subclass of the same shape and pin
that surface. A failing test here means the change needs the NeMo-RL owners'
agreement first; see the ``OpenAIDisaggServer`` class docstring.
"""

import asyncio
import importlib
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from test_common.serve.app_factory import isolated_serve_env
from test_common.serve.routes import effective_routes

from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    RouterConfig,
    ServerRole,
)
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    ResponsesRequest,
    ResponsesResponse,
    UCompletionRequest,
    UsageInfo,
)

pytestmark = pytest.mark.cpu_only

# The import paths of the class. A move that keeps the old path as a shim adds
# the new path here and keeps the old one, so the contract holds through both.
CONTRACT_MODULES = ("tensorrt_llm.serve.openai_disagg_server",)


@pytest.fixture(params=CONTRACT_MODULES)
def server_cls(request):
    return importlib.import_module(request.param).OpenAIDisaggServer


def _contract(item: str) -> str:
    return (
        f"{item}: NeMo-RL subclasses OpenAIDisaggServer and overrides this; "
        "agree the change with the NeMo-RL owners first "
        "(see the OpenAIDisaggServer docstring)"
    )


class _HeaderMiddleware:
    """Pure ASGI, the way NeMo-RL installs its request adaptor."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-contract", b"1"))
            await send(message)

        await self.app(scope, receive, send_with_header)


def _contract_subclass(server_cls):
    """A subclass shaped like NeMo-RL's ``OpenAIDisaggServerAdaptor``."""

    class _Adaptor(server_cls):
        def __init__(self, *args, **kwargs):
            self.marker = kwargs.pop("marker")
            self.wrapped = []
            super().__init__(*args, **kwargs)
            self.app.add_middleware(_HeaderMiddleware)

        async def __call__(self, host, port, sockets=None):
            raise AssertionError("the contract tests never serve")

        def _handle_exception(self, exception):
            if isinstance(exception, aiohttp.ClientResponseError) and 400 <= exception.status < 500:
                self._perf_metrics_collector.http_exceptions.inc()
                raise HTTPException(status_code=exception.status, detail=exception.message)
            return super()._handle_exception(exception)

        def _wrap_entry_point(self, entry_point, request_type=UCompletionRequest):
            self.wrapped.append((entry_point, request_type))
            inner = super()._wrap_entry_point(entry_point, request_type)

            async def wrapper(req: request_type, raw_req: Request) -> Response:
                response = await inner(req, raw_req)
                if isinstance(response, JSONResponse):
                    response.headers["x-wrapped"] = "1"
                return response

            return wrapper

    return _Adaptor


def _disagg_server_config() -> DisaggServerConfig:
    """Two workers on closed loopback ports: construction is network-free."""
    return DisaggServerConfig(
        server_configs=[
            CtxGenServerConfig(type="ctx", hostname="127.0.0.1", port=9),
            CtxGenServerConfig(type="gen", hostname="127.0.0.1", port=10),
        ],
        hostname="127.0.0.1",
        port=0,
        ctx_router_config=RouterConfig(server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(server_role=ServerRole.GENERATION),
        node_id=0,
    )


def _build_adaptor(server_cls):
    """Construct the subclass with NeMo-RL's call shape."""
    config = _disagg_server_config()
    return _contract_subclass(server_cls)(
        config, req_timeout_secs=7, coordinator_url=None, marker=1
    )


def _raw_request() -> Request:
    return Request(
        {
            "type": "http",
            "headers": [],
            "state": {"server_arrival_time": 1.0, "perf_metrics_records": []},
        }
    )


def _run_through_subclass(coroutine):
    """Await a wrapped request built by the subclass.

    A TypeError here means the base changed how it calls what the subclass
    overrides: NeMo-RL's _handle_exception takes exactly one positional
    argument (the exception) and its wrapper takes (req, raw_req).
    """
    try:
        return asyncio.run(coroutine)
    except TypeError as error:
        raise AssertionError(
            _contract(
                "_handle_exception called with exactly one positional argument (the "
                "exception) and the wrapper called with (req, raw_req)"
            )
        ) from error


def _chat_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-contract-test",
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def test_constructor_takes_config_timeout_and_coordinator_url(tmp_path, server_cls):
    config = _disagg_server_config()
    try:
        inspect.signature(server_cls).bind(config, req_timeout_secs=7, coordinator_url=None)
    except TypeError as error:
        raise AssertionError(
            _contract(
                "the constructor call shape (config, req_timeout_secs=..., coordinator_url=None)"
            )
        ) from error
    with isolated_serve_env(tmp_path):
        server = _build_adaptor(server_cls)
        assert server.marker == 1, _contract(
            "a subclass keyword argument popped before super().__init__"
        )


def test_app_and_routes_exist_when_init_returns(tmp_path, server_cls):
    with isolated_serve_env(tmp_path):
        server = _build_adaptor(server_cls)
        assert isinstance(server.app, FastAPI), _contract("self.app set when __init__ returns")
        post_routes = {
            path: route
            for path, route in effective_routes(server.app)
            if "POST" in (getattr(route, "methods", None) or ())
        }
        for path in ("/v1/completions", "/v1/chat/completions"):
            assert path in post_routes, _contract(
                f"the {path} route registered when __init__ returns"
            )
            assert post_routes[path].endpoint.__module__ == __name__, _contract(
                f"the {path} endpoint is the wrapper the subclass returned"
            )
        wrapped_types = [request_type for _, request_type in server.wrapped]
        assert any(t is CompletionRequest for t in wrapped_types), _contract(
            "_wrap_entry_point called with CompletionRequest, compared by identity"
        )
        assert any(t is ChatCompletionRequest for t in wrapped_types), _contract(
            "_wrap_entry_point called with ChatCompletionRequest, compared by identity"
        )
        subclass_endpoints = [
            route
            for _, route in effective_routes(server.app)
            if getattr(route, "endpoint", None) is not None
            and route.endpoint.__module__ == __name__
        ]
        assert len(server.wrapped) == len(subclass_endpoints), _contract(
            "every route built through _wrap_entry_point goes through the override"
        )


def test_middleware_added_after_init_runs(tmp_path, server_cls):
    with isolated_serve_env(tmp_path) as env:
        server = _build_adaptor(server_cls)
        response = env.client(server.app, lifespan=False).get("/version")
        assert response.status_code == 200, _contract(
            "a request served by self.app after construction"
        )
        assert response.headers.get("x-contract") == "1", _contract(
            "middleware added to self.app right after super().__init__ runs"
        )


def test_wrapper_result_reaches_the_client(tmp_path, server_cls):
    with isolated_serve_env(tmp_path):
        server = _build_adaptor(server_cls)
        chat_response = _chat_response()
        wrapper = server._wrap_entry_point(
            AsyncMock(return_value=chat_response), ChatCompletionRequest
        )
        req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])

        response = _run_through_subclass(wrapper(req, _raw_request()))

        assert isinstance(response, JSONResponse), _contract(
            "the base wrapper's non-streaming result, re-wrappable by a subclass"
        )
        assert response.headers.get("x-wrapped") == "1", _contract(
            "the subclass wrapper around the base wrapper's result"
        )
        assert json.loads(bytes(response.body)) == chat_response.model_dump(by_alias=True), (
            _contract("the response body the base wrapper serializes")
        )


def test_every_failure_reaches_the_overridden_handle_exception(tmp_path, server_cls):
    with isolated_serve_env(tmp_path):
        server = _build_adaptor(server_cls)
        assert callable(server._perf_metrics_collector.http_exceptions.inc), _contract(
            "_perf_metrics_collector.http_exceptions.inc"
        )
        server._perf_metrics_collector = SimpleNamespace(
            http_exceptions=SimpleNamespace(inc=Mock()),
            internal_errors=SimpleNamespace(inc=Mock()),
            queue_latency_seconds=SimpleNamespace(observe=Mock()),
            total_requests=SimpleNamespace(inc=Mock()),
            stream_requests=SimpleNamespace(inc=Mock()),
            nonstream_requests=SimpleNamespace(inc=Mock()),
            total_responses=SimpleNamespace(inc=Mock()),
        )
        req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])

        upstream = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=413, message="too long"
            )
        )
        with pytest.raises(HTTPException) as exc_info:
            _run_through_subclass(
                server._wrap_entry_point(upstream, ChatCompletionRequest)(req, _raw_request())
            )
        assert exc_info.value.status_code == 413, _contract(
            "a 4xx upstream error answered by the overridden _handle_exception"
        )
        assert exc_info.value.detail == "too long", _contract(
            "the overridden _handle_exception's own error text, not the base's"
        )
        assert server._perf_metrics_collector.http_exceptions.inc.call_count == 1, _contract(
            "the override counting on _perf_metrics_collector.http_exceptions"
        )

        unexpected = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(HTTPException) as exc_info:
            _run_through_subclass(
                server._wrap_entry_point(unexpected, ChatCompletionRequest)(req, _raw_request())
            )
        assert exc_info.value.status_code == 500, _contract(
            "errors the override delegates reach super()._handle_exception"
        )
        assert exc_info.value.detail == "Internal server error boom", _contract(
            "super()._handle_exception's 500 for a delegated error"
        )
        assert server._perf_metrics_collector.internal_errors.inc.call_count == 1, _contract(
            "super()._handle_exception counting internal errors"
        )


def test_call_serves_host_port_sockets(server_cls):
    assert inspect.iscoroutinefunction(server_cls.__call__), _contract(
        "__call__ as the coroutine that serves"
    )
    parameters = inspect.signature(server_cls.__call__).parameters
    assert list(parameters) == ["self", "host", "port", "sockets"], _contract(
        "__call__(self, host, port, sockets)"
    )
    assert parameters["sockets"].default is None, _contract(
        "the sockets argument of __call__ defaulting to None"
    )


def test_responses_route_works_through_two_arg_wrap_override(tmp_path, server_cls, monkeypatch):
    class _ExtendedResponsesRequest(ResponsesRequest):
        required_marker: str

    responses_response = ResponsesResponse(
        model="m",
        output=[],
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        service_tier="auto",
        status="completed",
        top_logprobs=0,
        truncation="disabled",
    )

    async def stream_frames():
        yield b'event: response.created\ndata: {"type":"response.created","sequence_number":0}\n\n'
        yield (
            b"event: response.completed\ndata: "
            b'{"type":"response.completed","sequence_number":1}\n\n'
        )

    handler_box = {}
    seen_markers = []
    background_ran = []

    async def fake_service_responses(self, req, hooks):
        return await handler_box["fn"](req, hooks)

    monkeypatch.setattr(OpenAIDisaggregatedService, "openai_responses", fake_service_responses)

    class _ResponsesAdaptor(_contract_subclass(server_cls)):
        def _wrap_entry_point(self, entry_point, request_type=UCompletionRequest):
            effective_type = (
                _ExtendedResponsesRequest if request_type is ResponsesRequest else request_type
            )
            inner = super()._wrap_entry_point(entry_point, effective_type)

            async def wrapper(req: effective_type, raw_req: Request) -> Response:
                if isinstance(req, _ExtendedResponsesRequest):
                    seen_markers.append(req.required_marker)
                response = await inner(req, raw_req)
                if isinstance(response, StreamingResponse):
                    response.status_code = 202
                    response.headers["x-wrapped-stream"] = "1"
                    response.background = BackgroundTask(lambda: background_ran.append(True))
                return response

            return wrapper

    with isolated_serve_env(tmp_path) as env:
        server = _ResponsesAdaptor(
            _disagg_server_config(), req_timeout_secs=7, coordinator_url=None, marker=1
        )
        client = env.client(server.app, lifespan=False)

        # Missing subclass-required field must fail FastAPI validation on POST /v1/responses.
        missing = client.post("/v1/responses", json={"model": "m", "input": "hi"})
        assert missing.status_code == 400, _contract(
            "/v1/responses must validate against the request model swapped by _wrap_entry_point"
        )
        assert "required_marker" in missing.text

        # Non-streaming POST /v1/responses passes the swapped model and subclass header.
        handler_box["fn"] = AsyncMock(return_value=responses_response)
        non_stream = client.post(
            "/v1/responses",
            json={"model": "m", "input": "hi", "required_marker": "non-stream"},
        )
        assert non_stream.status_code == 200
        assert non_stream.headers.get("x-wrapped") == "1"
        assert seen_markers == ["non-stream"]

        # Fast-path streaming POST /v1/responses preserves status, header, body, and background.
        handler_box["fn"] = AsyncMock(side_effect=lambda req, hooks: stream_frames())
        fast_stream = client.post(
            "/v1/responses",
            json={"model": "m", "input": "hi", "stream": True, "required_marker": "fast"},
        )
        assert fast_stream.status_code == 202
        assert fast_stream.headers.get("x-wrapped-stream") == "1"
        assert "event: response.created" in fast_stream.text
        assert "event: response.completed" in fast_stream.text
        assert seen_markers == ["non-stream", "fast"]
        assert background_ran == [True]

        # Slow-path early-open streaming through the registered /v1/responses endpoint.
        background_ran.clear()
        disagg_mod = importlib.import_module(server_cls.__module__)
        monkeypatch.setattr(disagg_mod, "_RESPONSES_STREAM_START_WAIT_S", 0.0)
        responses_endpoint = next(
            route.endpoint
            for path, route in effective_routes(server.app)
            if path == "/v1/responses" and "POST" in (getattr(route, "methods", None) or ())
        )

        async def run_slow_stream():
            gate = asyncio.Event()

            async def delayed_responses(req, hooks):
                await gate.wait()
                return stream_frames()

            handler_box["fn"] = delayed_responses
            streamed = await responses_endpoint(
                _ExtendedResponsesRequest(
                    model="m", input="hi", stream=True, required_marker="slow"
                ),
                _raw_request(),
            )
            assert isinstance(streamed, StreamingResponse), _contract(
                "streaming /v1/responses through a two-argument _wrap_entry_point override"
            )
            messages = []

            async def send(message):
                messages.append(message)
                if message["type"] == "http.response.start":
                    gate.set()

            async def receive():
                await asyncio.Future()
                return {"type": "http.disconnect"}

            await streamed({"type": "http"}, receive, send)
            return messages

        slow_messages = _run_through_subclass(asyncio.wait_for(run_slow_stream(), timeout=10.0))
        slow_start = next(m for m in slow_messages if m["type"] == "http.response.start")
        assert slow_start["status"] == 202
        slow_headers = {k.lower(): v for k, v in slow_start["headers"]}
        assert slow_headers.get(b"x-wrapped-stream") == b"1"
        slow_body = b"".join(
            m.get("body", b"") for m in slow_messages if m["type"] == "http.response.body"
        )
        assert b"event: response.created" in slow_body and b"event: response.completed" in slow_body
        assert seen_markers == ["non-stream", "fast", "slow"]
        assert background_ran == [True]
