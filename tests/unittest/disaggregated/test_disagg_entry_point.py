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
"""Behavior of the disaggregated server's entry-point wrapper, pinned.

``OpenAIDisaggServer._wrap_entry_point`` turns a service entry point into the
FastAPI handler of the OpenAI routes: it counts the request, applies the
chat-template permission, resolves the conversation identity from the headers,
builds the per-request hooks, runs the entry point, counts the response,
serializes it, and sends every failure through ``_handle_exception`` while the
exception is being handled. These tests pin that behavior - the per-request
counters, the status codes, the serialization and the single trace span - so
the wrapper's internals can be reorganized without changing what a client or
a metrics scrape observes.
"""

import asyncio
import contextlib
import json
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.responses import ResponseFormatTextJSONSchemaConfig, ResponseTextConfig
from opentelemetry.trace import SpanKind
from starlette.requests import Request
from test_common.serve.app_factory import DisaggConfig, isolated_serve_env

import tensorrt_llm.serve.openai_disagg_server as disagg_server_module
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.llmapi import tracing
from tensorrt_llm.serve.openai_disagg_server import RawRequestResponseHooks
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

COUNTER_NAMES = (
    "total_requests",
    "stream_requests",
    "nonstream_requests",
    "validation_exceptions",
    "http_exceptions",
    "internal_errors",
    "total_responses",
)


def _mock_collector() -> SimpleNamespace:
    """The seven server counters and the queue-latency histogram, as mocks."""
    return SimpleNamespace(
        **{name: SimpleNamespace(inc=Mock()) for name in COUNTER_NAMES},
        queue_latency_seconds=SimpleNamespace(observe=Mock()),
    )


def _server_with_counters(env):
    server = env.build_disagg_server(DisaggConfig())
    counters = _mock_collector()
    server._perf_metrics_collector = counters
    return server, counters


def _assert_counts(counters: SimpleNamespace, case: str, **expected: int) -> None:
    for name in COUNTER_NAMES:
        actual = getattr(counters, name).inc.call_count
        assert actual == expected.get(name, 0), (
            f"the {name} counter changed for {case}; _wrap_entry_point and its "
            "steps must keep the request accounting exactly"
        )


def _raw_request(*headers: tuple) -> Request:
    return Request(
        {
            "type": "http",
            "headers": list(headers),
            "state": {"server_arrival_time": 1.0, "perf_metrics_records": []},
        }
    )


def _chat_request(**overrides) -> ChatCompletionRequest:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


def _chat_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-entry-point-test",
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


def test_non_streaming_request_is_counted_and_serialized(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)
        chat_response = _chat_response()
        entry_point = AsyncMock(return_value=chat_response)
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)
        req = _chat_request()
        raw_req = _raw_request()

        response = asyncio.run(wrapper(req, raw_req))

        assert isinstance(response, JSONResponse)
        assert json.loads(bytes(response.body)) == chat_response.model_dump(by_alias=True)
        entry_point.assert_awaited_once()
        called_req, called_hooks = entry_point.await_args.args
        assert called_req is req
        assert isinstance(called_hooks, RawRequestResponseHooks)
        assert called_hooks.raw_req is raw_req
        _assert_counts(
            counters,
            "a non-streaming chat request",
            total_requests=1,
            nonstream_requests=1,
            total_responses=1,
        )


def test_streaming_request_returns_the_generator_unchanged(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)

        async def generated():
            yield b"data: [DONE]\n\n"

        generator = generated()
        entry_point = AsyncMock(return_value=generator)
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        response = asyncio.run(wrapper(_chat_request(stream=True), _raw_request()))

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert response.body_iterator is generator
        _assert_counts(
            counters,
            "a streaming chat request",
            total_requests=1,
            stream_requests=1,
            total_responses=1,
        )


def test_forbidden_chat_template_is_a_400_before_any_worker(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)
        entry_point = AsyncMock()
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(wrapper(_chat_request(chat_template="x"), _raw_request()))

        assert exc_info.value.status_code == 400
        entry_point.assert_not_awaited()
        _assert_counts(
            counters,
            "a request with a forbidden chat template",
            total_requests=1,
            nonstream_requests=1,
            http_exceptions=1,
        )


def test_conversation_id_comes_from_the_headers(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, _ = _server_with_counters(env)
        entry_point = AsyncMock(return_value=_chat_response())
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        asyncio.run(wrapper(_chat_request(), _raw_request((b"x-session-id", b"s"))))

        (called_req, _), _ = entry_point.await_args
        assert called_req.conversation_params is not None
        assert called_req.conversation_params.conversation_id == "s"


def test_upstream_error_keeps_its_status(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)
        entry_point = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=404, message='{"message": "nope"}'
            )
        )
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(wrapper(_chat_request(), _raw_request()))

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "nope"
        _assert_counts(
            counters,
            "an upstream worker error",
            total_requests=1,
            nonstream_requests=1,
            http_exceptions=1,
        )


def test_context_length_error_returns_the_worker_envelope(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)
        worker_body = OpenAIServer.create_error_response(
            context_length_exceeded_message(2048, 4096)
        ).body.decode()
        # The error the way openai_client raises one: post_json/_send_request
        # use message=f"{reason}: {body[:2048]}" with the worker's response text.
        entry_point = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=400,
                message=f"Bad Request: {worker_body}",
            )
        )
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        response = asyncio.run(wrapper(_chat_request(), _raw_request()))

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        assert json.loads(bytes(response.body))["code"] == "context_length_exceeded"
        _assert_counts(
            counters,
            "a context-length rejection",
            total_requests=1,
            nonstream_requests=1,
            http_exceptions=1,
        )


def test_engine_fatal_error_returns_none_after_sigint(tmp_path, monkeypatch):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)
        recorder = Mock()
        monkeypatch.setattr(signal, "raise_signal", recorder)
        monkeypatch.setattr(disagg_server_module, "record_termination_observation", Mock())
        entry_point = AsyncMock(side_effect=CppExecutorError("x"))
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        response = asyncio.run(wrapper(_chat_request(), _raw_request()))

        assert response is None
        recorder.assert_called_once_with(signal.SIGINT)
        _assert_counts(
            counters,
            "an engine fatal error",
            total_requests=1,
            nonstream_requests=1,
        )


def test_unexpected_error_is_a_500(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)
        entry_point = AsyncMock(side_effect=RuntimeError("boom"))
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(wrapper(_chat_request(), _raw_request()))

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal server error boom"
        _assert_counts(
            counters,
            "an unexpected error",
            total_requests=1,
            nonstream_requests=1,
            internal_errors=1,
        )


def test_serialization_failure_goes_to_the_error_path(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)

        class _ExplodingResponse:
            def model_dump(self, **kwargs):
                raise RuntimeError("cannot serialize")

        entry_point = AsyncMock(return_value=_ExplodingResponse())
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(wrapper(_chat_request(), _raw_request()))

        assert exc_info.value.status_code == 500
        _assert_counts(
            counters,
            "a response that fails to serialize",
            total_requests=1,
            nonstream_requests=1,
            total_responses=1,
            internal_errors=1,
        )


class _FakeTracer:
    def __init__(self):
        self.spans = []

    @contextlib.contextmanager
    def start_as_current_span(self, name, kind=None, context=None):
        self.spans.append((name, kind))
        yield SimpleNamespace(set_status=Mock(), record_exception=Mock())


def test_one_trace_span_per_request(tmp_path, monkeypatch):
    with isolated_serve_env(tmp_path) as env:
        server, _ = _server_with_counters(env)
        fake = _FakeTracer()
        monkeypatch.setattr(tracing, "global_otlp_tracer", lambda: fake)
        entry_point = AsyncMock(return_value=_chat_response())
        wrapper = server._wrap_entry_point(entry_point, ChatCompletionRequest)

        asyncio.run(wrapper(_chat_request(), _raw_request()))
        assert fake.spans == [("disaggregated_request", SpanKind.SERVER)]

        with pytest.raises(HTTPException):
            asyncio.run(wrapper(_chat_request(chat_template="x"), _raw_request()))
        assert fake.spans == [("disaggregated_request", SpanKind.SERVER)] * 2


def test_anthropic_messages_still_counts_like_chat(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)
        server._service = SimpleNamespace(
            openai_chat_completion=AsyncMock(return_value=_chat_response())
        )
        client = env.client(server.app, lifespan=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "m",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "hello"
        _assert_counts(
            counters,
            "an Anthropic Messages request",
            total_requests=1,
            nonstream_requests=1,
            total_responses=1,
        )


def test_non_streaming_responses_request_serializes_by_alias(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)
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
            text=ResponseTextConfig(
                format=ResponseFormatTextJSONSchemaConfig(
                    name="shape", schema={"type": "object"}, type="json_schema"
                )
            ),
            top_logprobs=0,
            truncation="disabled",
        )
        entry_point = AsyncMock(return_value=responses_response)
        wrapper = server._wrap_entry_point(entry_point, ResponsesRequest)

        response = asyncio.run(wrapper(ResponsesRequest(model="m", input="hi"), _raw_request()))

        assert isinstance(response, JSONResponse)
        body = json.loads(bytes(response.body))
        # by_alias serialization: text.format.schema goes out under its wire
        # name, not the Python field name schema_.
        assert body["text"]["format"]["schema"] == {"type": "object"}
        assert "schema_" not in body["text"]["format"]
        assert body == responses_response.model_dump(by_alias=True)
        _assert_counts(
            counters,
            "a non-streaming Responses request",
            total_requests=1,
            nonstream_requests=1,
            total_responses=1,
        )


def test_streamed_responses_request_is_guarded(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server, counters = _server_with_counters(env)

        async def generated():
            yield b"data: {}\n\n"

        generator = generated()
        entry_point = AsyncMock(return_value=generator)
        wrapper = server._wrap_entry_point(entry_point, ResponsesRequest)
        req = ResponsesRequest(model="m", input="hi", stream=True)

        response = asyncio.run(wrapper(req, _raw_request()))

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        # A relayed Responses stream is wrapped so that an early failure ends
        # with an `error` event; the raw generator must not go out as-is.
        assert response.body_iterator is not generator
        _assert_counts(
            counters,
            "a streamed Responses request",
            total_requests=1,
            stream_requests=1,
            total_responses=1,
        )
