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
"""``AnthropicMessagesServing`` over a fake ``ChatBackend``: what reaches the answer and how."""

import asyncio
import json
import logging

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from tensorrt_llm.logger import logger as trtllm_logger
from tensorrt_llm.serve._base.errors import EngineFatalError, ServingError
from tensorrt_llm.serve._base.request_context import RequestContext
from tensorrt_llm.serve.api.anthropic import serving as serving_module
from tensorrt_llm.serve.api.anthropic.protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.api.anthropic.serving import AnthropicMessagesServing, _error_type
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ErrorResponse,
    FunctionCall,
    ToolCall,
    UsageInfo,
)

pytestmark = pytest.mark.cpu_only

CLIENT_MODEL = "client-model"
RESPONSE_MODEL = "response-model"


class _Backend:
    """A ChatBackend that answers with a fixed result or raises a fixed error."""

    def __init__(self, *, result=None, error=None, model="x-model", message="x-message"):
        self.result = result
        self.error = error
        self.model = model
        self.message = message
        self.calls = []

    async def create_chat_completion(self, request, ctx):
        self.calls.append((request, ctx))
        if self.error is not None:
            raise self.error
        return self.result

    def stream_model(self, request):
        return self.model

    def error_message(self, error):
        return self.message


def _request(**overrides) -> AnthropicMessagesRequest:
    payload = {
        "model": CLIENT_MODEL,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return AnthropicMessagesRequest(**payload)


def _chat_response(*, tool_arguments=None) -> ChatCompletionResponse:
    tool_calls = []
    if tool_arguments is not None:
        tool_calls = [
            ToolCall(
                id="call_1", function=FunctionCall(name="get_weather", arguments=tool_arguments)
            )
        ]
    return ChatCompletionResponse(
        id="chatcmpl-serving-test",
        model=RESPONSE_MODEL,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello", tool_calls=tool_calls),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def _frames(*texts: str) -> list:
    chunks = [
        ChatCompletionStreamResponse(
            model=RESPONSE_MODEL, choices=[{"index": 0, "delta": {"content": text}}]
        )
        for text in texts
    ]
    return [f"data: {chunk.model_dump_json()}\n\n" for chunk in chunks] + ["data: [DONE]\n\n"]


async def _stream(frames: list):
    for frame in frames:
        yield frame


def _serving_error(status: int) -> ServingError:
    return ServingError(status, ErrorResponse(message="body message", type="t", code=status))


def _create(backend: _Backend, request: AnthropicMessagesRequest):
    serving = AnthropicMessagesServing(backend)
    return asyncio.run(serving.create_message(request, RequestContext.internal({})))


def _body(response: JSONResponse) -> dict:
    return json.loads(bytes(response.body))


def _events(response: StreamingResponse) -> list:
    async def collect():
        return [frame async for frame in response.body_iterator]

    events = []
    for frame in "".join(asyncio.run(collect())).strip().split("\n\n"):
        name, data = None, None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append((name, data))
    return events


@pytest.fixture
def error_records():
    """Messages of the ERROR records the TensorRT-LLM logger emits (it does not propagate to caplog)."""
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.ERROR)
    trtllm_logger._logger.addHandler(handler)
    try:
        yield records
    finally:
        trtllm_logger._logger.removeHandler(handler)


# -- create_message --------------------------------------------------------------


def test_non_streaming_result_is_translated_without_a_round_trip():
    backend = _Backend(result=_chat_response())

    response = _create(backend, _request())

    assert isinstance(response, JSONResponse) and response.status_code == 200
    body = _body(response)
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["model"] == RESPONSE_MODEL
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 2}
    (chat_request, ctx), *_ = backend.calls
    assert chat_request.model == CLIENT_MODEL and chat_request.stream is False
    assert isinstance(ctx, RequestContext)


def test_streaming_result_is_reframed_as_event_stream_from_the_backends_iterator(monkeypatch):
    stream = _stream(_frames("he", "llo"))
    backend = _Backend(result=stream)
    seen = {}

    async def fake_reframe(openai_sse, model):
        seen["iterator"] = openai_sse
        seen["model"] = model
        yield "event: ping\ndata: {}\n\n"

    monkeypatch.setattr(serving_module, "reframe_openai_stream", fake_reframe)

    response = _create(backend, _request(stream=True))

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    assert _events(response) == [("ping", {})]
    # The backend's stream object goes to the reframer as it is: no copy, no
    # wrapper, so the adapter reads the worker's bytes or the core's frames.
    assert seen["iterator"] is stream
    assert seen["model"] == "x-model"


def test_stream_without_chunks_names_the_backends_model():
    backend = _Backend(result=_stream(["data: [DONE]\n\n"]), model="x-model")

    response = _create(backend, _request(stream=True))

    events = _events(response)
    assert events[0][0] == "message_start"
    assert events[0][1]["message"]["model"] == "x-model"
    assert events[-1][0] == "message_stop"


def test_stream_with_chunks_adopts_the_chunks_model():
    backend = _Backend(result=_stream(_frames("he", "llo")), model="x-model")

    response = _create(backend, _request(stream=True))

    events = _events(response)
    assert events[0][1]["message"]["model"] == RESPONSE_MODEL
    text = "".join(
        data["delta"]["text"]
        for name, data in events
        if name == "content_block_delta" and data["delta"]["type"] == "text_delta"
    )
    assert text == "hello"


@pytest.mark.parametrize(
    "status, expected_type",
    [(400, "invalid_request_error"), (429, "invalid_request_error"), (503, "api_error")],
)
def test_serving_error_is_rendered_with_the_backends_message_and_the_status(status, expected_type):
    backend = _Backend(error=_serving_error(status), message="x-message")

    response = _create(backend, _request())

    assert response.status_code == status
    assert _body(response) == {
        "type": "error",
        "error": {"type": expected_type, "message": "x-message"},
    }


def test_engine_failure_is_a_500_api_error():
    backend = _Backend(error=EngineFatalError())

    response = _create(backend, _request())

    assert response.status_code == 500
    assert _body(response)["error"] == {"type": "api_error", "message": "Internal server error"}


def test_request_conversion_failure_is_a_400_before_the_backend_runs():
    backend = _Backend(result=_chat_response())

    response = _create(
        backend, _request(tools=[{"name": "web_search", "type": "web_search_20250305"}])
    )

    assert response.status_code == 400
    assert _body(response)["error"]["type"] == "invalid_request_error"
    assert "server tool" in _body(response)["error"]["message"]
    assert backend.calls == []


def test_unconvertible_chat_response_is_a_500_with_one_error_log(error_records):
    backend = _Backend(result=_chat_response(tool_arguments="{not-json"))

    response = _create(backend, _request())

    assert response.status_code == 500
    assert _body(response)["error"] == {"type": "api_error", "message": "Internal server error"}
    assert len(error_records) == 1
    assert "Invalid response from OpenAI chat pipeline" in error_records[0]


# -- run_batched_message ----------------------------------------------------------


def _run_batched(backend: _Backend, request: AnthropicMessagesRequest):
    serving = AnthropicMessagesServing(backend)
    return asyncio.run(serving.run_batched_message(request))


def test_batched_message_runs_an_internal_context_without_streaming():
    backend = _Backend(result=_chat_response())

    outcome, payload = _run_batched(backend, _request(stream=True))

    assert outcome == "succeeded"
    assert payload["content"] == [{"type": "text", "text": "hello"}]
    (chat_request, ctx), *_ = backend.calls
    # Streaming makes no sense for a batch: the result is collected, not watched.
    assert chat_request.stream is False
    assert isinstance(ctx, RequestContext)
    assert ctx.http_request is None
    assert dict(ctx.headers) == {}
    assert ctx.no_client_connection is True
    assert asyncio.run(ctx.json()) == chat_request.model_dump(exclude_none=True)


@pytest.mark.parametrize(
    "error, expected",
    [
        (_serving_error(400), {"type": "invalid_request_error", "message": "x-message"}),
        (_serving_error(503), {"type": "api_error", "message": "x-message"}),
        (EngineFatalError(), {"type": "api_error", "message": "Internal server error"}),
    ],
    ids=["rejected", "upstream_5xx", "engine_failure"],
)
def test_batched_message_reports_errors_as_errored_items(error, expected):
    backend = _Backend(error=error, message="x-message")

    assert _run_batched(backend, _request()) == ("errored", expected)


def test_batched_message_conversion_failure_does_not_reach_the_backend():
    backend = _Backend(result=_chat_response())

    outcome, payload = _run_batched(
        backend, _request(tools=[{"name": "web_search", "type": "web_search_20250305"}])
    )

    assert outcome == "errored"
    assert payload["type"] == "invalid_request_error"
    assert backend.calls == []


def test_batched_unconvertible_response_is_an_errored_item_with_one_log(error_records):
    backend = _Backend(result=_chat_response(tool_arguments="{not-json"))

    assert _run_batched(backend, _request()) == (
        "errored",
        {"type": "api_error", "message": "Internal server error"},
    )
    assert len(error_records) == 1
    assert "Invalid response from OpenAI chat pipeline in batch" in error_records[0]


# -- _error_type ----------------------------------------------------------------


@pytest.mark.parametrize(
    "status, expected",
    [
        (399, "api_error"),
        (400, "invalid_request_error"),
        (499, "invalid_request_error"),
        (500, "api_error"),
    ],
)
def test_error_type_boundaries(status, expected):
    assert _error_type(status) == expected
