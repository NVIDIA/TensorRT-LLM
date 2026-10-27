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
"""Route-level tests for the Anthropic Messages compatibility handlers."""

import asyncio
import json
import time
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from tensorrt_llm.executor.utils import RequestError, context_length_exceeded_message
from tensorrt_llm.inputs.chat_template_guard import UnusedChatTemplateKwargsError
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.render import chat as render_chat_module

# These tests are CPU-only (no GPU, engine or sockets) and run in the
# CPU-Generic CI stage, which selects with `-m cpu_only`.
pytestmark = pytest.mark.cpu_only

MODEL = "test-model"


def _request(**overrides):
    payload = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


def _chat_response(*, tool_arguments=None):
    tool_calls = []
    if tool_arguments is not None:
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(name="get_weather", arguments=tool_arguments),
            )
        ]
    return ChatCompletionResponse(
        id="chatcmpl-route-test",
        model=MODEL,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello", tool_calls=tool_calls),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def _json_chat_response(*, tool_arguments=None, status_code=200):
    return JSONResponse(
        content=_chat_response(tool_arguments=tool_arguments).model_dump(),
        status_code=status_code,
    )


def _streaming_chat_response():
    chunks = [
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"role": "assistant"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"content": "hello"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        ),
    ]

    async def source():
        for chunk in chunks:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(source(), media_type="text/event-stream")


def _make_route_client(server_kind, openai_response):
    app = FastAPI()
    # Body validation fails before the route function runs, so without the
    # server's own handler these tests would see FastAPI's 422 instead of the
    # 400 + Anthropic envelope a client actually gets.
    _anthropic_validation_handler(app)
    if server_kind == "standard":
        server = object.__new__(OpenAIServer)
        server.model = MODEL
        backend = AsyncMock(return_value=openai_response)
        server.openai_chat = backend
    else:
        server = object.__new__(OpenAIDisaggServer)
        server._service = SimpleNamespace(openai_chat_completion=object())
        backend = AsyncMock(return_value=openai_response)
        server._wrap_entry_point = Mock(return_value=backend)
    app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
    return TestClient(app), backend


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_converts_nonstream_response(server_kind):
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post("/v1/messages", json=_request())

    assert response.status_code == 200
    assert response.json() | {"id": "ignored"} == {
        "id": "ignored",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    chat_request = backend.await_args.args[0]
    assert chat_request.model == MODEL
    assert chat_request.max_completion_tokens == 64
    assert not chat_request.stream


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_accepts_thinking_display(server_kind):
    """The thinking config Claude Code sends is served, not answered with a 400."""
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post(
        "/v1/messages",
        json=_request(max_tokens=4096, thinking={"type": "adaptive", "display": "updates"}),
    )

    assert response.status_code == 200
    assert backend.await_args.args[0].chat_template_kwargs["enable_thinking"] is True


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_rejects_anthropic_server_tools(server_kind):
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post(
        "/v1/messages",
        json=_request(
            tools=[
                {
                    "name": "web_search",
                    "type": "web_search_20250305",
                }
            ]
        ),
    )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "server tool" in response.json()["error"]["message"]
    backend.assert_not_awaited()


@pytest.mark.parametrize(
    "block,tag,field",
    [
        pytest.param({"type": "text"}, "text", "text", id="text_without_text"),
        pytest.param({"type": "text", "text": 123}, "text", "text", id="text_with_non_string_text"),
        pytest.param({"type": "image"}, "image", "source", id="image_without_source"),
        pytest.param({"type": "tool_use", "id": "a"}, "tool_use", "name", id="tool_use_no_name"),
        pytest.param(
            {"type": "tool_result", "content": "x"},
            "tool_result",
            "tool_use_id",
            id="tool_result_without_tool_use_id",
        ),
    ],
)
def test_malformed_known_block_is_a_400_against_its_own_type(block, tag, field):
    """A known type with bad fields must fail as that type, not as the catch-all.

    The union is tagged on `type`, so the block is checked against the model the
    client asked for and the error points at the field it is missing. Trying
    members in order instead let these validate as AnthropicUnknownBlock, reach
    the adapter's dispatch on `type`, and raise AttributeError on an attribute
    the block never had -- a 500 for a plain client error.

    Errors carry `loc` as a tuple, so the tag and field appear adjacent.
    """
    client, backend = _make_route_client("standard", _json_chat_response())

    response = client.post(
        "/v1/messages",
        json=_request(messages=[{"role": "user", "content": [block]}]),
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    message = response.json()["error"]["message"]
    assert f"'{tag}', '{field}'" in message
    # The catch-all must not have absorbed it; that is the regression itself.
    assert "'unknown'," not in message
    backend.assert_not_awaited()


def test_unknown_block_type_still_reaches_the_adapter():
    """The catch-all exists for types this server does not model; keep it working.

    Tagging the union must not close it: an unmodelled type has to validate and
    be named by the adapter, rather than returning a wall of failed members.
    """
    client, backend = _make_route_client("standard", _json_chat_response())

    response = client.post(
        "/v1/messages",
        json=_request(messages=[{"role": "user", "content": [{"type": "document", "source": {}}]}]),
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "document" in response.json()["error"]["message"]
    backend.assert_not_awaited()


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_hides_invalid_generated_tool_arguments(server_kind):
    client, _ = _make_route_client(server_kind, _json_chat_response(tool_arguments="{not-json"))

    response = client.post("/v1/messages", json=_request())

    assert response.status_code == 500
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_reframes_streaming_response(server_kind):
    client, backend = _make_route_client(server_kind, _streaming_chat_response())

    response = client.post("/v1/messages", json=_request(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("event: message_start\n")
    assert "event: content_block_delta\n" in response.text
    assert '"text":"hello"' in response.text
    assert response.text.rstrip().endswith('event: message_stop\ndata: {"type":"message_stop"}')
    assert backend.await_args.args[0].stream


class _StreamedGeneration:
    """A streamed generation whose steps are already post-processed frames.

    After its steps it ends, raises ``then``, or (``hang=True``) waits forever
    for the next step, as a generation does while the model is still running.
    """

    prompt_token_ids = [1, 2, 3]

    def __init__(self, steps, *, then=None, hang=False):
        self._steps = list(steps)
        self._then = then
        self._hang = hang

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._steps:
            frames = self._steps.pop(0)
            return SimpleNamespace(
                finished=not self._steps,
                outputs=[SimpleNamespace(_postprocess_result=frames)],
            )
        if self._then is not None:
            raise self._then
        if self._hang:
            await asyncio.Event().wait()
        raise StopAsyncIteration


class _ModelConfig:
    model_type = "llama"
    vocab_size = 1024


def _chat_server(monkeypatch, generation):
    """A server whose real chat handlers run over ``generation``.

    The server is built with ``object.__new__``; prompt rendering and the
    engine are stubbed, and the engine's output is already post-processed into
    SSE frames.
    """

    async def render(**kwargs):
        return "rendered prompt"

    monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render)
    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server.allow_request_chat_template = False
    server.model_config = _ModelConfig()
    server.processor = None
    server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=1024))
    server.chat_template = None
    server.tool_parser = None
    server.tool_call_id_type = "random"
    server.multimodal_server_config = None
    server._is_visual_gen = False
    server._adjusted_steady_clock = SimpleNamespace(now=time.monotonic)
    server.generator = SimpleNamespace(
        args=SimpleNamespace(
            gather_generation_logits=False,
            reasoning_parser=None,
            backend="pytorch",
            guided_decoding_backend=None,
            num_postprocess_workers=1,
        ),
        generate_async=Mock(return_value=generation),
    )
    server.await_disconnected = AsyncMock()
    server._extract_metrics = AsyncMock()
    return server


async def _chat_stream(monkeypatch, generation):
    """The body of a streamed /v1/chat/completions response, and the server."""
    server = _chat_server(monkeypatch, generation)
    request = ChatCompletionRequest(
        model=MODEL, messages=[{"role": "user", "content": "hi"}], stream=True
    )
    raw_request = SimpleNamespace(state=SimpleNamespace(), headers={})
    response = await server.openai_chat(request, raw_request)
    assert isinstance(response, StreamingResponse), response.body
    return response.body_iterator, server


@pytest.fixture
def server_logger(monkeypatch):
    log = Mock()
    monkeypatch.setattr(openai_server_module, "logger", log)
    return log


def test_chat_stream_read_to_its_end_records_the_metrics(monkeypatch, server_logger):
    """Past [DONE] the chat stream records the request's metrics."""

    async def run():
        generation = _StreamedGeneration([["data: one\n\n"], ["data: two\n\n"]])
        stream, server = await _chat_stream(monkeypatch, generation)
        frames = [frame async for frame in stream]
        return frames, server

    frames, server = asyncio.run(run())

    assert frames == ["data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"]
    server._extract_metrics.assert_awaited_once()
    server_logger.error.assert_not_called()


def test_chat_stream_closed_by_its_consumer_logs_no_error(monkeypatch, server_logger):
    """A consumer that stops reading (GeneratorExit) is not a stream error."""

    async def run():
        generation = _StreamedGeneration([["data: one\n\n"], ["data: two\n\n"]])
        stream, server = await _chat_stream(monkeypatch, generation)
        assert await stream.__anext__() == "data: one\n\n"
        await stream.aclose()
        return server

    server = asyncio.run(run())

    server._extract_metrics.assert_not_awaited()
    server_logger.error.assert_not_called()


def test_chat_stream_cancelled_by_a_disconnect_logs_no_error(monkeypatch, server_logger):
    """A client that disconnects cancels the stream; that is not a stream error."""

    async def run():
        generation = _StreamedGeneration([["data: one\n\n"]], hang=True)
        stream, _ = await _chat_stream(monkeypatch, generation)
        first_frame = asyncio.Event()

        async def consume():
            async for _ in stream:
                first_frame.set()

        consumer = asyncio.create_task(consume())
        # Bounded, so a stream that fails before its first frame fails the
        # test instead of hanging it.
        await asyncio.wait_for(first_frame.wait(), timeout=5.0)
        # The stream is now waiting on the generation for its next step.
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    asyncio.run(run())

    server_logger.error.assert_not_called()


def test_chat_stream_logs_a_failure_and_raises_it(monkeypatch, server_logger):
    """A failure of the generation itself is still logged and raised."""

    async def run():
        generation = _StreamedGeneration([["data: one\n\n"]], then=RuntimeError("engine failed"))
        stream, _ = await _chat_stream(monkeypatch, generation)
        with pytest.raises(RuntimeError, match="engine failed"):
            async for _ in stream:
                pass

    asyncio.run(run())

    server_logger.error.assert_called_once()
    assert "engine failed" in server_logger.error.call_args.args[0]


# ---------------------------------------------------------------------------
# A prompt longer than the context
# ---------------------------------------------------------------------------

# The engine's rejection of a 210000-token prompt against a 200000-token context.
_OVER_LENGTH = context_length_exceeded_message(max_context_length=200000, num_prompt_tokens=210000)

# Anthropic's answer to such a prompt, the one Claude Code compacts its history on.
_PROMPT_TOO_LONG = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "prompt is too long: 210000 tokens > 199999 maximum",
    },
}


class _GatedGeneration(_StreamedGeneration):
    """A ``_StreamedGeneration`` whose first step waits for ``gate``, like a slow prefill."""

    def __init__(self, steps, gate, **kwargs):
        super().__init__(steps, **kwargs)
        self._gate = gate

    async def __anext__(self):
        await self._gate.wait()
        return await super().__anext__()


def _reply_steps():
    """A one-word reply as post-processed chat frames, one engine step each."""
    chunks = [
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"role": "assistant", "content": "hello"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        ),
    ]
    return [[f"data: {chunk.model_dump_json()}\n\n"] for chunk in chunks]


async def _messages_response(server, stream):
    """The response of ``server`` to a /v1/messages request, its body not read yet."""
    return await server.anthropic_messages(
        AnthropicMessagesRequest(**_request(stream=stream)),
        SimpleNamespace(state=SimpleNamespace(), headers={}),
    )


async def _read(response):
    """The body of ``response``, read to its end."""
    if isinstance(response, StreamingResponse):
        return "".join([frame async for frame in response.body_iterator])
    return response.body.decode()


async def _messages(server, stream):
    """The response of ``server`` to a /v1/messages request, and its body."""
    response = await _messages_response(server, stream)
    return response, await _read(response)


@pytest.fixture
def short_admission_wait(monkeypatch):
    """Shorten how long a streamed request waits for a rejection before it starts."""
    monkeypatch.setattr(
        "tensorrt_llm.serve.anthropic_adapter.STREAM_ADMISSION_WAIT_S", 0.01, raising=False
    )


def _rejecting_engine(where, stream):
    """``generate_async`` of an engine that rejects the prompt as too long.

    An engine in this process rejects it when the request is submitted; one in
    a worker process (the default) rejects it as the first result, which a
    streamed request reads as the failure of its first frame.
    """
    error = RequestError(_OVER_LENGTH)
    if where == "in_process":
        return Mock(side_effect=error)
    if stream:
        return Mock(return_value=_StreamedGeneration([], then=error))
    return Mock(
        return_value=SimpleNamespace(
            prompt_token_ids=[1, 2, 3], aresult=AsyncMock(side_effect=error)
        )
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("where", ["in_worker", "in_process"])
def test_over_length_prompt_is_answered_as_anthropic_answers_it(monkeypatch, where, stream):
    """An HTTP 400 invalid_request_error in Anthropic's wording, with no stream.

    Claude Code compacts its conversation when a request fails that way. A
    stream that opens and then fails, or OpenAI's wording, reads to it as some
    other error.
    """
    server = _chat_server(monkeypatch, generation=None)
    server.generator.generate_async = _rejecting_engine(where, stream)

    response, body = asyncio.run(_messages(server, stream))

    assert not isinstance(response, StreamingResponse)
    assert response.status_code == 400
    assert json.loads(body) == _PROMPT_TOO_LONG


def test_stream_starts_without_waiting_for_a_slow_first_token(monkeypatch, short_admission_wait):
    """The wait for a rejection is bounded; a long prefill does not hold the stream back."""

    async def run():
        gate = asyncio.Event()
        server = _chat_server(monkeypatch, _GatedGeneration(_reply_steps(), gate))
        response = await _messages_response(server, stream=True)
        # Started while the first token is still pending.
        assert isinstance(response, StreamingResponse)
        gate.set()
        return await _read(response)

    body = asyncio.run(run())

    assert body.startswith("event: message_start\n")
    assert '"text":"hello"' in body
    assert body.rstrip().endswith('data: {"type":"message_stop"}')


def test_rejection_after_the_wait_ends_the_stream_in_the_same_wording(
    monkeypatch, short_admission_wait
):
    """A rejection slower than the wait still says the prompt is too long."""

    async def run():
        gate = asyncio.Event()
        generation = _GatedGeneration([], gate, then=RequestError(_OVER_LENGTH))
        response = await _messages_response(_chat_server(monkeypatch, generation), stream=True)
        assert isinstance(response, StreamingResponse)
        gate.set()
        return await _read(response)

    body = asyncio.run(run())

    # The stream has started, so the error event has the type every stream failure has.
    event, data = body.strip().split("\n")
    assert event == "event: error"
    assert json.loads(data[len("data: ") :])["error"] == {
        "type": "api_error",
        "message": _PROMPT_TOO_LONG["error"]["message"],
    }


def test_other_failures_before_the_first_token_stay_in_the_stream(monkeypatch):
    """Only an over-length rejection turns into a 400; other failures are reported as before."""
    generation = _StreamedGeneration([], then=RequestError("engine failed"))
    server = _chat_server(monkeypatch, generation)

    response, body = asyncio.run(_messages(server, stream=True))

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert body == (
        "event: error\n"
        'data: {"type":"error","error":{"type":"api_error","message":"Internal server error"}}\n\n'
    )


def test_client_gone_before_the_first_token_leaves_nothing_running(
    monkeypatch, server_logger, short_admission_wait
):
    """A disconnect cancels the read the stream is waiting on, and logs no error."""

    async def run():
        server = _chat_server(monkeypatch, _StreamedGeneration([], hang=True))
        response = await _messages_response(server, stream=True)
        consumer = asyncio.ensure_future(response.body_iterator.__anext__())
        await asyncio.sleep(0.05)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    assert asyncio.run(run()) == []
    server_logger.error.assert_not_called()


@pytest.mark.parametrize(
    ("message", "reply"),
    [
        pytest.param(
            context_length_exceeded_message(8192, 8192),
            "prompt is too long: 8192 tokens > 8191 maximum",
            id="prompt_fills_the_context",
        ),
        pytest.param(
            f"Generation failed: {context_length_exceeded_message(8192, 9000)}",
            "prompt is too long: 9000 tokens > 8191 maximum",
            id="wrapped_rejection",
        ),
    ],
)
@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_prompt_too_long_carries_the_counts(server_kind, message, reply):
    """N and M are the prompt and the longest prompt the server accepts, so N > M holds."""
    client, _ = _make_route_client(server_kind, OpenAIServer.create_error_response(message))

    response = client.post("/v1/messages", json=_request(stream=True))

    assert response.status_code == 400
    assert response.json()["error"] == {"type": "invalid_request_error", "message": reply}


def _relayed_by_disagg_frontend(worker_response):
    """What the disaggregated frontend hands its routes for a worker's error response."""
    frontend = object.__new__(OpenAIDisaggServer)
    frontend._perf_metrics_collector = SimpleNamespace(
        http_exceptions=SimpleNamespace(inc=Mock()),
        internal_errors=SimpleNamespace(inc=Mock()),
    )
    upstream = aiohttp.ClientResponseError(
        request_info=Mock(),
        history=(),
        status=worker_response.status_code,
        message=f"Bad Request: {worker_response.body.decode()}",
    )
    try:
        return frontend._handle_exception(upstream)
    except HTTPException as e:
        return e


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("message", "status", "error"),
    [
        pytest.param(_OVER_LENGTH, 400, _PROMPT_TOO_LONG["error"], id="over_length"),
        pytest.param(
            "`max_tokens` (0) must be greater than 0",
            400,
            {"type": "invalid_request_error", "message": "`max_tokens` (0) must be greater than 0"},
            id="other_client_error",
        ),
        pytest.param(
            "engine overloaded",
            503,
            {"type": "api_error", "message": "engine overloaded"},
            id="server_error",
        ),
    ],
)
def test_disagg_route_answers_a_worker_rejection(stream, message, status, error):
    """A context worker's rejection reaches the client before any stream, streamed or not.

    The context phase never streams, so the frontend holds the worker's error
    before it starts a response, as a response for an over-length prompt and
    as an HTTPException for anything else.
    """
    worker_response = OpenAIServer.create_error_response(message, status_code=HTTPStatus(status))
    relayed = _relayed_by_disagg_frontend(worker_response)
    client, backend = _make_route_client("disagg", None)
    if isinstance(relayed, HTTPException):
        backend.side_effect = relayed
    else:
        backend.return_value = relayed

    response = client.post("/v1/messages", json=_request(stream=stream))

    assert response.status_code == status
    assert response.json() == {"type": "error", "error": error}


@pytest.fixture
def prometheus_multiproc_dir(tmp_path, monkeypatch):
    """PROMETHEUS_MULTIPROC_DIR for one test.

    prometheus_client picks and caches its value class the first time a metric
    is built with the variable set, so the class is restored as well as the
    environment; otherwise later metrics in the process fail without a
    directory.
    """
    import prometheus_client.values

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    saved_value_class = prometheus_client.values.ValueClass
    try:
        yield tmp_path
    finally:
        prometheus_client.values.ValueClass = saved_value_class


def test_standard_and_disagg_register_messages_route(prometheus_multiproc_dir):
    # register_routes mounts a prometheus multiprocess collector, which needs a
    # real PROMETHEUS_MULTIPROC_DIR; object.__new__ skips the startup that
    # creates one.

    standard = object.__new__(OpenAIServer)
    standard.app = FastAPI()
    standard.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    standard.use_harmony = False
    # register_routes() reads these to decide whether to mount the optional
    # control endpoints. Building the server with object.__new__ skips
    # __init__, so anything register_routes() touches has to be supplied.
    standard._enable_runtime_control_endpoints = False
    standard._enable_rl_control_endpoints = False
    standard.register_routes()

    disagg = object.__new__(OpenAIDisaggServer)
    disagg.app = FastAPI()
    disagg._service = SimpleNamespace(
        openai_completion=AsyncMock(),
        openai_chat_completion=AsyncMock(),
        anthropic_count_tokens=AsyncMock(),
        openai_responses=AsyncMock(),
    )
    disagg._perf_metrics_collector = SimpleNamespace(get_perf_metrics=AsyncMock())
    disagg._disagg_cluster_storage = None
    # register_routes reads this; the server is built with object.__new__ here,
    # so every attribute it touches has to be stubbed explicitly.
    disagg._coordinator = None
    disagg.register_routes()

    for server in (standard, disagg):
        paths = {route.path for route in server.app.routes}
        assert "/v1/messages" in paths

    # Claude Code calls count_tokens before most turns to size its context.
    # An unregistered route 404s every call, which the client cannot act on,
    # so registration is asserted separately from /v1/messages -- and on both
    # servers, because a client pointed at a disaggregated deployment cannot
    # tell the difference and must not lose the endpoint.
    for server in (standard, disagg):
        paths = {route.path for route in server.app.routes}
        assert "/v1/messages/count_tokens" in paths

    disagg_paths = {route.path for route in disagg.app.routes}
    assert "/v1/responses" in disagg_paths
    assert "/v1/models" in disagg_paths


def _batch_client(runner=None):
    """A TestClient over just the batch routes, with a stubbed item runner."""
    from tensorrt_llm.serve.anthropic_batches import AnthropicBatchStore

    async def ok(request):
        return "succeeded", {"type": "message", "role": "assistant", "content": []}

    app = FastAPI()
    _anthropic_validation_handler(app)
    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server._anthropic_batch_store = AnthropicBatchStore(runner=runner or ok)
    app.add_api_route("/v1/messages/batches", server.anthropic_create_batch, methods=["POST"])
    app.add_api_route("/v1/messages/batches", server.anthropic_list_batches, methods=["GET"])
    app.add_api_route(
        "/v1/messages/batches/{batch_id}", server.anthropic_get_batch, methods=["GET"]
    )
    app.add_api_route(
        "/v1/messages/batches/{batch_id}", server.anthropic_delete_batch, methods=["DELETE"]
    )
    app.add_api_route(
        "/v1/messages/batches/{batch_id}/cancel", server.anthropic_cancel_batch, methods=["POST"]
    )
    app.add_api_route(
        "/v1/messages/batches/{batch_id}/results", server.anthropic_batch_results, methods=["GET"]
    )
    return TestClient(app)


def _batch_body(*custom_ids):
    return {
        "requests": [
            {
                "custom_id": cid,
                "params": {
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            }
            for cid in custom_ids
        ]
    }


def test_batch_routes_are_registered_on_the_standard_server():
    standard = object.__new__(OpenAIServer)
    standard.app = FastAPI()
    standard.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    standard.use_harmony = False
    # register_routes() reads these to decide whether to mount the optional
    # control endpoints. Building the server with object.__new__ skips
    # __init__, so anything register_routes() touches has to be supplied.
    standard._enable_runtime_control_endpoints = False
    standard._enable_rl_control_endpoints = False
    standard.register_routes()

    paths = {route.path for route in standard.app.routes}
    for path in (
        "/v1/messages/batches",
        "/v1/messages/batches/{batch_id}",
        "/v1/messages/batches/{batch_id}/cancel",
        "/v1/messages/batches/{batch_id}/results",
    ):
        assert path in paths, f"{path} not registered"


def test_create_batch_returns_an_in_progress_batch():
    client = _batch_client()
    response = client.post("/v1/messages/batches", json=_batch_body("a", "b"))

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message_batch"
    assert body["id"].startswith("msgbatch_")
    assert body["processing_status"] in {"in_progress", "ended"}
    assert body["request_counts"]["processing"] + body["request_counts"]["succeeded"] == 2


def test_create_batch_rejects_duplicate_custom_ids():
    client = _batch_client()
    response = client.post("/v1/messages/batches", json=_batch_body("same", "same"))

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_unknown_batch_says_it_is_not_found_and_explains_why():
    """The in-memory limitation is the likeliest cause, so name it."""
    client = _batch_client()
    response = client.get("/v1/messages/batches/msgbatch_nope")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "not_found_error"
    assert "restart" in error["message"]


def test_results_of_an_unfinished_batch_are_400_not_404():
    """404 would tell a polling client the batch never existed.

    Driven directly rather than through TestClient: each TestClient call runs
    in its own portal, so the batch's background task is not guaranteed to
    still be in flight by the time the second request arrives, and the batch
    would have already ended.
    """
    from tensorrt_llm.serve.anthropic_batches import AnthropicBatchStore
    from tensorrt_llm.serve.anthropic_protocol import AnthropicBatchRequestItem

    async def scenario():
        blocked = asyncio.Event()

        async def never_finishes(request):
            await blocked.wait()
            return "succeeded", {"type": "message"}

        server = object.__new__(OpenAIServer)
        server._anthropic_batch_store = AnthropicBatchStore(runner=never_finishes)
        item = AnthropicBatchRequestItem(
            custom_id="a",
            params={
                "model": MODEL,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        batch = server._anthropic_batch_store.create([item])
        await asyncio.sleep(0)  # let the worker start and block

        response = await server.anthropic_batch_results(batch.id)
        assert response.status_code == 400
        assert "still processing" in json.loads(response.body)["error"]["message"]

        # An id that was never issued is a genuine 404, and must stay distinct.
        missing = await server.anthropic_batch_results("msgbatch_nope")
        assert missing.status_code == 404

        blocked.set()

    asyncio.run(scenario())


def test_interrupted_batch_still_reports_counts_that_sum():
    """A cancelled worker must not leave counts that under-sum.

    _run finalizes from a finally block, so a torn-down task would otherwise
    end the batch with fewer results than requests - and request_counts is the
    number clients use to decide whether every request was accounted for.
    """
    from tensorrt_llm.serve.anthropic_batches import AnthropicBatchStore
    from tensorrt_llm.serve.anthropic_protocol import AnthropicBatchRequestItem

    async def scenario():
        blocked = asyncio.Event()

        async def never_finishes(request):
            await blocked.wait()
            return "succeeded", {"type": "message"}

        store = AnthropicBatchStore(runner=never_finishes)
        items = [
            AnthropicBatchRequestItem(
                custom_id=f"id-{i}",
                params={
                    "model": MODEL,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            for i in range(4)
        ]
        batch = store.create(items)
        await asyncio.sleep(0)

        batch_record_task = store._batches[batch.id].task
        batch_record_task.cancel()
        try:
            await batch_record_task
        except asyncio.CancelledError:
            pass

        counts = store.get(batch.id).request_counts
        total = (
            counts.succeeded + counts.errored + counts.canceled + counts.expired + counts.processing
        )
        assert total == 4, f"counts sum to {total}, expected 4"

    asyncio.run(scenario())


def test_finished_batch_serves_ndjson_results():
    client = _batch_client()
    created = client.post("/v1/messages/batches", json=_batch_body("a", "b")).json()

    for _ in range(200):
        if (
            client.get(f"/v1/messages/batches/{created['id']}").json()["processing_status"]
            == "ended"
        ):
            break
        time.sleep(0.01)

    response = client.get(f"/v1/messages/batches/{created['id']}/results")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert {line["custom_id"] for line in lines} == {"a", "b"}


def test_list_batches_returns_a_page():
    client = _batch_client()
    client.post("/v1/messages/batches", json=_batch_body("a"))
    body = client.get("/v1/messages/batches").json()

    assert isinstance(body["data"], list) and body["data"]
    assert body["first_id"] == body["data"][0]["id"]
    assert body["has_more"] is False


def _handle(exception):
    """Drive OpenAIDisaggServer._handle_exception and capture what it raises."""
    server = object.__new__(OpenAIDisaggServer)
    server._perf_metrics_collector = SimpleNamespace(
        http_exceptions=SimpleNamespace(inc=Mock()),
        internal_errors=SimpleNamespace(inc=Mock()),
    )
    with pytest.raises(Exception) as excinfo:
        server._handle_exception(exception)
    return server, excinfo.value


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_counter"),
    [
        pytest.param(
            aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=400, message="bad request"
            ),
            400,
            "http_exceptions",
            id="upstream_http_status_is_propagated_not_masked_as_500",
        ),
        pytest.param(
            RuntimeError("boom"),
            500,
            "internal_errors",
            id="non_http_exceptions_still_become_500",
        ),
        pytest.param(
            UnusedChatTemplateKwargsError("chat_template_kwargs ['x'] are not referenced"),
            400,
            "http_exceptions",
            id="unused_chat_template_kwargs_is_the_clients_400_not_a_500",
        ),
    ],
)
def test_handle_exception_maps_status_and_counter(exception, expected_status, expected_counter):
    """Upstream status survives, and only for aiohttp errors.

    _handle_exception is shared by every route on the disagg server, including
    /v1/completions, so the aiohttp branch is cross-cutting and is pinned here
    for that reason. It reports an upstream ClientResponseError under its own
    status rather than flattening every upstream failure to 500: the Anthropic
    adapter derives Anthropic error types from that status, so a worker's 400
    arriving as a 500 would reach the client as an api_error rather than an
    invalid_request_error. The non-Anthropic routes gain the same fidelity - a
    429 or 503 survives instead of being flattened. The RuntimeError case pins
    the other half: anything that is not an aiohttp error still takes the
    generic 500 path.

    The counter matters as much as the status. Each exception must increment
    exactly one of the two, so an upstream 400 is not also filed as an internal
    error of this server.
    """
    server, raised = _handle(exception)
    collector = server._perf_metrics_collector
    incremented = {
        name
        for name in ("http_exceptions", "internal_errors")
        if getattr(collector, name).inc.called
    }

    assert raised.status_code == expected_status
    assert incremented == {expected_counter}


_FAKE_DEFAULT_TEMPLATE = "FAKE-DEFAULT-TEMPLATE"


class _FakeTokenizer:
    """Renders a prompt the way a chat template would, and counts words."""

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or _FAKE_DEFAULT_TEMPLATE

    def apply_chat_template(self, messages=None, **kwargs):
        if messages is None:
            # The chat pipeline passes the conversation by keyword.
            messages = kwargs.pop("conversation")
        parts = [str(m.get("content", "")) for m in messages]
        # Tools and the thinking prefix change the real prompt, so reflect them
        # here too - a count that ignored them would not be worth taking.
        for tool in kwargs.get("tools") or []:
            parts.append(str(tool))
        # A real template changes the rendered prompt, so a fake that dropped
        # it would make any test of template plumbing pass vacuously. The
        # tokenizer's own default template is not an override.
        template = kwargs.get("chat_template")
        if template and template != _FAKE_DEFAULT_TEMPLATE:
            parts.append(str(template))
        return " ".join(parts)

    def encode(self, text, add_special_tokens=True, **kwargs):
        return text.split()


def _anthropic_validation_handler(app):
    """Mirror the server's handler so boundary rejections keep the envelope.

    Model-level validation fails before the route function runs, so without
    this a bare test app answers with FastAPI's 422 shape while the real
    server answers 400 with the Anthropic envelope -- the test would pin the
    fixture's behaviour instead of the product's.
    """
    from fastapi.exceptions import RequestValidationError

    from tensorrt_llm.serve.anthropic_adapter import anthropic_error_response

    @app.exception_handler(RequestValidationError)
    async def _handler(request, exc):  # noqa: ANN001
        if request.url.path.startswith("/v1/messages"):
            return anthropic_error_response(str(exc), "invalid_request_error", 400)
        raise exc


def _count_tokens_client():
    app = FastAPI()
    _anthropic_validation_handler(app)
    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server.tokenizer = _FakeTokenizer()
    app.add_api_route("/v1/messages/count_tokens", server.anthropic_count_tokens, methods=["POST"])
    return TestClient(app)


def test_count_tokens_route_returns_a_count():
    client = _count_tokens_client()

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "one two three"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 3}


def test_count_tokens_counts_the_rendered_prompt_not_the_raw_messages():
    """Tools inflate the real prompt, so they have to inflate the count."""
    client = _count_tokens_client()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "one two three"}],
    }

    bare = client.post("/v1/messages/count_tokens", json=body).json()
    with_tools = client.post(
        "/v1/messages/count_tokens",
        json={
            **body,
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {"type": "object"},
                }
            ],
        },
    ).json()

    assert with_tools["input_tokens"] > bare["input_tokens"]


def test_count_tokens_honours_the_server_chat_template():
    """A server started with --chat_template must count the prompt it renders.

    The renderer only consults request.chat_template; openai_chat resolves
    `request.chat_template or self.chat_template`. Without the same resolution
    here, count_tokens reports a prompt the server never builds - the exact
    drift this endpoint exists to remove.
    """
    app = FastAPI()
    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server.tokenizer = _FakeTokenizer()
    # A template that measurably inflates the rendered prompt.
    server.chat_template = "SERVER-TEMPLATE-MARKER {{ messages }}"
    app.add_api_route("/v1/messages/count_tokens", server.anthropic_count_tokens, methods=["POST"])
    client = TestClient(app)

    with_template = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "one two three"}],
        },
    ).json()

    server.chat_template = None
    without = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "one two three"}],
        },
    ).json()

    assert with_template["input_tokens"] != without["input_tokens"], (
        "server chat_template had no effect on the count, so it was ignored"
    )


def test_count_tokens_rejects_a_malformed_request():
    client = _count_tokens_client()

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "bogus"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Disaggregated count_tokens forwarding
# ---------------------------------------------------------------------------


def _disagg_service(servers, post_json=None):
    """A disagg service with only what anthropic_count_tokens touches."""
    from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService

    service = object.__new__(OpenAIDisaggregatedService)
    service._ctx_router = SimpleNamespace(servers=servers)
    service._ctx_client = SimpleNamespace(post_json=post_json or AsyncMock())
    service._count_tokens_rr_counter = 0
    service.is_ready = AsyncMock(return_value=True)
    return service


def test_count_tokens_forwards_to_a_context_worker():
    """The disagg server has no tokenizer, so the count must come from a worker."""
    from tensorrt_llm.serve.anthropic_protocol import (
        AnthropicCountTokensRequest,
        AnthropicCountTokensResponse,
    )

    post_json = AsyncMock(return_value=AnthropicCountTokensResponse(input_tokens=7))
    service = _disagg_service(["ctx0:8000"], post_json)
    request = AnthropicCountTokensRequest(model=MODEL, messages=[{"role": "user", "content": "hi"}])

    response = asyncio.run(service.anthropic_count_tokens(request))

    assert response.input_tokens == 7
    endpoint, sent, response_type, server = post_json.await_args.args
    assert endpoint == "v1/messages/count_tokens"
    assert server == "ctx0:8000"
    assert response_type is AnthropicCountTokensResponse
    assert sent is request


def test_count_tokens_spreads_over_context_workers():
    """Round-robin: one worker must not absorb every count."""
    from tensorrt_llm.serve.anthropic_protocol import (
        AnthropicCountTokensRequest,
        AnthropicCountTokensResponse,
    )

    post_json = AsyncMock(return_value=AnthropicCountTokensResponse(input_tokens=1))
    service = _disagg_service(["ctx0:8000", "ctx1:8000"], post_json)
    request = AnthropicCountTokensRequest(model=MODEL, messages=[{"role": "user", "content": "hi"}])

    async def four_counts():
        for _ in range(4):
            await service.anthropic_count_tokens(request)

    asyncio.run(four_counts())

    picked = [call.args[3] for call in post_json.await_args_list]
    assert picked == ["ctx0:8000", "ctx1:8000", "ctx0:8000", "ctx1:8000"]


def test_count_tokens_without_context_workers_is_an_error_not_a_zero():
    """A missing worker must raise rather than report zero.

    Returning 0 would read as an empty prompt and silently skew a client's
    context budgeting.
    """
    from tensorrt_llm.serve.anthropic_protocol import AnthropicCountTokensRequest

    service = _disagg_service([])
    request = AnthropicCountTokensRequest(model=MODEL, messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(RuntimeError, match="No context servers"):
        asyncio.run(service.anthropic_count_tokens(request))


class _FakeSession:
    """Records the one POST post_json makes and replays a canned response."""

    def __init__(self, status=200, payload=None, text=""):
        self._status, self._payload, self._text = status, payload or {}, text
        self.sent = {}

    def post(self, url, data, headers):
        self.sent = {"url": url, "data": data, "headers": headers}
        session = self

        class _Response:
            status = session._status
            reason = "Bad Request"
            headers = {}
            # ClientResponseError.__str__ dereferences request_info.real_url, so
            # a bare None here fails while formatting the error rather than
            # while raising it.
            request_info = SimpleNamespace(real_url=url)
            history = ()

            async def json(self):
                return session._payload

            async def text(self):
                return session._text

        class _Ctx:
            async def __aenter__(self):
                return _Response()

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def test_post_json_sends_msgpack_the_worker_can_decode():
    """Exercises post_json's own serialisation rather than a mock of it.

    Every other test here substitutes post_json, so nothing covers the body it
    actually builds. The worker only decodes msgpack when X-TRTLLM-Msgpack is
    set, so a wrong header or encoding is a runtime failure on a disaggregated
    deployment that no mocked test can see.
    """
    import msgspec

    from tensorrt_llm.serve.anthropic_protocol import (
        AnthropicCountTokensRequest,
        AnthropicCountTokensResponse,
    )
    from tensorrt_llm.serve.openai_client import OpenAIHttpClient

    session = _FakeSession(payload={"input_tokens": 11})
    client = object.__new__(OpenAIHttpClient)
    client._session = session
    request = AnthropicCountTokensRequest(model=MODEL, messages=[{"role": "user", "content": "hi"}])

    response = asyncio.run(
        client.post_json(
            "v1/messages/count_tokens", request, AnthropicCountTokensResponse, "ctx0:8000"
        )
    )

    assert response.input_tokens == 11
    # The router hands out bare host:port, which aiohttp will not accept.
    assert session.sent["url"] == "http://ctx0:8000/v1/messages/count_tokens"
    assert session.sent["headers"]["X-TRTLLM-Msgpack"] == "1"
    assert msgspec.msgpack.decode(session.sent["data"])["model"] == MODEL


def test_post_json_raises_on_an_error_status():
    """A 4xx must raise, not be parsed as if it were a response body.

    response_type(**body) on an error payload would either throw a confusing
    validation error or, worse, construct a defaulted object.
    """
    import aiohttp

    from tensorrt_llm.serve.anthropic_protocol import (
        AnthropicCountTokensRequest,
        AnthropicCountTokensResponse,
    )
    from tensorrt_llm.serve.openai_client import OpenAIHttpClient

    client = object.__new__(OpenAIHttpClient)
    client._session = _FakeSession(status=400, text='{"message":"messages must not be empty"}')
    request = AnthropicCountTokensRequest(model=MODEL, messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(aiohttp.ClientResponseError, match="messages must not be empty"):
        asyncio.run(
            client.post_json(
                "v1/messages/count_tokens", request, AnthropicCountTokensResponse, "ctx0:8000"
            )
        )


@pytest.mark.parametrize(
    "message,expected",
    [
        pytest.param(
            'Bad Request: {"type":"error","error":{"type":"invalid_request_error",'
            '"message":"messages must not be empty"}}',
            "messages must not be empty",
            id="anthropic_envelope_is_unwrapped",
        ),
        pytest.param(
            'Bad Request: {"message":"plain body"}',
            "plain body",
            id="plain_message_body",
        ),
        pytest.param(
            "Bad Request: <html>gateway said no</html>",
            "context worker rejected the request with status 400",
            id="unrecognised_body_reports_status_only",
        ),
        pytest.param(
            "Bad Request: {truncated",
            "context worker rejected the request with status 400",
            id="unparsable_body_reports_status_only",
        ),
    ],
)
def test_upstream_error_never_leaks_the_worker_url(message, expected):
    """The client must learn what went wrong, not where the worker lives.

    str(ClientResponseError) embeds the upstream URL, so propagating it
    verbatim publishes a context worker's internal host and port to every
    caller -- and nests an error envelope inside an error envelope, which no
    Anthropic SDK unwraps.
    """
    from tensorrt_llm.serve.openai_disagg_server import _upstream_error_message

    error = aiohttp.ClientResponseError(
        request_info=Mock(), history=(), status=400, message=message
    )

    result = _upstream_error_message(error)

    assert result == expected
    assert "http://" not in result
    assert "8001" not in result


def test_create_batch_at_capacity_is_429_not_400():
    """A full store is a back-off signal, not a malformed request.

    400 tells the client to change the request, which cannot help here: the
    identical body succeeds once a batch ends. 429 is what makes a client retry
    instead of surfacing a hard failure to the user.

    This pins the status mapping only. Whether the store actually refuses at
    capacity is store policy, covered in test_anthropic_batches.py -- driving a
    real store to its limit through TestClient is not possible anyway, since
    each request runs in its own event loop and a batch's task does not
    survive between them.
    """
    from tensorrt_llm.serve.anthropic_batches import BatchStoreFullError

    app = FastAPI()
    server = object.__new__(OpenAIServer)
    server.model = MODEL

    def full(_requests):
        raise BatchStoreFullError(
            "the server is holding 100 unfinished batches (limit 100); retry once one of them ends"
        )

    server._anthropic_batch_store = SimpleNamespace(create=full)
    app.add_api_route("/v1/messages/batches", server.anthropic_create_batch, methods=["POST"])

    response = TestClient(app).post("/v1/messages/batches", json=_batch_body("a"))

    assert response.status_code == 429
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"
    # The message has to say what to do about it; an "invalid request" shape
    # would send the client editing a body that was never the problem.
    assert "retry" in body["error"]["message"].lower()


def test_await_disconnected_returns_immediately_for_a_synthetic_request():
    """A batched request has no socket, so the watchdog must not poll on it.

    is_disconnected() can never become true for a synthesised request, so
    without an explicit opt-out this coroutine loops at 1Hz for the life of
    the process -- one live task per batched request, each holding its Request
    and RequestOutput.
    """
    server = object.__new__(OpenAIServer)
    raw_request = OpenAIServer._synthetic_request(
        {
            "model": MODEL,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    promise = SimpleNamespace(finished=False, request_id="req", abort=Mock())

    async def run():
        # A real socket-backed request would never let this return; a bounded
        # wait keeps the failure a clear timeout rather than a hung suite.
        await asyncio.wait_for(
            OpenAIServer.await_disconnected(server, raw_request, promise),
            timeout=5.0,
        )

    asyncio.run(run())
    # Returning early must not be mistaken for a disconnect.
    promise.abort.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_list_batches_rejects_out_of_range_limit(limit):
    """Clamping hid both a client error and a surprising page size.

    limit=0 became 1 rather than an error, and the page was bounded by the
    retention limit -- how many batches are kept, which has nothing to do with
    how many fit on a page.
    """
    response = _batch_client().get(f"/v1/messages/batches?limit={limit}")

    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
