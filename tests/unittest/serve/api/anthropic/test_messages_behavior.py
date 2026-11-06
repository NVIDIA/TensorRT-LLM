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
"""What the aggregated and the disaggregated server answer on the Anthropic routes, pinned.

Both servers serve ``POST /v1/messages`` by translating the request into a chat
completion, running it through their own chat pipeline and translating the
answer back; the aggregated server also runs Message Batches over the same
path. The two pipelines answer some cases differently today (the model name of
a stream without chunks, the message of an unexpected error) and are meant to
keep doing so. These tests pin each server's own answer, case by case, so that
a change to how the Anthropic routes reach the chat pipelines can show it kept
every one of them.

Only HTTP and the engine seams are used. The aggregated chat path is stubbed
where it meets the engine (``generate_async`` and the response builder
``_create_chat_response``); the disaggregated one where it meets the service
(``openai_chat_completion``). The served model is ``server-model`` and every
request names ``client-model``, so an answer that echoes the wrong one is
caught.
"""

import json
import logging
import signal
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from test_common.serve.app_factory import AggConfig, DisaggConfig

import tensorrt_llm.serve.openai_disagg_server as disagg_server_module
import tensorrt_llm.serve.openai_server as openai_server_module
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.inputs.chat_template_guard import UnusedChatTemplateKwargsError
from tensorrt_llm.logger import logger as trtllm_logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

# threadleak is off: the TestClient's portal thread and the fake server's
# worker pools are still alive when the leak check runs; the serve_env fixture
# closes them in its teardown, which runs after that check.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

SIDES = ["aggregated", "disaggregated"]
SERVER_MODEL = "server-model"
CLIENT_MODEL = "client-model"
# Two chunks of streamed text. The accent gives the disaggregated byte stream a
# multi-byte character to be cut inside.
STREAMED_CHUNKS = ("hél", "lo")
STREAMED_TEXT = "".join(STREAMED_CHUNKS)
COUNTER_NAMES = (
    "total_requests",
    "stream_requests",
    "nonstream_requests",
    "validation_exceptions",
    "http_exceptions",
    "internal_errors",
    "total_responses",
)


def _changed(case: str, side: str, expected, got) -> str:
    return (
        f"Anthropic {case} on the {side} server changed: expected {expected!r}, got {got!r}. "
        "Each server must keep its own answer; the aggregated and disaggregated servers "
        "differ on purpose."
    )


# -- requests and engine answers -----------------------------------------------


def _request(text: str = "hello", **overrides) -> dict:
    payload = {
        "model": CLIENT_MODEL,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": text}],
    }
    payload.update(overrides)
    return payload


def _chat_response(*, tool_arguments=None) -> ChatCompletionResponse:
    """The chat pipeline's non-streaming answer: the text "hello", optionally with a tool call."""
    tool_calls = []
    if tool_arguments is not None:
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(name="get_weather", arguments=tool_arguments),
            )
        ]
    return ChatCompletionResponse(
        id="chatcmpl-messages-behavior",
        model=SERVER_MODEL,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello", tool_calls=tool_calls),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def _stream_frames(*texts: str) -> list:
    """The chat pipeline's SSE frames for a stream of text chunks, without the final [DONE]."""
    chunks = [
        ChatCompletionStreamResponse(
            model=SERVER_MODEL, choices=[{"index": 0, "delta": {"content": text}}]
        )
        for text in texts
    ]
    chunks.append(
        ChatCompletionStreamResponse(
            model=SERVER_MODEL,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )
    )
    return [f"data: {chunk.model_dump_json()}\n\n" for chunk in chunks]


def _promise() -> SimpleNamespace:
    """The engine's handle for a non-streaming request; the response builder is stubbed."""
    return SimpleNamespace(prompt_token_ids=[1, 2, 3])


class _StreamedGeneration:
    """A streamed generation whose steps are already post-processed SSE frames."""

    prompt_token_ids = [1, 2, 3]

    def __init__(self, steps):
        self._steps = list(steps)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._steps:
            raise StopAsyncIteration
        frames = self._steps.pop(0)
        return SimpleNamespace(
            finished=not self._steps,
            outputs=[SimpleNamespace(_postprocess_result=frames)],
        )


def _relayed_bytes(frames: list):
    """The worker's SSE bytes as the disaggregated service relays them.

    Delivered in two reads, cut inside the multi-byte character of the streamed
    text, as network reads need not align with characters or lines.
    """
    payload = "".join(frames).encode("utf-8")
    cut = payload.index("é".encode("utf-8")) + 1

    async def relay():
        yield payload[:cut]
        yield payload[cut:]

    return relay()


def _done_only():
    """A relayed stream finished by the context worker alone: only the [DONE] sentinel."""

    async def relay():
        yield b"data: [DONE]\n\n"

    return relay()


def _upstream_error(worker_body: str, status: int) -> aiohttp.ClientResponseError:
    """A worker rejection the way the disaggregated client raises it: reason, then the body."""
    return aiohttp.ClientResponseError(
        request_info=Mock(), history=(), status=status, message=f"Bad Request: {worker_body}"
    )


# -- servers -------------------------------------------------------------------


def _aggregated_server(env, *, generate_async, chat_response=None) -> OpenAIServer:
    """The aggregated server with its chat path stubbed where it meets the engine.

    Post-processing runs in worker mode, so a streamed generation's steps are
    the SSE frames themselves. The engine's preprocessing and the disconnect
    watcher are not part of any answer and are left out.
    """
    server = env.build_openai_server(AggConfig(model_name=SERVER_MODEL, num_postprocess_workers=1))
    server.generator.preprocess = None
    server.generator.generate_async = generate_async
    server.await_disconnected = AsyncMock()
    if chat_response is not None:
        server._create_chat_response = AsyncMock(return_value=chat_response)
    return server


def _disaggregated_server(env, openai_chat_completion):
    """The disaggregated server with the service stubbed and its request counters mocked."""
    server = env.build_disagg_server(DisaggConfig())
    counters = SimpleNamespace(
        **{name: SimpleNamespace(inc=Mock()) for name in COUNTER_NAMES},
        queue_latency_seconds=SimpleNamespace(observe=Mock()),
    )
    server._perf_metrics_collector = counters
    server._service = SimpleNamespace(openai_chat_completion=openai_chat_completion)
    return server, counters


def _post_messages(env, side: str, server, payload: dict):
    # The disaggregated lifespan connects to workers; these tests never run it.
    client = env.client(server.app, lifespan=side == "aggregated")
    return client.post("/v1/messages", json=payload)


def _assert_error(response, *, case: str, side: str, status: int, error_type: str, message: str):
    body = response.json()
    got = (response.status_code, body.get("type"), body.get("error"))
    expected = (status, "error", {"type": error_type, "message": message})
    assert got == expected, _changed(case, side, expected, got)


def _assert_counts(counters, case: str, **expected: int) -> None:
    for name in COUNTER_NAMES:
        got = getattr(counters, name).inc.call_count
        assert got == expected.get(name, 0), _changed(
            f"{case} ({name} counter)", "disaggregated", expected.get(name, 0), got
        )


def _events(body: str) -> list:
    """The (event name, decoded data) pairs of an Anthropic SSE body."""
    events = []
    for frame in body.strip().split("\n\n"):
        name, data = None, None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append((name, data))
    return events


def _streamed_text(events: list) -> str:
    return "".join(
        data["delta"]["text"]
        for name, data in events
        if name == "content_block_delta" and data["delta"]["type"] == "text_delta"
    )


@pytest.fixture
def error_records():
    """The ERROR records the TensorRT-LLM logger emits during the test."""
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture(level=logging.ERROR)
    trtllm_logger._logger.addHandler(handler)
    try:
        yield records
    finally:
        trtllm_logger._logger.removeHandler(handler)


@pytest.fixture
def engine_failure(monkeypatch):
    """What an engine failure does to the process, recorded instead of done.

    Both servers answer an engine failure by recording the termination and
    raising SIGINT; the recorder is patched in the module whose code raises it.
    """
    recorded = SimpleNamespace(signals=[], aggregated=Mock(), disaggregated=Mock())
    monkeypatch.setattr(signal, "raise_signal", recorded.signals.append)
    monkeypatch.setattr(openai_server_module, "_record_generator_termination", recorded.aggregated)
    monkeypatch.setattr(
        disagg_server_module, "record_termination_observation", recorded.disaggregated
    )
    return recorded


# -- /v1/messages --------------------------------------------------------------


@pytest.mark.parametrize("side", SIDES)
def test_non_streaming_success_is_the_translated_chat_response(side, serve_env):
    if side == "aggregated":
        server = _aggregated_server(
            serve_env, generate_async=Mock(return_value=_promise()), chat_response=_chat_response()
        )
        counters = None
    else:
        server, counters = _disaggregated_server(
            serve_env, AsyncMock(return_value=_chat_response())
        )

    response = _post_messages(serve_env, side, server, _request())

    body = response.json()
    got = (response.status_code, body.get("content"), body.get("model"))
    expected = (200, [{"type": "text", "text": "hello"}], SERVER_MODEL)
    assert got == expected, _changed("non-streaming success", side, expected, got)
    if counters is not None:
        _assert_counts(
            counters,
            "non-streaming success",
            total_requests=1,
            nonstream_requests=1,
            total_responses=1,
        )


@pytest.mark.parametrize("side", SIDES)
def test_streaming_success_reports_the_chunks_model_and_text(side, serve_env):
    frames = _stream_frames(*STREAMED_CHUNKS)
    if side == "aggregated":
        generation = _StreamedGeneration([[frame] for frame in frames])
        server = _aggregated_server(serve_env, generate_async=Mock(return_value=generation))
    else:
        relayed = _relayed_bytes(frames + ["data: [DONE]\n\n"])
        server, _ = _disaggregated_server(serve_env, AsyncMock(return_value=relayed))

    response = _post_messages(serve_env, side, server, _request(stream=True))

    assert response.status_code == 200, _changed(
        "streaming success", side, 200, (response.status_code, response.text)
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    got = (events[0][0], events[0][1]["message"]["model"], _streamed_text(events), events[-1][0])
    expected = ("message_start", SERVER_MODEL, STREAMED_TEXT, "message_stop")
    assert got == expected, _changed("streaming success", side, expected, got)


@pytest.mark.parametrize(
    "side, expected_model",
    [("aggregated", SERVER_MODEL), ("disaggregated", CLIENT_MODEL)],
)
def test_stream_without_chunks_names_each_servers_own_model(side, expected_model, serve_env):
    """With no chunk to take the model from, each server falls back to its own choice."""
    if side == "aggregated":
        generation = _StreamedGeneration([[]])
        server = _aggregated_server(serve_env, generate_async=Mock(return_value=generation))
    else:
        server, _ = _disaggregated_server(serve_env, AsyncMock(return_value=_done_only()))

    response = _post_messages(serve_env, side, server, _request(stream=True))

    assert response.status_code == 200, _changed(
        "stream without chunks", side, 200, (response.status_code, response.text)
    )
    events = _events(response.text)
    got = (events[0][0], events[0][1]["message"]["model"], events[-1][0])
    expected = ("message_start", expected_model, "message_stop")
    assert got == expected, _changed("stream without chunks", side, expected, got)


@pytest.mark.parametrize("side", SIDES)
def test_rejected_request_is_a_400_with_the_rejection(side, serve_env):
    if side == "aggregated":
        server = _aggregated_server(serve_env, generate_async=Mock(side_effect=ValueError("bad")))
    else:
        server, _ = _disaggregated_server(
            serve_env, AsyncMock(side_effect=UnusedChatTemplateKwargsError("bad"))
        )

    response = _post_messages(serve_env, side, server, _request())

    _assert_error(
        response,
        case="rejected request",
        side=side,
        status=400,
        error_type="invalid_request_error",
        message="bad",
    )


@pytest.mark.parametrize("side", SIDES)
def test_context_length_error_keeps_its_message(side, serve_env):
    message = context_length_exceeded_message(2048, 4096)
    if side == "aggregated":
        server = _aggregated_server(serve_env, generate_async=Mock(side_effect=ValueError(message)))
    else:
        worker_body = OpenAIServer.create_error_response(message).body.decode()
        server, _ = _disaggregated_server(
            serve_env, AsyncMock(side_effect=_upstream_error(worker_body, status=400))
        )

    response = _post_messages(serve_env, side, server, _request())

    _assert_error(
        response,
        case="context length",
        side=side,
        status=400,
        error_type="invalid_request_error",
        message=message,
    )


def test_upstream_error_keeps_the_worker_status_and_message(serve_env):
    error = aiohttp.ClientResponseError(
        request_info=Mock(),
        history=(),
        status=429,
        message='Too Many Requests: {"error": {"message": "slow down"}}',
    )
    server, counters = _disaggregated_server(serve_env, AsyncMock(side_effect=error))

    response = _post_messages(serve_env, "disaggregated", server, _request())

    _assert_error(
        response,
        case="upstream error",
        side="disaggregated",
        status=429,
        error_type="invalid_request_error",
        message="slow down",
    )
    _assert_counts(
        counters, "upstream error", total_requests=1, nonstream_requests=1, http_exceptions=1
    )


@pytest.mark.parametrize(
    "side, status, error_type, message",
    [
        ("aggregated", 400, "invalid_request_error", "boom"),
        ("disaggregated", 500, "api_error", "Internal server error boom"),
    ],
)
def test_unexpected_error_is_answered_each_servers_way(
    side, status, error_type, message, serve_env
):
    if side == "aggregated":
        server = _aggregated_server(
            serve_env, generate_async=Mock(side_effect=RuntimeError("boom"))
        )
        counters = None
    else:
        server, counters = _disaggregated_server(
            serve_env, AsyncMock(side_effect=RuntimeError("boom"))
        )

    response = _post_messages(serve_env, side, server, _request())

    _assert_error(
        response,
        case="unexpected error",
        side=side,
        status=status,
        error_type=error_type,
        message=message,
    )
    if counters is not None:
        _assert_counts(
            counters, "unexpected error", total_requests=1, nonstream_requests=1, internal_errors=1
        )


@pytest.mark.parametrize("side", SIDES)
def test_engine_failure_is_a_500_after_one_sigint(side, serve_env, engine_failure):
    if side == "aggregated":
        server = _aggregated_server(
            serve_env, generate_async=Mock(side_effect=CppExecutorError("dead"))
        )
        recorder = engine_failure.aggregated
    else:
        server, _ = _disaggregated_server(
            serve_env, AsyncMock(side_effect=CppExecutorError("dead"))
        )
        recorder = engine_failure.disaggregated

    response = _post_messages(serve_env, side, server, _request())

    _assert_error(
        response,
        case="engine failure",
        side=side,
        status=500,
        error_type="api_error",
        message="Internal server error",
    )
    got = (engine_failure.signals, recorder.call_count)
    expected = ([signal.SIGINT], 1)
    assert got == expected, _changed(
        "engine failure (SIGINT, termination record)", side, expected, got
    )


@pytest.mark.parametrize("side", SIDES)
def test_unconvertible_chat_response_is_a_500_with_one_error_log(side, serve_env, error_records):
    response_with_bad_tool_call = _chat_response(tool_arguments="{not-json")
    if side == "aggregated":
        server = _aggregated_server(
            serve_env,
            generate_async=Mock(return_value=_promise()),
            chat_response=response_with_bad_tool_call,
        )
    else:
        server, _ = _disaggregated_server(
            serve_env, AsyncMock(return_value=response_with_bad_tool_call)
        )

    response = _post_messages(serve_env, side, server, _request())

    _assert_error(
        response,
        case="unconvertible response",
        side=side,
        status=500,
        error_type="api_error",
        message="Internal server error",
    )
    messages = [record.getMessage() for record in error_records]
    assert len(messages) == 1 and "Invalid response from OpenAI chat pipeline" in messages[0], (
        _changed(
            "unconvertible response (error log)",
            side,
            "one record naming the invalid response",
            messages,
        )
    )


# -- /v1/messages/batches (aggregated only) --------------------------------------


def test_batch_items_report_each_outcome(serve_env, engine_failure):
    def generate_async(inputs, **kwargs):
        prompt = inputs.get("prompt") or ""
        if "reject me" in prompt:
            raise ValueError("bad")
        if "kill the engine" in prompt:
            raise CppExecutorError("dead")
        return _promise()

    server = _aggregated_server(
        serve_env, generate_async=generate_async, chat_response=_chat_response()
    )
    client = serve_env.client(server.app)
    items = [
        {"custom_id": custom_id, "params": _request(text)}
        for custom_id, text in (
            ("ok", "hello"),
            ("rejected", "reject me"),
            ("engine", "kill the engine"),
        )
    ]

    created = client.post("/v1/messages/batches", json={"requests": items})
    assert created.status_code == 200, created.text
    batch_id = created.json()["id"]
    deadline = time.monotonic() + 30
    while True:
        status = client.get(f"/v1/messages/batches/{batch_id}").json()["processing_status"]
        if status == "ended":
            break
        assert time.monotonic() < deadline, f"batch {batch_id} is still {status} after 30 s"
        time.sleep(0.01)
    results = client.get(f"/v1/messages/batches/{batch_id}/results")

    assert results.status_code == 200, results.text
    by_id = {
        line["custom_id"]: line["result"] for line in map(json.loads, results.text.splitlines())
    }
    got = {
        "ok": (
            by_id["ok"]["type"],
            by_id["ok"]["message"]["content"],
            by_id["ok"]["message"]["model"],
        ),
        "rejected": by_id["rejected"],
        "engine": by_id["engine"],
    }
    expected = {
        "ok": ("succeeded", [{"type": "text", "text": "hello"}], SERVER_MODEL),
        "rejected": {
            "type": "errored",
            "error": {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad"},
                "request_id": None,
            },
        },
        "engine": {
            "type": "errored",
            "error": {
                "type": "error",
                "error": {"type": "api_error", "message": "Internal server error"},
                "request_id": None,
            },
        },
    }
    assert got == expected, _changed("batch item outcomes", "aggregated", expected, got)
    assert engine_failure.signals == [signal.SIGINT], _changed(
        "batch engine failure (SIGINT)", "aggregated", [signal.SIGINT], engine_failure.signals
    )
