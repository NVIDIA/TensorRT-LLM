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
"""The chat completion handler's answers, pinned through the production code.

The HTTP-level tests here run on ``OpenAIServer.openai_chat`` before and after
it is split into a typed core and an HTTP wrapper, so the split can show that
it changed no answer: a rejected request keeps its 400 body byte for byte, an
engine failure keeps today's HTTP 200 with body ``null`` (and
``/health_generate`` keeps turning that into a 500), a response that cannot be
JSON-encoded keeps its 400, and a stream keeps its frames and media type.

Servers are built through the real constructor with the fake generator from
``tests/test_common/serve/``; only instance attributes are stubbed.
"""

import asyncio
import json
import logging
import signal
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.responses import Response
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import ScriptedOutput

from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.logger import logger
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve._base.errors import EngineFatalError, ServingError
from tensorrt_llm.serve._base.request_context import RequestContext
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

# threadleak is off: the tests create TestClients inside the test body, and the
# serve_env fixture closes them in its teardown, which runs after the leak check
# (precedent: tests/unittest/serve/api/ops/system/test_system_endpoints.py).
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL = "fake-model"
KEEP = (
    "openai_chat must keep this answer (status and body); see the branches of "
    "the HTTP wrapper openai_chat"
)
# The first rejection the handler itself makes (the request model already
# rejects a named tool_choice without tools): a named tool_choice on a
# pre-tokenized prompt.
NAMED_TOOL_ON_TOKEN_IDS_MESSAGE = (
    "tool_choice with a named function is not supported alongside "
    "`prompt_token_ids`; pass `messages` instead."
)
_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}


def _chat_body(**overrides):
    """A chat request whose prompt is pre-tokenized, so no rendering happens."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "prompt_token_ids": [1, 2, 3],
    }
    body.update(overrides)
    return body


def _named_tool_body():
    return _chat_body(
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )


def _chat_response(**choice_fields) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-chat-core-test",
        created=1700000000,
        model=MODEL,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="ok"),
                finish_reason="stop",
                **choice_fields,
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )


def _raw_request():
    """What the handler reads from a request when it is called directly."""
    return SimpleNamespace(state=SimpleNamespace(), headers={})


class _StreamedGeneration:
    """Engine output already post-processed into SSE frames.

    This is what the post-processing worker delivers: each step carries its
    frames in ``outputs[0]._postprocess_result``.
    """

    def __init__(self, steps, *, then=None):
        self._steps = list(steps)
        self._then = then

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._steps:
            frames = self._steps.pop(0)
            return SimpleNamespace(
                finished=not self._steps,
                outputs=[SimpleNamespace(_postprocess_result=frames)],
                candidate_metrics=None,
                metrics_dict=None,
            )
        if self._then is not None:
            raise self._then
        raise StopAsyncIteration


@pytest.fixture
def engine_failure_recorder(monkeypatch):
    """Record the two side effects of an engine failure instead of running them."""
    recorder = SimpleNamespace(signals=[], terminations=[])
    monkeypatch.setattr(signal, "raise_signal", recorder.signals.append)
    monkeypatch.setattr(
        openai_server, "_record_generator_termination", recorder.terminations.append
    )
    return recorder


def _build(serve_env, config=None) -> OpenAIServer:
    server = serve_env.build_openai_server(config or AggConfig())
    # The disconnect poll needs a client socket; none of these tests has one.
    server.await_disconnected = AsyncMock()
    return server


# ---------------------------------------------------------------------------
# HTTP answers that the split must keep
# ---------------------------------------------------------------------------


def test_http_wrapper_keeps_every_answer(serve_env, engine_failure_recorder):
    server = _build(serve_env, AggConfig(num_postprocess_workers=1))
    client = serve_env.client(server.app)

    # A request the handler rejects: 400 with today's body bytes.
    expected = OpenAIServer.create_error_response(
        message=NAMED_TOOL_ON_TOKEN_IDS_MESSAGE,
        err_type="BadRequestError",
        status_code=HTTPStatus.BAD_REQUEST,
    )
    response = client.post("/v1/chat/completions", json=_named_tool_body())
    assert response.status_code == 400, f"rejected request: {KEEP}"
    assert response.content == expected.body, f"rejected request: {KEEP}"

    # An engine failure: HTTP 200 with body null, after SIGINT.
    server.generator.script(ScriptedOutput(raise_on_call=CppExecutorError("dead")))
    response = client.post("/v1/chat/completions", json=_chat_body())
    assert response.status_code == 200, f"engine failure: {KEEP}"
    assert response.content == b"null", f"engine failure: {KEEP}"
    assert engine_failure_recorder.signals == [signal.SIGINT]
    assert len(engine_failure_recorder.terminations) == 1

    # A non-streaming success: the typed response, serialized.
    chat_response = _chat_response()
    server._create_chat_response = AsyncMock(return_value=chat_response)
    response = client.post("/v1/chat/completions", json=_chat_body())
    assert response.status_code == 200, f"non-streaming success: {KEEP}"
    assert response.json() == chat_response.model_dump(), f"non-streaming success: {KEEP}"

    # A streaming success: the frames, then [DONE], as text/event-stream.
    server.generator.generate_async = Mock(
        return_value=_StreamedGeneration([["data: a\n\n"], ["data: b\n\n"]])
    )
    response = client.post("/v1/chat/completions", json=_chat_body(stream=True))
    assert response.status_code == 200, f"streaming success: {KEEP}"
    assert response.headers["content-type"].startswith("text/event-stream"), (
        f"streaming success: {KEEP}"
    )
    assert response.text == "data: a\n\ndata: b\n\ndata: [DONE]\n\n", f"streaming success: {KEEP}"


def test_unencodable_response_is_still_a_400(serve_env, error_records):
    server = _build(serve_env)
    server._create_chat_response = AsyncMock(
        return_value=_chat_response(avg_decoded_tokens_per_iter=float("nan"))
    )
    response = serve_env.client(server.app).post("/v1/chat/completions", json=_chat_body())
    assert response.status_code == 400, f"unencodable response: {KEEP}"
    assert response.json()["message"].startswith(
        "Out of range float values are not JSON compliant"
    ), f"unencodable response: {KEEP}"
    # The wrapper answers this from its own ValueError branch, without the
    # traceback the generic handler would log.
    assert error_records == [], f"unencodable response: {KEEP}"


def test_health_generate_still_fails_with_500_on_engine_failure(serve_env, engine_failure_recorder):
    server = _build(serve_env)

    server.generator.script(ScriptedOutput(raise_on_call=CppExecutorError("dead")))
    response = asyncio.run(server.health_generate(_raw_request()))
    assert response.status_code == 500, f"/health_generate on an engine failure: {KEEP}"
    assert engine_failure_recorder.signals == [signal.SIGINT]

    server._create_chat_response = AsyncMock(return_value=_chat_response())
    response = asyncio.run(server.health_generate(_raw_request()))
    assert response.status_code == 200, f"/health_generate on a success: {KEEP}"
    assert response.body == b"Generation health check OK"


# ---------------------------------------------------------------------------
# The typed core: create_chat_completion
# ---------------------------------------------------------------------------


@pytest.fixture
def error_records():
    """Messages of the ERROR records the TRT-LLM logger emits during the test.

    The library logger does not propagate, so ``caplog`` would see nothing; a
    handler is attached to it instead. The positive control that the handler
    sees records is ``test_unexpected_error_is_a_serving_error_with_one_traceback``.
    """
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.ERROR)
    logger._logger.addHandler(handler)
    try:
        yield records
    finally:
        logger._logger.removeHandler(handler)


def _request(**overrides) -> ChatCompletionRequest:
    return ChatCompletionRequest(**_chat_body(**overrides))


def _run_core(server, request):
    return asyncio.run(
        server.create_chat_completion(request, RequestContext.from_http(_raw_request()))
    )


def test_typed_result_for_a_non_streaming_request(serve_env):
    server = _build(serve_env)
    chat_response = _chat_response()
    server._create_chat_response = AsyncMock(return_value=chat_response)

    result = _run_core(server, _request())

    assert result is chat_response
    assert not isinstance(result, Response)


def test_stream_is_returned_as_sse_strings(serve_env):
    server = _build(serve_env, AggConfig(num_postprocess_workers=1))
    server.generator.generate_async = Mock(
        return_value=_StreamedGeneration([["data: a\n\n"], ["data: b\n\n"]])
    )

    async def run():
        result = await server.create_chat_completion(
            _request(stream=True), RequestContext.from_http(_raw_request())
        )
        assert not isinstance(result, Response)
        return [frame async for frame in result]

    frames = asyncio.run(run())

    assert all(isinstance(frame, str) for frame in frames)
    assert frames == ["data: a\n\n", "data: b\n\n", "data: [DONE]\n\n"]


def test_named_tool_choice_on_token_ids_raises_serving_error(serve_env, error_records):
    server = _build(serve_env)
    expected = OpenAIServer.create_error_response(
        message=NAMED_TOOL_ON_TOKEN_IDS_MESSAGE, err_type="BadRequestError"
    )

    with pytest.raises(ServingError) as raised:
        _run_core(server, ChatCompletionRequest(**_named_tool_body()))

    assert raised.value.status == 400
    assert raised.value.body.model_dump() == json.loads(expected.body)
    assert error_records == [], "a rejected request is not an error of the server"


def test_value_error_is_a_serving_error_without_a_log(serve_env, error_records):
    server = _build(serve_env)
    server.generator.script(ScriptedOutput(raise_on_call=ValueError("bad")))

    with pytest.raises(ServingError) as raised:
        _run_core(server, _request())

    assert raised.value.status == 400
    assert raised.value.body.message == "bad"
    assert error_records == []


def test_unexpected_error_is_a_serving_error_with_one_traceback(serve_env, error_records):
    server = _build(serve_env)
    server.generator.script(ScriptedOutput(raise_on_call=RuntimeError("boom")))

    with pytest.raises(ServingError) as raised:
        _run_core(server, _request())

    assert raised.value.status == 400
    assert raised.value.body.message == "boom"
    assert len(error_records) == 1, error_records
    assert "RuntimeError: boom" in error_records[0]


def test_engine_failure_raises_engine_fatal_error_after_sigint(serve_env, engine_failure_recorder):
    server = _build(serve_env)
    server.generator.script(ScriptedOutput(raise_on_call=CppExecutorError("dead")))

    with pytest.raises(EngineFatalError):
        _run_core(server, _request())

    assert engine_failure_recorder.signals == [signal.SIGINT]
    assert len(engine_failure_recorder.terminations) == 1
