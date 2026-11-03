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
"""The dialect-neutral errors and the serving error boundary (``tensorrt_llm/serve/_base/errors.py``).

The boundary renders the typed errors through the registry of
``tensorrt_llm/serve/exception_handling/register.py`` and maps a handler's other exceptions
with its ladder. ``test_chat_wrapper_answers_as_before`` runs the chat completion handler on a
server built through the real constructor and checks that each of its former ``except``
branches still gives the same answer.
"""

import asyncio
import inspect
import json
import logging
import signal
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import ScriptedOutput

from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.logger import logger
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve._base.errors import (
    CHAT_LADDER,
    ContextLengthExceededError,
    DialectNeutralError,
    EngineFatalError,
    ErrorRule,
    InternalServingError,
    InvalidRequestError,
    ServingError,
    UnavailableError,
    UpstreamError,
    error_body,
    neutral_error,
    serving_error,
    serving_error_boundary,
)
from tensorrt_llm.serve.exception_handling import register as register_module
from tensorrt_llm.serve.exception_handling.register import (
    ANTHROPIC,
    OPENAI,
    ErrorRenderers,
    default_error_renderers,
    install_exception_handlers,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

# threadleak is off: TestClients are created in the test body and closed by the serve_env
# fixture's teardown, which runs after the leak check.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

CONTEXT_LENGTH_MESSAGE = context_length_exceeded_message(2048, 4096)
MODEL = "fake-model"
REQUEST_RULE = (
    "serving_error_boundary must find the request by its Request annotation and read "
    "app.state.error_renderers only when the scope has an app"
)


class _Body(BaseModel):
    value: int


def _openai_body(message, err_type, code):
    return {"object": "error", "message": message, "type": err_type, "param": None, "code": code}


def _anthropic_body(message, status):
    error_type = "invalid_request_error" if 400 <= status < 500 else "api_error"
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _json(response):
    return json.loads(response.body)


@pytest.fixture
def error_records():
    """Messages of the ERROR records the TRT-LLM logger emits during the test.

    The library logger does not propagate, so ``caplog`` would see nothing; a handler is
    attached instead. ERROR passes the logger's default level.
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


class _Recording(ErrorRenderers):
    """A registry that records what it is asked to render."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def render(self, dialect, request, error):
        self.calls.append((dialect, request, error))
        return JSONResponse(status_code=499, content={"recorded": True})


# ---------------------------------------------------------------------------
# The dialect-neutral error types
# ---------------------------------------------------------------------------


def test_neutral_error_defaults():
    invalid = InvalidRequestError("bad")
    assert (invalid.status, invalid.err_type, invalid.code, invalid.message, str(invalid)) == (
        400,
        "BadRequestError",
        None,
        "bad",
        "bad",
    )

    too_long = ContextLengthExceededError("too long")
    assert isinstance(too_long, InvalidRequestError)
    assert (too_long.status, too_long.err_type, too_long.code) == (
        400,
        "BadRequestError",
        "context_length_exceeded",
    )

    assert (UpstreamError("no", 404).status, UpstreamError("no", 404).err_type) == (
        404,
        "BadRequestError",
    )
    assert (UpstreamError("no", 500).status, UpstreamError("no", 500).err_type) == (
        500,
        "InternalServerError",
    )
    assert (UpstreamError("no", None).status, UpstreamError("no", None).err_type) == (
        502,
        "InternalServerError",
    )

    busy = UnavailableError("busy")
    assert (busy.status, busy.err_type, busy.retryable) == (503, "InternalServerError", True)
    assert UnavailableError("busy", retryable=False).retryable is False

    assert (InternalServingError("oops").status, InternalServingError("oops").err_type) == (
        500,
        "InternalServerError",
    )

    for error in (invalid, too_long, busy, UpstreamError("no"), InternalServingError("oops")):
        assert isinstance(error, DialectNeutralError)
    assert not issubclass(EngineFatalError, DialectNeutralError)
    assert not issubclass(ServingError, DialectNeutralError)

    expected_types = {
        (400, "bad"): InvalidRequestError,
        (400, CONTEXT_LENGTH_MESSAGE): ContextLengthExceededError,
        (499, "gone"): UpstreamError,
        (500, "oops"): InternalServingError,
        (502, "upstream"): UpstreamError,
        (503, "busy"): UnavailableError,
    }
    for (status, message), expected in expected_types.items():
        error = neutral_error(status, message)
        assert type(error) is expected, (status, message)
        assert (error.status, error.message) == (status, message)


NEUTRAL_CASES = [
    pytest.param(serving_error("bad"), 400, "BadRequestError", 400, id="serving-error"),
    pytest.param(
        serving_error("gone", "NotFoundError", HTTPStatus.NOT_FOUND),
        404,
        "NotFoundError",
        404,
        id="serving-404",
    ),
    pytest.param(InvalidRequestError("bad"), 400, "BadRequestError", 400, id="invalid-request"),
    pytest.param(
        ContextLengthExceededError("too long"),
        400,
        "BadRequestError",
        "context_length_exceeded",
        id="context-length",
    ),
    pytest.param(
        UpstreamError("worker said no", 404), 404, "BadRequestError", 404, id="upstream-4xx"
    ),
    pytest.param(
        UpstreamError("worker down", None), 502, "InternalServerError", 502, id="upstream-502"
    ),
    pytest.param(
        UpstreamError("client went away", 499), 499, "BadRequestError", 499, id="upstream-499"
    ),
    pytest.param(UnavailableError("busy"), 503, "InternalServerError", 503, id="unavailable"),
    pytest.param(InternalServingError("oops"), 500, "InternalServerError", 500, id="internal"),
]


@pytest.mark.parametrize("dialect", [OPENAI, ANTHROPIC])
@pytest.mark.parametrize("error, status, err_type, code", NEUTRAL_CASES)
def test_boundary_renders_serving_and_neutral_errors(dialect, error, status, err_type, code):
    @serving_error_boundary(dialect, renderers=default_error_renderers())
    async def handler(raw_request: Request) -> Response:
        raise error

    response = asyncio.run(handler(None))
    message = error.body.message if isinstance(error, ServingError) else error.message
    if dialect == OPENAI:
        expected = _openai_body(message, err_type, code)
    else:
        expected = _anthropic_body(message, status)
    assert (response.status_code, _json(response)) == (status, expected)


@pytest.mark.parametrize(
    "message", ["bad", CONTEXT_LENGTH_MESSAGE], ids=["plain", "context-length"]
)
def test_openai_rendering_equals_create_error_response(message):
    rendered = default_error_renderers().render(OPENAI, None, InvalidRequestError(message))
    reference = OpenAIServer.create_error_response(message)
    assert (rendered.status_code, rendered.body) == (reference.status_code, reference.body), (
        "the OPENAI rendering of a dialect-neutral error must use error_body "
        "(tensorrt_llm/serve/_base/errors.py) exactly as create_error_response does"
    )
    if message is CONTEXT_LENGTH_MESSAGE:
        assert _json(rendered)["code"] == "context_length_exceeded"


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_boundary_finds_the_request_by_annotation(monkeypatch):
    registry = _Recording()
    raw = Request({"type": "http", "headers": []})

    class Serving:
        @serving_error_boundary(OPENAI, renderers=registry)
        async def typed(self, request: _Body, raw_request: Request) -> Response:
            raise InvalidRequestError("bad")

        @serving_error_boundary(OPENAI, renderers=registry)
        async def stringly(self, request: _Body, raw_request: "Request") -> Response:
            raise InvalidRequestError("bad")

        @serving_error_boundary(OPENAI)
        async def unpinned(self, request: _Body, raw_request: Request) -> Response:
            raise InvalidRequestError("bad")

    body = _Body(value=1)
    asyncio.run(Serving().typed(body, raw))
    asyncio.run(Serving().stringly(body, raw_request=raw))
    asyncio.run(Serving().typed(body, None))

    # An app that never installed the registry is a wiring error, named as such rather than
    # silently answered with the default renderers.
    bare = Request({"type": "http", "headers": [], "app": FastAPI()})
    with pytest.raises(RuntimeError, match="install_exception_handlers"):
        asyncio.run(Serving().unpinned(body, bare))
    assert [call[1] for call in registry.calls] == [raw, raw, None], REQUEST_RULE
    assert all(call[1] is not body for call in registry.calls), REQUEST_RULE

    # Without a registry argument: the request's app registry when the scope has an app,
    # else the default renderers (patched here to tell them apart).
    fallback = _Recording()
    monkeypatch.setattr(register_module, "default_error_renderers", lambda **kwargs: fallback)
    app = FastAPI()
    app_registry = _Recording()
    install_exception_handlers(app, renderers=app_registry, default_dialect=OPENAI)

    asyncio.run(Serving().unpinned(body, None))
    asyncio.run(Serving().unpinned(body, Request({"type": "http", "headers": []})))
    assert len(fallback.calls) == 2 and app_registry.calls == [], REQUEST_RULE

    with_app = Request({"type": "http", "headers": [], "app": app})
    asyncio.run(Serving().unpinned(body, with_app))
    assert len(fallback.calls) == 2, REQUEST_RULE
    assert [call[1] for call in app_registry.calls] == [with_app], REQUEST_RULE


def test_boundary_keeps_signature_and_docstring():
    class Plain:
        async def handle(self, request: _Body, raw_request: Request) -> Response:
            """Handle a request."""
            return JSONResponse(content={"ok": True})

    class Guarded:
        @serving_error_boundary(OPENAI)
        async def handle(self, request: _Body, raw_request: Request) -> Response:
            """Handle a request."""
            return JSONResponse(content={"ok": True})

    assert inspect.signature(Guarded.handle) == inspect.signature(Plain.handle)
    assert inspect.signature(Guarded().handle) == inspect.signature(Plain().handle)
    assert (Guarded.handle.__name__, Guarded.handle.__doc__) == ("handle", "Handle a request.")

    def operation(handler):
        app = FastAPI()
        app.add_api_route("/x", handler, methods=["POST"])
        return app.openapi()["paths"]["/x"]["post"]

    plain, guarded = operation(Plain().handle), operation(Guarded().handle)
    for key in ("operationId", "summary", "description", "requestBody"):
        assert guarded.get(key) == plain.get(key), key
    assert guarded == plain


def test_boundary_engine_fatal():
    @serving_error_boundary(OPENAI, on_engine_fatal=lambda: None)
    async def answered() -> Response:
        raise EngineFatalError()

    @serving_error_boundary(OPENAI)
    async def propagated() -> Response:
        raise EngineFatalError()

    assert asyncio.run(answered()) is None
    with pytest.raises(EngineFatalError):
        asyncio.run(propagated())


@pytest.mark.parametrize("ladder", [CHAT_LADDER], ids=["chat"])
def test_boundary_ladders_match_today(ladder, error_records):
    @serving_error_boundary(OPENAI, ladder=ladder, renderers=default_error_renderers())
    async def handler(error: Exception) -> Response:
        raise error

    response = asyncio.run(handler(ValueError("bad")))
    assert (response.status_code, _json(response)) == (
        400,
        _openai_body("bad", "BadRequestError", 400),
    )
    assert error_records == [], "a ValueError is answered without a log"

    response = asyncio.run(handler(RuntimeError("boom")))
    assert (response.status_code, _json(response)) == (
        400,
        _openai_body("boom", "BadRequestError", 400),
    )
    assert len(error_records) == 1 and "RuntimeError: boom" in error_records[0], error_records


def test_boundary_rule_options(error_records):
    ladder = (
        ErrorRule(KeyError, HTTPStatus.NOT_FOUND, "NotFoundError", log="message"),
        ErrorRule(
            (TypeError, LookupError), HTTPStatus.INTERNAL_SERVER_ERROR, "InternalServerError"
        ),
    )

    @serving_error_boundary(ANTHROPIC, ladder=ladder, renderers=default_error_renderers())
    async def handler(error: Exception) -> Response:
        raise error

    response = asyncio.run(handler(KeyError("k")))
    assert (response.status_code, _json(response)) == (404, _anthropic_body("'k'", 404))
    assert len(error_records) == 1 and error_records[0].endswith(" 'k'"), (
        "log='message' logs str(error) at ERROR",
        error_records,
    )

    response = asyncio.run(handler(IndexError("i")))
    assert (response.status_code, _json(response)) == (500, _anthropic_body("i", 500))
    assert len(error_records) == 1, ("log='none' logs nothing", error_records)


def test_boundary_reraises_unmatched():
    @serving_error_boundary(OPENAI, ladder=(ErrorRule(ValueError, HTTPStatus.BAD_REQUEST),))
    async def handler() -> Response:
        raise KeyError("unmatched")

    with pytest.raises(KeyError):
        asyncio.run(handler())


# ---------------------------------------------------------------------------
# The chat completion handler through the boundary
# ---------------------------------------------------------------------------

NAMED_TOOL_ON_TOKEN_IDS_MESSAGE = (
    "tool_choice with a named function is not supported alongside `prompt_token_ids`; "
    "pass `messages` instead."
)
_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}


def _chat_body(**overrides):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "prompt_token_ids": [1, 2, 3],
    }
    body.update(overrides)
    return body


def _chat_response(**choice_fields) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-boundary-test",
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


class _UnserializableResponse(ChatCompletionResponse):
    def model_dump(self, *args, **kwargs):
        raise RuntimeError("boom")


@pytest.fixture
def engine_failure_recorder(monkeypatch):
    """Record the two side effects of an engine failure instead of running them."""
    recorder = SimpleNamespace(signals=[], terminations=[])
    monkeypatch.setattr(signal, "raise_signal", recorder.signals.append)
    monkeypatch.setattr(
        openai_server, "_record_generator_termination", recorder.terminations.append
    )
    return recorder


def test_chat_wrapper_answers_as_before(serve_env, error_records, engine_failure_recorder):
    server = serve_env.build_openai_server(AggConfig())
    server.await_disconnected = AsyncMock()
    client = serve_env.client(server.app)

    # The core's rejection: 400 with the body create_error_response gives.
    expected = OpenAIServer.create_error_response(NAMED_TOOL_ON_TOKEN_IDS_MESSAGE)
    response = client.post(
        "/v1/chat/completions",
        json=_chat_body(
            tools=[_TOOL], tool_choice={"type": "function", "function": {"name": "get_weather"}}
        ),
    )
    assert (response.status_code, response.content) == (400, expected.body)
    assert error_records == []

    # A response that cannot be encoded (ValueError): 400, no traceback.
    server._create_chat_response = AsyncMock(
        return_value=_chat_response(avg_decoded_tokens_per_iter=float("nan"))
    )
    response = client.post("/v1/chat/completions", json=_chat_body())
    message = response.json()["message"]
    assert message.startswith("Out of range float values are not JSON compliant")
    assert (response.status_code, response.content) == (
        400,
        OpenAIServer.create_error_response(message).body,
    )
    assert error_records == []

    # Any other failure while answering (RuntimeError): 400, one traceback at ERROR.
    server._create_chat_response = AsyncMock(
        return_value=_UnserializableResponse(**_chat_response().model_dump())
    )
    response = client.post("/v1/chat/completions", json=_chat_body())
    assert (response.status_code, response.content) == (
        400,
        OpenAIServer.create_error_response("boom").body,
    )
    assert len(error_records) == 1 and "RuntimeError: boom" in error_records[0], error_records

    # An engine failure: HTTP 200 with body null, after SIGINT.
    server._create_chat_response = None
    server.generator.script(ScriptedOutput(raise_on_call=CppExecutorError("dead")))
    response = client.post("/v1/chat/completions", json=_chat_body())
    assert (response.status_code, response.content) == (200, b"null")
    assert engine_failure_recorder.signals == [signal.SIGINT]
    assert len(engine_failure_recorder.terminations) == 1


def test_error_body_is_the_only_openai_body_builder():
    """A neutral error's OpenAI body is error_body's for the same three values."""
    # 499 has no HTTPStatus member: the body is built from the status itself and renders.
    error = UpstreamError("client went away", 499)
    rendered = default_error_renderers().render(OPENAI, None, error)
    assert rendered.status_code == 499
    assert _json(rendered) == error_body(error.message, error.err_type, error.status).model_dump()
