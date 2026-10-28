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
"""A degenerate generation reaches the client of every API as an error.

A model whose serving extension declares degenerate tokens gets their runs as
stop words on every generation request, and a generation that stops on one
fails with a RequestError carrying ``degenerate_generation_message`` (see
test_degenerate_token_runs.py). These tests drive the real route handlers on a
server built with ``object.__new__``: the engine is a stub whose result fails
the way the result layer fails it, in the server process or, prefixed, in a
postprocessing worker. CPU-only.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.executor.utils import RequestError, degenerate_generation_message
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve import serving_extensions
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.render import chat as render_chat_module
from tensorrt_llm.serve.serving_extensions import (
    DEGENERATE_TOKEN_RUN_LENGTH,
    DegenerateTokens,
    ServingExtension,
    register_serving_extension,
)

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
DECLARED_MODEL_TYPE = "degenerate-errors-test-model"
PAD = 2
LAST_ID = 129279
RUN = DEGENERATE_TOKEN_RUN_LENGTH
MESSAGE = degenerate_generation_message(LAST_ID, RUN)
# How the failure arrives when a postprocessing worker built the response.
WORKER_MESSAGE = f"Postprocessing error: {MESSAGE}"


@pytest.fixture(autouse=True)
def declaring_extension():
    """A model type whose extension declares two degenerate tokens."""

    @register_serving_extension(model_types=(DECLARED_MODEL_TYPE,))
    class DeclaringExtension(ServingExtension):
        def degenerate_tokens(self, hf_config):
            return DegenerateTokens(token_ids=(hf_config.pad_token_id, hf_config.vocab_size - 1))

    try:
        yield
    finally:
        serving_extensions._BY_MODEL_TYPE.pop(DECLARED_MODEL_TYPE, None)


class _DeclaredConfig:
    model_type = DECLARED_MODEL_TYPE
    pad_token_id = PAD
    vocab_size = LAST_ID + 1


class _UndeclaredConfig:
    model_type = "degenerate-errors-test-undeclared"
    pad_token_id = PAD
    vocab_size = LAST_ID + 1


def _engine_stop_words(sampling_params):
    """The stop words the engine gets, once the LLM has set the params up."""
    tokenizer = SimpleNamespace(
        eos_token_id=1, pad_token_id=None, encode=lambda text, add_special_tokens=False: [99]
    )
    sampling_params._setup(tokenizer, None, None)
    return sampling_params._get_stop_words()


def _step(*frames):
    """A streamed step whose frames a postprocessing worker already formatted."""
    return SimpleNamespace(outputs=[SimpleNamespace(_postprocess_result=list(frames))])


class _FailingPromise:
    """A generation result that yields ``steps``, then fails on a degenerate run."""

    def __init__(self, steps, message):
        self._steps = list(steps)
        self._message = message
        self.prompt_token_ids = [1, 2, 3]
        self.finished = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._steps:
            self._fail()
        return self._steps.pop(0)

    async def aresult(self):
        self._fail()

    def __await__(self):
        return self.aresult().__await__()

    def _fail(self):
        self.finished = True
        raise RequestError(self._message)


@pytest.fixture
def make_server(monkeypatch):
    """A server for ``config`` whose engine records each request and fails it."""

    async def fake_apply_chat_template(**kwargs):
        return "rendered prompt"

    monkeypatch.setattr(render_chat_module, "async_apply_chat_template", fake_apply_chat_template)

    async def fake_responses_preprocess(*, request, **kwargs):
        # Rendering is not under test here; the stop runs are added after it.
        return [1, 2, 3], request.to_sampling_params()

    monkeypatch.setattr(
        openai_server_module, "responses_api_request_preprocess", fake_responses_preprocess
    )
    monkeypatch.setattr(
        openai_server_module,
        "tokenize_harmony_chat_request",
        lambda request, harmony_adapter: [1, 2, 3],
    )

    def make(config=_DeclaredConfig, *, postproc_workers=0, steps=(), message=MESSAGE):
        submitted = []

        def generate_async(*, inputs, sampling_params, **kwargs):
            submitted.append(sampling_params)
            return _FailingPromise(steps, message)

        server = object.__new__(OpenAIServer)
        server.model = MODEL
        server.allow_request_chat_template = False
        server.model_config = config()
        server.processor = None
        server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=config.vocab_size))
        server.chat_template = None
        server.tool_parser = None
        server.tool_call_id_type = "random"
        server.multimodal_server_config = None
        server.use_harmony = False
        # Serves /harmony/v1/chat/completions, the chat route of gpt-oss models.
        server.harmony_adapter = SimpleNamespace(get_stop_tokens=lambda: [200002])
        server.enable_store = False
        server.conversation_store = None
        server._input_proc_executor = None
        server._adjusted_steady_clock = SimpleNamespace(now=lambda: 0.0)
        server.generator = SimpleNamespace(
            args=SimpleNamespace(
                gather_generation_logits=False,
                reasoning_parser=None,
                backend="pytorch",
                guided_decoding_backend=None,
                num_postprocess_workers=postproc_workers,
            ),
            generate_async=generate_async,
            input_processor=lambda prompt, sampling_params: ([1, 2, 3], None),
        )
        server.await_disconnected = AsyncMock()

        app = FastAPI()
        app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
        app.add_api_route("/harmony/v1/chat/completions", server.chat_harmony, methods=["POST"])
        app.add_api_route("/v1/completions", server.openai_completion, methods=["POST"])
        app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
        app.add_api_route("/v1/responses", server.openai_responses, methods=["POST"])
        # A Responses stream re-raises its failure after the terminal events.
        client = TestClient(app, raise_server_exceptions=False)
        return SimpleNamespace(server=server, client=client, submitted=submitted)

    return make


def _chat(**overrides):
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}
    body.update(overrides)
    return body


def _sse_data(text):
    return [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]


def _sse_events(text):
    """(event, data) pairs of an SSE body whose frames name their event."""
    events = []
    for frame in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


# --------------------------------------------------------------------------
# The declared runs reach the engine; undeclared models are unaffected
# --------------------------------------------------------------------------


# Every route that generates, with the body of a minimal request.
ROUTES = [
    ("/v1/chat/completions", _chat()),
    ("/harmony/v1/chat/completions", _chat()),
    ("/v1/completions", {"model": MODEL, "prompt": "hi", "max_tokens": 64}),
    ("/v1/responses", {"model": MODEL, "input": "hi"}),
]
CHAT_ROUTES = ["/v1/chat/completions", "/harmony/v1/chat/completions"]


@pytest.mark.parametrize(("path", "body"), ROUTES)
def test_declared_runs_are_stop_words_of_every_request(make_server, path, body):
    server = make_server()
    server.client.post(path, json=body)
    [sampling_params] = server.submitted
    assert _engine_stop_words(sampling_params)[-2:] == [[PAD] * RUN, [LAST_ID] * RUN]


@pytest.mark.parametrize(
    ("path", "body", "stop_words"),
    [
        ("/v1/chat/completions", _chat(stop_token_ids=[7]), [[7]]),
        ("/harmony/v1/chat/completions", _chat(), [[200002]]),
        ("/v1/completions", {"model": MODEL, "prompt": "hi", "max_tokens": 64}, []),
        ("/v1/responses", {"model": MODEL, "input": "hi"}, []),
    ],
)
def test_models_that_declare_nothing_keep_their_stop_words(make_server, path, body, stop_words):
    server = make_server(_UndeclaredConfig)
    server.client.post(path, json=body)
    [sampling_params] = server.submitted
    assert sampling_params._degenerate_token_runs == ()
    assert _engine_stop_words(sampling_params) == stop_words


# --------------------------------------------------------------------------
# Each API's client gets an error
# --------------------------------------------------------------------------


@pytest.mark.parametrize("message", [MESSAGE, WORKER_MESSAGE])
@pytest.mark.parametrize(("path", "body"), ROUTES)
def test_non_streamed_requests_fail_with_a_server_error(make_server, path, body, message):
    response = make_server(message=message).client.post(path, json=body)
    assert response.status_code == 500
    error = response.json()
    assert (error["type"], error["code"]) == ("InternalServerError", 500)
    assert MESSAGE in error["message"]


def test_openai_client_raises_for_a_non_streamed_chat(make_server):
    client = openai.OpenAI(
        api_key="unused",
        base_url="http://testserver/v1",
        http_client=make_server().client,
        max_retries=0,
    )
    with pytest.raises(openai.InternalServerError, match="degenerated"):
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("path", CHAT_ROUTES)
@pytest.mark.parametrize(
    ("postproc_workers", "steps", "message"),
    [
        pytest.param(0, (), MESSAGE, id="in_server"),
        pytest.param(1, (_step('data: {"id": "chunk-1"}\n\n'),), WORKER_MESSAGE, id="worker"),
    ],
)
def test_a_chat_stream_ends_with_an_error_event(
    make_server, path, postproc_workers, steps, message
):
    server = make_server(postproc_workers=postproc_workers, steps=steps, message=message)
    response = server.client.post(path, json=_chat(stream=True))

    assert response.status_code == 200
    data = _sse_data(response.text)
    # Whatever was streamed before the run stays as it was ...
    assert data[: len(steps)] == ['{"id": "chunk-1"}'][: len(steps)]
    # ... and the stream ends with an error rather than a finished reply.
    error = json.loads(data[-2])["error"]
    assert error["type"] == "server_error"
    assert MESSAGE in error["message"]
    assert data[-1] == "[DONE]"
    assert not any('"finish_reason"' in chunk for chunk in data)


def test_openai_client_raises_for_a_streamed_chat(make_server):
    client = openai.OpenAI(
        api_key="unused",
        base_url="http://testserver/v1",
        http_client=make_server().client,
        max_retries=0,
    )
    stream = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "hi"}], stream=True
    )
    with pytest.raises(openai.APIError, match="degenerated"):
        list(stream)


@pytest.mark.parametrize(
    ("postproc_workers", "message"),
    [(0, MESSAGE), (1, WORKER_MESSAGE)],
    ids=["in_server", "worker"],
)
def test_an_anthropic_stream_ends_with_an_error_event(make_server, postproc_workers, message):
    server = make_server(postproc_workers=postproc_workers, message=message)
    body = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    response = server.client.post("/v1/messages", json=body)

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error"]["type"] == "api_error"
    assert MESSAGE in events[-1][1]["error"]["message"]
    assert "message_stop" not in [name for name, _ in events]


def test_a_non_streamed_anthropic_request_fails_with_an_api_error(make_server):
    body = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
    response = make_server().client.post("/v1/messages", json=body)

    assert response.status_code == 500
    error = response.json()
    assert error["type"] == "error"
    assert error["error"]["type"] == "api_error"
    assert MESSAGE in error["error"]["message"]


@pytest.mark.parametrize(
    ("postproc_workers", "steps", "message"),
    [
        pytest.param(0, (), MESSAGE, id="in_server"),
        pytest.param(
            1,
            (
                _step(
                    "event: response.output_item.added\n"
                    'data: {"type": "response.output_item.added", "sequence_number": 2}\n\n'
                ),
            ),
            WORKER_MESSAGE,
            id="worker",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_responses_stream_ends_failed_without_output(
    make_server, postproc_workers, steps, message
):
    server = make_server(postproc_workers=postproc_workers, steps=steps, message=message)
    request = ResponsesRequest(model=MODEL, input="hi", stream=True)
    raw_request = SimpleNamespace(state=SimpleNamespace(no_client_connection=True), headers={})

    response = await server.server.openai_responses(request, raw_request)

    assert response.status_code == 200
    # The route re-raises the failure after the terminal events; the frames
    # sent before that are what the client receives. (An HTTP test client may
    # drop a streamed body whose app raised, so the body is read here.)
    frames = []
    with pytest.raises(RequestError):
        async for frame in response.body_iterator:
            frames.append(frame if isinstance(frame, str) else frame.decode())
    text = "".join(frames)
    names = [name for name, _ in _sse_events(text)]
    # The item opened before the run is never completed, so a client that keeps
    # items on response.output_item.done keeps nothing of this turn.
    assert names == [
        "response.created",
        "response.in_progress",
        *(["response.output_item.added"] if steps else []),
        "error",
        "response.failed",
    ]
    failed = _sse_events(text)[-1][1]["response"]
    assert failed["status"] == "failed"
    assert failed["output"] == []
    assert failed["error"]["code"] == "server_error"
    assert MESSAGE in failed["error"]["message"]
