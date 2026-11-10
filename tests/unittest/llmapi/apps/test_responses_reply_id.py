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
"""Offline tests for the id of a /v1/responses reply from a single server.

A reply is named ``resp_<request id>_<16 hex>`` after the executor client id,
the id the server's perf metrics and post-processor hook records carry.
"""

import json
import pickle
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import tensorrt_llm.serve.openai_server as server_module
import tensorrt_llm.serve.responses_utils as responses_utils
from tensorrt_llm.executor.result import CompletionOutput

from .openai_server_stub import make_stub_openai_server

pytestmark = pytest.mark.cpu_only


class _FinishedGeneration:
    """A generation the engine finished in one step, under ``request_id``.

    Given ``postproc_params``, a postprocessing worker builds the reply: it
    works on a copy of them taken when the request is submitted, as the
    executor pickles them, and the server reads the worker's result.
    """

    def __init__(self, request_id, postproc_params=None):
        self.request_id = request_id
        self.id = request_id
        self.prompt_token_ids = [1, 2, 3]
        self.cached_tokens = 0
        self.outputs = [
            CompletionOutput(index=0, text="Four.", token_ids=[4, 5], finish_reason="stop")
        ]
        self.finished = self._done = True
        self.error = None
        self._steps = [self]
        self._worker_params = None
        if postproc_params is not None:
            self._worker_params = pickle.loads(pickle.dumps(postproc_params))

    def _postprocess_in_the_worker(self):
        if self._worker_params is not None:
            worker = self._worker_params
            self.outputs[0]._postprocess_result = worker.post_processor(self, worker.postproc_args)

    async def aresult(self):
        self._postprocess_in_the_worker()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._steps:
            raise StopAsyncIteration
        self._postprocess_in_the_worker()
        return self._steps.pop()


def _responses_client(tmp_path, request_ids=(42,), postprocessing_workers=False):
    """A client of /v1/responses on a server whose engine names its requests ``request_ids``."""
    request_ids = iter(request_ids)
    model_config = type("ModelConfig", (), {"model_type": "qwen3", "vocab_size": 1024})()
    server = make_stub_openai_server(
        tmp_path,
        "test-model",
        model_config,
        lambda **kwargs: _FinishedGeneration(next(request_ids), kwargs["_postproc_params"]),
    )
    server.await_disconnected = AsyncMock()
    server._extract_metrics = AsyncMock()
    if postprocessing_workers:
        server.generator.args.num_postprocess_workers = 1
        # The constructor turns storage off when there are postprocessing workers.
        server.enable_store = False
    app = FastAPI()
    app.add_api_route("/v1/responses", server.openai_responses, methods=["POST"])
    return TestClient(app)


def _without_rendering(monkeypatch):
    """Skip request preprocessing: every request is the prompt [1, 2, 3]."""

    async def preprocess(**kwargs):
        return [1, 2, 3], kwargs["request"].to_sampling_params()

    monkeypatch.setattr(server_module, "responses_api_request_preprocess", preprocess)


def _rendering_faked(monkeypatch):
    """Run the real request preprocessing, store included, with rendering faked.

    Returns the messages each request is rendered from, in order.
    """
    rendered = []

    def parse(messages, model_config, multimodal_server_config=None):
        rendered.append(list(messages))

        async def no_media():
            return None, None

        return list(messages), no_media(), {}, None

    async def render(resources, **kwargs):
        return SimpleNamespace(token_ids=[1, 2, 3], text=None)

    resources = SimpleNamespace(extension=SimpleNamespace(serialize_tool=lambda tool: tool))
    monkeypatch.setattr(responses_utils, "parse_chat_messages_coroutines", parse)
    monkeypatch.setattr(
        responses_utils, "RenderResources", SimpleNamespace(from_parts=lambda **kw: resources)
    )
    monkeypatch.setattr(responses_utils, "render_conversation", render)
    monkeypatch.setattr(responses_utils, "resolve_top_level_model_type", lambda cfg: "qwen3")
    monkeypatch.setattr(
        responses_utils, "add_thinking_budget_logits_processor", lambda *a, **k: None
    )
    return rendered


def _streamed_responses(reply):
    """The type and response id of each streamed event that carries the response."""
    events = [
        json.loads(line.removeprefix("data: "))
        for line in reply.text.splitlines()
        if line.startswith("data: ")
    ]
    return [(event["type"], event["response"]["id"]) for event in events if "response" in event]


def _reply_name(reply, stream):
    """The id a reply goes by, which every streamed event carrying the response repeats."""
    assert reply.status_code == 200, reply.text
    if not stream:
        return reply.json()["id"]
    events = _streamed_responses(reply)
    assert events[0][0] == "response.created", events
    assert events[-1][0] == "response.completed", events
    names = {name for _, name in events}
    assert len(names) == 1, events
    return names.pop()


@pytest.mark.parametrize(
    "postprocessing_workers", [False, True], ids=["in_server", "postprocessing_workers"]
)
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_a_reply_is_named_after_the_executor_client_id(
    tmp_path, monkeypatch, stream, postprocessing_workers
):
    """The reply is ``resp_<executor client id>_<16 hex>``, in every event of a stream.

    A postprocessing worker builds the reply, or a stream's events after the
    opening pair, from a copy of the request taken before the executor gave the
    request its id; it names the copy as the server names its own.
    """
    _without_rendering(monkeypatch)
    client = _responses_client(tmp_path, postprocessing_workers=postprocessing_workers)
    body = {"model": "test-model", "input": "What is 2 + 2?", "stream": stream}

    reply = client.post("/v1/responses", json=body)

    assert re.fullmatch(r"resp_42_[0-9a-f]{16}", _reply_name(reply, stream))
    assert "x-trtllm-engine-request-id" not in reply.headers


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_a_request_id_the_client_set_is_the_reply_id(tmp_path, monkeypatch, stream):
    _without_rendering(monkeypatch)
    client = _responses_client(tmp_path)
    body = {"model": "test-model", "input": "What is 2 + 2?", "stream": stream}

    reply = client.post("/v1/responses", json={**body, "request_id": "resp_client"})

    assert _reply_name(reply, stream) == "resp_client"


def test_a_follow_up_to_a_named_reply_continues_its_conversation(tmp_path, monkeypatch):
    """The stored turn, input included, is found under the reply's name.

    A request's input is stored before the executor gives the request its id.
    """
    monkeypatch.delenv("TRTLLM_RESPONSES_API_DISABLE_STORE", raising=False)
    rendered = _rendering_faked(monkeypatch)
    client = _responses_client(tmp_path, request_ids=(42, 43))

    first = client.post("/v1/responses", json={"model": "test-model", "input": "What is 2 + 2?"})
    first_name = _reply_name(first, stream=False)
    second = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "And 3 + 3?", "previous_response_id": first_name},
    )

    assert re.fullmatch(r"resp_42_[0-9a-f]{16}", first_name)
    assert re.fullmatch(r"resp_43_[0-9a-f]{16}", _reply_name(second, stream=False))
    assert [(message["role"], message["content"]) for message in rendered[1]] == [
        ("user", "What is 2 + 2?"),
        ("assistant", "Four."),
        ("user", "And 3 + 3?"),
    ]
