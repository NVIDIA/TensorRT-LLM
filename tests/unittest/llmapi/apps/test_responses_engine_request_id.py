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
"""Offline tests for the engine request id header of /v1/responses replies.

A Responses reply is named with a random ``resp_`` id, so the header is what
ties it to the server's records of the request, which use the engine's id.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import tensorrt_llm.serve.openai_server as server_module
from tensorrt_llm.executor.result import CompletionOutput

from .openai_server_stub import make_stub_openai_server

pytestmark = pytest.mark.cpu_only

_HEADER = "x-trtllm-engine-request-id"


class _FinishedGeneration:
    """A generation the engine finished in one step, under ``request_id``."""

    def __init__(self, request_id):
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

    async def aresult(self):
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._steps:
            raise StopAsyncIteration
        return self._steps.pop()


def _responses_client(tmp_path, monkeypatch, request_id):
    """A client of /v1/responses on a server whose engine names its request ``request_id``."""

    async def preprocess(**kwargs):
        return [1, 2, 3], kwargs["request"].to_sampling_params()

    monkeypatch.setattr(server_module, "responses_api_request_preprocess", preprocess)
    model_config = type("ModelConfig", (), {"model_type": "qwen3", "vocab_size": 1024})()
    server = make_stub_openai_server(
        tmp_path,
        "test-model",
        model_config,
        lambda **kwargs: _FinishedGeneration(request_id),
    )
    server.await_disconnected = AsyncMock()
    server._extract_metrics = AsyncMock()
    app = FastAPI()
    app.add_api_route("/v1/responses", server.openai_responses, methods=["POST"])
    return TestClient(app)


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_a_reply_names_its_engine_request_in_a_header(tmp_path, monkeypatch, stream):
    """The header carries the engine's id for the request."""
    client = _responses_client(tmp_path, monkeypatch, request_id=42)
    body = {"model": "test-model", "input": "What is 2 + 2?", "stream": stream}

    reply = client.post("/v1/responses", json=body)

    assert reply.status_code == 200, reply.text
    assert reply.headers.get(_HEADER) == "42"
