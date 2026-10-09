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
"""The worker's streaming completion turns an executor-side request failure into an SSE error.

This is the serving half of the gen-only benchmark's insufficient-KV fail-fast:
the executor fails the request with an error response, and the client must see
that error end its stream rather than wait on a stream that never ends.
"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.utils import ErrorResponse
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.serve.openai_protocol import CompletionRequest
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

_KV_ERROR = (
    "Insufficient KV cache for gen-only benchmark mode: one or more requests "
    "are waiting for KV cache allocation"
)


class _FailingEngine:
    """LLM stand-in that fails every request the way the executor fails one.

    The error response travels through the real client-side result and the real
    per-request error handler, exactly as a worker's error response would.
    """

    def __init__(self, error_msg: str) -> None:
        self._error_msg = error_msg
        # The production handler that turns the error response into the
        # exception the server's stream generator sees. Held here because the
        # result only keeps a weak reference to it.
        self._worker = object.__new__(BaseWorker)
        self.args = SimpleNamespace(
            num_postprocess_workers=0,
            gather_generation_logits=False,
            backend="pytorch",
            reasoning_parser=None,
            cache_transceiver_config=None,
        )
        self.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=32000))

    def generate_async(self, inputs, sampling_params, _postproc_params, streaming, **_kwargs):
        request = GenerationRequest(
            inputs["prompt_token_ids"],
            sampling_params,
            streaming=streaming,
            postproc_params=_postproc_params,
        )
        request.set_id(1)
        result = GenerationResult(
            request, background_error_handler=self._worker._handle_background_error
        )
        result.aqueue.put(ErrorResponse(request.id, self._error_msg, request.id))
        return RequestOutput._from_generation_result(result)


def _bare_server(engine: _FailingEngine) -> OpenAIServer:
    server = object.__new__(OpenAIServer)
    server.generator = engine
    server.tokenizer = engine.tokenizer
    server.model = "test-model"
    server.model_config = None
    server._internal_disagg_auth_key = None
    server._adjusted_steady_clock = AdjustedSteadyClock(time_source=time.monotonic)
    return server


@pytest.mark.asyncio
async def test_a_failed_request_ends_the_completion_stream_with_an_error_event():
    server = _bare_server(_FailingEngine(_KV_ERROR))
    raw_request = SimpleNamespace(state=SimpleNamespace(no_client_connection=True), headers={})
    request = CompletionRequest(model="test-model", prompt=[1, 2, 3], max_tokens=4, stream=True)

    response = await server.openai_completion(request, raw_request)

    assert isinstance(response, StreamingResponse), getattr(response, "body", response)
    chunks = [chunk async for chunk in response.body_iterator]
    await asyncio.sleep(0)  # let the disconnect watcher task finish
    assert chunks[0].startswith("data: ")
    error = json.loads(chunks[0][len("data: ") :])["error"]
    assert error["message"] == _KV_ERROR
    assert error["type"] == "server_error"
    assert chunks[1] == "data: [DONE]\n\n"
