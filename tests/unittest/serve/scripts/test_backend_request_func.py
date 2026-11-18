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
"""The benchmark client counts a stream that ends with an SSE error event as failed."""

import contextlib
import json

import pytest

from tensorrt_llm.serve.scripts.backend_request_func import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    async_request_openai_completions,
)

pytestmark = pytest.mark.cpu_only

ERROR_MESSAGE = "A rank failure ended the request"
ERROR_EVENT = {
    "error": {"message": ERROR_MESSAGE, "type": "server_error", "code": None, "param": None}
}
USAGE_EVENT = {"choices": [], "usage": {"completion_tokens": 1}}


class _FakeContent:
    def __init__(self, body: bytes):
        self._body = body

    async def iter_any(self):
        yield self._body


class _FakeResponse:
    status = 200
    reason = "OK"

    def __init__(self, events: list):
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        self.content = _FakeContent((body + "data: [DONE]\n\n").encode())


class _FakeSession:
    """An aiohttp session stand-in whose POST streams the given SSE events."""

    def __init__(self, events: list):
        self._events = events

    @contextlib.asynccontextmanager
    async def post(self, url, json, headers):
        yield _FakeResponse(self._events)


ENDPOINTS = {
    "completions": (
        async_request_openai_completions,
        "http://localhost/v1/completions",
        {"choices": [{"text": "Hello"}]},
    ),
    "chat": (
        async_request_openai_chat_completions,
        "http://localhost/v1/chat/completions",
        {"choices": [{"delta": {"content": "Hello"}}]},
    ),
}


async def _run(endpoint: str, events_after_token: list):
    request_func, api_url, token_event = ENDPOINTS[endpoint]
    request = RequestFuncInput(
        prompt="Hi", api_url=api_url, prompt_len=1, output_len=8, model="model"
    )
    session = _FakeSession([token_event, *events_after_token])
    return await request_func(request, streaming=True, session=session)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_error_event_after_first_token_fails_request(endpoint):
    output = await _run(endpoint, [ERROR_EVENT])

    assert not output.success
    assert ERROR_MESSAGE in output.error


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_stream_without_error_event_succeeds(endpoint):
    output = await _run(endpoint, [USAGE_EVENT])

    assert output.success
    assert output.error == ""
    assert output.generated_text == "Hello"
    assert output.output_tokens == 1
