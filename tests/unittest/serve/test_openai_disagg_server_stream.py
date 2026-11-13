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
"""The disaggregated server pulls a stream's first chunk before it sends the response headers.

A generation server that fails before its first token (for example on a KV-cache transfer error)
then becomes an HTTP error response instead of a 200 whose body is cut off.
"""

from types import SimpleNamespace

import aiohttp
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.datastructures import Headers
from test_common.serve.app_factory import DisaggConfig

from tensorrt_llm.serve.openai_protocol import CompletionRequest

pytestmark = pytest.mark.cpu_only


def _raw_request():
    return SimpleNamespace(
        headers=Headers(headers={}), state=SimpleNamespace(server_arrival_time=0.0)
    )


@pytest.mark.asyncio
async def test_stream_failing_before_first_chunk_returns_http_error(serve_env):
    async def failing_generation_stream():
        raise aiohttp.ClientPayloadError("Response payload is not completed")
        yield b"unreachable"

    async def entry_point(request, hooks):
        return failing_generation_stream()

    server = serve_env.build_disagg_server(DisaggConfig())
    wrapper = server._wrap_entry_point(entry_point, CompletionRequest)
    request = CompletionRequest(model="test-model", prompt="hello", stream=True)

    with pytest.raises(HTTPException) as exc_info:
        await wrapper(request, _raw_request())

    assert exc_info.value.status_code == 500
    assert "Response payload is not completed" in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"], []])
async def test_stream_forwards_every_chunk(serve_env, chunks):
    closed = []

    async def generation_stream():
        try:
            for chunk in chunks:
                yield chunk
        finally:
            closed.append(True)

    async def entry_point(request, hooks):
        return generation_stream()

    server = serve_env.build_disagg_server(DisaggConfig())
    wrapper = server._wrap_entry_point(entry_point, CompletionRequest)
    request = CompletionRequest(model="test-model", prompt="hello", stream=True)

    response = await wrapper(request, _raw_request())

    assert isinstance(response, StreamingResponse)
    assert [chunk async for chunk in response.body_iterator] == chunks
    assert closed == [True]
