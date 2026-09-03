# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _msgpack_encoder
from tensorrt_llm.serve.openai_server import _MsgspecRequest, _MsgspecRoute

PAYLOAD = {
    "model": "m",
    "prompt": [1, 2, 3],
    "max_tokens": 4,
    "disaggregated_params": {"request_type": "context_only"},
}


def _request(body: bytes, headers: dict) -> _MsgspecRequest:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/completions",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return _MsgspecRequest(scope, receive)


@pytest.mark.asyncio
async def test_orchestrator_body_round_trips_through_worker():
    request = _request(_msgpack_encoder.encode(PAYLOAD), MSGPACK_HEADERS)
    assert await request.json() == PAYLOAD


@pytest.mark.asyncio
async def test_json_body_without_the_header_is_unchanged():
    headers = {"Content-Type": "application/json"}
    request = _request(json.dumps(PAYLOAD).encode(), headers)
    assert await request.json() == PAYLOAD


@pytest.mark.asyncio
async def test_empty_body_decodes_to_an_empty_dict():
    assert await _request(b"", MSGPACK_HEADERS).json() == {}


class _Body(BaseModel):
    value: int


def test_malformed_msgpack_is_a_422():
    # _MsgspecRoute is installed on every trtllm-serve app, so any client can
    # set the header and post arbitrary bytes: that must be a validation
    # error, not an unhandled decode failure.
    app = FastAPI()
    app.router.route_class = _MsgspecRoute

    @app.post("/echo")
    async def echo(body: _Body):
        return {"value": body.value}

    with TestClient(app) as client:
        response = client.post("/echo", content=b"\xc1garbage", headers=MSGPACK_HEADERS)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"
