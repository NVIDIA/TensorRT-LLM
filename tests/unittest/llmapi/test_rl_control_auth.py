# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.rl_control_auth import RL_CONTROL_AUTH_HEADER, build_rl_control_auth_headers


def _make_server(enabled: bool) -> OpenAIServer:
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    server.generator = object.__new__(AsyncLLM)
    server.generator.collective_rpc = AsyncMock()
    server._enable_rl_control_endpoints = enabled
    server._rl_control_api_key = "secret"
    server._register_rl_control_routes()
    return server


def _post_signed(client: TestClient, endpoint: str, payload: dict, key: str = "secret"):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(build_rl_control_auth_headers(key, body))
    return client.post(endpoint, content=body, headers=headers)


def test_rl_control_routes_require_key():
    with pytest.raises(ValueError, match="rl_control_api_key is required"):
        OpenAIServer(
            generator=object.__new__(AsyncLLM),
            model="model",
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
            enable_rl_control_endpoints=True,
        )


def test_rl_control_routes_require_async_llm():
    with pytest.raises(ValueError, match="require AsyncLLM"):
        OpenAIServer(
            generator=MagicMock(),
            model="model",
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
            enable_rl_control_endpoints=True,
            rl_control_api_key="secret",
        )


def test_rl_control_routes_disabled_by_default():
    server = _make_server(enabled=False)
    paths = {route.path for route in server.app.routes}

    assert "/release_memory" not in paths
    assert "/resume_memory" not in paths
    assert "/update_weights" not in paths


@pytest.mark.parametrize(
    "endpoint",
    [
        "/release_memory",
        "/resume_memory",
        "/update_weights",
    ],
)
def test_rl_control_routes_require_auth(endpoint):
    server = _make_server(enabled=True)

    with TestClient(server.app) as client:
        response = client.post(endpoint, json={"tags": ["kv_cache"]})

    assert response.status_code == 401
    server.generator.collective_rpc.assert_not_awaited()


@pytest.mark.parametrize(
    "endpoint",
    [
        "/release_memory",
        "/resume_memory",
        "/update_weights",
    ],
)
def test_rl_control_routes_reject_wrong_key(endpoint):
    server = _make_server(enabled=True)

    with TestClient(server.app) as client:
        response = _post_signed(client, endpoint, {"tags": ["kv_cache"]}, key="wrong")

    assert response.status_code == 401
    server.generator.collective_rpc.assert_not_awaited()


def test_rl_control_routes_reject_non_ascii_signature():
    server = _make_server(enabled=True)

    with TestClient(server.app) as client:
        response = client.post(
            "/release_memory",
            json={"tags": ["kv_cache"]},
            headers=[
                (b"content-type", b"application/json"),
                (RL_CONTROL_AUTH_HEADER.encode("ascii"), b"\xff"),
            ],
        )

    assert response.status_code == 401
    server.generator.collective_rpc.assert_not_awaited()


def test_rl_control_routes_reject_signature_for_different_body():
    server = _make_server(enabled=True)
    signed_body = json.dumps({"tags": ["kv_cache"]}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(build_rl_control_auth_headers("secret", signed_body))

    with TestClient(server.app) as client:
        response = client.post(
            "/release_memory",
            content=json.dumps({"tags": ["model"]}),
            headers=headers,
        )

    assert response.status_code == 401
    server.generator.collective_rpc.assert_not_awaited()


@pytest.mark.parametrize(
    ("endpoint", "payload", "rpc_name", "rpc_args"),
    [
        ("/release_memory", {"tags": ["kv_cache"]}, "sleep", (["kv_cache"],)),
        ("/resume_memory", {"tags": ["model"]}, "wakeup", (["model"],)),
        ("/update_weights", {"weights": None}, "update_weights", (None,)),
    ],
)
def test_rl_control_routes_accept_valid_signature(endpoint, payload, rpc_name, rpc_args):
    server = _make_server(enabled=True)

    with TestClient(server.app) as client:
        response = _post_signed(client, endpoint, payload)

    assert response.status_code == 200
    server.generator.collective_rpc.assert_awaited_once_with(rpc_name, args=rpc_args)


def test_release_memory_requires_tags():
    server = _make_server(enabled=True)

    with TestClient(server.app) as client:
        response = _post_signed(client, "/release_memory", {})

    assert response.status_code == 422
    server.generator.collective_rpc.assert_not_awaited()
