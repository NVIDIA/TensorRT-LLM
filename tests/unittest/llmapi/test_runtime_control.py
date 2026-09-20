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

import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm import LLM
from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.llmapi import RuntimeMemoryStatus, SleepConfig
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.runtime_control_auth import (
    RUNTIME_CONTROL_NONCE_HEADER,
    RUNTIME_CONTROL_TIMESTAMP_HEADER,
    RuntimeControlReplayCache,
    build_runtime_control_auth_headers,
    validate_runtime_control_request,
)

pytestmark = pytest.mark.cpu_only


def _make_server(*, asynchronous: bool = False, replay_cache_capacity: int = 4096) -> OpenAIServer:
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    generator_cls = AsyncLLM if asynchronous else LLM
    server.generator = object.__new__(generator_cls)
    server.generator.args = SimpleNamespace(sleep_config=SleepConfig())
    method_factory = AsyncMock if asynchronous else MagicMock
    server.generator.release = method_factory()
    server.generator.resume = method_factory()
    server.generator.get_memory_status = method_factory(
        return_value=RuntimeMemoryStatus(state="running", parked_tags=[])
    )
    server._enable_runtime_control_endpoints = True
    server._runtime_control_api_key = "secret"
    server._runtime_control_replay_cache = RuntimeControlReplayCache(replay_cache_capacity)
    server._register_runtime_control_routes()
    return server


def _signed_headers(
    method: str,
    path: str,
    body: bytes,
    key: str = "secret",
    *,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    headers.update(
        build_runtime_control_auth_headers(
            key, method, path, body, timestamp=timestamp, nonce=nonce
        )
    )
    return headers


def _post(client: TestClient, path: str, payload=None):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    return client.post(path, content=body, headers=_signed_headers("POST", path, body))


def test_runtime_control_routes_are_authenticated():
    server = _make_server()

    with TestClient(server.app) as client:
        response = client.post("/release_memory", content=b"")

    assert response.status_code == 401
    server.generator.release.assert_not_called()


def test_signature_covers_exact_body():
    server = _make_server()
    signed_body = json.dumps({"tags": ["model"]}).encode("utf-8")
    other_body = json.dumps({"tags": ["kv_cache"]}).encode("utf-8")

    with TestClient(server.app) as client:
        response = client.post(
            "/release_memory",
            content=other_body,
            headers=_signed_headers("POST", "/release_memory", signed_body),
        )

    assert response.status_code == 401
    server.generator.release.assert_not_called()


@pytest.mark.parametrize(
    ("signed_method", "signed_path"),
    [
        ("GET", "/release_memory"),
        ("POST", "/resume_memory"),
        ("GET", "/memory_status"),
    ],
)
def test_signature_covers_http_method_and_route(signed_method, signed_path):
    server = _make_server()
    body = b""

    with TestClient(server.app) as client:
        response = client.post(
            "/release_memory",
            content=body,
            headers=_signed_headers(signed_method, signed_path, body),
        )

    assert response.status_code == 401
    server.generator.release.assert_not_called()


@pytest.mark.parametrize("payload", [None, {}])
def test_release_omitted_tags_selects_defaults(payload):
    server = _make_server()

    with TestClient(server.app) as client:
        response = _post(client, "/release_memory", payload)

    assert response.status_code == 200
    server.generator.release.assert_called_once_with(None)


def test_explicit_tags_are_forwarded_to_sync_llm():
    server = _make_server()

    with TestClient(server.app) as client:
        response = _post(client, "/resume_memory", {"tags": ["model", "kv_cache"]})

    assert response.status_code == 200
    server.generator.resume.assert_called_once_with(["model", "kv_cache"])


@pytest.mark.asyncio
async def test_async_generator_is_awaited_directly():
    server = _make_server(asynchronous=True)

    response = await server.runtime_release_memory(None)

    assert response.status_code == 200
    server.generator.release.assert_awaited_once_with(None)


def test_memory_status_signs_empty_body():
    server = _make_server()
    headers = _signed_headers("GET", "/memory_status", b"")

    with TestClient(server.app) as client:
        response = client.get("/memory_status", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"state": "running", "parked_tags": []}


def test_invalid_worker_status_is_an_internal_error():
    server = _make_server()
    server.generator.get_memory_status.side_effect = ValueError("invalid worker status")
    headers = _signed_headers("GET", "/memory_status", b"")

    with TestClient(server.app) as client:
        response = client.get("/memory_status", headers=headers)

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "invalid worker status"


@pytest.mark.parametrize(
    "error, expected_status",
    [
        (ValueError("bad tag"), 400),
        (RuntimeError("Runtime memory is already parked"), 409),
        (RuntimeError("Cannot resume runtime memory while state is 'waking'."), 409),
        (OSError("worker failed"), 500),
    ],
)
def test_release_error_mapping(error, expected_status):
    server = _make_server()
    server.generator.release.side_effect = error

    with TestClient(server.app) as client:
        response = _post(client, "/release_memory", {"tags": ["model"]})

    assert response.status_code == expected_status
    assert response.json()["error"]["message"] == str(error)


def test_explicit_empty_tags_are_rejected():
    server = _make_server()

    with TestClient(server.app) as client:
        response = _post(client, "/release_memory", {"tags": []})

    assert response.status_code == 400
    server.generator.release.assert_not_called()


@pytest.mark.parametrize(
    "missing_header",
    [RUNTIME_CONTROL_TIMESTAMP_HEADER, RUNTIME_CONTROL_NONCE_HEADER],
)
def test_runtime_control_rejects_missing_replay_headers(missing_header):
    server = _make_server()
    body = b""
    headers = _signed_headers("POST", "/release_memory", body)
    headers.pop(missing_header)

    with TestClient(server.app) as client:
        response = client.post("/release_memory", content=body, headers=headers)

    assert response.status_code == 401
    server.generator.release.assert_not_called()


@pytest.mark.parametrize(
    ("header", "invalid_value"),
    [
        (RUNTIME_CONTROL_TIMESTAMP_HEADER, "not-a-timestamp"),
        (RUNTIME_CONTROL_NONCE_HEADER, "0" * 31),
        (RUNTIME_CONTROL_NONCE_HEADER, "A" * 32),
    ],
)
def test_runtime_control_rejects_malformed_replay_headers(header, invalid_value):
    server = _make_server()
    body = b""
    headers = _signed_headers("POST", "/release_memory", body)
    headers[header] = invalid_value

    with TestClient(server.app) as client:
        response = client.post("/release_memory", content=body, headers=headers)

    assert response.status_code == 401
    server.generator.release.assert_not_called()


@pytest.mark.parametrize("timestamp_delta", [-301, 301])
def test_runtime_control_rejects_timestamps_outside_window(timestamp_delta):
    server = _make_server()
    body = b""
    headers = _signed_headers(
        "POST",
        "/release_memory",
        body,
        timestamp=int(time.time()) + timestamp_delta,
    )

    with TestClient(server.app) as client:
        response = client.post("/release_memory", content=body, headers=headers)

    assert response.status_code == 401
    server.generator.release.assert_not_called()


@pytest.mark.parametrize("timestamp_delta", [-300, 300])
def test_runtime_control_accepts_timestamp_window_boundaries(timestamp_delta):
    current_time = 1_000
    headers = _signed_headers(
        "POST",
        "/release_memory",
        b"",
        timestamp=current_time + timestamp_delta,
    )

    validate_runtime_control_request(
        "secret",
        "POST",
        "/release_memory",
        b"",
        headers,
        RuntimeControlReplayCache(),
        current_time=current_time,
    )


def test_runtime_control_rejects_duplicate_request_nonce():
    server = _make_server()
    body = b""
    headers = _signed_headers("POST", "/release_memory", body)

    with TestClient(server.app) as client:
        first_response = client.post("/release_memory", content=body, headers=headers)
        replay_response = client.post("/release_memory", content=body, headers=headers)

    assert first_response.status_code == 200
    assert replay_response.status_code == 401
    server.generator.release.assert_called_once_with(None)


def test_runtime_control_rejects_nonce_reuse_for_changed_request():
    server = _make_server()
    timestamp = int(time.time())
    nonce = "a" * 32
    body = b""
    release_headers = _signed_headers(
        "POST",
        "/release_memory",
        body,
        timestamp=timestamp,
        nonce=nonce,
    )
    resume_headers = _signed_headers(
        "POST",
        "/resume_memory",
        body,
        timestamp=timestamp,
        nonce=nonce,
    )

    with TestClient(server.app) as client:
        release_response = client.post("/release_memory", content=body, headers=release_headers)
        replay_response = client.post("/resume_memory", content=body, headers=resume_headers)

    assert release_response.status_code == 200
    assert replay_response.status_code == 401
    server.generator.release.assert_called_once_with(None)
    server.generator.resume.assert_not_called()


def test_runtime_control_replay_cache_fails_closed_at_capacity():
    server = _make_server(replay_cache_capacity=1)
    body = b""

    with TestClient(server.app) as client:
        first_response = client.post(
            "/release_memory",
            content=body,
            headers=_signed_headers("POST", "/release_memory", body, nonce="a" * 32),
        )
        capacity_response = client.post(
            "/resume_memory",
            content=body,
            headers=_signed_headers("POST", "/resume_memory", body, nonce="b" * 32),
        )

    assert first_response.status_code == 200
    assert capacity_response.status_code == 401
    server.generator.resume.assert_not_called()


def test_runtime_control_replay_cache_removes_only_expired_entries():
    replay_cache = RuntimeControlReplayCache(max_entries=1)
    first_headers = _signed_headers("POST", "/release_memory", b"", timestamp=1_000, nonce="a" * 32)
    validate_runtime_control_request(
        "secret",
        "POST",
        "/release_memory",
        b"",
        first_headers,
        replay_cache,
        current_time=1_000,
    )

    replacement_headers = _signed_headers(
        "POST", "/release_memory", b"", timestamp=1_301, nonce="b" * 32
    )
    validate_runtime_control_request(
        "secret",
        "POST",
        "/release_memory",
        b"",
        replacement_headers,
        replay_cache,
        current_time=1_301,
    )


def test_invalid_signature_does_not_consume_nonce_capacity():
    replay_cache = RuntimeControlReplayCache(max_entries=1)
    invalid_headers = _signed_headers(
        "POST",
        "/release_memory",
        b"",
        key="wrong-secret",
        timestamp=1_000,
        nonce="a" * 32,
    )
    with pytest.raises(ValueError, match="authentication"):
        validate_runtime_control_request(
            "secret",
            "POST",
            "/release_memory",
            b"",
            invalid_headers,
            replay_cache,
            current_time=1_000,
        )

    valid_headers = _signed_headers("POST", "/release_memory", b"", timestamp=1_000, nonce="b" * 32)
    validate_runtime_control_request(
        "secret",
        "POST",
        "/release_memory",
        b"",
        valid_headers,
        replay_cache,
        current_time=1_000,
    )


def test_runtime_control_rejects_concurrent_nonce_reuse():
    replay_cache = RuntimeControlReplayCache()
    timestamp = int(time.time())
    headers = _signed_headers(
        "POST",
        "/release_memory",
        b"",
        timestamp=timestamp,
        nonce="a" * 32,
    )

    def validate_once():
        try:
            validate_runtime_control_request(
                "secret",
                "POST",
                "/release_memory",
                b"",
                headers,
                replay_cache,
                current_time=timestamp,
            )
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        accepted = list(executor.map(lambda _: validate_once(), range(2)))

    assert sorted(accepted) == [False, True]


def test_legacy_and_generic_controls_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        OpenAIServer(
            generator=MagicMock(),
            model="model",
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
            enable_rl_control_endpoints=True,
            rl_control_api_key="legacy-secret",
            enable_runtime_control_endpoints=True,
            runtime_control_api_key="secret",
        )
