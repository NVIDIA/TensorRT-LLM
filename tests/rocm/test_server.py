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

import pytest
from fastapi.testclient import TestClient

from tensorrt_llm.rocm.server import create_app

pytestmark = pytest.mark.cpu_only


def test_health_models_text_and_chat(engine, monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_API_KEY", raising=False)
    with TestClient(create_app(engine, "tiny")) as client:
        assert client.get("/health").json()["backend"] == "rocm"
        assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"
        response = client.post(
            "/v1/completions",
            json={
                "model": "tiny",
                "prompt": ["tok4 tok5", "tok6"],
                "max_tokens": 3,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["object"] == "text_completion" and len(data["choices"]) == 2
        assert data["usage"] == {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}
        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "tok4 tok5"}],
                "max_tokens": 3,
                "temperature": 0,
            },
        )
        assert chat.status_code == 200, chat.text
        assert chat.json()["choices"][0]["message"]["role"] == "assistant"


def test_server_rejects_streaming_wrong_models_and_unknown_options(engine, monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_API_KEY", raising=False)
    with TestClient(create_app(engine, "tiny")) as client:
        for values, status in (
            ({"model": "other"}, 404),
            ({"stream": True}, 400),
            ({"logprobs": 3}, 422),
        ):
            response = client.post(
                "/v1/completions", json={"model": "tiny", "prompt": "tok4", **values}
            )
            assert response.status_code == status, response.text
        invalid = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny",
                "messages": [{"role": "tool", "content": "tok4"}],
            },
        )
        assert invalid.status_code == 422


def test_optional_bearer_auth(engine, monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_API_KEY", "local-test-key")
    with TestClient(create_app(engine, "tiny")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        )
        assert (
            client.get("/v1/models", headers={"Authorization": "Bearer local-test-key"}).status_code
            == 200
        )


def test_chat_template_is_required(engine, monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_API_KEY", raising=False)
    engine.tokenizer.chat_template = None
    with TestClient(create_app(engine, "tiny")) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "tok4"}],
                "max_tokens": 1,
            },
        )
        assert response.status_code == 400
        assert "chat template" in response.json()["detail"]
