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
"""The render endpoints: the standalone app and mounting the routes on a worker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.serve.render import (
    GENERATE_REQUEST_SCHEMA_VERSION,
    GenerateRequest,
    ServingRender,
    attach_render_routes,
    mount_render_endpoints,
    render_chat,
    render_endpoints_enabled,
)
from tensorrt_llm.serve.render._http import build_render_app
from tensorrt_llm.serve.serving_extensions import ServingExtension

from .render_helpers import SERVER_TEMPLATE, WEATHER_TOOL, chat_request, make_tokenizer, resources

pytestmark = pytest.mark.cpu_only

CHAT_BODY = {
    "model": "m",
    "messages": [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "this is a test"},
        {"role": "user", "content": "get the weather"},
    ],
}


@pytest.fixture(scope="module")
def tokenizer():
    return make_tokenizer()


@pytest.fixture
def client(tokenizer):
    return TestClient(build_render_app(resources(tokenizer), served_model_name="m"))


class TestStandaloneApp:
    def test_serves_exactly_the_render_routes_health_and_server_info(self, client) -> None:
        paths = {route.path for route in client.app.routes}

        assert "/v1/chat/completions/render" in paths
        assert "/v1/completions/render" in paths
        assert "/health" in paths
        assert "/server_info" in paths
        # No executor here: no inference routes.
        assert "/v1/chat/completions" not in paths
        assert "/v1/completions" not in paths
        assert client.post("/v1/chat/completions", json=CHAT_BODY).status_code == 404

    def test_health_and_server_info(self, client, tokenizer) -> None:
        assert client.get("/health").json() == {"status": "ok"}

        info = client.get("/server_info").json()

        assert info["role"] == "renderer"
        assert info["model"] == "m"
        assert info["schema_version"] == GENERATE_REQUEST_SCHEMA_VERSION
        assert info["render_fingerprint"] == resources(tokenizer).fingerprint()

    def test_chat_render_returns_the_prepared_request(self, client, tokenizer) -> None:
        response = client.post("/v1/chat/completions/render", json=CHAT_BODY)

        assert response.status_code == 200, response.text
        prepared = GenerateRequest.model_validate(response.json())
        expected = render_chat(chat_request(), resources(tokenizer)).token_ids
        assert prepared.token_ids == expected
        assert prepared.kind == "chat"
        assert prepared.tokens_trusted is True
        assert "fingerprint" not in response.json()
        assert prepared.request == CHAT_BODY
        # Top-level token_ids, so a client that reads only that field works.
        assert response.json()["token_ids"] == expected

    def test_chat_render_matches_the_python_api_for_tools(self, client, tokenizer) -> None:
        body = {**CHAT_BODY, "tools": [WEATHER_TOOL]}

        response = client.post("/v1/chat/completions/render", json=body)

        assert response.status_code == 200, response.text
        expected = render_chat(chat_request(tools=[WEATHER_TOOL]), resources(tokenizer)).token_ids
        assert response.json()["token_ids"] == expected

    def test_dynamo_gateway_fields_are_ignored(self, client) -> None:
        plain = client.post("/v1/chat/completions/render", json=CHAT_BODY).json()
        with_gateway_fields = client.post(
            "/v1/chat/completions/render",
            json={
                **CHAT_BODY,
                "nvext": {"agent_hints": {"osl": 128}, "cache_namespace": "tenant-a"},
                "cache_namespace": "tenant-a",
            },
        )

        assert with_gateway_fields.status_code == 200, with_gateway_fields.text
        assert with_gateway_fields.json()["token_ids"] == plain["token_ids"]
        # The prepared request carries the body without the gateway-only fields.
        assert "nvext" not in with_gateway_fields.json()["request"]

    def test_extra_ignored_fields_can_be_configured(self, client, monkeypatch) -> None:
        monkeypatch.setenv("TRTLLM_RENDER_IGNORED_FIELDS", "x_gateway_hint, x_other")

        response = client.post(
            "/v1/chat/completions/render",
            json={**CHAT_BODY, "x_gateway_hint": 1, "x_other": 2},
        )

        assert response.status_code == 200, response.text

    def test_an_unknown_field_is_rejected(self, client) -> None:
        response = client.post("/v1/chat/completions/render", json={**CHAT_BODY, "surprise": 1})
        assert response.status_code == 400
        assert response.json()["type"] == "BadRequestError"

    def test_malformed_json_is_a_400(self, client) -> None:
        response = client.post(
            "/v1/chat/completions/render",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400

    def test_a_non_object_body_is_a_400(self, client) -> None:
        assert client.post("/v1/chat/completions/render", json=[1, 2]).status_code == 400

    def test_multimodal_input_is_rejected_with_a_400(self, client) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "http://example.invalid/x.png"}},
                    ],
                }
            ],
        }

        response = client.post("/v1/chat/completions/render", json=body)

        assert response.status_code == 400
        assert "Multimodal" in response.json()["message"]

    def test_a_named_tool_choice_comes_back_untrusted(self, client) -> None:
        body = {
            **CHAT_BODY,
            "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }

        prepared = client.post("/v1/chat/completions/render", json=body).json()

        assert prepared["tokens_trusted"] is False
        assert prepared["untrusted_reason"] == "named_tool_choice"
        assert prepared["token_ids"]

    def test_the_server_template_shapes_the_ids(self, tokenizer) -> None:
        templated = TestClient(
            build_render_app(resources(tokenizer, default_chat_template=SERVER_TEMPLATE))
        )
        plain = TestClient(build_render_app(resources(tokenizer)))

        with_template = templated.post("/v1/chat/completions/render", json=CHAT_BODY).json()
        without = plain.post("/v1/chat/completions/render", json=CHAT_BODY).json()

        assert with_template["token_ids"] != without["token_ids"]
        # The difference is visible in the identity published on /server_info.
        assert (
            templated.get("/server_info").json()["render_fingerprint"]["digest"]
            != plain.get("/server_info").json()["render_fingerprint"]["digest"]
        )


class TestCompletionsRender:
    def _render(self, client, prompt, **extra):
        return client.post("/v1/completions/render", json={"model": "m", "prompt": prompt, **extra})

    def test_a_string_prompt_tokenizes_with_special_tokens_by_default(
        self, client, tokenizer
    ) -> None:
        response = self._render(client, "hello world")

        assert response.status_code == 200, response.text
        (prepared,) = response.json()
        # Completions add special tokens by default; the chat route does not.
        assert prepared["token_ids"] == tokenizer.encode("hello world", add_special_tokens=True)
        assert prepared["kind"] == "completion"
        assert prepared["request"]["prompt"] == prepared["token_ids"]

    def test_special_tokens_and_truncation_follow_the_request(self, client, tokenizer) -> None:
        (without_bos,) = self._render(client, "hello world", add_special_tokens=False).json()
        (truncated,) = self._render(client, "hello world", truncate_prompt_tokens=2).json()

        assert without_bos["token_ids"] == tokenizer.encode("hello world", add_special_tokens=False)
        assert truncated["token_ids"] == tokenizer.encode("hello world")[:2]

    def test_a_batch_of_strings_returns_one_prepared_request_per_prompt(
        self, client, tokenizer
    ) -> None:
        prepared = self._render(client, ["hello", "world"]).json()

        assert [item["token_ids"] for item in prepared] == [
            tokenizer.encode("hello"),
            tokenizer.encode("world"),
        ]

    def test_token_prompts_pass_through(self, client) -> None:
        (single,) = self._render(client, [5, 6, 7]).json()
        batch = self._render(client, [[1, 2], [3]]).json()

        assert single["token_ids"] == [5, 6, 7]
        assert [item["token_ids"] for item in batch] == [[1, 2], [3]]


class TestEmbeddedMounting:
    def test_the_routes_are_added_to_any_app_and_match_the_standalone_app(self, tokenizer) -> None:
        app = FastAPI()
        attach_render_routes(app, ServingRender(lambda: resources(tokenizer)))
        embedded = TestClient(app).post("/v1/chat/completions/render", json=CHAT_BODY)
        standalone = TestClient(build_render_app(resources(tokenizer))).post(
            "/v1/chat/completions/render", json=CHAT_BODY
        )

        assert embedded.status_code == 200
        assert embedded.json() == standalone.json()

    def test_serving_workers_mount_them_only_when_enabled(self, monkeypatch) -> None:
        monkeypatch.delenv("TRTLLM_ENABLE_RENDER_ENDPOINTS", raising=False)
        assert render_endpoints_enabled() is False
        monkeypatch.setenv("TRTLLM_ENABLE_RENDER_ENDPOINTS", "1")
        assert render_endpoints_enabled() is True

    @pytest.mark.parametrize("enabled", [False, True])
    def test_a_worker_mounts_the_render_routes_only_when_enabled(
        self, tokenizer, monkeypatch, enabled
    ) -> None:
        if enabled:
            monkeypatch.setenv("TRTLLM_ENABLE_RENDER_ENDPOINTS", "1")
        else:
            monkeypatch.delenv("TRTLLM_ENABLE_RENDER_ENDPOINTS", raising=False)
        server = SimpleNamespace(tokenizer=tokenizer, model_config=None, generator=None)
        app = FastAPI()

        mounted = mount_render_endpoints(app, server)

        paths = {route.path for route in app.routes}
        assert mounted is enabled
        assert ("/v1/chat/completions/render" in paths) is enabled
        assert ("/v1/completions/render" in paths) is enabled

    def test_the_decision_can_be_forced(self, tokenizer) -> None:
        server = SimpleNamespace(tokenizer=tokenizer, model_config=None, generator=None)
        app = FastAPI()

        assert mount_render_endpoints(app, server, enabled=True) is True
        paths = {route.path for route in app.routes}
        assert "/v1/chat/completions/render" in paths
        # Executing ids uses the worker's existing token-in routes; only the two render
        # routes are added (FastAPI's own docs routes are not ours).
        added = {path for path in paths if not path.startswith(("/docs", "/redoc", "/openapi"))}
        assert added == {"/v1/chat/completions/render", "/v1/completions/render"}


class TestHarmonyConversionErrors:
    """Harmony reports messages it cannot convert as a RuntimeError; the chat route answers 400."""

    class _FailingHarmony(ServingExtension):
        def render_prompt(self, request, res=None):
            raise RuntimeError("Failed to convert messages to harmony tokens: boom")

    def test_a_conversion_failure_is_a_400_not_a_500(self, tokenizer) -> None:
        app = build_render_app(
            resources(tokenizer, use_harmony=True, extension=self._FailingHarmony())
        )

        response = TestClient(app).post("/v1/chat/completions/render", json=CHAT_BODY)

        assert response.status_code == 400
        assert "boom" in response.text
