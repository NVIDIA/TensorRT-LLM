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
"""The render endpoints: standalone app, embedded mounting, and ``POST /generate``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.serve.render import (
    GENERATE_REQUEST_SCHEMA_VERSION,
    GenerateRequest,
    ServingRender,
    attach_generate_route,
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
        # No executor here: no inference routes and no /generate.
        assert "/v1/chat/completions" not in paths
        assert "/v1/completions" not in paths
        assert "/generate" not in paths
        assert client.post("/v1/chat/completions", json=CHAT_BODY).status_code == 404
        assert client.post("/generate", json={}).status_code == 404

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
        assert prepared.fingerprint == resources(tokenizer).fingerprint()
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
        assert with_template["fingerprint"]["digest"] != without["fingerprint"]["digest"]


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
    def test_a_worker_mounts_render_and_generate_only_when_enabled(
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
        assert ("/generate" in paths) is enabled
        assert ("/v1/chat/completions/render" in paths) is enabled
        assert ("/v1/completions/render" in paths) is enabled

    def test_the_decision_can_be_forced(self, tokenizer) -> None:
        server = SimpleNamespace(tokenizer=tokenizer, model_config=None, generator=None)
        app = FastAPI()

        assert mount_render_endpoints(app, server, enabled=True) is True
        assert "/generate" in {route.path for route in app.routes}


class TestGenerate:
    """``POST /generate`` runs the original request with the ids already filled in."""

    @pytest.fixture
    def worker(self, tokenizer):
        server = SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(model_type="render-test-model"),
            processor=None,
            chat_template=None,
            allow_request_chat_template=False,
            generator=SimpleNamespace(args=SimpleNamespace(reasoning_parser=None)),
        )
        seen = {}

        async def openai_chat(request, raw_request):
            seen["chat"] = request
            return {"ok": "chat"}

        async def openai_completion(request, raw_request):
            seen["completion"] = request
            return {"ok": "completion"}

        server.openai_chat = openai_chat
        server.openai_completion = openai_completion
        app = FastAPI()
        attach_generate_route(app, server)
        return SimpleNamespace(client=TestClient(app), server=server, seen=seen)

    def _prepared(self, tokenizer, **overrides):
        prepared = (
            TestClient(build_render_app(resources(tokenizer)))
            .post("/v1/chat/completions/render", json=CHAT_BODY)
            .json()
        )
        prepared.update(overrides)
        return prepared

    def test_runs_the_chat_route_with_the_prepared_token_ids(self, worker, tokenizer) -> None:
        prepared = self._prepared(tokenizer)

        response = worker.client.post("/generate", json=prepared)

        assert response.status_code == 200, response.text
        assert response.json() == {"ok": "chat"}
        request = worker.seen["chat"]
        assert request.prompt_token_ids == prepared["token_ids"]
        assert [m["content"] for m in request.messages] == [
            m["content"] for m in CHAT_BODY["messages"]
        ]

    def test_a_harmony_worker_runs_its_harmony_chat_route(self, worker) -> None:
        # A gpt-oss worker serves chat on chat_harmony, which parses channels and tool
        # calls out of the token stream; the plain chat route would return them raw.
        from tensorrt_llm.serve.render import RenderResources

        async def chat_harmony(request, raw_request):
            worker.seen["harmony"] = request
            return {"ok": "harmony"}

        worker.server.use_harmony = True
        worker.server.chat_harmony = chat_harmony
        prepared = {
            "schema_version": GENERATE_REQUEST_SCHEMA_VERSION,
            "kind": "chat",
            "fingerprint": RenderResources.from_server(worker.server).fingerprint(),
            "token_ids": [7, 8, 9],
            "tokens_trusted": True,
            "request": CHAT_BODY,
        }

        response = worker.client.post("/generate", json=prepared)

        assert response.status_code == 200, response.text
        assert response.json() == {"ok": "harmony"}
        assert worker.seen["harmony"].prompt_token_ids == [7, 8, 9]
        assert "chat" not in worker.seen

    def test_runs_the_completions_route_for_a_completion_request(self, worker, tokenizer) -> None:
        (prepared,) = (
            TestClient(build_render_app(resources(tokenizer)))
            .post("/v1/completions/render", json={"model": "m", "prompt": "hello world"})
            .json()
        )

        response = worker.client.post("/generate", json=prepared)

        assert response.status_code == 200, response.text
        assert response.json() == {"ok": "completion"}
        assert worker.seen["completion"].prompt == prepared["token_ids"]

    def test_an_unknown_schema_version_is_rejected(self, worker, tokenizer) -> None:
        response = worker.client.post(
            "/generate", json=self._prepared(tokenizer, schema_version=99)
        )

        assert response.status_code == 400
        assert "schema_version" in response.json()["message"]
        assert "chat" not in worker.seen

    def test_a_different_rendering_configuration_is_rejected(self, worker, tokenizer) -> None:
        other = TestClient(
            build_render_app(resources(tokenizer, default_chat_template=SERVER_TEMPLATE))
        )
        prepared = other.post("/v1/chat/completions/render", json=CHAT_BODY).json()

        response = worker.client.post("/generate", json=prepared)

        assert response.status_code == 409
        assert "differs" in response.json()["message"]
        assert "chat" not in worker.seen

    def test_untrusted_ids_are_rejected(self, worker, tokenizer) -> None:
        response = worker.client.post(
            "/generate",
            json=self._prepared(
                tokenizer, tokens_trusted=False, untrusted_reason="named_tool_choice"
            ),
        )

        assert response.status_code == 400
        assert "named_tool_choice" in response.json()["message"]
        assert "chat" not in worker.seen

    def test_a_malformed_body_is_a_400(self, worker) -> None:
        assert worker.client.post("/generate", json={"token_ids": "nope"}).status_code == 400
        assert worker.client.post("/generate", json=[1]).status_code == 400

    def test_the_worker_fingerprint_is_computed_once(self, worker, tokenizer) -> None:
        prepared = self._prepared(tokenizer)
        worker.client.post("/generate", json=prepared)
        first = worker.server._render_fingerprint

        worker.client.post("/generate", json=prepared)

        assert worker.server._render_fingerprint is first


class TestGenerateSkipsRendering:
    """Through the real chat handler: a prepared request is not rendered again."""

    def test_the_chat_handler_receives_the_ids_and_never_renders(
        self, tokenizer, monkeypatch
    ) -> None:
        from unittest.mock import AsyncMock

        from tensorrt_llm.serve.openai_protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
            UsageInfo,
        )
        from tensorrt_llm.serve.openai_server import OpenAIServer
        from tensorrt_llm.serve.render import chat as render_chat_module

        rendered_calls = []

        async def render_must_not_run(**kwargs):
            rendered_calls.append(kwargs)
            return "should not happen"

        monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render_must_not_run)
        inputs = {}

        def generate_async(*, inputs: dict | list, **kwargs):
            inputs_holder["inputs"] = inputs
            return SimpleNamespace(prompt_token_ids=[1, 2, 3], finished=True)

        inputs_holder = inputs
        server = object.__new__(OpenAIServer)
        server.model = "m"
        server.allow_request_chat_template = False
        server.model_config = SimpleNamespace(model_type="render-test-model", vocab_size=1000)
        server.processor = None
        server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=1000))
        server.chat_template = None
        server.tool_parser = None
        server.tool_call_id_type = "random"
        server.multimodal_server_config = None
        server.generator = SimpleNamespace(
            args=SimpleNamespace(
                gather_generation_logits=False,
                reasoning_parser=None,
                backend="pytorch",
                guided_decoding_backend=None,
                num_postprocess_workers=0,
            ),
            generate_async=generate_async,
        )
        server.await_disconnected = AsyncMock()
        server._create_chat_response = AsyncMock(
            return_value=ChatCompletionResponse(
                id="x",
                model="m",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
            )
        )
        # The worker and the renderer share a tokenizer and template here, so
        # their fingerprints agree.
        server._render_fingerprint = resources(tokenizer).fingerprint()
        app = FastAPI()
        attach_generate_route(app, server)
        prepared = (
            TestClient(build_render_app(resources(tokenizer)))
            .post("/v1/chat/completions/render", json={**CHAT_BODY, "max_tokens": 4})
            .json()
        )

        response = TestClient(app).post("/generate", json=prepared)

        assert response.status_code == 200, response.text
        assert inputs["inputs"]["prompt_token_ids"] == prepared["token_ids"]
        assert rendered_calls == []


def _real_chat_worker(tokenizer, *, model_type="render-test-model", reasoning_parser=None):
    """A worker whose real chat route runs, with the engine and response building stubbed.

    Returns ``(app_with_generate, captured)``; ``captured["kwargs"]`` holds what the route
    handed the engine, including the postprocessing arguments it derived.
    """
    from unittest.mock import AsyncMock

    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
        UsageInfo,
    )
    from tensorrt_llm.serve.openai_server import OpenAIServer

    captured = {}

    def generate_async(*, inputs, **kwargs):
        captured["inputs"] = inputs
        captured["kwargs"] = kwargs
        return SimpleNamespace(prompt_token_ids=[1, 2, 3], finished=True)

    server = object.__new__(OpenAIServer)
    server.model = "m"
    server.allow_request_chat_template = False
    server.model_config = SimpleNamespace(model_type=model_type, vocab_size=1000)
    server.processor = None
    server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=1000))
    server.chat_template = None
    server.tool_parser = None
    server.tool_call_id_type = "random"
    server.multimodal_server_config = None
    server.generator = SimpleNamespace(
        args=SimpleNamespace(
            gather_generation_logits=False,
            reasoning_parser=reasoning_parser,
            backend="pytorch",
            guided_decoding_backend=None,
            num_postprocess_workers=0,
        ),
        generate_async=generate_async,
    )
    server.await_disconnected = AsyncMock()
    server._create_chat_response = AsyncMock(
        return_value=ChatCompletionResponse(
            id="x",
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )
    )
    app = FastAPI()
    attach_generate_route(app, server)
    return app, server, captured


class TestGenerateThroughTheRealChatRoute:
    """What the worker's chat route derives must survive the render -> /generate hop."""

    def test_messages_the_request_model_rejects_still_reach_the_route(self, tokenizer) -> None:
        # Object-valued tool-call arguments are rejected by the request model; the chat
        # route then reads the messages from the raw JSON body. For /generate that body
        # is the prepared-request envelope, which has no top-level "messages".
        app, server, captured = _real_chat_worker(tokenizer)
        server._render_fingerprint = resources(tokenizer).fingerprint()
        body = {
            "model": "m",
            "max_tokens": 4,
            "messages": [
                {"role": "user", "content": "hello world"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "paris"}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                {"role": "assistant", "content": "it is sunny"},
            ],
        }
        prepared = (
            TestClient(build_render_app(resources(tokenizer)))
            .post("/v1/chat/completions/render", json=body)
            .json()
        )

        response = TestClient(app).post("/generate", json=prepared)

        assert response.status_code == 200, response.text
        postproc_args = server._create_chat_response.call_args.args[1].postproc_args
        # Taken from the parsed conversation: empty if the route fell back to nothing.
        assert postproc_args.last_message_content == "it is sunny"

    @pytest.fixture
    def usage_extension(self):
        from tensorrt_llm.serve import serving_extensions
        from tensorrt_llm.serve.serving_extensions import register_serving_extension

        key = "render-test-usage-model"

        @register_serving_extension(model_types=(key,))
        class UsageExtension(ServingExtension):
            def prompt_tokens_excluded_from_usage(self, request):
                # Like kimi_k3: the trailing generation opener is not reported as usage,
                # and only a request that is rendered here has one.
                return (
                    3 if request.add_generation_prompt and request.prompt_token_ids is None else 0
                )

        try:
            yield key
        finally:
            serving_extensions._BY_MODEL_TYPE.pop(key, None)

    def test_the_usage_adjustment_decided_at_render_time_reaches_the_worker(
        self, tokenizer, usage_extension
    ) -> None:
        from tensorrt_llm.serve.serving_extensions import get_serving_extension

        render_resources = resources(
            tokenizer, model_type=usage_extension, extension=get_serving_extension(usage_extension)
        )
        app, server, captured = _real_chat_worker(tokenizer, model_type=usage_extension)
        server._render_fingerprint = render_resources.fingerprint()
        prepared = (
            TestClient(build_render_app(render_resources))
            .post("/v1/chat/completions/render", json={**CHAT_BODY, "max_tokens": 4})
            .json()
        )

        response = TestClient(app).post("/generate", json=prepared)

        assert prepared["context"]["prompt_tokens_excluded_from_usage"] == 3
        assert response.status_code == 200, response.text
        # Without the context the route sees ``prompt_token_ids`` and adjusts nothing.
        postproc_args = server._create_chat_response.call_args.args[1].postproc_args
        assert postproc_args.num_prompt_tokens_offset == 3

    def test_the_thinking_mode_read_off_the_rendered_prompt_reaches_the_worker(self) -> None:
        # The template prefills "<think>"; a parser that takes its mode from the prompt must
        # see it as open even though the worker never renders the prompt.
        thinking_tokenizer = make_tokenizer(
            chat_template=(
                "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
                "{% if add_generation_prompt %}<assistant><think>{% endif %}"
            )
        )
        render_resources = resources(thinking_tokenizer, reasoning_parser="poolside_v1")
        app, server, captured = _real_chat_worker(
            thinking_tokenizer, reasoning_parser="poolside_v1"
        )
        server._render_fingerprint = render_resources.fingerprint()
        prepared = (
            TestClient(build_render_app(render_resources))
            .post("/v1/chat/completions/render", json={**CHAT_BODY, "max_tokens": 4})
            .json()
        )

        response = TestClient(app).post("/generate", json=prepared)

        assert prepared["context"]["resolved_thinking"] is True
        assert response.status_code == 200, response.text
        postproc_args = server._create_chat_response.call_args.args[1].postproc_args
        assert postproc_args.chat_template_kwargs["thinking"] is True

    def test_a_client_cannot_set_the_prepared_context_on_the_normal_route(self, tokenizer) -> None:
        # The context is a private attribute set only by /generate; a field of the same
        # name in an ordinary chat request is just an unknown field.
        from pydantic import ValidationError

        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        with pytest.raises(ValidationError, match="_render_context"):
            ChatCompletionRequest.model_validate(
                {**CHAT_BODY, "_render_context": {"prompt_tokens_excluded_from_usage": 99}}
            )


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
