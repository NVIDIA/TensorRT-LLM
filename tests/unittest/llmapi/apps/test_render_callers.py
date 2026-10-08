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
"""Every caller of the prompt-preparation core: governor, router, count_tokens, Responses."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm.serve.render import RenderResources, render_chat

from .render_helpers import SERVER_TEMPLATE, WEATHER_TOOL, chat_request, make_tokenizer, resources

pytestmark = pytest.mark.cpu_only

MODEL_TYPE = "render-test-model"


@pytest.fixture(scope="module")
def tokenizer():
    return make_tokenizer()


@pytest.fixture(autouse=True)
def _no_legacy(monkeypatch):
    monkeypatch.delenv("TRTLLM_RENDER_LEGACY", raising=False)


class TestResourceGovernor:
    def _governor(self, tokenizer, chat_template=None):
        from tensorrt_llm.serve.resource_governor import ResourceGovernor

        governor = object.__new__(ResourceGovernor)
        governor.tokenizer = tokenizer
        governor.model_config = SimpleNamespace(model_type=MODEL_TYPE)
        governor.processor = None
        governor.chat_template = chat_template
        return governor

    async def _convert(self, governor, **overrides):
        kwargs = {
            "messages": [{"role": "user", "content": "hello world"}],
            "tool_dicts": None,
            "add_generation_prompt": True,
            "documents": None,
            "chat_template": None,
            "chat_template_kwargs": None,
        }
        kwargs.update(overrides)
        return await governor._convert_messages(**kwargs)

    @pytest.mark.asyncio
    async def test_tokenizes_the_prompt_the_chat_route_executes(self, tokenizer) -> None:
        governor = self._governor(tokenizer, chat_template=SERVER_TEMPLATE)
        request = chat_request(messages=[{"role": "user", "content": "hello world"}])

        ids = await self._convert(governor)

        # The server template applies, and ids match the chat route's tokenization.
        expected = render_chat(request, resources(tokenizer, default_chat_template=SERVER_TEMPLATE))
        assert expected.text.startswith("SERVER:")
        assert ids == expected.token_ids

    @pytest.mark.asyncio
    async def test_the_legacy_switch_restores_the_request_only_template(
        self, tokenizer, monkeypatch
    ) -> None:
        governor = self._governor(tokenizer, chat_template=SERVER_TEMPLATE)
        modern = await self._convert(governor)
        monkeypatch.setenv("TRTLLM_RENDER_LEGACY", "1")

        legacy = await self._convert(governor)

        assert legacy != modern
        request = chat_request(messages=[{"role": "user", "content": "hello world"}])
        assert legacy == render_chat(request, resources(tokenizer)).token_ids


class _Router:
    """A KV-aware router whose tokenizer is the test tokenizer."""

    @staticmethod
    def build(tokenizer, servers):
        from tensorrt_llm.serve.router import KvCacheAwareRouter

        router = KvCacheAwareRouter(
            server_role=None,
            servers=servers,
            use_tokens=False,
            max_batch_size=32,
            tokens_per_block=32,
            use_harmony=False,
        )
        patcher = mock.patch.object(router, "_get_tokenizer", return_value=tokenizer)
        patcher.start()
        router._test_patcher = patcher
        return router

    @staticmethod
    def worker_fingerprint(tokenizer):
        """A worker's fingerprint, built from a worker-shaped server and not from the router.

        Copying the router's own fingerprint would make the write-back tests agree by
        construction and hide any difference between how the two sides describe the
        same rendering configuration.
        """
        from tensorrt_llm.inputs.registry import DefaultInputProcessor

        server = SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(model_type=None),
            processor=None,
            chat_template=None,
            allow_request_chat_template=False,
            generator=SimpleNamespace(
                args=SimpleNamespace(reasoning_parser=None),
                input_processor=DefaultInputProcessor(None, None, tokenizer),
            ),
        )
        return RenderResources.from_server(server).fingerprint()


@pytest.fixture
def router(tokenizer):
    built = _Router.build(tokenizer, ["server1", "server2"])
    yield built
    built._test_patcher.stop()


class TestRouterWriteBack:
    def _expected_ids(self, tokenizer, request):
        return render_chat(
            request, resources(tokenizer, allow_request_chat_template=True)
        ).token_ids

    def _trust(self, router, tokenizer, servers=("server1", "server2")):
        fingerprint = _Router.worker_fingerprint(tokenizer)
        router._server_info = {name: {"render_fingerprint": fingerprint} for name in servers}

    def test_ids_are_forwarded_when_every_worker_reports_a_matching_fingerprint(
        self, router, tokenizer
    ) -> None:
        self._trust(router, tokenizer)
        request = chat_request()

        ids = router._tokenize(request)[0]

        assert ids == self._expected_ids(tokenizer, request)
        assert request.prompt_token_ids == ids

    def test_ids_are_not_forwarded_when_a_worker_reports_no_fingerprint(
        self, router, tokenizer
    ) -> None:
        fingerprint = _Router.worker_fingerprint(tokenizer)
        router._server_info = {
            "server1": {"render_fingerprint": fingerprint},
            "server2": {},  # an older worker
        }
        request = chat_request()

        ids = router._tokenize(request)[0]

        assert request.prompt_token_ids is None
        # Routing still uses the router's own ids.
        assert ids == self._expected_ids(tokenizer, request)
        assert router._render_fallbacks == 1

    def test_ids_are_not_forwarded_when_a_worker_renders_differently(
        self, router, tokenizer
    ) -> None:
        fingerprint = _Router.worker_fingerprint(tokenizer)
        other = {**fingerprint, "digest": "a-different-digest"}
        router._server_info = {
            "server1": {"render_fingerprint": fingerprint},
            "server2": {"render_fingerprint": other},
        }
        request = chat_request()

        router._tokenize(request)

        assert request.prompt_token_ids is None

    def test_ids_are_not_forwarded_when_no_worker_has_reported_yet(self, router) -> None:
        router._server_info = {}
        request = chat_request()

        router._tokenize(request)

        assert request.prompt_token_ids is None

    def test_a_named_tool_choice_is_never_forwarded(self, router, tokenizer) -> None:
        self._trust(router, tokenizer)
        request = chat_request(
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        ids = router._tokenize(request)[0]

        assert ids  # still used for routing
        assert request.prompt_token_ids is None

    def test_media_is_routed_on_an_estimate_and_never_forwarded(self, router, tokenizer) -> None:
        self._trust(router, tokenizer)
        request = chat_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "http://example.invalid/x.png"}},
                    ],
                }
            ]
        )

        ids = router._tokenize(request)[0]

        assert ids
        assert request.prompt_token_ids is None

    def test_a_request_that_is_already_pre_tokenized_is_left_alone(self, router, tokenizer) -> None:
        self._trust(router, tokenizer)
        request = chat_request(prompt_token_ids=[4, 5, 6])

        assert router._tokenize(request) == [[4, 5, 6]]
        assert request.prompt_token_ids == [4, 5, 6]

    def test_the_legacy_switch_forwards_ids_without_asking_the_workers(
        self, router, monkeypatch
    ) -> None:
        monkeypatch.setenv("TRTLLM_RENDER_LEGACY", "1")
        router._server_info = {}
        request = chat_request()

        ids = router._tokenize(request)[0]

        assert request.prompt_token_ids == ids


class TestCountTokens:
    def _client(self, tokenizer, chat_template=None):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tensorrt_llm.serve.openai_server import OpenAIServer

        app = FastAPI()
        server = object.__new__(OpenAIServer)
        server.model = "m"
        server.tokenizer = tokenizer
        server.chat_template = chat_template
        app.add_api_route(
            "/v1/messages/count_tokens", server.anthropic_count_tokens, methods=["POST"]
        )
        return TestClient(app)

    BODY = {"model": "m", "messages": [{"role": "user", "content": "hello world"}]}

    def test_counts_the_tokens_the_chat_route_would_execute(self, tokenizer) -> None:
        response = self._client(tokenizer).post("/v1/messages/count_tokens", json=self.BODY)

        assert response.status_code == 200, response.text
        executed = render_chat(
            chat_request(messages=[{"role": "user", "content": "hello world"}]),
            resources(tokenizer),
        ).token_ids
        assert response.json() == {"input_tokens": len(executed)}

    def test_the_server_template_is_part_of_the_count(self, tokenizer) -> None:
        plain = self._client(tokenizer).post("/v1/messages/count_tokens", json=self.BODY).json()
        templated = (
            self._client(tokenizer, chat_template=SERVER_TEMPLATE)
            .post("/v1/messages/count_tokens", json=self.BODY)
            .json()
        )
        assert templated != plain

    def test_the_legacy_switch_restores_the_old_count(self, tokenizer, monkeypatch) -> None:
        modern = self._client(tokenizer).post("/v1/messages/count_tokens", json=self.BODY).json()
        monkeypatch.setenv("TRTLLM_RENDER_LEGACY", "1")

        legacy = self._client(tokenizer).post("/v1/messages/count_tokens", json=self.BODY).json()

        # The old count encoded with the tokenizer's default special tokens (BOS),
        # which the chat route does not add; the shared pipeline matches the route.
        assert legacy["input_tokens"] == modern["input_tokens"] + 1


class TestResponsesInputTokens:
    @pytest.mark.asyncio
    async def test_the_server_template_applies_to_responses_input(
        self, tokenizer, monkeypatch
    ) -> None:
        import tensorrt_llm.serve.responses_utils as responses_utils

        async def fake_input_messages(request, prev_msgs):
            return [{"role": "user", "content": "hello world"}]

        monkeypatch.setattr(responses_utils, "_create_input_messages", fake_input_messages)
        monkeypatch.setattr(
            responses_utils, "_get_chat_completion_function_tools", lambda tools: []
        )
        monkeypatch.setattr(responses_utils, "reasoning_chat_template_kwargs", lambda request: {})
        monkeypatch.setattr(
            responses_utils, "reasoning_injected_chat_template_keys", lambda request: set()
        )
        request = mock.Mock()
        request.tools = None
        request.store = False

        async def tokens(chat_template):
            ids, _mm = await responses_utils._create_input_tokens(
                request=request,
                prev_response=None,
                prev_msgs=None,
                conversation_store=None,
                enable_store=False,
                tokenizer=tokenizer,
                model_config=SimpleNamespace(model_type=MODEL_TYPE),
                processor=None,
                chat_template=chat_template,
            )
            return ids

        with_template = await tokens(SERVER_TEMPLATE)
        without = await tokens(None)

        expected = render_chat(
            chat_request(messages=[{"role": "user", "content": "hello world"}]),
            resources(tokenizer, default_chat_template=SERVER_TEMPLATE),
        )
        assert with_template == expected.token_ids
        assert with_template != without
