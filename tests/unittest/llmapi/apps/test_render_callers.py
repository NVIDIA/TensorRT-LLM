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

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm.serve.render import RenderResources, render_chat
from tensorrt_llm.serve.router_utils import KV_CACHE_HASH_ALGO_DEFAULT

from .render_helpers import (
    KIMI_K3,
    SERVER_TEMPLATE,
    WEATHER_TOOL,
    chat_request,
    fallback_processor,
    make_tokenizer,
    resources,
)

pytestmark = pytest.mark.cpu_only

MODEL_TYPE = "render-test-model"


@pytest.fixture(scope="module")
def tokenizer():
    return make_tokenizer()


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
    def worker_fingerprint(tokenizer, reasoning_parser=None):
        """A worker's fingerprint, built from a worker-shaped server and not from the router.

        Copying the router's own fingerprint would make the write-back tests agree by
        construction and hide any difference between how the two sides describe the
        same rendering configuration. A worker's ``processor`` is ``AutoProcessor``'s result,
        which is the tokenizer itself for a text-only checkpoint, not ``None``. The worker's
        reasoning parser is not part of the rendering configuration, but the router reads it.
        """
        from tensorrt_llm.inputs.registry import DefaultInputProcessor

        server = SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(model_type=None),
            processor=fallback_processor(tokenizer),
            chat_template=None,
            allow_request_chat_template=False,
            generator=SimpleNamespace(
                args=SimpleNamespace(reasoning_parser=reasoning_parser),
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

    def test_ids_are_not_forwarded_when_a_worker_has_no_server_info(
        self, router, tokenizer
    ) -> None:
        # server2 is active, but fetching its /server_info failed, so it has no entry.
        self._trust(router, tokenizer, servers=("server1",))
        request = chat_request()

        with mock.patch("tensorrt_llm.serve.router_utils.logger") as logger:
            router._tokenize(request)

        assert request.prompt_token_ids is None
        # The warning names the worker that blocks forwarding, and why.
        message = logger.warning.call_args.args[0]
        assert "server2 (no rendering fingerprint reported)" in message
        assert "server1" not in message

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

        with mock.patch("tensorrt_llm.serve.router_utils.logger") as logger:
            router._tokenize(request)

        assert request.prompt_token_ids is None
        assert "server2 (renders differently" in logger.warning.call_args.args[0]

    def test_the_fallback_is_reported_again_only_when_its_cause_changes(
        self, router, tokenizer
    ) -> None:
        fingerprint = _Router.worker_fingerprint(tokenizer)
        router._server_info = {"server1": {"render_fingerprint": fingerprint}, "server2": {}}

        with mock.patch("tensorrt_llm.serve.router_utils.logger") as logger:

            def reports():
                return [c for c in logger.warning.call_args_list if "not forwarded" in c.args[0]]

            router._tokenize(chat_request())
            router._tokenize(chat_request())
            assert len(reports()) == 1  # the same cause is not repeated

            router._server_info["server2"] = {"render_fingerprint": {**fingerprint, "digest": "x"}}
            router._tokenize(chat_request())
            assert len(reports()) == 2  # a different cause is

            self._trust(router, tokenizer)
            forwarded = chat_request()
            router._tokenize(forwarded)
            assert forwarded.prompt_token_ids is not None
            router._server_info["server2"] = {}
            router._tokenize(chat_request())
            assert len(reports()) == 3  # forwarding resumed, then broke again

    def test_concurrent_decisions_never_forward_to_a_mismatching_worker(
        self, router, tokenizer
    ) -> None:
        # _tokenize runs on several threads; the decision must not live on the instance.
        import sys
        import threading

        fingerprint = _Router.worker_fingerprint(tokenizer)
        router._server_info = {
            "server1": {"render_fingerprint": fingerprint},
            "server2": {"render_fingerprint": {**fingerprint, "digest": "a-different-digest"}},
        }
        assert router._may_write_back_token_ids("m") is False
        assert not {"_write_back_blockers", "_thinking_blockers"} & set(vars(router))
        wrong = []

        def decide():
            for _ in range(4000):
                if router._may_write_back_token_ids("m"):
                    wrong.append(1)

        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # switch threads as often as possible
        try:
            threads = [threading.Thread(target=decide) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120)
        finally:
            sys.setswitchinterval(interval)
        assert not any(thread.is_alive() for thread in threads)
        assert wrong == []

    def test_ids_are_not_forwarded_when_no_worker_has_reported_yet(self, router) -> None:
        router._server_info = {}
        request = chat_request()

        router._tokenize(request)

        assert request.prompt_token_ids is None

    THINKING_TEMPLATE = (
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
        "{% if add_generation_prompt %}<assistant><think>{% endif %}"
    )

    def _thinking_router(self, reasoning_parser):
        thinking_tokenizer = make_tokenizer(chat_template=self.THINKING_TEMPLATE)
        built = _Router.build(thinking_tokenizer, ["server1"])
        built._server_info = {
            "server1": {
                "render_fingerprint": _Router.worker_fingerprint(
                    thinking_tokenizer, reasoning_parser=reasoning_parser
                )
            }
        }
        return built

    @pytest.mark.parametrize("parser", ["poolside_v1", "laguna"])
    def test_ids_are_not_forwarded_to_a_worker_whose_parser_reads_the_mode_off_the_prompt(
        self, parser
    ) -> None:
        # A forwarded request is an ordinary chat request carrying prompt_token_ids: the
        # worker never sees the rendered text, so a reasoning mode read off the prompt (here a
        # template that prefills "<think>") would be lost. The worker must render it itself.
        built = self._thinking_router(parser)
        try:
            request = chat_request()

            ids = built._tokenize(request)[0]

            assert ids  # still used for routing
            assert request.prompt_token_ids is None
        finally:
            built._test_patcher.stop()

    @pytest.mark.parametrize("parser", [None, "qwen3", "deepseek-r1"])
    def test_ids_are_forwarded_to_a_worker_whose_parser_ignores_the_prompt(self, parser) -> None:
        built = self._thinking_router(parser)
        try:
            request = chat_request()

            ids = built._tokenize(request)[0]

            assert request.prompt_token_ids == ids
        finally:
            built._test_patcher.stop()

    def test_a_worker_that_does_not_say_which_parser_it_runs_is_not_given_the_ids(self) -> None:
        built = self._thinking_router(None)
        try:
            info = built._server_info["server1"]["render_fingerprint"]
            del info["info"]["reasoning_parser"]
            request = chat_request()

            built._tokenize(request)

            assert request.prompt_token_ids is None
        finally:
            built._test_patcher.stop()

    def test_qwen3_without_thinking_is_forwarded_to_a_qwen3_parser_worker(
        self, router, tokenizer
    ) -> None:
        # The real Qwen3 template prefills an empty think block for enable_thinking=false; the
        # qwen3 parser does not read that off the prompt, so nothing is lost by forwarding.
        fingerprint = _Router.worker_fingerprint(tokenizer, reasoning_parser="qwen3")
        router._server_info = {
            name: {"render_fingerprint": fingerprint} for name in router._servers
        }
        request = chat_request(chat_template_kwargs={"enable_thinking": False})

        ids = router._tokenize(request)[0]

        assert request.prompt_token_ids == ids
        assert ids == self._expected_ids(
            tokenizer, chat_request(chat_template_kwargs={"enable_thinking": False})
        )

    def test_a_request_a_serving_extension_rejects_is_routed_on_an_estimate(
        self, monkeypatch
    ) -> None:
        # kimi_k3 refuses out-of-policy sampling parameters. The router has to leave that to
        # the worker (which answers the request) rather than fail it itself.
        monkeypatch.setenv("TRTLLM_KIMI_PARAM_POLICY", "1")
        k3 = make_tokenizer(name=KIMI_K3)
        built = _Router.build(k3, ["server1"])
        try:
            built._model_path = "/model"
            built._model_types["/model"] = "kimi_k3"
            request = chat_request(n=2)

            ids = built._tokenize(request)[0]

            assert ids  # routed on the request's text
            assert request.prompt_token_ids is None  # and left to the worker, which answers it
        finally:
            built._test_patcher.stop()

    def test_the_conversation_key_does_not_depend_on_whether_ids_were_forwarded(
        self, router, tokenizer
    ) -> None:
        # The router keeps a conversation on one worker by the first messages and the first
        # prompt tokens. Whether the ids are also sent to the workers must not move it.
        router._routing_hash_algo = KV_CACHE_HASH_ALGO_DEFAULT
        self._trust(router, tokenizer)
        forwarded = chat_request()
        key_forwarded = router._routing_key_sync(forwarded)["conv_key"]
        assert forwarded.prompt_token_ids is not None

        router._server_info = {}
        held_back = chat_request()
        key_held_back = router._routing_key_sync(held_back)["conv_key"]
        assert held_back.prompt_token_ids is None

        assert key_forwarded == key_held_back

    def test_ids_are_forwarded_when_the_prompt_carries_no_decisions(
        self, router, tokenizer
    ) -> None:
        self._trust(router, tokenizer)
        request = chat_request()

        router._tokenize(request)

        assert request.prompt_token_ids is not None

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

    def test_a_request_the_router_rendered_but_did_not_write_back_keeps_its_tool_calls(
        self, router
    ) -> None:
        # `tool_calls` validates into a single-use iterator that rendering would drain, and the
        # worker renders the history itself when the ids are not written back.
        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        request = chat_request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ]
        )

        router._tokenize(request)  # no worker reported a fingerprint: nothing is written back

        assert request.prompt_token_ids is None
        sent = request.model_dump(mode="json", exclude_unset=True)["messages"][1]
        assert [item["id"] for item in sent["tool_calls"]] == ["c1"]

    def test_a_harmony_router_renders_without_loading_a_tokenizer(self) -> None:
        # Harmony renders ids itself. A client that sends the served name, with no tokenizer
        # directory configured, must not make the router try to load a tokenizer for it.
        from tensorrt_llm.serve.router import KvCacheAwareRouter

        harmony_router = KvCacheAwareRouter(
            server_role=None,
            servers=["server1"],
            use_tokens=False,
            max_batch_size=32,
            tokens_per_block=32,
            use_harmony=True,
        )
        with mock.patch.object(harmony_router, "_get_tokenizer", side_effect=AssertionError):
            res = harmony_router._render_resources("gpt-oss-20b")
            fingerprint = res.fingerprint()

        assert res.use_harmony and fingerprint["compared"]["use_harmony"]

    def test_an_explicit_use_harmony_renders_without_a_resolvable_model_type(
        self, monkeypatch
    ) -> None:
        # No checkpoint path, so no model type: the operator sets use_harmony, as the router's
        # warning says. The request must render through Harmony, not fail in the template path.
        from utils.llm_data import llm_datasets_root

        from tensorrt_llm.serve.chat_tokenization import tokenize_harmony_chat_request
        from tensorrt_llm.serve.router import KvCacheAwareRouter

        vocab = Path(llm_datasets_root()) / "tiktoken_vocab"
        if not vocab.is_dir():
            pytest.skip("the Harmony vocabulary is not available")
        monkeypatch.setenv("TIKTOKEN_RS_CACHE_DIR", str(vocab))
        monkeypatch.setenv("TIKTOKEN_ENCODINGS_BASE", str(vocab))
        router = KvCacheAwareRouter(
            server_role=None,
            servers=["server1"],
            use_tokens=False,
            max_batch_size=32,
            tokens_per_block=32,
            use_harmony=True,
        )
        request = chat_request()

        rendered = render_chat(request, router._render_resources("gpt-oss-20b"), tokenize=True)

        assert rendered.token_ids == tokenize_harmony_chat_request(chat_request())

    def test_a_harmony_conversion_error_is_routed_on_no_ids(self, harmony_vocab) -> None:
        # A tool without a description cannot be converted to Harmony.
        from tensorrt_llm.serve.router import KvCacheAwareRouter

        router = KvCacheAwareRouter(
            server_role=None,
            servers=["server1"],
            use_tokens=False,
            max_batch_size=32,
            tokens_per_block=32,
            use_harmony=True,
        )
        no_description = {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {}}},
        }
        named = {"type": "function", "function": {"name": "f"}}
        for tool_choice in (None, named):
            request = chat_request(tools=[no_description], tool_choice=tool_choice)

            assert router._tokenize(request) == [[]]
            assert request.prompt_token_ids is None

    def test_object_valued_tool_arguments_are_routed_on_no_ids_not_on_a_partial_render(
        self, router, tokenizer
    ) -> None:
        # The exact render spends the message iterators; a second render would drop the tool call.
        self._trust(router, tokenizer)
        request = chat_request(
            messages=[
                {"role": "user", "content": "weather in paris?"},
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
                {"role": "user", "content": "thanks"},
            ],
            tools=[WEATHER_TOOL],
        )

        assert router._tokenize(request) == [[]]
        assert request.prompt_token_ids is None

    def test_a_request_that_is_already_pre_tokenized_is_left_alone(self, router, tokenizer) -> None:
        self._trust(router, tokenizer)
        request = chat_request(prompt_token_ids=[4, 5, 6])

        assert router._tokenize(request) == [[4, 5, 6]]
        assert request.prompt_token_ids == [4, 5, 6]


class TestCoordinatorFrontendWriteBack:
    """A coordinator frontend never fetches /server_info; the coordinator publishes summaries."""

    def _published(self, tokenizer, servers=("server1", "server2")):
        from tensorrt_llm.serve.render import fingerprint_summary

        summary = fingerprint_summary(_Router.worker_fingerprint(tokenizer))
        return {name: summary for name in servers}

    def test_a_summary_matches_exactly_like_the_fingerprint_it_came_from(self, tokenizer) -> None:
        from tensorrt_llm.serve.render import fingerprint_summary, fingerprints_match

        full = _Router.worker_fingerprint(tokenizer)
        other = {**full, "digest": "a-different-digest"}

        assert fingerprints_match(full, fingerprint_summary(full))
        assert not fingerprints_match(full, fingerprint_summary(other))
        assert fingerprint_summary(None) is None

    def test_ids_are_forwarded_when_the_published_fingerprints_match(
        self, router, tokenizer
    ) -> None:
        # The frontend's own _server_info stays empty, as in coordinator mode.
        router._server_info = {}
        router.set_published_render_fingerprints(self._published(tokenizer))
        request = chat_request()

        ids = router._tokenize(request)[0]

        assert request.prompt_token_ids == ids

    def test_ids_are_not_forwarded_when_a_published_fingerprint_differs(
        self, router, tokenizer
    ) -> None:
        published = self._published(tokenizer)
        published["server2"] = {**published["server2"], "digest": "a-different-digest"}
        router.set_published_render_fingerprints(published)
        request = chat_request()

        router._tokenize(request)

        assert request.prompt_token_ids is None

    def test_a_server_the_coordinator_has_not_reported_is_a_mismatch(
        self, router, tokenizer
    ) -> None:
        router.set_published_render_fingerprints(self._published(tokenizer, servers=("server1",)))
        request = chat_request()

        router._tokenize(request)

        assert request.prompt_token_ids is None

    def test_a_worker_with_no_fingerprint_is_a_mismatch(self, router, tokenizer) -> None:
        published = self._published(tokenizer)
        published["server2"] = None  # an older worker
        router.set_published_render_fingerprints(published)
        request = chat_request()

        router._tokenize(request)

        assert request.prompt_token_ids is None


class TestServerInfoFingerprint:
    @pytest.mark.asyncio
    async def test_the_first_call_hashes_off_the_event_loop(self, monkeypatch) -> None:
        import asyncio
        import json

        from tensorrt_llm.serve.openai_server import OpenAIServer

        where = []

        class Resources:
            def fingerprint(self):
                try:
                    asyncio.get_running_loop()
                    where.append("event loop")
                except RuntimeError:
                    where.append("worker thread")
                return {"digest": "d"}

        monkeypatch.setattr(RenderResources, "from_server", classmethod(lambda cls, s: Resources()))
        server = object.__new__(OpenAIServer)
        server.generator = SimpleNamespace(disaggregated_params=None, args=None)

        first = json.loads((await server.get_server_info()).body)
        second = json.loads((await server.get_server_info()).body)

        assert first["render_fingerprint"] == second["render_fingerprint"] == {"digest": "d"}
        assert where == ["worker thread"]  # computed once, then cached


class TestCountTokens:
    def _client(
        self,
        tokenizer,
        chat_template=None,
        use_harmony=False,
        model_type=None,
        input_processor=None,
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tensorrt_llm.serve.openai_server import OpenAIServer

        app = FastAPI()
        server = object.__new__(OpenAIServer)
        server.model = "m"
        server.tokenizer = tokenizer
        server.chat_template = chat_template
        if use_harmony:
            server.use_harmony = True
            server.model_config = SimpleNamespace(model_type="gpt_oss")
        elif model_type is not None:
            server.model_config = SimpleNamespace(model_type=model_type)
        if input_processor is not None:
            server.generator = SimpleNamespace(input_processor=input_processor)
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

    def test_a_harmony_server_counts_the_harmony_prompt(self, tokenizer, harmony_vocab) -> None:
        from tensorrt_llm.serve.chat_tokenization import tokenize_harmony_chat_request

        client = self._client(tokenizer, use_harmony=True)

        response = client.post("/v1/messages/count_tokens", json=self.BODY)

        assert response.status_code == 200, response.text
        expected = tokenize_harmony_chat_request(
            chat_request(messages=[{"role": "user", "content": "hello world"}])
        )
        assert response.json() == {"input_tokens": len(expected)}

    def test_a_harmony_server_estimates_what_harmony_cannot_convert(
        self, tokenizer, harmony_vocab
    ) -> None:
        # A tool without a description is rejected by Harmony; the count falls back to the
        # template count.
        body = {
            **self.BODY,
            "tools": [{"name": "f", "input_schema": {"type": "object", "properties": {}}}],
        }

        response = self._client(tokenizer, use_harmony=True).post(
            "/v1/messages/count_tokens", json=body
        )

        assert response.status_code == 200, response.text
        assert response.json()["input_tokens"] > 0

    def test_a_request_with_media_is_counted_with_its_system_blocks_tools_and_tool_calls(
        self, tokenizer
    ) -> None:
        # Media is refused by the exact render; the count still covers the rest of the prompt.
        tool = {
            "name": "get_weather",
            "description": "Get the weather.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
        image = {"type": "image", "source": {"type": "url", "url": "http://example.invalid/x.png"}}
        text = {"type": "text", "text": "look at this"}

        def count(first_turn, tool_input) -> int:
            body = {
                "model": "m",
                "system": "be terse",
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "tools": [tool],
                "messages": [
                    {"role": "user", "content": first_turn},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "get_weather",
                                "input": tool_input,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t1", "content": "sunny"}
                        ],
                    },
                ],
            }
            response = self._client(tokenizer).post("/v1/messages/count_tokens", json=body)
            assert response.status_code == 200, response.text
            return response.json()["input_tokens"]

        paris = {"city": "paris", "units": "metric"}
        assert count([text, image], paris) == count([text], paris)  # the media adds nothing
        assert count([text, image], paris) > count([text, image], {})  # the tool call is counted

    def test_a_passthrough_model_with_its_own_input_processor_is_counted(self, tokenizer) -> None:
        # A passthrough model carries its own input processor; the count uses the text tokenizer.
        import tensorrt_llm._torch.models.modeling_mistral  # noqa: F401

        class OwnProcessor:
            tokenizer = None

        client = self._client(
            tokenizer, model_type="mistral_common", input_processor=OwnProcessor()
        )

        response = client.post("/v1/messages/count_tokens", json=self.BODY)

        assert response.status_code == 200, response.text
        plain = self._client(tokenizer).post("/v1/messages/count_tokens", json=self.BODY).json()
        assert response.json() == plain

    def test_the_server_template_is_part_of_the_count(self, tokenizer) -> None:
        plain = self._client(tokenizer).post("/v1/messages/count_tokens", json=self.BODY).json()
        templated = (
            self._client(tokenizer, chat_template=SERVER_TEMPLATE)
            .post("/v1/messages/count_tokens", json=self.BODY)
            .json()
        )
        assert templated != plain


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
        monkeypatch.setattr(
            responses_utils, "reasoning_chat_template_kwargs", lambda request, **kwargs: {}
        )
        monkeypatch.setattr(
            responses_utils, "reasoning_injected_chat_template_keys", lambda request: set()
        )
        request = mock.Mock()
        request.tools = None
        request.store = False

        async def tokens(chat_template):
            return await responses_utils._create_input_tokens(
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

        with_template = await tokens(SERVER_TEMPLATE)
        without = await tokens(None)

        expected = render_chat(
            chat_request(messages=[{"role": "user", "content": "hello world"}]),
            resources(tokenizer, default_chat_template=SERVER_TEMPLATE),
        )
        assert with_template == expected.token_ids
        assert with_template != without
