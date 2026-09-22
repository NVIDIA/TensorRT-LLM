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
"""Contract tests for the per-model serving-extension registry."""

from types import SimpleNamespace

import pytest

from tensorrt_llm.serve import serving_extensions
from tensorrt_llm.serve.serving_extensions import (
    ServingExtension,
    apply_model_chat_extensions,
    register_serving_extension,
    structured_output_format_for,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def scratch_extension():
    """Register a recording extension under unique keys; unregister after."""
    model_key = "serving-extensions-test-model"
    parser_key = "serving-extensions-test-parser"

    @register_serving_extension(model_types=(model_key,), reasoning_parsers=(parser_key,))
    class RecordingExtension(ServingExtension):
        applied_requests: list = []

        def apply_chat_extensions(self, request) -> None:
            self.applied_requests.append(request)

        def structured_output_format(self, content, chat_template_kwargs):
            return {"content": content, "chat_template_kwargs": chat_template_kwargs}

    try:
        yield SimpleNamespace(model_key=model_key, parser_key=parser_key, cls=RecordingExtension)
    finally:
        serving_extensions._BY_MODEL_TYPE.pop(model_key, None)
        serving_extensions._BY_REASONING_PARSER.pop(parser_key, None)


class TestChatExtensionDispatch:
    def test_registered_model_type_dispatches(self, scratch_extension) -> None:
        request = SimpleNamespace()
        apply_model_chat_extensions(request, scratch_extension.model_key)
        assert scratch_extension.cls.applied_requests == [request]

    @pytest.mark.parametrize("model_type", [None, "unregistered-model"])
    def test_unregistered_model_type_is_a_no_op(self, scratch_extension, model_type) -> None:
        apply_model_chat_extensions(SimpleNamespace(), model_type)
        assert scratch_extension.cls.applied_requests == []


class TestStructuredOutputDispatch:
    def test_registered_parser_returns_bound_hook(self, scratch_extension) -> None:
        hook = structured_output_format_for(scratch_extension.parser_key)
        content = {"type": "json_schema", "json_schema": {"type": "object"}}
        kwargs = {"flag": True}
        assert hook(content, kwargs) == {"content": content, "chat_template_kwargs": kwargs}

    @pytest.mark.parametrize("parser", [None, "unregistered-parser"])
    def test_unregistered_parser_returns_none(self, scratch_extension, parser) -> None:
        assert structured_output_format_for(parser) is None

    def test_base_class_default_means_raw_grammar(self) -> None:
        assert ServingExtension().structured_output_format({}, None) is None

    def test_hook_none_result_falls_back_to_raw_grammar(self) -> None:
        """A registered hook returning None leaves the grammar unwrapped.

        End to end through _response_format_to_guided_decoding_params,
        matching the contract documented on ServingExtension.
        """
        import json

        from tensorrt_llm.serve.openai_protocol import (
            ResponseFormat,
            _response_format_to_guided_decoding_params,
        )

        parser_key = "serving-extensions-none-parser"

        @register_serving_extension(reasoning_parsers=(parser_key,))
        class RawGrammarExtension(ServingExtension):
            pass

        try:
            params = _response_format_to_guided_decoding_params(
                ResponseFormat(type="json_object"),
                reasoning_parser=parser_key,
                chat_template_kwargs=None,
            )
            assert params.structural_tag is None
            assert params.json_object is True

            # And a dict result is wrapped into the structural tag.
            serving_extensions._BY_REASONING_PARSER[parser_key].structured_output_format = (
                lambda content, chat_template_kwargs: {"type": "sequence", "elements": [content]}
            )
            params = _response_format_to_guided_decoding_params(
                ResponseFormat(type="json_object"),
                reasoning_parser=parser_key,
                chat_template_kwargs=None,
            )
            assert json.loads(params.structural_tag)["format"]["type"] == "sequence"
        finally:
            serving_extensions._BY_REASONING_PARSER.pop(parser_key, None)


class TestOpenAIChatIntegration:
    """``openai_chat`` runs the model-type hook, and before rendering.

    The dispatch tests above call ``apply_model_chat_extensions`` directly, so
    they would still pass if ``openai_chat`` dropped the call or moved it past
    ``async_apply_chat_template``. This drives the real handler on a server
    built with ``object.__new__`` (the pattern in
    ``test_openai_chat_disagg_multimodal.py``), with the renderer and engine
    stubbed.
    """

    MODEL_KEY = "serving-extensions-chat-model"
    MARKER = {"serving_extension_ran": True}

    @pytest.fixture
    def ordering_extension(self):
        """Extension that logs its turn and stamps ``chat_template_kwargs``."""
        events: list = []
        marker = self.MARKER

        @register_serving_extension(model_types=(self.MODEL_KEY,))
        class OrderingExtension(ServingExtension):
            def apply_chat_extensions(self, request) -> None:
                events.append("extension")
                request.chat_template_kwargs = {
                    **(request.chat_template_kwargs or {}),
                    **marker,
                }

        try:
            yield events
        finally:
            serving_extensions._BY_MODEL_TYPE.pop(self.MODEL_KEY, None)

    @pytest.fixture
    def chat_client(self, monkeypatch, ordering_extension):
        """``openai_chat`` on a bare app; returns (client, events, rendered)."""
        from unittest.mock import AsyncMock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tensorrt_llm.serve import openai_server
        from tensorrt_llm.serve.openai_protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
            UsageInfo,
        )
        from tensorrt_llm.serve.openai_server import OpenAIServer

        events = ordering_extension
        rendered: dict = {}

        class _StubModelConfig:
            # Class attribute: resolve_top_level_model_type reads the type.
            model_type = TestOpenAIChatIntegration.MODEL_KEY
            vocab_size = 1024

        async def fake_apply_chat_template(**kwargs) -> str:
            events.append("render")
            rendered.update(kwargs)
            return "rendered prompt"

        monkeypatch.setattr(openai_server, "async_apply_chat_template", fake_apply_chat_template)

        def generate_async(*, inputs, **kwargs):
            return SimpleNamespace(prompt_token_ids=[1, 2, 3], finished=True)

        server = object.__new__(OpenAIServer)
        server.model = "test-model"
        server.allow_request_chat_template = False
        server.model_config = _StubModelConfig()
        server.processor = None
        server.tokenizer = SimpleNamespace(
            tokenizer=SimpleNamespace(vocab_size=_StubModelConfig.vocab_size)
        )
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
                id="chatcmpl-serving-extensions-test",
                model="test-model",
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
        app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
        return TestClient(app), events, rendered

    def test_extension_runs_before_prompt_rendering(self, chat_client) -> None:
        client, events, rendered = chat_client

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"client_flag": 1},
                "max_tokens": 4,
            },
        )

        assert response.status_code == 200, response.text
        # Removed call -> ["render"]; moved after rendering -> ["render", "extension"].
        assert events == ["extension", "render"]
        # The renderer saw the request the extension mutated, with the
        # client's own kwargs preserved.
        assert rendered["chat_template_kwargs"] == {"client_flag": 1, **self.MARKER}
