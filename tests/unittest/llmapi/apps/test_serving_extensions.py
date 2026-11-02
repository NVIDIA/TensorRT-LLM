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

import sys
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve import serving_extensions
from tensorrt_llm.serve.serving_extensions import (
    ServingExtension,
    apply_model_chat_extensions,
    register_serving_extension,
    required_tool_choice_mode,
    structured_output_format_for,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("kimi_k3", "template"),
        ("nemotron_h_omni", "grammar"),
        ("nemotron_h", "grammar"),
        ("qwen3", None),
        ("qwen3_next", None),
        ("unknown", None),
        (None, None),
    ],
)
def test_required_tool_choice_mode(model_type, expected):
    assert required_tool_choice_mode(model_type) == expected


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


class TestRequiredToolChoiceDispatch:
    def test_registered_extension_defaults_to_unsupported(self, scratch_extension) -> None:
        assert required_tool_choice_mode(scratch_extension.model_key) is None


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


class TestBuiltinExtensions:
    """Built-in extensions resolve through the registry."""

    @pytest.fixture
    def clean_registry(self, monkeypatch):
        """Registry as if nothing had consulted it: empty tables, built-ins not yet imported.

        Earlier tests (or the concrete imports below) populate the registry as
        a side effect, so without this the lookups would pass even if
        ``_load_builtin_extensions`` stopped importing the built-in package.
        The package is also dropped from ``sys.modules`` so the lookup can only
        succeed through the lazy import; monkeypatch restores everything after.
        """
        monkeypatch.setattr(serving_extensions, "_builtins_loaded", False)
        monkeypatch.setattr(serving_extensions, "_BY_MODEL_TYPE", {})
        monkeypatch.setattr(serving_extensions, "_BY_REASONING_PARSER", {})
        for name in [
            mod for mod in sys.modules if mod.startswith(serving_extensions._BUILTINS_PACKAGE)
        ]:
            monkeypatch.delitem(sys.modules, name)
        parent = sys.modules["tensorrt_llm.serve"]
        monkeypatch.delattr(parent, "extensions", raising=False)

    def test_kimi_k3_resolves_to_kimi_extension(self, clean_registry) -> None:
        hook = structured_output_format_for("kimi_k3")

        from tensorrt_llm.serve.extensions.kimi_k3 import KimiK3ServingExtension

        assert isinstance(hook.__self__, KimiK3ServingExtension)
        assert isinstance(serving_extensions._BY_MODEL_TYPE["kimi_k3"], KimiK3ServingExtension)

    @pytest.mark.parametrize(
        "model_type,expected",
        [("kimi_k3", "template"), ("nemotron_h", "grammar"), ("nemotron_h_omni", "grammar")],
    )
    def test_required_tool_choice_loads_builtins(self, clean_registry, model_type, expected):
        assert required_tool_choice_mode(model_type) == expected

    @pytest.mark.parametrize("model_type", ["deepseek_v41", "deepseek_v41_text"])
    def test_deepseek_v41_resolves_to_its_extension(self, clean_registry, model_type) -> None:
        extension = serving_extensions.get_serving_extension(model_type)

        from tensorrt_llm.serve.extensions.deepseek_v41 import DeepseekV41ServingExtension

        assert isinstance(extension, DeepseekV41ServingExtension)

    def test_gpt_oss_resolves_to_gpt_oss_extension(self, clean_registry) -> None:
        hook = structured_output_format_for("gpt_oss")

        from tensorrt_llm.serve.extensions.gpt_oss import GptOssServingExtension

        assert isinstance(hook.__self__, GptOssServingExtension)
        # gpt-oss is also registered by model_type, which owns its Harmony
        # prompt rendering and output mode.
        assert isinstance(serving_extensions._BY_MODEL_TYPE["gpt_oss"], GptOssServingExtension)

    def test_kimi_param_policy_applies_via_generic_dispatch(self, monkeypatch) -> None:
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        monkeypatch.setenv("TRTLLM_KIMI_PARAM_POLICY", "1")
        messages = [{"role": "user", "content": "hi"}]

        request = ChatCompletionRequest(model="m", messages=messages, top_p=1.0)
        apply_model_chat_extensions(request, "kimi_k3")
        assert request.top_p == 0.95

        with pytest.raises(ValueError, match="n is fixed at 1"):
            apply_model_chat_extensions(
                ChatCompletionRequest(model="m", messages=messages, n=2), "kimi_k3"
            )

        # Other model types are untouched by the Kimi policy.
        request = ChatCompletionRequest(model="m", messages=messages, top_p=1.0, n=2)
        apply_model_chat_extensions(request, "llama")
        assert (request.top_p, request.n) == (1.0, 2)

    def test_kimi_chat_kwargs_derived_via_generic_dispatch(self) -> None:
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "effort": "high"},
            stream=True,
        )
        apply_model_chat_extensions(request, "kimi_k3")
        assert request.chat_template_kwargs == {"thinking": True, "thinking_effort": "high"}
        assert request.stream_options is not None
        assert request.stream_options.include_usage is True

    def test_gpt_oss_structured_output_triggers_on_final_channel(self) -> None:
        import json

        from tensorrt_llm.serve.openai_protocol import (
            ResponseFormat,
            _response_format_to_guided_decoding_params,
        )

        params = _response_format_to_guided_decoding_params(
            ResponseFormat(type="json_schema", json_schema={"schema": {"type": "object"}}),
            reasoning_parser="gpt_oss",
            chat_template_kwargs=None,
        )
        # Key-wise: the structural-tag model adds defaulted fields on dump.
        fmt = json.loads(params.structural_tag)["format"]
        final = "<|start|>assistant<|channel|>final<|message|>"
        assert fmt["type"] == "triggered_tags"
        assert fmt["triggers"] == [final]
        assert len(fmt["tags"]) == 1
        tag = fmt["tags"][0]
        assert tag["begin"] == final
        assert tag["end"] == ""
        assert tag["content"]["type"] == "json_schema"
        assert tag["content"]["json_schema"] == {"type": "object"}
        assert fmt["stop_after_first"] is True

    @pytest.mark.parametrize(
        ("chat_template_kwargs", "expect_tag"),
        [(None, True), ({"thinking": True}, True), ({"thinking": False}, False)],
    )
    def test_kimi_structured_output_follows_thinking_mode(
        self, chat_template_kwargs, expect_tag
    ) -> None:
        import json

        from tensorrt_llm.serve.openai_protocol import (
            ResponseFormat,
            _response_format_to_guided_decoding_params,
        )

        params = _response_format_to_guided_decoding_params(
            ResponseFormat(type="json_object"),
            reasoning_parser="kimi_k3",
            chat_template_kwargs=chat_template_kwargs,
        )
        if not expect_tag:
            assert params.structural_tag is None
            assert params.json_object is True
            return
        fmt = json.loads(params.structural_tag)["format"]
        assert fmt["type"] == "triggered_tags"
        assert fmt["triggers"] == ["<|open|>response<|sep|>"]
        assert fmt["tags"][0]["end"] == "<|close|>response<|sep|>"
        assert fmt["stop_after_first"] is True


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

        from tensorrt_llm.serve.openai_protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
            UsageInfo,
        )
        from tensorrt_llm.serve.openai_server import OpenAIServer
        from tensorrt_llm.serve.render import chat as render_chat_module

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

        monkeypatch.setattr(
            render_chat_module, "async_apply_chat_template", fake_apply_chat_template
        )

        def generate_async(*, inputs, **kwargs):
            rendered["generation"] = kwargs
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
        server._input_proc_executor = None
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

        rendered["create_chat_response"] = server._create_chat_response

        app = FastAPI()
        app.state.server = server
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

    def test_required_tool_choice_is_rejected_unless_the_extension_allows_it(
        self, chat_client
    ) -> None:
        client, _events, _rendered = chat_client

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_WEATHER_TOOL],
                "tool_choice": "required",
                "max_tokens": 4,
            },
        )

        assert response.status_code == 400, response.text
        assert "tool_choice='required' is not supported" in response.text

    def test_extension_hooks_drive_tools_and_the_usage_offset(self, chat_client) -> None:
        client, _events, rendered = chat_client

        class HookExtension(ServingExtension):
            def allows_required_tool_choice(self) -> bool:
                return True

            def serialize_tool(self, tool) -> dict:
                return {"custom": tool.function.name}

            def prompt_tokens_excluded_from_usage(self, request) -> int:
                return 5

        serving_extensions._BY_MODEL_TYPE[self.MODEL_KEY] = HookExtension()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_WEATHER_TOOL],
                "tool_choice": "required",
                "max_tokens": 4,
            },
        )

        assert response.status_code == 200, response.text
        assert rendered["tools"] == [{"custom": "get_weather"}]
        # _create_chat_response(promise, postproc_params, raw_request, ...)
        postproc_params = rendered["create_chat_response"].call_args.args[1]
        assert postproc_params.postproc_args.num_prompt_tokens_offset == 5

    @pytest.fixture(params=["nemotron_h_omni", "nemotron_h"])
    def required_chat_client(self, chat_client, monkeypatch, request):
        from tensorrt_llm.serve.render import chat as render_chat_module

        client, events, rendered = chat_client
        server = client.app.state.server
        type(server.model_config).model_type = request.param
        server.tool_parser = "qwen3_coder"
        server.generator.args.reasoning_parser = "nemotron-v3"
        server.generator.args.guided_decoding_backend = "xgrammar"

        async def render_prompt(**kwargs):
            thinking = rendered.get("thinking", True)
            return "<|im_start|>assistant\n<think>" + ("\n" if thinking else "</think>")

        monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render_prompt)
        return client, server, rendered

    @staticmethod
    def required_body():
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hello."}],
            "tools": [{"type": "function", "function": {"name": "ping"}}],
            "tool_choice": "required",
            "max_tokens": 64,
        }

    def test_required_template_mode_uses_kimi_extension(self, chat_client, monkeypatch):
        from tensorrt_llm.serve import chat_utils

        # Keep text-only routing independent of multimodal model initialization.
        monkeypatch.setattr(
            chat_utils.MULTIMODAL_PLACEHOLDER_REGISTRY, "get_content_format", lambda _: None
        )
        client, _, rendered = chat_client
        server = client.app.state.server
        type(server.model_config).model_type = "kimi_k3"
        server.tool_parser = "kimi_k3"
        server.generator.args.reasoning_parser = "kimi_k3"
        server.generator.args.guided_decoding_backend = None

        response = client.post("/v1/chat/completions", json=self.required_body())

        assert response.status_code == 200, response.text
        assert rendered["chat_template_kwargs"]["tool_choice"] == "required"
        assert rendered["generation"]["sampling_params"].guided_decoding is None
        postproc = server._create_chat_response.call_args.args[1].postproc_args
        assert postproc.tool_parser_grammar_constrained is False

    def test_required_grammar_admission_uses_extension_hook(self, required_chat_client):
        client, server, rendered = required_chat_client
        model_type = "required-grammar-test-model"
        type(server.model_config).model_type = model_type

        @register_serving_extension(model_types=(model_type,))
        class RequiredGrammarExtension(ServingExtension):
            def required_tool_choice_mode(self):
                return "grammar"

        try:
            response = client.post("/v1/chat/completions", json=self.required_body())
        finally:
            serving_extensions._BY_MODEL_TYPE.pop(model_type, None)

        assert response.status_code == 200, response.text
        assert rendered["generation"]["sampling_params"].guided_decoding is not None

    @pytest.mark.parametrize("thinking", [False, True])
    @pytest.mark.parametrize("workers", [0, 4])
    def test_required_attaches_mandatory_grammar(self, required_chat_client, thinking, workers):
        import xgrammar

        client, server, rendered = required_chat_client
        rendered["thinking"] = thinking
        server.generator.args.num_postprocess_workers = workers
        body = self.required_body()
        body["chat_template_kwargs"] = {"enable_thinking": not thinking}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        params = rendered["generation"]["sampling_params"].guided_decoding
        assert params is not None
        compiler = xgrammar.GrammarCompiler(
            xgrammar.TokenizerInfo(
                [bytes([i]) for i in range(256)] + [b"<EOS>"], stop_token_ids=[256]
            )
        )
        compiled = compiler.compile_structural_tag(params.structural_tag)
        matcher = xgrammar.GrammarMatcher(compiled)
        if thinking:
            assert matcher.accept_string("Considering.</think>")
        assert not matcher.accept_token(256)
        assert matcher.accept_string("<tool_call>\n<function=ping>\n</function>\n</tool_call>")
        assert matcher.accept_token(256)
        postproc = server._create_chat_response.call_args.args[1].postproc_args
        assert postproc.chat_template_kwargs["enable_thinking"] is thinking
        assert postproc.tool_parser_grammar_constrained is True

    @pytest.mark.parametrize(
        "change",
        [
            {"tools": []},
            {"prompt_token_ids": [1, 2]},
            {"prompt_token_ids_b64": "AQAAAA=="},
            {"add_generation_prompt": False},
            {"disaggregated_params": {"request_type": "context_only"}},
            {"response_format": {"type": "json_object"}},
        ],
    )
    def test_required_rejects_conflicting_requests(self, required_chat_client, change):
        client, server, rendered = required_chat_client
        response = client.post("/v1/chat/completions", json={**self.required_body(), **change})
        # The bare app uses FastAPI's 422 for request-schema errors.
        expected_status = 422 if change.get("tools") == [] else 400
        assert response.status_code == expected_status, response.text
        assert "generation" not in rendered

    @pytest.mark.parametrize(
        "model_type,parser,backend",
        [
            ("qwen3", "qwen3_coder", "xgrammar"),
            ("qwen3_moe", "qwen3_coder", "xgrammar"),
            ("qwen3_next", "qwen3_coder", "xgrammar"),
            ("nemotron_h_omni", "qwen3", "xgrammar"),
            ("nemotron_h_omni", None, "xgrammar"),
            ("nemotron_h_omni", "qwen3_coder", None),
            ("nemotron_h_omni", "qwen3_coder", "llguidance"),
        ],
    )
    def test_required_rejects_unsupported_configuration(
        self, required_chat_client, model_type, parser, backend
    ):
        client, server, rendered = required_chat_client
        type(server.model_config).model_type = model_type
        server.tool_parser = parser
        server.generator.args.guided_decoding_backend = backend
        response = client.post("/v1/chat/completions", json=self.required_body())
        assert response.status_code == 400, response.text
        assert "generation" not in rendered

    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "object", "properties": {}, "additionalProperties": True},
            {"type": "object", "properties": {}, "additionalProperties": {}},
            {"type": "object", "required": "x"},
        ],
    )
    def test_required_schema_errors_are_client_errors(self, required_chat_client, schema):
        client, server, rendered = required_chat_client
        body = self.required_body()
        body["tools"][0]["function"]["parameters"] = schema
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 400
        assert "generation" not in rendered
        assert "EBNF" not in response.text
        assert "json_schema_converter" not in response.text

    @pytest.mark.parametrize(
        "reasoning_parser", ["NEMOTRON-V3", "nano-v3", "NANO-V3", "k-exaone", "K-EXAONE"]
    )
    def test_required_uses_parser_capabilities(
        self, required_chat_client, monkeypatch, reasoning_parser
    ):
        from tensorrt_llm.serve.render import chat as render_chat_module
        from tensorrt_llm.serve.tool_parser.qwen3_coder_parser import Qwen3CoderToolParser
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

        client, server, rendered = required_chat_client
        monkeypatch.setitem(ToolParserFactory.parsers, "custom_xml", Qwen3CoderToolParser)
        server.tool_parser = "CUSTOM_XML"
        server.generator.args.reasoning_parser = reasoning_parser

        async def render_prompt(**kwargs):
            return "<assistant></think>\n"

        monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render_prompt)
        response = client.post("/v1/chat/completions", json=self.required_body())
        assert response.status_code == 200, response.text
        assert rendered["generation"]["sampling_params"].guided_decoding is not None
        args = server._create_chat_response.call_args.args[1].postproc_args
        assert args.chat_template_kwargs["enable_thinking"] is False

    @pytest.mark.parametrize("reasoning_parser", [None, "deepseek-r1"])
    def test_required_reports_missing_reasoning_capability(
        self, required_chat_client, monkeypatch, reasoning_parser
    ):
        from unittest.mock import AsyncMock

        from tensorrt_llm.serve.render import chat as render_chat_module

        client, server, rendered = required_chat_client
        server.generator.args.reasoning_parser = reasoning_parser
        render_prompt = AsyncMock()
        monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render_prompt)

        response = client.post("/v1/chat/completions", json=self.required_body())

        assert response.status_code == 400, response.text
        assert "need a reasoning parser that resolves its mode" in response.text
        assert "--reasoning_parser nemotron-v3" in response.text
        render_prompt.assert_not_called()
        assert "generation" not in rendered

    def test_required_rejects_unresolved_prompt(self, required_chat_client, monkeypatch):
        from unittest.mock import AsyncMock

        from tensorrt_llm.serve.render import chat as render_chat_module

        client, server, rendered = required_chat_client
        monkeypatch.setattr(
            render_chat_module, "async_apply_chat_template", AsyncMock(return_value="assistant:")
        )
        response = client.post("/v1/chat/completions", json=self.required_body())
        assert response.status_code == 400, response.text
        assert (
            "Cannot resolve required-tool reasoning state from the rendered prompt" in response.text
        )
        assert "--reasoning_parser" not in response.text
        assert "generation" not in rendered

    @pytest.mark.parametrize("reasoning_parser", ["nemotron-v3", "nano-v3", "k-exaone"])
    @pytest.mark.parametrize("thinking", [False, True])
    @pytest.mark.parametrize("tool_choice", [None, "auto"])
    def test_non_required_chat_resolves_mode_from_prompt(
        self, chat_client, monkeypatch, reasoning_parser, thinking, tool_choice
    ):
        from unittest.mock import AsyncMock

        from tensorrt_llm.serve.render import chat as render_chat_module

        client, _, rendered = chat_client
        server = client.app.state.server
        server.generator.args.reasoning_parser = reasoning_parser
        prompt = "<|im_start|>assistant\n<think>" + ("\n" if thinking else "</think>")
        monkeypatch.setattr(
            render_chat_module, "async_apply_chat_template", AsyncMock(return_value=prompt)
        )
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hello."}],
            "chat_template_kwargs": {"thinking": not thinking, "enable_thinking": not thinking},
        }
        if tool_choice is not None:
            body.update(tools=self.required_body()["tools"], tool_choice=tool_choice)
            server.tool_parser = "qwen3_coder"

        response = client.post("/v1/chat/completions", json=body)

        assert response.status_code == 200, response.text
        assert rendered["generation"]["sampling_params"].guided_decoding is None
        postproc = server._create_chat_response.call_args.args[1].postproc_args
        assert postproc.chat_template_kwargs["thinking"] is thinking
        assert postproc.chat_template_kwargs["enable_thinking"] is thinking
        assert postproc.tool_parser_grammar_constrained is False

    def test_auto_does_not_attach_required_grammar(self, required_chat_client):
        client, server, rendered = required_chat_client
        response = client.post(
            "/v1/chat/completions", json={**self.required_body(), "tool_choice": "auto"}
        )
        assert response.status_code == 200, response.text
        assert rendered["generation"]["sampling_params"].guided_decoding is None
        postproc = server._create_chat_response.call_args.args[1].postproc_args
        assert postproc.tool_parser_grammar_constrained is False


def _chat_request(**overrides):
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    fields = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    fields.update(overrides)
    return ChatCompletionRequest(**fields)


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


class TestExtensionHooks:
    """Per-model hooks: defaults match the generic path, built-ins override."""

    def test_unregistered_model_type_gets_the_all_defaults_extension(self) -> None:
        for model_type in ("not-a-registered-model", "", None):
            extension = serving_extensions.get_serving_extension(model_type)
            assert type(extension) is ServingExtension

    def test_registered_model_type_gets_its_extension(self, scratch_extension) -> None:
        extension = serving_extensions.get_serving_extension(scratch_extension.model_key)
        assert isinstance(extension, scratch_extension.cls)

    def test_default_hooks_match_the_generic_path(self) -> None:
        extension = serving_extensions.get_serving_extension(None)
        request = _chat_request(tools=[_WEATHER_TOOL])

        assert extension.render_prompt(request) is None
        assert extension.output_mode() is serving_extensions.OutputMode.TEXT
        assert extension.allows_required_tool_choice() is False
        assert extension.serialize_tool(request.tools[0]) == request.tools[0].model_dump()
        assert extension.dynamic_tools(request.messages) == []
        assert extension.prompt_tokens_excluded_from_usage(request) == 0

    def test_load_builtin_extensions_keeps_its_private_alias(self) -> None:
        assert (
            serving_extensions._load_builtin_extensions
            is serving_extensions.load_builtin_extensions
        )

    @pytest.fixture
    def kimi(self):
        return serving_extensions.get_serving_extension("kimi_k3")

    def test_kimi_k3_allows_required_tool_choice(self, kimi) -> None:
        assert kimi.allows_required_tool_choice() is True

    def test_kimi_k3_serializes_tools_without_null_defaults(self, kimi) -> None:
        request = _chat_request(tools=[_WEATHER_TOOL])
        tool = request.tools[0]

        assert kimi.serialize_tool(tool) == tool.model_dump(exclude_none=True)
        # The default dump carries pydantic-injected nulls that K3 must not render.
        assert kimi.serialize_tool(tool) != tool.model_dump()

    def test_kimi_k3_collects_message_level_tools(self, kimi) -> None:
        messages = [{"role": "system", "content": "", "tools": [_WEATHER_TOOL]}]
        assert kimi.dynamic_tools(messages) == [_WEATHER_TOOL]
        assert kimi.dynamic_tools([{"role": "user", "content": "hi"}]) == []

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            pytest.param({}, 3, id="native_template_with_generation_prompt"),
            pytest.param({"add_generation_prompt": False}, 0, id="no_generation_prompt"),
            pytest.param({"prompt_token_ids": [1, 2, 3]}, 0, id="pre_tokenized"),
            pytest.param({"prompt_token_ids_b64": "AQAAAA=="}, 0, id="b64_pre_tokenized"),
            pytest.param({"chat_template": "{{ messages }}"}, 0, id="request_template"),
        ],
    )
    def test_kimi_k3_excludes_the_generation_channel_opener(
        self, kimi, overrides, expected
    ) -> None:
        request = _chat_request(**overrides)
        assert kimi.prompt_tokens_excluded_from_usage(request) == expected

    def test_gpt_oss_declares_harmony_output(self) -> None:
        extension = serving_extensions.get_serving_extension("gpt_oss")
        assert extension.output_mode() is serving_extensions.OutputMode.HARMONY_TOKENS

    @pytest.fixture
    def deepseek_v41(self):
        return serving_extensions.get_serving_extension("deepseek_v41")

    @staticmethod
    def _namespaced_tool(namespace="multi_agent_v1", name="spawn_agent", description=None):
        from tensorrt_llm.serve.openai_protocol import (
            FunctionDefinition,
            NamespacedFunctionToolParam,
        )

        return NamespacedFunctionToolParam(
            function=FunctionDefinition(
                name=f"{namespace}.{name}",
                description=description or "Sub-agents.",
                parameters={"type": "object", "properties": {}},
            ),
            namespace=namespace,
            namespace_description="Sub-agents.",
            declared_name=name,
            declared_description=description,
        )

    def test_deepseek_v41_dumps_a_tool_without_a_namespace_unchanged(self, deepseek_v41) -> None:
        tool = _chat_request(tools=[_WEATHER_TOOL]).tools[0]

        assert deepseek_v41.serialize_tool(tool) == tool.model_dump()

    def test_other_models_get_a_namespaced_tool_as_a_plain_function(self) -> None:
        from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam

        tool = self._namespaced_tool()
        plain = ChatCompletionToolsParam(function=tool.function)

        assert tool.model_dump() == plain.model_dump()
        for model_type in (None, "kimi_k3", "nemotron_h", "qwen3"):
            extension = serving_extensions.get_serving_extension(model_type)
            assert extension.serialize_tool(tool) == extension.serialize_tool(plain)

    @pytest.mark.parametrize("description", [None, "Spawn one."])
    def test_deepseek_v41_dumps_a_namespaced_tool_in_the_checkpoint_form(
        self, deepseek_v41, description
    ) -> None:
        """The namespace beside the function, as the checkpoint's encoder reads it."""
        tool = self._namespaced_tool(description=description)

        assert deepseek_v41.serialize_tool(tool) == {
            "type": "function",
            "namespace": {"name": "multi_agent_v1", "description": "Sub-agents."},
            "function": {
                "name": "spawn_agent",
                "description": description,
                "parameters": {"type": "object", "properties": {}},
                "strict": None,
            },
        }

    @pytest.mark.parametrize("namespace, name", [("a::b", "tool"), ("ns", "a::tool")])
    def test_deepseek_v41_keeps_a_name_its_encoder_rejects_flattened(
        self, deepseek_v41, namespace, name
    ) -> None:
        tool = self._namespaced_tool(namespace=namespace, name=name)

        assert deepseek_v41.serialize_tool(tool) == tool.model_dump()
