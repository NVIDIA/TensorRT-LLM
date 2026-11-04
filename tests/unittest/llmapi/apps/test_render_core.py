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
"""The prompt-preparation core: rendering, tokenization, fingerprint, wire types."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import gc
import warnings
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.serve import serving_extensions
from tensorrt_llm.serve.render import (
    RENDERED_REQUEST_SCHEMA_VERSION,
    HarmonyConversionError,
    RenderedRequest,
    RenderResources,
    UnsupportedRenderError,
    compute_fingerprint,
    fingerprint_mismatch,
    fingerprint_summary,
    fingerprints_match,
    render_chat,
    render_chat_for_length,
    render_conversation,
    worker_resolves_thinking,
)
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

from .render_helpers import (
    KIMI_K3,
    LLAMA,
    QWEN3,
    REQUEST_TEMPLATE,
    SERVER_TEMPLATE,
    WEATHER_TOOL,
    chat_request,
    checkpoint_dir,
    copy_of,
    fallback_processor,
    harmony_resources,
    make_tokenizer,
    model_dir,
    real_processor,
    resources,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def tokenizer():
    return make_tokenizer()


def _expected_ids(tokenizer, request, *, add_special_tokens=False, tools=None, template=None):
    text = tokenizer.apply_chat_template(
        [dict(m) for m in request.messages],
        tokenize=False,
        add_generation_prompt=request.add_generation_prompt,
        tools=tools,
        chat_template=template,
    )
    return text, tokenizer.encode(text, add_special_tokens=add_special_tokens)


class TestRenderChat:
    def test_renders_and_tokenizes_like_the_template_and_the_tokenizer(self, tokenizer) -> None:
        request = chat_request()
        text, ids = _expected_ids(tokenizer, request)

        result = render_chat(request, resources(tokenizer))

        assert result.text == text
        assert result.token_ids == ids
        assert result.prefix_applied is False

    def test_tools_reach_the_template(self, tokenizer) -> None:
        request = chat_request(tools=[WEATHER_TOOL])
        text, ids = _expected_ids(tokenizer, request, tools=[t.model_dump() for t in request.tools])

        result = render_chat(request, resources(tokenizer))

        assert "get_weather" in result.text  # the real template prints the tool schema
        assert result.token_ids == ids

    def test_tokenize_false_leaves_only_the_text(self, tokenizer) -> None:
        result = render_chat(chat_request(), resources(tokenizer), tokenize=False)
        assert result.text is not None
        assert result.token_ids is None

    @pytest.mark.parametrize("add_special_tokens", [False, True])
    def test_add_special_tokens_follows_thechat_request(
        self, tokenizer, add_special_tokens
    ) -> None:
        request = chat_request(add_special_tokens=add_special_tokens)
        _text, ids = _expected_ids(tokenizer, request, add_special_tokens=add_special_tokens)

        assert render_chat(request, resources(tokenizer)).token_ids == ids

    def test_truncation_follows_thechat_request(self, tokenizer) -> None:
        request = chat_request(truncate_prompt_tokens=5)
        full = render_chat(chat_request(), resources(tokenizer)).token_ids

        # The loader a worker uses truncates on the left, keeping the end of the prompt.
        assert tokenizer.truncation_side == "left"
        assert render_chat(request, resources(tokenizer)).token_ids == full[-5:]

    def test_rendering_leaves_no_unawaited_coroutine(self, tokenizer) -> None:
        # The conversation parser hands back a coroutine for the media it would load. Nothing is
        # loaded here, and Python reports one left alone as "never awaited" on every render.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            render_chat(chat_request(), resources(tokenizer))
            gc.collect()
        assert not [w for w in caught if "never awaited" in str(w.message)]

    def test_a_router_without_a_model_config_still_parses_for_the_model_type(self) -> None:
        # kimi_k3 keeps a tool call's unparsable argument string verbatim, the way the worker
        # (which has the model config) parses it. A caller that has only the model type must
        # get the same prompt, not an error.
        tokenizer = make_tokenizer(name=KIMI_K3)
        request = chat_request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{not json"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "x"},
                {"role": "user", "content": "go"},
            ]
        )

        rendered = render_chat(
            request, RenderResources.from_tokenizer(tokenizer, model_type="kimi_k3")
        )

        assert "{not json" in rendered.text

    def test_a_pre_tokenized_request_renders_to_its_own_ids(self, tokenizer) -> None:
        result = render_chat(chat_request(prompt_token_ids=[4, 5, 6]), resources(tokenizer))
        assert result.token_ids == [4, 5, 6]
        assert result.text is None

    def test_a_request_template_wins_over_the_server_template(self, tokenizer) -> None:
        res = resources(tokenizer, default_chat_template=SERVER_TEMPLATE)

        assert render_chat(chat_request(), res, tokenize=False).text.startswith("SERVER:")
        request = chat_request(chat_template=REQUEST_TEMPLATE)
        res = resources(
            tokenizer, default_chat_template=SERVER_TEMPLATE, allow_request_chat_template=True
        )
        assert render_chat(request, res, tokenize=False).text.startswith("REQUEST:")

    def test_a_request_template_is_rejected_unless_the_server_allows_it(self, tokenizer) -> None:
        request = chat_request(chat_template=REQUEST_TEMPLATE)
        with pytest.raises(ValueError, match="chat_template"):
            render_chat(request, resources(tokenizer))

    def test_multimodal_input_is_rejected(self, tokenizer) -> None:
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
        with pytest.raises(UnsupportedRenderError, match="Multimodal"):
            render_chat(request, resources(tokenizer))

    def test_the_length_of_a_refused_request_renders_the_text_and_template_without_the_media(
        self, tokenizer
    ) -> None:
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            length_render = render_chat_for_length(request, resources(tokenizer))
            gc.collect()

        text_only = chat_request(messages=[{"role": "user", "content": "what is this"}])
        assert length_render.text == render_chat(text_only, resources(tokenizer)).text
        assert length_render.token_ids
        assert not [w for w in caught if "never awaited" in str(w.message)]

        # A model with its own input processor is refused; its length uses the text tokenizer.
        import tensorrt_llm._torch.models.modeling_mistral  # noqa: F401

        class OwnProcessor:
            tokenizer = None

        own = resources(tokenizer, model_type="mistral_common", input_processor=OwnProcessor())
        with pytest.raises(UnsupportedRenderError):
            render_chat(chat_request(), own)
        assert (
            render_chat_for_length(chat_request(), own).token_ids
            == render_chat(chat_request(), resources(tokenizer)).token_ids
        )

    def test_the_length_counts_the_tool_calls_of_a_request_with_media_and_leaves_the_request_intact(
        self, tokenizer
    ) -> None:
        image = {"type": "image_url", "image_url": {"url": "http://example.invalid/x.png"}}

        def request():
            return chat_request(
                messages=[
                    {"role": "user", "content": "weather in paris?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "paris"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                    {"role": "user", "content": [{"type": "text", "text": "and here"}, image]},
                ],
                tools=[WEATHER_TOOL],
            )

        same_request = request()
        first = render_chat_for_length(same_request, resources(tokenizer))
        second = render_chat_for_length(same_request, resources(tokenizer))

        assert "paris" in first.text and "get_weather" in first.text  # the tool call is rendered
        assert second.token_ids == first.token_ids  # and the request is not spent by the first
        # The caller's request keeps its tool calls (the router forwards it) and its other fields.
        assert all(
            isinstance(dict(message).get("tool_calls"), list)
            for message in same_request.messages
            if dict(message).get("tool_calls") is not None
        )
        assert same_request.prompt_token_ids is None
        assert "image_url" in str(list(same_request.messages))

    def test_the_length_keeps_a_tool_choice_that_changes_which_tools_are_rendered(
        self, tokenizer, harmony_vocab
    ) -> None:
        image = {"type": "image_url", "image_url": {"url": "http://example.invalid/x.png"}}
        text = {"type": "text", "text": "what is this"}

        def length(tool_choice):
            request = chat_request(
                messages=[{"role": "user", "content": [text, image]}],
                tools=[WEATHER_TOOL],
                tool_choice=tool_choice,
            )
            return render_chat_for_length(request, resources(tokenizer)).token_ids

        def exact(tool_choice):
            request = chat_request(
                messages=[{"role": "user", "content": "what is this"}],
                tools=[WEATHER_TOOL],
                tool_choice=tool_choice,
            )
            return render_chat(request, resources(tokenizer)).token_ids

        assert length("none") == exact("none")
        assert length("auto") == exact("auto")
        # A forcing choice is cleared: the length is that of the auto render.
        assert length("required") == exact("auto")

        # Harmony removes the tools for "none".
        def harmony_length(tool_choice):
            request = chat_request(tools=[WEATHER_TOOL], tool_choice=tool_choice)
            return render_chat_for_length(request, harmony_resources()).token_ids

        assert harmony_length("none") == render_chat(chat_request(), harmony_resources()).token_ids
        assert len(harmony_length("auto")) > len(harmony_length("none"))

    def test_the_length_of_a_kimi_k3_request_keeps_the_extensions_tool_serialization(self) -> None:
        # kimi_k3 dumps tools without null defaults.
        tokenizer = make_tokenizer(name=KIMI_K3)
        res = resources(
            tokenizer,
            model_type="kimi_k3",
            extension=serving_extensions.get_serving_extension("kimi_k3"),
        )
        image = {"type": "image_url", "image_url": {"url": "http://example.invalid/x.png"}}
        with_media = chat_request(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}, image]}],
            tools=[WEATHER_TOOL],
        )
        text_only = chat_request(messages=[{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL])

        assert render_chat_for_length(with_media, res).text == render_chat(text_only, res).text

        # kimi_k3 keeps a tool-call argument that is not JSON.
        call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "not json"},
                }
            ],
        }
        lenient = chat_request(messages=[with_media.messages[0], call], tools=[WEATHER_TOOL])
        lenient_text = chat_request(messages=[text_only.messages[0], call], tools=[WEATHER_TOOL])

        assert render_chat_for_length(lenient, res).text == render_chat(lenient_text, res).text

    def test_only_a_harmony_conversion_error_falls_back_to_the_template_for_a_length(
        self, tokenizer
    ) -> None:
        class FailingHarmony(ServingExtension):
            def __init__(self, error):
                self.error = error

            def render_prompt(self, request, res=None):
                raise self.error

        def resources_failing_with(error):
            return resources(
                tokenizer, use_harmony=True, extension=FailingHarmony(error), model_type="gpt_oss"
            )

        fallback = render_chat_for_length(
            chat_request(), resources_failing_with(HarmonyConversionError("no description"))
        )
        assert fallback.token_ids == render_chat(chat_request(), resources(tokenizer)).token_ids
        with pytest.raises(RuntimeError, match="boom"):
            render_chat_for_length(chat_request(), resources_failing_with(RuntimeError("boom")))

    def test_a_non_default_input_processor_is_rejected_not_approximated(self, tokenizer) -> None:
        class SpecialProcessor:
            tokenizer = None

        res = resources(tokenizer, input_processor=SpecialProcessor())
        with pytest.raises(UnsupportedRenderError, match="SpecialProcessor"):
            render_chat(chat_request(), res)

    def test_the_default_input_processor_gives_the_same_ids_as_plain_tokenization(
        self, tokenizer
    ) -> None:
        request = chat_request(add_special_tokens=True)
        plain = render_chat(request, resources(tokenizer)).token_ids
        processor = DefaultInputProcessor(None, None, tokenizer)

        via_processor = render_chat(request, resources(tokenizer, input_processor=processor))

        assert via_processor.token_ids == plain

    def test_the_router_cache_hook_replaces_the_plain_encode(self, tokenizer) -> None:
        calls = []

        def encode_rendered(text, tok):
            calls.append(text)
            return [7, 7]

        res = resources(tokenizer)
        assert render_chat(chat_request(), res, encode_rendered=encode_rendered).token_ids == [7, 7]
        # Special tokens or truncation mean a call the hook was not written for.
        render_chat(chat_request(add_special_tokens=True), res, encode_rendered=encode_rendered)
        assert len(calls) == 1


class TestExtensions:
    @pytest.fixture(params=["template", "grammar"])
    def extension_model(self, request):
        key = "render-core-test-model"
        mode = request.param

        @register_serving_extension(model_types=(key,))
        class TestExtension(ServingExtension):
            def apply_chat_extensions(self, request) -> None:
                request.chat_template_kwargs = {**(request.chat_template_kwargs or {}), "ran": 1}

            def allows_required_tool_choice(self) -> bool:
                return mode == "template"

            def required_tool_choice_mode(self) -> serving_extensions.RequiredToolChoiceMode | None:
                if mode == "grammar":
                    return "grammar"
                return super().required_tool_choice_mode()

            def serialize_tool(self, tool) -> dict:
                return {"type": "function", "function": {"name": "renamed_" + tool.function.name}}

        try:
            yield key
        finally:
            serving_extensions._BY_MODEL_TYPE.pop(key, None)

    def test_required_tool_choice_is_rejected_without_extension_support(self, tokenizer) -> None:
        request = chat_request(tools=[WEATHER_TOOL], tool_choice="required")
        # A ValueError for the routes (a 400) and an UnsupportedRenderError for callers
        # that only need a length (the router, count_tokens).
        with pytest.raises(UnsupportedRenderError, match="tool_choice='required' is not supported"):
            render_chat(request, resources(tokenizer))
        with pytest.raises(ValueError):
            render_chat(request, resources(tokenizer))

    def test_messages_the_request_model_rejects_need_the_raw_body(self, tokenizer) -> None:
        # Object-valued tool-call arguments: the chat route accepts them by falling
        # back to the raw JSON body. Without the body they cannot be recovered, and
        # rendering without the tool call would be silently wrong.
        body = {
            "model": "m",
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
            ],
        }
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        with pytest.raises(UnsupportedRenderError, match="raw request body"):
            render_chat(ChatCompletionRequest(**body), resources(tokenizer), tokenize=False)

        template = (
            "{% for m in messages %}{% if m.tool_calls %}[call:{{ m.tool_calls[0].function.name }}]"
            "{% endif %}{{ m.content }}|{% endfor %}"
        )
        rendered = render_chat(
            ChatCompletionRequest(**body),
            resources(tokenizer, default_chat_template=template),
            tokenize=False,
            raw_messages=body["messages"],
        )
        assert "[call:get_weather]" in rendered.text

    def test_a_model_that_builds_its_prompt_in_its_own_processor_is_not_rendered(
        self, tokenizer
    ) -> None:
        # Registers the passthrough content format for this model type.
        import tensorrt_llm._torch.models.modeling_mistral  # noqa: F401

        with pytest.raises(UnsupportedRenderError, match="own input processor"):
            render_chat(chat_request(), resources(tokenizer, model_type="mistral_common"))

    def test_a_named_tool_choice_is_refused_for_harmony_like_the_chat_route(
        self, harmony_vocab
    ) -> None:
        request = chat_request(
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        with pytest.raises(UnsupportedRenderError, match="named function is not yet supported"):
            render_chat(request, harmony_resources())
        # A length (a router routing a request the worker will refuse) gets Harmony's ids.
        assert render_chat_for_length(request, harmony_resources()).token_ids

    def test_the_extension_preprocesses_and_serializes_tools(
        self, tokenizer, extension_model
    ) -> None:
        request = chat_request(tools=[WEATHER_TOOL], tool_choice="required")
        # The real template does not reference the extension's `ran` kwarg; this one does.
        template = (
            "[ran:{{ ran }}]{% for t in tools %}[tool:{{ t.function.name }}]{% endfor %}"
            "{{ messages[-1].content }}"
        )
        res = resources(
            tokenizer,
            model_type=extension_model,
            extension=serving_extensions.get_serving_extension(extension_model),
            default_chat_template=template,
        )

        result = render_chat(request, res)

        assert request.chat_template_kwargs == {"ran": 1}
        assert result.text.startswith("[ran:1]")
        assert "[tool:renamed_get_weather]" in result.text

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"tools": [WEATHER_TOOL], "tool_choice": "auto"},
            {"reasoning_effort": "high"},
            {
                "messages": [
                    {"role": "developer", "content": "Be brief."},
                    {"role": "user", "content": "hello world"},
                ]
            },
        ],
        ids=["plain", "tools", "reasoning_effort", "developer_message"],
    )
    def test_harmony_renders_the_adapters_ids(self, harmony_vocab, overrides) -> None:
        from tensorrt_llm.serve.chat_tokenization import tokenize_harmony_chat_request

        result = render_chat(chat_request(**overrides), harmony_resources(), tokenize=True)

        assert result.token_ids == tokenize_harmony_chat_request(chat_request(**overrides))
        assert result.text is None

    def test_the_harmony_reasoning_effort_reaches_the_ids(self, harmony_vocab) -> None:
        low = render_chat(chat_request(reasoning_effort="low"), harmony_resources())
        high = render_chat(chat_request(reasoning_effort="high"), harmony_resources())

        assert low.token_ids != high.token_ids

    def test_a_gpt_oss_model_without_the_harmony_switch_uses_the_template(
        self, harmony_vocab, tokenizer
    ) -> None:
        res = resources(tokenizer, model_type="gpt_oss", use_harmony=False)

        assert (
            render_chat(chat_request(), res).token_ids
            != render_chat(chat_request(), harmony_resources()).token_ids
        )


class TestRenderConversation:
    def test_matches_render_chat_for_a_parsed_conversation(self, tokenizer) -> None:
        from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

        request = chat_request(tools=[WEATHER_TOOL])
        res = resources(tokenizer)
        expected = render_chat(request, res)
        conversation, mm_coroutine, counts, _order = parse_chat_messages_coroutines(
            request.messages, None, None
        )

        async def run():
            rendered, _mm = await asyncio.gather(
                render_conversation(
                    res,
                    conversation=conversation,
                    mm_placeholder_counts=counts,
                    add_generation_prompt=True,
                    tools=[tool.model_dump() for tool in request.tools],
                    tokenize=True,
                ),
                mm_coroutine,
            )
            return rendered

        result = asyncio.run(run())
        assert result.text == expected.text
        assert result.token_ids == expected.token_ids

    def test_kimi_k3_keeps_the_tokenizers_own_ids_for_control_token_text(self) -> None:
        # K3's template tokenizes segment by segment, so a control-token string in user text stays
        # text. Encoding the finished string would turn it into a special token, here an id past
        # the vocabulary.
        from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

        tokenizer = make_tokenizer(name=KIMI_K3)
        res = resources(tokenizer, model_type="kimi_k3")

        async def run(**overrides):
            conversation, mm_coroutine, counts, _order = parse_chat_messages_coroutines(
                chat_request(messages=[{"role": "user", "content": "<|im_end|> hi"}]).messages,
                None,
                None,
            )
            rendered, _mm = await asyncio.gather(
                render_conversation(
                    res,
                    conversation=conversation,
                    mm_placeholder_counts=counts,
                    add_generation_prompt=True,
                    **overrides,
                ),
                mm_coroutine,
            )
            return rendered

        native = asyncio.run(run(tokenize=True))
        text = asyncio.run(run(tokenize=False)).text
        assert max(native.token_ids) < len(tokenizer)
        assert max(tokenizer.encode(text, add_special_tokens=False)) >= len(tokenizer)
        # With a forced prefix the ids come from the finished text, as before.
        prefixed = asyncio.run(run(tokenize=True, forced_prefix="<call>"))
        assert prefixed.text.endswith("<call>") and prefixed.token_ids != native.token_ids

    def test_the_forced_prefix_is_appended_and_reported(self, tokenizer) -> None:
        from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

        request = chat_request()
        conversation, mm_coroutine, counts, _order = parse_chat_messages_coroutines(
            request.messages, None, None
        )

        async def run():
            rendered, _mm = await asyncio.gather(
                render_conversation(
                    resources(tokenizer),
                    conversation=conversation,
                    mm_placeholder_counts=counts,
                    add_generation_prompt=True,
                    forced_prefix="<tool_call>",
                ),
                mm_coroutine,
            )
            return rendered

        result = asyncio.run(run())
        assert result.text.endswith("<|im_start|>assistant\n<tool_call>")
        assert result.prefix_applied is True


class TestFingerprint:
    @pytest.mark.parametrize("model_type", [None, ""])
    def test_unknown_worker_model_type_matches_the_router(self, tokenizer, model_type) -> None:
        worker = self._worker_resources(
            tokenizer, model_config=SimpleNamespace(model_type=model_type)
        )
        router = RenderResources.from_tokenizer(tokenizer)
        assert fingerprints_match(worker.fingerprint(), router.fingerprint())

    @pytest.mark.parametrize("local,remote", [(None, "qwen3"), ("", "qwen3"), ("qwen2", "qwen3")])
    def test_known_model_types_remain_distinct(self, tokenizer, local, remote) -> None:
        base = resources(tokenizer)
        assert not fingerprints_match(
            dataclasses.replace(base, model_type=local).fingerprint(),
            dataclasses.replace(base, model_type=remote).fingerprint(),
        )

    def test_a_different_server_template_changes_the_digest(self, tokenizer) -> None:
        plain = compute_fingerprint(resources(tokenizer))
        with_template = compute_fingerprint(
            resources(tokenizer, default_chat_template=SERVER_TEMPLATE)
        )
        assert not fingerprints_match(plain, with_template)

    def test_a_different_tokenizer_changes_the_digest(self, tokenizer) -> None:
        other = make_tokenizer(name=LLAMA)
        assert not fingerprints_match(
            compute_fingerprint(resources(tokenizer)), compute_fingerprint(resources(other))
        )

    def test_the_harmony_digest_does_not_depend_on_the_model_type(self, harmony_vocab) -> None:
        # A router told to use Harmony that cannot resolve the model type still matches a
        # worker that knows it.
        assert fingerprints_match(
            compute_fingerprint(harmony_resources(model_type=None)),
            compute_fingerprint(harmony_resources()),
        )

    def test_a_mismatch_says_why(self, tokenizer) -> None:
        fingerprint = compute_fingerprint(resources(tokenizer))
        other = compute_fingerprint(resources(tokenizer, default_chat_template=SERVER_TEMPLATE))
        incomplete = {
            **fingerprint,
            "compared": {**fingerprint["compared"], "tokenizer": {"complete": False}},
        }

        assert fingerprint_mismatch(fingerprint, fingerprint) is None
        assert "no rendering fingerprint reported" in fingerprint_mismatch(fingerprint, None)
        assert "renders differently" in fingerprint_mismatch(fingerprint, other)
        assert "tokenizer cannot be fully identified" in fingerprint_mismatch(
            fingerprint, incomplete
        )
        assert "tokenizer cannot be fully identified" in fingerprint_mismatch(
            incomplete, fingerprint
        )

    @pytest.mark.parametrize(
        "parser,resolved,expected",
        [
            ("poolside_v1", {"poolside_v1": True}, True),
            ("poolside_v1", {"poolside_v1": False}, True),
            ("qwen3", {"poolside_v1": True}, False),
            (None, {"poolside_v1": True}, False),
            ("poolside_v1", {}, False),
        ],
    )
    def test_only_the_workers_own_reasoning_parser_decides_what_the_prompt_must_carry(
        self, tokenizer, parser, resolved, expected
    ) -> None:
        fingerprint = compute_fingerprint(resources(tokenizer, reasoning_parser=parser))

        assert worker_resolves_thinking(fingerprint, resolved) is expected
        # A published summary says the same.
        assert worker_resolves_thinking(fingerprint_summary(fingerprint), resolved) is expected

    def test_a_fingerprint_that_does_not_name_the_parser_is_judged_to_need_the_prompt(self) -> None:
        assert worker_resolves_thinking({"digest": "d"}, {"poolside_v1": True})
        assert not worker_resolves_thinking({"digest": "d"}, {})

    def test_the_extension_and_harmony_switch_change_the_digest(self, tokenizer) -> None:
        plain = compute_fingerprint(resources(tokenizer))
        harmony = compute_fingerprint(resources(tokenizer, use_harmony=True))
        extended = compute_fingerprint(resources(tokenizer, extension=SimpleExtension()))
        assert not fingerprints_match(plain, harmony)
        assert not fingerprints_match(plain, extended)

    def test_settings_that_do_not_change_the_ids_are_not_compared(self, tokenizer) -> None:
        plain = compute_fingerprint(resources(tokenizer))
        parsers = compute_fingerprint(
            resources(tokenizer, tool_parser="qwen3", reasoning_parser="qwen3")
        )
        assert fingerprints_match(plain, parsers)
        assert parsers["info"] == {"tool_parser": "qwen3", "reasoning_parser": "qwen3"}

    @pytest.mark.parametrize("local,remote", [(None, {"digest": "x"}), ({"digest": "x"}, None)])
    def test_a_missing_fingerprint_never_matches(self, local, remote) -> None:
        assert not fingerprints_match(local, remote)

    def _worker_resources(self, tokenizer, **overrides) -> RenderResources:
        """Resources built the way a worker builds them, not copied from the router's.

        A worker's ``processor`` is ``AutoProcessor.from_pretrained(...)``, which returns a
        tokenizer for a text-only checkpoint; ``processor=None`` would hide every difference
        that only exists in that real shape.
        """
        server = SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(model_type="render-test-model"),
            processor=fallback_processor(tokenizer),
            chat_template=None,
            allow_request_chat_template=False,
            generator=SimpleNamespace(
                args=SimpleNamespace(reasoning_parser=None),
                input_processor=DefaultInputProcessor(None, None, tokenizer),
            ),
        )
        for name, value in overrides.items():
            setattr(server, name, value)
        return RenderResources.from_server(server)

    def _router_resources(self, tokenizer) -> RenderResources:
        return RenderResources.from_tokenizer(tokenizer, model_type="render-test-model")

    def test_a_worker_whose_processor_is_the_tokenizer_matches_a_router_without_one(
        self, tokenizer
    ) -> None:
        # The real shape of every text-only worker: AutoProcessor falls back to the tokenizer,
        # so the "processor" carries the tokenizer's own template. The router has no processor.
        worker = self._worker_resources(tokenizer)
        assert worker.processor.chat_template == tokenizer.chat_template
        assert worker.input_processor is not None  # the engine's DefaultInputProcessor

        assert fingerprints_match(
            compute_fingerprint(self._router_resources(tokenizer)), compute_fingerprint(worker)
        )

    def test_a_processor_with_a_different_template_does_not_match_a_router_without_one(
        self, tokenizer
    ) -> None:
        # A vision model's processor can carry its own template; the router cannot know it.
        processor = copy.copy(real_processor())
        processor.chat_template = "PROCESSOR:{{ x }}"
        worker = self._worker_resources(tokenizer, processor=processor)

        assert not fingerprints_match(
            compute_fingerprint(self._router_resources(tokenizer)), compute_fingerprint(worker)
        )

    def test_a_server_template_makes_the_processors_template_irrelevant(self, tokenizer) -> None:
        # The server template wins when it is set, so a difference in a template that is never
        # used must not change the identity.
        def with_processor_template(template):
            processor = copy.copy(real_processor())
            processor.chat_template = template
            return resources(tokenizer, default_chat_template=SERVER_TEMPLATE, processor=processor)

        first, second = with_processor_template("ONE"), with_processor_template("TWO")

        assert fingerprints_match(compute_fingerprint(first), compute_fingerprint(second))

    @pytest.mark.parametrize("variant", ["tool_use", "alternate"])
    def test_every_named_template_is_part_of_the_identity(self, variant) -> None:
        # A request can select a named template, so a difference in any variant matters, also
        # one the default and the tool variant do not reach.
        plain = "{% for m in messages %}{{ m.content }}{% endfor %}"

        def tokenizer_with(**overrides):
            templates = {"default": plain, "tool_use": plain + "T", "alternate": plain + "A"}
            return make_tokenizer(chat_template={**templates, **overrides})

        same = compute_fingerprint(resources(tokenizer_with()))
        assert fingerprints_match(same, compute_fingerprint(resources(tokenizer_with())))
        changed = compute_fingerprint(resources(tokenizer_with(**{variant: plain + "X"})))
        assert not fingerprints_match(same, changed)

    def test_the_truncation_side_is_part_of_the_identity(self) -> None:
        # Which end `truncate_prompt_tokens` keeps is tokenizer state, not part of its encoding.
        left, right = make_tokenizer(), make_tokenizer()
        left.truncation_side, right.truncation_side = "left", "right"

        assert not fingerprints_match(
            compute_fingerprint(resources(left)), compute_fingerprint(resources(right))
        )

    def test_a_tokenizer_without_a_template_has_no_template_identity(self) -> None:
        # Kimi-K3 renders in Python and has no Jinja template; its own identity covers it.
        templates = compute_fingerprint(resources(make_tokenizer(name=KIMI_K3)))["compared"][
            "templates"
        ]

        assert templates == {"named": None, "without_tools": None, "with_tools": None}

    def test_a_worker_with_its_own_input_processor_does_not_match_a_router(self, tokenizer) -> None:
        class OwnInputProcessor:
            pass

        router = RenderResources.from_tokenizer(tokenizer, model_type="render-test-model")
        worker = dataclasses.replace(
            self._worker_resources(tokenizer), input_processor=OwnInputProcessor()
        )

        assert not fingerprints_match(compute_fingerprint(router), compute_fingerprint(worker))

    def test_equal_vocabularies_with_a_different_normalizer_do_not_match(self, tokenizer) -> None:
        from tokenizers import normalizers

        other = copy_of()
        other.backend_tokenizer.normalizer = normalizers.Lowercase()
        assert other.get_vocab() == tokenizer.get_vocab()
        assert type(other) is type(tokenizer)

        assert not fingerprints_match(
            compute_fingerprint(resources(tokenizer)), compute_fingerprint(resources(other))
        )

    def test_equal_vocabularies_with_a_different_pre_tokenizer_do_not_match(
        self, tokenizer
    ) -> None:
        from tokenizers import pre_tokenizers

        other = copy_of()
        other.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        assert other.get_vocab() == tokenizer.get_vocab()

        assert not fingerprints_match(
            compute_fingerprint(resources(tokenizer)), compute_fingerprint(resources(other))
        )

    def test_an_incomplete_encoding_identity_is_never_trusted(self) -> None:
        # Every tokenizer these tests can load is identifiable, so this is the rule itself:
        # identical digests are not enough when the encoding could not be identified.
        incomplete = {"digest": "d", "compared": {"tokenizer": {"complete": False}}}

        assert not fingerprints_match(incomplete, incomplete)

    def test_an_extension_revision_changes_the_digest(self, tokenizer) -> None:
        class NewRevision(SimpleExtension):
            render_version = 2

        old = compute_fingerprint(resources(tokenizer, extension=SimpleExtension()))
        new = compute_fingerprint(resources(tokenizer, extension=NewRevision()))

        assert not fingerprints_match(old, new)

    def test_the_same_checkpoint_at_two_locations_has_one_fingerprint(self, tmp_path) -> None:
        # The standalone renderer and the worker typically see the same files at different
        # mount points; the identity is the content, not where it was loaded from.
        from transformers import AutoTokenizer

        (tmp_path / "mount_a").mkdir()
        (tmp_path / "mount_b").mkdir()
        first = checkpoint_dir(tmp_path / "mount_a")
        second = checkpoint_dir(tmp_path / "mount_b")
        a, b = AutoTokenizer.from_pretrained(first), AutoTokenizer.from_pretrained(second)
        # The loading provenance really does differ.
        assert a.init_kwargs["name_or_path"] != b.init_kwargs["name_or_path"]

        assert fingerprints_match(
            compute_fingerprint(resources(a)), compute_fingerprint(resources(b))
        )

    def test_serving_requests_does_not_change_the_fingerprint(self) -> None:
        # Transformers enables truncation on the live backend while encoding a request and
        # does not restore it, and the backend's serialization includes it. A fingerprint
        # computed after the first truncated request must equal one computed before it.
        from transformers import AutoTokenizer

        served = AutoTokenizer.from_pretrained(model_dir(QWEN3))
        fresh = AutoTokenizer.from_pretrained(model_dir(QWEN3))
        before = compute_fingerprint(resources(served))

        served.encode("hello world this is a test", truncation=True, max_length=3)

        assert served.backend_tokenizer.truncation is not None  # the request state is set
        assert fingerprints_match(before, compute_fingerprint(resources(served)))
        assert fingerprints_match(
            compute_fingerprint(resources(fresh)), compute_fingerprint(resources(served))
        )
        # Canonicalizing never touches the live tokenizer other requests are using.
        assert served.backend_tokenizer.truncation["max_length"] == 3

    def test_harmony_has_an_explicit_encoding_identity(self, tokenizer) -> None:
        fingerprint = compute_fingerprint(resources(tokenizer, use_harmony=True))

        assert fingerprint["compared"]["harmony"]["encoding"] == "HARMONY_GPT_OSS"
        assert "version" in fingerprint["compared"]["harmony"]


class TestPreparationContext:
    PREFILLED_OPEN = (
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
        "{% if add_generation_prompt %}<assistant><think>{% endif %}"
    )
    PREFILLED_CLOSED = PREFILLED_OPEN.replace("<think>", "</think>")
    # MiniMax-M3 in adaptive mode: the generation prompt is the bare assistant header.
    M3_ADAPTIVE = (
        "{% for m in messages %}]~b]{{ m.role }}\n{{ m.content }}[e~[\n{% endfor %}"
        "{% if add_generation_prompt %}]~b]ai\n{% endif %}"
    )

    def _context(self, template, **overrides):
        res = resources(make_tokenizer(chat_template=template), **overrides)
        return render_chat(chat_request(), res).context

    def test_the_mode_is_recorded_for_every_parser_that_reads_it_off_the_prompt(self) -> None:
        # No reasoning parser is configured on this renderer: the context must not depend on
        # a setting that the compatibility check ignores.
        context = self._context(self.PREFILLED_OPEN)

        assert context["resolved_thinking"]["poolside_v1"] is True
        assert context["resolved_thinking"]["laguna"] is True  # an alias of the same parser

    def test_a_prompt_that_closes_thinking_is_recorded_as_off(self) -> None:
        assert self._context(self.PREFILLED_CLOSED)["resolved_thinking"]["poolside_v1"] is False

    def test_a_prompt_that_prefills_nothing_records_nothing(self, tokenizer) -> None:
        assert render_chat(chat_request(), resources(tokenizer)).context["resolved_thinking"] == {}

    def test_the_minimax_m3_adaptive_header_is_recorded_as_off(self) -> None:
        # The template prefills nothing after its header, so only that parser resolves the prompt.
        assert self._context(self.M3_ADAPTIVE)["resolved_thinking"] == {"minimax_m3": False}

    def test_no_generation_prompt_records_nothing(self) -> None:
        res = resources(make_tokenizer(chat_template=self.PREFILLED_OPEN))

        context = render_chat(chat_request(add_generation_prompt=False), res).context

        assert context["resolved_thinking"] == {}

    def test_only_a_context_without_decisions_is_trivial(self) -> None:
        from tensorrt_llm.serve.render import PreparedContext

        assert PreparedContext().is_trivial()
        assert not PreparedContext(prompt_tokens_excluded_from_usage=3).is_trivial()
        assert not PreparedContext(resolved_thinking={"poolside_v1": True}).is_trivial()


class SimpleExtension(ServingExtension):
    pass


class TestResourcesFromServer:
    def test_reads_the_server_at_call_time_with_defaults_for_missing_attributes(
        self, tokenizer
    ) -> None:
        server = SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(model_type="render-test-model"),
            processor=None,
            chat_template=SERVER_TEMPLATE,
            tool_parser="qwen3",
            allow_request_chat_template=True,
            generator=SimpleNamespace(args=SimpleNamespace(reasoning_parser="qwen3")),
        )

        res = RenderResources.from_server(server)

        assert res.model_type == "render-test-model"
        assert res.default_chat_template == SERVER_TEMPLATE
        assert res.allow_request_chat_template is True
        assert res.tool_parser == "qwen3"
        assert res.reasoning_parser == "qwen3"
        assert res.use_harmony is False
        assert res.input_processor is None


class TestRenderedRequest:
    def test_round_trips_and_carries_the_schema_version(self) -> None:
        body = RenderedRequest(token_ids=[1, 2, 3], request={"model": "m"})
        again = RenderedRequest.model_validate_json(body.model_dump_json())

        assert again == body
        assert again.schema_version == RENDERED_REQUEST_SCHEMA_VERSION

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RenderedRequest(token_ids=[1], surprise=True)

    def test_an_unset_request_id_is_generated(self) -> None:
        first, second = RenderedRequest(token_ids=[1]), RenderedRequest(token_ids=[1])

        assert first.request_id != second.request_id

    def test_a_per_response_fingerprint_is_no_longer_part_of_the_schema(self) -> None:
        with pytest.raises(ValidationError):
            RenderedRequest(fingerprint={"digest": "abc"}, token_ids=[1])
