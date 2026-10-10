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
import dataclasses
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.serve import serving_extensions
from tensorrt_llm.serve.render import (
    GENERATE_REQUEST_SCHEMA_VERSION,
    GenerateRequest,
    RenderResources,
    UnsupportedRenderError,
    compute_fingerprint,
    fingerprints_match,
    legacy_render_enabled,
    render_chat,
    render_conversation,
)
from tensorrt_llm.serve.serving_extensions import (
    OutputMode,
    ServingExtension,
    register_serving_extension,
)

from .render_helpers import (
    REQUEST_TEMPLATE,
    SERVER_TEMPLATE,
    WEATHER_TOOL,
    chat_request,
    make_tokenizer,
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
        assert result.tokens_trusted is True
        assert result.prefix_applied is False

    def test_tools_reach_the_template(self, tokenizer) -> None:
        request = chat_request(tools=[WEATHER_TOOL])
        text, ids = _expected_ids(tokenizer, request, tools=[WEATHER_TOOL])

        result = render_chat(request, resources(tokenizer))

        assert "[tools:get_weather]" in result.text
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

        assert render_chat(request, resources(tokenizer)).token_ids == full[:5]

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

    def test_named_tool_choice_marks_the_ids_untrusted(self, tokenizer) -> None:
        request = chat_request(
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        result = render_chat(request, resources(tokenizer))

        assert result.tokens_trusted is False
        assert result.untrusted_reason == "named_tool_choice"
        assert result.token_ids  # still usable for routing

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
    @pytest.fixture
    def extension_model(self):
        key = "render-core-test-model"

        @register_serving_extension(model_types=(key,))
        class TestExtension(ServingExtension):
            def apply_chat_extensions(self, request) -> None:
                request.chat_template_kwargs = {**(request.chat_template_kwargs or {}), "ran": 1}

            def allows_required_tool_choice(self) -> bool:
                return True

            def serialize_tool(self, tool) -> dict:
                return {"type": "function", "function": {"name": "renamed_" + tool.function.name}}

        try:
            yield key
        finally:
            serving_extensions._BY_MODEL_TYPE.pop(key, None)

    def test_required_tool_choice_is_rejected_without_extension_support(self, tokenizer) -> None:
        request = chat_request(tools=[WEATHER_TOOL], tool_choice="required")
        # A ValueError for the routes (a 400) and an UnsupportedRenderError for callers
        # that only need an estimate (the router, count_tokens).
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
        self, tokenizer
    ) -> None:
        class NeverRendered(ServingExtension):
            def render_prompt(self, request, res=None):
                raise AssertionError("the request must be refused before Harmony renders it")

        request = chat_request(
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        with pytest.raises(UnsupportedRenderError, match="named function is not yet supported"):
            render_chat(request, resources(tokenizer, use_harmony=True, extension=NeverRendered()))

    def test_the_extension_preprocesses_and_serializes_tools(
        self, tokenizer, extension_model
    ) -> None:
        request = chat_request(tools=[WEATHER_TOOL], tool_choice="required")
        res = resources(
            tokenizer,
            model_type=extension_model,
            extension=serving_extensions.get_serving_extension(extension_model),
        )

        result = render_chat(request, res)

        assert request.chat_template_kwargs == {"ran": 1}
        assert result.text.startswith("[ran:1]")
        assert "[tools:renamed_get_weather]" in result.text

    def test_harmony_models_render_through_the_extension(self, tokenizer) -> None:
        class HarmonyLike(ServingExtension):
            def render_prompt(self, request, res=None):
                return [9, 8, 7]

            def output_mode(self):
                return OutputMode.HARMONY_TOKENS

        res = resources(tokenizer, extension=HarmonyLike(), use_harmony=True)
        assert render_chat(chat_request(), res).token_ids == [9, 8, 7]
        # Without the Harmony switch the extension's renderer is not consulted.
        res = resources(tokenizer, extension=HarmonyLike(), use_harmony=False)
        assert render_chat(chat_request(), res).token_ids != [9, 8, 7]


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
                    tools=[WEATHER_TOOL],
                    tokenize=True,
                ),
                mm_coroutine,
            )
            return rendered

        result = asyncio.run(run())
        assert result.text == expected.text
        assert result.token_ids == expected.token_ids

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
        assert result.text.endswith("<assistant><tool_call>")
        assert result.prefix_applied is True


class TestFingerprint:
    def test_equal_resources_have_equal_fingerprints(self, tokenizer) -> None:
        first = compute_fingerprint(resources(tokenizer))
        second = compute_fingerprint(resources(tokenizer))
        assert fingerprints_match(first, second)
        assert first["digest"] == second["digest"]

    def test_a_different_server_template_changes_the_digest(self, tokenizer) -> None:
        plain = compute_fingerprint(resources(tokenizer))
        with_template = compute_fingerprint(
            resources(tokenizer, default_chat_template=SERVER_TEMPLATE)
        )
        assert not fingerprints_match(plain, with_template)

    def test_a_different_tokenizer_changes_the_digest(self, tokenizer) -> None:
        other = make_tokenizer(chat_template=SERVER_TEMPLATE)
        assert not fingerprints_match(
            compute_fingerprint(resources(tokenizer)), compute_fingerprint(resources(other))
        )

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

    def test_a_router_without_a_processor_does_not_match_a_server_with_one(self, tokenizer) -> None:
        server = resources(tokenizer, processor=SimpleNamespace(chat_template="PROCESSOR:{{ x }}"))
        router = RenderResources.from_tokenizer(tokenizer, model_type="render-test-model")
        assert not fingerprints_match(compute_fingerprint(router), compute_fingerprint(server))

    def _worker_resources(self, tokenizer, **overrides) -> RenderResources:
        """Resources built the way a worker builds them, not copied from the router's."""
        server = SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(model_type="render-test-model"),
            processor=None,
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

    def test_a_router_matches_an_independently_built_worker_for_plain_text(self, tokenizer) -> None:
        # The router has only a tokenizer; the worker has the engine's DefaultInputProcessor.
        # Both are the same plain-text tokenization.
        router = RenderResources.from_tokenizer(tokenizer, model_type="render-test-model")
        worker = self._worker_resources(tokenizer)

        assert fingerprints_match(compute_fingerprint(router), compute_fingerprint(worker))

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

        other = make_tokenizer()
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

        other = make_tokenizer()
        other.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        assert other.get_vocab() == tokenizer.get_vocab()

        assert not fingerprints_match(
            compute_fingerprint(resources(tokenizer)), compute_fingerprint(resources(other))
        )

    def test_a_tokenizer_whose_encoding_cannot_be_identified_is_never_trusted(self) -> None:
        class OpaqueTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}

        fingerprint = compute_fingerprint(resources(OpaqueTokenizer()))

        assert fingerprint["compared"]["tokenizer"]["complete"] is False
        # Even identical digests on both sides are not enough.
        assert not fingerprints_match(fingerprint, fingerprint)

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

        first, second = tmp_path / "mount_a" / "model", tmp_path / "mount_b" / "model"
        make_tokenizer().save_pretrained(first)
        make_tokenizer().save_pretrained(second)
        a, b = AutoTokenizer.from_pretrained(first), AutoTokenizer.from_pretrained(second)
        # The loading provenance really does differ.
        assert a.init_kwargs["name_or_path"] != b.init_kwargs["name_or_path"]

        assert fingerprints_match(
            compute_fingerprint(resources(a)), compute_fingerprint(resources(b))
        )

    def test_serving_requests_does_not_change_the_fingerprint(self, tmp_path) -> None:
        # Transformers enables truncation on the live backend while encoding a request and
        # does not restore it, and the backend's serialization includes it. A fingerprint
        # computed after the first truncated request must equal one computed before it.
        from transformers import AutoTokenizer

        make_tokenizer().save_pretrained(tmp_path)
        served = AutoTokenizer.from_pretrained(tmp_path)
        fresh = AutoTokenizer.from_pretrained(tmp_path)
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


class TestGenerateRequest:
    def test_round_trips_and_carries_the_schema_version(self) -> None:
        body = GenerateRequest(
            fingerprint={"digest": "abc"}, token_ids=[1, 2, 3], request={"model": "m"}
        )
        again = GenerateRequest.model_validate_json(body.model_dump_json())

        assert again == body
        assert again.schema_version == GENERATE_REQUEST_SCHEMA_VERSION
        assert again.tokens_trusted is True

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GenerateRequest(fingerprint={}, token_ids=[1], surprise=True)


class TestLegacySwitch:
    def test_off_by_default_and_enabled_by_the_environment(self, monkeypatch) -> None:
        monkeypatch.delenv("TRTLLM_RENDER_LEGACY", raising=False)
        assert legacy_render_enabled() is False
        monkeypatch.setenv("TRTLLM_RENDER_LEGACY", "1")
        assert legacy_render_enabled() is True
