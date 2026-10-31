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
"""CPU-only unit tests for the MuseSpark serving extensions.

Covers the request-schema acceptance of the extended reasoning-effort scale
(up to `xhigh`), the routing of a request-level `reasoning_effort`
into the chat-template kwargs for muse_spark deployments, whose chat
template reads the effort from its kwargs only, the Responses-API
reasoning routing, which must not inject template-only kwargs the active
template never reads, and the structured-output placement under the
`musespark` reasoning parser. No GPU or checkpoint required.
"""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm.inputs.chat_template_guard import validate_chat_template_kwargs
from tensorrt_llm.serve import responses_utils
from tensorrt_llm.serve.anthropic_adapter import convert_anthropic_request
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.harmony_adapter import maybe_transform_reasoning_effort
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ResponseFormat,
    ResponsesRequest,
    _response_format_to_guided_decoding_params,
)
from tensorrt_llm.serve.responses_utils import reasoning_chat_template_kwargs
from tensorrt_llm.serve.serving_extensions import (
    apply_model_chat_extensions,
    structured_output_format_for,
)

from .render_helpers import byte_level_tokenizer, responses_prompt

# The L0 CPU stage invokes registered files with `-m cpu_only`; without this
# marker every test is deselected and pytest exits with code 5.
pytestmark = pytest.mark.cpu_only

USER_MSG = {"role": "user", "content": "hello"}

# The serve-side effort contract, restated independently of the constant in
# tensorrt_llm/serve/extensions/musespark.py (importing the table under test
# could not catch a wrong table): API value -> level handed to the template.
# The floor-intent values (`none`, legacy `minimal`) map to `low` explicitly
# rather than falling through to a high template default. A contract change
# edits this table and the one in musespark.py, nothing else.
EFFORT_CONTRACT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
    "none": "low",
    "minimal": "low",
}
# The template's own levels (identity rows) and the protocol aliases.
MUSESPARK_EFFORTS = [k for k, v in EFFORT_CONTRACT.items() if k == v]
EFFORT_ALIASES = [(k, v) for k, v in EFFORT_CONTRACT.items() if k != v]
# Every effort the request schema accepts.
API_EFFORTS = list(EFFORT_CONTRACT)


def make_request(**kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("model", "musespark_open")
    kwargs.setdefault("messages", [USER_MSG])
    return ChatCompletionRequest(**kwargs)


class TestReasoningEffortSchema:
    """The request schema accepts every effort the templates support."""

    @pytest.mark.parametrize("effort", API_EFFORTS)
    def test_all_supported_efforts_accepted(self, effort: str) -> None:
        req = make_request(reasoning_effort=effort)
        assert req.reasoning_effort == effort

    def test_unknown_effort_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_request(reasoning_effort="xxhigh")


class TestMuseSparkEffortRouting:
    def apply(self, model_type: str = "muse_spark", **kwargs) -> ChatCompletionRequest:
        req = make_request(**kwargs)
        apply_model_chat_extensions(req, model_type)
        return req

    def kwargs_of(self, req: ChatCompletionRequest) -> dict:
        return req.chat_template_kwargs or {}

    @pytest.mark.parametrize("effort", MUSESPARK_EFFORTS)
    def test_top_level_effort_routed_to_template_kwargs(self, effort: str) -> None:
        req = self.apply(reasoning_effort=effort)
        assert self.kwargs_of(req)["reasoning_effort"] == effort

    def test_non_musespark_untouched(self) -> None:
        req = self.apply(model_type="llama", reasoning_effort="xhigh")
        assert req.chat_template_kwargs is None

    def test_kimi_model_untouched(self) -> None:
        # The registry dispatches kimi_k3 to its own extension, which derives
        # `thinking_effort`; the MuseSpark-style `reasoning_effort` kwarg must
        # not appear there.
        req = self.apply(model_type="kimi_k3", reasoning_effort="high")
        assert "reasoning_effort" not in (req.chat_template_kwargs or {})

    def test_default_effort_not_derived(self) -> None:
        # The field defaults to LOW via pydantic; a client that never sent
        # reasoning_effort must not have an effort injected.
        req = self.apply()
        assert req.chat_template_kwargs is None

    def test_explicit_null_effort_not_derived(self) -> None:
        req = self.apply(reasoning_effort=None)
        assert req.chat_template_kwargs is None

    @pytest.mark.parametrize(("effort", "expected"), EFFORT_ALIASES)
    def test_protocol_aliases_map_to_nearest_musespark_level(
        self, effort: str, expected: str
    ) -> None:
        # The template rejects anything outside its own levels, so the
        # protocol values it lacks map to the nearest level instead of
        # falling through to the template default.
        req = self.apply(reasoning_effort=effort)
        assert self.kwargs_of(req)["reasoning_effort"] == expected

    def test_client_chat_template_kwargs_win(self) -> None:
        req = self.apply(reasoning_effort="xhigh", chat_template_kwargs={"reasoning_effort": "low"})
        assert self.kwargs_of(req)["reasoning_effort"] == "low"

    def test_other_client_kwargs_preserved(self) -> None:
        req = self.apply(reasoning_effort="xhigh", chat_template_kwargs={"foo": "bar"})
        assert self.kwargs_of(req) == {"reasoning_effort": "xhigh", "foo": "bar"}

    # An adapter that derives the kwarg from an API field (the Anthropic
    # adapter with output_config.effort) lists it in
    # injected_chat_template_kwargs; those are normalized, client-written
    # kwargs are not.
    @pytest.mark.parametrize(
        ("injected", "expected"),
        EFFORT_ALIASES + [("HIGH", "high"), ("xhigh", "xhigh")],
    )
    def test_injected_kwarg_normalized(self, injected: str, expected: str) -> None:
        req = self.apply(
            chat_template_kwargs={"reasoning_effort": injected, "enable_thinking": True},
            injected_chat_template_kwargs=["enable_thinking", "reasoning_effort"],
        )
        assert self.kwargs_of(req) == {"reasoning_effort": expected, "enable_thinking": True}

    def test_injected_unknown_effort_dropped_to_template_default(self) -> None:
        req = self.apply(
            chat_template_kwargs={"reasoning_effort": "ultra", "enable_thinking": True},
            injected_chat_template_kwargs=["enable_thinking", "reasoning_effort"],
        )
        assert self.kwargs_of(req) == {"enable_thinking": True}

    def test_client_written_kwarg_not_normalized(self) -> None:
        # Not listed as injected, so it is the client's literal value and is
        # forwarded verbatim (the template then rejects it with a 400).
        req = self.apply(chat_template_kwargs={"reasoning_effort": "HIGH"})
        assert self.kwargs_of(req) == {"reasoning_effort": "HIGH"}

    def test_injected_kwarg_untouched_for_other_models(self) -> None:
        req = self.apply(
            model_type="llama",
            chat_template_kwargs={"reasoning_effort": "max"},
            injected_chat_template_kwargs=["reasoning_effort"],
        )
        assert self.kwargs_of(req) == {"reasoning_effort": "max"}


class TestAnthropicEffortReachesMuseSparkTemplate:
    """The Anthropic path end to end.

    output_config.effort is forwarded by the model-agnostic adapter as an
    injected kwarg, and the MuseSpark extension normalizes it on the resulting
    chat request, so Anthropic's own top tier `max` reaches the template as
    `xhigh`.
    """

    def convert(self, effort: str, model_type: str = "muse_spark") -> dict:
        request = AnthropicMessagesRequest(
            model="musespark_open",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            output_config={"effort": effort},
        )
        chat = convert_anthropic_request(request)
        assert "reasoning_effort" in (chat.injected_chat_template_kwargs or [])
        apply_model_chat_extensions(chat, model_type)
        return chat.chat_template_kwargs or {}

    @pytest.mark.parametrize(
        ("effort", "expected"), [("max", "xhigh"), ("high", "high"), ("Low", "low")]
    )
    def test_effort_normalized_for_musespark(self, effort: str, expected: str) -> None:
        assert self.convert(effort)["reasoning_effort"] == expected

    def test_unknown_effort_left_to_template_default(self) -> None:
        assert "reasoning_effort" not in self.convert("ultra")

    def test_other_models_get_the_verbatim_value(self) -> None:
        assert self.convert("max", model_type="llama")["reasoning_effort"] == "max"


# A template in the MuseSpark style: it reads the effort level from its
# kwargs but implements no thinking toggle.
TEMPLATE_EFFORT_ONLY = (
    "{%- if reasoning_effort %}Reasoning: {{ reasoning_effort }}.{%- endif %}"
    "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
)

# A template in the DeepSeek-V4 style: it reads both the thinking toggle and
# the effort level.
TEMPLATE_EFFORT_AND_THINKING = (
    "{%- if thinking %}<think>{%- endif %}"
    "{%- if reasoning_effort %}Reasoning: {{ reasoning_effort }}.{%- endif %}"
    "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
)

# A template that implements no reasoning controls at all.
TEMPLATE_PLAIN = "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"

# A template whose reads the analysis cannot see: they may sit in the include.
TEMPLATE_WITH_INCLUDE = (
    "{%- include 'reasoning.jinja' -%}"
    "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
)


def make_responses_request(**kwargs) -> ResponsesRequest:
    kwargs.setdefault("model", "musespark_open")
    kwargs.setdefault("input", "hello")
    return ResponsesRequest(**kwargs)


class TestResponsesReasoningKwargsInjection:
    """/v1/responses reasoning routing is template-aware.

    The routing maps `reasoning.effort` into chat_template_kwargs and drops
    injected keys the active template never reads, so what it constructs
    passes the render-site guard even without the injected-key exemption.
    The thinking-family keys (`thinking`, `enable_thinking`) are kept on
    both sides: the reasoning parsers consume them from chat_template_kwargs
    after generation, so the guard accepts them everywhere.
    """

    def test_effort_only_template_keeps_effort_and_parser_consumed_thinking(self):
        req = make_responses_request(reasoning={"effort": "high"})
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_ONLY)
        # `thinking` survives even though the template never reads it: the
        # reasoning parsers consume it post-generation, so the guard accepts
        # it everywhere and dropping it could change the parser's mode.
        assert kwargs == {"reasoning_effort": "high", "thinking": True}

    def test_template_reading_neither_key_drops_the_template_only_one(self):
        # The injection stays template-conditional for template-only keys:
        # against a template that reads neither key, `reasoning_effort` is
        # dropped (the guard would reject it as unreferenced), while the
        # parser-consumed `thinking` is kept.
        req = make_responses_request(reasoning={"effort": "high"})
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_PLAIN)
        assert kwargs == {"thinking": True}
        validate_chat_template_kwargs(TEMPLATE_PLAIN, kwargs)

    def test_template_the_analysis_cannot_see_into_gets_only_the_parser_consumed_key(self):
        # The include may or may not read `reasoning_effort`, so it is left out
        # and the template keeps its own default. `thinking` stays: the
        # reasoning parsers consume it whatever the template reads.
        req = make_responses_request(reasoning={"effort": "high"})
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_WITH_INCLUDE)
        assert kwargs == {"thinking": True}

    def test_effort_only_template_injection_passes_the_guard(self):
        # End to end: what the routing constructs for a reasoning-bearing
        # Responses request passes the same guard check the render site
        # applies (a raise here is the 400).
        req = make_responses_request(reasoning={"effort": "high"})
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_ONLY)
        validate_chat_template_kwargs(TEMPLATE_EFFORT_ONLY, kwargs)

    def test_thinking_template_keeps_both_keys(self):
        req = make_responses_request(reasoning={"effort": "high"})
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_AND_THINKING)
        assert kwargs == {"reasoning_effort": "high", "thinking": True}
        validate_chat_template_kwargs(TEMPLATE_EFFORT_AND_THINKING, kwargs)

    def test_no_template_source_keeps_full_injection(self):
        # The reasoning-parser call sites and the custom non-Jinja tokenizers
        # pass no template source; they must keep the full derivation.
        req = make_responses_request(reasoning={"effort": "high"})
        assert reasoning_chat_template_kwargs(req) == {
            "reasoning_effort": "high",
            "thinking": True,
        }

    def test_client_sent_kwargs_are_not_filtered(self):
        # Filtering covers only the serve's own injection. A kwarg the client
        # sent is forwarded untouched, and the guard still rejects it when
        # the template never reads it - that rejection is correct and stays.
        # `disable_reasoning` rather than `thinking` demonstrates it: the
        # thinking-family keys are parser-consumed and accepted everywhere.
        req = SimpleNamespace(
            chat_template_kwargs={"disable_reasoning": True},
            reasoning=SimpleNamespace(effort="high"),
        )
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_ONLY)
        assert kwargs == {
            "disable_reasoning": True,
            "reasoning_effort": "high",
            "thinking": True,
        }
        with pytest.raises(ValueError, match="disable_reasoning"):
            validate_chat_template_kwargs(TEMPLATE_EFFORT_ONLY, kwargs)

    def test_request_without_reasoning_injects_nothing(self):
        req = make_responses_request()
        assert reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_ONLY) == {}

    @pytest.mark.parametrize(
        ("effort", "expected"), [("none", EFFORT_CONTRACT["none"]), ("high", "high")]
    )
    def test_musespark_model_type_normalizes_injected_effort(self, effort: str, expected: str):
        # The derived effort goes through the model's normalizer: OpenAI's
        # `none` has no MuseSpark level and maps to the floor instead of
        # reaching the template, which would reject it with a 400.
        req = make_responses_request(reasoning={"effort": effort})
        kwargs = reasoning_chat_template_kwargs(
            req, template_source=TEMPLATE_EFFORT_ONLY, model_type="muse_spark"
        )
        assert kwargs == {"reasoning_effort": expected, "thinking": True}
        validate_chat_template_kwargs(TEMPLATE_EFFORT_ONLY, kwargs)

    def test_unknown_model_type_forwards_effort_verbatim(self):
        req = make_responses_request(reasoning={"effort": "minimal"})
        kwargs = reasoning_chat_template_kwargs(
            req, template_source=TEMPLATE_EFFORT_ONLY, model_type="llama"
        )
        assert kwargs == {"reasoning_effort": "minimal", "thinking": True}

    def test_none_turns_thinking_off_unless_the_model_maps_it(self):
        # Without a normalizer, `none` turns thinking off; the MuseSpark
        # template cannot turn reasoning off, so its normalizer maps `none`
        # to the lowest level instead.
        req = make_responses_request(reasoning={"effort": "none"})
        assert reasoning_chat_template_kwargs(
            req, template_source=TEMPLATE_EFFORT_ONLY, model_type="llama"
        ) == {"thinking": False}
        assert reasoning_chat_template_kwargs(
            req, template_source=TEMPLATE_EFFORT_ONLY, model_type="muse_spark"
        ) == {"reasoning_effort": EFFORT_CONTRACT["none"], "thinking": True}

    def test_client_effort_kwarg_wins_over_derived_and_is_not_normalized(self):
        req = SimpleNamespace(
            chat_template_kwargs={"reasoning_effort": "HIGH"},
            reasoning=SimpleNamespace(effort="none"),
        )
        kwargs = reasoning_chat_template_kwargs(
            req, template_source=TEMPLATE_EFFORT_ONLY, model_type="muse_spark"
        )
        assert kwargs == {"reasoning_effort": "HIGH", "thinking": True}


class TestResponsesRenderedEffort:
    """`_create_input_tokens` renders with the normalized effort, only where the template reads it."""

    @staticmethod
    async def render(template: str, monkeypatch):
        """The rendered prompt and the kwargs the render got, for `reasoning.effort="none"`."""
        captured = {}
        render_conversation = responses_utils.render_conversation

        async def spy(*args, **kwargs):
            captured.update(kwargs)
            return await render_conversation(*args, **kwargs)

        monkeypatch.setattr(responses_utils, "render_conversation", spy)
        request = make_responses_request(reasoning={"effort": "none"})
        prompt = await responses_prompt(
            request, byte_level_tokenizer(template), model_type="muse_spark"
        )
        return prompt, captured["chat_template_kwargs"]

    @pytest.mark.asyncio
    async def test_effort_reaching_the_template_is_normalized(self, monkeypatch):
        prompt, kwargs = await self.render(TEMPLATE_EFFORT_ONLY, monkeypatch)
        assert kwargs == {"reasoning_effort": EFFORT_CONTRACT["none"], "thinking": True}
        assert prompt == f"Reasoning: {EFFORT_CONTRACT['none']}.hello"

    @pytest.mark.asyncio
    async def test_template_that_reads_no_effort_is_not_given_one(self, monkeypatch):
        prompt, kwargs = await self.render(TEMPLATE_PLAIN, monkeypatch)
        assert kwargs == {"thinking": True}
        assert prompt == "hello"


class TestHarmonyTransformCoversExtendedScale:
    """No KeyError from `maybe_transform_reasoning_effort` on accepted efforts.

    It runs on harmony-only paths, where the extended-scale values map to the
    nearest harmony level.
    """

    @pytest.mark.parametrize("effort", API_EFFORTS)
    def test_no_keyerror_for_any_accepted_effort(self, effort: str) -> None:
        maybe_transform_reasoning_effort(effort)

    def test_extended_efforts_map_to_nearest_harmony_level(self) -> None:
        from openai_harmony import ReasoningEffort

        assert maybe_transform_reasoning_effort("minimal") == ReasoningEffort.LOW
        assert maybe_transform_reasoning_effort("xhigh") == ReasoningEffort.HIGH


# The markers of the forced reasoning message, restated rather than read from
# MuseSparkReasoningParser for the same reason as EFFORT_CONTRACT.
REASONING_BEGIN = " to=self<|message|>"
REASONING_END = "<|eom|><|start|>assistant to=user<|message|>"
CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
}


class TestStructuredOutputPlacement:
    """Response formats take the generic reasoning-parser placement.

    No hook is keyed by the `musespark` parser, so the constraint follows a
    forced private reasoning message and applies to the answer message, at
    every reasoning effort.
    """

    def test_no_parser_keyed_hook(self) -> None:
        assert structured_output_format_for("musespark") is None

    @pytest.mark.parametrize("chat_template_kwargs", [None, {"reasoning_effort": "low"}])
    @pytest.mark.parametrize(
        ("response_format", "expected_schema"),
        [
            (ResponseFormat(type="json_object"), {"type": "object"}),
            (
                ResponseFormat(
                    type="json_schema", json_schema={"name": "city", "schema": CITY_SCHEMA}
                ),
                CITY_SCHEMA,
            ),
        ],
        ids=["json_object", "json_schema"],
    )
    def test_reasoning_precedes_the_constrained_answer(
        self, response_format, expected_schema, chat_template_kwargs
    ) -> None:
        params = _response_format_to_guided_decoding_params(
            response_format,
            reasoning_parser="musespark",
            chat_template_kwargs=chat_template_kwargs,
        )
        # Key-wise: the structural-tag model adds defaulted fields on dump.
        fmt = json.loads(params.structural_tag)["format"]
        assert fmt["type"] == "sequence"
        reasoning, answer = fmt["elements"]
        assert reasoning["type"] == "tag"
        assert reasoning["begin"] == REASONING_BEGIN
        assert reasoning["content"]["type"] == "any_text"
        assert reasoning["end"] == REASONING_END
        assert answer["type"] == "json_schema"
        assert answer["json_schema"] == expected_schema
