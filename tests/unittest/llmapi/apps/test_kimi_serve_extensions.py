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
"""CPU-only unit tests for the Kimi K3 serving extensions.

Covers the Kimi Vendor Verifier (KVV) API contract implemented in
`tensorrt_llm/serve`: request validators, the Kimi extension-to-chat-template
mapping and its precedence rules, the immutable sampling-parameter policy, the
kimi_k3 `response_format` guided-decoding branch, and the (env-gated)
strict-tools structural-tag grammar builder. No GPU or checkpoint required.
"""

import json

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.extensions.kimi_k3 import (
    dynamic_tool_dicts,
    enforce_kimi_param_policy,
    validate_kimi_dynamic_tools,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    StreamOptions,
    _response_format_to_guided_decoding_params,
)
from tensorrt_llm.serve.serving_extensions import apply_model_chat_extensions
from tensorrt_llm.serve.tool_parser.kimi_k3_tool_parser import (
    KimiK3ToolParser,
    _escape_attr,
    _parse_attrs,
    _unescape_attr,
)

# The L0 CPU stage invokes registered files with `-m cpu_only`; without this
# marker every test is deselected and pytest exits with code 5.
pytestmark = pytest.mark.cpu_only

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather of a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

USER_MSG = {"role": "user", "content": "what is the weather in beijing?"}


def make_request(**kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("model", "hf-kimi-k3")
    kwargs.setdefault("messages", [USER_MSG])
    return ChatCompletionRequest(**kwargs)


def dynamic_system_msg(tools: list, **extra) -> dict:
    return {"role": "system", "content": "", "tools": tools, **extra}


class TestToolChoiceValidation:
    def test_required_with_tools_accepted(self) -> None:
        req = make_request(tools=[WEATHER_TOOL], tool_choice="required")
        assert req.tool_choice == "required"

    def test_required_without_tools_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"tools.*must be set"):
            make_request(tool_choice="required")

    def test_required_with_empty_tools_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"tools.*must be set"):
            make_request(tools=[], tool_choice="required")

    def test_required_with_dynamic_only_tools_accepted(self) -> None:
        req = make_request(
            messages=[dynamic_system_msg([WEATHER_TOOL]), USER_MSG], tool_choice="required"
        )
        assert req.tool_choice == "required"

    def test_named_with_dynamic_only_tools_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"tools.*must be set"):
            make_request(
                messages=[dynamic_system_msg([WEATHER_TOOL]), USER_MSG],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            )

    def test_auto_without_tools_accepted(self) -> None:
        req = make_request(tool_choice="auto")
        assert req.tool_choice == "auto"

    def test_tools_default_tool_choice_to_auto(self) -> None:
        assert make_request(tools=[WEATHER_TOOL]).tool_choice == "auto"

    def test_dynamic_only_tools_default_tool_choice_to_auto(self) -> None:
        req = make_request(messages=[dynamic_system_msg([WEATHER_TOOL]), USER_MSG])
        assert req.tool_choice == "auto"

    def test_no_tools_defaults_to_none(self) -> None:
        assert make_request().tool_choice == "none"


class TestMessageToolsCarrierValidation:
    """Carrier-role validation for message-level tools.

    The role restriction must reject at the raw-payload layer: union
    validation strips unknown keys from non-system messages, so a
    serving-side check would never see them.
    """

    def test_tools_in_user_message_rejected(self) -> None:
        with pytest.raises(ValidationError, match="only allowed on system"):
            make_request(
                messages=[
                    {
                        "role": "user",
                        "content": "hi",
                        "tools": [WEATHER_TOOL],
                    },
                    USER_MSG,
                ]
            )

    def test_tools_in_assistant_message_rejected(self) -> None:
        with pytest.raises(ValidationError, match="only allowed on system"):
            make_request(
                messages=[
                    USER_MSG,
                    {
                        "role": "assistant",
                        "content": "hello",
                        "tools": [WEATHER_TOOL],
                    },
                    USER_MSG,
                ]
            )

    def test_null_tools_key_ignored_everywhere(self) -> None:
        req = make_request(
            messages=[
                {"role": "user", "content": "hi", "tools": None},
                {"role": "system", "content": "be nice", "tools": None},
                USER_MSG,
            ]
        )
        assert dynamic_tool_dicts(req.messages) == []

    def test_system_tools_key_survives_validation(self) -> None:
        req = make_request(messages=[dynamic_system_msg([WEATHER_TOOL]), USER_MSG])
        assert dynamic_tool_dicts(req.messages) == [WEATHER_TOOL]

    def test_system_tools_with_content_survives_validation(self) -> None:
        # Content correctness is enforced by the kimi-gated serving layer,
        # but the key itself must not be silently stripped by the union.
        req = make_request(
            messages=[dynamic_system_msg([WEATHER_TOOL], content="not empty"), USER_MSG]
        )
        assert dynamic_tool_dicts(req.messages) == [WEATHER_TOOL]


class TestKimiDynamicToolsValidation:
    def check(self, messages: list, **kwargs) -> None:
        validate_kimi_dynamic_tools(make_request(messages=messages, **kwargs))

    def test_valid_dynamic_tool_passes(self) -> None:
        self.check([dynamic_system_msg([WEATHER_TOOL]), USER_MSG])

    def test_absent_content_passes(self) -> None:
        self.check([{"role": "system", "tools": [WEATHER_TOOL]}, USER_MSG])

    def test_strict_false_passes(self) -> None:
        tool = json.loads(json.dumps(WEATHER_TOOL))
        tool["function"]["strict"] = False
        self.check([dynamic_system_msg([tool]), USER_MSG])

    def test_nonempty_content_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty content"):
            self.check([dynamic_system_msg([WEATHER_TOOL], content="x"), USER_MSG])

    def test_tools_not_array_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be an array"):
            self.check(
                [
                    {
                        "role": "system",
                        "content": "",
                        "tools": {"type": "function"},
                    },
                    USER_MSG,
                ]
            )

    def test_tool_item_not_object_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            self.check([dynamic_system_msg([None]), USER_MSG])

    def test_missing_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported message-level"):
            self.check([dynamic_system_msg([{"function": {"name": "x"}}]), USER_MSG])

    def test_bogus_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported message-level"):
            self.check(
                [dynamic_system_msg([{"type": "bogus", "function": {"name": "x"}}]), USER_MSG]
            )

    def test_missing_function_rejected(self) -> None:
        with pytest.raises(ValueError, match="`function` object"):
            self.check([dynamic_system_msg([{"type": "function"}]), USER_MSG])

    @pytest.mark.parametrize(
        "name",
        [
            "1bad_name",
            "bad@name",
            "",
            "a" * 257,
            "get_weather\n",
        ],
    )
    def test_invalid_names_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="Invalid message-level"):
            self.check(
                [dynamic_system_msg([{"type": "function", "function": {"name": name}}]), USER_MSG]
            )

    @pytest.mark.parametrize("name", ["a" * 256, "Get_weather-2", "_x"])
    def test_valid_names_accepted(self, name: str) -> None:
        self.check(
            [dynamic_system_msg([{"type": "function", "function": {"name": name}}]), USER_MSG]
        )

    def test_duplicate_within_message_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate tool name"):
            self.check([dynamic_system_msg([WEATHER_TOOL, WEATHER_TOOL]), USER_MSG])

    def test_duplicate_across_messages_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate tool name"):
            self.check(
                [
                    dynamic_system_msg([WEATHER_TOOL]),
                    dynamic_system_msg([WEATHER_TOOL]),
                    USER_MSG,
                ]
            )

    def test_duplicate_against_request_tools_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate tool name"):
            self.check(
                [dynamic_system_msg([WEATHER_TOOL]), USER_MSG],
                tools=[WEATHER_TOOL],
                tool_choice="auto",
            )


class TestKimiExtensionMapping:
    def apply(self, model_type: str = "kimi_k3", **kwargs) -> ChatCompletionRequest:
        req = make_request(**kwargs)
        apply_model_chat_extensions(req, model_type)
        return req

    def kwargs_of(self, req: ChatCompletionRequest) -> dict:
        return req.chat_template_kwargs or {}

    def test_non_kimi_untouched(self) -> None:
        req = self.apply(model_type="llama", thinking={"type": "disabled"}, top_p=1.0)
        assert req.chat_template_kwargs is None
        assert req.top_p == 1.0

    def test_thinking_effort_wins_over_reasoning_effort(self) -> None:
        req = self.apply(
            thinking={"type": "enabled", "keep": "all", "effort": "low"}, reasoning_effort="max"
        )
        assert self.kwargs_of(req)["thinking"] is True
        assert self.kwargs_of(req)["thinking_effort"] == "low"

    @pytest.mark.parametrize("effort", ["low", "high", "max"])
    def test_reasoning_effort_applies_when_thinking_effort_absent(self, effort: str) -> None:
        req = self.apply(thinking={"type": "enabled", "keep": "all"}, reasoning_effort=effort)
        assert self.kwargs_of(req)["thinking_effort"] == effort

    def test_reasoning_effort_none_disables_thinking_when_alone(self) -> None:
        req = self.apply(reasoning_effort="none")
        assert self.kwargs_of(req)["thinking"] is False
        assert "thinking_effort" not in self.kwargs_of(req)

    def test_reasoning_effort_none_does_not_override_explicit_thinking(self) -> None:
        req = self.apply(thinking={"type": "enabled", "keep": "all"}, reasoning_effort="none")
        assert self.kwargs_of(req)["thinking"] is True
        assert "thinking_effort" not in self.kwargs_of(req)

    def test_no_effort_derived_for_explicitly_disabled_thinking(self) -> None:
        req = self.apply(thinking={"type": "disabled"}, reasoning_effort="high")
        assert self.kwargs_of(req)["thinking"] is False
        assert "thinking_effort" not in self.kwargs_of(req)

    def test_medium_effort_has_no_k3_equivalent(self) -> None:
        req = self.apply(reasoning_effort="medium")
        assert "thinking_effort" not in self.kwargs_of(req)
        assert "thinking" not in self.kwargs_of(req)

    def test_k3_rejects_medium_thinking_effort(self) -> None:
        with pytest.raises(ValueError, match="thinking.effort must be one of low, high, max"):
            self.apply(thinking={"type": "enabled", "effort": "medium"})

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
    def test_k31_thinking_efforts(self, effort: str) -> None:
        # Kimi K3.1's chat template adds the medium effort.
        req = self.apply(model_type="kimi_k31", thinking={"type": "enabled", "effort": effort})
        assert self.kwargs_of(req)["thinking_effort"] == effort
        req = self.apply(model_type="kimi_k31", reasoning_effort=effort)
        assert self.kwargs_of(req)["thinking_effort"] == effort

    def test_default_reasoning_effort_not_mapped(self) -> None:
        req = self.apply()
        assert req.chat_template_kwargs is None

    def test_client_chat_template_kwargs_win(self) -> None:
        req = self.apply(chat_template_kwargs={"thinking": True}, reasoning_effort="none")
        assert self.kwargs_of(req)["thinking"] is True

    def test_stream_options_defaulted_for_streaming(self) -> None:
        req = self.apply(stream=True)
        assert isinstance(req.stream_options, StreamOptions)
        assert req.stream_options.include_usage is True

    def test_stream_options_untouched_for_non_streaming(self) -> None:
        assert self.apply().stream_options is None

    def test_tool_choice_required_derived_with_tools(self) -> None:
        req = self.apply(tools=[WEATHER_TOOL], tool_choice="required")
        assert self.kwargs_of(req)["tool_choice"] == "required"

    def test_tool_choice_none_derived_with_tools(self) -> None:
        req = self.apply(tools=[WEATHER_TOOL], tool_choice="none")
        assert self.kwargs_of(req)["tool_choice"] == "none"

    def test_tool_choice_required_derived_with_dynamic_only_tools(self) -> None:
        req = self.apply(
            messages=[dynamic_system_msg([WEATHER_TOOL]), USER_MSG], tool_choice="required"
        )
        assert self.kwargs_of(req)["tool_choice"] == "required"

    def test_tool_choice_auto_not_derived(self) -> None:
        req = self.apply(tools=[WEATHER_TOOL], tool_choice="auto")
        assert "tool_choice" not in self.kwargs_of(req)

    def test_response_format_json_object_derived(self) -> None:
        req = self.apply(response_format={"type": "json_object"})
        assert self.kwargs_of(req)["response_format"] == "json_object"

    def test_response_format_json_schema_derived(self) -> None:
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}
        req = self.apply(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "schema": schema,
                    "strict": True,
                },
            }
        )
        assert self.kwargs_of(req)["response_format"] == "json_schema"
        assert self.kwargs_of(req)["response_schema"] == schema

    @pytest.mark.parametrize(
        "wrapper, msg",
        [
            ({"schema": {"type": "object"}}, "non-empty"),
            ({"name": "", "schema": {"type": "object"}}, "non-empty"),
            ({"name": "weather"}, "`schema` object"),
            (
                {"name": "weather", "schema": {"type": "object"}, "strict": "yes"},
                "must be a boolean",
            ),
        ],
    )
    def test_response_format_json_schema_wrapper_validation(self, wrapper: dict, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            self.apply(response_format={"type": "json_schema", "json_schema": wrapper})


class TestKimiParamPolicy:
    @pytest.fixture(autouse=True)
    def _enable_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The policy is opt-in (off by default); these tests exercise its
        # enabled semantics as a KVV certification run would.
        monkeypatch.setenv("TRTLLM_KIMI_PARAM_POLICY", "1")

    def enforce(self, **kwargs) -> ChatCompletionRequest:
        req = make_request(**kwargs)
        enforce_kimi_param_policy(req)
        return req

    def test_policy_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRTLLM_KIMI_PARAM_POLICY", raising=False)
        req = self.enforce(top_p=0.8, temperature=2.0, n=2)
        # Existing deployments keep today's behavior: no rejection, no
        # coercion, unless a certification run opts in with =1.
        assert req.top_p == 0.8
        assert req.temperature == 2.0
        assert req.n == 2

    def test_top_p_none_coerced(self) -> None:
        assert self.enforce().top_p == 0.95

    def test_top_p_one_coerced(self) -> None:
        assert self.enforce(top_p=1.0).top_p == 0.95

    def test_top_p_pinned_value_accepted(self) -> None:
        assert self.enforce(top_p=0.95).top_p == 0.95

    def test_top_p_other_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_p is fixed"):
            self.enforce(top_p=0.8)

    @pytest.mark.parametrize("temperature", [0.0, 0.6, 1.0])
    def test_temperature_in_range_accepted(self, temperature: float) -> None:
        self.enforce(temperature=temperature)

    @pytest.mark.parametrize("temperature", [-0.1, 1.1, 2.0])
    def test_temperature_out_of_range_rejected(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            self.enforce(temperature=temperature)

    def test_penalties_rejected(self) -> None:
        with pytest.raises(ValueError, match="presence_penalty"):
            self.enforce(presence_penalty=0.5)
        with pytest.raises(ValueError, match="frequency_penalty"):
            self.enforce(frequency_penalty=0.5)

    def test_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="n is fixed"):
            self.enforce(n=2)

    def test_policy_env_off_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRTLLM_KIMI_PARAM_POLICY", "0")
        req = self.enforce(top_p=0.8, temperature=2.0, n=2)
        # Fully unconstrained: no rejection and no coercion.
        assert req.top_p == 0.8
        assert req.temperature == 2.0
        assert req.n == 2


class TestKimiResponseFormatGuidedDecoding:
    def test_thinking_mode_builds_triggered_tags_on_response_channel(self) -> None:
        from tensorrt_llm.serve.openai_protocol import ResponseFormat

        params = _response_format_to_guided_decoding_params(
            ResponseFormat(type="json_object"),
            reasoning_parser="kimi_k3",
            chat_template_kwargs={"thinking": True},
        )
        stag = json.loads(params.structural_tag)
        fmt = stag["format"]
        assert fmt["type"] == "triggered_tags"
        assert fmt["triggers"] == ["<|open|>response<|sep|>"]
        assert fmt["tags"][0]["begin"] == "<|open|>response<|sep|>"
        assert fmt["tags"][0]["end"] == "<|close|>response<|sep|>"
        assert fmt["stop_after_first"] is True

    def test_non_thinking_mode_returns_raw_grammar(self) -> None:
        from tensorrt_llm.serve.openai_protocol import ResponseFormat

        params = _response_format_to_guided_decoding_params(
            ResponseFormat(type="json_object"),
            reasoning_parser="kimi_k3",
            chat_template_kwargs={"thinking": False},
        )
        assert params.structural_tag is None
        assert params.json_object is True


class TestKimiK3StrictGrammar:
    def build(self, monkeypatch: pytest.MonkeyPatch, tools: list, gate: str = "1") -> dict | None:
        monkeypatch.setenv("TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR", gate)
        parser = KimiK3ToolParser()
        return parser.build_strict_structural_tag_format(
            [ChatCompletionToolsParam.model_validate(t) for t in tools]
        )

    def test_env_gate_off_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strict = json.loads(json.dumps(WEATHER_TOOL))
        strict["function"]["strict"] = True
        assert self.build(monkeypatch, [strict], gate="0") is None

    def test_empty_tools_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.build(monkeypatch, []) is None

    def test_format_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strict = json.loads(json.dumps(WEATHER_TOOL))
        strict["function"]["strict"] = True
        loose = {
            "type": "function",
            "function": {"name": "search", "parameters": {"type": "object"}},
        }
        fmt = self.build(monkeypatch, [strict, loose])
        assert fmt["type"] == "triggered_tags"
        assert fmt["triggers"] == ["<|open|>tools<|sep|>"]
        # The deadlock traps: these must stay False or the grammar forbids
        # the think/response text before the section and the message close
        # after it.
        assert fmt["at_least_one"] is False
        assert fmt["stop_after_first"] is False
        section = fmt["tags"][0]
        assert section["begin"] == "<|open|>tools<|sep|>"
        assert section["end"] == "<|close|>tools<|sep|>"
        calls = section["content"]
        assert calls["type"] == "tags_with_separator"
        assert calls["separator"] == ""
        assert calls["at_least_one"] is True
        strict_tag, loose_tag = calls["tags"]
        assert strict_tag["begin"] == '<|open|>call tool="get_weather"'
        elements = strict_tag["content"]["elements"]
        assert elements[0]["type"] == "regex"
        assert elements[1] == {
            "type": "const_string",
            "value": '<|sep|><|open|>json type="object"<|sep|>',
        }
        assert elements[2]["type"] == "json_schema"
        assert elements[2]["json_schema"] == WEATHER_TOOL["function"]["parameters"]
        assert strict_tag["end"] == "<|close|>json<|sep|><|close|>call<|sep|>"
        assert loose_tag["content"] == {"type": "any_text"}
        assert loose_tag["end"] == "<|close|>call<|sep|>"

    def test_tool_name_attribute_escaping_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        name = 'we"ird&name'
        tool = {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        fmt = self.build(monkeypatch, [tool])
        begin = fmt["tags"][0]["content"]["tags"][0]["begin"]
        escaped = _escape_attr(name)
        # Pin the exact K3 dialect: only '&' and '"' have escaped forms.
        assert escaped == "we&quot;ird&amp;name"
        assert _escape_attr("a<b>c") == "a<b>c"
        assert escaped in begin
        assert _unescape_attr(escaped) == name
        assert _parse_attrs(f'tool="{escaped}" index="1"') == {
            "tool": name,
            "index": "1",
        }

    def test_angle_bracket_tool_name_skips_grammar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # '<' has no escaped form in the K3 wire format and would break the
        # parser's attribute regex; the builder must fall back rather than
        # emit a grammar the parser cannot read back.
        tool = {
            "type": "function",
            "function": {
                "name": "bad<name",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
        assert self.build(monkeypatch, [tool]) is None

    def test_grammar_compiles_with_xgrammar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        xgrammar = pytest.importorskip("xgrammar")
        strict = json.loads(json.dumps(WEATHER_TOOL))
        strict["function"]["strict"] = True
        fmt = self.build(monkeypatch, [strict])
        xgrammar.Grammar.from_structural_tag(json.dumps({"type": "structural_tag", "format": fmt}))


class TestKimiK3RequiredToolGrammar:
    """tool_choice="required": the grammar admits only messages that call a tool."""

    CALL = (
        '<|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Beijing<|close|>argument<|sep|>'
        "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
    )

    @staticmethod
    def accepts(tools: list, text: str, thinking: bool) -> bool:
        import xgrammar
        from xgrammar.testing import _is_grammar_accept_string

        tag = KimiK3ToolParser().build_required_structural_tag(
            [ChatCompletionToolsParam.model_validate(t) for t in tools], thinking=thinking
        )
        return _is_grammar_accept_string(xgrammar.Grammar.from_structural_tag(tag), text)

    def test_thinking_message_with_a_call(self) -> None:
        text = "Look it up.<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
        assert self.accepts([WEATHER_TOOL], text + self.CALL, thinking=True)

    def test_non_thinking_message_starts_in_the_response(self) -> None:
        assert self.accepts([WEATHER_TOOL], "On it.<|close|>response<|sep|>" + self.CALL, False)
        # The think channel is closed in the prompt already.
        text = "x<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>" + self.CALL
        assert not self.accepts([WEATHER_TOOL], text, thinking=False)

    def test_a_tool_call_is_mandatory(self) -> None:
        answer = "Sunny.<|close|>response<|sep|><|close|>message<|sep|>"
        assert not self.accepts([WEATHER_TOOL], answer, thinking=False)

    def test_only_declared_tools_and_arguments(self) -> None:
        prefix = "<|close|>response<|sep|>"
        other_tool = self.CALL.replace("get_weather", "get_time")
        assert not self.accepts([WEATHER_TOOL], prefix + other_tool, thinking=False)
        # city is required, so the call needs an argument.
        empty = (
            '<|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
        )
        assert not self.accepts([WEATHER_TOOL], prefix + empty, thinking=False)

    def test_non_string_arguments_follow_their_schema(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "forecast",
                "parameters": {
                    "type": "object",
                    "properties": {"days": {"type": "integer", "minimum": 1}},
                },
            },
        }

        def call(value: str) -> str:
            return (
                '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="forecast" '
                'index="1"<|sep|><|open|>argument key="days" type="number"<|sep|>'
                f"{value}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
                "<|close|>message<|sep|>"
            )

        assert self.accepts([tool], call("3"), thinking=False)
        assert not self.accepts([tool], call("three"), thinking=False)

    @staticmethod
    def one_call(tool: str, key: str, xtml_type: str, value: str) -> str:
        return (
            f'<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="{tool}" index="1"<|sep|>'
            f'<|open|>argument key="{key}" type="{xtml_type}"<|sep|>{value}<|close|>argument<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
        )

    def test_referenced_parameter_follows_its_definition(self) -> None:
        # KVV tool_call_json_schema TestReferences:16 as the verifier sends it:
        # the parameter is a $ref into $defs, and the definition refers to itself.
        case = {
            "type": "object",
            "required": ["node"],
            "properties": {
                "node": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "required": ["next"],
                            "properties": {
                                "next": {"$ref": "#/$defs/case"},
                                "value": {"type": "string"},
                            },
                        },
                    ]
                }
            },
        }
        tool = {
            "type": "function",
            "function": {
                "name": "walle",
                "parameters": {
                    "type": "object",
                    "required": ["value"],
                    "additionalProperties": False,
                    "properties": {"value": {"$ref": "#/$defs/case"}},
                    "$defs": {"case": case},
                },
            },
        }
        for value in ('{"node": null}', '{"node": {"next": {"node": null}}}'):
            assert self.accepts([tool], self.one_call("walle", "value", "object", value), False)
        for value in ('{"value": null}', '{"next": null}', '{"node": {"next": {"value": "x"}}}'):
            assert not self.accepts([tool], self.one_call("walle", "value", "object", value), False)
        # Written as a raw string, the JSON text would reach the tool as a string.
        raw = self.one_call("walle", "value", "string", '{"node": null}')
        assert not self.accepts([tool], raw, thinking=False)

    def test_root_reference_is_the_parameters_document(self) -> None:
        # "#" points at the whole parameters object (which requires "node"),
        # not at the property it appears in.
        tool = {
            "type": "function",
            "function": {
                "name": "walle",
                "parameters": {
                    "type": "object",
                    "required": ["node"],
                    "properties": {
                        "node": {
                            "anyOf": [
                                {"type": "null"},
                                {
                                    "type": "object",
                                    "required": ["next"],
                                    "properties": {"next": {"$ref": "#"}},
                                },
                            ]
                        }
                    },
                },
            },
        }
        assert self.accepts([tool], self.one_call("walle", "node", "null", "null"), False)
        nested = self.one_call("walle", "node", "object", '{"next": {"node": null}}')
        assert self.accepts([tool], nested, thinking=False)
        no_node = self.one_call("walle", "node", "object", '{"next": {"next": null}}')
        assert not self.accepts([tool], no_node, thinking=False)
        # Each tag's value has the type its attribute names.
        for xtml_type, value in (("null", '{"next": {"node": null}}'), ("object", "null")):
            mismatch = self.one_call("walle", "node", xtml_type, value)
            assert not self.accepts([tool], mismatch, thinking=False)

    def test_union_parameters_get_a_tag_per_type(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "pick",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {
                            "anyOf": [
                                {"type": "integer", "minimum": 1},
                                {"type": "string", "enum": ["auto"]},
                            ]
                        },
                        "m": {
                            "oneOf": [
                                {"type": "null"},
                                {"type": "array", "items": {"type": "integer"}},
                            ]
                        },
                    },
                },
            },
        }
        for key, xtml_type, value in (
            ("n", "number", "3"),
            ("n", "string", "auto"),
            ("m", "null", "null"),
            ("m", "array", "[1, 2]"),
        ):
            assert self.accepts([tool], self.one_call("pick", key, xtml_type, value), False)
        for key, xtml_type, value in (
            ("n", "string", "manual"),
            ("n", "boolean", "true"),
            ("m", "array", '["a"]'),
            ("m", "object", "{}"),
            ("m", "null", "[1]"),
            ("m", "array", "null"),
        ):
            assert not self.accepts([tool], self.one_call("pick", key, xtml_type, value), False)

    def test_unconstrained_parameters_stay_permissive(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"description": "anything"},
                        "y": {"$ref": "https://example.com/schema.json"},
                    },
                },
            },
        }
        assert self.accepts([tool], self.one_call("echo", "x", "string", "hi"), False)
        assert self.accepts([tool], self.one_call("echo", "x", "object", '{"a": 1}'), False)
        # A reference outside the document cannot be resolved, so its value is free.
        assert self.accepts([tool], self.one_call("echo", "y", "custom", "raw"), False)

    def test_tool_names_the_wire_format_cannot_carry(self) -> None:
        tool = json.loads(json.dumps(WEATHER_TOOL))
        tool["function"]["name"] = "a<b"
        parser = KimiK3ToolParser()
        assert (
            parser.build_required_structural_tag(
                [ChatCompletionToolsParam.model_validate(tool)], thinking=True
            )
            is None
        )


@pytest.mark.parametrize(
    "tail, expected",
    [
        ("<|open|>user<|sep|>hi<|close|>user<|sep|><|open|>think<|sep|>", True),
        ("<|open|>user<|sep|>hi<|close|>user<|sep|><|open|>response<|sep|>", False),
        ("<|open|>user<|sep|>hi<|close|>user<|sep|>", None),
    ],
)
def test_kimi_k3_thinking_resolves_from_the_prompt(tail: str, expected) -> None:
    from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

    assert ReasoningParserFactory.resolve_prefilled_thinking("kimi_k3", tail) is expected


class TestRequiredToolGrammarDisaggregated:
    """A disaggregated request builds the tool_choice="required" grammar on both workers."""

    @staticmethod
    def validate(**kwargs) -> None:
        from tensorrt_llm.sampling_params import SamplingParams
        from tensorrt_llm.serve.openai_server import _validate_required_tool_grammar_request

        request = make_request(tools=[WEATHER_TOOL], tool_choice="required", **kwargs)
        _validate_required_tool_grammar_request(
            request,
            SamplingParams(),
            guided_decoding_backend="xgrammar",
            tool_parser="kimi_k3",
            reasoning_parser="kimi_k3",
        )

    def test_native_and_context_requests(self) -> None:
        self.validate()
        self.validate(disaggregated_params={"request_type": "context_only"})

    def test_generation_worker_uses_the_relayed_mode(self) -> None:
        self.validate(
            prompt_token_ids=[1, 2, 3],
            disaggregated_params={"request_type": "generation_only", "resolved_thinking": True},
        )
        with pytest.raises(ValueError, match="relayed by the context worker"):
            self.validate(
                prompt_token_ids=[1, 2, 3],
                disaggregated_params={"request_type": "generation_only"},
            )

    def test_pre_tokenized_native_request_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not prompt_token_ids"):
            self.validate(prompt_token_ids=[1, 2, 3])


def test_agent_client_chat_request_with_prompt_cache_key_validates() -> None:
    """kimi-code sends prompt_cache_key on every chat request; it is accepted and ignored."""
    body = {
        "model": "kimi-k3.1",
        "messages": [{"role": "user", "content": "hello"}],
        "prompt_cache_key": "session-1234",
        "thinking": {"type": "enabled", "effort": "max", "keep": "all"},
        "max_completion_tokens": 4096,
        "stream": True,
    }
    request = ChatCompletionRequest.model_validate(body)
    assert request.prompt_cache_key == "session-1234"
    assert request.thinking.effort == "max" and request.thinking.keep == "all"
    assert request.max_completion_tokens == 4096
    # The key never becomes a cache salt, which would isolate each session's prefix cache.
    assert request.cache_salt is None
    # The router re-serializes the request for the workers; the field survives the round trip.
    assert ChatCompletionRequest.model_validate(request.model_dump(exclude_unset=True)) == request


@pytest.mark.parametrize(
    "model_type, tool_arg, reasoning_arg, expected",
    [
        # Kimi K3.1 checkpoints get the kimi_k3 parsers when no option is given.
        ("kimi_k31", None, None, ("kimi_k3", "kimi_k3")),
        # Kimi K3 keeps explicit-only parsers.
        ("kimi_k3", None, None, (None, None)),
        ("kimi_k3", "auto", "kimi_k3", ("auto", "kimi_k3")),
        ("kimi_k3", "none", "none", (None, None)),
        # "none" keeps a parser off; explicit choices are kept.
        ("kimi_k31", "none", "none", (None, None)),
        ("kimi_k31", "none", None, (None, "kimi_k3")),
        ("kimi_k31", "kimi_k2", "auto", ("kimi_k2", "auto")),
        # Other models keep the previous behaviour: no parser unless asked for.
        ("llama", None, None, (None, None)),
        ("llama", "none", "none", (None, None)),
    ],
)
def test_serve_default_kimi_k3_parsers(tmp_path, model_type, tool_arg, reasoning_arg, expected):
    from tensorrt_llm.commands.serve import _default_kimi_k3_parsers

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    assert _default_kimi_k3_parsers(str(tmp_path), tool_arg, reasoning_arg) == expected
