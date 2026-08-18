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

from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    StreamOptions,
    _response_format_to_guided_decoding_params,
)
from tensorrt_llm.serve.openai_server import (
    _apply_kimi_chat_extensions,
    _dynamic_tool_dicts,
    _enforce_kimi_param_policy,
    _validate_kimi_dynamic_tools,
)
from tensorrt_llm.serve.tool_parser.kimi_k3_tool_parser import (
    KimiK3ToolParser,
    _escape_attr,
    _parse_attrs,
    _unescape_attr,
)

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
        assert _dynamic_tool_dicts(req.messages) == []

    def test_system_tools_key_survives_validation(self) -> None:
        req = make_request(messages=[dynamic_system_msg([WEATHER_TOOL]), USER_MSG])
        assert _dynamic_tool_dicts(req.messages) == [WEATHER_TOOL]

    def test_system_tools_with_content_survives_validation(self) -> None:
        # Content correctness is enforced by the kimi-gated serving layer,
        # but the key itself must not be silently stripped by the union.
        req = make_request(
            messages=[dynamic_system_msg([WEATHER_TOOL], content="not empty"), USER_MSG]
        )
        assert _dynamic_tool_dicts(req.messages) == [WEATHER_TOOL]


class TestKimiDynamicToolsValidation:
    def check(self, messages: list, **kwargs) -> None:
        _validate_kimi_dynamic_tools(make_request(messages=messages, **kwargs))

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
        _apply_kimi_chat_extensions(req, model_type)
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
    def enforce(self, **kwargs) -> ChatCompletionRequest:
        req = make_request(**kwargs)
        _enforce_kimi_param_policy(req)
        return req

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
