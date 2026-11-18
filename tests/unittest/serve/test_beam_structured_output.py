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
"""Beam structured-output constraints begin at the final-channel body."""

import json

import pytest
import xgrammar
from openai.types.responses import ResponseFormatTextJSONSchemaConfig

from tensorrt_llm.llmapi.reasoning_parser import BeamHarmonyTextParser
from tensorrt_llm.serve.openai_protocol import (
    ResponseFormat,
    _response_format_text_config_to_guided_decoding_params,
    _response_format_to_guided_decoding_params,
)
from tensorrt_llm.serve.serving_extensions import (
    OutputMode,
    get_serving_extension,
    validate_tools_for,
)

pytestmark = pytest.mark.cpu_only

FINAL = "<|channel|>final<|message|>"
SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
@pytest.mark.threadleak(enabled=False)
def test_beam_tools_rejected_before_generation(serve_env, endpoint: str, stream: bool) -> None:
    from test_common.serve.app_factory import AggConfig

    server = serve_env.build_openai_server(
        AggConfig(model_type="beam", reasoning_parser="beam_harmony_text")
    )
    function = {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}
    payload = {"model": "fake-model", "stream": stream, "tool_choice": "auto"}
    if endpoint == "chat/completions":
        payload["messages"] = [{"role": "user", "content": "What is the weather?"}]
        payload["tools"] = [{"type": "function", "function": function}]
    else:
        payload["input"] = "What is the weather?"
        payload["tools"] = [{"type": "function", **function, "strict": False}]
        payload["store"] = False
    response = serve_env.client(server.app).post(f"/v1/{endpoint}", json=payload)
    assert response.status_code == 400
    assert "Tool calling is not supported" in response.json()["message"]
    assert server.generator.generate_calls == []


def test_tool_validation_is_scoped_to_beam_parser() -> None:
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    with pytest.raises(ValueError, match="Tool calling is not supported"):
        validate_tools_for("beam_harmony_text", tools)
    for empty in (None, []):
        validate_tools_for("beam_harmony_text", empty)
    for parser in (None, "deepseek-r1", "unregistered-parser"):
        validate_tools_for(parser, tools)
    get_serving_extension(None).validate_tools(tools)


@pytest.mark.parametrize(
    "response_format,content",
    [
        ({"type": "json_object"}, {"type": "json_schema", "json_schema": {"type": "object"}}),
        (
            {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}},
            {"type": "json_schema", "json_schema": SCHEMA},
        ),
        ({"type": "regex", "regex": "[0-9]+"}, {"type": "regex", "pattern": "[0-9]+"}),
        ({"type": "ebnf", "ebnf": 'root ::= "5"'}, {"type": "grammar", "grammar": 'root ::= "5"'}),
    ],
)
def test_chat_constraint_starts_at_final_channel(response_format: dict, content: dict) -> None:
    params = _response_format_to_guided_decoding_params(
        ResponseFormat(**response_format), reasoning_parser="beam_harmony_text"
    )
    fmt = json.loads(params.structural_tag)["format"]
    assert fmt["type"] == "triggered_tags"
    assert fmt["triggers"] == [FINAL]
    assert fmt["stop_after_first"] is True
    assert len(fmt["tags"]) == 1
    tag = fmt["tags"][0]
    assert tag["begin"] == FINAL
    assert tag["end"] == ""
    for key, value in content.items():
        assert tag["content"][key] == value


def test_responses_schema_uses_same_constraint() -> None:
    params = _response_format_text_config_to_guided_decoding_params(
        ResponseFormatTextJSONSchemaConfig(type="json_schema", name="answer", schema=SCHEMA),
        reasoning_parser="beam_harmony_text",
    )
    chat_params = _response_format_to_guided_decoding_params(
        ResponseFormat(type="json_schema", json_schema={"schema": SCHEMA}),
        reasoning_parser="beam_harmony_text",
    )
    assert json.loads(params.structural_tag) == json.loads(chat_params.structural_tag)
    extension = get_serving_extension("beam")
    assert extension.output_mode() == OutputMode.TEXT
    assert extension.render_prompt(None) is None


@pytest.mark.parametrize(
    "prefix,reasoning",
    [
        ("", ""),
        ("<|start|>assistant", ""),
        (
            "<|channel|>analysis<|message|>12 - 7 = 5.<|end|><|start|>assistant",
            "12 - 7 = 5.",
        ),
    ],
)
def test_constrained_answer_survives_whole_and_streamed_parsing(
    prefix: str, reasoning: str
) -> None:
    params = _response_format_to_guided_decoding_params(
        ResponseFormat(type="json_object"), reasoning_parser="beam_harmony_text"
    )
    compiler = xgrammar.GrammarCompiler(
        xgrammar.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<EOS>"], stop_token_ids=[256])
    )
    compiled = compiler.compile_structural_tag(params.structural_tag)
    matcher = xgrammar.GrammarMatcher(compiled)
    header = prefix + FINAL
    assert matcher.accept_string(header)
    assert not matcher.accept_string("not JSON")
    answer = '{"answer": 5}'
    assert matcher.accept_string(answer)
    assert matcher.accept_token(256)

    reply = header + answer
    whole = BeamHarmonyTextParser().parse(reply)
    assert whole.content == answer
    assert whole.reasoning_content == reasoning
    parser = BeamHarmonyTextParser()
    chunks = [parser.parse_delta(char) for char in reply] + [parser.finish()]
    assert "".join(chunk.content for chunk in chunks) == answer
    assert "".join(chunk.reasoning_content for chunk in chunks) == reasoning
