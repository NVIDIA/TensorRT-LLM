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
"""Inkling tool-call parsing, batch and streaming.

Inkling frames a call with special tokens and carries the tool name twice --
once in the ``<|message_model|>`` header and once inside the JSON payload:

    <|message_model|>get_weather<|content_invoke_tool_json|>
        {"name":"get_weather","args":{"city":"NYC"}}<|end_message|>

Three things are easy to get wrong and are asserted here rather than assumed:
the two names disagreeing (the call must be dropped, not executed under a
guess), a control token split across streaming deltas (its first half must not
leak as visible text), and streaming having to agree with a one-shot parse for
every possible chunk boundary.
"""

import json

import pytest

from tensorrt_llm.llmapi.inkling_tokens import (
    INKLING_CONTENT_TEXT,
    INKLING_END_MESSAGE,
    INKLING_INVOKE_TOOL_JSON,
    INKLING_MESSAGE_MODEL,
)
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam, FunctionDefinition
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

MM, IT, EM, CT = (
    INKLING_MESSAGE_MODEL,
    INKLING_INVOKE_TOOL_JSON,
    INKLING_END_MESSAGE,
    INKLING_CONTENT_TEXT,
)


def _tools(*names):
    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name=n, description=n, parameters={"type": "object", "properties": {}}
            ),
        )
        for n in names
    ]


def _parser():
    return ToolParserFactory.create_tool_parser("inkling")


def _call(name, args, header=None):
    head = f"{MM}{header if header is not None else name}"
    return f'{head}{IT}{{"name":"{name}","args":{json.dumps(args)}}}{EM}'


def test_registered_and_needs_raw_special_tokens():
    """The parser is reachable by name and declares that it needs the tokens.

    Without the flag the serving layer strips the delimiters before the parser
    runs, and there is nothing left to detect.
    """
    parser = _parser()
    assert parser.needs_raw_special_tokens is True
    assert parser.has_tool_call(_call("f", {}))
    assert not parser.has_tool_call("plain answer")


def test_auto_resolves_from_model_type(tmp_path):
    """`--tool_parser auto` has to find Inkling from its HF config."""
    from tensorrt_llm.serve.tool_parser.tool_parser_factory import resolve_auto_tool_parser

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "inkling_mm_model"}))
    assert resolve_auto_tool_parser(str(tmp_path)) == "inkling"


def test_single_call_batch():
    text = f"{CT}Let me look it up.{EM}" + _call("get_weather", {"city": "NYC"})
    result = _parser().detect_and_parse(text, _tools("get_weather"))
    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert json.loads(result.calls[0].parameters) == {"city": "NYC"}
    # Framing must not survive into visible text.
    assert "<|" not in result.normal_text
    assert result.normal_text.strip() == "Let me look it up."


def test_two_calls_batch():
    text = _call("a", {"x": 1}) + _call("b", {"y": 2})
    result = _parser().detect_and_parse(text, _tools("a", "b"))
    assert [c.name for c in result.calls] == ["a", "b"]
    assert [c.tool_index for c in result.calls] == [0, 1]


def test_header_payload_mismatch_is_dropped():
    """The header and the payload naming different tools is not a guess to make.

    Executing either name could call the wrong function; the call is dropped and
    the surrounding text is still cleaned.
    """
    text = f"{CT}before{EM}" + _call("real", {"a": 1}, header="other")
    result = _parser().detect_and_parse(text, _tools("real", "other"))
    assert result.calls == []
    assert "<|" not in result.normal_text
    # The rejected region is dropped, not regurgitated as an answer.
    assert "real" not in result.normal_text


def test_undeclared_tool_is_still_surfaced():
    """A hallucinated tool surfaces as a structured call, as OpenAI does.

    The harness can then return a tool error and let the model correct itself,
    instead of the invocation degrading into terminal answer text.
    """
    result = _parser().detect_and_parse(_call("not_declared", {"a": 1}), _tools("declared"))
    assert len(result.calls) == 1
    assert result.calls[0].name == "not_declared"


def test_no_tool_call_passthrough():
    result = _parser().detect_and_parse(f"{CT}just an answer{EM}", _tools("f"))
    assert result.calls == []
    assert result.normal_text.strip() == "just an answer"


def _stream(text, splits, tools):
    parser = _parser()
    edges = [0] + list(splits) + [len(text)]
    normal, calls = "", []
    for a, b in zip(edges, edges[1:]):
        r = parser.parse_streaming_increment(text[a:b], tools)
        normal += r.normal_text
        calls.extend(r.calls)
    return normal, calls


def _assembled(calls):
    """Reassemble streamed name/argument deltas into whole calls."""
    out = {}
    for c in calls:
        entry = out.setdefault(c.tool_index, {"name": None, "args": ""})
        if c.name:
            entry["name"] = c.name
        entry["args"] += c.parameters
    return [(v["name"], v["args"]) for _, v in sorted(out.items())]


def test_streaming_matches_batch_char_by_char():
    """Char-by-char streaming must reproduce the one-shot parse.

    This walks a split through the middle of every control token, which is the
    part of the parser most likely to leak framing.
    """
    tools = _tools("get_weather")
    text = f"{CT}checking{EM}" + _call("get_weather", {"city": "NYC"})
    batch = _parser().detect_and_parse(text, tools)
    normal, calls = _stream(text, range(1, len(text)), tools)
    assert "<|" not in normal
    assert normal.strip() == batch.normal_text.strip()
    assert _assembled(calls) == [("get_weather", batch.calls[0].parameters)]


@pytest.mark.parametrize("split", [3, 12, 27, 41, 55])
def test_streaming_no_control_token_leak_at_any_split(split):
    """A control token cut in half must not emit its first half as text."""
    tools = _tools("f")
    text = f"{CT}hello{EM}" + _call("f", {"k": "v"})
    if split >= len(text):
        pytest.skip("split beyond text")
    normal, calls = _stream(text, [split], tools)
    assert "<|" not in normal, f"leaked framing at split {split}: {normal!r}"
    assert _assembled(calls)[0][0] == "f"


def test_streaming_two_calls_get_distinct_indices():
    """Consecutive calls must not collide on tool_index 0.

    An abandoned call that reset the counter would slice the next call's
    arguments against the wrong tool's already-streamed text.
    """
    tools = _tools("a", "b")
    text = _call("a", {"x": 1}) + _call("b", {"y": 2})
    _, calls = _stream(text, range(1, len(text)), tools)
    assembled = _assembled(calls)
    assert [name for name, _ in assembled] == ["a", "b"]


def test_streaming_rejected_call_does_not_shift_the_next_one():
    """After a dropped call the next valid one still parses correctly."""
    tools = _tools("good")
    text = _call("real", {"a": 1}, header="other") + _call("good", {"b": 2})
    _, calls = _stream(text, range(1, len(text)), tools)
    assembled = _assembled(calls)
    assert [name for name, _ in assembled] == ["good"]
    assert json.loads(assembled[0][1]) == {"b": 2}


def test_structure_info_round_trip():
    """The structural tag must frame exactly what the parser accepts."""
    info = _parser().structure_info()("get_weather")
    text = info.begin + '{"city":"NYC"}' + info.end
    result = _parser().detect_and_parse(text, _tools("get_weather"))
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {"city": "NYC"}


def test_reasoning_parser_then_tool_parser_yields_a_call():
    """The composition trtllm-serve actually runs, which no test covered.

    `postprocess_handlers` applies the reasoning parser first and gives its
    content to the tool parser. Every other test here feeds the framed form
    directly, so they all passed while the served path returned
    `content: {"name": ...}` with `tool_calls: []` -- measured against the BF16
    release (job 6032875). The two parsers are only correct together.
    """
    from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

    raw = (
        f"<|content_thinking|>The user wants the weather.<|end_message|>"
        f'{MM}get_weather{IT}{{"name":"get_weather","args":{{"city":"Paris"}}}}{EM}'
    )
    reasoning = ReasoningParserFactory.create_reasoning_parser("inkling")
    parsed = reasoning.parse(raw)
    assert parsed.reasoning_content == "The user wants the weather."

    parser = ToolParserFactory.create_tool_parser("inkling")
    result = parser.detect_and_parse(parsed.content, _tools("get_weather"))
    assert [c.name for c in result.calls] == ["get_weather"]
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}
    assert result.normal_text.strip() == ""
