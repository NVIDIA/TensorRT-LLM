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
"""Tests for the MuseSpark <atem:function_calls> tool-call format."""

import json
import random

import pytest

from tensorrt_llm.llmapi.reasoning_parser import MuseSparkReasoningParser
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam, FunctionDefinition
from tensorrt_llm.serve.tool_parser.core_types import StreamingParseResult
from tensorrt_llm.serve.tool_parser.musespark_parser import MuseSparkToolParser

pytestmark = pytest.mark.cpu_only

MS_S, MS_M, MS_EOM, MS_EOT = "<|start|>", "<|message|>", "<|eom|>", "<|eot|>"


def musespark_block(*invokes):
    return "<atem:function_calls>\n" + "".join(invokes) + "</atem:function_calls>"


def musespark_invoke(name, **params):
    body = "".join(f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params.items())
    return f'<atem:invoke name="{name}">\n{body}</atem:invoke>\n'


MS_WEATHER = musespark_block(musespark_invoke("get_weather", city="Paris", days="3", metric="true"))
MS_TIME = musespark_block(musespark_invoke("get_time", tz="UTC"))


@pytest.fixture
def musespark_sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                        "metric": {"type": "boolean"},
                        "opts": {"type": "object"},
                        "code": {"type": "string"},
                    },
                },
            )
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_time", parameters={"type": "object", "properties": {}}
            )
        ),
    ]


@pytest.fixture
def musespark_parser():
    return MuseSparkToolParser()


def musespark_calls_as_dicts(calls):
    return [(c.tool_index, c.name, json.loads(c.parameters)) for c in calls]


def musespark_stream_chunks(musespark_parser, tools, text, chunks):
    """Feed ``text`` in chunks; return (normal_text, [(idx, name, args_json)])."""
    text_parts = []
    names, args = {}, {}
    pos = 0
    results = []
    for n in chunks:
        results.append(musespark_parser.parse_streaming_increment(text[pos : pos + n], tools))
        pos += n
    if pos < len(text):
        results.append(musespark_parser.parse_streaming_increment(text[pos:], tools))
    results.append(musespark_parser.finish(tools))
    for r in results:
        text_parts.append(r.normal_text)
        for c in r.calls:
            if c.name is not None:
                assert c.tool_index not in names, "name sent twice"
                names[c.tool_index] = c.name
                assert c.parameters == ""
            else:
                assert c.tool_index in names, "arguments before name"
                args[c.tool_index] = args.get(c.tool_index, "") + c.parameters
    assembled = [(i, names[i], args.get(i, "")) for i in sorted(names)]
    return "".join(text_parts), assembled


# ------------------------------------------------------------- has_tool_call
def test_musespark_has_tool_call(musespark_parser):
    assert musespark_parser.has_tool_call(f" to=get_weather{MS_M}{MS_WEATHER}")
    assert musespark_parser.has_tool_call(MS_WEATHER)
    assert not musespark_parser.has_tool_call("just text")
    assert not musespark_parser.has_tool_call('<atem:invoke name="x">')


def test_musespark_flags(musespark_parser):
    assert MuseSparkToolParser.needs_raw_special_tokens is True
    assert musespark_parser.supports_structural_tag() is True
    info = musespark_parser.structure_info()("get_weather")
    assert info.begin.startswith("<atem:function_calls>")
    assert 'name="get_weather"' in info.begin
    assert info.end.endswith("</atem:function_calls>")
    assert info.trigger == "<atem:function_calls>"


# ---------------------------------------------------------- detect_and_parse
def test_musespark_detect_and_parse_no_tool_call(musespark_parser, musespark_sample_tools):
    r = musespark_parser.detect_and_parse("The weather is nice.", musespark_sample_tools)
    assert r == StreamingParseResult(normal_text="The weather is nice.")


def test_musespark_detect_and_parse_single_with_schema(musespark_parser, musespark_sample_tools):
    text = f" to=get_weather{MS_M}{MS_WEATHER}"
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert r.normal_text == ""
    assert musespark_calls_as_dicts(r.calls) == [
        (0, "get_weather", {"city": "Paris", "days": 3, "metric": True})
    ]


def test_musespark_detect_and_parse_without_schema_matches_reference_typing(musespark_parser):
    # Reference: json.loads else raw string; no schema consultation.
    text = musespark_block(
        musespark_invoke("f", a="123", b="true", c="null", d="x y", e='"q"', f="1e3")
    )
    r = musespark_parser.detect_and_parse(text, [])
    assert musespark_calls_as_dicts(r.calls) == [
        (0, "f", {"a": 123, "b": True, "c": None, "d": "x y", "e": "q", "f": 1000.0})
    ]


def test_musespark_declared_string_is_never_retyped(musespark_parser, musespark_sample_tools):
    text = musespark_block(musespark_invoke("get_weather", city="123", code='"q"'))
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert musespark_calls_as_dicts(r.calls) == [(0, "get_weather", {"city": "123", "code": '"q"'})]


def test_musespark_parameter_value_not_trimmed(musespark_parser):
    text = musespark_block(
        '<atem:invoke name="f"><atem:parameter name="a"> x \n</atem:parameter></atem:invoke>'
    )
    r = musespark_parser.detect_and_parse(text, [])
    assert musespark_calls_as_dicts(r.calls) == [(0, "f", {"a": " x \n"})]


def test_musespark_nested_json_parameter(musespark_parser, musespark_sample_tools):
    opts = '{"units": ["c", "f"], "flags": {"verbose": true, "n": null}}'
    text = musespark_block(musespark_invoke("get_weather", city="Paris", opts=opts))
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert musespark_calls_as_dicts(r.calls) == [
        (0, "get_weather", {"city": "Paris", "opts": json.loads(opts)})
    ]


def test_musespark_multiple_invokes_in_one_block(musespark_parser, musespark_sample_tools):
    text = musespark_block(
        musespark_invoke("get_weather", city="Paris"), musespark_invoke("get_time", tz="UTC")
    )
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert musespark_calls_as_dicts(r.calls) == [
        (0, "get_weather", {"city": "Paris"}),
        (1, "get_time", {"tz": "UTC"}),
    ]


def test_musespark_parallel_calls_across_messages_raw_stream(
    musespark_parser, musespark_sample_tools
):
    # Raw decoded stream (no reasoning parser): headers, <|eom|>, <|eot|>.
    text = (
        f" to=self{MS_M}think{MS_EOM}{MS_S}assistant to=get_weather{MS_M}{MS_WEATHER}{MS_EOM}"
        f"{MS_S}assistant to=get_time{MS_M}{MS_TIME}{MS_EOT}"
    )
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert r.normal_text == ""
    assert [(c.tool_index, c.name) for c in r.calls] == [(0, "get_weather"), (1, "get_time")]


def test_musespark_parallel_calls_after_reasoning_parser(musespark_parser, musespark_sample_tools):
    text = (
        f" to=self{MS_M}think{MS_EOM}{MS_S}assistant to=commentary{MS_M}Checking.{MS_EOM}"
        f"{MS_S}assistant to=get_weather{MS_M}{MS_WEATHER}{MS_EOM}"
        f"{MS_S}assistant to=get_time{MS_M}{MS_TIME}{MS_EOT}"
    )
    content = MuseSparkReasoningParser().parse(text).content
    r = musespark_parser.detect_and_parse(content, musespark_sample_tools)
    assert r.normal_text == "Checking."
    assert [(c.tool_index, c.name) for c in r.calls] == [(0, "get_weather"), (1, "get_time")]


def test_musespark_commentary_before_call_is_kept(musespark_parser, musespark_sample_tools):
    # Reference drops this prose (content=None once a call is found).
    text = f" to=commentary{MS_M}Let me check.{MS_EOM}{MS_S}assistant to=get_weather{MS_M}{MS_WEATHER}{MS_EOT}"
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert r.normal_text == "Let me check."
    assert len(r.calls) == 1


def test_musespark_reasoning_span_suppressed_without_reasoning_parser(
    musespark_parser, musespark_sample_tools
):
    text = f" to=self{MS_M}private{MS_EOM}{MS_S}assistant to=user{MS_M}Hi{MS_EOT}"
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert r == StreamingParseResult(normal_text="Hi")


def test_musespark_escaped_angle_pipe_text_in_body(musespark_parser, musespark_sample_tools):
    text = musespark_block(musespark_invoke("get_weather", city="a < b | c <|x|> d"))
    r = musespark_parser.detect_and_parse(
        f" to=user{MS_M}x <| y{MS_EOT}" + text, musespark_sample_tools
    )
    assert r.normal_text == "x <| y"
    assert musespark_calls_as_dicts(r.calls) == [(0, "get_weather", {"city": "a < b | c <|x|> d"})]


def test_musespark_unterminated_block(musespark_parser, musespark_sample_tools):
    text = (
        '<atem:function_calls>\n<atem:invoke name="get_time">'
        '<atem:parameter name="tz">UTC</atem:parameter></atem:invoke>'
    )
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert musespark_calls_as_dicts(r.calls) == [(0, "get_time", {"tz": "UTC"})]


def test_musespark_unterminated_invoke(musespark_parser, musespark_sample_tools):
    """A generation cut off before </atem:invoke> still yields its call.

    The streaming side already does this -- see
    test_musespark_streaming_truncated_block_closed_by_finish, which uses the
    same text. The non-streaming side required the closing tag and returned no
    call at all, so one generation produced a tool call when streamed and none
    when not. That is what made the serving smoke's streamed-vs-unstreamed
    check fail against a live server whose reply hit max_tokens.
    """
    text = (
        '<atem:function_calls><atem:invoke name="get_time">'
        '<atem:parameter name="tz">UTC</atem:parameter>'
    )
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert musespark_calls_as_dicts(r.calls) == [(0, "get_time", {"tz": "UTC"})]


def test_musespark_unterminated_invoke_drops_half_written_parameter(
    musespark_parser, musespark_sample_tools
):
    """Only parameters that closed are kept.

    The cut lands inside the second parameter's value, so `days` never gets its
    </atem:parameter>. Reporting it would invent an argument the model had not
    finished writing.
    """
    text = (
        '<atem:function_calls><atem:invoke name="get_weather">'
        '<atem:parameter name="city">Paris</atem:parameter>'
        '<atem:parameter name="days">3'
    )
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert musespark_calls_as_dicts(r.calls) == [(0, "get_weather", {"city": "Paris"})]


def test_musespark_unterminated_invoke_after_a_complete_one(
    musespark_parser, musespark_sample_tools
):
    """Two invokes, only the second truncated: both are reported.

    An unterminated invoke has to end at the next invoke's opening tag, or the
    first call swallows the second.
    """
    text = (
        "<atem:function_calls>"
        '<atem:invoke name="get_time">'
        '<atem:parameter name="tz">UTC</atem:parameter></atem:invoke>'
        '<atem:invoke name="get_weather">'
        '<atem:parameter name="city">Paris</atem:parameter>'
    )
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert musespark_calls_as_dicts(r.calls) == [
        (0, "get_time", {"tz": "UTC"}),
        (1, "get_weather", {"city": "Paris"}),
    ]


def test_musespark_truncated_invoke_agrees_streamed_and_unstreamed(
    musespark_parser, musespark_sample_tools
):
    """The contract the serving smoke checks, pinned at the parser level.

    The same truncated text through both entry points must produce the same
    calls.
    """
    text = (
        '<atem:function_calls><atem:invoke name="get_weather">'
        '<atem:parameter name="city">Paris</atem:parameter>'
    )
    unstreamed = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    unstreamed_calls = [(c.tool_index, c.name, c.parameters) for c in unstreamed.calls]

    musespark_parser._reset_streaming_state()
    _, streamed_calls = musespark_stream_chunks(
        musespark_parser, musespark_sample_tools, text, [len(text)]
    )
    assert unstreamed_calls == streamed_calls


def test_musespark_optional_quotes_and_dotted_names(musespark_parser):
    text = musespark_block(
        "<atem:invoke name=media.create_image>"
        "<atem:parameter name='prompt'>a cat</atem:parameter>"
        "</atem:invoke>"
    )
    r = musespark_parser.detect_and_parse(text, [])
    assert musespark_calls_as_dicts(r.calls) == [(0, "media.create_image", {"prompt": "a cat"})]


def test_musespark_bare_json_invoke_body_from_structural_tag(
    musespark_parser, musespark_sample_tools
):
    info = musespark_parser.structure_info()("get_weather")
    text = info.begin + '{"city": "Paris", "days": 2}' + info.end
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert musespark_calls_as_dicts(r.calls) == [(0, "get_weather", {"city": "Paris", "days": 2})]


def test_musespark_malformed_tool_header_returns_text(musespark_parser, musespark_sample_tools):
    text = f" to=get_weather{MS_M}oops no block{MS_EOT}"
    r = musespark_parser.detect_and_parse(text, musespark_sample_tools)
    assert r.calls == []
    assert r.normal_text == "oops no block"


def test_musespark_no_params_gives_empty_object(musespark_parser, musespark_sample_tools):
    r = musespark_parser.detect_and_parse(
        musespark_block('<atem:invoke name="get_time"></atem:invoke>'), musespark_sample_tools
    )
    assert musespark_calls_as_dicts(r.calls) == [(0, "get_time", {})]


# ------------------------------------------------------------------ streaming
def test_musespark_streaming_plain_text_passthrough(musespark_parser, musespark_sample_tools):
    # A trailing space is held one chunk: it could begin a " to=" header.
    r = musespark_parser.parse_streaming_increment("Hello ", musespark_sample_tools)
    assert r == StreamingParseResult(normal_text="Hello")
    r = musespark_parser.parse_streaming_increment("world", musespark_sample_tools)
    assert r.normal_text == " world"
    r = musespark_parser.parse_streaming_increment("!", musespark_sample_tools)
    assert r.normal_text == "!"


def test_musespark_streaming_holds_partial_bot_token(musespark_parser, musespark_sample_tools):
    r = musespark_parser.parse_streaming_increment("text <atem:fun", musespark_sample_tools)
    assert r.normal_text == "text"
    r = musespark_parser.parse_streaming_increment("ny", musespark_sample_tools)  # not a tag
    assert r.normal_text == " <atem:funny"


def test_musespark_streaming_header_never_leaks(musespark_parser, musespark_sample_tools):
    for d in [f"{MS_S}assis", "tant to=get_", "weather", MS_M]:
        r = musespark_parser.parse_streaming_increment(d, musespark_sample_tools)
        assert r.normal_text == "" and r.calls == []


def test_musespark_streaming_bare_header_first_message(musespark_parser, musespark_sample_tools):
    # The tool name arrives split mid-word, so the parser has to hold the
    # partial header until the name completes. Sliced rather than written out
    # as two literals: the halves are not words and read as typos.
    name = "get_weather"
    chunks = [" to", "=" + name[:6], name[6:], MS_M, "<atem:function_calls>"]
    for d in chunks:
        r = musespark_parser.parse_streaming_increment(d, musespark_sample_tools)
        assert r.normal_text == "" and r.calls == []


def test_musespark_streaming_reasoning_span_suppressed(musespark_parser, musespark_sample_tools):
    chunks = [f" to=self{MS_M}", "priv", "ate<|e", f"om|>{MS_S}assistant to=user{MS_M}Hi", MS_EOT]
    out = [
        musespark_parser.parse_streaming_increment(c, musespark_sample_tools).normal_text
        for c in chunks
    ]
    assert "".join(out) == "Hi"


def test_musespark_streaming_name_then_prefix_deltas(musespark_parser, musespark_sample_tools):
    chunks = [
        '<atem:function_calls>\n<atem:invoke name="get_weather">\n',
        '<atem:parameter name="city">Par',
        'is</atem:parameter>\n<atem:parameter name="days">3</atem:parameter>\n',
        "</atem:invoke>\n</atem:function_calls>",
    ]
    r0 = musespark_parser.parse_streaming_increment(chunks[0], musespark_sample_tools)
    assert [(c.tool_index, c.name, c.parameters) for c in r0.calls] == [(0, "get_weather", "")]
    r1 = musespark_parser.parse_streaming_increment(chunks[1], musespark_sample_tools)
    assert r1.calls == []  # parameter not closed yet
    r2 = musespark_parser.parse_streaming_increment(chunks[2], musespark_sample_tools)
    assert [c.parameters for c in r2.calls] == ['{"city": "Paris", "days": 3']
    r3 = musespark_parser.parse_streaming_increment(chunks[3], musespark_sample_tools)
    assert [c.parameters for c in r3.calls] == ["}"]
    assert musespark_parser.prev_tool_call_arr == [
        {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}
    ]
    assert musespark_parser.streamed_args_for_tool == ['{"city": "Paris", "days": 3}']


def test_musespark_streaming_indices_continue_across_blocks(
    musespark_parser, musespark_sample_tools
):
    text = f" to=get_weather{MS_M}{MS_WEATHER}{MS_EOM}{MS_S}assistant to=get_time{MS_M}{MS_TIME}{MS_EOT}"
    normal, calls = musespark_stream_chunks(
        musespark_parser, musespark_sample_tools, text, [len(text)]
    )
    assert normal == ""
    assert calls == [
        (0, "get_weather", '{"city": "Paris", "days": 3, "metric": true}'),
        (1, "get_time", '{"tz": "UTC"}'),
    ]


def test_musespark_streaming_truncated_block_closed_by_finish(
    musespark_parser, musespark_sample_tools
):
    text = '<atem:function_calls><atem:invoke name="get_time"><atem:parameter name="tz">UTC</atem:parameter>'
    normal, calls = musespark_stream_chunks(
        musespark_parser, musespark_sample_tools, text, [len(text)]
    )
    assert calls == [(0, "get_time", '{"tz": "UTC"}')]


def test_musespark_finish_releases_held_text(musespark_parser, musespark_sample_tools):
    assert (
        musespark_parser.parse_streaming_increment("abc <atem", musespark_sample_tools).normal_text
        == "abc"
    )
    assert musespark_parser.finish(musespark_sample_tools).normal_text == " <atem"


EQUIV_TEXTS = [
    f" to=get_weather{MS_M}{MS_WEATHER}{MS_EOT}",
    f" to=self{MS_M}think{MS_EOM}{MS_S}assistant to=get_weather{MS_M}{MS_WEATHER}{MS_EOM}"
    f"{MS_S}assistant to=get_time{MS_M}{MS_TIME}{MS_EOT}",
    f" to=commentary{MS_M}Let me check the weather.{MS_EOM}{MS_S}assistant to=get_weather{MS_M}{MS_WEATHER}{MS_EOT}",
    musespark_block(
        musespark_invoke(
            "get_weather", city="Paris", opts='{"units": ["c", "f"], "flags": {"verbose": true}}'
        ),
        musespark_invoke("get_time", tz="UTC"),
    ),
    f" to=user{MS_M}Plain answer with a < b | c and <|odd|> bits{MS_EOT}",
    f" to=user{MS_M}No tools here.{MS_EOT}",
]


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("text", EQUIV_TEXTS)
def test_musespark_stream_matches_detect_and_parse(seed, text, musespark_sample_tools):
    rng = random.Random(seed)
    chunks, remaining = [], len(text)
    while remaining > 0:
        n = min(rng.randint(1, 9), remaining)
        chunks.append(n)
        remaining -= n
    expected = MuseSparkToolParser().detect_and_parse(text, musespark_sample_tools)
    normal, calls = musespark_stream_chunks(
        MuseSparkToolParser(), musespark_sample_tools, text, chunks
    )
    assert normal.strip() == expected.normal_text
    assert [(i, n, json.loads(a)) for i, n, a in calls] == musespark_calls_as_dicts(expected.calls)


@pytest.mark.parametrize("seed", range(10))
def test_musespark_reasoning_then_tool_parser_chain_streaming(seed, musespark_sample_tools):
    """Check the reasoning-to-tool parser chain end to end.

    The serving layer feeds the reasoning parser's content deltas into the
    tool parser (postprocess_handlers.py).
    """
    text = (
        f" to=self{MS_M}Think.{MS_EOM}{MS_S}assistant to=commentary{MS_M}Looking up.{MS_EOM}"
        f"{MS_S}assistant to=get_weather{MS_M}{MS_WEATHER}{MS_EOM}"
        f"{MS_S}assistant to=get_time{MS_M}{MS_TIME}{MS_EOT}"
    )
    rng = random.Random(seed)
    rp, tp = MuseSparkReasoningParser(), MuseSparkToolParser()
    reasoning, normal, names, args = [], [], {}, {}
    pos = 0
    while pos < len(text):
        n = min(rng.randint(1, 8), len(text) - pos)
        r = rp.parse_delta(text[pos : pos + n])
        pos += n
        reasoning.append(r.reasoning_content)
        t = (
            tp.parse_streaming_increment(r.content, musespark_sample_tools)
            if r.content
            else StreamingParseResult()
        )
        normal.append(t.normal_text)
        for c in t.calls:
            if c.name:
                names[c.tool_index] = c.name
            else:
                args[c.tool_index] = args.get(c.tool_index, "") + c.parameters
    tail = rp.finish()
    reasoning.append(tail.reasoning_content)
    t = (
        tp.parse_streaming_increment(tail.content, musespark_sample_tools)
        if tail.content
        else StreamingParseResult()
    )
    t2 = tp.finish(musespark_sample_tools)
    for res in (t, t2):
        normal.append(res.normal_text)
        for c in res.calls:
            if c.name:
                names[c.tool_index] = c.name
            else:
                args[c.tool_index] = args.get(c.tool_index, "") + c.parameters
    assert "".join(reasoning) == "Think."
    assert "".join(normal) == "Looking up."
    assert names == {0: "get_weather", 1: "get_time"}
    assert json.loads(args[0]) == {"city": "Paris", "days": 3, "metric": True}
    assert json.loads(args[1]) == {"tz": "UTC"}
