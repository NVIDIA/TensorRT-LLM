# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-response and streaming tests for the Beam Harmony text parser."""

import pytest

from tensorrt_llm.llmapi.reasoning_parser import BeamHarmonyTextParser, ReasoningParserFactory

pytestmark = pytest.mark.cpu_only

#: The parser's registry key.
PARSER_KEY = "beam_harmony_text"

#: (reply text after the generation prefix, expected content, expected reasoning_content)
REPLIES = {
    "analysis_then_final": (
        "<|channel|>analysis<|message|>Add the two numbers.<|end|><|start|>assistant"
        "<|channel|>final<|message|>2 + 3 = 5.<|return|>",
        "2 + 3 = 5.",
        "Add the two numbers.",
    ),
    "final_only": ("<|channel|>final<|message|>Paris.<|return|>", "Paris.", ""),
    "no_headers": ("Plain text without channels.", "Plain text without channels.", ""),
    "text_before_message": ("Pre<|message|>Hello<|return|>", "PreHello", ""),
    "text_before_channel": (
        "Pre<|channel|>analysis<|message|>Think.<|end|><|start|>assistant"
        "<|channel|>final<|message|>Hello<|return|>",
        "PreHello",
        "Think.",
    ),
    "text_before_constraint": (
        'Pre<|constrain|>json<|message|>{"ok":true}<|return|>',
        'Pre{"ok":true}',
        "",
    ),
    "stopped_in_analysis": ("<|channel|>analysis<|message|>Still thinking", "", "Still thinking"),
    "stopped_in_final": (
        "<|channel|>analysis<|message|>a<|end|><|start|>assistant<|channel|>final<|message|>Par",
        "Par",
        "a",
    ),
    "multiline_bodies": (
        "<|channel|>analysis<|message|>Line 1\n\nLine 2 <b>x</b><|end|><|start|>assistant"
        "<|channel|>final<|message|>\\(12 - 7 = 5\\)\n\nANSWER: 5<|return|>",
        "\\(12 - 7 = 5\\)\n\nANSWER: 5",
        "Line 1\n\nLine 2 <b>x</b>",
    ),
    "literal_angle_brackets": (
        "<|channel|>final<|message|>a <| b and <|x|><|return|>",
        "a <| b and <|x|>",
        "",
    ),
    "quoted_header_markers": (
        '<|channel|>final<|message|>Split on "<|channel|>". The header is '
        "<|start|>assistant; <|message|> starts the body; <|constrain|> labels its format."
        "<|return|>ignored",
        'Split on "<|channel|>". The header is '
        "<|start|>assistant; <|message|> starts the body; <|constrain|> labels its format.",
        "",
    ),
    "quoted_markers_in_analysis": (
        "<|channel|>analysis<|message|>Explain <|channel|>final<|message|>, "
        "<|start|>assistant and <|constrain|>json.<|end|><|start|>assistant"
        "<|channel|>final<|message|>These are header markers.<|return|>",
        "These are header markers.",
        "Explain <|channel|>final<|message|>, <|start|>assistant and <|constrain|>json.",
    ),
    "partial_header_marker_at_eof": (
        "<|channel|>final<|message|>Example: <|chan",
        "Example: <|chan",
        "",
    ),
    "partial_terminator_at_eof": (
        "<|channel|>final<|message|>Example: <|ret",
        "Example: <|ret",
        "",
    ),
    "end_closes_body": (
        "<|channel|>final<|message|>done<|end|>ignored",
        "done",
        "",
    ),
    "tool_channel_ignored": (
        "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|call|>",
        "",
        "",
    ),
    "terminated_by_call": ("<|channel|>final<|message|>done<|call|>ignored", "done", ""),
}


def _stream(text, pieces):
    parser = BeamHarmonyTextParser()
    content = reasoning = ""
    for piece in pieces:
        delta = parser.parse_delta(piece)
        content += delta.content
        reasoning += delta.reasoning_content
    tail = parser.finish()
    return content + tail.content, reasoning + tail.reasoning_content


@pytest.mark.parametrize("name", sorted(REPLIES))
def test_whole_reply(name):
    text, content, reasoning = REPLIES[name]
    result = BeamHarmonyTextParser().parse(text)
    assert (result.content, result.reasoning_content) == (content, reasoning)


@pytest.mark.parametrize("name", sorted(REPLIES))
def test_every_fragmentation_agrees_with_the_whole_reply(name):
    text, content, reasoning = REPLIES[name]
    assert _stream(text, list(text)) == (content, reasoning)
    for cut in range(len(text) + 1):
        assert _stream(text, [text[:cut], text[cut:]]) == (content, reasoning), cut
    for size in (2, 3, 5, 7, 11):
        pieces = [text[i : i + size] for i in range(0, len(text), size)]
        assert _stream(text, pieces) == (content, reasoning), size


def test_body_holds_only_partial_terminators() -> None:
    parser = BeamHarmonyTextParser()
    assert parser.parse_delta("<|channel|>final<|message|><|chan").content == "<|chan"
    assert parser.parse_delta("nel|><|ret").content == "nel|>"
    assert parser.parse_delta("urn|>ignored").content == ""
    assert parser.finish().content == ""


@pytest.mark.parametrize("long_header", [False, True])
def test_delimiter_heavy_reply(long_header: bool) -> None:
    if long_header:
        text = "<|channel|>final" + "<|constrain|>json" * 4096
        text += "<|message|>answer<|return|>ignored"
        expected = ("answer", "")
    else:
        cycle = (
            "<|channel|>analysis<|message|>think<|end|>"
            "<|channel|>final<|message|>answer<|channel|>literal<|end|>"
        )
        text = cycle * 4096 + "<|return|>ignored"
        expected = ("answer<|channel|>literal" * 4096, "think" * 4096)
    result = BeamHarmonyTextParser().parse(text)
    assert (result.content, result.reasoning_content) == expected
    assert _stream(text, [text[i : i + 257] for i in range(0, len(text), 257)]) == expected


def test_parse_is_self_contained_and_repeatable():
    text, content, reasoning = REPLIES["analysis_then_final"]
    parser = BeamHarmonyTextParser()
    parser.parse_delta("<|channel|>analysis<|message|>left over")
    for _ in range(2):
        result = parser.parse(text)
        assert (result.content, result.reasoning_content) == (content, reasoning)


def test_registered_under_its_key():
    parser = ReasoningParserFactory.create_reasoning_parser(PARSER_KEY)
    assert isinstance(parser, BeamHarmonyTextParser)
    assert ReasoningParserFactory.needs_raw_special_tokens(PARSER_KEY)
