# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Beam NVFP4 target's channel parser and family routing (generated inputs)."""

import json

import pytest
import torch

from tensorrt_llm.llmapi.reasoning_parser import BeamHarmonyTextParser, ReasoningParserFactory

pytestmark = pytest.mark.cpu_only

#: The parser's registry key and the channel delimiters it consumes.
PARSER_KEY = "beam_harmony_text"
TAGS = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)

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
    assert not any(tag in result.content or tag in result.reasoning_content for tag in TAGS)


@pytest.mark.parametrize("name", sorted(REPLIES))
def test_every_fragmentation_agrees_with_the_whole_reply(name):
    text, content, reasoning = REPLIES[name]
    assert _stream(text, list(text)) == (content, reasoning)
    for cut in range(len(text) + 1):
        assert _stream(text, [text[:cut], text[cut:]]) == (content, reasoning), cut
    for size in (2, 3, 5, 7, 11):
        pieces = [text[i : i + size] for i in range(0, len(text), size)]
        assert _stream(text, pieces) == (content, reasoning), size


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


def test_load_pretrained_config_preserves_beam_rope(tmp_path):
    from tensorrt_llm._torch.models.modeling_beam import BeamConfig
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    rope = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": None,
    }
    config = {
        "model_type": "beam",
        "num_hidden_layers": 6,
        "torch_dtype": "bfloat16",
        "rope_parameters": rope,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = load_pretrained_config(str(tmp_path))
    assert isinstance(loaded, BeamConfig)
    assert loaded.name_or_path == str(tmp_path)
    assert loaded.torch_dtype == torch.bfloat16
    assert loaded.rope_parameters == rope
