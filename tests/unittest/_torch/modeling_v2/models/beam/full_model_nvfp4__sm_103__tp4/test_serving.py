# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Beam NVFP4 target's channel parser and family routing (generated inputs)."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget, routing_module
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.serving import (  # noqa: E501
    PARSER_KEY,
    TAGS,
    BeamHarmonyTextParser,
)
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

pytestmark = pytest.mark.cpu_only

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


def _beam_config(**overrides):
    fields = dict(
        num_hidden_layers=52,
        hidden_size=5120,
        num_experts=256,
        quantization_config={"quant_algo": "NVFP4"},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _mapping(**overrides):
    fields = dict(
        world_size=4, tp_size=4, pp_size=1, moe_tp_size=4, moe_ep_size=1, enable_attention_dp=False
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_route_claims_only_the_supported_deployment():
    route = routing_module("BeamForCausalLM").route
    target = route(_beam_config(), _mapping(), (10, 3), None)
    assert target.__name__ == "ModelingV2BeamFullModelNvfp4Sm103Tp4"
    declined = [
        (_beam_config(), _mapping(), (10, 0), None),
        (_beam_config(num_hidden_layers=6), _mapping(), (10, 3), None),
        (_beam_config(quantization_config=None), _mapping(), (10, 3), None),
        (_beam_config(), _mapping(tp_size=2, pp_size=2, moe_tp_size=2), (10, 3), None),
        (_beam_config(), _mapping(moe_tp_size=1, moe_ep_size=4), (10, 3), None),
        (_beam_config(), _mapping(enable_attention_dp=True), (10, 3), None),
        (_beam_config(), _mapping(), (10, 3), object()),
    ]
    for args in declined:
        with pytest.raises(NoTarget):
            route(*args)
