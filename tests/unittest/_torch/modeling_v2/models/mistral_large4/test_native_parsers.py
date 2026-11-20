# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mistral_large_4 tool-call and reasoning parsers.

Every case runs through the serving glue streamed in every two-way split, token-sized
pieces and seeded random pieces, and must give the one-shot result.

The round-trip tests need MISTRAL_LARGE4_CHECKPOINT_DIR (its tekken.json): mistral-common
encodes an assistant message, the native tokenizer decodes the ids whole and token by token
with special tokens kept, and the parsers must return the message.
"""

from __future__ import annotations

import json
import os
import random
import re

import pytest

from tensorrt_llm.llmapi.reasoning_parser import (
    MistralLarge4ReasoningParser,
    ReasoningParserFactory,
)
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.chat_utils import make_tool_call_id
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam
from tensorrt_llm.serve.openai_server import (
    _build_tool_strict_guided_decoding_params,
    _configure_parser_special_token_decoding,
)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    apply_reasoning_parser,
    apply_tool_parser,
)
from tensorrt_llm.serve.tool_parser.mistral_large_4_parser import MistralLarge4ToolParser
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

pytestmark = pytest.mark.cpu_only

PARSER = "mistral_large_4"
MARKERS = ("[TOOL_CALLS]", "[ARGS]", "[THINK]", "[/THINK]", "</s>")


def _tool(name, strict=None):
    function = {"name": name, "parameters": {"type": "object", "properties": {}}}
    if strict is not None:
        function["strict"] = strict
    return ChatCompletionToolsParam.model_validate({"type": "function", "function": function})


TOOLS = [
    _tool(name)
    for name in (
        "bash",
        "get_current_weather",
        "get_weather",
    )
]

# (model output, reasoning, content, [(name, arguments)]) through both parsers.
COMBINED_CASES = [
    # The call after reasoning stays out of the reasoning.
    (
        '[THINK]Need weather.[/THINK]Checking.[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}</s>',
        "Need weather.",
        "Checking.",
        [("get_weather", '{"city": "Paris"}')],
    ),
    # [TOOL_CALLS] also ends an unclosed reasoning block.
    (
        '[THINK]need weather [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}',
        "need weather",
        "",
        [("get_weather", '{"city": "Paris"}')],
    ),
    # A [THINK] inside the arguments belongs to them.
    (
        '[TOOL_CALLS]bash[ARGS]{"command": "echo [THINK]"}',
        "",
        "",
        [("bash", '{"command": "echo [THINK]"}')],
    ),
]


def _args(reasoning_parser=True, **kwargs):
    return ChatPostprocArgs(
        role="assistant",
        model="mistral-large-4",
        tools=TOOLS,
        reasoning_parser=PARSER if reasoning_parser else None,
        tool_parser=PARSER,
        **kwargs,
    )


def _parse_whole(text, reasoning_parser=True):
    args = _args(reasoning_parser)
    content, reasoning = apply_reasoning_parser(args, 0, text, streaming=False)
    content, calls = apply_tool_parser(args, 0, content, streaming=False)
    return reasoning, content, [(call.name, call.parameters) for call in calls]


def _parse_stream(pieces, reasoning_parser=True):
    """The serving glue over streamed pieces; checks the call deltas' shape on the way."""
    args = _args(reasoning_parser)
    reasoning, content, calls = [], [], {}
    for number, piece in enumerate(pieces):
        finished = number == len(pieces) - 1
        delta, reasoning_delta = apply_reasoning_parser(args, 0, piece, True, finished)
        delta, items = apply_tool_parser(args, 0, delta, True, finished)
        reasoning.append(reasoning_delta)
        content.append(delta)
        for item in items:
            if item.name is not None:
                assert item.tool_index not in calls, "a call was opened twice"
                assert item.tool_index == len(calls), "calls must open in order"
                calls[item.tool_index] = [item.name, item.parameters]
            else:
                assert item.tool_index == len(calls) - 1, "arguments of a call that is not last"
                calls[item.tool_index][1] += item.parameters
    return "".join(reasoning), "".join(content), [tuple(call) for call in calls.values()]


def _splits(text, seed=0):
    yield [text]
    yield list(text)
    for cut in range(1, len(text)):
        yield [text[:cut], text[cut:]]
    rng = random.Random(seed)
    for _ in range(20):
        cuts = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(1, 6))))
        yield [text[start:end] for start, end in zip([0, *cuts], [*cuts, len(text)])]


def _assert_stream_matches(text, reasoning_parser=True):
    whole = _parse_whole(text, reasoning_parser)
    for pieces in _splits(text):
        if not pieces[-1]:
            continue
        reasoning, content, calls = _parse_stream(pieces, reasoning_parser)
        assert calls == whole[2], pieces
        assert reasoning == whole[0], pieces
        # A streamed call cannot take back whitespace already sent before it.
        assert (content.strip() if calls else content) == whole[1], pieces


@pytest.mark.parametrize("text,reasoning,content,calls", COMBINED_CASES)
def test_reasoning_and_tool_calls(text, reasoning, content, calls):
    assert _parse_whole(text) == (reasoning, content, calls)


@pytest.mark.parametrize(
    "text",
    [case[0] for case in COMBINED_CASES if case[0]],
)
def test_streaming_matches_the_one_shot_parse(text):
    _assert_stream_matches(text)
    _assert_stream_matches(text, reasoning_parser=False)


def test_markers_never_reach_the_client():
    for text in [case[0] for case in COMBINED_CASES]:
        reasoning, content, calls = _parse_whole(text)
        for marker in MARKERS:
            assert marker not in content, (text, marker)
            assert marker not in reasoning, (text, marker)
            assert all(marker not in name for name, _ in calls), (text, marker)


def test_tool_choice_none_strips_the_calls():
    args = _args(tool_choice="none")
    text = 'Sure.[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}</s>'
    content, reasoning = apply_reasoning_parser(args, 0, text, streaming=False)
    assert apply_tool_parser(args, 0, content, streaming=False) == ("Sure.", [])


def test_registration_and_serving_flags():
    assert isinstance(ToolParserFactory.create_tool_parser(PARSER), MistralLarge4ToolParser)
    assert isinstance(
        ReasoningParserFactory.create_reasoning_parser(PARSER), MistralLarge4ReasoningParser
    )
    assert ToolParserFactory.parsers[PARSER].streaming_matches_whole_parse

    # The reasoning delimiters are control tokens, so every request keeps them.
    params = SamplingParams()
    _configure_parser_special_token_decoding(params, PARSER, None, has_tools=False)
    assert params.skip_special_tokens is False
    # The tool parser alone keeps them only for requests with tools.
    params = SamplingParams()
    _configure_parser_special_token_decoding(params, None, PARSER, has_tools=False)
    assert params.skip_special_tokens is True
    _configure_parser_special_token_decoding(params, None, PARSER, has_tools=True)
    assert params.skip_special_tokens is False


def test_strict_tools_are_not_grammar_constrained():
    """The xgrammar vocabulary cannot write [TOOL_CALLS], so no tool grammar is built."""
    assert _build_tool_strict_guided_decoding_params([_tool("get_weather", True)], PARSER) is None


def test_tool_call_ids_take_the_mistral_form():
    ids = {make_tool_call_id(id_type="mistral") for _ in range(100)}
    assert len(ids) == 100
    assert all(re.fullmatch(r"[A-Za-z0-9]{9}", call_id) for call_id in ids)


@pytest.fixture(scope="module")
def tokenizer():
    if not os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR"):
        pytest.skip("needs checkpoint tokenizer")
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    return MistralTokenizer.from_pretrained(os.environ["MISTRAL_LARGE4_CHECKPOINT_DIR"])


def _assistant_ids(tokenizer, assistant, after):
    """The ids mistral-common writes for an assistant turn, through its end of sequence."""
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.tool_calls import Function, Tool

    tools = [
        Tool(function=Function(name=tool.function.name, description="", parameters={}))
        for tool in TOOLS
    ]
    request = ChatCompletionRequest(
        messages=[UserMessage(content="Q?"), assistant, *after], tools=tools
    )
    ids = tokenizer.mistral.encode_chat_completion(request).tokens
    start = ids.index(tokenizer.tokenizer.get_control_token("[/INST]")) + 1
    return ids[start : ids.index(tokenizer.tokenizer.eos_id, start) + 1]


def _streamed_deltas(tokenizer, ids):
    """Text deltas as the executor streams them, one token at a time."""
    text, states, deltas = "", None, []
    for index, token in enumerate(ids):
        new_text, states = tokenizer.decode_incrementally(
            [token],
            prev_text=text,
            states=states,
            flush=index == len(ids) - 1,
            skip_special_tokens=False,
        )
        deltas.append(new_text[len(text) :])
        text = new_text
    return deltas


def _assistant_cases():
    from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
    from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage, UserMessage
    from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall

    calls = [
        ToolCall(
            id="abcDEF123",
            function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}'),
        ),
        ToolCall(
            id="xyzXYZ789",
            function=FunctionCall(
                name="get_current_weather",
                arguments='{"city": "巴黎", "note": "🦜 𠜎 «quoted» \\"x\\"", "days": [1, 2]}',
            ),
        ),
    ]
    results = [ToolMessage(content="sunny", tool_call_id=call.id) for call in calls]
    thanks = [UserMessage(content="Thanks")]
    return [
        (AssistantMessage(content="It is 4."), thanks, "", "It is 4.", []),
        (AssistantMessage(content=None, tool_calls=calls), results, "", "", calls),
        (
            AssistantMessage(content="Let me check.", tool_calls=calls[:1]),
            results[:1],
            "",
            "Let me check.",
            calls[:1],
        ),
        (
            AssistantMessage(content=[ThinkChunk(thinking="Add 2 and 2."), TextChunk(text="4.")]),
            thanks,
            "Add 2 and 2.",
            "4.",
            [],
        ),
        (
            AssistantMessage(content=[ThinkChunk(thinking="Need both.")], tool_calls=calls),
            results,
            "Need both.",
            "",
            calls,
        ),
    ]


def test_round_trip_through_the_native_tokenizer(tokenizer):
    split_bytes = False
    for assistant, after, reasoning, content, calls in _assistant_cases():
        ids = _assistant_ids(tokenizer, assistant, after)
        split_bytes |= any("�" in tokenizer.tokenizer.id_to_piece(token) for token in ids)
        whole = tokenizer.decode(ids, skip_special_tokens=False)
        deltas = _streamed_deltas(tokenizer, ids)
        # Streaming keeps the control tokens decode() keeps, and splits no character.
        assert "".join(deltas) == whole
        expected = [(call.function.name, json.loads(call.function.arguments)) for call in calls]
        for got in (_parse_whole(whole), _parse_stream(deltas)):
            assert got[0] == reasoning
            assert got[1] == content
            assert [(name, json.loads(arguments)) for name, arguments in got[2]] == expected
    assert split_bytes, "no case covers a character split across byte tokens"
