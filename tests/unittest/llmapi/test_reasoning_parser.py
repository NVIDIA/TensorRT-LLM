# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools
import json
import os
import random
from pathlib import Path

import pytest

from tensorrt_llm.llmapi.reasoning_parser import (
    MODEL_TYPE_TO_REASONING_PARSER, MuseSparkReasoningParser,
    NemotronV3ReasoningParser, ReasoningParserFactory,
    resolve_auto_reasoning_parser)

pytestmark = pytest.mark.cpu_only

R1_START, R1_END = "<think>", "</think>"


@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    ("a b", "", "a b"),
    (f"{R1_END} a b", " a b", ""),
    (f"a {R1_END} b", " b", "a "),
    (f"a b {R1_END}", "", "a b "),
    (f"{R1_START} a {R1_END} b", " b", f"{R1_START} a "),
])
def test_deepseek_r1_reasoning_parser(text: str, content: str,
                                      reasoning_context: str):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(("delta_texts", "content", "reasoning_context"), [
    (["a", "b"], ["", ""], ["a", "b"]),
    ([R1_END, "a", "b"], ["", "a", "b"], ["", "", ""]),
    (["a", R1_END, "b"], ["", "", "b"], ["a", "", ""]),
    (["a", "b", R1_END], ["", "", ""], ["a", "b", ""]),
    (["a", f"l{R1_END}", "b"], ["", "", "b"], ["a", "l", ""]),
    (["a", f"l{R1_END}r", "b"], ["", "r", "b"], ["a", "l", ""]),
    (["a", f"{R1_END}r", "b"], ["", "r", "b"], ["a", "", ""]),
])
def test_deepseek_r1_reasoning_parser_stream(delta_texts: list, content: list,
                                             reasoning_context: list):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(
    ("parser_key", "text"),
    [
        # `finish()` flushes as reasoning: the stream starts inside the
        # reasoning block, so the withheld `<` is reasoning output.
        ("deepseek-r1", "a <"),
        # `finish()` flushes as content: no reasoning block was entered.
        ("qwen3", "a<"),
        # `finish()` discards: the buffer holds exactly a delimiter. Passes on
        # `main` too - it guards against a fix that leaks the tag instead.
        ("deepseek-r1", f"a{R1_END}"),
    ])
def test_deepseek_r1_reasoning_parser_stream_matches_non_stream(
        parser_key: str, text: str) -> None:
    """Streaming char-by-char then finishing must match a non-streaming parse.

    One `(parser_key, text)` pair per branch of `finish()`. This is the
    contract the missing flush violated, so it subsumes example-based tests
    of the individual branches.
    """
    expected = ReasoningParserFactory.create_reasoning_parser(parser_key).parse(
        text)
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        parser_key)
    content, reasoning_context = "", ""
    for char in text:
        result = reasoning_parser.parse_delta(char)
        content += result.content
        reasoning_context += result.reasoning_content
    result = reasoning_parser.finish()
    content += result.content
    reasoning_context += result.reasoning_content
    assert content == expected.content
    assert reasoning_context == expected.reasoning_content


def test_deepseek_r1_reasoning_parser_finish_flushes_partial_tag() -> None:
    """A partial tag arriving alongside text must still be flushed.

    Such a delta fills `_buffer` through the `rfind` branch of `parse_delta`,
    which one-character-at-a-time streaming never reaches - and it is the
    shape a real stream delivers.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    assert reasoning_parser.parse_delta("a </thin").reasoning_content == "a "
    assert reasoning_parser.finish().reasoning_content == "</thin"


@pytest.mark.parametrize(("chat_template_kwargs", "flushed"), [
    ({
        "thinking": True
    }, "<"),
    ({
        "thinking": False
    }, ""),
])
def test_deepseek_v4_reasoning_parser_finish_delegates(
        chat_template_kwargs: dict[str, bool], flushed: str) -> None:
    """`finish()` must reach whichever parser the thinking flag selected.

    `DeepSeekV4ReasoningParser` delegates to two different targets:
    `DeepSeekR1Parser`, which now flushes, and `IdentityReasoningParser`,
    which withholds nothing to flush.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek_v4", chat_template_kwargs)
    reasoning_parser.parse_delta("a <")
    result = reasoning_parser.finish()
    assert result.reasoning_content == flushed
    assert result.content == ""


@pytest.mark.parametrize("chat_template_kwargs", [{
    "thinking": True
}, {
    "enable_thinking": True
}])
def test_deepseek_v4_reasoning_parser_extracts_when_thinking(
        chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek_v4", chat_template_kwargs)

    result = reasoning_parser.parse(f"hidden{R1_END}visible")

    assert result.content == "visible"
    assert result.reasoning_content == "hidden"


def test_deepseek_v4_reasoning_parser_streams_when_thinking():
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek_v4", {"enable_thinking": True})

    deltas = ["hid", f"den{R1_END}visible", " tail"]
    results = [reasoning_parser.parse_delta(delta) for delta in deltas]

    assert [result.content for result in results] == ["", "visible", " tail"]
    assert [result.reasoning_content
            for result in results] == ["hid", "den", ""]


TOOL_START = "<|tool_calls_section_begin|>"


@pytest.mark.parametrize(
    ("text", "content", "reasoning_context"),
    [
        # Standard <think>...</think> patterns.
        ("a<think>b</think>c", "c", "b"),
        ("<think>a</think>b", "b", "a"),
        ("<think>a", "", "a"),
        ("a", "a", ""),
        ("<think>", "", ""),
        # Interleaved thinking: tool call section implicitly ends reasoning.
        (f"<think>reasoning{TOOL_START}tool_call_data",
         f"{TOOL_START}tool_call_data", "reasoning"),
        # </think> before tool call section: standard end wins.
        (f"<think>reasoning</think>text{TOOL_START}tool_call_data",
         f"text{TOOL_START}tool_call_data", "reasoning"),
        # No <think> tag at all – just content.
        (f"content{TOOL_START}tool_call_data",
         f"content{TOOL_START}tool_call_data", ""),
    ])
def test_kimi_k2_reasoning_parser(text: str, content: str,
                                  reasoning_context: str):
    """Test kimi_k2 reasoning parser (non-streaming).

    Kimi-K2-Thinking generates <think>...</think> tags and may implicitly
    end reasoning via <|tool_calls_section_begin|>.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("kimi_k2")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_context"),
    [
        # Standard streaming cases (same as qwen3).
        (["<think>a", "l</think>r", "b"], ["", "r", "b"], ["a", "l", ""]),
        (["<th", "ink>a</think>b"], ["", "b"], ["", "a"]),
        (["<think>a</th", "ink>b"], ["", "b"], ["a", ""]),
        (["<think>", "a</think>b"], ["", "b"], ["", "a"]),
        (["<think>a</think>", "b"], ["", "b"], ["a", ""]),
        # Interleaved thinking: tool call section implicitly ends reasoning.
        # When the tool token arrives as a full token, the parser buffers it
        # (prefix check) and emits it combined with the next delta.
        (
            ["<think>", "reasoning", TOOL_START, "tool_data"],
            ["", "", "", TOOL_START + "tool_data"],
            ["", "reasoning", "", ""],
        ),
        # Tool section arrives combined with preceding reasoning text.
        (
            ["<think>", "reasoning" + TOOL_START + "tool_data"],
            ["", TOOL_START + "tool_data"],
            ["", "reasoning"],
        ),
        # Partial start-tag at end of delta should be buffered (not leaked).
        (
            ["content<th", "ink>reason</think>after"],
            ["content", "after"],
            ["", "reason"],
        ),
        # Partial tool section tag at end of content after </think>.
        (
            ["<think>reason</think>content<|tool", "_calls_section_begin|>td"],
            ["content", TOOL_START + "td"],
            ["reason", ""],
        ),
    ])
def test_kimi_k2_reasoning_parser_stream(delta_texts: list, content: list,
                                         reasoning_context: list):
    """Test kimi_k2 reasoning parser streaming."""
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("kimi_k2")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i], \
            f"Step {i}: delta={delta_text!r}, expected content={content[i]!r}, got {result.content!r}"
        assert result.reasoning_content == reasoning_context[i], \
            f"Step {i}: delta={delta_text!r}, expected reasoning={reasoning_context[i]!r}, got {result.reasoning_content!r}"


@pytest.mark.parametrize(
    ("parser_name", "delta_texts", "content", "reasoning_context"),
    [
        # Interleaved thinking with reasoning_at_start=True (deepseek-r1, minimax_m2):
        # Model output starts directly with reasoning (no <think> tag).
        # Simulates: reason1</think>text1<think>reason2</think>text2
        (
            "minimax_m2",
            ["reason1", R1_END, "text1", R1_START, "reason2", R1_END, "text2"],
            ["", "", "text1", "", "", "", "text2"],
            ["reason1", "", "", "", "reason2", "", ""],
        ),
        (
            "deepseek-r1",
            ["reason1", R1_END, "text1", R1_START, "reason2", R1_END, "text2"],
            ["", "", "text1", "", "", "", "text2"],
            ["reason1", "", "", "", "reason2", "", ""],
        ),
        # Interleaved thinking with reasoning_at_start=False (qwen3, kimi_k2):
        # Model output contains <think>...</think> tags.
        # Simulates: <think>reason1</think>content1<think>reason2</think>content2
        (
            "qwen3",
            [
                R1_START, "reason1", R1_END, "content1", R1_START, "reason2",
                R1_END, "content2"
            ],
            ["", "", "", "content1", "", "", "", "content2"],
            ["", "reason1", "", "", "", "reason2", "", ""],
        ),
        (
            "kimi_k2",
            [
                R1_START, "reason1", R1_END, "content1", R1_START, "reason2",
                R1_END, "content2"
            ],
            ["", "", "", "content1", "", "", "", "content2"],
            ["", "reason1", "", "", "", "reason2", "", ""],
        ),
        # Kimi-K2 interleaved thinking: reasoning interrupted by tool calls.
        # Simulates: <think>reasoning<|tool_calls_section_begin|>tool_data
        # Note: when TOOL_START arrives as a full token, the parser buffers it
        # (prefix check) and emits it combined with the next delta.
        (
            "kimi_k2",
            [R1_START, "reasoning", TOOL_START, "tool_data"],
            ["", "", "", TOOL_START + "tool_data"],
            ["", "reasoning", "", ""],
        ),
    ],
)
def test_interleaved_thinking_stream(parser_name: str, delta_texts: list,
                                     content: list, reasoning_context: list):
    """Test that streaming parsers correctly handle interleaved thinking.

    Interleaved thinking allows models to reason between tool calls,
    producing multiple <think>...</think> blocks within a single generation.
    The streaming parser must correctly transition between reasoning and
    content modes across multiple think blocks.

    For kimi_k2, reasoning may also be implicitly ended by the tool call
    section token <|tool_calls_section_begin|> without an explicit </think>.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        parser_name)
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i], \
            f"Step {i}: delta={delta_text!r}, expected content={content[i]!r}, got {result.content!r}"
        assert result.reasoning_content == reasoning_context[i], \
            f"Step {i}: delta={delta_text!r}, expected reasoning={reasoning_context[i]!r}, got {result.reasoning_content!r}"


@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    ("a<think>b</think>c", "c", "b"),
    ("<think>a</think>b", "b", "a"),
    ("<think>a", "", "a"),
    ("a", "a", ""),
    ("<think>", "", ""),
])
def test_qwen3_reasoning_parser(text: str, content: str,
                                reasoning_context: str):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(("delta_texts", "content", "reasoning_context"), [
    (["<think>a", "l</think>r", "b"], ["", "r", "b"], ["a", "l", ""]),
    (["<th", "ink>a</think>b"], ["", "b"], ["", "a"]),
    (["<think>a</th", "ink>b"], ["", "b"], ["a", ""]),
    (["<think>", "a</think>b"], ["", "b"], ["", "a"]),
    (["<think>a</think>", "b"], ["", "b"], ["a", ""]),
    (["<think>a</th", "ank></th", "ink>b"], ["", "", "b"
                                             ], ["a", "</thank>", ""]),
])
def test_qwen3_reasoning_parser_stream(delta_texts: list, content: list,
                                       reasoning_context: list):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    (f"hidden{R1_END}visible", "visible", "hidden"),
    (f"{R1_END}visible", "visible", ""),
    (R1_END, "", ""),
    ("unterminated", "", "unterminated"),
])
def test_poolside_v1_reasoning_parser_when_thinking(
        text: str, content: str, reasoning_context: str) -> None:
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", {"enable_thinking": True})
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize("chat_template_kwargs", [
    {
        "enable_thinking": False
    },
    {
        "thinking": False
    },
])
def test_poolside_v1_reasoning_parser_when_not_thinking(
        chat_template_kwargs: dict[str, bool]) -> None:
    """Output is all visible content when thinking is off.

    The template emits `</think>` into the prompt in that mode, so the model
    output carries no markers at all.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", chat_template_kwargs)
    result = reasoning_parser.parse("visible")
    assert result.content == "visible"
    assert result.reasoning_content == ""


@pytest.mark.parametrize(("delta_texts", "content", "reasoning_context"), [
    (["a", f"l{R1_END}r", "b"], ["", "r", "b"], ["a", "l", ""]),
    (["a</th", "ink>b"], ["", "b"], ["a", ""]),
    (["", f"a{R1_END}b"], ["", "b"], ["", "a"]),
    ([f"a{R1_END}", "b"], ["", "b"], ["a", ""]),
    (["a</th", "ank></th", "ink>b"], ["", "", "b"], ["a", "</thank>", ""]),
])
def test_poolside_v1_reasoning_parser_stream(
        delta_texts: list[str], content: list[str],
        reasoning_context: list[str]) -> None:
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", {"enable_thinking": True})
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


def test_laguna_alias_of_poolside_v1() -> None:
    """`laguna` is kept as an alias so existing deployments keep working."""
    laguna = ReasoningParserFactory.create_reasoning_parser("laguna")
    poolside = ReasoningParserFactory.create_reasoning_parser("poolside_v1")
    assert type(laguna) is type(poolside)
    assert MODEL_TYPE_TO_REASONING_PARSER["laguna"] == "poolside_v1"


@pytest.mark.parametrize("parser", ["poolside_v1", "laguna"])
@pytest.mark.parametrize(("tail", "expected"), [(R1_START, True),
                                                (R1_END, False)])
def test_alias_resolves_identically(parser: str, tail: str,
                                    expected: bool) -> None:
    assert ReasoningParserFactory.resolve_prefilled_thinking(
        parser, f"<assistant>{tail}") is expected


@pytest.mark.parametrize(("prompt", "expected"), [
    (f"<user>hi</user>\n<assistant>{R1_START}", True),
    (f"<user>hi</user>\n<assistant>{R1_END}", False),
    (f"<user>hi</user>\n<assistant>\n{R1_END}", False),
    (f"<user>hi</user>\n<assistant>{R1_START}\n", True),
    ("<user>hi</user>\n<assistant>", None),
    (f"a prompt quoting {R1_END} mid-turn\n<assistant>", None),
])
def test_resolve_prefilled_thinking(prompt: str, expected: bool | None) -> None:
    assert ReasoningParserFactory.resolve_prefilled_thinking(
        "poolside_v1", prompt) is expected


def test_resolve_prefilled_thinking_unknown_parser() -> None:
    assert ReasoningParserFactory.resolve_prefilled_thinking(
        "not-a-parser", R1_START) is None


@pytest.mark.parametrize(("tail", "expected"), [(R1_START, True),
                                                (R1_END, False)])
def test_resolve_prefilled_thinking_opted_in(tail: str, expected: bool) -> None:
    assert ReasoningParserFactory.resolve_prefilled_thinking(
        "poolside_v1", f"<assistant>{tail}") is expected


@pytest.mark.parametrize("parser", ["poolside_v1", "laguna"])
@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    ("a<think>b</think>c", "c", "b"),
    ("<think>a</think>b", "b", "a"),
    ("<think>a", "", "a"),
    ("a", "a", ""),
])
def test_unresolved_mode_keeps_previous_behaviour(
        parser: str, text: str, content: str, reasoning_context: str) -> None:
    """With no mode supplied, split on a `<think>` the model emitted itself.

    Nothing resolves the mode for the offline LLM API, the disaggregated
    generation server, or `add_generation_prompt=false`. Falling back to
    `IdentityReasoningParser` there would drop the split these models still
    need, since they emit the tags in multi-turn and tool-calling flows.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(parser)

    result = reasoning_parser.parse(text)

    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize("kwargs", [{
    "enable_thinking": False
}, {
    "thinking": False
}])
def test_explicit_no_thinking_still_uses_identity(
        kwargs: dict[str, bool]) -> None:
    """An explicit off is a resolved mode, so the template closed reasoning."""
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", kwargs)

    result = reasoning_parser.parse("a<think>b</think>c")

    assert result.content == "a<think>b</think>c"
    assert result.reasoning_content == ""


def test_clearing_only_enable_thinking_leaves_reasoning_on() -> None:
    """Why the server writes both keys: the parser ORs them.

    If it cleared only `enable_thinking`, a `thinking=True` the caller sent
    would still force reasoning mode and the answer would land in the wrong
    field. This pins the OR so that shortcut cannot be taken silently.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", {
            "thinking": True,
            "enable_thinking": False
        })

    result = reasoning_parser.parse(f"hidden{R1_END}visible")

    assert result.reasoning_content == "hidden"
    assert result.content == "visible"


@pytest.mark.parametrize("sent", [{
    "thinking": True
}, {
    "enable_thinking": True
}, {
    "thinking": True,
    "enable_thinking": True
}])
def test_resolved_mode_overrides_whatever_the_caller_sent(
        sent: dict[str, bool]) -> None:
    """Both keys written, as the server does, so the resolved mode wins."""
    resolved = {**sent, "thinking": False, "enable_thinking": False}
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", resolved)

    result = reasoning_parser.parse("visible")

    assert result.content == "visible"
    assert result.reasoning_content == ""


@pytest.mark.parametrize("parser", [
    "deepseek-r1", "deepseek_v4", "qwen3", "qwen3_5", "minimax_m2",
    "minimax_m3", "nemotron-v3", "nano-v3", "gemma4", "kimi_k2", "kimi_k25"
])
def test_resolve_prefilled_thinking_requires_opt_in(parser: str) -> None:
    """Parsers that have not opted in must never be resolved from the prompt.

    `deepseek_v4` shares the base class and `nemotron-v3` / `nano-v3` also read
    `enable_thinking`, so without the flag they would silently pick up a mode
    the server inferred.
    """
    # Otherwise a typo or a dropped registration passes vacuously, since an
    # unknown name also resolves to None.
    assert parser in ReasoningParserFactory.keys()
    for tail in (R1_START, R1_END, ""):
        assert ReasoningParserFactory.resolve_prefilled_thinking(
            parser, f"<assistant>{tail}") is None


@pytest.mark.parametrize(
    ("prompt_tail", "model_output", "content", "reasoning_content"), [
        (R1_START, f"hidden{R1_END}visible", "visible", "hidden"),
        (R1_END, "visible", "visible", ""),
    ])
def test_poolside_v1_mode_resolved_from_prompt(prompt_tail: str,
                                               model_output: str, content: str,
                                               reasoning_content: str) -> None:
    """Mirror the server path: resolve from the prompt, then parse.

    A request that sends no chat template kwargs must still land in the mode
    the template actually rendered.
    """
    prompt = f"<user>hi</user>\n<assistant>{prompt_tail}"
    thinking = ReasoningParserFactory.resolve_prefilled_thinking(
        "poolside_v1", prompt)
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", {"enable_thinking": thinking})
    result = reasoning_parser.parse(model_output)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


TOOL_CALL = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


@pytest.mark.parametrize(
    ("text", "content", "reasoning_context", "chat_template_kwargs"),
    [
        ("a b", "", "a b", None),
        (f"{R1_END} a b", " a b", "", None),
        (f"a {R1_END} b", " b", "a ", None),
        (f"a b {R1_END}", "", "a b ", None),
        (f"{R1_START} a {R1_END} b", " b", f"{R1_START} a ", None),
        # All without reasoning_context.
        ("a b", "a b", "", {
            "enable_thinking": False
        }),
        (f"{R1_END} a b", f"{R1_END} a b", "", {
            "enable_thinking": False
        }),
        (f"a {R1_END} b", f"a {R1_END} b", "", {
            "enable_thinking": False
        }),
        (f"a b {R1_END}", f"a b {R1_END}", "", {
            "enable_thinking": False
        }),
        # force_nonempty_content swaps reasoning into content when content is
        # empty (reasoning_at_start stays True, so parsing is unchanged).
        ("a b", "a b", "", {
            "force_nonempty_content": True
        }),
        (f"a {R1_END} b", " b", "a ", {
            "force_nonempty_content": True
        }),
        (f"a b {R1_END}", "a b ", "", {
            "force_nonempty_content": True
        }),
        # NVBug 6060281: whitespace-only content after </redacted_thinking> must
        # still trigger the reasoning-to-content swap when force_nonempty_content.
        (f"a {R1_END}\n", "a ", "", {
            "force_nonempty_content": True
        }),
        (f"a {R1_END} \t ", "a ", "", {
            "force_nonempty_content": True
        }),
        # NVBug 6082303: <tool_call> as implicit end-of-reasoning.
        # No </think> before <tool_call> — tool call must appear in content.
        (f"I need weather{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"{TOOL_CALL}get_weather{TOOL_CALL_END}", "I need weather", None),
        # </think> before <tool_call> — standard end wins.
        (f"reasoning{R1_END}{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"{TOOL_CALL}get_weather{TOOL_CALL_END}", "reasoning", None),
        # <tool_call> with enable_thinking=False — all goes to content.
        (f"content{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"content{TOOL_CALL}get_weather{TOOL_CALL_END}", "", {
             "enable_thinking": False
         }),
        # <tool_call> with force_nonempty_content — content is non-empty, no swap.
        (f"reasoning{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"{TOOL_CALL}get_weather{TOOL_CALL_END}", "reasoning", {
             "force_nonempty_content": True
         }),
        # <think> then <tool_call> without </think> with enable_thinking=False:
        # reasoning_at_start is False so parse() looks for <think> tag,
        # strips it, then tool_call acts as implicit end-of-reasoning.
        (f"{R1_START}reasoning{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"{TOOL_CALL}get_weather{TOOL_CALL_END}", "reasoning", {
             "enable_thinking": False
         }),
    ])
def test_nano_v3_reasoning_parser(text: str, content: str,
                                  reasoning_context: str,
                                  chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    result = reasoning_parser.parse(text)
    print(f"text: {text}, result: {result}")
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_context", "chat_template_kwargs"),
    [
        (["a", "b"], ["", ""], ["a", "b"], None),
        ([R1_END, "a", "b"], ["", "a", "b"], ["", "", ""], None),
        (["a", R1_END, "b"], ["", "", "b"], ["a", "", ""], None),
        (["a", "b", R1_END], ["", "", ""], ["a", "b", ""], None),
        (["a", f"l{R1_END}", "b"], ["", "", "b"], ["a", "l", ""], None),
        (["a", f"l{R1_END}r", "b"], ["", "r", "b"], ["a", "l", ""], None),
        (["a", f"{R1_END}r", "b"], ["", "r", "b"], ["a", "", ""], None),
        # All without reasoning_context.
        (["a", "b"], ["a", "b"], ["", ""], {
            "enable_thinking": False
        }),
        ([R1_END, "a", "b"], ["", f"{R1_END}a", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", R1_END, "b"], ["a", "", f"{R1_END}b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", "b", R1_END], ["a", "b", ""], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"l{R1_END}", "b"], ["a", f"l{R1_END}", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"l{R1_END}r", "b"], ["a", f"l{R1_END}r", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"{R1_END}r", "b"], ["a", f"{R1_END}r", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        # NVBug 6082303: <tool_call> as implicit end-of-reasoning in streaming.
        # Single-token <tool_call> after reasoning deltas.
        (
            ["I need ", "weather", TOOL_CALL, "get_weather"],
            ["", "", f"{TOOL_CALL}", "get_weather"],
            ["I need ", "weather", "", ""],
            None,
        ),
        # <tool_call> combined with preceding reasoning text in one delta.
        (
            ["reasoning" + TOOL_CALL + "tool_data"],
            [TOOL_CALL + "tool_data"],
            ["reasoning"],
            None,
        ),
        # </think> before <tool_call> — standard handling.
        (
            ["reasoning", R1_END, TOOL_CALL + "data"],
            ["", "", TOOL_CALL + "data"],
            ["reasoning", "", ""],
            None,
        ),
        # Token-level chunks mimicking the bug scenario from NVBug 6082303.
        (
            [
                "I", " need to check", " the weather in", " New York City",
                TOOL_CALL, "<function=get_weather>", "<parameter=location>NYC",
                "</parameter>", "</function>", TOOL_CALL_END
            ],
            [
                "", "", "", "", TOOL_CALL, "<function=get_weather>",
                "<parameter=location>NYC", "</parameter>", "</function>",
                TOOL_CALL_END
            ],
            [
                "I", " need to check", " the weather in", " New York City", "",
                "", "", "", "", ""
            ],
            None,
        ),
        # Parent buffers trailing "<" (prefix of "</think>") from a text
        # token, then the full <tool_call> special token arrives atomically.
        (
            ["reasoning<", "<tool_call>data"],
            ["", "<tool_call>data"],
            ["reasoning", "<"],
            None,
        ),
        # force_nonempty_content + streaming tool call: tool_call ends
        # reasoning, accumulated reasoning is discarded (not swapped into
        # content), and content carries the tool markup.
        (
            ["I need ", "weather", TOOL_CALL, "get_weather"],
            ["", "", TOOL_CALL, "get_weather"],
            ["I need ", "weather", "", ""],
            {
                "force_nonempty_content": True
            },
        ),
        # enable_thinking=False + streaming tool call: everything is content
        # (no buffering since <tool_call> is not a prefix of <think>).
        (
            ["content", TOOL_CALL, "data"],
            ["content", TOOL_CALL, "data"],
            ["", "", ""],
            {
                "enable_thinking": False
            },
        ),
    ])
def test_nano_v3_reasoning_parser_stream(delta_texts: list, content: list,
                                         reasoning_context: list,
                                         chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        print(f"delta_text: {delta_text}, result: {result}")
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(
    ("delta_texts", "finish_content", "finish_reasoning",
     "chat_template_kwargs"),
    [
        (["a", "b"], "", "", None),
        ([R1_END, "a", "b"], "", "", None),
        (["a", R1_END, "b"], "", "", None),
        (["a", "b"], "", "", {
            "enable_thinking": False
        }),
        ([f"{R1_START}a", "b"], "", "", {
            "enable_thinking": False
        }),
        (["a", "b"], "", "", {
            "force_nonempty_content": False
        }),
        (["a", "b"], "ab", "", {
            "force_nonempty_content": True
        }),
        ([R1_END, "a", "b"], "", "", {
            "force_nonempty_content": True
        }),
        # NVBug 6082303: <tool_call> ends reasoning,
        # finish should return empty (tag acted as
        # implicit closing).
        (["reasoning", "<tool_call>data"], "", "", None),
        (["reasoning", "<tool_call>data"], "", "", {
            "force_nonempty_content": True
        }),
    ])
def test_nano_v3_reasoning_parser_finish(delta_texts: list, finish_content: str,
                                         finish_reasoning: str,
                                         chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    for delta_text in delta_texts:
        reasoning_parser.parse_delta(delta_text)
    result = reasoning_parser.finish()
    assert result.content == finish_content
    assert result.reasoning_content == finish_reasoning


# ---------------------------------------------------------------------------
# Auto-detection tests for resolve_auto_reasoning_parser
# ---------------------------------------------------------------------------


def _write_config(model_dir: str, model_type: str):
    """Write a minimal config.json with the given model_type."""
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"model_type": model_type}, f)


def _write_tokenizer_config(model_dir: str, chat_template: str):
    """Write a minimal tokenizer_config.json with the given chat_template."""
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"chat_template": chat_template}, f)


# Hybrid Qwen3: chat template contains "enable_thinking" → use "qwen3" parser
_HYBRID_TEMPLATE = (
    "{%- if enable_thinking is not defined %}{% set enable_thinking = true %}"
    "{% endif %}{%- if add_generation_prompt %}{%- if enable_thinking %}"
    "{{- '<|im_start|>assistant\\n<think>\\n' }}{%- else %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}{%- endif %}")

# Forced-thinking Qwen3: no "enable_thinking" but has "<think>" → "deepseek-r1"
_FORCED_THINKING_TEMPLATE = ("{%- if add_generation_prompt %}"
                             "{{- '<|im_start|>assistant\\n<think>\\n' }}"
                             "{%- endif %}")

# Forced-non-thinking Qwen3: no "enable_thinking" and no "<think>" → None
_FORCED_NON_THINKING_TEMPLATE = ("{%- if add_generation_prompt %}"
                                 "{{- '<|im_start|>assistant\\n' }}"
                                 "{%- endif %}")


def test_auto_detect_qwen3_hybrid(tmp_path):
    """Hybrid Qwen3 model with enable_thinking toggle → 'qwen3' parser."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _HYBRID_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "qwen3"


def test_auto_detect_qwen3_forced_thinking(tmp_path):
    """Forced-thinking Qwen3 model → 'deepseek-r1' parser."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B-Thinking-2507")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _FORCED_THINKING_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "deepseek-r1"


def test_auto_detect_qwen3_forced_non_thinking(tmp_path):
    """Forced-non-thinking Qwen3 model → None (no parser needed)."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B-Instruct-2507")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _FORCED_NON_THINKING_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_qwen3_no_tokenizer_config(tmp_path):
    """Qwen3 model without tokenizer_config.json → falls back to 'qwen3'."""
    model_dir = str(tmp_path / "Qwen3-SomeModel")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "qwen3"


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("glm_moe_dsa", "glm_moe_dsa"),  # GLM-5, GLM-5.2
        # GLM-4.5/4.6 and GLM-4.7 share glm4_moe but differ on whether the
        # template prefills <think>; GLM-5.3-Flash prefills it whatever
        # enable_thinking says. Neither is resolved automatically.
        ("glm4_moe", None),
        ("glm5_next", None),
    ])
def test_auto_detect_glm(tmp_path, model_type, expected):
    model_dir = str(tmp_path / "GLM")
    os.makedirs(model_dir)
    _write_config(model_dir, model_type)

    assert resolve_auto_reasoning_parser(model_dir) == expected


def test_auto_detect_deepseek_r1(tmp_path):
    """DeepSeek R1 model → 'deepseek-r1' parser."""
    model_dir = str(tmp_path / "DeepSeek-R1")
    os.makedirs(model_dir)
    _write_config(model_dir, "deepseek_v3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "deepseek-r1"


def test_auto_detect_deepseek_non_r1(tmp_path):
    """DeepSeek non-R1 model → None."""
    model_dir = str(tmp_path / "DeepSeek-V3")
    os.makedirs(model_dir)
    _write_config(model_dir, "deepseek_v3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_unknown_model(tmp_path):
    """Unknown model type → None."""
    model_dir = str(tmp_path / "SomeUnknownModel")
    os.makedirs(model_dir)
    _write_config(model_dir, "unknown_type")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_no_config(tmp_path):
    """No config.json → None."""
    model_dir = str(tmp_path / "EmptyDir")
    os.makedirs(model_dir)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_gemma4(tmp_path):
    """Gemma 4 model → 'gemma4' parser."""
    model_dir = str(tmp_path / "gemma-4-26B-A4B-it")
    os.makedirs(model_dir)
    _write_config(model_dir, "gemma4")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "gemma4"


def test_auto_detect_laguna(tmp_path):
    """Laguna model → 'poolside_v1' parser."""
    model_dir = str(tmp_path / "Laguna")
    os.makedirs(model_dir)
    _write_config(model_dir, "laguna")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "poolside_v1"


def test_auto_detect_exaone_moe(tmp_path: Path) -> None:
    model_dir = str(tmp_path / "K-EXAONE")
    os.makedirs(model_dir)
    _write_config(model_dir, "exaone_moe")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "k-exaone"

    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "k-exaone")
    assert isinstance(reasoning_parser, NemotronV3ReasoningParser)


@pytest.mark.parametrize("model_type",
                         ["nemotron_h", "nemotron_h_puzzle", "nemotron_h_omni"])
def test_auto_detect_nemotron_h(tmp_path, model_type):
    """Nemotron-H models → 'nemotron-v3' parser (preferred over 'nano-v3').

    `nemotron_h_omni` (Nemotron 3.5 Super VL) has a `model_type` that diverges
    from its architecture string, and the lookup is exact-match, so without its
    own row `--reasoning_parser auto` resolves to None.
    """
    model_dir = str(tmp_path / model_type)
    os.makedirs(model_dir)
    _write_config(model_dir, model_type)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "nemotron-v3"


def test_nemotron_v3_alias_same_parser():
    """'nemotron-v3' and the legacy 'nano-v3' resolve to the same parser."""
    nemotron = ReasoningParserFactory.create_reasoning_parser("nemotron-v3")
    nano = ReasoningParserFactory.create_reasoning_parser("nano-v3")
    assert isinstance(nemotron, NemotronV3ReasoningParser)
    assert isinstance(nano, NemotronV3ReasoningParser)


# ---------------------------------------------------------------------------
# Gemma 4 reasoning parser tests
# ---------------------------------------------------------------------------

G4_OPEN, G4_CLOSE = "<|channel>", "<channel|>"


@pytest.mark.parametrize(
    ("text", "content", "reasoning_content"),
    [
        # No reasoning block: everything is content.
        ("hello world", "hello world", ""),
        # Channel block wrapping reasoning, followed by content.
        (f"{G4_OPEN}thought\nreasoning{G4_CLOSE}answer", "answer",
         "thought\nreasoning"),
        # Content before and after the reasoning block.
        (f"pre{G4_OPEN}r{G4_CLOSE}post", "prepost", "r"),
        # Unterminated channel: remainder treated as reasoning.
        (f"{G4_OPEN}abc", "", "abc"),
        # Multiple interleaved channel blocks.
        (f"a{G4_OPEN}r1{G4_CLOSE}b{G4_OPEN}r2{G4_CLOSE}c", "abc", "r1r2"),
        # Empty reasoning block (e.g. prefilled when enable_thinking=False).
        (f"{G4_OPEN}thought\n{G4_CLOSE}answer", "answer", "thought\n"),
    ],
)
def test_gemma4_reasoning_parser(text: str, content: str,
                                 reasoning_content: str):
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    result = parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_content"),
    [
        # No reasoning: plain content streams through.
        (["a", "b"], ["a", "b"], ["", ""]),
        # Open and close in a single delta.
        ([f"{G4_OPEN}r{G4_CLOSE}c"], ["c"], ["r"]),
        # Delimiters split across deltas.
        ([G4_OPEN, "r", G4_CLOSE, "c"], ["", "", "", "c"], ["", "r", "", ""]),
        # Partial open tag held back until complete.
        (["pre<|cha", "nnel>r<chan", "nel|>post"], ["pre", "", "post"
                                                    ], ["", "r", ""]),
        # Two reasoning blocks interleaved with content.
        (
            [f"{G4_OPEN}r1{G4_CLOSE}c1", f"{G4_OPEN}r2{G4_CLOSE}c2"],
            ["c1", "c2"],
            ["r1", "r2"],
        ),
        # Partial close tag at end of delta buffered.
        ([f"{G4_OPEN}reason<chan", "nel|>tail"], ["", "tail"], ["reason", ""]),
    ],
)
def test_gemma4_reasoning_parser_stream(delta_texts: list, content: list,
                                        reasoning_content: list):
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    for i, delta in enumerate(delta_texts):
        result = parser.parse_delta(delta)
        assert result.content == content[i], (
            f"Step {i}: delta={delta!r} expected content={content[i]!r} "
            f"got {result.content!r}")
        assert result.reasoning_content == reasoning_content[i], (
            f"Step {i}: delta={delta!r} expected reasoning="
            f"{reasoning_content[i]!r} got {result.reasoning_content!r}")


def test_gemma4_reasoning_parser_finish_flushes_buffer():
    """finish() should flush any buffered trailing text."""
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    # Send a partial open tag; parser holds it back.
    parser.parse_delta("some text<|cha")
    # Stream ended mid-tag: the held-back suffix flushes as content.
    result = parser.finish()
    assert result.content == "<|cha"
    assert result.reasoning_content == ""


def test_gemma4_reasoning_parser_finish_unterminated_reasoning():
    """Verify finish() flushes a held-back partial close tag as reasoning.

    When the stream ends mid-channel with a buffered partial close tag, the
    remainder should surface as reasoning content.
    """
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    # Enter reasoning; stream ends with a partial close tag (held back).
    stream = parser.parse_delta(f"{G4_OPEN}reasoning_start<chan")
    assert stream.reasoning_content == "reasoning_start"
    assert stream.content == ""
    # finish() should release the buffered "<chan" as reasoning since we are
    # still inside the channel block.
    result = parser.finish()
    assert result.content == ""
    assert result.reasoning_content == "<chan"


# ---------------------------------------------------------------------------
# Kimi K3 reasoning parser tests
#
# Fixture strings are transcribed from the checkpoint's `encoding_k3.py`
# rendering (the authoritative XTML chat template): tags render as
# `<|open|>tag key="value"<|sep|>` / `<|close|>tag<|sep|>` with no
# whitespace between segments, and the generation prompt ends inside
# `<|open|>think<|sep|>` (or `<|open|>response<|sep|>` when thinking=False),
# so completions start mid-channel and end with
# `<|close|>message<|sep|><|end_of_msg|>`.
# ---------------------------------------------------------------------------

K3_OPEN, K3_CLOSE, K3_SEP, K3_EOM = ("<|open|>", "<|close|>", "<|sep|>",
                                     "<|end_of_msg|>")

# One get_weather call, exactly as encoding_k3._render_assistant_segments
# renders it (attributes space-prefixed, index 1-based, string args raw).
K3_TOOLS_SECTION = (f'{K3_OPEN}tools{K3_SEP}'
                    f'{K3_OPEN}call tool="get_weather" index="1"{K3_SEP}'
                    f'{K3_OPEN}argument key="location" type="string"{K3_SEP}'
                    f'NYC'
                    f'{K3_CLOSE}argument{K3_SEP}'
                    f'{K3_CLOSE}call{K3_SEP}'
                    f'{K3_CLOSE}tools{K3_SEP}')

K3_MSG_END = f"{K3_CLOSE}message{K3_SEP}{K3_EOM}"


def _k3_completion(reasoning: str,
                   content: str,
                   tools_section: str = "",
                   terminated: bool = True) -> str:
    """A thinking-mode completion (prompt already opened the think channel)."""
    text = (f"{reasoning}{K3_CLOSE}think{K3_SEP}"
            f"{K3_OPEN}response{K3_SEP}{content}")
    if terminated:
        text += f"{K3_CLOSE}response{K3_SEP}{tools_section}{K3_MSG_END}"
    return text


@pytest.mark.parametrize(
    ("text", "content", "reasoning_content"),
    [
        # Fully terminated think + response message.
        (_k3_completion("step by step", "The answer is 4."), "The answer is 4.",
         "step by step"),
        # Tool-calling message: the tools section passes through into content
        # verbatim so the kimi_k3 tool parser can consume it downstream.
        (_k3_completion("pick a tool", "Checking.", K3_TOOLS_SECTION),
         "Checking." + K3_TOOLS_SECTION, "pick a tool"),
        # Length-capped mid-think: everything is reasoning.
        ("unterminated reasoning", "", "unterminated reasoning"),
        # Length-capped mid-response: finish() flushes the response tail.
        (_k3_completion("r", "partial resp",
                        terminated=False), "partial resp", "r"),
        # Empty think body.
        (_k3_completion("", "only content"), "only content", ""),
    ])
def test_kimi_k3_reasoning_parser(text: str, content: str,
                                  reasoning_content: str):
    parser = ReasoningParserFactory.create_reasoning_parser("kimi_k3")
    result = parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


@pytest.mark.parametrize(
    ("text", "content", "reasoning_content"),
    [
        # thinking=False: prompt opened the response channel directly.
        (f"plain answer{K3_CLOSE}response{K3_SEP}{K3_MSG_END}", "plain answer",
         ""),
        (f"answer{K3_CLOSE}response{K3_SEP}{K3_TOOLS_SECTION}{K3_MSG_END}",
         "answer" + K3_TOOLS_SECTION, ""),
    ])
def test_kimi_k3_reasoning_parser_non_thinking(text: str, content: str,
                                               reasoning_content: str):
    parser = ReasoningParserFactory.create_reasoning_parser(
        "kimi_k3", {"thinking": False})
    result = parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_content"),
    [
        # Plain reasoning streams straight through.
        (["a", "b"], ["", ""], ["a", "b"]),
        # Channel switch split across deltas mid-marker.
        (
            [
                "rea",
                f"son{K3_CLOSE}thi",  # codespell:ignore thi
                f"nk{K3_SEP}{K3_OPEN}response{K3_SEP}c",
                "d"
            ],
            ["", "", "c", "d"],
            ["rea", "son", "", ""]),
        # A partial marker at the end of a delta is held back, then released
        # as reasoning once it turns out not to be a marker.
        (["a<|clo", "x"], ["", ""], ["a", "<|clox"]),
        # Structural close/open pair arriving as one delta.
        ([f"r{K3_CLOSE}think{K3_SEP}{K3_OPEN}response{K3_SEP}c"], ["c"], ["r"]),
        # tools_pass: `<|close|>tools<|sep|>` has an internal `<`, so the
        # suffix hold must consider mid-marker splits like `...<|close|>to`.
        ([
            f"r{K3_CLOSE}think{K3_SEP}{K3_OPEN}response{K3_SEP}",
            f"{K3_OPEN}tools{K3_SEP}CALL{K3_CLOSE}to",
            f"ols{K3_SEP}{K3_MSG_END}",
        ], [
            "",
            f"{K3_OPEN}tools{K3_SEP}CALL",
            f"{K3_CLOSE}tools{K3_SEP}",
        ], ["r", "", ""]),
        # Message terminator split across deltas produces no output.
        ([
            f"r{K3_CLOSE}think{K3_SEP}{K3_OPEN}response{K3_SEP}c",
            "<|close|>mes", f"sage{K3_SEP}{K3_EOM}"
        ], ["c", "", ""], ["r", "", ""]),
    ])
def test_kimi_k3_reasoning_parser_stream(delta_texts: list, content: list,
                                         reasoning_content: list):
    parser = ReasoningParserFactory.create_reasoning_parser("kimi_k3")
    for i, delta_text in enumerate(delta_texts):
        result = parser.parse_delta(delta_text)
        assert result.content == content[i], \
            f"Step {i}: delta={delta_text!r}, expected content={content[i]!r}, got {result.content!r}"
        assert result.reasoning_content == reasoning_content[i], \
            f"Step {i}: delta={delta_text!r}, expected reasoning={reasoning_content[i]!r}, got {result.reasoning_content!r}"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7])
@pytest.mark.parametrize("thinking", [True, False])
def test_kimi_k3_reasoning_parser_stream_matches_parse(chunk_size: int,
                                                       thinking: bool):
    """Streaming in arbitrary chunkings must reproduce the one-shot parse.

    This sweeps every marker-split position, which is the riskiest logic in
    the parser (`_trailing_partial_marker` suffix holds).
    """
    if thinking:
        text = _k3_completion("Let me think.", "Answer: 4.", K3_TOOLS_SECTION)
    else:
        text = (f"Answer: 4.{K3_CLOSE}response{K3_SEP}{K3_TOOLS_SECTION}"
                f"{K3_MSG_END}")
    kwargs = None if thinking else {"thinking": False}

    oneshot = ReasoningParserFactory.create_reasoning_parser("kimi_k3",
                                                             kwargs).parse(text)

    streamer = ReasoningParserFactory.create_reasoning_parser("kimi_k3", kwargs)
    content, reasoning = [], []
    for start in range(0, len(text), chunk_size):
        result = streamer.parse_delta(text[start:start + chunk_size])
        content.append(result.content)
        reasoning.append(result.reasoning_content)
    tail = streamer.finish()
    content.append(tail.content)
    reasoning.append(tail.reasoning_content)

    assert "".join(content) == oneshot.content
    assert "".join(reasoning) == oneshot.reasoning_content


def test_kimi_k3_needs_raw_special_tokens():
    """The K3 delimiters are special tokens.

    The serving layer keys off this flag to disable skip_special_tokens for
    the request.
    """
    assert ReasoningParserFactory.needs_raw_special_tokens("kimi_k3") is True
    assert ReasoningParserFactory.needs_raw_special_tokens("KIMI_K3") is True
    assert ReasoningParserFactory.needs_raw_special_tokens(
        "deepseek-r1") is False
    assert ReasoningParserFactory.needs_raw_special_tokens(
        "no_such_parser") is False


def test_auto_detect_kimi_k3(tmp_path):
    """Kimi K3 model → 'kimi_k3' parser."""
    model_dir = str(tmp_path / "Kimi-K3")
    os.makedirs(model_dir)
    _write_config(model_dir, "kimi_k3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "kimi_k3"


_ARGS = "<arg_value>grep -c \"</think><tool_call>\" log</arg_value></tool_call>"
_OFF = {"enable_thinking": False}

# (key, chat_template_kwargs, text, content, reasoning). `glm`, `glm47` and
# `glm_moe_dsa` start inside the prefilled block; `glm45` waits for `<think>`.
_GLM_CASES = [
    ("glm", None, "plan</think>The answer is 42.", "The answer is 42.", "plan"),
    # A repeated close is dropped before any content or before the first call.
    ("glm", None, "recovered.</think>Let me retry.</think><tool_call>f",
     "Let me retry.<tool_call>f", "recovered."),
    ("glm", None, "plan</think></think></think>done", "done", "plan"),
    ("glm", None, "plan</think></think>", "", "plan"),
    ("glm", None, "plan</think>It printed </think> here.",
     "It printed </think> here.", "plan"),
    ("glm", None, "r</think>text</think>", "text</think>", "r"),
    # `<think>` never opens a second block.
    ("glm", None, "r</think>a<think>b\n</", "a<think>b\n</", "r"),
    ("glm", None, "a<think>", "", "a<think>"),
    # A tool call ends the block; text that only resembles a marker is kept.
    ("glm", None, "Plan.<tool_call>f<arg_key>m</arg_key></tool_call>",
     "<tool_call>f<arg_key>m</arg_key></tool_call>", "Plan."),
    ("glm", None, "plan</think>say<tool_call>f", "say<tool_call>f", "plan"),
    ("glm", None, "x</t<tool_call>y", "<tool_call>y", "x</t"),
    ("glm", None, "Planning <tools are useful", "",
     "Planning <tools are useful"),
    ("glm", None, "done<tool_c", "", "done<tool_c"),
    # Tool-call arguments are passed through verbatim.
    ("glm47", None, "Done.</think><tool_call>exec" + _ARGS,
     "<tool_call>exec" + _ARGS, "Done."),
    ("glm47", None, "Plan.<tool_call>exec" + _ARGS, "<tool_call>exec" + _ARGS,
     "Plan."),
    # With thinking off the template closed the block in the prompt.
    ("glm47", _OFF, "Paris.", "Paris.", ""),
    ("glm_moe_dsa", _OFF, "</think>Paris.</think><tool_call>f",
     "Paris.<tool_call>f", ""),
    # The block opens only at the start of the output.
    ("glm45", None, "\n<think>let me think</think>Answer.", "Answer.",
     "let me think"),
    ("glm45", None, " <think>plan<tool_call>f", "<tool_call>f", "plan"),
    ("glm45", None, "\nAnswer.", "\nAnswer.", ""),
    ("glm45", None, "Answer.<think>x</think>y", "Answer.<think>x</think>y", ""),
    ("glm45", None, "</think>a", "a", ""),
    ("glm45", None, "hello<tool_call>f", "hello<tool_call>f", ""),
]


def _glm_stream(key: str, kwargs: dict | None,
                deltas: list[str]) -> tuple[str, str]:
    parser = ReasoningParserFactory.create_reasoning_parser(key, kwargs)
    results = [parser.parse_delta(delta) for delta in deltas]
    results.append(parser.finish())
    content = "".join(r.content for r in results)
    return content, "".join(r.reasoning_content for r in results)


@pytest.mark.parametrize(("key", "kwargs", "text", "content", "reasoning"),
                         _GLM_CASES)
def test_glm_reasoning_parser(key: str, kwargs: dict | None, text: str,
                              content: str, reasoning: str) -> None:
    expected = (content, reasoning)
    result = ReasoningParserFactory.create_reasoning_parser(key,
                                                            kwargs).parse(text)
    assert (result.content, result.reasoning_content) == expected
    # Small chunks spread a marker over many deltas; the three-way splits put
    # two delta boundaries at every pair of offsets.
    for size in (1, 2, 4):
        chunks = [text[i:i + size] for i in range(0, len(text), size)]
        assert _glm_stream(key, kwargs, chunks) == expected, chunks
    for i, j in itertools.combinations_with_replacement(range(len(text) + 1),
                                                        2):
        chunks = [text[:i], text[i:j], text[j:]]
        assert _glm_stream(key, kwargs, chunks) == expected, chunks


@pytest.mark.parametrize("key", ["glm", "glm45"])
def test_glm_streaming_matches_parse_on_random_text(key: str) -> None:
    pieces = [
        "<think>", "</think>", "<tool_call>", "</tool_call>", "<", "</", "<t",
        "</th", "<tool_", "think>", "call>", "a", " ", "\n"
    ]
    rng = random.Random(0)
    for _ in range(3000):
        text = "".join(rng.choices(pieces, k=rng.randint(0, 10)))
        whole = ReasoningParserFactory.create_reasoning_parser(key).parse(text)
        assert "<tool_call>" not in whole.reasoning_content
        cuts = [i for i in range(1, len(text)) if rng.random() < 0.3]
        chunks = [text[i:j] for i, j in zip([0] + cuts, cuts + [len(text)])]
        expected = (whole.content, whole.reasoning_content)
        assert _glm_stream(key, None, chunks) == expected, chunks


def test_deepseek_r1_keeps_a_second_close_as_content() -> None:
    result = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1").parse("a</think>b</think><tool_call>f")
    assert result.content == "b</think><tool_call>f"

# ---------------------------------------------------------------------------
# MuseSpark header/message chat format. Ported from the standalone parser
# package that was validated against the reference vLLM reasoning parser.
# ---------------------------------------------------------------------------
import random  # noqa: E402

MS_S, MS_M, MS_EOM, MS_EOT = "<|start|>", "<|message|>", "<|eom|>", "<|eot|>"
MS_TOOL_BODY = ('<atem:function_calls>\n<atem:invoke name="get_weather">\n'
                '<atem:parameter name="city">Paris</atem:parameter>\n'
                '</atem:invoke>\n</atem:function_calls>')
MS_TOOL_BODY2 = ('<atem:function_calls>\n<atem:invoke name="get_time">\n'
                 '<atem:parameter name="tz">UTC</atem:parameter>\n'
                 '</atem:invoke>\n</atem:function_calls>')


def musespark_make_parser():
    return ReasoningParserFactory.create_reasoning_parser("musespark")


def musespark_stream_chunks(parser, text, chunks):
    """Feed ``text`` in the given chunk sizes; return (content, reasoning)."""
    content, reasoning = [], []
    pos = 0
    for n in chunks:
        r = parser.parse_delta(text[pos:pos + n])
        content.append(r.content)
        reasoning.append(r.reasoning_content)
        pos += n
    if pos < len(text):
        r = parser.parse_delta(text[pos:])
        content.append(r.content)
        reasoning.append(r.reasoning_content)
    tail = parser.finish()
    content.append(tail.content)
    reasoning.append(tail.reasoning_content)
    return "".join(content), "".join(reasoning)


# -------------------------------------------------------------- non-streaming
MS_CASES = [
    # reasoning only (truncated / max_tokens, no end marker)
    (f" to=self{MS_M}Let me think", "", "Let me think"),
    # reasoning closed, nothing else
    (f" to=self{MS_M}Let me think{MS_EOM}", "", "Let me think"),
    # reasoning + final answer (reference docstring shape 1)
    (f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=user{MS_M}C{MS_EOT}", "C", "R"
     ),
    # bare <|message|> = public CoT (reference shape 2)
    (f"{MS_M}R{MS_EOM}{MS_S}assistant to=user{MS_M}C{MS_EOT}", "C", "R"),
    # final answer only (shape 3)
    (f" to=user{MS_M}C{MS_EOT}", "C", ""),
    # <|end_of_text|> stop
    (f" to=user{MS_M}C<|end_of_text|>", "C", ""),
    # Guided decoding pins the whole generation to one value, so the model
    # writes no routing header and the body runs straight into the end marker.
    # Both markers have to keep it: a forced tool_choice decodes exactly this
    # shape and the body is the call's entire arguments value.
    ('{"a": 2, "b": 3}<|end_of_text|>', '{"a": 2, "b": 3}', ""),
    ('{"a": 2, "b": 3}' + MS_EOT, '{"a": 2, "b": 3}', ""),
    # commentary + final answer: commentary is content (deviation from ref)
    (f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=commentary{MS_M}Working on it.{MS_EOM}"
     f"{MS_S}assistant to=user{MS_M}Done.{MS_EOT}", "Working on it.Done.", "R"),
    # commentary + tool call: header replayed for the tool parser
    (f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=commentary{MS_M}Checking.{MS_EOM}"
     f"{MS_S}assistant to=get_weather{MS_M}{MS_TOOL_BODY}{MS_EOT}",
     f"Checking.{MS_S}assistant to=get_weather{MS_M}{MS_TOOL_BODY}", "R"),
    # direct tool call as the first message (header has no <|start|>assistant)
    (f" to=get_weather{MS_M}{MS_TOOL_BODY}{MS_EOT}",
     f" to=get_weather{MS_M}{MS_TOOL_BODY}", ""),
    # dotted recipient
    (f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=media.create_image{MS_M}{MS_TOOL_BODY}{MS_EOT}",
     f"{MS_S}assistant to=media.create_image{MS_M}{MS_TOOL_BODY}", "R"),
    # parallel calls across messages: end markers dropped, headers kept
    (f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=get_weather{MS_M}{MS_TOOL_BODY}{MS_EOM}"
     f"{MS_S}assistant to=get_time{MS_M}{MS_TOOL_BODY2}{MS_EOT}",
     f"{MS_S}assistant to=get_weather{MS_M}{MS_TOOL_BODY}"
     f"{MS_S}assistant to=get_time{MS_M}{MS_TOOL_BODY2}", "R"),
    # body text containing "<|" that is not a marker, and '<' / '|' chars
    (f" to=self{MS_M}a < b | c <|not_a_token|> d{MS_EOM}{MS_S}assistant to=user{MS_M}"
     f"x <| y{MS_EOT}", "x <| y", "a < b | c <|not_a_token|> d"),
    # developer / assistant recipients are dropped
    (f" to=developer{MS_M}secret{MS_EOM}{MS_S}assistant to=user{MS_M}C{MS_EOT}",
     "C", ""),
    # <|eot|> straight after <|eom|>
    (f" to=self{MS_M}R{MS_EOM}{MS_EOT}", "", "R"),
    # text that is not a header at all streams as content (reference default)
    ("Hello there", "Hello there", ""),
    # model skipped <|eom|> before the next header
    (f" to=self{MS_M}R{MS_S}assistant to=user{MS_M}C{MS_EOT}", "C", "R"),
    # empty
    ("", "", ""),
]


@pytest.mark.parametrize(("text", "content", "reasoning"), MS_CASES)
def test_musespark_parse(text, content, reasoning):
    r = musespark_make_parser().parse(text)
    assert (r.content, r.reasoning_content) == (content, reasoning)


def test_musespark_parse_resets_state():
    p = musespark_make_parser()
    p.parse(f" to=self{MS_M}R")
    r = p.parse(f" to=user{MS_M}C{MS_EOT}")
    assert (r.content, r.reasoning_content) == ("C", "")


# ----------------------------------------------------- stripped-token fallback
MS_STRIPPED_CASES = [
    # ref _split_on_bare_recipients: self + user
    (" to=selfLet me thinkassistant to=userThe answer", "The answer",
     "Let me think"),
    # user only
    (" to=userThe answer", "The answer", ""),
    # self only: never returned as content
    (" to=selfprivate thought", "", "private thought"),
    # no header at all: ordinary content
    ("plain text", "plain text", ""),
    # commentary survives as content
    (" to=selfRassistant to=commentaryWorkingassistant to=userDone",
     "WorkingDone", "R"),
]


@pytest.mark.parametrize(("text", "content", "reasoning"), MS_STRIPPED_CASES)
def test_musespark_parse_stripped_special_tokens(text, content, reasoning):
    r = musespark_make_parser().parse(text)
    assert (r.content, r.reasoning_content) == (content, reasoning)


@pytest.mark.parametrize(("text", "content", "reasoning"), MS_STRIPPED_CASES)
def test_musespark_stream_stripped_special_tokens(text, content, reasoning):
    # No <|message|> ever arrives: header-shaped text is held until finish().
    p = musespark_make_parser()
    got = musespark_stream_chunks(p, text, [1] * len(text))
    assert got == (content, reasoning)


def test_musespark_stream_stripped_never_leaks_reasoning_before_finish():
    p = musespark_make_parser()
    text = " to=selfLet me think about this carefully"
    for ch in text:
        r = p.parse_delta(ch)
        assert r.content == "" and r.reasoning_content == ""
    tail = p.finish()
    assert tail.content == ""
    assert tail.reasoning_content == "Let me think about this carefully"


# ------------------------------------------------------------------ streaming
# Delta tables mirror the reference _extract_reasoning_streaming case table.
MS_DELTA_CASES = [
    # header buffering: nothing emitted until <|message|> resolves recipient
    ([" to", "=se", "lf", MS_M, "R"], ["", "", "", "",
                                       ""], ["", "", "", "", "R"]),
    # case 3: reasoning starts in this delta
    ([f" to=self{MS_M}Rea", "soning"], ["", ""], ["Rea", "soning"]),
    # case 1: reasoning ends in delta, content starts in the same delta
    ([
        f" to=self{MS_M}R", f"est{MS_EOM}{MS_S}assistant to=user{MS_M}Con",
        "tent", MS_EOT
    ], ["", "Con", "tent", ""], ["R", "est", "", ""]),
    # case 2: between reasoning end and content start, nothing emitted
    ([
        f" to=self{MS_M}R{MS_EOM}", f"{MS_S}assistant", " to=", "user", MS_M,
        "C", MS_EOT
    ], ["", "", "", "", "", "C", ""], ["R", "", "", "", "", "", ""]),
    # case 5/6: content starts in delta, continues, ends (case 4)
    (
        [f" to=user{MS_M}Hel", "lo", f" world{MS_EOT}"],  # codespell:ignore
        [
            "Hel",  # codespell:ignore
            "lo",  # codespell:ignore
            " world"
        ],
        ["", "", ""]),
    # partial end marker held back, then released as content
    ([f" to=user{MS_M}a<|e", "x", "b"], ["a", "<|ex", "b"], ["", "", ""]),
    # partial end marker that completes into <|eot|>
    ([f" to=user{MS_M}a<|e", f"ot|>"], ["a", ""], ["", ""]),  # codespell:ignore
    # bare <|message|> public CoT
    ([f"{MS_M}think",
      f"{MS_EOM}{MS_S}assistant to=user{MS_M}C{MS_EOT}"], ["",
                                                           "C"], ["think", ""]),
    # tool handoff: header replayed once the recipient is known, body streams
    (
        [
            " to=get_we",
            f"ather{MS_M}<atem:",  # codespell:ignore
            "function_calls>"
        ],  # codespell:ignore
        ["", f" to=get_weather{MS_M}<atem:", "function_calls>"],
        ["", "", ""]),
    # default: text that cannot be a header streams as content immediately
    (["Hello", " world"], ["Hello", " world"], ["", ""]),
    # developer channel suppressed
    ([
        f" to=developer{MS_M}secret{MS_EOM}{MS_S}assistant to=user{MS_M}ok{MS_EOT}"
    ], ["ok"], [""]),
]


@pytest.mark.parametrize(("deltas", "content", "reasoning"), MS_DELTA_CASES)
def test_musespark_parse_delta(deltas, content, reasoning):
    p = musespark_make_parser()
    got_c, got_r = [], []
    for d in deltas:
        r = p.parse_delta(d)
        got_c.append(r.content)
        got_r.append(r.reasoning_content)
    assert got_c == content
    assert got_r == reasoning


def test_musespark_finish_flushes_truncated_reasoning():
    p = musespark_make_parser()
    p.parse_delta(f" to=self{MS_M}abc<|e")
    tail = p.finish()
    assert tail.reasoning_content == "<|e" and tail.content == ""


def test_musespark_finish_flushes_truncated_content():
    p = musespark_make_parser()
    assert p.parse_delta(f" to=user{MS_M}abc<").content == "abc"
    assert p.finish().content == "<"


def test_musespark_headerless_body_survives_end_marker():
    """A forced tool call's arguments are the whole response.

    ``tool_choice={"type":"function",...}`` constrains generation to the
    arguments value, so nothing that looks like a routing header is ever
    emitted and the body runs straight into ``<|end_of_text|>``. The header
    state used to clear its buffer on that marker, which threw the arguments
    away: the caller saw HTTP 200, an empty message and no tool call.

    Streaming already got this right (the marker is not in the buffer when the
    first chunk is routed), so this also pins the streamed and non-streamed
    paths to the same answer.
    """
    arguments = '{"a": 2, "b": 3}'
    for marker in ("<|end_of_text|>", MS_EOT):
        result = musespark_make_parser().parse(arguments + marker)
        assert result.content == arguments
        assert result.reasoning_content == ""

        streamed = musespark_make_parser()
        got = musespark_stream_chunks(streamed, arguments + marker,
                                      [1] * len(arguments + marker))
        assert got == (arguments, "")


def test_musespark_end_marker_still_ends_a_completed_turn():
    """The fix must not resurrect text the parser is meant to drop.

    After a closed message the header state sees ``<|start|>assistant`` (or a
    fragment of it) followed by the end marker; that is a header remnant, not a
    body, and it stays dropped.
    """
    for head in ("", MS_S + "assistant", " to=us"):
        result = musespark_make_parser().parse(
            f" to=user{MS_M}C{MS_EOM}{head}{MS_EOT}")
        assert result.content == "C"
        assert result.reasoning_content == ""


def test_musespark_finish_drops_partial_header_after_message():
    p = musespark_make_parser()
    p.parse_delta(f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=us")
    tail = p.finish()
    assert tail == type(tail)()


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize(("text", "_c", "_r"), MS_CASES + MS_STRIPPED_CASES)
def test_musespark_stream_matches_parse(seed, text, _c, _r):
    rng = random.Random(seed)
    chunks = []
    remaining = len(text)
    while remaining > 0:
        n = rng.randint(1, 7)
        chunks.append(min(n, remaining))
        remaining -= chunks[-1]
    expected = musespark_make_parser().parse(text)
    got = musespark_stream_chunks(musespark_make_parser(), text, chunks)
    assert got == (expected.content, expected.reasoning_content)


class _CountingMuseSparkParser(MuseSparkReasoningParser):
    """Turns "the parser stopped consuming" from a hang into a failure.

    The bug these tests pin wedged a live server: the header and body states
    handed the same buffer back and forth, `_feed`'s loop never ended, and
    because that loop runs in the server's event loop the whole frontend
    stopped answering, including ``/health``. A plain assertion on the parsed
    text would reproduce it as a test run that never finishes, which is a much
    worse signal than a failed assertion.
    """

    STEP_LIMIT = 500

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0

    def _step(self, content, reasoning):
        self.steps += 1
        if self.steps > self.STEP_LIMIT:
            raise AssertionError(
                f"parser made {self.steps} steps without finishing: state "
                f"{self._state!r}, buffer {self._buffer[:64]!r}")
        return super()._step(content, reasoning)


@pytest.mark.parametrize(("text", "content", "reasoning"), [
    (f"{MS_S}hello world", "hello world", ""),
    (f"{MS_S}xy<|end_of_text|>", "xy", ""),
    (f" to=self{MS_M}think{MS_S}zz", "zz", "think"),
    (f" to=user{MS_M}A{MS_S}not a header", "Anot a header", ""),
])
def test_musespark_stray_start_marker_terminates(text, content, reasoning):
    """``<|start|>`` that is not followed by a header must still be consumed.

    The body state leaves ``<|start|>`` in the buffer so the header regexes see
    the canonical ``<|start|>assistant to=`` form. When the header state then
    ruled the text out as a header it handed the buffer back unchanged, the
    body state found the marker at index 0, emitted nothing and handed it back
    again. Under concurrent load with ``--ignore-eos`` the model emits exactly
    this, and the server stopped serving.
    """
    p = _CountingMuseSparkParser()
    result = p.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning


def test_musespark_stray_start_marker_terminates_streamed():
    p = _CountingMuseSparkParser()
    got_c, got_r = [], []
    for delta in (" to=self", MS_M, "think", MS_S, "zz", "!"):
        r = p.parse_delta(delta)
        got_c.append(r.content)
        got_r.append(r.reasoning_content)
    tail = p.finish()
    assert "".join(got_r) == "think"
    assert "".join(got_c) + tail.content == "zz!"


def test_musespark_feed_is_bounded_even_if_a_step_stops_consuming():
    """The backstop, independent of the specific cycle above.

    ``_feed`` runs in the server's event loop, so a step sequence that stops
    consuming must degrade to a bad parse, never to a spin.
    """

    class _NeverConsumes(MuseSparkReasoningParser):

        def __init__(self):
            super().__init__()
            self.steps = 0

        def _step(self, content, reasoning):
            self.steps += 1
            return True  # always "more to do", never consumes

    p = _NeverConsumes()
    result = p.parse_delta("abcdefgh")
    assert result.content == "" and result.reasoning_content == ""
    assert p.steps <= len("abcdefgh") + 8
    assert p._buffer == "abcdefgh"


MS_FUZZ_PIECES = [
    MS_S,
    MS_M,
    MS_EOM,
    MS_EOT,
    "<|end_of_text|>",
    " to=self",
    " to=user",
    " to=commentary",
    " to=developer",
    " to=get_weather",
    "assistant",
    "hello",
    " ",
    "",
    "<|start",
    "to=",
    "<atem:function_calls>",
    '{"a": 2}',
    "\n",
    "<|",
]  # codespell:ignore


@pytest.mark.parametrize("seed", range(300))
def test_musespark_always_consumes(seed):
    """No input makes the parser stop consuming, batched or streamed.

    The specific cycle above is pinned by name; this pins the property behind
    it. The parser runs inside the server's event loop, so an input it never
    finishes with is not a bad answer, it is a dead server, and marker soup is
    exactly what unconstrained decoding produces. The step counter turns a
    regression into a failure instead of a test run that never returns.
    """
    rng = random.Random(seed)
    text = "".join(
        rng.choice(MS_FUZZ_PIECES) for _ in range(rng.randint(1, 14)))

    _CountingMuseSparkParser().parse(text)

    streamed = _CountingMuseSparkParser()
    pos = 0
    while pos < len(text):
        n = rng.randint(1, 6)
        streamed.parse_delta(text[pos:pos + n])
        pos += n
    streamed.finish()


def test_musespark_registry_and_flag():
    assert "musespark" in ReasoningParserFactory.keys()
    assert ReasoningParserFactory.needs_raw_special_tokens("musespark")
    assert MuseSparkReasoningParser.needs_raw_special_tokens is True
    assert isinstance(musespark_make_parser(), MuseSparkReasoningParser)
