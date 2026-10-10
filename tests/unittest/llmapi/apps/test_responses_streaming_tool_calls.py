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
"""Unit tests for Responses API streaming tool call emission.

These drive `_generate_streaming_event` with real reasoning and tool parsers:
what they check is the agreement between the incremental and the whole-text
parse, which mocked parsers cannot exhibit.
"""

import json
import random
from types import SimpleNamespace

import pytest
from openai.types.responses.tool import FunctionTool

from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingEventsHelper,
    _accumulate_tool_call_fragments,
    _assembled_tool_calls,
    _create_output_content,
    _generate_streaming_event,
    _get_chat_completion_function_tools,
)
from tensorrt_llm.serve.tool_parser.core_types import ToolCallItem
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

pytestmark = pytest.mark.cpu_only

_MARKUP = ("<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>")

# A recorded GLM-4.7 response in its original stream frames: reasoning, one
# sentence, then two calls to a namespaced tool. The frame boundaries put one
# closed and one open call in the parser's buffer at the same time.
_REAL_FRAMES = [
    "Let",
    (
        " me start by understanding the problem. I need to:\n1. Read the problem definition\n2."
        " Understand the kernel-factory-schemas\n3. Read the problem files\n4. Implement a CUDA"
        " kernel\n\nLet me begin by reading the problem files and understanding what I'm workin"  # codespell:ignore
        "g with.</think>I"
    ),
    (
        "'ll start by reading the problem definition and understanding the schema, then dive in"
        "to the kernel implementation.<tool_call>functions.exec<arg_key>input</arg_key><arg_val"
        "ue>\n// Read the problem definition and related files\nconst out = await tools.exec_co"  # codespell:ignore ue
        "mmand({cmd: `python3 - <<'PYEOF'\nimport json\nd = json.load(open"
    ),
    (
        "('/tmp/problem/definition.json'))\nprint(\"Description:\", d.get('description', ''"
        "))\nprint(\"\\nFixed axes:\", {k: v['value'] for k, v in d.get('axes', {}).items() i"
        "f v.get('type') == 'const'})\nprint(\"Variable axes:\", [k for k, v in d.get('axes"
        "', {}).items() if v.get('type')"
    ),
    (
        " == 'var'})\nprint(\"\\nInputs:\")\nfor n, s in d['inputs'].items():  print(f\"  {n}:"
        " shape={s['shape']}, dtype={s['dtype']}\")\nprint(\"Outputs (pre-allocated):\")\nfor "
        "n, s in d['outputs'].items(): print(f\"  {n}: shape={s['shape']}, dtype={s"
    ),
    (
        "['dtype']}\")\nin_args  = list(d['inputs'].keys())\nout_args = list(d['outputs']."
        "keys())\nprint(f\"\\nRequired signature: def run({', '.join(in_args + out_args)}) -> "
        'None:")\nPYEOF\necho "=== workload ==="\nhead -3 /tmp/problem/workload.jsonl\necho "=='
        '= language ==="\ncat /'
    ),
    (
        'tmp/problem/language.txt\necho "=== user_prompt ==="\ncat /tmp/problem/user_prompt.txt'
        "\n`, yield_time_ms: 15000});\ntext(out.output);\n</arg_value></tool_call><tool_call>fu"
        "nctions.exec<arg_key>input</arg_key><arg_value>\n// Read the baseline metrics\nconst o"
        "ut = await tools.exec_command({cmd: `cat /tmp/problem/baseline/metrics.json`, yield"
    ),
    "_time_ms: 5000});\ntext(out.output);\n</arg_value></tool_call>",
]

# The only visible assistant text in that response.
_REAL_TEXT = (
    "I'll start by reading the problem definition and understanding "
    "the schema, then dive into the kernel implementation."
)

_GET_TIME_CALL = "<tool_call>get_time<arg_key>city</arg_key><arg_value>{}</arg_value></tool_call>"


def _get_time_call(city):
    return _GET_TIME_CALL.format(city)


def _namespace_tool(namespace="functions", name="exec"):
    """A namespaced tool, offered to the model as `<namespace>.<name>`."""
    inner = SimpleNamespace(
        name=name,
        type="function",
        description="run a command",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}},
    )
    return SimpleNamespace(name=namespace, type="namespace", description="tools", tools=[inner])


def _function_tool(name="get_time", properties=("city",)):
    return FunctionTool(
        name=name,
        type="function",
        strict=False,
        parameters={"type": "object", "properties": {p: {"type": "string"} for p in properties}},
    )


def _drive(
    frames,
    tools=None,
    finish=True,
    tool_parser_id="glm47",
    reasoning_parser_id="glm47",
    helper=None,
    parsers=None,
):
    """Feed `frames` through the streaming event generator, one chunk each."""
    helper = helper or ResponsesStreamingEventsHelper()
    request = SimpleNamespace(
        tools=list(tools) if tools is not None else [_namespace_tool()], tool_choice="auto"
    )
    reasoning_parser_dict, tool_parser_dict = parsers or ({}, {})
    events, accumulated = [], ""
    for i, frame in enumerate(frames):
        accumulated += frame
        events.extend(
            _generate_streaming_event(
                output=SimpleNamespace(index=0, text=accumulated, text_diff=frame),
                request=request,
                finished_generation=finish and i == len(frames) - 1,
                streaming_events_helper=helper,
                reasoning_parser_id=reasoning_parser_id,
                tool_parser_id=tool_parser_id,
                reasoning_parser_dict=reasoning_parser_dict,
                tool_parser_dict=tool_parser_dict,
            )
        )
    return events


def _payloads(events, event_type):
    return [e.text for e in events if e.type == event_type]


def _texts(events):
    return _payloads(events, "response.output_text.done")


def _items(events, event_type, item_type):
    return [e.item for e in events if e.type == event_type and e.item.type == item_type]


def _call_items(events):
    return _items(events, "response.output_item.done", "function_call")


def _first_index(events, event_type, item_type):
    return next(
        i for i, e in enumerate(events) if e.type == event_type and e.item.type == item_type
    )


def _assert_no_markup(payloads):
    for payload in payloads:
        for marker in _MARKUP:
            assert marker not in payload, f"{marker!r} reached the client as text: {payload!r}"


class TestRecordedResponse:
    def test_one_message_item_then_both_calls(self):
        events = _drive(_REAL_FRAMES)

        assert _texts(events) == [_REAL_TEXT]
        assert _first_index(events, "response.output_item.done", "message") < _first_index(
            events, "response.output_item.added", "function_call"
        )
        calls = _call_items(events)
        assert [(c.name, c.namespace) for c in calls] == [("exec", "functions")] * 2
        assert all(json.loads(c.arguments)["input"] for c in calls)

    @pytest.mark.parametrize("finish", [False, True])
    def test_every_prefix_streams_done_payloads_equal_to_their_deltas(self, finish):
        """Only a stream finishing inside a call may release markup, as text."""
        delta_of = {
            "response.output_text.done": "response.output_text.delta",
            "response.reasoning_text.done": "response.reasoning_text.delta",
        }
        for k in range(1, len(_REAL_FRAMES) + 1):
            accumulated = "".join(_REAL_FRAMES[:k])
            events = _drive(_REAL_FRAMES[:k], finish=finish)
            streamed = {}
            for event in events:
                if event.type in delta_of.values():
                    key = (event.item_id, event.type)
                    streamed[key] = streamed.get(key, "") + event.delta
                elif event.type in delta_of:
                    assert event.text == streamed.get((event.item_id, delta_of[event.type]), "")

            start = accumulated.rfind("<tool_call>")
            texts = _texts(events)
            if finish and start != -1 and "</tool_call>" not in accumulated[start:]:
                assert texts[-1].endswith(accumulated[start:])
                texts[-1] = texts[-1][: -len(accumulated[start:])]
            _assert_no_markup(texts)

    def test_any_chunking_gives_the_same_result(self):
        full_text = "".join(_REAL_FRAMES)
        rng = random.Random(20260918)
        for _ in range(30):
            cuts = sorted(rng.sample(range(1, len(full_text)), rng.randint(1, 40)))
            frames = [full_text[a:b] for a, b in zip([0] + cuts, cuts + [len(full_text)])]
            events = _drive(frames)
            assert _texts(events) == [_REAL_TEXT]
            assert [c.name for c in _call_items(events)] == ["exec", "exec"]

    def test_a_stream_cut_off_mid_call_releases_the_markup_as_text(self):
        raw = "".join(_REAL_FRAMES[:3])
        events = _drive(_REAL_FRAMES[:3])

        assert _texts(events) == [_REAL_TEXT + raw[raw.index("<tool_call>") :]]
        assert _call_items(events) == []

    def test_a_stream_cut_off_after_one_call_keeps_that_call(self):
        events = _drive(_REAL_FRAMES[:7])
        assert [c.name for c in _call_items(events)] == ["exec"]

    def test_a_finished_output_seen_twice_emits_its_calls_once(self):
        helper, parsers = ResponsesStreamingEventsHelper(), ({}, {})
        assert len(_call_items(_drive(_REAL_FRAMES, helper=helper, parsers=parsers))) == 2
        assert _call_items(_drive([""], helper=helper, parsers=parsers)) == []


class TestTransitions:
    def test_a_call_straight_after_reasoning_closes_the_reasoning_item(self):
        events = _drive(
            ["Deciding what to do.</think>", _get_time_call("Beijing")], tools=[_function_tool()]
        )
        assert _items(events, "response.output_item.done", "message") == []
        assert _payloads(events, "response.reasoning_text.done") == ["Deciding what to do."]
        assert _first_index(events, "response.output_item.done", "reasoning") < _first_index(
            events, "response.output_item.added", "function_call"
        )
        assert [c.name for c in _call_items(events)] == ["get_time"]

    @pytest.mark.parametrize(
        "frames",
        [
            ["Thinking.</think>Before. ", _get_time_call("Beijing"), "After."],
            ["Thinking.</think>Before. ", _get_time_call("Beijing") + "After."],
        ],
    )
    def test_text_after_a_call_opens_a_new_message_item(self, frames):
        events = _drive(frames, tools=[_function_tool()])
        assert _texts(events) == ["Before. ", "After."]
        assert len(_call_items(events)) == 1

    def test_two_whole_calls_in_the_last_chunk_are_both_reported(self):
        events = _drive(
            ["Thinking.</think>", _get_time_call("Beijing") + _get_time_call("Paris")],
            tools=[_function_tool()],
        )
        assert [json.loads(c.arguments)["city"] for c in _call_items(events)] == [
            "Beijing",
            "Paris",
        ]

    def test_an_abort_closes_the_open_item(self):
        events = _drive(["Thinking.</think>Partial answer", ""])
        assert _texts(events) == ["Partial answer"]
        assert len(_items(events, "response.output_item.done", "message")) == 1

    def test_no_tool_parser_leaves_the_markup_as_text(self):
        events = _drive(["Thinking.</think>Answer <tool_call>x</tool_call>"], tool_parser_id=None)
        assert _texts(events) == ["Answer <tool_call>x</tool_call>"]
        assert _call_items(events) == []

    def test_an_empty_delta_opens_nothing(self):
        assert _drive([""], finish=False) == []


_CALLS_BEGIN, _CALLS_END = "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"
_CALL_BEGIN, _CALL_END, _SEP = "<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>", "<｜tool▁sep｜>"
_NYC, _AI = json.dumps({"location": "NYC"}), json.dumps({"query": "AI"})

# Parsers whose increments are not guaranteed to match their whole-text parse
# take their calls from one whole-text parse at the end of the stream.
_WHOLE_TEXT_PARSER_OUTPUTS = {
    "qwen3": (
        'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n'
        '</tool_call>\n<tool_call>\n{"name": "search_web", "arguments": {"query": "AI"}}\n'
        "</tool_call>"
    ),
    "qwen3_coder": "Checking.\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>",
    "deepseek_v3": (
        f"Checking.{_CALLS_BEGIN}{_CALL_BEGIN}function{_SEP}get_weather\n```json\n{_NYC}\n```"
        f"{_CALL_END}{_CALL_BEGIN}function{_SEP}search_web\n```json\n{_AI}\n```{_CALL_END}"
        f"{_CALLS_END}"
    ),
    "deepseek_v31": (
        f"Checking.{_CALLS_BEGIN}{_CALL_BEGIN}get_weather{_SEP}{_NYC}{_CALL_END}"
        f"{_CALL_BEGIN}search_web{_SEP}{_AI}{_CALL_END}{_CALLS_END}"
    ),
    "minimax_m2": (
        'Checking.<minimax:tool_call><invoke name="get_weather"><parameter name="location">'
        'NYC</parameter></invoke><invoke name="search_web"><parameter name="query">AI'
        "</parameter></invoke></minimax:tool_call>"
    ),
    "gemma4": 'Checking.<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|>',
}


@pytest.mark.parametrize("parser_id", sorted(_WHOLE_TEXT_PARSER_OUTPUTS))
def test_whole_text_parser_calls_are_streamed_once_and_repeated_in_the_snapshot(parser_id):
    text = _WHOLE_TEXT_PARSER_OUTPUTS[parser_id]
    tools = [
        _function_tool("get_weather", ("location",)),
        _function_tool("search_web", ("query",)),
        _function_tool("get_time", ()),
    ]
    expected = ToolParserFactory.create_tool_parser(parser_id).detect_and_parse(
        text, _get_chat_completion_function_tools(tools)
    )
    expected_calls = [(c.name, json.loads(c.parameters or "{}")) for c in expected.calls]
    assert expected_calls

    for size in (1, 3, 8, len(text)):
        helper = ResponsesStreamingEventsHelper()
        frames = [text[i : i + size] for i in range(0, len(text), size)]
        events = _drive(
            frames, tools=tools, tool_parser_id=parser_id, reasoning_parser_id=None, helper=helper
        )
        calls = _call_items(events)
        assert [(c.name, json.loads(c.arguments)) for c in calls] == expected_calls, size

        items, _messages, _reasoning = _create_output_content(
            SimpleNamespace(outputs=[SimpleNamespace(index=0, text=text)]),
            tool_parser=parser_id,
            tools=tools,
            streamed_tool_calls=helper.emitted_tool_call_items,
            streamed_item_ids=helper.emitted_item_ids,
        )
        assert [i for i in items if i.type == "function_call"] == calls


class TestFragmentAssembly:
    def test_fragments_concatenate_in_order_under_the_first_name(self):
        fragments = {}
        _accumulate_tool_call_fragments(
            fragments,
            [
                ToolCallItem(tool_index=0, name="get_time", parameters=""),
                ToolCallItem(tool_index=0, name=None, parameters='{"city": "Par'),
                ToolCallItem(tool_index=0, name=None, parameters='is"}'),
            ],
        )
        assert _assembled_tool_calls(fragments) == [
            ToolCallItem(tool_index=0, name="get_time", parameters='{"city": "Paris"}')
        ]

    @pytest.mark.parametrize(
        "name, parameters",
        [
            ("get_time", '{"a": '),
            ("get_time", '{"cmd": }'),
            ("get_time", '{"a": NaN}'),
            ("get_time", ""),
            (None, '{"a": 1}'),
        ],
    )
    def test_calls_a_client_cannot_run_are_skipped(self, name, parameters):
        fragments = {}
        good = ToolCallItem(tool_index=0, name="get_time", parameters='{"a": 1}')
        bad = ToolCallItem(tool_index=1, name=name, parameters=parameters)
        _accumulate_tool_call_fragments(fragments, [good, bad])
        assert _assembled_tool_calls(fragments) == [good]

    def test_fragments_are_kept_apart_per_output_and_per_request(self):
        first, second = ResponsesStreamingEventsHelper(), ResponsesStreamingEventsHelper()
        for index, name in ((0, "first"), (1, "second")):
            _accumulate_tool_call_fragments(
                first.tool_call_fragments(index),
                [ToolCallItem(tool_index=0, name=name, parameters="{}")],
            )
        assert [c.name for c in _assembled_tool_calls(first.tool_call_fragments(0))] == ["first"]
        assert [c.name for c in _assembled_tool_calls(first.tool_call_fragments(1))] == ["second"]
        assert second.tool_call_fragments(0) == {}
