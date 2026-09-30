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

"""Unit tests for Responses API streaming tool call emission (TRTLLM-9605).

These drive `_generate_streaming_event` with the real GLM-4.7 reasoning and
tool parsers rather than with mocked ones. The defect these tests exist for -
tool-call markup published as the assistant's message - lived entirely in the
disagreement between what the incremental parser had streamed and what a
whole-text re-parse concluded afterwards, so a test that mocks the parsers
cannot see it.
"""

import json
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai.types.responses.tool import FunctionTool

from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingEventsHelper,
    _accumulate_tool_call_fragments,
    _assembled_tool_calls,
    _generate_streaming_event,
)
from tensorrt_llm.serve.tool_parser.core_types import ToolCallItem

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only

_REASONING_PARSER = "glm47"
_TOOL_PARSER = "glm47"

# Markup that must never reach a client as assistant text - with one
# sanctioned exception: a stream that ends inside an unterminated call
# releases the raw markup as message text, because the final response's
# whole-text re-parse keeps it as message text too and the two views must
# agree (see _flush_tool_parser and TestEndOfStream below).
_MARKUP = ("<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>")

# One real GLM-4.7 response, verbatim, from a recorded agent run: eight stream
# frames carrying reasoning, a sentence of prose, and two tool calls. Feeding
# frames 0-6 used to publish 296 characters of `normal_text` beginning with the
# prose and continuing straight into `<tool_call>functions.exec<arg_key>...`,
# cut off mid-word at `yield`; the whole response yields 116 clean characters
# and two calls. The frame boundaries matter - they are what put one closed
# call and one open call in the accumulated text at the same time - so they are
# kept exactly as recorded.
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

# Everything the frames above put between `</think>` and the first
# `<tool_call>`: the only visible assistant text in that response.
_REAL_TEXT = (
    "I'll start by reading the problem definition and understanding "
    "the schema, then dive into the kernel implementation."
)


def _make_output(text: str, text_diff: str, index: int = 0):
    """A stand-in for one RequestOutput as the streaming loop sees it."""
    return SimpleNamespace(index=index, text=text, text_diff=text_diff)


def _namespace_tool(namespace: str = "functions", name: str = "exec"):
    """A namespaced tool, offered to the model as `<namespace>.<name>`.

    This is the shape the recorded response was produced with, and the reason
    its calls are named `functions.exec`.
    """
    inner = SimpleNamespace(
        name=name,
        type="function",
        description="run a command",
        parameters={
            "type": "object",
            "properties": {"input": {"type": "string"}},
        },
    )
    return SimpleNamespace(name=namespace, type="namespace", description="tools", tools=[inner])


def _function_tool(name: str = "get_time"):
    return FunctionTool(
        name=name,
        type="function",
        strict=False,
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    )


def _make_request(tools=None):
    """A minimal ResponsesRequest-like object.

    SimpleNamespace rather than a nested class: a class body does not close
    over the enclosing function's locals (unlike a nested function), so
    ``tools = tools or []`` inside one raises NameError on the right-hand name.
    """
    return SimpleNamespace(tools=list(tools) if tools else [])


def _drive(
    frames,
    tools=None,
    finish=True,
    helper=None,
    tool_parser_id=_TOOL_PARSER,
    reasoning_parser_id=_REASONING_PARSER,
):
    """Feed `frames` through the streaming event generator, one chunk each.

    Returns the flat list of events, which is what a client would receive.
    """
    helper = helper or ResponsesStreamingEventsHelper()
    request = _make_request(tools if tools is not None else [_namespace_tool()])
    reasoning_parser_dict, tool_parser_dict = {}, {}
    events, accumulated = [], ""
    for i, frame in enumerate(frames):
        accumulated += frame
        events.extend(
            _generate_streaming_event(
                output=_make_output(accumulated, frame),
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


def _of_type(events, event_type):
    return [e for e in events if getattr(e, "type", None) == event_type]


def _payloads(events, event_type):
    return [e.text for e in _of_type(events, event_type)]


def _done_payloads(events):
    """Every terminal text payload a client would display, text or reasoning."""
    return _payloads(events, "response.output_text.done") + _payloads(
        events, "response.reasoning_text.done"
    )


def _items(events, event_type, item_type):
    return [
        e.item for e in _of_type(events, event_type) if getattr(e.item, "type", None) == item_type
    ]


def _call_items(events):
    return _items(events, "response.output_item.done", "function_call")


def _assert_no_markup(events):
    for payload in _done_payloads(events):
        for marker in _MARKUP:
            assert marker not in payload, (
                f"{marker!r} reached the client as assistant text: {payload!r}"
            )


class TestTheDefect:
    """Edge case 1: two or more calls in one response."""

    def test_text_item_closes_at_the_first_call_with_no_markup(self):
        """The whole recorded response, replayed frame by frame.

        Exactly one message item, holding only the prose that preceded the
        first call, and both calls recovered. Before the fix this same replay
        produced a 296-character message item running from the prose straight
        into `<tool_call>functions.exec<arg_key>input</arg_key><arg_value>`.
        """
        events = _drive(_REAL_FRAMES)

        _assert_no_markup(events)
        assert _payloads(events, "response.output_text.done") == [_REAL_TEXT]

        calls = _call_items(events)
        assert [c.name for c in calls] == ["exec", "exec"]
        assert [c.namespace for c in calls] == ["functions", "functions"]
        for call in calls:
            # Arguments a client cannot parse are arguments it cannot run.
            assert json.loads(call.arguments)["input"]

    def test_the_prose_is_one_item_not_two(self):
        """The chunk carrying both the prose and the call start is one turn.

        The delta has to be appended to the open item before the call closes
        it. If the close ran first, the prose in that chunk would land in a
        second message item - the split happens mid-sentence, and a client
        rendering only the last item shows only the fragment.
        """
        events = _drive(_REAL_FRAMES)
        assert len(_items(events, "response.output_item.done", "message")) == 1

    def test_no_prefix_of_the_response_leaks_markup(self):
        """Feeding frames 0..k for every k must never *leak* markup.

        The failing window was narrow - it opened once one call had closed
        while another was still open - so the whole prefix family is replayed
        rather than just the end state.

        One shape is a release, not a leak: a prefix that finishes inside an
        unterminated call flushes the raw remainder as its trailing message
        item, because the final response's whole-text re-parse keeps that
        text too and the two views of one generation must agree (see
        _flush_tool_parser). That remainder is pinned exactly - the bytes
        from the last unclosed `<tool_call>` to the end - and everything
        else still has to be clean; an unfinished stream (finish=False)
        releases nothing at all.
        """

        def pending_markup(accumulated):
            start = accumulated.rfind("<tool_call>")
            if start == -1 or "</tool_call>" in accumulated[start:]:
                return None
            return accumulated[start:]

        for k in range(1, len(_REAL_FRAMES) + 1):
            for finish in (False, True):
                events = _drive(_REAL_FRAMES[:k], finish=finish)
                released = pending_markup("".join(_REAL_FRAMES[:k])) if finish else None
                for payload in _done_payloads(events):
                    if released is not None and payload == released:
                        continue
                    for marker in _MARKUP:
                        assert marker not in payload, (
                            f"k={k} finish={finish}: {marker!r} reached the "
                            f"client as assistant text: {payload!r}"
                        )

    def test_done_payload_is_exactly_the_deltas_that_were_streamed(self):
        """The invariant, stated directly, for every prefix.

        Every enumerated case is a consequence of this one property, and it
        catches the regressions the enumerated cases miss: once a done payload
        is defined as the sum of the deltas already sent, text that was never
        streamed cannot appear in it.
        """
        delta_of = {
            "response.output_text.done": "response.output_text.delta",
            "response.reasoning_text.done": "response.reasoning_text.delta",
        }
        for k in range(1, len(_REAL_FRAMES) + 1):
            for finish in (False, True):
                events = _drive(_REAL_FRAMES[:k], finish=finish)
                streamed: dict[tuple[str, str], str] = {}
                for event in events:
                    event_type = getattr(event, "type", None)
                    if event_type in delta_of.values():
                        key = (event.item_id, event_type)
                        streamed[key] = streamed.get(key, "") + event.delta
                    elif event_type in delta_of:
                        key = (event.item_id, delta_of[event_type])
                        assert event.text == streamed.get(key, ""), (
                            f"k={k} finish={finish}: done payload differs from "
                            f"the deltas sent for {event.item_id}"
                        )

    def test_any_chunking_of_the_response_gives_the_same_result(self):
        """The recorded frame boundaries are one chunking out of many.

        Whether a call arrives whole in one chunk or spread over ten is a
        property of the decoder's timing, not of the response, so every
        chunking has to produce the same text and the same two calls. Replayed
        against the previous implementation this fails on 1742 of 2000
        randomised chunkings; the boundaries are seeded so a failure is
        reproducible.
        """
        full_text = "".join(_REAL_FRAMES)
        rng = random.Random(20260918)

        for _ in range(30):
            cuts = sorted(rng.sample(range(1, len(full_text)), rng.randint(1, 40)))
            frames = [full_text[a:b] for a, b in zip([0] + cuts, cuts + [len(full_text)])]
            events = _drive(frames)

            _assert_no_markup(events)
            assert _payloads(events, "response.output_text.done") == [_REAL_TEXT]
            assert [c.name for c in _call_items(events)] == ["exec", "exec"]

    def test_two_whole_calls_in_one_chunk_are_both_reported(self):
        """A parser reports at most one finished call per increment.

        Glm47ToolParser returns as soon as it sees a `</tool_call>` and keeps
        the rest buffered for the next increment. When the stream ends on that
        chunk there is no next increment, so without draining the parser at
        end of stream the second call is silently lost.
        """
        events = _drive(
            [
                "Thinking.</think>",
                "<tool_call>get_time<arg_key>city</arg_key>"
                "<arg_value>Beijing</arg_value></tool_call>"
                "<tool_call>get_time<arg_key>city</arg_key>"
                "<arg_value>Paris</arg_value></tool_call>",
            ],
            tools=[_function_tool()],
        )

        calls = _call_items(events)
        assert [c.name for c in calls] == ["get_time", "get_time"]
        assert [json.loads(c.arguments)["city"] for c in calls] == ["Beijing", "Paris"]


class TestTransitions:
    """Edge cases 2, 3 and 4: what closes, and what opens, around a call."""

    def test_call_straight_after_reasoning_emits_no_message_item(self):
        """Edge case 2/3: `</think><tool_call>` closes the reasoning item.

        There is no text item to close, and none may be invented: an empty
        message item is a turn the model never took.
        """
        events = _drive(
            [
                "Deciding what to do.</think>",
                "<tool_call>get_time<arg_key>city</arg_key>"
                "<arg_value>Beijing</arg_value></tool_call>",
            ],
            tools=[_function_tool()],
        )

        assert _items(events, "response.output_item.done", "message") == []
        assert _payloads(events, "response.reasoning_text.done") == ["Deciding what to do."]
        assert [c.name for c in _call_items(events)] == ["get_time"]

    def test_reasoning_item_closes_before_the_call_item_is_added(self):
        """Edge case 3, on ordering: no call is nested inside another item.

        A client that is still inside a reasoning item when a function_call
        item arrives attributes the call to the reasoning.
        """
        events = _drive(
            [
                "Deciding what to do.</think>",
                "<tool_call>get_time<arg_key>city</arg_key>"
                "<arg_value>Beijing</arg_value></tool_call>",
            ],
            tools=[_function_tool()],
        )

        types = [getattr(e, "type", None) for e in events]
        reasoning_done = types.index("response.reasoning_text.done")
        call_added = next(
            i
            for i, e in enumerate(events)
            if getattr(e, "type", None) == "response.output_item.added"
            and getattr(e.item, "type", None) == "function_call"
        )
        assert reasoning_done < call_added

    def test_text_after_a_call_opens_a_new_message_item(self):
        """Edge case 4, constructed - no recorded response has text after a call.

        All 35 multi-call records have nothing between `</tool_call>` and the
        next `<tool_call>`, and nothing after the last one, so this shape only
        exists here.
        """
        events = _drive(
            [
                "Thinking.</think>Before. ",
                "<tool_call>get_time<arg_key>city</arg_key>"
                "<arg_value>Beijing</arg_value></tool_call>",
                "After.",
            ],
            tools=[_function_tool()],
        )

        _assert_no_markup(events)
        assert _payloads(events, "response.output_text.done") == ["Before. ", "After."]

    def test_text_in_the_same_chunk_as_the_closing_call_tag(self):
        """Edge case 4 again, with the call and the trailing text in one chunk.

        The parser keeps everything after `</tool_call>` in its buffer and only
        returns it on the next increment, so a stream that ends on that chunk
        relies on the end-of-stream flush to release it. Without the flush the
        sentence is silently dropped.
        """
        events = _drive(
            [
                "Thinking.</think>Before. ",
                "<tool_call>get_time<arg_key>city</arg_key>"
                "<arg_value>Beijing</arg_value></tool_call>After.",
            ],
            tools=[_function_tool()],
        )

        _assert_no_markup(events)
        assert _payloads(events, "response.output_text.done") == ["Before. ", "After."]

    def test_whitespace_only_delta_between_calls_opens_a_message_item(self):
        r"""Edge case 5, recording what happens rather than what should.

        SGLang suppresses a whitespace-only message between calls because
        qwen3-coder separates its calls with `\n`. GLM does not, so no GLM
        traffic produces this, and suppressing it here would contradict the
        rule directly above it in _generate_streaming_event: a whitespace-only
        *first* delta must still open an item, or Codex CLI reports
        "OutputTextDelta without active item" and prints nothing at all.
        Reconciling the two needs its own decision, so this test pins the
        current behaviour and will fail loudly if someone changes it by
        accident.
        """
        events = _drive(
            [
                "Thinking.</think>",
                "<tool_call>get_time<arg_key>city</arg_key>"
                "<arg_value>Beijing</arg_value></tool_call>",
                "\n",
                "<tool_call>get_time<arg_key>city</arg_key>"
                "<arg_value>Paris</arg_value></tool_call>",
            ],
            tools=[_function_tool()],
        )

        _assert_no_markup(events)
        assert _payloads(events, "response.output_text.done") == ["\n"]
        assert len(_call_items(events)) == 2


class TestEndOfStream:
    """Edge cases 6 and 7: streams that stop before the model was done."""

    def test_stream_cut_off_mid_call_releases_the_markup_as_text(self):
        """Edge case 6: the parser is still holding an unterminated call.

        Dropping those bytes was the first design, but the final response
        never dropped them: its whole-text re-parse needs the closing tag,
        fails to read the fragment as a call, and falls back to publishing it
        as message text - so the snapshot carried a 429-character message the
        stream never showed (trace tr_8f312e9973954fa784ae7dc8f45e9d3e). The
        views have to agree, and of the two ways to agree, keeping the text
        wins: these turns are recorded as training data, where silent loss is
        worse than a call that visibly never closed. So the flush releases
        the raw buffer as a trailing message item. The text already streamed
        is unaffected, and the item carrying the fragment is separate because
        the message item closed when the call was announced, chunks earlier.
        """
        raw = "".join(_REAL_FRAMES[:3])
        unterminated = raw[raw.index("<tool_call>") :]

        with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
            events = _drive(_REAL_FRAMES[:3])

        assert _payloads(events, "response.output_text.done") == [
            _REAL_TEXT,
            unterminated,
        ]
        # The release has to be visible in the log, not silent: the model
        # emitted a call it never finished, and someone reading the recorded
        # turn should find that said somewhere.
        assert any("tool call" in call.args[0] for call in mock_logger.warning.call_args_list)

    def test_stream_cut_off_mid_call_reports_no_half_built_call(self):
        """A call whose arguments never closed is not a call a client can run.

        Its accumulated arguments are a JSON prefix. The non-streaming
        endpoint reports no call for the same text, and the two must agree.
        """
        events = _drive(_REAL_FRAMES[:3])
        assert _call_items(events) == []

    def test_stream_cut_off_after_one_call_keeps_that_call(self):
        """Dropping the unfinished call must not drop the finished ones.

        Frames 0-6 close the first call and open the second, so exactly one
        call survives.
        """
        events = _drive(_REAL_FRAMES[:7])
        assert [c.name for c in _call_items(events)] == ["exec"]
        assert json.loads(_call_items(events)[0].arguments)["input"]

    def test_abort_closes_the_open_item(self):
        """Edge case 7: an aborted stream still finalises what it had.

        An abort surfaces as a final chunk carrying no new text, and an item
        left open has no terminal state - Codex CLI echoes it back on the next
        turn without a `status`, which the next request is rejected for.
        """
        events = _drive(["Thinking.</think>Partial answer", ""])

        assert _payloads(events, "response.output_text.done") == ["Partial answer"]
        assert len(_items(events, "response.output_item.done", "message")) == 1


class TestUnchangedPaths:
    """Edge cases 10 and 11, and the regressions the structure already fixed."""

    def test_no_tool_parser_leaves_the_text_alone(self):
        """Edge case 10: with no tool parser, nothing is a call.

        The markup is ordinary text on this path, and it still has to come
        back as exactly what was streamed.
        """
        events = _drive(["Thinking.</think>Answer <tool_call>x</tool_call>"], tool_parser_id=None)

        assert _payloads(events, "response.output_text.done") == ["Answer <tool_call>x</tool_call>"]
        assert _call_items(events) == []

    def test_empty_delta_opens_nothing(self):
        """Edge case 11."""
        assert _drive([""], finish=False) == []

    def test_reasoning_and_text_in_one_chunk_still_emits_the_reasoning(self):
        """Regression: the chunk that spans `</think>` carries both halves.

        The reasoning part has to be flushed in the delta branch or it is
        never emitted; measured at 56% of reasoning items across four fleets,
        and 100% of reasoning short enough to fit one chunk.
        """
        events = _drive(["I should answer.</think>Answer."])

        assert _payloads(events, "response.reasoning_text.done") == ["I should answer."]
        assert _payloads(events, "response.output_text.done") == ["Answer."]

    def test_item_is_opened_before_a_whitespace_only_first_delta(self):
        """Regression: a delta with no item open makes a client drop the turn.

        Codex CLI reports "OutputTextDelta without active item" and prints
        nothing. Short replies hit it, because a leading whitespace token is
        more likely to be the whole first delta.
        """
        events = _drive(["Thinking.</think> ", "hi"])

        types = [getattr(e, "type", None) for e in events]
        first_delta = types.index("response.output_text.delta")
        message_added = next(
            i
            for i, e in enumerate(events)
            if getattr(e, "type", None) == "response.output_item.added"
            and getattr(e.item, "type", None) == "message"
        )
        assert message_added < first_delta
        assert _payloads(events, "response.output_text.done") == [" hi"]

    def test_message_item_is_closed_before_the_call_item_is_added(self):
        """Regression: a call must never be nested inside a message item."""
        events = _drive(_REAL_FRAMES)

        message_done = next(
            i
            for i, e in enumerate(events)
            if getattr(e, "type", None) == "response.output_item.done"
            and getattr(e.item, "type", None) == "message"
        )
        call_added = next(
            i
            for i, e in enumerate(events)
            if getattr(e, "type", None) == "response.output_item.added"
            and getattr(e.item, "type", None) == "function_call"
        )
        assert message_done < call_added

    def test_calls_are_emitted_once_if_the_finished_output_arrives_twice(self):
        """Regression: double emission must stay impossible.

        `emitted_tool_calls` existed because the old per-chunk re-parse
        re-reported the same calls on every chunk. Emission is driven by
        accumulated fragments now, so the counter guards a different thing -
        the emission block running for more than one chunk of the same
        output - but it still has to guard it.
        """
        helper = ResponsesStreamingEventsHelper()
        request = _make_request([_namespace_tool()])
        reasoning_parser_dict, tool_parser_dict = {}, {}

        def step(text_diff, accumulated, finished):
            return list(
                _generate_streaming_event(
                    output=_make_output(accumulated, text_diff),
                    request=request,
                    finished_generation=finished,
                    streaming_events_helper=helper,
                    reasoning_parser_id=_REASONING_PARSER,
                    tool_parser_id=_TOOL_PARSER,
                    reasoning_parser_dict=reasoning_parser_dict,
                    tool_parser_dict=tool_parser_dict,
                )
            )

        events, accumulated = [], ""
        for frame in _REAL_FRAMES:
            accumulated += frame
            events += step(frame, accumulated, frame is _REAL_FRAMES[-1])
        assert len(_call_items(events)) == 2

        # A second finished chunk for the same output carries no new text, and
        # must not re-report the calls already emitted for it.
        assert _call_items(step("", accumulated, True)) == []


class TestFragmentAssembly:
    """The accumulator that turns parser fragments back into whole calls."""

    def test_fragments_are_concatenated_in_order(self):
        fragments: dict = {}
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

    def test_a_later_fragment_does_not_erase_the_name(self):
        """Only a call's first fragment carries its name; the rest carry None."""
        fragments: dict = {}
        _accumulate_tool_call_fragments(
            fragments,
            [
                ToolCallItem(tool_index=0, name="get_time", parameters=""),
                ToolCallItem(tool_index=0, name=None, parameters="{}"),
            ],
        )
        assert _assembled_tool_calls(fragments)[0].name == "get_time"

    def test_the_unfinished_call_is_dropped_and_the_finished_ones_kept(self):
        fragments: dict = {}
        _accumulate_tool_call_fragments(
            fragments,
            [
                ToolCallItem(tool_index=0, name="get_time", parameters='{"a": 1}'),
                ToolCallItem(tool_index=1, name="get_time", parameters='{"a": '),
            ],
        )

        assembled = _assembled_tool_calls(fragments, unfinished_tool_index=1)
        assert [c.parameters for c in assembled] == ['{"a": 1}']

    def test_a_call_whose_name_never_arrived_is_dropped(self):
        """Matches parse_base_json on the non-streaming path."""
        fragments: dict = {}
        _accumulate_tool_call_fragments(
            fragments, [ToolCallItem(tool_index=0, parameters='{"a": 1}')]
        )
        assert _assembled_tool_calls(fragments) == []

    def test_fragments_are_kept_apart_per_output(self):
        """Edge case 8: one parser instance per output, one fragment set each.

        `tool_parser_dict` is keyed by output index, so two outputs of the same
        request number their calls independently - both start at tool_index 0 -
        and merging them would concatenate one output's arguments onto the
        other's.
        """
        helper = ResponsesStreamingEventsHelper()
        _accumulate_tool_call_fragments(
            helper.tool_call_fragments(0),
            [ToolCallItem(tool_index=0, name="first", parameters='{"a": 1}')],
        )
        _accumulate_tool_call_fragments(
            helper.tool_call_fragments(1),
            [ToolCallItem(tool_index=0, name="second", parameters='{"b": 2}')],
        )

        assert [c.name for c in _assembled_tool_calls(helper.tool_call_fragments(0))] == ["first"]
        assert [c.name for c in _assembled_tool_calls(helper.tool_call_fragments(1))] == ["second"]

    def test_two_helpers_do_not_share_fragments(self):
        """Two requests in flight at once must not share an accumulator.

        A mutable default on the state tracker's class body would be one
        object shared by every request, so two concurrent streams would
        accumulate into each other's calls.
        """
        first, second = (ResponsesStreamingEventsHelper(), ResponsesStreamingEventsHelper())
        _accumulate_tool_call_fragments(
            first.tool_call_fragments(0),
            [ToolCallItem(tool_index=0, name="only_mine", parameters="{}")],
        )

        assert second.tool_call_fragments(0) == {}
