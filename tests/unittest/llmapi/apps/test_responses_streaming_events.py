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
"""Offline tests for Responses API streaming event construction.

Clients track streamed content by output item: a delta is attached to the
item whose id it carries, and an item is only usable once it has been opened
with response.output_item.added and closed with response.output_item.done.
Codex CLI drops an entire turn - printing nothing - if a delta arrives with
an id it has not seen opened.
"""

import pytest

from tensorrt_llm.serve.responses_utils import ResponsesStreamingEventsHelper

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _helper():
    return ResponsesStreamingEventsHelper()


# ---------------------------------------------------------------------------
# Item identity
# ---------------------------------------------------------------------------


def test_message_item_gets_a_non_empty_id():
    """Regression: current_item_id had no writer, so every event carried "".

    A delta whose item_id is empty matches no item the client has opened, so
    the client discards the text and the user sees no reply at all.
    """
    helper = _helper()
    list(helper.get_message_output_added_events())
    assert helper.item_id
    assert helper.item_id.startswith("msg_")


def test_reasoning_item_gets_a_non_empty_id():
    helper = _helper()
    list(helper.get_reasoning_output_added_events())
    assert helper.item_id.startswith("rs_")


def test_events_share_the_open_item_id():
    helper = _helper()
    added = list(helper.get_message_output_added_events())
    item_id = helper.item_id
    assert added[0].item.id == item_id
    assert added[1].item_id == item_id
    assert helper.get_text_delta_event("hello", []).item_id == item_id


def test_a_new_item_gets_a_new_id():
    """Regression: closing an item left item_id set, so the next item reused it.

    Two output items sharing one id makes the stream ambiguous for a client
    keying its state on item_id.
    """
    helper = _helper()
    list(helper.get_message_output_added_events())
    first = helper.item_id

    # What the close path does.
    helper.is_output_item_added_sent = False
    helper.output_index_increment()

    list(helper.get_message_output_added_events())
    assert helper.item_id != first
    assert helper.item_id.startswith("msg_")


def test_added_events_are_idempotent_while_an_item_is_open():
    """Deltas may call the opener every time; only the first must emit."""
    helper = _helper()
    first = list(helper.get_message_output_added_events())
    second = list(helper.get_message_output_added_events())
    assert len(first) == 2  # output_item.added + content_part.added
    assert second == []


# ---------------------------------------------------------------------------
# Text accumulation used to close an item at end of generation
# ---------------------------------------------------------------------------


def test_text_buffer_accumulates_and_drains():
    helper = _helper()
    helper.append_text("Hello")
    helper.append_text(" world")
    assert helper.take_text() == "Hello world"
    # Draining leaves nothing behind, so a later close cannot re-emit it.
    assert helper.take_text() == ""


def test_reasoning_buffer_is_separate_from_text():
    helper = _helper()
    helper.append_text("answer")
    helper.append_reasoning("thinking")
    assert helper.take_reasoning() == "thinking"
    assert helper.take_text() == "answer"


def test_done_events_carry_the_open_item_id():
    helper = _helper()
    list(helper.get_message_output_added_events())
    item_id = helper.item_id
    assert helper.get_text_done_event("hi", []).item_id == item_id
    assert helper.get_content_part_done_event(_output_text("hi")).item_id == item_id


def _output_text(text):
    from openai.types.responses import ResponseOutputText

    return ResponseOutputText(text=text, annotations=[], type="output_text", logprobs=None)
