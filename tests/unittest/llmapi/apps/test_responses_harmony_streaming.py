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
"""Responses API streaming of Harmony (gpt-oss) output.

A response carries several tokens when stream_interval > 1 or when draft tokens
are accepted. The streamed events must describe the output the same way however
its tokens are grouped into responses.
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.result import CompletionOutput
from tensorrt_llm.serve.harmony_adapter import get_harmony_adapter
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import ResponsesStreamingProcessor

pytestmark = pytest.mark.cpu_only

_REASONING = "The user asks where the order is. Look it up, then answer briefly."
_ANSWER = "Your order #W2378156 shipped on Monday 🚀 and arrives by Thursday."

_OUTPUTS = {
    "reasoning_then_answer": (
        f"<|channel|>analysis<|message|>{_REASONING}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>{_ANSWER}<|return|>"
    ),
    "preamble_then_tool_call": (
        f"<|channel|>analysis<|message|>{_REASONING}<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>Checking the order.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_order <|constrain|>json"
        '<|message|>{"order_id": "#W2378156"}<|call|>'
    ),
    # Cut off by max_tokens: the last item is never closed.
    "truncated_answer": (
        f"<|channel|>analysis<|message|>{_REASONING}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>{_ANSWER[:30]}"
    ),
}


def _stream(text, tokens_per_response):
    """Stream one Harmony output; returns the event payloads of each response."""
    adapter = get_harmony_adapter()
    tokens = adapter.encoding.encode(text, allowed_special="all")
    request = ResponsesRequest(model="gpt-oss", input="Where is my order?", stream=True)
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="gpt-oss",
        use_harmony=True,
    )
    result = SimpleNamespace(outputs=[CompletionOutput(index=0)], _done=False)
    output = result.outputs[0]
    responses = []
    try:
        for start in range(0, len(tokens), tokens_per_response):
            output._last_token_ids_len = len(output.token_ids)
            output.token_ids.extend(tokens[start : start + tokens_per_response])
            result._done = start + tokens_per_response >= len(tokens)
            frames = processor.process_single_output(result)
            responses.append([json.loads(frame.split("data: ", 1)[1]) for frame in frames])
    finally:
        adapter.cleanup_stream_state(processor.stream_request_id)
    return responses


def _client_view(responses):
    """The events as a client reads them: deltas of one content part joined, no sequence numbers."""
    view = []
    for event in (event for events in responses for event in events):
        event = {key: value for key, value in event.items() if key != "sequence_number"}
        last = view[-1] if view else None
        if (
            last is not None
            and event["type"].endswith(".delta")
            and event["type"] == last["type"]
            and event["output_index"] == last["output_index"]
        ):
            last["delta"] += event["delta"]
        else:
            view.append(event)
    return view


def _deltas(responses, event_type):
    return "".join(e["delta"] for events in responses for e in events if e["type"] == event_type)


def test_one_token_per_response_streams_reasoning_then_answer(harmony_vocab):
    responses = _stream(_OUTPUTS["reasoning_then_answer"], 1)

    assert _deltas(responses, "response.reasoning_text.delta") == _REASONING
    assert _deltas(responses, "response.output_text.delta") == _ANSWER


@pytest.mark.parametrize("tokens_per_response", [2, 3, 20])
@pytest.mark.parametrize("output", sorted(_OUTPUTS))
def test_events_do_not_depend_on_tokens_per_response(harmony_vocab, output, tokens_per_response):
    expected = _client_view(_stream(_OUTPUTS[output], 1))

    assert _client_view(_stream(_OUTPUTS[output], tokens_per_response)) == expected


def test_a_response_sends_one_delta_per_content_part(harmony_vocab):
    responses = _stream(_OUTPUTS["reasoning_then_answer"], 20)

    assert _deltas(responses, "response.reasoning_text.delta") == _REASONING
    assert _deltas(responses, "response.output_text.delta") == _ANSWER
    for events in responses:
        parts = [(e["type"], e["output_index"]) for e in events if e["type"].endswith(".delta")]
        assert len(parts) == len(set(parts))


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_parsed_messages_follow_the_parser(harmony_vocab, batch_size):
    """Messages are converted again only after a stop token, and never go stale."""
    adapter = get_harmony_adapter()
    tokens = adapter.encoding.encode(_OUTPUTS["preamble_then_tool_call"], allowed_special="all")
    request_id = f"messages-follow-parser-{batch_size}"
    state = adapter.create_stream_state(request_id)
    try:
        for start in range(0, len(tokens), batch_size):
            messages = state.process_token_batch_to_messages(tokens[start : start + batch_size])
            assert messages == state.get_parser().messages
        assert len(messages) == 3
    finally:
        adapter.cleanup_stream_state(request_id)
