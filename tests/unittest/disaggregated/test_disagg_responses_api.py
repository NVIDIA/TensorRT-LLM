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
"""Offline tests for the Responses API on the disaggregated path.

The disaggregated orchestrator holds no tokenizer and no engine: it splits one
client request into a context request and a generation request, relays a
KV-cache handle between two workers, and streams the second worker's bytes
back. These tests pin the parts of that split that differ for Responses,
because the Completions and Chat Completions shapes carry the handoff on
``choices[0]`` and a Responses response has no ``choices`` at all.
"""

import asyncio

import pytest

from tensorrt_llm.serve.openai_disagg_service import _ctx_handoff_slots, _drop_ctx_handoff
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    DisaggregatedParams,
    ResponsesRequest,
    ResponsesResponse,
)

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _responses_response(finish_reason="length", disagg=True, prompt_token_ids=None):
    return ResponsesResponse(
        model="m",
        output=[],
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        service_tier="auto",
        status="incomplete",
        top_logprobs=0,
        truncation="disabled",
        finish_reason=finish_reason,
        prompt_token_ids=prompt_token_ids,
        disaggregated_params=DisaggregatedParams(
            request_type="context_only",
            ctx_request_id=7,
            disagg_request_id=7,
        )
        if disagg
        else None,
    )


# ---------------------------------------------------------------------------
# The handoff carrier
# ---------------------------------------------------------------------------


def test_responses_response_carries_the_handoff_at_top_level():
    """No ``choices`` to hang it on, so the orchestrator reads it directly."""
    response = _responses_response()
    slots = _ctx_handoff_slots(response)
    assert slots == [response]
    assert slots[0].disaggregated_params.ctx_request_id == 7


def test_chat_response_still_carries_the_handoff_per_choice():
    """The pre-existing shape must keep working unchanged."""
    response = ChatCompletionResponse(
        model="m",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "length",
                "disaggregated_params": {
                    "request_type": "context_only",
                    "ctx_request_id": 3,
                },
            }
        ],
        usage={"prompt_tokens": 1, "total_tokens": 2, "completion_tokens": 1},
    )
    slots = _ctx_handoff_slots(response)
    assert len(slots) == 1
    assert slots[0].disaggregated_params.ctx_request_id == 3


def test_dropping_the_handoff_clears_it():
    """A response that never reaches a generation worker must not leak it.

    The field names an internal KV-cache transfer; relaying it to the client
    would describe a transfer that never happens.
    """
    response = _responses_response()
    _drop_ctx_handoff(response)
    assert response.disaggregated_params is None


# ---------------------------------------------------------------------------
# finish_reason: why `status` is not enough
# ---------------------------------------------------------------------------


def test_finish_reason_is_carried_unmapped():
    """``status`` cannot express the distinction the orchestrator needs.

    "length" and "not_finished" both map onto the public status "incomplete",
    but only the raw value tells the orchestrator a generation phase is still
    pending, and ``finish_reason_mapping`` used to raise outright on
    "not_finished".
    """
    response = _responses_response(finish_reason="not_finished")
    assert response.status == "incomplete"
    assert response.finish_reason == "not_finished"


def test_not_finished_maps_to_incomplete_rather_than_raising():
    """Regression: the context worker is capped at one token.

    It hands the request off without finishing, so the engine reports
    "not_finished" -- a value the mapping did not handle, which surfaced as a
    500 from the context worker rather than a completed handoff.
    """
    from tensorrt_llm.serve.responses_utils import finish_reason_mapping

    assert finish_reason_mapping("not_finished") == "incomplete"
    assert finish_reason_mapping("length") == "incomplete"
    assert finish_reason_mapping("stop") == "completed"


def test_an_unknown_finish_reason_names_itself():
    from tensorrt_llm.serve.responses_utils import finish_reason_mapping

    with pytest.raises(RuntimeError, match="wat"):
        finish_reason_mapping("wat")


# ---------------------------------------------------------------------------
# The relayed prompt
# ---------------------------------------------------------------------------


def test_relayed_prompt_token_ids_are_used_verbatim():
    """The generation worker must not re-render the chat template.

    Re-rendering is not merely wasted work: a template that varies per render
    would produce a prompt the context worker never prefilled, and the
    relayed KV cache would not correspond to it.
    """
    from tensorrt_llm.serve.responses_utils import _relayed_prompt_token_ids

    request = ResponsesRequest(model="m", input="hi", prompt_token_ids=[1, 2, 3])
    assert _relayed_prompt_token_ids(request) == [1, 2, 3]


def test_relayed_prompt_token_ids_b64_round_trips():
    """The orchestrator relays one base64 string instead of an int list."""
    import base64

    import numpy as np

    from tensorrt_llm.serve.responses_utils import _relayed_prompt_token_ids

    encoded = base64.b64encode(np.asarray([5, 6, 7], dtype=np.int32).tobytes()).decode("ascii")
    request = ResponsesRequest(model="m", input="hi", prompt_token_ids_b64=encoded)
    assert _relayed_prompt_token_ids(request) == [5, 6, 7]


def test_an_ordinary_client_request_has_no_relayed_prompt():
    """A client sends neither field, so the worker tokenizes as usual."""
    from tensorrt_llm.serve.responses_utils import _relayed_prompt_token_ids

    assert _relayed_prompt_token_ids(ResponsesRequest(model="m", input="hi")) is None


# ---------------------------------------------------------------------------
# Streaming a request that never reached a generation worker
# ---------------------------------------------------------------------------


def _drain(generator):
    async def collect():
        return [chunk async for chunk in generator]

    return asyncio.run(collect())


def test_ctx_only_stream_ends_with_response_completed():
    """Regression: the completions terminator hangs a Responses client.

    The Responses protocol has no ``[DONE]`` sentinel -- a client watches for
    ``response.completed``. Emitting ``data: [DONE]`` leaves it waiting for an
    event that never arrives, and drops the answer the context worker already
    produced.
    """
    from tensorrt_llm.serve.responses_utils import responses_done_generator

    chunks = _drain(responses_done_generator(_responses_response()))
    body = b"".join(chunks).decode("utf-8")

    assert "event: response.completed" in body
    assert "[DONE]" not in body
    # A well-formed run opens before it closes.
    assert body.index("response.created") < body.index("response.completed")


def test_ctx_only_stream_events_are_sse_framed():
    from tensorrt_llm.serve.responses_utils import responses_done_generator

    chunks = _drain(responses_done_generator(_responses_response()))
    for chunk in chunks:
        text = chunk.decode("utf-8")
        assert text.startswith("event: ")
        assert "\ndata: " in text
        assert text.endswith("\n\n")


# ---------------------------------------------------------------------------
# Routing a Responses request to the right worker endpoint
# ---------------------------------------------------------------------------


def test_client_dispatches_responses_requests_to_v1_responses():
    """The endpoint is chosen by request type; an unmapped type raises.

    Without this branch a Responses request reaching the client raised
    "Invalid request type", which is how the route would have failed had only
    the orchestrator side been wired up.
    """
    import inspect

    from tensorrt_llm.serve.openai_client import OpenAIClient

    source = inspect.getsource(OpenAIClient.send_request)
    assert "ResponsesRequest" in source
    assert "v1/responses" in source


# ---------------------------------------------------------------------------
# Fields the orchestrator needs on the wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "disaggregated_params",
        "conversation_params",
        "prompt_token_ids",
        "prompt_token_ids_b64",
    ],
)
def test_request_carries_the_orchestrator_fields(field):
    """``_wrap_entry_point`` and the split read these off every request.

    ``conversation_params`` in particular is read unconditionally by
    conversation-id resolution, so its absence is an AttributeError on the
    first request rather than a missing feature.
    """
    request = ResponsesRequest(model="m", input="hi")
    assert hasattr(request, field)
    assert getattr(request, field) is None


def test_request_fields_survive_serialization_to_a_worker():
    """The client serializes with exclude_unset, so set fields must persist."""
    request = ResponsesRequest(model="m", input="hi")
    request.disaggregated_params = DisaggregatedParams(
        request_type="generation_only", ctx_request_id=11
    )
    request.prompt_token_ids = [1, 2]

    payload = request.model_dump_json(exclude_unset=True)
    assert "generation_only" in payload
    assert "prompt_token_ids" in payload

    revived = ResponsesRequest.model_validate_json(payload)
    assert revived.disaggregated_params.ctx_request_id == 11
    assert revived.prompt_token_ids == [1, 2]
