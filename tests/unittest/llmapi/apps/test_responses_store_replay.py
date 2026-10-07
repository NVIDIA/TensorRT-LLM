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
"""Offline tests for Responses storage and replay, and the endpoint up to preprocessing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai.types.responses import ResponseFunctionToolCall

from tensorrt_llm.serve.openai_protocol import DisaggregatedParams, ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ConversationHistoryStore,
    _create_input_messages,
    _create_output_messages,
    request_preprocess,
)

pytestmark = pytest.mark.cpu_only


def _call(call_id="call_1", name="exec", arguments='{"cmd": "ls"}'):
    return ResponseFunctionToolCall(
        type="function_call",
        id=f"fc_{call_id}",
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
    )


def _tool_result_item(call_id="call_1", output="ok"):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _contents(text=None, reasoning=None, calls=()):
    return {
        "text_content": text,
        "reasoning_content": reasoning,
        "tool_calls": list(calls),
    }


# ---------------------------------------------------------------------------
# The store writer: a tool-call-bearing turn is an assistant turn
# ---------------------------------------------------------------------------

_STORED_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "exec", "arguments": '{"cmd": "ls"}'},
}


@pytest.mark.parametrize(
    "contents, expected",
    [
        (
            _contents(calls=[_call()]),
            [{"role": "assistant", "content": None, "tool_calls": [_STORED_CALL]}],
        ),
        (
            _contents(text="Running it.", calls=[_call()]),
            [{"role": "assistant", "content": "Running it.", "tool_calls": [_STORED_CALL]}],
        ),
        (_contents(text="Hello."), [{"role": "assistant", "content": "Hello."}]),
    ],
    ids=["calls_only", "text_and_calls", "text_only"],
)
def test_a_turn_is_stored_with_its_calls(contents, expected):
    assert _create_output_messages(contents) == expected


def test_a_reasoning_turn_carries_reasoning_and_calls():
    (message,) = _create_output_messages(_contents(reasoning="think", calls=[_call()]))
    assert message["reasoning"] == "think"
    assert message["tool_calls"] == [_STORED_CALL]


# ---------------------------------------------------------------------------
# Replay: strip the reasoning, keep the calls
# ---------------------------------------------------------------------------


def _replay(prev_msgs, request_input="hi", **request_overrides):
    request = ResponsesRequest(model="m", input=request_input, **request_overrides)
    return asyncio.run(_create_input_messages(request=request, prev_msgs=prev_msgs))


def test_replaying_a_reasoning_call_turn_preserves_the_calls():
    stored = _create_output_messages(_contents(reasoning="think", calls=[_call()]))
    replayed = _replay(stored)
    assert all("reasoning" not in m for m in replayed)
    carried = [m for m in replayed if m.get("tool_calls")]
    assert carried and carried[0]["tool_calls"][0]["id"] == "call_1"


def test_replay_drops_a_reasoning_only_turn_and_keeps_others_as_stored():
    plain = {"role": "assistant", "content": None, "tool_calls": [_STORED_CALL]}
    replayed = _replay(_create_output_messages(_contents(reasoning="think")) + [plain])
    assert replayed[0] is plain
    assert replayed[1:] == [{"role": "user", "content": "hi"}]


def test_trimming_a_stored_conversation_leaves_the_callers_messages_whole():
    store = ConversationHistoryStore(resp_capacity=1)
    messages = [{"role": ("user", "assistant")[i % 2], "content": str(i)} for i in range(8)]
    original = list(messages)
    asyncio.run(store.store_messages("resp_1", messages, None))
    assert messages == original
    assert len(store.conversations[store.response_to_conversation["resp_1"]]) < len(original)


# ---------------------------------------------------------------------------
# The full chain: write -> store -> read via previous_response_id -> tool result
# ---------------------------------------------------------------------------


async def _chain(output_contents, call_id):
    store = ConversationHistoryStore()
    first = ResponsesRequest(model="m", input="do the thing", request_id="resp_first")
    input_msgs = await _create_input_messages(request=first, prev_msgs=[])
    await store.store_messages("resp_first", input_msgs, None)

    output_msgs = _create_output_messages(output_contents)
    await store.store_response(
        resp=SimpleNamespace(id="resp_first"), resp_msgs=output_msgs, prev_resp_id=None
    )

    prev_msgs = await store.get_conversation_history("resp_first")
    follow_up = ResponsesRequest(
        model="m",
        input=[_tool_result_item(call_id=call_id)],
        previous_response_id="resp_first",
        store=False,
        request_id="resp_second",
    )
    return await _create_input_messages(request=follow_up, prev_msgs=prev_msgs)


@pytest.mark.parametrize(
    "contents,call_id",
    [
        (_contents(calls=[_call()]), "call_1"),
        (_contents(reasoning="let me think", calls=[_call(call_id="call_2")]), "call_2"),
    ],
    ids=["tool-only-turn", "reasoning-plus-call-turn"],
)
def test_the_stored_call_precedes_its_result_on_replay(contents, call_id):
    messages = asyncio.run(_chain(contents, call_id))
    call_pos = [
        i
        for i, m in enumerate(messages)
        if any(c.get("id") == call_id for c in (m.get("tool_calls") or []))
    ]
    result_pos = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    ]
    assert call_pos and result_pos, f"orphaned result: {messages}"
    assert call_pos[0] < result_pos[0]
    assert all("reasoning" not in m for m in messages)


# ---------------------------------------------------------------------------
# The two store switches are independent
# ---------------------------------------------------------------------------


async def _preprocess(monkeypatch, store_flag):
    """Run the real request_preprocess with rendering faked; the store is real."""
    import tensorrt_llm.serve.responses_utils as ru

    store = ConversationHistoryStore()
    await store.store_messages(
        "resp_prev",
        [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "earlier answer"},
        ],
        None,
    )

    captured = {}

    def fake_parse(messages, model_config):
        captured["messages"] = list(messages)

        async def mm():
            return None, None

        return list(messages), mm(), {}, None

    async def fake_template(**kwargs):
        return [1, 2, 3]

    monkeypatch.setattr(ru, "parse_chat_messages_coroutines", fake_parse)
    monkeypatch.setattr(ru, "async_apply_chat_template", fake_template)
    monkeypatch.setattr(ru, "resolve_top_level_model_type", lambda cfg: "llama")
    monkeypatch.setattr(ru, "add_thinking_budget_logits_processor", lambda *a, **k: None)
    monkeypatch.setattr(
        ResponsesRequest, "to_sampling_params", lambda self, **kw: SimpleNamespace()
    )

    request = ResponsesRequest(
        model="m",
        input="next turn",
        previous_response_id="resp_prev",
        store=store_flag,
        request_id="resp_new",
    )
    await request_preprocess(
        request=request,
        prev_response=None,
        conversation_store=store,
        enable_store=True,
        use_harmony=False,
        tokenizer=None,
        model_config=None,
        processor=None,
        reasoning_parser=None,
    )
    saw_prior = any(m.get("content") == "earlier answer" for m in captured["messages"])
    persisted = "resp_new" in store.response_to_conversation
    return saw_prior, persisted


@pytest.mark.parametrize("store_flag", [False, True])
def test_prior_context_is_read_whether_or_not_the_turn_is_stored(monkeypatch, store_flag):
    saw_prior, persisted = asyncio.run(_preprocess(monkeypatch, store_flag=store_flag))
    assert saw_prior
    assert persisted == store_flag


def _server_up_to_preprocess(monkeypatch):
    """An OpenAIServer whose /v1/responses handler stops at preprocessing."""
    import tensorrt_llm.serve.openai_server as server_module
    from tensorrt_llm.serve.openai_server import OpenAIServer

    server = object.__new__(OpenAIServer)
    server.model = "test-model"
    server.enable_store = True
    server._is_visual_gen = False
    server.use_harmony = False
    server.tokenizer = None
    server.model_config = None
    server.processor = None
    server.tool_parser = None
    server.metrics_collector = None
    server.generator = SimpleNamespace(
        args=SimpleNamespace(reasoning_parser=None, num_postprocess_workers=0)
    )
    server.conversation_store = ConversationHistoryStore()
    asyncio.run(
        server.conversation_store.store_response(
            resp=SimpleNamespace(id="resp_prev"), resp_msgs=[], prev_resp_id=None
        )
    )

    captured = {}

    async def fake_preprocess(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after the handoff")

    monkeypatch.setattr(server_module, "responses_api_request_preprocess", fake_preprocess)
    raw_request = SimpleNamespace(
        state=SimpleNamespace(),
        headers={},
        url=SimpleNamespace(path="/v1/responses"),
        json=AsyncMock(return_value={}),
    )
    return server, raw_request, captured


def test_the_endpoint_hands_preprocess_the_server_switch(monkeypatch):
    """A store=false request must not flip the flag preprocessing receives."""
    server, raw_request, captured = _server_up_to_preprocess(monkeypatch)
    request = ResponsesRequest(
        model="test-model", input="hi", previous_response_id="resp_prev", store=False
    )
    asyncio.run(server.openai_responses(request, raw_request))

    assert captured.get("enable_store") is True
    assert captured["request"].store is False


def test_protected_disagg_fields_are_checked_before_anything_is_rendered(monkeypatch):
    server, raw_request, captured = _server_up_to_preprocess(monkeypatch)
    server._validate_internal_disagg_request = Mock(side_effect=ValueError("unsigned"))
    request = ResponsesRequest(
        model="test-model",
        input="hi",
        disaggregated_params=DisaggregatedParams(request_type="generation_only"),
    )
    response = asyncio.run(server.openai_responses(request, raw_request))

    assert response.status_code == 400
    assert captured == {}


@pytest.mark.parametrize(
    "fields, warned",
    [({}, False), ({"parallel_tool_calls": True}, False), ({"parallel_tool_calls": False}, True)],
)
def test_unenforced_parallel_tool_calls_is_logged_only_when_requested(monkeypatch, fields, warned):
    server, raw_request, _ = _server_up_to_preprocess(monkeypatch)
    request = ResponsesRequest(model="test-model", input="hi", **fields)
    with patch("tensorrt_llm.serve.openai_server.logger") as mock_logger:
        asyncio.run(server.openai_responses(request, raw_request))
    keys = [c.kwargs.get("key") for c in mock_logger.warning_once.call_args_list]
    assert ("responses_parallel_tool_calls_unenforced" in keys) is warned
