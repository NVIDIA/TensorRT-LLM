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
"""Offline tests: the Responses store must not lose tool calls or prior context.

Three ways a multi-turn tool conversation lost its thread, each pinned here:

* The store writer only attached tool calls to the reasoning message, so a
  turn that called tools without reasoning stored them nowhere - and a turn
  that was *nothing but* tool calls stored no assistant message at all.
* A stored reasoning+calls turn survived storage but not replay: the replay
  filter dropped the whole message for its "reasoning" key, calls included, so
  the client's tool RESULT replayed with no call before it - an orphan the
  model cannot pair.
* The server folded ``request.store`` into the one flag it hands
  preprocessing, so a follow-up sent with ``store=false`` had its
  ``previous_response_id`` loaded and validated - and then generated with no
  history at all, silently.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import ResponseFunctionToolCall

from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ConversationHistoryStore,
    _create_input_messages,
    _create_output_messages,
    request_preprocess,
)

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
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


def test_a_tool_only_turn_is_stored():
    """No text, no reasoning - the calls alone used to store nothing."""
    messages = _create_output_messages(_contents(calls=[_call()]))
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "exec"


def test_a_text_turn_keeps_its_calls():
    """Text plus calls without reasoning stored the text and dropped the calls."""
    messages = _create_output_messages(_contents(text="Running it.", calls=[_call()]))
    carried = [m for m in messages if m.get("tool_calls")]
    assert carried and carried[0]["content"] == "Running it."


def test_a_text_only_turn_stores_the_old_shape():
    assert _create_output_messages(_contents(text="Hello.")) == [
        {"role": "assistant", "content": "Hello."}
    ]


def test_a_reasoning_turn_still_carries_reasoning_and_calls():
    """The path that already worked, kept byte-compatible."""
    (message,) = _create_output_messages(_contents(reasoning="think", calls=[_call()]))
    assert message["reasoning"] == "think"
    assert message["tool_calls"][0]["id"] == "call_1"


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


def test_replaying_a_reasoning_only_turn_still_drops_it():
    stored = _create_output_messages(_contents(reasoning="think"))
    replayed = _replay(stored)
    assert all("reasoning" not in m and not m.get("tool_calls") for m in replayed)


def test_a_plain_assistant_call_message_replays_byte_identical():
    plain = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_9", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
        ],
    }
    replayed = _replay([plain])
    assert replayed[0] is plain


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
        # The follow-up itself opting out of storage must not cost it the
        # history it names.
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
    """Run the real request_preprocess with the heavy leaves faked.

    Chat-template rendering and sampling-params construction need a tokenizer
    and the engine bindings; the gates under test - fetch prior context, then
    persist or not - sit above both, so those leaves are replaced with
    recorders and the store itself stays real.
    """
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
        # The server-side switch, which is all the server passes: retrieval is
        # gated downstream on previous_response_id, persistence on
        # request.store.
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


def test_store_false_still_sees_prior_context_and_is_not_persisted(monkeypatch):
    saw_prior, persisted = asyncio.run(_preprocess(monkeypatch, store_flag=False))
    assert saw_prior, "previous_response_id was loaded and then ignored"
    assert not persisted


def test_store_true_sees_prior_context_and_is_persisted(monkeypatch):
    saw_prior, persisted = asyncio.run(_preprocess(monkeypatch, store_flag=True))
    assert saw_prior
    assert persisted


def test_the_endpoint_hands_preprocess_the_server_switch(monkeypatch):
    """A store=false request must not flip the flag preprocessing receives.

    The endpoint used to pass ``enable_store and request.store``, which reads
    as one switch but gates two: it suppressed retrieval of the prior context
    along with persistence. Only the wiring is under test, so preprocessing is
    replaced with a recorder that stops the handler right after the handoff.
    """
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

    request = ResponsesRequest(
        model="test-model", input="hi", previous_response_id="resp_prev", store=False
    )
    raw_request = SimpleNamespace(
        state=SimpleNamespace(),
        headers={},
        url=SimpleNamespace(path="/v1/responses"),
        json=AsyncMock(return_value={}),
    )
    asyncio.run(server.openai_responses(request, raw_request))

    assert captured.get("enable_store") is True
    assert captured["request"].store is False
