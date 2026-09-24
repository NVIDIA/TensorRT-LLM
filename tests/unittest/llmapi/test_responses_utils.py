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

from types import SimpleNamespace

import pytest
from openai_harmony import Message

from tensorrt_llm.serve.responses_utils import ConversationHistoryStore


@pytest.fixture
def progress_checked_store(monkeypatch: pytest.MonkeyPatch) -> ConversationHistoryStore:
    store = ConversationHistoryStore()
    pop_conversation = store._pop_conversation_by_conversation_id

    def pop_with_progress_check(conversation_id: str) -> None:
        previous_length = len(store.conversations[conversation_id])
        pop_conversation(conversation_id)
        # Fail a non-progressing regression instead of hanging the test runner.
        assert len(store.conversations[conversation_id]) < previous_length

    monkeypatch.setattr(store, "_pop_conversation_by_conversation_id", pop_with_progress_check)
    return store


def _message(
    role: str, content: str, use_harmony: bool, channel: str | None = None
) -> Message | dict[str, str]:
    if not use_harmony:
        return {"role": role, "content": content}
    message = Message.from_role_and_content(role, content)
    return message.with_channel(channel) if channel is not None else message


def _turns(count: int):
    messages = []
    for index in range(count):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"user {index}",
                },
                {
                    "role": "assistant",
                    "content": f"assistant {index}",
                },
            ]
        )
    return messages


@pytest.mark.asyncio
async def test_store_response_trims_pre_stored_request_conversation():
    store = ConversationHistoryStore(resp_capacity=1)

    await store.store_messages("resp_1", _turns(3), prev_resp_id=None)
    await store.store_response(
        SimpleNamespace(id="resp_1"),
        [
            {
                "role": "assistant",
                "content": "final",
            }
        ],
        prev_resp_id=None,
    )

    conversation = await store.get_conversation_history("resp_1")

    assert len(conversation) <= store.conversation_capacity


@pytest.mark.asyncio
async def test_store_response_trims_previous_response_conversation(monkeypatch):
    store = ConversationHistoryStore(resp_capacity=1)

    await store.store_messages("resp_prev", _turns(2), prev_resp_id=None)

    def fail_if_unmapped_response_id_is_used(_):
        raise AssertionError("conversation trim used an unmapped response id")

    monkeypatch.setattr(store, "_pop_conversation", fail_if_unmapped_response_id_is_used)

    await store.store_response(
        SimpleNamespace(id="resp_next"),
        [
            {
                "role": "assistant",
                "content": "next",
            }
        ],
        prev_resp_id="resp_prev",
    )

    conversation = await store.get_conversation_history("resp_next")

    assert len(conversation) <= store.conversation_capacity
    assert (
        store.response_to_conversation["resp_next"] == store.response_to_conversation["resp_prev"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_harmony", [False, True], ids=["chat", "harmony"])
@pytest.mark.parametrize("prefix_roles", [(), ("system",), ("system", "developer")])
async def test_store_messages_trims_unfinished_user_history(
    progress_checked_store: ConversationHistoryStore,
    use_harmony: bool,
    prefix_roles: tuple[str, ...],
) -> None:
    store = progress_checked_store
    prefix = [_message(role, role, use_harmony) for role in prefix_roles]
    inputs = [_message("user", str(i), use_harmony) for i in range(store.conversation_capacity + 1)]

    await store.store_messages("resp_1", prefix + inputs, prev_resp_id=None)

    history = await store.get_conversation_history("resp_1")
    assert history == prefix + inputs[-(store.conversation_capacity - len(prefix)) :]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_harmony", [False, True], ids=["chat", "harmony"])
async def test_store_messages_trims_complete_turns_with_instructions(
    progress_checked_store: ConversationHistoryStore, use_harmony: bool
) -> None:
    store = progress_checked_store
    prefix = [_message(role, role, use_harmony) for role in ("system", "developer")]
    turns = [
        _message(role, str(i), use_harmony, channel="final" if role == "assistant" else None)
        for i in range(40)
        for role in ("user", "assistant")
    ]

    await store.store_messages("resp_1", prefix + turns, prev_resp_id=None)

    history = await store.get_conversation_history("resp_1")
    assert history == prefix + turns[-(store.conversation_capacity - len(prefix)) :]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_harmony", [False, True], ids=["chat", "harmony"])
@pytest.mark.parametrize("message_count", [0, 63, 64])
async def test_store_messages_keeps_history_within_capacity(
    progress_checked_store: ConversationHistoryStore, use_harmony: bool, message_count: int
) -> None:
    store = progress_checked_store
    messages = [_message("user", str(i), use_harmony) for i in range(message_count)]

    await store.store_messages("resp_1", list(messages), prev_resp_id=None)

    assert await store.get_conversation_history("resp_1") == messages


@pytest.mark.asyncio
@pytest.mark.parametrize("use_harmony", [False, True], ids=["chat", "harmony"])
async def test_store_messages_bounds_instruction_only_history(
    progress_checked_store: ConversationHistoryStore, use_harmony: bool
) -> None:
    store = progress_checked_store
    messages = [
        _message("system", str(i), use_harmony) for i in range(store.conversation_capacity + 1)
    ]

    await store.store_messages("resp_1", list(messages), prev_resp_id=None)

    assert (
        await store.get_conversation_history("resp_1") == messages[-store.conversation_capacity :]
    )


@pytest.mark.asyncio
async def test_store_response_trims_harmony_history_without_final_channel(
    progress_checked_store: ConversationHistoryStore,
) -> None:
    store = progress_checked_store
    prefix = [_message(role, role, True) for role in ("system", "developer")]
    inputs = [_message("user", "question", True)]
    await store.store_messages("resp_1", prefix + inputs, prev_resp_id=None)
    outputs = [
        _message("assistant", str(i), True, channel="analysis")
        for i in range(store.conversation_capacity + 1)
    ]

    await store.store_response(SimpleNamespace(id="resp_1"), outputs, prev_resp_id=None)

    history = await store.get_conversation_history("resp_1")
    assert history == prefix + outputs[-(store.conversation_capacity - len(prefix)) :]
