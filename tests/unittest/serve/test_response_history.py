# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from openai_harmony import Message

from tensorrt_llm.serve.responses_utils import ConversationHistoryStore

pytestmark = pytest.mark.cpu_only


def _messages(use_harmony, user, answer=None):
    entries = [("user", user)]
    if answer is not None:
        entries.append(("assistant", answer))
    if use_harmony:
        return [Message.from_role_and_content(role, text) for role, text in entries]
    return [{"role": role, "content": text} for role, text in entries]


@pytest.mark.parametrize("use_harmony", [False, True], ids=["chat", "harmony"])
@pytest.mark.parametrize("overlap", [False, True], ids=["sequential", "overlapping"])
def test_response_history_keeps_parent_and_sibling_turns_separate(use_harmony, overlap):
    async def scenario():
        store = ConversationHistoryStore()
        parent = _messages(use_harmony, "parent", "parent answer")
        await store.store_messages("parent", parent[:1], None)
        await store.store_response(SimpleNamespace(id="parent"), parent[1:])

        left = parent + _messages(use_harmony, "left")
        right = parent + _messages(use_harmony, "right")
        await store.store_messages("left", left, "parent")
        if overlap:
            await store.store_messages("right", right, "parent")
        left_answer = _messages(use_harmony, "unused", "left answer")[1:]
        await store.store_response(SimpleNamespace(id="left"), left_answer, "parent")
        assert await store.get_conversation_history("parent") == parent
        assert await store.get_conversation_history("left") == left + left_answer

        if not overlap:
            # This is the input the next request renders from its previous_response_id.
            sibling_input = await store.get_conversation_history("parent")
            await store.store_messages(
                "right", sibling_input + _messages(use_harmony, "right"), "parent"
            )
        right_answer = _messages(use_harmony, "unused", "right answer")[1:]
        await store.store_response(SimpleNamespace(id="right"), right_answer, "parent")
        assert await store.get_conversation_history("parent") == parent
        assert await store.get_conversation_history("left") == left + left_answer
        assert await store.get_conversation_history("right") == right + right_answer

    asyncio.run(scenario())


@pytest.mark.parametrize("use_harmony", [False, True], ids=["chat", "harmony"])
def test_response_history_without_pending_input_copies_parent(use_harmony):
    async def scenario():
        store = ConversationHistoryStore()
        parent = _messages(use_harmony, "parent", "parent answer")
        await store.store_messages("parent", parent, None)
        answer = _messages(use_harmony, "unused", "child answer")[1:]
        await store.store_response(SimpleNamespace(id="child"), answer, "parent")
        assert await store.get_conversation_history("parent") == parent
        assert await store.get_conversation_history("child") == parent + answer

    asyncio.run(scenario())


def test_response_history_stays_within_existing_lru_limits():
    async def scenario():
        store = ConversationHistoryStore(resp_capacity=2, max_conversations=2)
        for index in range(3):
            response_id = str(index)
            messages = _messages(False, response_id, f"answer {index}")
            await store.store_messages(response_id, messages[:1], None)
            await store.store_response(SimpleNamespace(id=response_id), messages[1:])
        assert await store.load_response("0") is None
        assert await store.get_conversation_history("0") == []
        assert await store.get_conversation_history("1") == _messages(False, "1", "answer 1")
        assert await store.get_conversation_history("2") == _messages(False, "2", "answer 2")
        assert len(store.responses) == len(store.conversations) == 2
        assert len(store.response_to_conversation) == len(store.conversation_to_response) == 2

    asyncio.run(scenario())
