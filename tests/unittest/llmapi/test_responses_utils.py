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

from tensorrt_llm.serve.responses_utils import ConversationHistoryStore


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
