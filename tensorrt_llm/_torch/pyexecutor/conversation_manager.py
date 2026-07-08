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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from ...logger import logger
from ...runtime.kv_cache_manager_v2 import CommittedBlockRecord
from .llm_request import LlmRequest

if TYPE_CHECKING:
    from ...runtime.kv_cache_manager_v2 import _KVCache


def _request_conversation_id(request: LlmRequest) -> Optional[str]:
    if request.is_dummy_request:
        return None
    conversation_params = request.py_conversation_params
    if conversation_params is None:
        return None
    conversation_id = conversation_params.conversation_id.strip()
    return conversation_id or None


@dataclass(slots=True)
class ConversationState:
    current_request_id: Optional[int] = None
    committed_block_record: Optional[CommittedBlockRecord] = None


class ConversationManager:
    """Track request ordering and the live request for each conversation."""

    def __init__(self) -> None:
        self._conversation_states: Dict[str, ConversationState] = {}

    def record_conversation(self, request: LlmRequest, kv_cache: "_KVCache") -> None:
        """Record a completed request, replacing the previous turn on success."""
        request_id = request.py_request_id
        conversation_id = _request_conversation_id(request)
        if conversation_id is None:
            return

        state = self._conversation_states[conversation_id]
        if state.current_request_id != request_id:
            return

        block_record = kv_cache.record_committed_blocks()
        if block_record is None:
            logger.warning(
                f"Committed blocks for request {request_id} in conversation "
                f"{conversation_id} have been dropped."
            )
        else:
            previous_record = state.committed_block_record
            state.committed_block_record = block_record
            if previous_record is not None:
                previous_record.release()

        state.current_request_id = None

    def update_conversation(self, request: LlmRequest) -> None:
        """Register a request as the active turn of its conversation."""
        conversation_id = _request_conversation_id(request)
        if conversation_id is None:
            return
        request_id = request.py_request_id
        state = self._conversation_states.setdefault(conversation_id, ConversationState())
        current_request_id = state.current_request_id
        if current_request_id is not None and current_request_id != request_id:
            logger.warning(
                f"Conversation {conversation_id} already has current request "
                f"{current_request_id}. Request {request_id} will ignore "
                "conversation params."
            )
            return

        state.current_request_id = request_id

    def clear(self) -> None:
        """Clear state after reusable KV-cache blocks have been cleared."""
        self._conversation_states.clear()
