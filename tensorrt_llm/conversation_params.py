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


@dataclass(slots=True, kw_only=True)
class ConversationParams:
    """Conversation parameters.

    Args:
        conversation_id (str): Stable multi-turn conversation id used for routing.
    """

    conversation_id: str

    def __post_init__(self) -> None:
        if self.conversation_id is None:
            raise ValueError("conversation_id must be non-empty")
        conversation_id = str(self.conversation_id).strip()
        if not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        self.conversation_id = conversation_id
