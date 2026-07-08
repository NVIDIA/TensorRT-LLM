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

from typing import Any, Mapping, Optional, Protocol

# Supported HTTP header protocol for external clients, gateways, or proxies
# that carry a stable multi-turn identifier outside the JSON body. Body
# ``conversation_params.conversation_id`` is canonical when both body and
# headers are set; the serve edge copies the first non-empty header value into
# ``request.conversation_params`` only when the body omits it. Routers then read
# ``conversation_params.conversation_id`` to keep later turns of the same
# conversation on the same backend when sticky conversation routing is enabled.
CONVERSATION_ID_HEADERS = (
    "x-session-id",
    "x-correlation-id",
    "x-session-affinity",
    "x-multi-turn-session-id",
)


class RequestWithConversationParams(Protocol):
    conversation_params: Any


def get_request_conversation_id(request: RequestWithConversationParams) -> Optional[str]:
    conversation_params = request.conversation_params
    if conversation_params is None:
        return None
    return conversation_params.conversation_id


def extract_conversation_id_from_headers(headers: Optional[Mapping[str, str]]) -> Optional[str]:
    if headers is None:
        return None
    lower_headers = {str(key).lower(): value for key, value in headers.items()}
    for header_name in CONVERSATION_ID_HEADERS:
        conversation_id = lower_headers.get(header_name)
        if conversation_id is None:
            continue
        conversation_id = str(conversation_id).strip()
        if conversation_id:
            return conversation_id
    return None


def resolve_request_conversation_id(
    request: RequestWithConversationParams,
    headers: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return conversation_params.conversation_id populated at the serve edge.

    Body ``conversation_params.conversation_id`` is canonical. Headers are used
    only when the body does not provide an id.
    """
    conversation_params = request.conversation_params
    if conversation_params is not None:
        return conversation_params.conversation_id

    conversation_id = extract_conversation_id_from_headers(headers)
    if conversation_id is not None:
        from tensorrt_llm.serve.openai_protocol import ConversationParams

        request.conversation_params = ConversationParams(conversation_id=conversation_id)
    return conversation_id
