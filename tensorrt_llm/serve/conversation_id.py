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
#
# This module implements only the base "body > header" resolution (precedence
# steps 3 and 4 below). The disagg sub-agent parent-session override
# (``conversation_affinity_header_for_subagents``) is layered on TOP of this at
# the disagg serve edge -- see ``OpenAIDisaggServer._extract_conversation_id``
# for the full precedence:
#   1. [future TODO] a sub-agent parent-affinity id from the request BODY.
#   2. the configured sub-agent parent-session HEADER.
#   3. ``conversation_params.conversation_id`` from the request BODY.
#   4. a conversation-id HEADER (below).
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
    return None if conversation_params is None else conversation_params.conversation_id


def extract_subagent_parent_id(
    headers: Optional[Mapping[str, str]],
    subagent_affinity_header: Optional[str],
) -> Optional[str]:
    """Return the sub-agent parent-session header value, stripped, or None.

    When a deployment configures ``conversation_affinity_header_for_subagents``
    (e.g. the Dynamo header ``X-Dynamo-Parent-Session-ID`` that an agent gateway
    attaches to every sub-agent request but NOT to a main-agent request), this
    returns that header's value -- the id a sub-agent should co-locate on. A
    main-agent request lacks the header and yields None.
    """
    if not subagent_affinity_header or headers is None:
        return None
    lower_headers = {str(key).lower(): value for key, value in headers.items()}
    parent_id = lower_headers.get(str(subagent_affinity_header).strip().lower())
    if parent_id is None:
        return None
    parent_id = str(parent_id).strip()
    return parent_id or None


def extract_conversation_id_from_headers(
    headers: Optional[Mapping[str, str]],
) -> Optional[str]:
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

    Base "body > header" resolution: body
    ``conversation_params.conversation_id`` takes precedence over the
    conversation-id headers, which are consulted only when the body omits it.
    The disagg sub-agent parent-session override is applied on top of this by
    ``OpenAIDisaggServer._extract_conversation_id``; see the module docstring.
    """
    conversation_params = request.conversation_params
    if conversation_params is not None:
        return conversation_params.conversation_id

    conversation_id = extract_conversation_id_from_headers(headers)
    if conversation_id is not None:
        from tensorrt_llm.serve.openai_protocol import ConversationParams

        request.conversation_params = ConversationParams(conversation_id=conversation_id)
    return conversation_id
