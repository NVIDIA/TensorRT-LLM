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

# Disagg edge-to-worker routing key, independent of conversation history.
SUBAGENT_AFFINITY_HEADER = "x-trtllm-subagent-affinity-id"

# Supported HTTP header protocol for external clients, gateways, or proxies
# that carry a stable multi-turn identifier outside the JSON body. Body
# ``conversation_params.conversation_id`` is canonical when both body and
# headers are set; the serve edge copies the first non-empty header value into
# ``request.conversation_params`` only when the body omits it. Routers then read
# ``conversation_params.conversation_id`` to keep later turns of the same
# conversation on the same backend when sticky conversation routing is enabled.
#
# The Claude Code headers are listed first because a client that sends one also
# sends nothing else on this list; ordering them ahead of the generic names
# keeps the common case a single lookup. They mirror the set the Anthropic
# adapter already reads for audit records, so the Messages API gets the same
# conversation identity the audit log records rather than a second notion of a
# session.
CONVERSATION_ID_HEADERS = (
    "x-claude-code-session-id",
    "x-claude-session-id",
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


def get_request_subagent_affinity_id(
    request: RequestWithConversationParams,
) -> Optional[str]:
    """Return the parent-session routing key set by the disagg edge."""
    conversation_params = request.conversation_params
    if conversation_params is None:
        return None
    return getattr(conversation_params, "subagent_affinity_id", None)


def get_request_routing_id(request: RequestWithConversationParams) -> Optional[str]:
    """Return the parent affinity key when present, else the conversation id."""
    return get_request_subagent_affinity_id(request) or get_request_conversation_id(request)


def extract_subagent_parent_id(
    headers: Optional[Mapping[str, str]],
    subagent_affinity_header: Optional[str],
) -> Optional[str]:
    """Read the parent-session id from the configured gateway header."""
    if not subagent_affinity_header or headers is None:
        return None
    lower_headers = {str(key).lower(): value for key, value in headers.items()}
    parent_id = lower_headers.get(str(subagent_affinity_header).strip().lower())
    if parent_id is None:
        return None
    parent_id = str(parent_id).strip()
    return parent_id or None


def extract_subagent_affinity_id_from_headers(
    headers: Optional[Mapping[str, str]],
) -> Optional[str]:
    """Read the parent affinity key forwarded by the disagg edge."""
    if headers is None:
        return None
    lower_headers = {str(key).lower(): value for key, value in headers.items()}
    affinity_id = lower_headers.get(SUBAGENT_AFFINITY_HEADER)
    if affinity_id is None:
        return None
    affinity_id = str(affinity_id).strip()
    return affinity_id or None


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

    Body ``conversation_params.conversation_id`` takes precedence over headers.
    """
    conversation_params = request.conversation_params
    if conversation_params is not None:
        return conversation_params.conversation_id

    conversation_id = extract_conversation_id_from_headers(headers)
    if conversation_id is not None:
        from tensorrt_llm.serve.openai_protocol import ConversationParams

        request.conversation_params = ConversationParams(conversation_id=conversation_id)
    return conversation_id
