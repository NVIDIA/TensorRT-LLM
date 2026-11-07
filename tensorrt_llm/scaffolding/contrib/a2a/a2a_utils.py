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
"""Thin client wrapper around the official ``a2a-sdk`` (Agent2Agent protocol).

The wrapper keeps the rest of Scaffolding (worker/controller) free of any
hard dependency on ``a2a-sdk``: the SDK is imported lazily inside the methods
that actually talk to a remote agent, and the worker only ever sees the
normalized :class:`AgentInfo` view defined below.  This mirrors how the MCP
contrib isolates the ``mcp`` package inside ``mcp_utils.MCPClient``.
"""

import uuid
from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentInfo:
    """Protocol-agnostic view of a remote agent, derived from its agent card.

    Decoupling the worker/controller from ``a2a-sdk`` types means tests can
    inject fake connections without installing the SDK.
    """

    name: str
    description: str = ""
    skills: List[str] = field(default_factory=list)


def _extract_text_from_response(response) -> str:
    """Best-effort extraction of textual content from an A2A send-message response.

    The ``a2a-sdk`` response schema has shifted across versions, so we walk the
    structure defensively and fall back to ``str(response)`` if no text part is
    found rather than raising on an unexpected layout.
    """
    texts: List[str] = []

    def _walk(obj):
        # A ``TextPart`` exposes a ``text`` attribute directly.
        text = getattr(obj, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
        # ``Part`` wraps the concrete part under ``root`` in recent SDKs.
        root = getattr(obj, "root", None)
        if root is not None and root is not obj:
            _walk(root)
        # A ``Message``/``Task`` carries a list of ``parts``.
        parts = getattr(obj, "parts", None)
        if parts:
            for part in parts:
                _walk(part)

    result = getattr(response, "root", response)
    result = getattr(result, "result", result)
    _walk(result)

    return "\n".join(texts) if texts else str(response)


class A2AAgentConnection:
    """A connection to a single remote agent that speaks the A2A protocol."""

    def __init__(self):
        self._httpx_client = None
        self._client = None
        self._agent_card = None

    async def connect(self, base_url: str):
        """Resolve the remote agent card and create an A2A client for it."""
        try:
            import httpx
            from a2a.client import A2ACardResolver, A2AClient
        except ImportError as e:
            raise ImportError(
                "The A2A contrib requires the 'a2a-sdk' and 'httpx' packages. "
                "Install them with `pip install a2a-sdk httpx`."
            ) from e

        self._httpx_client = httpx.AsyncClient()
        resolver = A2ACardResolver(httpx_client=self._httpx_client, base_url=base_url.rstrip("/"))
        self._agent_card = await resolver.get_agent_card()
        self._client = A2AClient(httpx_client=self._httpx_client, agent_card=self._agent_card)

    def get_agent_info(self) -> AgentInfo:
        """Normalize the resolved agent card into an :class:`AgentInfo`."""
        card = self._agent_card
        skills = []
        for skill in getattr(card, "skills", None) or []:
            name = getattr(skill, "name", None) or getattr(skill, "id", "")
            if name:
                skills.append(name)
        return AgentInfo(
            name=card.name, description=getattr(card, "description", "") or "", skills=skills
        )

    async def send_message(self, message: str) -> str:
        """Send a text message to the remote agent and return its text reply."""
        from a2a.types import MessageSendParams, SendMessageRequest

        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "messageId": uuid.uuid4().hex,
                }
            ),
        )
        response = await self._client.send_message(request)
        return _extract_text_from_response(response)

    async def cleanup(self):
        """Close the underlying HTTP client."""
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
