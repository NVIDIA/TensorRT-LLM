# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker endpoints of an LLM server.

``/health_generate``, ``/v1/data_transceiver_state``, ``/energy_metrics``,
``/steady_clock_offset``, ``/kv_cache_events`` and ``/server_info``.
"""

from typing import Any, Optional, Protocol

from fastapi import Request
from fastapi.responses import Response

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.multi_frontend import SharedClockOffset
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.render import RenderResources


class _ChatBackend(Protocol):
    """What ``health_generate`` calls on the chat serving object.

    A Protocol rather than an import: packages under ``serve/api/`` do not
    import each other, so the composition root injects the chat serving
    object.
    """

    async def openai_chat(self, request: ChatCompletionRequest,
                          raw_request: Request) -> Response:
        ...


class WorkerServing(BaseServing):
    """The worker endpoints, served from injected dependencies."""

    def __init__(self, *, model: str, generator: Any,
                 adjusted_steady_clock: AdjustedSteadyClock,
                 shared_clock_offset: Optional[SharedClockOffset],
                 chat_serving: _ChatBackend,
                 render_resources: RenderResources) -> None:
        self.model = model
        self.generator = generator
        self._adjusted_steady_clock = adjusted_steady_clock
        self._shared_clock_offset = shared_clock_offset
        self._chat_serving = chat_serving
        self._render_resources = render_resources
        # The composition root's lifespan hands the energy monitor over once
        # it exists.
        self.energy_monitor = None
        # get_server_info computes the rendering fingerprint once, then
        # caches it here.
        self._render_fingerprint = None
