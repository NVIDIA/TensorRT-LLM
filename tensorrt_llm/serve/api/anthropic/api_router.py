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
"""The Anthropic Messages router of the aggregated server.

``/v1/messages``, ``/v1/messages/count_tokens`` and the six Message Batches
routes. The disaggregated frontend keeps its own ``/v1/messages`` route and
calls the same serving class.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Query, Request, Response

from tensorrt_llm.serve._base.request_context import RequestContext
from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicCreateBatchRequest,
    AnthropicMessagesRequest,
)
from tensorrt_llm.serve.api.anthropic.serving import (
    AnthropicBatchesServing,
    AnthropicCountTokensServing,
    AnthropicMessagesServing,
)

# The serving objects the composition root sets on app.state before attaching.
APP_STATE_KEYS = (
    "anthropic_serving_messages",
    "anthropic_serving_count_tokens",
    "anthropic_serving_batches",
)

router = make_router()


# No app.state defaults: wiring bugs fail loudly.
def _messages(raw_request: Request) -> AnthropicMessagesServing:
    serving: AnthropicMessagesServing = raw_request.app.state.anthropic_serving_messages
    return serving


def _count_tokens(raw_request: Request) -> AnthropicCountTokensServing:
    serving: AnthropicCountTokensServing = raw_request.app.state.anthropic_serving_count_tokens
    return serving


def _batches(raw_request: Request) -> AnthropicBatchesServing:
    serving: AnthropicBatchesServing = raw_request.app.state.anthropic_serving_batches
    return serving


async def anthropic_messages(request: AnthropicMessagesRequest, raw_request: Request) -> Response:
    """Serve the Anthropic Messages API on top of the chat pipeline."""
    return await _messages(raw_request).create_message(
        request, RequestContext.from_http(raw_request)
    )


async def anthropic_count_tokens(
    request: AnthropicCountTokensRequest, raw_request: Request
) -> Response:
    """Report how many input tokens a Messages request would consume."""
    return await _count_tokens(raw_request).anthropic_count_tokens(request)


async def anthropic_create_batch(
    request: AnthropicCreateBatchRequest, raw_request: Request
) -> Response:
    return await _batches(raw_request).anthropic_create_batch(request)


async def anthropic_list_batches(
    raw_request: Request,
    limit: int = Query(20, ge=1, le=100),
    after_id: Optional[str] = None,
    before_id: Optional[str] = None,
) -> Response:
    return await _batches(raw_request).anthropic_list_batches(
        limit=limit, after_id=after_id, before_id=before_id
    )


async def anthropic_get_batch(batch_id: str, raw_request: Request) -> Response:
    return await _batches(raw_request).anthropic_get_batch(batch_id)


async def anthropic_delete_batch(batch_id: str, raw_request: Request) -> Response:
    return await _batches(raw_request).anthropic_delete_batch(batch_id)


async def anthropic_cancel_batch(batch_id: str, raw_request: Request) -> Response:
    return await _batches(raw_request).anthropic_cancel_batch(batch_id)


async def anthropic_batch_results(batch_id: str, raw_request: Request) -> Response:
    return await _batches(raw_request).anthropic_batch_results(batch_id)


# Registration order is match order. The literal batch paths come before the
# parameterized ones, as they always did.
router.add_api_route("/v1/messages", anthropic_messages, methods=["POST"])
# Claude Code calls count_tokens before most turns to decide whether to
# compact its context; an unregistered route 404s every call.
router.add_api_route("/v1/messages/count_tokens", anthropic_count_tokens, methods=["POST"])
router.add_api_route("/v1/messages/batches", anthropic_create_batch, methods=["POST"])
router.add_api_route("/v1/messages/batches", anthropic_list_batches, methods=["GET"])
router.add_api_route("/v1/messages/batches/{batch_id}", anthropic_get_batch, methods=["GET"])
router.add_api_route("/v1/messages/batches/{batch_id}", anthropic_delete_batch, methods=["DELETE"])
router.add_api_route(
    "/v1/messages/batches/{batch_id}/cancel", anthropic_cancel_batch, methods=["POST"]
)
router.add_api_route(
    "/v1/messages/batches/{batch_id}/results", anthropic_batch_results, methods=["GET"]
)


def attach_router(app: FastAPI) -> None:
    for key in APP_STATE_KEYS:
        if not hasattr(app.state, key):
            raise RuntimeError(
                f"app.state.{key} must be set before attaching "
                "tensorrt_llm.serve.api.anthropic.api_router; the composition root sets it."
            )
    app.include_router(router)
