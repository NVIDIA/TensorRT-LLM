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
"""HTTP routes of the render endpoints, mountable on any FastAPI app."""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .prompt_types import GenerateRequest
from .serving import RenderRequestError, ServingRender

CHAT_RENDER_PATH = "/v1/chat/completions/render"
COMPLETION_RENDER_PATH = "/v1/completions/render"

# The render and generate routes are a separate exposed surface; a normal
# ``trtllm-serve`` worker mounts them only when this is set.
ENABLE_ENV = "TRTLLM_ENABLE_RENDER_ENDPOINTS"


def render_endpoints_enabled() -> bool:
    """Whether a serving worker mounts the render and generate routes."""
    return os.getenv(ENABLE_ENV, "0") == "1"


def error_response(error: RenderRequestError) -> JSONResponse:
    """The OpenAI-shaped error body for a rejected request."""
    # Deferred: openai_protocol pulls in the whole request-model stack.
    from tensorrt_llm.serve.openai_protocol import ErrorResponse

    body = ErrorResponse(
        message=error.message, type=error.err_type, code=error.status_code
    ).model_dump()
    return JSONResponse(content=body, status_code=error.status_code)


async def _json_body(raw_request: Request) -> dict:
    try:
        body = await raw_request.json()
    except ValueError as error:
        raise RenderRequestError(400, f"Request body is not valid JSON: {error}") from error
    if not isinstance(body, dict):
        raise RenderRequestError(400, "Request body must be a JSON object.")
    return body


def attach_render_routes(app: FastAPI, serving: ServingRender) -> None:
    """Mount the two render routes on ``app``.

    Registered with ``add_api_route`` so the app's own route class applies, as it
    does for the inference routes.
    """

    async def render_chat_completions(raw_request: Request):
        try:
            return await serving.render_chat(await _json_body(raw_request))
        except RenderRequestError as error:
            return error_response(error)

    async def render_completions(raw_request: Request):
        try:
            return await serving.render_completion(await _json_body(raw_request))
        except RenderRequestError as error:
            return error_response(error)

    app.add_api_route(
        CHAT_RENDER_PATH,
        render_chat_completions,
        methods=["POST"],
        response_model=GenerateRequest,
    )
    app.add_api_route(
        COMPLETION_RENDER_PATH,
        render_completions,
        methods=["POST"],
        response_model=List[GenerateRequest],
    )
