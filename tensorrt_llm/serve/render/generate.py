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
"""``POST /generate``: execute a prepared request without rendering it again.

The body is a :class:`GenerateRequest` produced by a render endpoint. The worker
checks that it understands the schema version and that the renderer used the same
rendering configuration, then runs the original request through its normal route
with the prompt token ids already filled in, so rendering and tokenization are
skipped. Scheduling, streaming, tool and reasoning parsing, usage, cancellation
and errors are those of the route the request belongs to.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request

from .api_router import _json_body, attach_render_routes, error_response, render_endpoints_enabled
from .fingerprint import fingerprints_match
from .prompt_types import GENERATE_REQUEST_SCHEMA_VERSION, GenerateRequest
from .resources import RenderResources
from .serving import RenderRequestError, ServingRender

GENERATE_PATH = "/generate"


def _server_fingerprint(server: Any) -> Dict[str, Any]:
    fingerprint = getattr(server, "_render_fingerprint", None)
    if fingerprint is None:
        fingerprint = RenderResources.from_server(server).fingerprint()
        server._render_fingerprint = fingerprint
    return fingerprint


def _validate(body: GenerateRequest, server: Any) -> None:
    if body.schema_version != GENERATE_REQUEST_SCHEMA_VERSION:
        raise RenderRequestError(
            400,
            f"Unsupported schema_version {body.schema_version}; this server understands "
            f"{GENERATE_REQUEST_SCHEMA_VERSION}.",
        )
    if not body.tokens_trusted:
        raise RenderRequestError(
            400,
            "The prepared token ids are not trusted "
            f"({body.untrusted_reason or 'unspecified'}); send the original request "
            "to the inference route instead.",
        )
    if not fingerprints_match(_server_fingerprint(server), body.fingerprint):
        raise RenderRequestError(
            409,
            "The renderer's configuration (chat template, tokenizer, model type or "
            "extension) differs from this server's, so its token ids cannot be executed here.",
            err_type="ConflictError",
        )


def attach_generate_route(app: FastAPI, server: Any) -> None:
    """Mount ``POST /generate`` on a serving worker's app."""

    async def generate(raw_request: Request):
        from pydantic import ValidationError

        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest

        try:
            try:
                body = GenerateRequest.model_validate(await _json_body(raw_request))
            except ValidationError as error:
                raise RenderRequestError(400, str(error)) from error
            _validate(body, server)
            try:
                if body.kind == "chat":
                    request = ChatCompletionRequest.model_validate(
                        {**body.request, "prompt_token_ids": body.token_ids}
                    )
                else:
                    request = CompletionRequest.model_validate(
                        {**body.request, "prompt": body.token_ids}
                    )
            except ValidationError as error:
                raise RenderRequestError(400, str(error)) from error
        except RenderRequestError as error:
            return error_response(error)
        # The routes fall back to the raw JSON body's messages when the request model
        # rejects them (object-valued tool-call arguments); here the HTTP body is the
        # prepared-request envelope, so hand the route the original request body instead.
        # (Starlette caches the parsed body in ``_json``; ``Request.json()`` returns it.)
        raw_request._json = copy.deepcopy(body.request)
        if body.kind == "chat":
            # Decisions derived from the rendered prompt, which the route cannot make
            # again from token ids. A private attribute, so a client cannot set it.
            request._render_context = body.context.model_dump()
            # The route a worker serves chat on: a Harmony (gpt-oss) worker's chat route
            # parses channels and tool calls out of the token stream.
            chat_route = (
                server.chat_harmony if getattr(server, "use_harmony", False) else server.openai_chat
            )
            return await chat_route(request, raw_request)
        return await server.openai_completion(request, raw_request)

    app.add_api_route(GENERATE_PATH, generate, methods=["POST"])


def mount_render_endpoints(app: FastAPI, server: Any, *, enabled: Optional[bool] = None) -> bool:
    """Mount the render routes and ``/generate`` on a serving worker's app.

    They are a separate exposed surface, so a worker mounts them only when
    ``TRTLLM_ENABLE_RENDER_ENDPOINTS=1`` (or ``enabled`` says so). Returns
    whether they were mounted.
    """
    if not (render_endpoints_enabled() if enabled is None else enabled):
        return False
    attach_render_routes(app, ServingRender(lambda: RenderResources.from_server(server)))
    attach_generate_route(app, server)
    return True
