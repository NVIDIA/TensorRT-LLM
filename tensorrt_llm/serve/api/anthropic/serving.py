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
"""Serving classes of the Anthropic Messages dialect.

Messages, count_tokens and the Message Batches, served from values the
composition root injects; no class here reads a server attribute.
"""

import asyncio
import json
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base.chat_backend import ChatBackend
from tensorrt_llm.serve._base.errors import EngineFatalError, ServingError
from tensorrt_llm.serve._base.request_context import RequestContext
from tensorrt_llm.serve.api.anthropic.adapter import (
    AnthropicRequestError, AnthropicResponseError, anthropic_error_response,
    convert_anthropic_count_tokens_request, convert_anthropic_request,
    convert_chat_response, reframe_openai_stream)
from tensorrt_llm.serve.api.anthropic.batches import (AnthropicBatchStore,
                                                      BatchItemRunner,
                                                      BatchStoreFullError,
                                                      results_to_jsonl)
from tensorrt_llm.serve.api.anthropic.protocol import (
    AnthropicBatchDeleteResponse, AnthropicBatchList,
    AnthropicCountTokensRequest, AnthropicCountTokensResponse,
    AnthropicCreateBatchRequest, AnthropicMessagesRequest)
from tensorrt_llm.serve.openai_protocol import ChatCompletionResponse
from tensorrt_llm.serve.render import (HarmonyConversionError, RenderResources,
                                       UnsupportedRenderError, render_chat,
                                       render_chat_for_length)


def _error_type(status: int) -> str:
    """The Anthropic error type both servers derive from an HTTP status."""
    return "invalid_request_error" if 400 <= status < 500 else "api_error"


class AnthropicMessagesServing:
    """The Anthropic Messages orchestration, for any ChatBackend.

    One class serves the aggregated server (over its typed chat core), the
    disaggregated frontend (over its prepare, run and finish steps) and the
    aggregated Message Batches. What differs between the servers, the model a
    stream without chunks announces and how an error's message is extracted,
    comes from the backend.
    """

    def __init__(self, backend: ChatBackend) -> None:
        self._backend = backend

    async def create_message(self, request: AnthropicMessagesRequest,
                             ctx: RequestContext) -> Response:
        """Serve the Anthropic Messages API on top of the chat pipeline.

        The Anthropic request is translated into a ChatCompletionRequest,
        handed to the chat backend unchanged (chat template, tool parser and
        post-processing are reused verbatim), and the typed result is
        translated back: JSON body for non-streaming, SSE reframing for
        streaming.
        """
        try:
            chat_request = convert_anthropic_request(request)
        except (AnthropicRequestError, ValidationError) as e:
            return anthropic_error_response(str(e), "invalid_request_error",
                                            400)
        try:
            result = await self._backend.create_chat_completion(
                chat_request, ctx)
        except EngineFatalError:
            return anthropic_error_response("Internal server error",
                                            "api_error", 500)
        except ServingError as e:
            return anthropic_error_response(self._backend.error_message(e),
                                            _error_type(e.status), e.status)
        if not isinstance(result, ChatCompletionResponse):
            return StreamingResponse(
                content=reframe_openai_stream(
                    result, model=self._backend.stream_model(chat_request)),
                media_type="text/event-stream",
            )
        try:
            anthropic_response = convert_chat_response(result)
        except (AnthropicResponseError, ValidationError, json.JSONDecodeError):
            logger.error("Invalid response from OpenAI chat pipeline:\n"
                         f"{traceback.format_exc()}")
            return anthropic_error_response("Internal server error",
                                            "api_error", 500)
        return JSONResponse(content=anthropic_response.model_dump(
            exclude_none=True))

    async def run_batched_message(
            self,
            request: AnthropicMessagesRequest) -> Tuple[str, Dict[str, Any]]:
        """Run one batched Messages request through the chat backend."""
        # Streaming makes no sense for a batch: the result is collected, not
        # watched. Forcing it off here means a client that sets stream=true in
        # params gets a usable message instead of an SSE body in its .jsonl.
        request = request.model_copy(update={"stream": False})
        try:
            chat_request = convert_anthropic_request(request)
        except (AnthropicRequestError, ValidationError) as e:
            return "errored", {
                "type": "invalid_request_error",
                "message": str(e)
            }
        # A batched request has no client connection behind it: the context
        # carries the seeded state and the converted request as its raw JSON.
        ctx = RequestContext.internal(
            chat_request.model_dump(exclude_none=True))
        try:
            result = await self._backend.create_chat_completion(
                chat_request, ctx)
        except EngineFatalError:
            return "errored", {
                "type": "api_error",
                "message": "Internal server error"
            }
        except ServingError as e:
            return "errored", {
                "type": _error_type(e.status),
                "message": self._backend.error_message(e)
            }
        try:
            anthropic_response = convert_chat_response(result)
        except (AnthropicResponseError, ValidationError, json.JSONDecodeError):
            logger.error(
                "Invalid response from OpenAI chat pipeline in batch:\n"
                f"{traceback.format_exc()}")
            return "errored", {
                "type": "api_error",
                "message": "Internal server error"
            }
        return "succeeded", anthropic_response.model_dump(exclude_none=True)


class AnthropicCountTokensServing:
    """Counts the input tokens of a Messages request over the rendered prompt."""

    def __init__(self, *, render_resources: Callable[[],
                                                     RenderResources]) -> None:
        # Called per request: the resources are read at call time, as the
        # chat route reads them.
        self._render_resources = render_resources

    async def anthropic_count_tokens(
            self, request: AnthropicCountTokensRequest) -> Response:
        """Report how many input tokens a Messages request would consume.

        The count is taken over the rendered prompt rather than over the raw
        messages, so system blocks, tool definitions and the thinking prefix
        are all included - those are exactly what make a client's own estimate
        drift from the server's.
        """
        try:
            chat_request = convert_anthropic_count_tokens_request(request)
        except AnthropicRequestError as e:
            return anthropic_error_response(str(e), "invalid_request_error",
                                            400)
        except ValidationError as e:
            return anthropic_error_response(str(e), "invalid_request_error",
                                            400)

        try:

            # Both the render and the encode are synchronous and both scale
            # with prompt size. Claude Code calls this before most turns with
            # the whole conversation attached, so running them inline would
            # block the event loop -- and therefore every other in-flight
            # request on this server -- for a full template render plus a
            # tokenizer pass. The chat path avoids this the same way.
            def _count() -> int:
                # The count is over the prompt tokens the chat route would
                # execute: same pipeline, same template, same tokenization.
                resources = self._render_resources()
                try:
                    rendered = render_chat(chat_request,
                                           resources,
                                           tokenize=True)
                except (UnsupportedRenderError, HarmonyConversionError):
                    # Media, an own-processor model or an unconvertible Harmony
                    # request: count the text, tools and template without the media.
                    rendered = render_chat_for_length(chat_request, resources)
                return len(rendered.token_ids)

            input_tokens = await asyncio.to_thread(_count)
        except Exception:
            logger.error("count_tokens failed to render the prompt:\n"
                         f"{traceback.format_exc()}")
            return anthropic_error_response("Internal server error",
                                            "api_error", 500)

        return JSONResponse(content=AnthropicCountTokensResponse(
            input_tokens=input_tokens).model_dump())


class AnthropicBatchesServing:
    """The Message Batches routes of the aggregated server."""

    def __init__(self, *, runner: BatchItemRunner) -> None:
        # The batch methods create the store on first use, with this runner.
        self._run_batched_message = runner
        self._anthropic_batch_store: Optional[AnthropicBatchStore] = None

    def _batch_store(self) -> AnthropicBatchStore:
        """Created on first use so servers that never batch pay nothing."""
        if getattr(self, "_anthropic_batch_store", None) is None:
            self._anthropic_batch_store = AnthropicBatchStore(
                runner=self._run_batched_message)
        return self._anthropic_batch_store

    async def anthropic_create_batch(
            self, request: AnthropicCreateBatchRequest) -> Response:
        try:
            batch = self._batch_store().create(request.requests)
        except BatchStoreFullError as e:
            # 429, not 400: the request is valid and an identical one will
            # succeed later, so the client should back off rather than edit it.
            return anthropic_error_response(str(e), "rate_limit_error", 429)
        except ValueError as e:
            return anthropic_error_response(str(e), "invalid_request_error",
                                            400)
        return JSONResponse(content=batch.model_dump())

    async def anthropic_list_batches(
            self,
            # Validated here rather than clamped in the store: clamping turned
            # limit=0 into 1 instead of a 400, and bounded the page against the
            # retention limit, which is an unrelated quantity.
            limit: int = Query(20, ge=1, le=100),
            after_id: Optional[str] = None,
            before_id: Optional[str] = None) -> Response:
        try:
            batches, has_more = self._batch_store().list(limit=limit,
                                                         after_id=after_id,
                                                         before_id=before_id)
        except LookupError as e:
            # Saying so beats an empty page: an empty page is indistinguishable
            # from a finished walk, so the client stops early having silently
            # skipped every batch past the cursor.
            return anthropic_error_response(
                f"Unknown pagination cursor {e.args[0]!r}. The batch may have "
                "been deleted or expired; restart the listing without a cursor.",
                "invalid_request_error", 400)
        listing = AnthropicBatchList(
            data=batches,
            # Reported honestly rather than hardcoded false: a client that
            # asked for N and received N cannot otherwise tell a full page from
            # the end of the list, and silently stops early. It pages on with
            # ?after_id=<last_id>.
            has_more=has_more,
            first_id=batches[0].id if batches else None,
            last_id=batches[-1].id if batches else None,
        )
        return JSONResponse(content=listing.model_dump())

    async def anthropic_get_batch(self, batch_id: str) -> Response:
        batch = self._batch_store().get(batch_id)
        if batch is None:
            return self._no_such_batch(batch_id)
        return JSONResponse(content=batch.model_dump())

    async def anthropic_cancel_batch(self, batch_id: str) -> Response:
        batch = self._batch_store().cancel(batch_id)
        if batch is None:
            return self._no_such_batch(batch_id)
        return JSONResponse(content=batch.model_dump())

    async def anthropic_delete_batch(self, batch_id: str) -> Response:
        try:
            deleted = self._batch_store().delete(batch_id)
        except ValueError as e:
            return anthropic_error_response(str(e), "invalid_request_error",
                                            400)
        if not deleted:
            return self._no_such_batch(batch_id)
        return JSONResponse(content=AnthropicBatchDeleteResponse(
            id=batch_id).model_dump())

    async def anthropic_batch_results(self, batch_id: str) -> Response:
        store = self._batch_store()
        results = store.results(batch_id)
        if results is None:
            if not store.has(batch_id):
                return self._no_such_batch(batch_id)
            # Known batch, but not finished. 404 would say "no such batch",
            # which is wrong and would stop a client from polling.
            return anthropic_error_response(
                f"Message Batch {batch_id} is still processing; results are "
                "available once processing_status is 'ended'",
                "invalid_request_error", 400)
        return Response(content=results_to_jsonl(results),
                        media_type="application/x-ndjson")

    @staticmethod
    def _no_such_batch(batch_id: str) -> Response:
        # Batches are held in memory only, so an id from before a restart lands
        # here. Say so, rather than leaving the caller to wonder.
        return anthropic_error_response(
            f"Message Batch {batch_id} not found. Batches are stored in memory "
            "and do not survive a server restart.", "not_found_error", 404)
