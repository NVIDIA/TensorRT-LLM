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
import traceback
from typing import Callable, Optional

from fastapi import Query
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.api.anthropic.adapter import (
    AnthropicRequestError, anthropic_error_response,
    convert_anthropic_count_tokens_request)
from tensorrt_llm.serve.api.anthropic.batches import (AnthropicBatchStore,
                                                      BatchItemRunner,
                                                      BatchStoreFullError,
                                                      results_to_jsonl)
from tensorrt_llm.serve.api.anthropic.protocol import (
    AnthropicBatchDeleteResponse, AnthropicBatchList,
    AnthropicCountTokensRequest, AnthropicCountTokensResponse,
    AnthropicCreateBatchRequest)
from tensorrt_llm.serve.render import (HarmonyConversionError, RenderResources,
                                       UnsupportedRenderError, render_chat,
                                       render_chat_for_length)


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
