# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/embeddings`` on an encode-only ``LLM``."""

import array
import asyncio
import base64
import traceback
from http import HTTPStatus
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.api.openai.embeddings.batcher import (EncodeBatcher,
                                                              InputTooLongError,
                                                              QueueFullError)
from tensorrt_llm.serve.openai_protocol import (EmbeddingRequest,
                                                EmbeddingResponse,
                                                EmbeddingResponseData,
                                                EmbeddingUsageInfo)


class EmbeddingServing(BaseServing):
    """``POST /v1/embeddings`` on an encode-only ``LLM``.

    Concurrent requests are coalesced by ``EncodeBatcher``; the composition
    root keeps a reference to the batcher, because the lifespan starts and
    stops it and the health probe reports it.
    """

    def __init__(self, *, model: str, generator: Any,
                 embedding_max_queue_delay: float,
                 embedding_max_queue_size: int) -> None:
        self.model = model
        self.generator = generator
        self._embedding_max_queue_delay = embedding_max_queue_delay
        self._embedding_max_queue_size = embedding_max_queue_size
        self.embedding_batcher: Optional[EncodeBatcher] = None
        self._init_embedding_batcher()

    def _init_embedding_batcher(self):
        """Create the encode dynamic batcher for the embedding server.

        The batcher coalesces concurrent /v1/embeddings requests into a single
        `llm.encode()` call. `encode()` is synchronous; the batcher runs it in
        the default executor so the event loop stays responsive.
        """

        def encode_fn(token_ids_batch):
            # token_ids_batch: list of pre-tokenized token-id lists. encode()
            # returns one EncoderOutput per item, in input order.
            return self.generator.encode(token_ids_batch)

        # The limits encode() checks every call against; see LLM.encode_limits.
        limits = self.generator.encode_limits
        self.embedding_batcher = EncodeBatcher(
            encode_fn,
            max_batch_size=limits.max_batch_size,
            max_queue_delay=self._embedding_max_queue_delay,
            max_queue_size=self._embedding_max_queue_size,
            max_num_tokens=limits.max_num_tokens,
            max_seq_len=limits.max_seq_len,
        )

    @staticmethod
    def _normalize_embedding_input(inp):
        """Normalize EmbeddingRequest.input into a list of items.

        Each item is either a string (to tokenize) or a pre-tokenized token-id
        list. Mirrors the four OpenAI input forms: str | list[str] | list[int] |
        list[list[int]].
        """
        if isinstance(inp, str):
            return [inp]
        if isinstance(inp, list):
            if len(inp) == 0:
                return []
            if isinstance(inp[0], int):
                return [inp]  # a single pre-tokenized input
            return list(inp)  # list of strings or list of token-id lists
        raise TypeError(f"Unsupported embedding input type: {type(inp)}")

    async def openai_embedding(self, request: EmbeddingRequest,
                               raw_request: Request) -> Response:
        try:
            # `dimensions` is a Matryoshka-embedding knob (truncate-then-renormalize
            # a sentence-embedding vector), matching vLLM, which rejects it for
            # non-Matryoshka models. None of the models served here are
            # Matryoshka-trained: BERT classifiers / reward models emit raw
            # [num_labels] / [seq_len, num_labels] tensors, and Qwen3-Embedding emits
            # a fixed-width pooled vector with no nested sub-dimensions. Truncating
            # any of these is meaningless, so reject. (Honoring `dimensions` for a
            # genuinely Matryoshka model would be a fast-follow.)
            if request.dimensions is not None:
                return self.create_error_response(
                    "`dimensions` is only supported by Matryoshka-trained "
                    "text-embedding models; none of the models served by this "
                    "endpoint are Matryoshka-trained.",
                    status_code=HTTPStatus.BAD_REQUEST)

            items = self._normalize_embedding_input(request.input)
            if not items:
                return self.create_error_response("`input` must not be empty.")

            # Tokenize strings via the same input processor used by /v1/completions,
            # honoring add_special_tokens. Pre-tokenized inputs are used as-is.
            sampling_params = SamplingParams(
                add_special_tokens=request.add_special_tokens)
            token_ids_list = []
            for item in items:
                if isinstance(item, str):
                    token_ids, _ = await asyncio.to_thread(
                        self.generator.input_processor, prompt_inputs(item),
                        sampling_params)
                else:
                    token_ids = item
                token_ids_list.append(token_ids)

            try:
                # Validate every input's length up front so an oversize item fails
                # the whole request before any item is enqueued — otherwise gather()
                # cancels the awaiters but their already-queued inputs still run
                # encode() (wasted GPU work).
                for token_ids in token_ids_list:
                    self.embedding_batcher.validate_input(token_ids)
                results = await asyncio.gather(*[
                    self.embedding_batcher.submit(token_ids)
                    for token_ids in token_ids_list
                ])
            except InputTooLongError as e:
                return self.create_error_response(
                    str(e), status_code=HTTPStatus.BAD_REQUEST)
            except QueueFullError as e:
                return self.create_error_response(
                    str(e), status_code=HTTPStatus.TOO_MANY_REQUESTS)

            data = []
            for idx, encoder_output in enumerate(results):
                # Model-agnostic: serialize whatever per-request tensor encode()
                # emits (classification logits, reward score, pooled vector, ...).
                embedding = encoder_output.logits.flatten().tolist()
                if request.encoding_format == "base64":
                    embedding = base64.b64encode(
                        array.array("f", embedding).tobytes()).decode("utf-8")
                data.append(
                    EmbeddingResponseData(index=idx, embedding=embedding))

            num_prompt_tokens = sum(len(t) for t in token_ids_list)
            usage = EmbeddingUsageInfo(prompt_tokens=num_prompt_tokens,
                                       total_tokens=num_prompt_tokens)
            response = EmbeddingResponse(model=request.model,
                                         data=data,
                                         usage=usage)
            return JSONResponse(content=response.model_dump())
        except Exception as e:
            # Unexpected fault (not a client error): surface as 500, not 400.
            logger.error(traceback.format_exc())
            return self.create_error_response(
                str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
