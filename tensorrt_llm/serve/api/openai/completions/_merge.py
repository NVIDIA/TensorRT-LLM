# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functions of ``CompletionServing.openai_completion``, lifted from its closures.

Their free variables are keyword-only parameters: ``serving`` is the
``CompletionServing`` object, ``raw_request`` the HTTP request and
``disaggregated_params`` the request's disaggregated parameters.
"""

import asyncio
import json
import traceback
from typing import Any, AsyncIterator, List, Optional

from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (CompletionResponse,
                                                CompletionResponseChoice,
                                                PromptTokensDetails, UsageInfo)


async def completion_response(promise: RequestOutput,
                              postproc_params: Optional[PostprocParams], *,
                              serving, disaggregated_params,
                              raw_request) -> CompletionResponse:
    response = await promise
    if response.error is not None:
        raise RuntimeError(f"Generation failed: {response.error}")
    pp_result = serving._postproc.apply(response, postproc_params)
    if disaggregated_params and disaggregated_params.request_type and disaggregated_params.request_type == "context_only":
        # Include prompt token ids for context-only requests
        pp_result.prompt_token_ids = response.prompt_token_ids
    await serving._extract_metrics(response, raw_request)
    return pp_result


def merge_completion_responses(responses: List[CompletionResponse], *,
                               serving) -> CompletionResponse:
    all_choices: List[CompletionResponseChoice] = []
    all_prompt_token_ids: List[List[int]] = []
    num_prompt_tokens = num_gen_tokens = num_cached_tokens = 0
    for rsp in responses:
        choices, usage = rsp.choices, rsp.usage
        all_choices.extend(choices)
        num_prompt_tokens += usage.prompt_tokens
        num_gen_tokens += usage.completion_tokens
        num_cached_tokens += usage.prompt_tokens_details.cached_tokens
        # Aggregate prompt token ids for context-only requests
        if rsp.prompt_token_ids is not None:
            all_prompt_token_ids.append(rsp.prompt_token_ids)

    usage_info = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_gen_tokens,
        total_tokens=num_gen_tokens + num_prompt_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=num_cached_tokens, ),
    )
    merged_rsp = CompletionResponse(
        model=serving.model,
        choices=all_choices,
        usage=usage_info,
        prompt_token_ids=all_prompt_token_ids,
    )
    return merged_rsp


async def completion_generator(promise: RequestOutput,
                               params: Optional[PostprocParams], *, serving,
                               raw_request):
    try:
        async for output in promise:
            pp_result = serving._postproc.apply(output, params)
            for pp_res in pp_result:
                yield pp_res
        await serving._extract_metrics(output, raw_request)
    except Exception as e:
        logger.error(traceback.format_exc())
        # StreamingResponse commits HTTP 200 before the first
        # chunk, so we cannot change the status code.  Yield
        # an SSE error event so the stream terminates cleanly
        # instead of breaking the HTTP connection.
        error_data = json.dumps({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": None,
                "param": None,
            }
        })
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"


async def merge_generators(generators: List[AsyncIterator[Any]]):
    result_queue = asyncio.Queue()
    finished = [False] * len(generators)

    async def producer(generator: AsyncIterator[Any], idx: int):
        async for output in generator:
            await result_queue.put(output)
        finished[idx] = True

    tasks = [
        asyncio.create_task(producer(generator, idx))
        for idx, generator in enumerate(generators)
    ]

    while not all(finished) or not result_queue.empty():
        output = await result_queue.get()
        yield output
    await asyncio.gather(*tasks)


async def generator_wrapper(generator: AsyncIterator[Any], *, serving,
                            raw_request):
    first_response = await anext(generator)
    raw_request.state.server_first_token_time = serving._adjusted_steady_clock.now(
    )
    yield first_response
    async for output in generator:
        yield output
    yield "data: [DONE]\n\n"
