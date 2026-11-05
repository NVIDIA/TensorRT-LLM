# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-sent-event generators shared by the chat serving paths."""

import traceback
from typing import AsyncGenerator

from tensorrt_llm._utils import nvtx_mark
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger


async def chat_stream_generator(promise: RequestOutput,
                                postproc_params: PostprocParams, *, server,
                                raw_request) -> AsyncGenerator[str, None]:
    try:
        if not server.postproc_worker_enabled:
            post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
        first_response = await anext(promise)
        raw_request.state.server_first_token_time = server._adjusted_steady_clock.now(
        )
        pp_results = first_response.outputs[
            0]._postprocess_result if server.postproc_worker_enabled else post_processor(
                first_response, args)
        for pp_res in pp_results:
            yield pp_res
        # Making sure we can handle the situation where there is only one response
        res = first_response
        async for res in promise:
            pp_results = res.outputs[
                0]._postprocess_result if server.postproc_worker_enabled else post_processor(
                    res, args)
            for pp_res in pp_results:
                yield pp_res
        yield "data: [DONE]\n\n"
        await server._extract_metrics(res, raw_request)
        nvtx_mark("generation ends")
    except Exception:
        # Only failures are logged: a consumer that stops reading closes
        # the stream with GeneratorExit, and a client that disconnects
        # cancels it, and neither is an error of this stream.
        logger.error(traceback.format_exc())
        raise
