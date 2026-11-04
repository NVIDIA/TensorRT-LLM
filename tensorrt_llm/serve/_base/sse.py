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
                                ctx) -> AsyncGenerator[str, None]:
    try:
        first_response = await anext(promise)
        ctx.state.server_first_token_time = server._adjusted_steady_clock.now()
        pp_results = server._postproc.apply(first_response, postproc_params)
        for pp_res in pp_results:
            yield pp_res
        # Making sure we can handle the situation where there is only one response
        res = first_response
        async for res in promise:
            pp_results = server._postproc.apply(res, postproc_params)
            for pp_res in pp_results:
                yield pp_res
        yield "data: [DONE]\n\n"
        await server._extract_metrics(res, ctx)
        nvtx_mark("generation ends")
    except Exception:
        # Only failures are logged: a consumer that stops reading closes
        # the stream with GeneratorExit, and a client that disconnects
        # cancels it, and neither is an error of this stream.
        logger.error(traceback.format_exc())
        raise
