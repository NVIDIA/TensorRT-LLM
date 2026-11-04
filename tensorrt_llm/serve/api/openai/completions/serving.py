# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/completions``."""

import asyncio
import functools
import os
import signal
import time
import traceback
import uuid
from typing import List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.inputs.data import TokensPrompt
from tensorrt_llm.llmapi import tracing
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.thinking_budget import \
    add_thinking_budget_logits_processor
from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base.serving import (GenerateBaseServing,
                                              _record_generator_termination)
from tensorrt_llm.serve.api.openai.completions._merge import (
    completion_generator, completion_response, generator_wrapper,
    merge_completion_responses, merge_generators)
from tensorrt_llm.serve.conversation_id import resolve_request_conversation_id
from tensorrt_llm.serve.openai_protocol import (CompletionRequest,
                                                to_llm_conversation_params,
                                                to_llm_disaggregated_params)
from tensorrt_llm.serve.postprocess_handlers import (
    CompletionPostprocArgs, completion_response_post_processor,
    completion_stream_post_processor)


class CompletionServing(GenerateBaseServing):
    """``POST /v1/completions``.

    Takes no dependency beyond those of ``GenerateBaseServing``; the
    composition root builds it with ``_generate_serving_kwargs()``.
    """

    async def openai_completion(self, request: CompletionRequest,
                                raw_request: Request) -> Response:

        try:
            if isinstance(request.prompt, list) and not request.prompt:
                return self.create_error_response("'prompt' must not be empty.")
            if isinstance(request.prompt, str) or \
                (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            stream_response_id = None
            stream_created = None
            if request.stream and len(prompts) > 1:
                stream_response_id = f"cmpl-{uuid.uuid4().hex}"
                stream_created = int(time.time())

            promises: List[RequestOutput] = []
            postproc_params_collection: List[Optional[PostprocParams]] = []
            # Pass the model vocabulary size so ``logit_bias`` can be
            # expanded into an embedding bias tensor in the sampler.
            sampling_params = request.to_sampling_params(
                vocab_size=self._logit_bias_vocab_size(),
                gather_generation_logits=self.generator.args.
                gather_generation_logits,
                backend=self.generator.args.backend)
            add_thinking_budget_logits_processor(
                sampling_params,
                reasoning_parser=self.generator.args.reasoning_parser,
                tokenizer=self.tokenizer,
            )
            # TODO: better way to enable metrics
            if len(os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")) > 0:
                sampling_params.return_perf_metrics = True
            self._validate_internal_disagg_request(request, raw_request)
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)
            resolve_request_conversation_id(
                request, None if raw_request is None else raw_request.headers)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            for idx, prompt in enumerate(prompts):
                postproc_args = CompletionPostprocArgs.from_request(request)
                self._apply_spec_decode_stats_opt_in(postproc_args)
                postproc_args.prompt_idx = idx
                postproc_args.stream_response_id = stream_response_id
                postproc_args.stream_created = stream_created
                if request.echo:
                    postproc_args.prompt = prompt
                postproc_params = PostprocParams(
                    post_processor=completion_stream_post_processor
                    if request.stream else completion_response_post_processor,
                    postproc_args=postproc_args,
                )
                trace_headers = (None if raw_request is None else
                                 tracing.extract_trace_headers(
                                     raw_request.headers))

                prompt = prompt_inputs(prompt)
                if prompt.get("prompt") is not None:
                    loop = asyncio.get_event_loop()
                    prompt_token_ids, extra_processed_inputs = await loop.run_in_executor(
                        self._input_proc_executor,
                        functools.partial(self.generator.input_processor,
                                          prompt, sampling_params))
                    tokens_prompt = TokensPrompt(
                        prompt_token_ids=prompt_token_ids)
                else:
                    tokens_prompt = prompt

                promise = self.generator.generate_async(
                    inputs=tokens_prompt,
                    sampling_params=sampling_params,
                    _postproc_params=postproc_params,
                    streaming=request.stream,
                    lora_request=request.lora_request,
                    disaggregated_params=disaggregated_params,
                    conversation_params=conversation_params,
                    trace_headers=trace_headers,
                    kv_hints=request.kv_hints,
                    priority=request.priority if request.priority is not None
                    else DEFAULT_REQUEST_PRIORITY)
                asyncio.create_task(
                    self.await_disconnected(raw_request, promise))
                if not self.postproc_worker_enabled:
                    postproc_args.tokenizer = self.tokenizer
                    postproc_args.num_prompt_tokens = len(
                        promise.prompt_token_ids)
                promises.append(promise)
                postproc_params_collection.append(
                    None if self.postproc_worker_enabled else postproc_params)

            if request.stream:
                generators = [
                    completion_generator(promise,
                                         params,
                                         serving=self,
                                         raw_request=raw_request) for promise,
                    params in zip(promises, postproc_params_collection)
                ]
                response_generator = merge_generators(generators) if len(
                    promises) > 1 else generators[0]
                return StreamingResponse(content=generator_wrapper(
                    response_generator, serving=self, raw_request=raw_request),
                                         media_type="text/event-stream")
            else:
                rsps = await asyncio.gather(*[
                    completion_response(
                        promise,
                        params,
                        serving=self,
                        disaggregated_params=disaggregated_params,
                        raw_request=raw_request) for promise, params in zip(
                            promises, postproc_params_collection)
                ])
                response = merge_completion_responses(
                    rsps, serving=self) if len(rsps) > 1 else rsps[0]
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            _record_generator_termination(self.generator)
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))
