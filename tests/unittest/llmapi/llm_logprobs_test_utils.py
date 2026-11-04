# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Optional

import torch

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

from .model_paths import llama_model_path


def llm_return_logprobs_test_harness(
    prompt_logprobs: Optional[int],
    logprobs: Optional[int],
    return_context_logits: bool,
    return_generation_logits: bool,
    tp_size=1,
    streaming=False,
    backend=None,
):
    kv_cache_args_extra = {}
    if streaming:
        # need this so that context_logits / prompt_logprobs are not dropped
        # in the 2nd reuse of llm.generate() in streaming mode
        kv_cache_args_extra["enable_block_reuse"] = False

    with LLM(
        llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4, **kv_cache_args_extra),
        tensor_parallel_size=tp_size,
        gather_generation_logits=True,
    ) as llm:
        prompts = ["A B C D E F G H I J K"]
        sampling_params = SamplingParams(
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            return_context_logits=return_context_logits,
            return_generation_logits=return_generation_logits,
        )

        for output in llm.generate(prompts, sampling_params):
            context_logits = output.context_logits
            generation_logits = output.outputs[0].generation_logits
            logprobs_result = output.outputs[0].logprobs
            prompt_logprobs_result = output.outputs[0].prompt_logprobs
            token_ids = output.outputs[0].token_ids

            # ensure logits are dropped unless users specify return_context_logits=True
            if prompt_logprobs and not return_context_logits:
                assert context_logits is None

            if logprobs and not return_generation_logits:
                assert generation_logits is None

            if return_context_logits:
                assert isinstance(context_logits, torch.Tensor)

            if return_generation_logits:
                assert isinstance(generation_logits, torch.Tensor)

            if prompt_logprobs:
                assert prompt_logprobs_result
                prompt_tokens = output.prompt_token_ids[1:] + token_ids[:1]
                assert len(prompt_logprobs_result) == len(prompt_tokens)
                for token, step_logprobs in zip(prompt_tokens, prompt_logprobs_result):
                    assert len(step_logprobs) in {prompt_logprobs, prompt_logprobs + 1}
                    assert token in step_logprobs
                    top_ranks = sorted(value.rank for value in step_logprobs.values())[
                        :prompt_logprobs
                    ]
                    assert top_ranks == list(range(1, prompt_logprobs + 1))

            if logprobs:
                assert logprobs_result
                assert len(logprobs_result) == len(token_ids)
                for token, step_logprobs in zip(token_ids, logprobs_result):
                    assert len(step_logprobs) in {logprobs, logprobs + 1}
                    assert token in step_logprobs
                    logprob_items = [
                        (value.logprob, value.rank) for value in step_logprobs.values()
                    ]
                    sorted_by_rank = sorted(logprob_items, key=lambda x: x[1])
                    assert [rank for _, rank in sorted_by_rank[:logprobs]] == list(
                        range(1, logprobs + 1)
                    )
                    for i in range(logprobs - 1):
                        assert sorted_by_rank[i][0] >= sorted_by_rank[i + 1][0]

        if streaming:

            async def task(id: int, prompt: str):
                logprobs_result_streaming = []
                async for output in llm.generate_async(prompt, sampling_params, streaming=True):
                    logprobs_result_streaming += output.outputs[0].logprobs_diff

                # comparing streaming logprobs result to non-streaming
                assert logprobs_result_streaming == logprobs_result
                assert output.outputs[0].prompt_logprobs == prompt_logprobs_result

            async def main():
                tasks = [task(id, prompt) for id, prompt in enumerate(prompts)]
                await asyncio.gather(*tasks)

            asyncio.run(main())
