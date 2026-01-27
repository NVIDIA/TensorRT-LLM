# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import os
from functools import partial
from typing import List

import pytest
import torch
from transformers import AutoTokenizer
from utils.llm_data import llm_models_root
from utils.torch_ref import RefHFModel

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


async def generate_batch_async(
    hf_model: RefHFModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    responses: List[List[int]],
    prompt_max_len: int = 1024,
    micro_batch_size: int = 16,
) -> List[torch.Tensor]:
    """
    Async wrapper for generate_batch_with_padding.
    Runs the synchronous model inference in a thread pool.

    Args:
        hf_model: RefHFModel instance
        input_ids: Input token IDs
        attention_mask: Attention mask
        position_ids: Position IDs
        responses: List of response token IDs for each sample
        prompt_max_len: Maximum prompt length
        micro_batch_size: Size of micro batches for processing

    Returns:
        List of logprobs tensors, one per sample
    """
    loop = asyncio.get_event_loop()

    func = partial(
        hf_model.generate_batch_with_padding,
        prompt_max_len=prompt_max_len,
        micro_batch_size=micro_batch_size,
    )

    result = await loop.run_in_executor(
        None,  # Use default executor
        func,
        input_ids,
        attention_mask,
        position_ids,
        responses,
    )
    return result


def compare_logprobs(logprobs_list, ref_new_token_logprobs_list):
    """
    logprobs_list: List[torch.Tensor] - LLM logprob values
    ref_new_token_logprobs_list: List[torch.Tensor] - Ref logprobs

    Compares logprobs for each prompt separately.
    """
    assert len(logprobs_list) == len(ref_new_token_logprobs_list)

    final_max_diff = float("-inf")
    final_min_diff = float("inf")
    final_mean_diff = 0.0
    for llm_logprobs_i, ref_logprobs_i in zip(logprobs_list, ref_new_token_logprobs_list):
        logprobs_diff = ref_logprobs_i - llm_logprobs_i
        max_diff = logprobs_diff.max().item()
        min_diff = logprobs_diff.min().item()
        mean_diff = logprobs_diff.mean().item()

        final_max_diff = max(final_max_diff, max_diff)
        final_min_diff = min(final_min_diff, min_diff)
        final_mean_diff += mean_diff

    final_mean_diff = final_mean_diff / len(logprobs_list)
    # Given e^(-2.30) ≈ 0.1, the probability ratio should not drop below 0.1x
    assert abs(final_min_diff) < 2.30, (
        f"Final Min diff: {final_min_diff:.6f} is below threshold -2.30"
    )


@pytest.mark.gpu4
@pytest.mark.parametrize("model_dir", ["Qwen2-7B-Instruct"])
@pytest.mark.parametrize("sampler_type", ["TRTLLMSampler"])
@pytest.mark.parametrize("allreduce_strategy", ["NCCL", "AUTO"])
def test_accuracy_with_allreduce_strategy(model_dir, sampler_type, allreduce_strategy):
    """Test accuracy with different allreduce strategies.

    This test validates that both NCCL and AUTO allreduce strategies produce
    correct logprobs compared to HuggingFace reference implementation.

    Expected behavior:
        - allreduce_strategy="NCCL": Accuracy assertion PASSES
        - allreduce_strategy="AUTO": Accuracy assertion PASSES
    """
    model_dir = str(llm_models_root() / model_dir)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt_text = "The president of the United States is"
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    del tokenizer

    test_prompts = [prompt] * 256

    llm_logprobs = []
    llm_responses = []

    kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.6)
    llm = LLM(
        model=model_dir,
        backend="pytorch",
        orchestrator_type="ray",
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        kv_cache_config=kv_cache_config,
        max_seq_len=2048,
        max_batch_size=256,
        max_num_tokens=8192,
        tensor_parallel_size=4,
        sampler_type=sampler_type,
        allreduce_strategy=allreduce_strategy,
    )

    sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=1024, logprobs=1)
    outputs = llm.generate(test_prompts, sampling_params)

    for output in outputs:
        token_ids = output.outputs[0].token_ids
        logprobs_list = output.outputs[0].logprobs  # list[dict[int, Logprob]]
        # Extract logprob values from the list of dicts
        logprob_values = [
            logprobs[token_id].logprob for token_id, logprobs in zip(token_ids, logprobs_list)
        ]
        llm_responses.append(token_ids)
        llm_logprobs.append(torch.tensor(logprob_values, dtype=torch.float32, device="cuda"))

    del llm
    torch.cuda.empty_cache()

    input_ids, attention_mask, position_ids = RefHFModel.pad_data(test_prompts, llm_responses)

    # Split data across GPUs
    num_gpus = 4
    micro_batch_size = 16
    batch_size = input_ids.shape[0]
    samples_per_gpu = (batch_size + num_gpus - 1) // num_gpus

    dp_hf_models = []
    for device_id in range(num_gpus):
        hf_model = RefHFModel(model_dir, device_id)
        dp_hf_models.append(hf_model)

    # Split input data and responses into chunks for each GPU
    input_ids_chunks = []
    attention_mask_chunks = []
    position_ids_chunks = []
    responses_chunks = []

    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = min((i + 1) * samples_per_gpu, batch_size)

        if start_idx < batch_size:
            input_ids_chunks.append(input_ids[start_idx:end_idx])
            attention_mask_chunks.append(attention_mask[start_idx:end_idx])
            position_ids_chunks.append(position_ids[start_idx:end_idx])
            responses_chunks.append(llm_responses[start_idx:end_idx])

    # Process each chunk on its corresponding GPU asynchronously
    async def process_all_chunks(hf_models: List[RefHFModel]):
        tasks = []
        for i, (input_chunk, attn_chunk, pos_chunk, resp_chunk) in enumerate(
            zip(input_ids_chunks, attention_mask_chunks, position_ids_chunks, responses_chunks)
        ):
            task = generate_batch_async(
                hf_models[i],
                input_chunk,
                attn_chunk,
                pos_chunk,
                resp_chunk,
                prompt_max_len=1024,
                micro_batch_size=micro_batch_size,
            )
            tasks.append(task)
        return await asyncio.gather(*tasks)

    ref_logprobs_chunks = asyncio.run(process_all_chunks(dp_hf_models))

    # Move all tensors to cuda:0 and flatten the list
    # Each GPU returns a list of logprobs tensors
    ref_new_token_logprobs = []
    for i, logprobs_list in enumerate(ref_logprobs_chunks):
        for logprobs in logprobs_list:
            ref_new_token_logprobs.append(logprobs.to("cuda:0"))

    assert len(ref_new_token_logprobs) == batch_size, (
        f"Count mismatch: got {len(ref_new_token_logprobs)}, expected {batch_size}"
    )

    del dp_hf_models
    torch.cuda.empty_cache()

    # Compare LLM logprobs vs HF reference
    compare_logprobs(llm_logprobs, ref_new_token_logprobs)
