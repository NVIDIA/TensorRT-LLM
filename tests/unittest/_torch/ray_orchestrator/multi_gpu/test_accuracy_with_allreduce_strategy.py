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
from typing import List, Tuple

import pytest
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


class HFModel:
    def __init__(self, model_name: str, device_id: int):
        self.device_id = device_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(f"cuda:{device_id}")

    def generate_batch_with_padding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        responses: List[List[int]],
        prompt_max_len: int = 1024,
        micro_batch_size: int = 16,
    ):
        """
        Synchronous inference on a batch with micro-batching.
        Directly extracts response logprobs to save memory.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            responses: List of response token IDs for each sample
            prompt_max_len: Maximum prompt length (default 1024)
            micro_batch_size: Size of each micro batch to avoid OOM

        Returns:
            List of logprobs tensors, one per sample [response_len]
        """
        # Move tensors to the correct device
        input_ids = input_ids.to(f"cuda:{self.device_id}")
        attention_mask = attention_mask.to(f"cuda:{self.device_id}")
        position_ids = position_ids.to(f"cuda:{self.device_id}")

        batch_size = input_ids.shape[0]
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        all_response_logprobs = []

        with torch.no_grad():
            for micro_idx in range(num_micro_batches):
                start_idx = micro_idx * micro_batch_size
                end_idx = min((micro_idx + 1) * micro_batch_size, batch_size)

                # Extract micro batch
                micro_input_ids = input_ids[start_idx:end_idx]
                micro_attention_mask = attention_mask[start_idx:end_idx]
                micro_position_ids = position_ids[start_idx:end_idx]

                # Forward pass
                outputs = self.model(
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                    position_ids=micro_position_ids,
                )

                # Extract response logprobs for each sample in this micro batch
                micro_logits = outputs.logits  # [micro_batch_size, seq_len, vocab_size]

                for i in range(micro_logits.shape[0]):
                    sample_idx = start_idx + i
                    response = responses[sample_idx]
                    response_len = len(response)

                    # Extract logits for predicting response tokens
                    # For predicting response[j], we need logits at position prompt_max_len-1+j
                    response_logits = micro_logits[
                        i, prompt_max_len - 1 : prompt_max_len - 1 + response_len, :
                    ]

                    # Convert to logprobs
                    response_logprobs = torch.log_softmax(response_logits, dim=-1)

                    # Extract logprobs for the actual generated tokens
                    response_tensor = torch.tensor(
                        response, dtype=torch.long, device=response_logprobs.device
                    )
                    ref_logprob_for_tokens = torch.gather(
                        response_logprobs, dim=-1, index=response_tensor.unsqueeze(-1)
                    ).squeeze(-1)

                    all_response_logprobs.append(ref_logprob_for_tokens)

                # Free memory immediately after processing each micro batch
                del outputs, micro_logits
                torch.cuda.empty_cache()

        return all_response_logprobs


async def generate_batch_async(
    hf_model: HFModel,
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
        hf_model: HFModel instance
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


def pad_data(
    original_prompts: List[List[int]],
    generated_token_ids_list: List[List[int]],
    prompt_max_len: int = 1024,
    response_max_len: int = 1024,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad the data to the maximum length.

    Structure:
    [left_pad | actual_prompt | actual_response | right_pad]
    |<--  prompt_max_len=1024  -->|<--  response_max_len=1024  -->|

    Args:
        original_prompts: List of prompt token IDs, len = batch_size
        generated_token_ids_list: List of response token IDs, len = batch_size
        prompt_max_len: Maximum length for prompt section (default 1024)
        response_max_len: Maximum length for response section (default 1024)
        pad_token_id: Token ID for padding (default 0)
    Returns:
        input_ids: Tensor of shape [batch_size, prompt_max_len + response_max_len]
        attention_mask: Tensor of shape [batch_size, prompt_max_len + response_max_len]
        position_ids: Tensor of shape [batch_size, prompt_max_len + response_max_len]
    """
    batch_size = len(original_prompts)
    total_len = prompt_max_len + response_max_len

    for i, (prompt, response) in enumerate(zip(original_prompts, generated_token_ids_list)):
        assert len(prompt) <= prompt_max_len, (
            f"Batch {i}: Prompt length {len(prompt)} exceeds max {prompt_max_len}"
        )
        assert len(response) <= response_max_len, (
            f"Batch {i}: Response length {len(response)} exceeds max {response_max_len}"
        )

    # Build batch tensors [batch_size, 2048]
    batch_input_ids = torch.full(
        (batch_size, total_len), pad_token_id, dtype=torch.long, device="cuda"
    )
    batch_attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long, device="cuda")
    batch_position_ids = torch.zeros((batch_size, total_len), dtype=torch.long, device="cuda")

    response_lens = []

    for i in range(batch_size):
        prompt_tokens = original_prompts[i]
        response_tokens = generated_token_ids_list[i]

        prompt_len = len(prompt_tokens)
        response_len = len(response_tokens)
        response_lens.append(response_len)

        left_pad_len = prompt_max_len - prompt_len

        # Fill input_ids: [left_pad | prompt | response | right_pad]
        prompt_start = left_pad_len
        prompt_end = prompt_max_len
        response_start = prompt_max_len
        response_end = prompt_max_len + response_len

        batch_input_ids[i, prompt_start:prompt_end] = torch.tensor(
            prompt_tokens, dtype=torch.long, device="cuda"
        )
        batch_input_ids[i, response_start:response_end] = torch.tensor(
            response_tokens, dtype=torch.long, device="cuda"
        )

        # Fill attention_mask: 1 for actual tokens, 0 for padding
        batch_attention_mask[i, prompt_start:response_end] = 1

        # Fill position_ids: sequential for actual tokens
        actual_seq_len = prompt_len + response_len
        batch_position_ids[i, prompt_start:response_end] = torch.arange(
            actual_seq_len, dtype=torch.long, device="cuda"
        )
        # Right padding keeps the last position value
        if response_len < response_max_len:
            batch_position_ids[i, response_end:] = actual_seq_len - 1

    return batch_input_ids, batch_attention_mask, batch_position_ids


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

    The default allreduce_strategy (AUTO) produced wrong logprobs with large batch size,
    causing VeRL integration to fail to converge. There may be an issue with the
    customAllReduce kernels.

    Tracked: NVBug (https://nvbugs/5727691)

    Expected behavior:
        - allreduce_strategy="NCCL": Accuracy assertion PASSES
        - allreduce_strategy="AUTO": Accuracy assertion FAILS
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
    try:
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
    finally:
        if ray.is_initialized():
            ray.shutdown()

    torch.cuda.empty_cache()
    input_ids, attention_mask, position_ids = pad_data(test_prompts, llm_responses)

    # Split data across GPUs
    num_gpus = 4
    micro_batch_size = 16
    batch_size = input_ids.shape[0]
    samples_per_gpu = (batch_size + num_gpus - 1) // num_gpus

    dp_hf_models = []
    for device_id in range(num_gpus):
        hf_model = HFModel(model_dir, device_id)
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
    async def process_all_chunks(hf_models: List[HFModel]):
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
    if allreduce_strategy == "AUTO":
        with pytest.raises(AssertionError, match=r"Final Min diff: .* is below threshold -2\.30"):
            compare_logprobs(llm_logprobs, ref_new_token_logprobs)
    else:
        compare_logprobs(llm_logprobs, ref_new_token_logprobs)
