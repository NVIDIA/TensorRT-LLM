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

import unittest

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig


@pytest.mark.high_cuda_memory
def test_eagle3_one_model_kv_cache_reuse() -> None:
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        enable_partial_reuse=True,
        use_kv_cache_manager_v2=True,
        tokens_per_block=32,
        max_tokens=8192,
    )
    llm_common_config = dict(
        model=target_model_dir,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1]),
        max_batch_size=1,
        kv_cache_config=kv_cache_config,
        max_seq_len=8192,
    )
    llm_spec = LLM(
        **llm_common_config,
        speculative_config=Eagle3DecodingConfig(
            max_draft_len=4,
            speculative_model=eagle_model_dir,
            eagle3_one_model=True,
        ),
    )
    prompt = ("The quick brown fox jumped over the lazy dog. " * 20).strip() + " Once upon a time,"
    poison_prompt = "X Y Z W A B C D E F G H I J K L M N O P Q R S T U V " * 60
    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0,
        ignore_eos=True,
        return_perf_metrics=True,
    )

    llm_spec.generate(poison_prompt, SamplingParams(max_tokens=8, temperature=0))
    cold_result = llm_spec.generate(prompt, sampling_params)
    reuse_result = llm_spec.generate(prompt, sampling_params)
    llm_spec.shutdown()

    cold_acceptance = float(cold_result.avg_decoded_tokens_per_iter)
    reuse_acceptance = float(reuse_result.avg_decoded_tokens_per_iter)
    assert cold_result.outputs[0].text == reuse_result.outputs[0].text
    assert cold_acceptance > 1.0
    assert reuse_acceptance > 1.0
    assert reuse_result.cached_tokens >= 64
    assert reuse_acceptance >= cold_acceptance - 0.2


if __name__ == "__main__":
    unittest.main()
