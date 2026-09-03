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

import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, DraftTargetDecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root


@pytest.mark.parametrize("use_cuda_graph,attn_backend", [[False, "TRTLLM"], [True, "TRTLLM"]])
@pytest.mark.high_cuda_memory
def test_qwen3_draft_target(use_cuda_graph: bool, attn_backend: str):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 30:
        pytest.skip("Not enough memory to load target and draft models")

    models_path = llm_models_root()
    draft_model_dir = f"{models_path}/Qwen3/Qwen3-0.6B"
    target_model_dir = f"{models_path}/Qwen3/Qwen3-8B"

    max_batch_size = 2
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, max_batch_size]) if use_cuda_graph else None

    llm_common_config = dict(
        model=target_model_dir,
        backend="pytorch",
        attn_backend=attn_backend,
        disable_overlap_scheduler=True,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
    )

    spec_config = DraftTargetDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=draft_model_dir,
    )

    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]
    # Eight tokens require at least two DraftTarget iterations while avoiding
    # later shape-sensitive BF16 greedy ties observed on L40S.
    max_tokens = 8
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(prompts, sampling_params)
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    llm_ref.shutdown()

    for prompt, result_spec, result_ref in zip(prompts, results_spec, results_ref, strict=True):
        spec_output = result_spec.outputs[0]
        ref_output = result_ref.outputs[0]
        assert len(spec_output.token_ids) == max_tokens
        assert len(ref_output.token_ids) == max_tokens
        assert spec_output.token_ids == ref_output.token_ids, (
            f"DraftTarget output tokens differ from greedy reference for prompt {prompt!r}: "
            f"speculative={spec_output.token_ids}, reference={ref_output.token_ids}"
        )


@pytest.mark.high_cuda_memory
def test_qwen3_draft_target_rejection():
    """DraftTarget one-model with rejection sampling on: the rejection path
    (draft-prob capture -> fail-closed guard -> rejection acceptance) runs
    end-to-end with non-greedy sampling and produces coherent output."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 30:
        pytest.skip("Not enough memory to load target and draft models")

    models_path = llm_models_root()
    target_model_dir = f"{models_path}/Qwen3/Qwen3-8B"
    draft_model_dir = f"{models_path}/Qwen3/Qwen3-0.6B"

    spec_config = DraftTargetDecodingConfig(
        max_draft_len=4,
        speculative_model=draft_model_dir,
        use_rejection_sampling=True,
    )

    llm = LLM(
        model=target_model_dir,
        backend="pytorch",
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        max_batch_size=2,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=8192),
        max_num_tokens=2048,
        speculative_config=spec_config,
    )
    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]
    # Non-greedy so rejection sampling actually engages (all-greedy bypasses it).
    sampling_params = SamplingParams(
        max_tokens=32, temperature=0.8, top_p=0.95, top_k=50, seed=1234
    )
    outputs = llm.generate(prompts, sampling_params)
    llm.shutdown()

    assert len(outputs) == len(prompts)
    for out in outputs:
        assert len(out.outputs[0].token_ids) > 0
        assert out.outputs[0].text.strip()


if __name__ == "__main__":
    unittest.main()
