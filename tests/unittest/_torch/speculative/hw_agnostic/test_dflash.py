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
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

PROMPTS = [
    "The capital of France is",
    "The president of the United States is",
    "The future of AI is",
]


def _make_llm_config(
    target_model_dir: str,
    dflash_model_dir: str,
    disable_overlap_scheduler: bool,
    max_draft_len: int = 4,
    max_batch_size: int = 4,
):
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=2048)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True)
    spec_config = DFlashDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=dflash_model_dir,
    )
    return dict(
        model=target_model_dir,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=2048,
        enable_chunked_prefill=False,
        speculative_config=spec_config,
    )


def _run_and_check(llm_config: dict, min_avg_accepted: float):
    llm = LLM(**llm_config)
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=256, temperature=0))
    llm.shutdown()

    avg_accepted = [o.avg_decoded_tokens_per_iter - 1 for o in outputs]
    mean_accepted = sum(avg_accepted) / len(avg_accepted)
    assert mean_accepted >= min_avg_accepted, (
        f"mean avg accepted {mean_accepted:.2f} < threshold {min_avg_accepted} "
        f"(per request: {[f'{a:.2f}' for a in avg_accepted]})"
    )


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
def test_dflash_qwen3_8b(disable_overlap_scheduler: bool):
    """Test DFlash with Qwen3-8B BF16: CUDA graphs, padding, and draft acceptance."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    llm_config = _make_llm_config(
        target_model_dir=f"{models_path}/Qwen3/Qwen3-8B",
        dflash_model_dir=f"{models_path}/Qwen3-8B-DFlash-b16",
        disable_overlap_scheduler=disable_overlap_scheduler,
    )
    _run_and_check(llm_config, min_avg_accepted=1.0)


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
def test_dflash_qwen3_5_4b(disable_overlap_scheduler: bool):
    """Test DFlash with Qwen3.5-4B BF16: CUDA graphs, padding, and draft acceptance."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    llm_config = _make_llm_config(
        target_model_dir=f"{models_path}/Qwen3.5-4B",
        dflash_model_dir=f"{models_path}/Qwen3.5-4B-DFlash",
        disable_overlap_scheduler=disable_overlap_scheduler,
    )
    _run_and_check(llm_config, min_avg_accepted=1.0)


if __name__ == "__main__":
    unittest.main()
