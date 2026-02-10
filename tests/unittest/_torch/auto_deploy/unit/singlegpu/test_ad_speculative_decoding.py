# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from test_common.llm_data import with_mocked_hf_download_for_single_gpu

from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig


@pytest.mark.skip(
    reason="OOM on A30 GPUs on CI - speculative model loading does not support model_kwargs reduction"
)
@pytest.mark.parametrize("use_hf_speculative_model", [False, True])
@with_mocked_hf_download_for_single_gpu
def test_ad_speculative_decoding_smoke(use_hf_speculative_model: bool):
    """Test speculative decoding with AutoDeploy using the build_and_run_ad main()."""

    # Use a simple test prompt
    test_prompt = "What is the capital of France?"

    # Get base model config
    experiment_config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")
    speculative_model_hf_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if use_hf_speculative_model:
        # NOTE: this will still mock out the actual HuggingFace download
        speculative_model = speculative_model_hf_id
    else:
        speculative_model = get_small_model_config(speculative_model_hf_id)["args"]["model"]

    # Configure speculative decoding with a draft model
    spec_config = DraftTargetDecodingConfig(max_draft_len=3, speculative_model=speculative_model)

    # Configure KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.01,
    )

    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["speculative_config"] = spec_config
    experiment_config["args"]["kv_cache_config"] = kv_cache_config
    experiment_config["args"]["disable_overlap_scheduler"] = True
    experiment_config["args"]["max_num_tokens"] = 64

    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = test_prompt

    print(f"Experiment config: {experiment_config}")

    cfg = ExperimentConfig(**experiment_config)

    # Add sampling parameters (deterministic with temperature=0.0)
    cfg.prompt.sp_kwargs = {
        "max_tokens": 50,
        "top_k": None,
        "temperature": 0.0,
        "seed": 42,
    }

    print(f"Experiment config: {experiment_config}")
    print("Generating outputs with speculative decoding...")
    results = main(cfg)

    # Validate that we got output
    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1, "Should have exactly one prompt/output pair"

    prompt, generated_text = prompts_and_outputs[0]
    assert prompt == test_prompt, f"Prompt mismatch: expected '{test_prompt}', got '{prompt}'"
    assert len(generated_text) > 0, "Generated text should not be empty"

    print("Speculative decoding smoke test passed!")
