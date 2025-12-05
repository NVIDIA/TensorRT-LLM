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

from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm.llmapi.llm_args import SamplerType


def test_ad_trtllm_sampler_smoke():
    """Test TRTLLMSampler in AutoDeploy smoke test."""
    # Get small model config
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    experiment_config = get_small_model_config(model_id)

    # Configure for TRTLLMSampler
    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["sampler_type"] = SamplerType.TRTLLMSampler

    # Setup simple prompt
    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = {"prompt": "What is the capital of France?"}
    experiment_config["prompt"]["sp_kwargs"] = {
        "max_tokens": 10,
        "temperature": 1.0,
        "top_k": 1,
    }

    print(f"Experiment config: {experiment_config}")
    cfg = ExperimentConfig(**experiment_config)

    print("Running smoke test with TRTLLMSampler...")
    results = main(cfg)

    # Basic assertion that we got some output
    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1
    assert len(prompts_and_outputs[0][1]) > 0
