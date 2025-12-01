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

from tensorrt_llm.llmapi import GuidedDecodingParams


def test_ad_guided_decoding_regex_e2e():
    """Test guided decoding with regex pattern validation using the build_and_run_ad main()."""
    test_case = {
        "prompt": "What is the capital of France?",
        "regex": r"I don't know, I am a randomly initialized model|Paris",
        "valid_responses": ["I don't know, I am a randomly initialized model", "Paris"],
    }

    guided_decoding_backend = "xgrammar"

    experiment_config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")

    # DemoLLM runtime does not support guided decoding. Need to set runtime to trtllm.
    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["guided_decoding_backend"] = guided_decoding_backend

    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = {"prompt": test_case["prompt"]}

    cfg = ExperimentConfig(**experiment_config)

    # Need to introduce the guided decoding params after ExperimentConfig construction
    # because otherwise they get unpacked as a dict.
    cfg.prompt.sp_kwargs = {
        "max_tokens": 10,
        "top_k": None,
        "temperature": 0.1,
        "guided_decoding": GuidedDecodingParams(regex=test_case["regex"]),
    }

    print(f"Experiment config: {experiment_config}")
    print("Generating outputs...")
    results = main(cfg)
    print("Results:", results)

    # Parse and validate: output should be a prefix of one of the valid responses
    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1
    generated_text = prompts_and_outputs[0][1].strip()

    valid_responses = test_case["valid_responses"]
    is_valid_prefix = any(response.startswith(generated_text) for response in valid_responses)
    assert is_valid_prefix, (
        f"Output is not a valid prefix of any expected response.\n"
        f"Generated: '{generated_text}'\n"
        f"Valid responses: {valid_responses}"
    )
