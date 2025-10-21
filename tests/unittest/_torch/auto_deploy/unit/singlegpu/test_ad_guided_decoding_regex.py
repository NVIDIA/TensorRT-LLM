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

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy import LLM
from tensorrt_llm.llmapi import GuidedDecodingParams


@pytest.mark.parametrize("guided_decoding_backend", ["xgrammar", "llguidance"])
def test_ad_guided_decoding_regex_e2e(guided_decoding_backend: str):
    """Test guided decoding with regex pattern validation.

    This test constructs an LLM in the AutoDeploy backend configured to output text matching a regex pattern,
    and checks that the generated outputs match the expected regex.

    Goal is to check that the AutoDeploy backend consistently works with regex-based guided decoding end-to-end.
    """
    # Define test cases with prompts and their corresponding regex patterns
    test_cases = [
        {
            "prompt": "What is the capital of France?",
            "regex": r"I don't know, I am a randomly initialized model|Paris",
            "valid_responses": ["I don't know, I am a randomly initialized model", "Paris"],
        },
        {
            "prompt": (
                "Let's echo the following statement: 'This is a string meant to test that "
                "the guided decoding sampler restricts a random network to a particular regex.'"
            ),
            "regex": (
                r"This is a string meant to test that the guided decoding sampler restricts "
                r"a random network to a particular regex\."
            ),
            "valid_responses": [
                "This is a string meant to test that the guided decoding sampler restricts "
                "a random network to a particular regex."
            ],
        },
    ]

    # Get small model config and prepare LLM kwargs
    experiment_config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llm_kwargs = experiment_config["args"]
    llm_kwargs["guided_decoding_backend"] = guided_decoding_backend

    # Instantiate the LLM with AutoDeploy backend.
    llm = LLM(**llm_kwargs)

    try:
        for test_idx, test_case in enumerate(test_cases):
            prompt = [{"prompt": test_case["prompt"]}]
            regex_pattern = test_case["regex"]
            valid_responses = test_case["valid_responses"]

            sampling_params = SamplingParams(
                max_tokens=10,
                top_k=None,
                temperature=0.1,
                guided_decoding=GuidedDecodingParams(regex=regex_pattern),
            )

            # Generate outputs with guided decoding
            print(f"\nTest case {test_idx + 1}: Running guided decoding with regex pattern...")
            print(f"Prompt: {test_case['prompt']}")
            print(f"Regex: {regex_pattern}")

            outputs = llm.generate(
                prompt,
                sampling_params=sampling_params,
            )

            print(f"Generated {len(outputs)} outputs")

            # Validate each output matches the regex pattern as a prefix
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text.strip()

                # Test that the output is a prefix of one of the valid responses
                # Valid prefixes: "P", "Pa", "Par", "Pari", "Paris", "I", "I ", "I d", "I don", etc.
                is_valid_prefix = any(
                    response.startswith(generated_text) or generated_text.startswith(response)
                    for response in valid_responses
                )

                assert is_valid_prefix, (
                    f"Test case {test_idx + 1}, Output {i} is not a valid prefix of '{valid_responses}'\n"
                    f"Generated text: '{generated_text}'"
                )

                print(f"Output {i} successfully matched as prefix: '{generated_text}'")

    finally:
        llm.shutdown()
