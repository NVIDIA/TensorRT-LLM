# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Module test_gpt test gpt examples."""

import defs.ci_profiler
import pytest
from defs.common import similar, similarity_score

from tensorrt_llm import LLM
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.sampling_params import SamplingParams


@pytest.mark.skip_less_device_memory(
    20000)  # Conservative 20GB requirement for GPT-OSS-20B
@pytest.mark.parametrize("gpt_oss_model_root", [
    "gpt-oss-20b",
], indirect=True)
@pytest.mark.parametrize("llm_lora_model_root",
                         ['gpt-oss-20b-lora-adapter_NIM_r8'],
                         indirect=True)
def test_gpt_oss_20b_lora_torch(gpt_example_root, llm_venv, gpt_oss_model_root,
                                llm_datasets_root, llm_rouge_root, engine_dir,
                                cmodel_dir, llm_lora_model_root):
    """Run GPT-OSS-20B with LoRA adapter using Torch backend."""

    print(f"Using LoRA from: {llm_lora_model_root}")

    defs.ci_profiler.start("test_gpt_oss_20b_lora_torch")

    lora_config = LoraConfig(
        lora_dir=[llm_lora_model_root],
        max_lora_rank=8,  # Match adapter_config.json "r": 8
        max_loras=1,
        max_cpu_loras=1,
    )

    with LLM(model=gpt_oss_model_root, lora_config=lora_config) as llm:

        prompts = [
            "User: Message Mason saying that we should compete in next week's football tournament, and tell him that the winner will get $100.\n\nAssistant: "
        ]

        sampling_params = SamplingParams(max_tokens=50)

        lora_request = [LoRARequest("gpt-oss-lora", 0, llm_lora_model_root)]

        print("Running inference with real LoRA adapter...")
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_request)

        expected_output = " Hey Mason! I hope you're doing well. I was thinking about the next week's football tournament and I wanted to give you a hint that we should compete in it. The winner will be a great opportunity for us to win $100.\n\nUser:"

        for i, output in enumerate(outputs):
            print(f"Prompt {i+1}: {prompts[i]}")
            print(f"Response {i+1}: {output.outputs[0].text}")
            print("-" * 50)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0
        generated_text = outputs[0].outputs[0].text
        similarity = similarity_score(generated_text, expected_output)
        assert similar(generated_text, expected_output, threshold=0.8), \
            f"Output similarity too low (similarity={similarity:.2%})!\nExpected: {repr(expected_output)}\nGot: {repr(generated_text)}"

    defs.ci_profiler.stop("test_gpt_oss_20b_lora_torch")
    print(
        f"test_gpt_oss_20b_lora_torch: {defs.ci_profiler.elapsed_time_in_sec('test_gpt_oss_20b_lora_torch')} sec"
    )
