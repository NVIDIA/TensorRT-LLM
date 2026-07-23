# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import defs.ci_profiler
import pytest
from defs.common import test_llm_torch_multi_lora_support
from defs.conftest import get_sm_version, skip_pre_ada

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.fixture(scope="module")
def phi_example_root(llm_root, llm_venv):
    "Get phi example root"
    example_root = os.path.join(llm_root, "examples", "models", "core", "phi")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.mark.skip(
    reason="TODO: Resolve an import issue with transformers's LossKwargs")
@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llm_phi_model_root", ['Phi-4-mini-instruct'],
                         indirect=True)
def test_phi_4_mini_instruct_with_bf16_lora_torch(
        phi_example_root, llm_datasets_root, qcache_dir_without_install_package,
        llm_venv, engine_dir, llm_phi_model_root):
    """Run Phi-4-mini-instruct with multiple dummy LoRAs using LLM-API Torch backend."""

    print("Testing with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")
    test_llm_torch_multi_lora_support(hf_model_dir=llm_phi_model_root,
                                      llm_venv=llm_venv,
                                      num_loras=2,
                                      lora_rank=8,
                                      target_hf_modules=["qkv_proj"],
                                      target_trtllm_modules=["attn_qkv"],
                                      zero_lora_weights=True,
                                      tensor_parallel_size=1)
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
