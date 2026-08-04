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
from defs.common import test_llm_torch_multi_lora_support, venv_check_call
from defs.conftest import get_sm_version, skip_post_blackwell, skip_pre_ada
from defs.trt_test_alternative import check_call

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


@skip_pre_ada
@pytest.mark.parametrize("data_type", ['float16', 'bfloat16'])
@pytest.mark.parametrize("qformat", ['fp8'])
@pytest.mark.parametrize("llm_phi_model_root", [
    pytest.param("phi-2", marks=skip_post_blackwell),
    pytest.param("Phi-3-mini-128k-instruct", marks=skip_post_blackwell),
    pytest.param("Phi-3-small-128k-instruct", marks=skip_post_blackwell),
    pytest.param("Phi-3.5-mini-instruct", marks=skip_post_blackwell),
    "Phi-3.5-MoE-instruct", "Phi-4-mini-instruct"
],
                         indirect=True)
def test_llm_phi_quantization_1gpu(data_type, llm_phi_model_root, llm_venv,
                                   cmodel_dir, engine_dir, phi_example_root,
                                   llm_datasets_root, llm_rouge_root, qformat):
    "Run phi quantization tests"
    # Workaround for Modelopt can't convert Phi-3 on multi GPUs.
    gpu_constraint = {"CUDA_VISIBLE_DEVICES": "0"}

    print("Convert checkpoint by modelopt...")
    convert_cmd = [
        f"{phi_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_phi_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={data_type}",
        f"--qformat={qformat}",
        f"--kv_cache_dtype={qformat}",
        f"--output_dir={cmodel_dir}",
    ]
    venv_check_call(llm_venv, convert_cmd, env=gpu_constraint)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={cmodel_dir}",
        f"--output_dir={engine_dir}",
        "--max_input_len=3000",
        "--max_seq_len=3100",
        f"--max_batch_size={16}",
    ]

    build_env = {
        **llm_venv._new_env,
        **gpu_constraint
    } if llm_venv._new_env else gpu_constraint
    check_call(" ".join(build_cmd), shell=True, env=build_env)

    print("Run summarize...")
    threshold_score = 24.0
    model_name = os.path.basename(llm_phi_model_root)
    if model_name == "phi-2":
        threshold_score = 22.0

    summary_cmd = [
        f"{phi_example_root}/../../../summarize.py",
        "--test_trt_llm",
        f"--hf_model_dir={llm_phi_model_root}",
        f"--tokenizer_dir={llm_phi_model_root}",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={threshold_score}",
        "--max_ite=40",
        f"--batch_size={16}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]

    venv_check_call(llm_venv, summary_cmd, env=gpu_constraint)


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
