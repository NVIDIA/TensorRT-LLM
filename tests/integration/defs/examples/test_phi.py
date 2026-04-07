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
import csv
import os

import defs.ci_profiler
import pytest
from defs.common import (convert_weights, quantize_data,
                         test_llm_torch_multi_lora_support, venv_check_call)
from defs.conftest import (get_sm_version, skip_fp8_pre_ada,
                           skip_post_blackwell, skip_pre_ada)
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


@pytest.mark.parametrize("data_type", ["float16", "fp8"],
                         ids=["base_fp16", "base_fp8"])
@pytest.mark.parametrize("lora_data_type", ["float16"], ids=["lora_fp16"])
@pytest.mark.parametrize("llm_phi_model_root", ["Phi-3-mini-4k-instruct"],
                         indirect=True)
@pytest.mark.parametrize("llm_lora_model_root",
                         ["Phi-3-mini-4k-instruct-ru-lora"],
                         indirect=True)
def test_llm_phi_lora_1gpu(data_type, lora_data_type, phi_example_root,
                           llm_phi_model_root, llm_datasets_root, llm_venv,
                           cmodel_dir, engine_dir, llm_lora_model_root,
                           qcache_dir_without_install_package):
    "run phi lora test on 1gpu"
    print("Converting checkpoint...")
    model_name = 'phi-3-lora'
    if data_type == 'fp8':
        skip_fp8_pre_ada(use_fp8=True)
        if get_sm_version() >= 100:
            pytest.skip("FP8 is not supported on post-Blackwell architectures")
        model_dir = quantize_data(
            llm_venv,
            phi_example_root,
            model_dir=llm_phi_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="fp8",
            kv_cache_dtype="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512)
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=phi_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llm_phi_model_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--gemm_plugin=auto",
        "--max_batch_size=8",
        f"--lora_dir={llm_lora_model_root}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        1, 1815, 366, 3867, 5837, 304, 17545, 18240, 310, 9892, 16397, 322,
        8338, 265, 29888, 21211, 29973, 306, 29915, 29885, 3063, 363, 907, 1230,
        322, 9045, 29891, 9522, 5547, 393, 11039, 403, 1716, 285, 21211, 29889,
        29871
    ]

    ref_2 = [
        1815, 366, 3867, 5837, 304, 17545, 18240, 310, 9892, 16397, 322, 8338,
        265, 29888, 21211, 29973, 13, 13, 7900, 22137, 29901, 315, 13946, 368,
        29991, 2266, 526, 777, 907, 1230, 5837, 304, 13389, 9892, 16397, 322
    ]

    input_text = "Can you provide ways to eat combinations of bananas and dragonfruits?"

    print(f"Run inference with lora id 0...")
    venv_check_call(llm_venv, [
        f"{phi_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict or data_type != "float16"

    print(f"Run inference with lora id -1...")
    venv_check_call(llm_venv, [
        f"{phi_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_phi_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]

    assert ref_2 == predict or data_type != "float16"


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

    expected_outputs = {
        'Phi-4-mini-instruct': ["...", "...", "...", "...", "..."],
    }

    print("Testing with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")
    model_name = os.path.basename(llm_phi_model_root).lower()
    test_llm_torch_multi_lora_support(
        hf_model_dir=llm_phi_model_root,
        llm_venv=llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["qkv_proj"],
        target_trtllm_modules=["attn_qkv"],
        zero_lora_weights=True,
        tensor_parallel_size=1,
        expected_outputs=expected_outputs[model_name])
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
