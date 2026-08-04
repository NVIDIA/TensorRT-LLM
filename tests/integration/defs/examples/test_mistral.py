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
"""Module test_mistral test mistral examples."""
import multiprocessing

import defs.ci_profiler
import psutil
import pytest
from defs.common import (convert_weights, test_llm_torch_multi_lora_support,
                         venv_check_call)
from defs.conftest import (get_device_count, get_sm_version,
                           skip_post_blackwell, skip_pre_ada)
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


def get_optimal_jobs():
    cpu_count = multiprocessing.cpu_count()
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    memory_per_job = 4
    memory_based_jobs = int(available_memory / memory_per_job)
    system_load = psutil.getloadavg()[0] / cpu_count
    if system_load > 0.7:
        cpu_factor = 0.5
    else:
        cpu_factor = 0.75
    cpu_based_jobs = max(1, int(cpu_count * cpu_factor))
    optimal_jobs = max(1, min(cpu_based_jobs, memory_based_jobs))
    return optimal_jobs


@skip_post_blackwell  #nvbug 5298661
@pytest.mark.parametrize(
    "run_type",
    ['inference', 'summarization_long', 'chunked_summarization_long'])
@pytest.mark.parametrize("max_attention_window", [4096],
                         ids=['max_attention_window_size_4096'])
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("llm_mistral_model_root", ['mistral-7b-v0.1'],
                         indirect=True)
def test_llm_mistral_v1_1gpu(run_type, data_type, llama_example_root,
                             max_attention_window, llm_mistral_model_root,
                             llm_datasets_root, llm_rouge_root, llm_venv,
                             cmodel_dir, engine_dir):

    print("Build engines...")
    if run_type == "summarization_long":
        model_name = 'mistral-{}'.format(run_type)
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llm_mistral_model_root,
                                    data_type=data_type)
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}",
            "--max_input_len",
            "6400",
            f"--max_batch_size={1}",
            "--max_seq_len",
            "6528",
            f"--gpt_attention_plugin={data_type}",
            f"--gemm_plugin={data_type}",
            "--context_fmha=enable",
            "--use_paged_context_fmha=disable",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

        print("Run long context summarize...")
        # using shorter input length since A30 doesn't have enough device memory.
        summary_cmd = [
            f"{llama_example_root}/summarize_long.py",
            "--test_trt_llm",
            "--test_hf",
            "--hf_model_location",
            f"{llm_mistral_model_root}",
            "--data_type",
            "fp16",
            f"--engine_dir={engine_dir}",
            f"--max_attention_window_size={max_attention_window}",
            "--max_ite",
            "3",
            "--max_input_len",
            "6400",
            "--tensorrt_llm_rouge1_threshold",
            "90",
            "--check_accuracy",
        ]
        # https://nvbugs/4658787
        # WAR before summarize_long.py can work offline
        env = {"HF_DATASETS_OFFLINE": "0"}
        venv_check_call(llm_venv, summary_cmd, env=env)

        # multi block + sliding window attention tests.
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}",
            "--max_input_len",
            "6400",
            "--max_seq_len",
            "6528",
            f"--gpt_attention_plugin={data_type}",
            f"--gemm_plugin={data_type}",
            "--use_paged_context_fmha=disable",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

        print("Run long context summarize with multi_block_mode enabled...")
        # using shorter input length since A30 doesn't have enough device memory.
        summary_cmd = [
            f"{llama_example_root}/summarize_long.py", "--test_trt_llm",
            "--test_hf", "--hf_model_location", f"{llm_mistral_model_root}",
            "--data_type", "fp16", f"--engine_dir={engine_dir}",
            f"--max_attention_window_size={max_attention_window}", "--max_ite",
            "3", "--max_input_len", "6400", "--tensorrt_llm_rouge1_threshold",
            "90", "--check_accuracy"
        ]
        venv_check_call(llm_venv, summary_cmd, env=env)

    elif run_type == "chunked_summarization_long":
        model_name = 'mistral-{}'.format(run_type)
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llm_mistral_model_root,
                                    data_type=data_type)
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}",
            "--max_input_len",
            "6400",
            "--max_num_tokens=2048",
            "--use_paged_context_fmha=enable",
            f"--max_batch_size={1}",
            "--max_seq_len",
            "6528",
            f"--gpt_attention_plugin={data_type}",
            f"--gemm_plugin={data_type}",
            "--context_fmha=enable",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

        print("Run long context summarize...")
        summary_cmd = [
            f"{llama_example_root}/../../../summarize.py",
            "--eval_task=summarize_long", "--test_trt_llm", "--test_hf",
            "--hf_model_dir", f"{llm_mistral_model_root}", "--data_type",
            "fp16", f"--engine_dir={engine_dir}",
            f"--max_attention_window_size={max_attention_window}",
            "--max_input_length", "6400", "--tensorrt_llm_rouge1_threshold",
            "21", "--check_accuracy", "--enable_chunked_context"
        ]
        # https://nvbugs/4658787
        # WAR before summarize_long.py can work offline
        env = {"HF_DATASETS_OFFLINE": "0"}
        venv_check_call(llm_venv, summary_cmd, env=env)


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llm_mistral_model_root", [
    'mistral-7b-v0.1',
    'mistral-nemo-instruct-2407',
],
                         indirect=True)
def test_mistral_with_bf16_lora_torch(llama_example_root, llm_datasets_root,
                                      qcache_dir_without_install_package,
                                      llm_venv, engine_dir,
                                      llm_mistral_model_root):
    """Run Mistral models with multiple dummy LoRAs using LLM-API Torch backend."""

    if "mistral-nemo-instruct-2407" in llm_mistral_model_root.lower():
        tensor_parallel_size = 2
        if get_device_count() < 2:
            pytest.skip(
                "Skipping: mistral-nemo-instruct-2407 model requires 2 GPUs")
    else:
        tensor_parallel_size = 1

    print(f"Testing {llm_mistral_model_root} with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")
    test_llm_torch_multi_lora_support(
        hf_model_dir=llm_mistral_model_root,
        llm_venv=llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        zero_lora_weights=True,
        tensor_parallel_size=tensor_parallel_size)
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
    print(
        f"test_llm_torch_multi_lora_support: {defs.ci_profiler.elapsed_time_in_sec('test_llm_torch_multi_lora_support')} sec"
    )
