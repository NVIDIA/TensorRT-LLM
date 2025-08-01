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
import platform

import psutil
import pytest
from defs.common import (convert_weights, quantize_data,
                         test_multi_lora_support, venv_check_call)
from defs.conftest import skip_post_blackwell, skip_pre_ada
from defs.trt_test_alternative import check_call


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


@pytest.fixture(autouse=True, scope="module")
def mistral_example_root(llm_venv):
    if platform.system() != "Windows":
        # https://github.com/Dao-AILab/flash-attention/issues/345
        # No wheel for flash-attn on windows and compilation fails locally.
        max_jobs = get_optimal_jobs()
        install_cmd = [
            f"MAX_JOBS={max_jobs}",
            "python3",
            "-m",
            "pip",
            "install",
            "--upgrade",
            "flash-attn==2.4.2",
        ]

        check_call(" ".join(install_cmd), shell=True, env=llm_venv._new_env)


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
@pytest.mark.parametrize("llm_mistral_model_root", ['komt-mistral-7b-v1'],
                         indirect=True)
@pytest.mark.parametrize("llm_lora_model_root", ['komt-mistral-7b-v1-lora'],
                         indirect=True)
def test_llm_mistral_lora_1gpu(llama_example_root, llm_mistral_model_root,
                               llm_datasets_root, llm_venv, engine_dir,
                               llm_lora_model_root, qcache_dir):
    "run mistral lora test on 1gpu"
    print("Quantization...")
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llm_mistral_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="float16",
        qformat="fp8",
        quantize_dir=qcache_dir,
        calib_size=512,
        kv_cache_dtype="fp8")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--lora_dir={llm_lora_model_root}",
        "--lora_plugin=auto",
        "--gemm_plugin=auto",
        "--max_batch_size=8",
        "--max_input_len=32256",
        "--max_seq_len=33280",
        "--use_paged_context_fmha=enable",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    input_text = "[INST]오늘은 날씨가 아주 좋다 내가 공원에 갔을 때 [/INST]"

    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        f"--input_text={input_text}",
        f"--tokenizer_dir={llm_mistral_model_root}",
        f"--engine_dir={engine_dir}",
        "--max_output_len=1024",
        "--max_attention_window_size=4096",
        "--lora_task_uids=0",
        "--temperature=0.8",
        "--top_p=0.8",
        "--top_k=100",
        "--random_seed=0",
    ]

    venv_check_call(llm_venv, run_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("mistral_nemo_minitron_model_root",
                         ['Mistral-NeMo-Minitron-8B-Instruct'],
                         indirect=True)
def test_mistral_nemo_minitron_fp8_with_bf16_lora(
    llama_example_root,
    mistral_nemo_minitron_model_root,
    llm_datasets_root,
    qcache_dir,
    llm_rouge_root,
    llm_venv,
    engine_dir,
    num_beams=1,
):
    "Run Mistral Nemo Minitron 8B with multiple pseudo LoRAs."

    # Quantize the base model to fp8.
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=mistral_nemo_minitron_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="bfloat16",
        qformat="fp8",
        quantize_dir=qcache_dir,
        calib_size=32,
        kv_cache_dtype="fp8")

    test_multi_lora_support(
        hf_model_dir=mistral_nemo_minitron_model_root,
        tllm_ckpt_dir=qmodel_dir,
        engine_dir=engine_dir,
        llm_venv=llm_venv,
        example_root=llama_example_root,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
        zero_lora_weights=True,
    )
