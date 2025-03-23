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
import os
import platform
import uuid

import pytest
from defs.common import (convert_weights, generate_summary_cmd, quantize_data,
                         test_multi_lora_support, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import (evaltool_mmlu_post_process,
                           evaltool_wikilingua_post_process, get_device_memory,
                           skip_post_blackwell, skip_pre_ada)
from defs.trt_test_alternative import check_call
from evaltool.constants import (EVALTOOL_INFERENCE_SERVER_STARTUP_SCRIPT,
                                EVALTOOL_INFERENCE_SERVER_STOP_SCRIPT,
                                EVALTOOL_MMLU_CONFIG, EVALTOOL_MMLU_RESULT_FILE,
                                EVALTOOL_WIKILINGUA_CONFIG,
                                EVALTOOL_WIKILINGUA_RESULT_FILE)

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import BuildConfig, CalibConfig, QuantAlgo, QuantConfig


@pytest.fixture(autouse=True, scope="module")
def mistral_example_root(llm_venv):
    if platform.system() != "Windows":
        # https://github.com/Dao-AILab/flash-attention/issues/345
        # No wheel for flash-attn on windows and compilation fails locally.
        llm_venv.run_cmd(
            ['-m', 'pip', 'install', '--upgrade', 'flash-attn==2.4.2'])


@pytest.mark.parametrize("run_type", [
    'inference', 'summarization', 'summarization_long',
    'chunked_summarization_long'
])
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
    if run_type == "inference":
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
            f"--max_beam_width=4",
            f"--gpt_attention_plugin={data_type}",
            f"--gemm_plugin={data_type}",
            "--max_input_len=1024",
            "--max_batch_size=1",
            "--context_fmha=enable",
            "--max_seq_len=2048",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

        print("Run inference...")
        venv_check_call(llm_venv, [
            f"{llama_example_root}/../run.py",
            "--max_output_len=512",
            f"--tokenizer_dir={llm_mistral_model_root}",
            f"--engine_dir={engine_dir}",
            f"--max_attention_window_size={max_attention_window}",
        ])

    elif run_type == "summarization":
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
            f"--max_beam_width=4",
            f"--max_batch_size={1}",
            f"--max_input_len={1024}",
            f"--gpt_attention_plugin={data_type}",
            f"--gemm_plugin={data_type}",
            "--context_fmha=enable",
            "--max_seq_len=2048",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

        print("Run summarize...")
        summary_cmd = [
            f"{llama_example_root}/../summarize.py",
            "--test_trt_llm",
            "--hf_model_dir",
            f"{llm_mistral_model_root}",
            "--data_type",
            "fp16",
            f"--engine_dir={engine_dir}",
            "--tensorrt_llm_rouge1_threshold",
            "22",
            "--check_accuracy",
            f"--dataset_dir={llm_datasets_root}",
            f"--rouge_dir={llm_rouge_root}",
            f"--max_ite=100",
        ]
        venv_check_call(llm_venv, summary_cmd)

        print("Run summarize with beam_width = 2...")
        summary_cmd = [
            f"{llama_example_root}/../summarize.py",
            "--test_trt_llm",
            "--hf_model_dir",
            f"{llm_mistral_model_root}",
            "--data_type",
            "fp16",
            "--num_beams",
            "2",
            f"--engine_dir={engine_dir}",
            "--tensorrt_llm_rouge1_threshold",
            "22",
            "--check_accuracy",
            f"--dataset_dir={llm_datasets_root}",
            f"--rouge_dir={llm_rouge_root}",
            f"--max_ite=100",
        ]
        venv_check_call(llm_venv, summary_cmd)

        print("Run summarize with beam_width = 4...")
        summary_cmd = [
            f"{llama_example_root}/../summarize.py",
            "--test_trt_llm",
            "--hf_model_dir",
            f"{llm_mistral_model_root}",
            "--data_type",
            "fp16",
            "--num_beams",
            "4",
            f"--engine_dir={engine_dir}",
            "--tensorrt_llm_rouge1_threshold",
            "22",
            "--check_accuracy",
            f"--dataset_dir={llm_datasets_root}",
            f"--rouge_dir={llm_rouge_root}",
            f"--max_ite=100",
        ]
        venv_check_call(llm_venv, summary_cmd)

    elif run_type == "summarization_long":
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
            f"{llama_example_root}/../summarize.py",
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


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llm_mistral_model_root", ['mistral-7b-v0.1'],
                         indirect=True)
def test_llm_mistral_v1_smooth_quant_4gpus(llama_example_root,
                                           llm_mistral_model_root,
                                           llm_datasets_root, llm_rouge_root,
                                           llm_venv, cmodel_dir, engine_dir):
    "Run smooth quant test on 4 gpus"
    data_type = "float16"
    # --per_token & --per_channel are mandatory
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model="mistral-sq",
        model_path=llm_mistral_model_root,
        tp_size=4,
        pp_size=1,
        smoothquant=0.5,
        per_channel=True,
        per_token=True,
        data_type=data_type,
        calib_dataset=f"{llm_datasets_root}/ccdv/cnn_dailymail")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--max_input_len=1024",
        "--max_batch_size=1",
        "--context_fmha=enable",
        "--max_beam_width=4",
        "--workers=4",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llm_mistral_model_root,
                                       data_type="fp16",
                                       num_beams=4,
                                       engine_dir=engine_dir,
                                       tensorrt_llm_rouge1_threshold=23,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.parametrize("run_type", ['inference', 'summarization'])
@pytest.mark.parametrize("mistral_nemo_model_root", ['Mistral-Nemo-12b-Base'],
                         indirect=True)
def test_llm_mistral_nemo_fp8_quantization_1gpu(mistral_nemo_model_root,
                                                llama_example_root,
                                                run_type,
                                                llm_datasets_root,
                                                llm_rouge_root,
                                                llm_venv,
                                                cmodel_dir,
                                                engine_dir,
                                                qcache_dir,
                                                data_type='bfloat16',
                                                num_beams=1):
    if num_beams > 2 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    # Quantize HF llama checkpoint into FP8 format
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=mistral_nemo_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat="fp8",
        quantize_dir=qcache_dir,
        calib_size=512,
        kv_cache_dtype="fp8")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if run_type == "inference":
        print("Run inference...")
        venv_check_call(llm_venv, [
            f"{llama_example_root}/../run.py",
            "--max_output_len=50",
            f"--tokenizer_dir={mistral_nemo_model_root}",
            f"--engine_dir={engine_dir}",
            f"--num_beams={num_beams}",
        ])
    elif run_type == "summarization":
        print("Run summarize...")
        tensorrt_llm_rouge1_threshold = 24

        summary_cmd = generate_summary_cmd(
            llama_example_root,
            hf_model_dir=mistral_nemo_model_root,
            data_type=data_type,
            engine_dir=engine_dir,
            tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
            num_beams=num_beams,
            dataset_dir=llm_datasets_root,
            rouge_dir=llm_rouge_root)

        venv_check_call(llm_venv, summary_cmd)


@skip_pre_ada
@pytest.mark.parametrize("mistral_nemo_minitron_model_root",
                         ['Mistral-NeMo-Minitron-8B-Instruct'],
                         indirect=True)
def test_llm_mistral_nemo_minitron_fp8_quantization(
        mistral_nemo_minitron_model_root,
        llama_example_root,
        llm_datasets_root,
        llm_rouge_root,
        llm_venv,
        engine_dir,
        qcache_dir,
        qformat='fp8',
        num_beams=1):
    "Run Mistral Nemo Minitron 8B quantization."
    data_type = "bfloat16"
    tp_size, pp_size = 1, 1
    world_size = tp_size * pp_size

    print("Quantizing engine...")
    # Quantize HF llama checkpoint into FP8 format.
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=mistral_nemo_minitron_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat=qformat,
        quantize_dir=qcache_dir,
        tp_size=tp_size,
        pp_size=pp_size,
        calib_size=512)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        f"--moe_plugin={data_type}",
        f"--max_beam_width={num_beams}",
        "--context_fmha=enable",
        f"--workers={world_size}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    tensorrt_llm_rouge1_threshold = 22.0

    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=mistral_nemo_minitron_model_root,
        data_type=data_type,
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("qformat", ['fp8'])
@pytest.mark.parametrize("llm_mistral_model_root", ['mistral-7b-v0.1'],
                         indirect=True)
def test_llm_mistral_quantization_8gpus_summary(
        llama_example_root, llm_mistral_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, engine_dir, num_beams, qcache_dir, qformat):
    "run mixtral fp8 on 2gpus"
    data_type = "float16"
    tp_size, pp_size = 4, 2
    world_size = tp_size * pp_size

    print("Quantizing engine...")
    # Quantize HF llama checkpoint into FP8 format
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llm_mistral_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat=qformat,
        quantize_dir=qcache_dir,
        tp_size=tp_size,
        pp_size=pp_size,
        calib_size=32)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        f"--moe_plugin={data_type}",
        f"--max_beam_width={num_beams}",
        "--context_fmha=enable",
        f"--workers={world_size}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    tensorrt_llm_rouge1_threshold = 22.0

    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llm_mistral_model_root,
        data_type="fp16",
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.parametrize("llm_mistral_model_root", ['mistral-7b-v0.1'],
                         indirect=True)
def test_mistal_evaltool(llama_example_root, llm_mistral_model_root, llm_venv,
                         cmodel_dir, engine_dir, evaltool_root):

    print("Build engines...")

    data_type = "float16"
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model='mistral',
                                model_path=llm_mistral_model_root,
                                data_type=data_type)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--gather_context_logits",
        "--max_batch_size=8",
        "--max_input_len=5000",
        "--max_seq_len=7048",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Lm evaluation harness")

    # start inference server
    start_inference_server = [
        EVALTOOL_INFERENCE_SERVER_STARTUP_SCRIPT, "-e", engine_dir, "-t",
        llm_mistral_model_root, "-d", evaltool_root, "-m", "256"
    ]
    check_call(" ".join(start_inference_server), shell=True)

    task_list = ['mmlu', 'wikilingua']

    try:
        for task in task_list:
            project_id = str(uuid.uuid4())
            if task == "wikilingua":
                config_file = EVALTOOL_WIKILINGUA_CONFIG
                result_file = EVALTOOL_WIKILINGUA_RESULT_FILE

            if task == "mmlu":
                config_file = EVALTOOL_MMLU_CONFIG
                result_file = EVALTOOL_MMLU_RESULT_FILE

            model_name = os.path.basename(llm_mistral_model_root)
            # Update config dynamically
            import yaml
            with open(config_file, 'r') as f:
                lm_eval_config = yaml.safe_load(f)
                lm_eval_config['model']['llm_name'] = model_name
                lm_eval_config['model'][
                    'tokenizer_path'] = llm_mistral_model_root

            config_file = os.path.join(llm_venv.get_working_directory(),
                                       "lm_eval_config.yaml")
            with open(config_file, 'w') as f:
                yaml.dump(lm_eval_config, f)

            # launch evaluation
            run_cmd = [
                f"cd {evaltool_root}",
                "&&",
                "source .venv/bin/activate",
                "&&",
                "python3",
                "evaltool/interfaces/cli/main.py",
                "project",
                "launch",
                f"--eval_project_config_file '{config_file}'",
                "--infra_name local",
                f"--output_dir '{llm_venv.get_working_directory()}'",
                f"--project_id {project_id}",
            ]
            check_call(" ".join(run_cmd), shell=True, executable="/bin/bash")

            # process result
            result_path = f"{llm_venv.get_working_directory()}/{project_id}/{result_file}"
            check_call(f"cat {result_path}", shell=True)

            if task == 'mmlu':
                evaltool_mmlu_post_process(result_path, 0.6408, 0.006)
            if task == 'wikilingua':
                evaltool_wikilingua_post_process(result_path, 0.2443, 0.003)

    finally:
        # stop the server
        check_call(f"{EVALTOOL_INFERENCE_SERVER_STOP_SCRIPT}", shell=True)


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
        f"{llama_example_root}/../run.py",
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


@skip_post_blackwell
@skip_pre_ada
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("quant", ['int4', 'int4_awq', 'int8_awq'])
@pytest.mark.parametrize("llm_mistral_model_root", ['mistral-7b-v0.3'],
                         indirect=True)
def test_llm_mistral_quantization_4gpus_llmapi(llama_example_root,
                                               llm_mistral_model_root,
                                               llm_datasets_root, llm_venv,
                                               engine_dir, quant,
                                               mmlu_dataset_root):
    "run mixtral weight only int4/int8 on 4gpus"

    tp_size = 4

    if quant == 'int4':
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16)
    elif quant == 'int4_awq':
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
    elif quant == 'int8_awq':
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)

    calib_config = CalibConfig(
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        calib_batches=512,
        calib_max_seq_length=2048)

    build_config = BuildConfig()
    build_config.max_batch_size = 1
    build_config.max_input_len = 1900
    build_config.plugin_config.context_fmha = True
    build_config.plugin_config.paged_kv_cache = True
    build_config.plugin_config._use_paged_context_fmha = True

    llm = LLM(model=llm_mistral_model_root,
              auto_parallel_world_size=tp_size,
              tensor_parallel_size=tp_size,
              build_config=build_config,
              quant_config=quant_config,
              calib_config=calib_config)

    llm.save(engine_dir)

    prompt = "You are a friendly AI agent who can provide assistance to the customer regarding their recent order."

    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=128)

    with llm:
        output = llm.generate(prompt, sampling_params)
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )
        # Assert that output contains "Assistant" or "AI agent"
        generated_text = output.outputs[0].text.strip()
        assert ("Assistant" in generated_text) or (
            "AI agent" in generated_text
        ), "Generated text should start with either 'Assistant' or 'AI agent'"

    del llm

    threshold = 0.55 if 'int4' in quant else 0.6

    mmlu_cmd = [
        f"{llama_example_root}/../mmlu_llmapi.py",
        "--data_dir",
        f"{mmlu_dataset_root}",
        "--hf_model_dir",
        f"{llm_mistral_model_root}",
        "--engine_dir",
        f"{engine_dir}",
        "--backend=tensorrt",
        "--check_accuracy",
        f"--accuracy_threshold={threshold}",
        f"--tp_size={tp_size}",
    ]

    venv_check_call(llm_venv, mmlu_cmd)
