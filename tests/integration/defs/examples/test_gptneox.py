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
"""Module test_gptneox test gptneox examples."""
import os

import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import get_device_memory
from defs.trt_test_alternative import check_call


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("context_fmha_type",
                         ['enabled', 'enabled_with_fp32_acc', 'disabled'])
@pytest.mark.parametrize("use_weight_only_groupwise_quant_matmul_plugin",
                         [True, False],
                         ids=[
                             "enable_weight_only_groupwise_quant_matmul_plugin",
                             "disable_weight_only_groupwise_quant_matmul_plugin"
                         ])
def test_llm_gptneox_single_gpu_summary(
        gptneox_example_root, llm_gptneox_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, engine_dir, use_gemm_plugin,
        context_fmha_type, use_weight_only_groupwise_quant_matmul_plugin,
        use_attention_plugin, num_beams):
    "Build & run gptneox on single gpu."
    if (num_beams == 4 or not use_weight_only_groupwise_quant_matmul_plugin) \
        and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    print("Building engines...")

    gptneox_gptq_safetensors_root = os.path.join(
        llm_gptneox_model_root, "..", "int4-quantized-gptq-awq",
        "gptneox-20b-4bit-gs128.safetensors")
    if use_weight_only_groupwise_quant_matmul_plugin:
        use_weight_only = True
        weight_only_precision = 'int4_gptq'
        quant_ckpt_path = gptneox_gptq_safetensors_root
    else:
        use_weight_only = False
        weight_only_precision = None
        quant_ckpt_path = None
    dtype = "float16"
    model_name = 'gpt-neox-20b'
    cmodel_dir = os.path.join(engine_dir, "cmodel")
    cmodel_dir = convert_weights(llm_venv=llm_venv,
                                 example_root=gptneox_example_root,
                                 cmodel_dir=cmodel_dir,
                                 model=model_name,
                                 model_path=llm_gptneox_model_root,
                                 data_type=dtype,
                                 use_weight_only=use_weight_only,
                                 weight_only_precision=weight_only_precision,
                                 quant_ckpt_path=quant_ckpt_path)

    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={cmodel_dir}",
        f"--max_batch_size={1}", f"--max_input_len={1024}",
        f"--output_dir={engine_dir}", f"--max_beam_width={num_beams}"
    ]
    if use_attention_plugin:
        #gpt-neox paged kv cache is not implemented yet
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
        if context_fmha_type == 'enabled':
            build_cmd.append("--context_fmha=enable")
        elif context_fmha_type == "disabled":
            build_cmd.append("--context_fmha=disable")
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])

    if use_gemm_plugin:
        build_cmd.append(f"--gemm_plugin={dtype}")
    else:
        build_cmd.append("--gemm_plugin=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    summary_cmd = generate_summary_cmd(gptneox_example_root,
                                       engine_dir=engine_dir,
                                       hf_model_dir=llm_gptneox_model_root,
                                       batch_size=1,
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold=14,
                                       data_type="fp16",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    if context_fmha_type == "enabled_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "use_parallel_embedding", [True, False],
    ids=["enable_parallel_embedding", "disable_parallel_embedding"])
@pytest.mark.parametrize("use_weight_only_groupwise_quant_matmul_plugin",
                         [True, False],
                         ids=[
                             "enable_weight_only_groupwise_quant_matmul_plugin",
                             "disable_weight_only_groupwise_quant_matmul_plugin"
                         ])
def test_llm_gptneox_summary_1node_2gpus(
        gptneox_example_root, llm_gptneox_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, cmodel_dir, engine_dir,
        use_parallel_embedding, use_weight_only_groupwise_quant_matmul_plugin,
        num_beams):
    "Build & run gptneox on 2 gpus."
    if num_beams == 4 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    print("Building engines...")

    gptneox_gptq_safetensors_root = os.path.join(
        llm_gptneox_model_root, "..", "int4-quantized-gptq-awq",
        "gptneox-20b-4bit-gs128.safetensors")

    if use_weight_only_groupwise_quant_matmul_plugin:
        use_weight_only = True
        weight_only_precision = 'int4_gptq'
        quant_ckpt_path = gptneox_gptq_safetensors_root
    else:
        use_weight_only = False
        weight_only_precision = None
        quant_ckpt_path = None

    dtype = "float16"
    model_name = 'gpt-neox-20b'

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=gptneox_example_root,
                                gpus=2,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llm_gptneox_model_root,
                                data_type=dtype,
                                use_parallel_embedding=use_parallel_embedding,
                                use_weight_only=use_weight_only,
                                weight_only_precision=weight_only_precision,
                                quant_ckpt_path=quant_ckpt_path)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--max_beam_width={num_beams}",
        "--max_batch_size=16",
        "--max_input_len=1024",
        "--max_seq_len=2048",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        "--context_fmha=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{gptneox_example_root}/../run.py",
        "--max_output_len=50",
        f"--tokenizer_dir={llm_gptneox_model_root}",
        f"--engine_dir={engine_dir}",
    ]
    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "-np", "2", "--allow-run-as-root", "--oversubscribe"],
        run_cmd)

    print("Run summary...")
    summary_cmd = generate_summary_cmd(gptneox_example_root,
                                       engine_dir=engine_dir,
                                       hf_model_dir=llm_gptneox_model_root,
                                       batch_size=1,
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold=13.5,
                                       data_type="fp16",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "-np", "2", "--allow-run-as-root", "--oversubscribe"],
        summary_cmd)
