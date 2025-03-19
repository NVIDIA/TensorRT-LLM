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
"""Module test_mpt test mpt examples."""
import os

import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import skip_pre_ada
from defs.trt_test_alternative import check_call


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("data_type", ['bfloat16'])
@pytest.mark.parametrize("use_plugins", [True, False],
                         ids=['enable_plugins', 'disable_plugins'])
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_context_fmha', 'enable_context_fmha_fp32_acc', 'disable_fmha'])
def test_llm_mpt_7b_1node_4gpus(mpt_example_root, llm_venv,
                                llm_mpt_7b_model_root, llm_datasets_root,
                                llm_rouge_root, cmodel_dir, engine_dir,
                                data_type, use_plugins, context_fmha_type):
    "mpt 7b test on 4gpus"
    print("Converting MPT weights...")
    model_name = os.path.basename(llm_mpt_7b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=mpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_mpt_7b_model_root,
                               data_type=data_type,
                               gpus=4)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={4}",
        f"--max_input_len={2048}",
        f"--max_seq_len={2560}",
        f"--workers={4}",
    ]

    if use_plugins:
        if context_fmha_type == "enable_fmha":
            build_cmd.append("--context_fmha=enable")
        elif context_fmha_type == "disable_fmha":
            build_cmd.append("--context_fmha=disable")
        build_cmd.extend([
            f"--gpt_attention_plugin={data_type}", f"--gemm_plugin={data_type}"
        ])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--gemm_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    summary_cmd = generate_summary_cmd(mpt_example_root,
                                       hf_model_dir=llm_mpt_7b_model_root,
                                       engine_dir=engine_dir,
                                       data_type="fp16",
                                       tensorrt_llm_rouge1_threshold=18,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    if context_fmha_type == "enable_context_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("data_type", ['bfloat16'])
@pytest.mark.parametrize("use_plugins", [True, False],
                         ids=['enable_plugins', 'disable_plugins'])
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_context_fmha', 'enable_context_fmha_fp32_acc', 'disable_fmha'])
def test_llm_mpt_30b_1node_4gpus(mpt_example_root, llm_venv,
                                 llm_mpt_30b_model_root, llm_datasets_root,
                                 llm_rouge_root, cmodel_dir, engine_dir,
                                 data_type, use_plugins, context_fmha_type):
    "mpt 30b test on 4gpus"
    print("Converting MPT weights...")
    model_name = os.path.basename(llm_mpt_30b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=mpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_mpt_30b_model_root,
                               data_type=data_type,
                               gpus=4)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={4}",
        f"--max_input_len={1024}",
        f"--max_seq_len={1124}",
        f"--workers={4}",
    ]

    if use_plugins:
        if context_fmha_type == "enable_fmha":
            build_cmd.append("--context_fmha=enable")
        elif context_fmha_type == "disable_fmha":
            build_cmd.append("--context_fmha=disable")

        build_cmd.extend([
            f"--gpt_attention_plugin={data_type}", f"--gemm_plugin={data_type}"
        ])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--gemm_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    summary_cmd = generate_summary_cmd(mpt_example_root,
                                       hf_model_dir=llm_mpt_30b_model_root,
                                       engine_dir=engine_dir,
                                       data_type="fp16",
                                       tensorrt_llm_rouge1_threshold=17,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    if context_fmha_type == "enable_context_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_context_fmha', 'enable_context_fmha_fp32_acc', 'disable_fmha'])
def test_llm_mpt_7b_1node_1gpu(mpt_example_root, llm_venv,
                               llm_mpt_7b_model_root, llm_datasets_root,
                               llm_rouge_root, cmodel_dir, engine_dir,
                               context_fmha_type):
    "mpt-7b test on one gpu"
    ckpt_dir = convert_weights(llm_venv, mpt_example_root, cmodel_dir, "mpt-7b",
                               llm_mpt_7b_model_root)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={2}",
        f"--max_input_len={1024}",
        f"--max_beam_width={5}",
        "--gemm_plugin=float16",
    ]

    if context_fmha_type == "enable_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disable_fmha":
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    summary_cmd = generate_summary_cmd(mpt_example_root,
                                       hf_model_dir=llm_mpt_7b_model_root,
                                       engine_dir=engine_dir,
                                       data_type="fp16",
                                       tensorrt_llm_rouge1_threshold=20,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    if context_fmha_type == "enable_context_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


# transformers compatibility issues
# ImportError: cannot import name '_expand_mask' from 'transformers.models.bloom.modeling_bloom'
def test_llm_mpt_125m_summary(mpt_example_root, llm_venv,
                              llm_mpt_125m_model_root, llm_datasets_root,
                              llm_rouge_root, cmodel_dir, engine_dir,
                              update_transformers):
    "mpt-125m summary test"
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=mpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="mpt-125m",
                               model_path=llm_mpt_125m_model_root,
                               data_type="float32")

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        "--gpt_attention_plugin=float32",
        "--gemm_plugin=float32",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running summary...")
    summary_cmd = generate_summary_cmd(mpt_example_root,
                                       hf_model_dir=llm_mpt_125m_model_root,
                                       engine_dir=engine_dir,
                                       batch_size=1,
                                       data_type="fp32",
                                       tensorrt_llm_rouge1_threshold=10,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@skip_pre_ada
def test_llm_mpt_7b_fp8_summary(mpt_example_root, llm_mpt_7b_model_root,
                                llm_datasets_root, llm_rouge_root, llm_venv,
                                engine_dir, qcache_dir):
    "Build & Run mpt 7b with fp8."
    # Quantize HF mpt 7b checkpoint into FP8 format
    quantize_cmd = [
        f"{mpt_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_mpt_7b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--dtype=float16",
        "--qformat=fp8",
        "--kv_cache_dtype=fp8",
        f"--output_dir={qcache_dir}/quantized_fp8",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qcache_dir}/quantized_fp8/",
        f"--output_dir={engine_dir}",
        f"--max_input_len={1024}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run mpt-7b fp8...')
    summary_cmd = generate_summary_cmd(mpt_example_root,
                                       hf_model_dir=llm_mpt_7b_model_root,
                                       engine_dir=engine_dir,
                                       data_type="fp16",
                                       tensorrt_llm_rouge1_threshold=20,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


def test_llm_mpt_7b_awq_int4_summary(mpt_example_root, llm_mpt_7b_model_root,
                                     llm_datasets_root, llm_rouge_root,
                                     llm_venv, engine_dir, qcache_dir):
    "Build & Run mpt 7b with awq int4 gpus"
    # Quantize HF mpt-7b checkpoint into int4 format
    quantize_cmd = [
        f"{mpt_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_mpt_7b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--dtype=float16",
        "--qformat=int4_awq",
        "--calib_size=32",
        f"--output_dir={qcache_dir}/quantized_int4",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qcache_dir}/quantized_int4/",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={64}",
        f"--max_input_len={1024}",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run mpt-7b awq int4...')
    summary_cmd = generate_summary_cmd(mpt_example_root,
                                       hf_model_dir=llm_mpt_7b_model_root,
                                       engine_dir=engine_dir,
                                       data_type="fp16",
                                       tensorrt_llm_rouge1_threshold=20,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("data_type", ['int8', 'int4'])
def test_llm_mpt_7b_weight_only(mpt_example_root, llm_venv,
                                llm_mpt_7b_model_root, llm_datasets_root,
                                llm_rouge_root, cmodel_dir, engine_dir,
                                data_type):
    "mpt-7b test with weight only"
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=mpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="mpt-7b",
                               model_path=llm_mpt_7b_model_root,
                               weight_only_precision=data_type)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={64}",
        f"--max_input_len={1024}",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    # For weight-only int4, mpt-7b has bad accuracy while mpt-125m and
    # mpt-30b has comparable accuracy with FP16.
    summary_cmd = generate_summary_cmd(mpt_example_root,
                                       hf_model_dir=llm_mpt_7b_model_root,
                                       engine_dir=engine_dir,
                                       data_type="fp16",
                                       tensorrt_llm_rouge1_threshold=20,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


# transformers compatibility issues
# ImportError: cannot import name '_expand_mask' from 'transformers.models.bloom.modeling_bloom'
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
def test_llm_replit_code_v1_5_3b_1node_2gpus(mpt_example_root, llm_venv,
                                             llm_replit_code_v1_5_3b_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             cmodel_dir, engine_dir, num_beams,
                                             update_transformers):
    "replit code v1_5 3b test with 2gpus"
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=mpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="mpt_replit_code",
                               model_path=llm_replit_code_v1_5_3b_model_root,
                               data_type="bfloat16",
                               gpus=2)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={16}",
        f"--max_input_len={1024}",
        f"--max_beam_width={num_beams}",
        "--gemm_plugin=bfloat16",
        "--gpt_attention_plugin=bfloat16",
        "--workers=2",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    summary_cmd = generate_summary_cmd(
        mpt_example_root,
        hf_model_dir=llm_replit_code_v1_5_3b_model_root,
        engine_dir=engine_dir,
        data_type="fp16",
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=10,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)
