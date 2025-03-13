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
"""Module test_baichuan test baichuan examples."""
import os

import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import skip_post_blackwell, skip_pre_ada
from defs.trt_test_alternative import check_call


@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("dtype", [
    "float16", "bfloat16",
    pytest.param("int8", marks=skip_post_blackwell),
    pytest.param("int4", marks=skip_post_blackwell)
])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root",
                         ["v1_7b", "v1_13b", "v2_7b", "v2_13b"],
                         indirect=True)
def test_llm_baichuan_single_gpu_summary(baichuan_example_root,
                                         llm_baichuan_model_version_and_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, engine_dir,
                                         use_attention_plugin, use_gemm_plugin,
                                         dtype):
    "Build & run baichuan on single gpu."
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    print(f"Building engines for baichuan_{baichuan_model_version} ...")

    if baichuan_model_version == "v1_7b" and dtype == "int4":
        pytest.skip(
            "Baichuan V1 7B performs not well when using int4 weight only.")

    convert_cmd = [
        f"{baichuan_example_root}/convert_checkpoint.py",
        "--model_version",
        baichuan_model_version,
        "--model_dir",
        baichuan_model_root,
        "--output_dir",
        f"{engine_dir}/ckpt",
    ]
    if dtype == "bfloat16":
        convert_cmd.extend(["--dtype", "bfloat16"])
    else:
        convert_cmd.extend(["--dtype", "float16"])
    if dtype in ("int8", "int4"):
        convert_cmd.extend(["--use_weight_only"])
    if dtype == "int4":
        convert_cmd.extend(["--weight_only_precision", "int4"])
    venv_check_call(llm_venv, convert_cmd)

    build_cmd = [
        "trtllm-build",
        "--checkpoint_dir",
        f"{engine_dir}/ckpt",
        "--max_seq_len=1074",
        "--output_dir",
        f"{engine_dir}/engines",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
    ]
    if use_attention_plugin:
        if dtype == "bfloat16":
            build_cmd.extend(["--gpt_attention_plugin", "bfloat16"])
        else:
            build_cmd.extend(["--gpt_attention_plugin", "float16"])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])

    if use_gemm_plugin:
        if dtype == "bfloat16":
            build_cmd.extend(["--gemm_plugin", "bfloat16"])
        else:
            build_cmd.extend(["--gemm_plugin", "float16"])
    else:
        build_cmd.extend(["--gemm_plugin", "disable"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{baichuan_example_root}/../run.py", "--max_output_len=50",
        "--engine_dir", f"{engine_dir}/engines", "--tokenizer_dir",
        baichuan_model_root
    ]
    venv_check_call(llm_venv, run_cmd)

    print(f"Run summary for baichuan_{baichuan_model_version}...")
    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", baichuan_model_root, "--engine_dir",
        f"{engine_dir}/engines", "--data_type", "fp16",
        "--tensorrt_llm_rouge1_threshold=18", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("dtype", [
    "float16", "bfloat16",
    pytest.param("int8", marks=skip_post_blackwell),
    pytest.param("int4", marks=skip_post_blackwell)
])
@pytest.mark.parametrize("parallel_build", [True, False],
                         ids=["parallel_build", "serial_build"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root",
                         ["v1_7b", "v1_13b", "v2_7b", "v2_13b"],
                         indirect=True)
def test_llm_baichuan_1node_2gpus(baichuan_example_root,
                                  llm_baichuan_model_version_and_root,
                                  llm_datasets_root, llm_rouge_root, llm_venv,
                                  engine_dir, use_attention_plugin,
                                  use_gemm_plugin, dtype, parallel_build):
    "Build & run baichuan on 2 gpus."
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    print(f"Building engines for baichuan_{baichuan_model_version} ...")

    if baichuan_model_version == "v1_7b" and dtype == "int4":
        pytest.skip(
            "Baichuan V1 7B performs not well when using int4 weight only.")

    convert_cmd = [
        f"{baichuan_example_root}/convert_checkpoint.py",
        "--model_version",
        baichuan_model_version,
        "--model_dir",
        baichuan_model_root,
        "--output_dir",
        f"{engine_dir}/ckpt",
        "--tp_size",
        "2",
        "--pp_size",
        "1",
    ]
    if dtype == "bfloat16":
        convert_cmd.extend(["--dtype", "bfloat16"])
    else:
        convert_cmd.extend(["--dtype", "float16"])
    if dtype in ("int8", "int4"):
        convert_cmd.extend(["--use_weight_only"])
    if dtype == "int4":
        convert_cmd.extend(["--weight_only_precision", "int4"])
    venv_check_call(llm_venv, convert_cmd)

    build_cmd = [
        "trtllm-build",
        "--checkpoint_dir",
        f"{engine_dir}/ckpt",
        "--max_seq_len=1074",
        "--output_dir",
        f"{engine_dir}/engines",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
    ]

    if use_attention_plugin:
        if dtype == "bfloat16":
            build_cmd.extend(["--gpt_attention_plugin", "bfloat16"])
        else:
            build_cmd.extend(["--gpt_attention_plugin", "float16"])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])
    if use_gemm_plugin:
        if dtype == "bfloat16":
            build_cmd.extend(["--gemm_plugin", "bfloat16"])
        else:
            build_cmd.extend(["--gemm_plugin", "float16"])
    else:
        build_cmd.extend(["--gemm_plugin", "disable"])

    if parallel_build:
        build_cmd.extend(["--workers", "2"])
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{baichuan_example_root}/../run.py", "--max_output_len=50",
        "--engine_dir", f"{engine_dir}/engines", "--tokenizer_dir",
        baichuan_model_root
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        run_cmd)

    print(f"Run summary for baichuan_{baichuan_model_version}...")
    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", baichuan_model_root, "--engine_dir",
        f"{engine_dir}/engines", "--data_type", "fp16",
        "--tensorrt_llm_rouge1_threshold=18", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    # For 7b models + int8/int4 weight only + TP > 1 cases,
    # we may need more samples to get stable rouge scores
    if "7b" in baichuan_model_version and dtype in ("int8", "int4"):
        summary_cmd.extend(["--max_ite", "40"])
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("per_token_channel", [True, False],
                         ids=["enable_ptpc", "disable_ptpc"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root",
                         ["v1_7b", "v1_13b", "v2_7b", "v2_13b"],
                         indirect=True)
def test_llm_baichuan_smoothquant_summary(baichuan_example_root,
                                          llm_baichuan_model_version_and_root,
                                          llm_datasets_root, llm_rouge_root,
                                          per_token_channel, llm_venv,
                                          engine_dir):
    "Build & run baichuan with smoothquant on single gpu."
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    model_name = f"baichuan_{baichuan_model_version}-smoothquant-0.8"

    print(f"Converting model for {model_name} ...")
    convert_cmd = [
        f"{baichuan_example_root}/convert_checkpoint.py",
        "--model_version",
        baichuan_model_version,
        "--model_dir",
        baichuan_model_root,
        "--smoothquant",
        "0.8",
        "--dtype",
        "float16",
        "--output_dir",
        f"{engine_dir}/ckpt",
        f"--calib_dataset={llm_datasets_root}/ccdv/cnn_dailymail",
    ]
    if per_token_channel:
        convert_cmd.extend(["--per_token", "--per_channel"])
    venv_check_call(llm_venv, convert_cmd)

    print(f"Building engines for {model_name} ...")
    build_cmd = [
        "trtllm-build",
        "--checkpoint_dir",
        f"{engine_dir}/ckpt",
        "--max_seq_len=1074",
        f"--max_batch_size={8}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
        "--output_dir",
        f"{engine_dir}/engines",
        "--gpt_attention_plugin",
        "float16",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{baichuan_example_root}/../run.py", "--max_output_len=50",
        "--engine_dir", f"{engine_dir}/engines", "--tokenizer_dir",
        baichuan_model_root
    ]
    venv_check_call(llm_venv, run_cmd)

    print(f"Run summary for {model_name} ...")

    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        baichuan_model_root,
        "--engine_dir",
        f"{engine_dir}/engines",
        "--data_type",
        "fp16",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
        "--max_ite",
        "100",
        "--batch_size",
        "8",
    ]

    if per_token_channel:
        summary_cmd.extend(
            ["--tensorrt_llm_rouge1_threshold=28", "--check_accuracy"])

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("int8_kv_cache", [True, False],
                         ids=["enable_int8_kv_cache", "disable_int8_kv_cache"])
@pytest.mark.parametrize("per_token_channel", [True, False],
                         ids=["enable_ptpc", "disable_ptpc"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root",
                         ["v1_7b", "v1_13b", "v2_7b", "v2_13b"],
                         indirect=True)
def test_llm_baichuan_smoothquant_1node_2gpus_summary(
        baichuan_example_root, llm_baichuan_model_version_and_root,
        llm_datasets_root, llm_rouge_root, per_token_channel, llm_venv,
        engine_dir, cmodel_dir, int8_kv_cache):
    "Build & run baichuan with smoothquant on 2 gpus."
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    model_name = f"baichuan_{baichuan_model_version}-smoothquant-0.8-tp2"

    print(f"Converting model for {model_name} ...")
    ckpt_dir = convert_weights(
        llm_venv,
        baichuan_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=baichuan_model_root,
        data_type="float16",
        smoothquant=0.8,
        tp_size=2,
        pp_size=1,
        per_channel=per_token_channel,
        per_token=per_token_channel,
        int8_kv_cache=int8_kv_cache,
        calib_dataset=f"{llm_datasets_root}/ccdv/cnn_dailymail")

    print(f"Building engines for {model_name} ...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_seq_len=1074",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
        "--gpt_attention_plugin=float16",
        "--workers=2",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{baichuan_example_root}/../run.py",
        f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={baichuan_model_root}",
        "--max_output_len=50",
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        run_cmd)

    print(f"Run summary for baichuan_{baichuan_model_version}...")
    summary_cmd = generate_summary_cmd(baichuan_example_root,
                                       engine_dir=engine_dir,
                                       data_type='fp16',
                                       hf_model_dir=baichuan_model_root,
                                       tensorrt_llm_rouge1_threshold=19,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root",
                         ["v1_7b", "v1_13b", "v2_7b", "v2_13b"],
                         indirect=True)
def test_llm_baichuan_int8_kv_single_gpu_summary(
        baichuan_example_root, llm_baichuan_model_version_and_root,
        llm_datasets_root, llm_rouge_root, llm_venv, engine_dir, cmodel_dir,
        use_attention_plugin, use_gemm_plugin):
    "Baichuan with int8 kv test on single gpu"
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    model_name = f"baichuan_{baichuan_model_version}_kv_cache"

    print(f"Converting model for {model_name} ...")
    quantize_cmd = [
        f"{baichuan_example_root}/../quantization/quantize.py",
        f"--model_dir={baichuan_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--dtype=float16",
        "--kv_cache_dtype=int8",
        f"--output_dir={cmodel_dir}",
        "--calib_size=512",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    print(f"Building engines for {model_name} ...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={cmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
    ]
    if use_attention_plugin:
        build_cmd.extend(["--gpt_attention_plugin", "float16"])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])
    if use_gemm_plugin:
        build_cmd.extend(["--gemm_plugin", "float16"])
    else:
        build_cmd.extend(["--gemm_plugin", "disable"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{baichuan_example_root}/../run.py", "--max_output_len=50",
        "--engine_dir", engine_dir, "--tokenizer_dir", baichuan_model_root
    ]

    venv_check_call(llm_venv, run_cmd)

    print(f"Run summary for {model_name}...")
    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", baichuan_model_root, "--engine_dir", engine_dir,
        "--data_type", "fp16", "--tensorrt_llm_rouge1_threshold=19",
        "--check_accuracy", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@skip_pre_ada
@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize(
    "fp8_context_fmha", ["enable_fp8_context_fmha", "disable_fp8_context_fmha"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root",
                         ["v1_7b", "v1_13b", "v2_7b", "v2_13b"],
                         indirect=True)
def test_llm_baichuan_single_gpu_fp8_summary(
        baichuan_example_root, llm_baichuan_model_version_and_root,
        llm_datasets_root, llm_rouge_root, llm_venv, engine_dir, qcache_dir,
        fp8_context_fmha):
    """
        RCCA https://nvbugs/4348560
        RCCA https://nvbugs/4538035
    """
    baichuan_model_root = llm_baichuan_model_version_and_root[1]

    # Quantize HF baichuan checkpoint into FP8 format
    quantize_cmd = [
        f"{baichuan_example_root}/../quantization/quantize.py",
        f"--model_dir={baichuan_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--dtype=float16",
        "--qformat=fp8",
        "--kv_cache_dtype=fp8",
        f"--output_dir={qcache_dir}/quantized_fp8",
        "--calib_size=256",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qcache_dir}/quantized_fp8",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
    ]

    if "enable" in fp8_context_fmha:
        build_cmd.extend([
            "--use_fp8_context_fmha=enable", "--use_paged_context_fmha=enable",
            "--paged_kv_cache=enable"
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        f"{baichuan_model_root}",
        "--data_type",
        "fp16",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("group_size", [64, 128], ids=["gs64", "gs128"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root",
                         ["v1_7b", "v1_13b", "v2_7b", "v2_13b"],
                         indirect=True)
def test_llm_baichuan_single_gpu_int4_awq_summary(
        baichuan_example_root, llm_baichuan_model_version_and_root,
        llm_datasets_root, llm_rouge_root, group_size, llm_venv, engine_dir,
        qcache_dir):
    "Build & Run Baichuan with INT4 AWQ"
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    model_name = f"baichuan_{baichuan_model_version}_int4_awq_gs{group_size}"

    # Quantize HF baichuan checkpoint into INT4 AWQ format
    quantize_cmd = [
        f"{baichuan_example_root}/../quantization/quantize.py",
        f"--model_dir={baichuan_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--dtype=float16",
        "--qformat=int4_awq",
        f"--awq_block_size={group_size}",
        f"--output_dir={qcache_dir}/{model_name}",
        "--calib_size=32",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    print(f"Building engines for {model_name} ...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qcache_dir}/{model_name}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{baichuan_model_root}", "--data_type", "fp16",
        f"--engine_dir={engine_dir}", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_host_memory(480000)
@pytest.mark.parametrize("group_size", [64, 128], ids=["gs64", "gs128"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root",
                         ["v1_7b", "v1_13b", "v2_7b", "v2_13b"],
                         indirect=True)
def test_llm_baichuan_int4_awq_1node_2gpus(baichuan_example_root,
                                           llm_baichuan_model_version_and_root,
                                           llm_datasets_root, llm_rouge_root,
                                           group_size, llm_venv, engine_dir,
                                           qcache_dir):
    "Build & Run Baichuan with INT4 AWQ and TP = 2"
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    model_name = f"baichuan_{baichuan_model_version}_int4_awq_gs{group_size}"

    # Quantize HF baichuan checkpoint into INT4 AWQ format
    quantize_cmd = [
        f"{baichuan_example_root}/../quantization/quantize.py",
        f"--model_dir={baichuan_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--dtype=float16",
        "--qformat=int4_awq",
        f"--awq_block_size={group_size}",
        f"--output_dir={qcache_dir}/{model_name}",
        "--calib_size=32",
        "--tp_size=2",
        "--pp_size=1",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    print(f"Building engines for {model_name} ...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qcache_dir}/{model_name}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{baichuan_model_root}", "--data_type", "fp16",
        f"--engine_dir={engine_dir}", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("group_size", [64], ids=["gs64"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root", ["v2_13b"],
                         indirect=True)
def test_llm_baichuan_single_gpu_int4_gptq_summary(
        baichuan_example_root, llm_baichuan_model_version_and_root,
        llm_datasets_root, llm_rouge_root, group_size, llm_venv, engine_dir):
    "Build & Run Baichuan with INT4 AWQ"
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    model_name = f"baichuan_{baichuan_model_version}_int4_gptq_gs{group_size}"

    print(f"Building engines for {model_name} ...")
    baichuan_gptq_safetensors_path = os.path.join(
        baichuan_model_root, "..", "int4-quantized-gptq-awq",
        "baichuan-2-13b-4bit-gs64.safetensors")
    convert_cmd = [
        f"{baichuan_example_root}/convert_checkpoint.py",
        f"--model_version={baichuan_model_version}",
        f"--model_dir={baichuan_model_root}",
        f"--quant_ckpt_path={baichuan_gptq_safetensors_path}",
        "--dtype=float16",
        "--use_weight_only",
        "--weight_only_precision=int4_gptq",
        f"--group_size={group_size}",
        f"--output_dir={engine_dir}/ckpt",
    ]
    venv_check_call(llm_venv, convert_cmd)

    print(f"Building engines for {model_name} ...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={engine_dir}/ckpt",
        "--max_seq_len=1074",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
        f"--output_dir={engine_dir}/engines",
        "--gpt_attention_plugin=float16",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{baichuan_example_root}/../run.py",
        "--max_output_len=50",
        f"--engine_dir={engine_dir}/engines",
        f"--tokenizer_dir={baichuan_model_root}",
    ]
    venv_check_call(llm_venv, run_cmd)

    print(f"Run summary for baichuan_{baichuan_model_version}...")
    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py", "--test_trt_llm",
        f"--hf_model_dir={baichuan_model_root}",
        f"--engine_dir={engine_dir}/engines", "--data_type=fp16",
        "--tensorrt_llm_rouge1_threshold=26", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_host_memory(480000)
@pytest.mark.parametrize("group_size", [64], ids=["gs64"])
@pytest.mark.parametrize("llm_baichuan_model_version_and_root", ["v2_13b"],
                         indirect=True)
def test_llm_baichuan_int4_gptq_1node_2gpus(baichuan_example_root,
                                            llm_baichuan_model_version_and_root,
                                            llm_datasets_root, llm_rouge_root,
                                            group_size, llm_venv, engine_dir):
    "Build & Run Baichuan with INT4 AWQ and TP = 2"
    baichuan_model_version = llm_baichuan_model_version_and_root[0]
    baichuan_model_root = llm_baichuan_model_version_and_root[1]
    model_name = f"baichuan_{baichuan_model_version}_int4_gptq_gs{group_size}"

    print(f"Building engines for {model_name} ...")
    baichuan_gptq_safetensors_path = os.path.join(
        baichuan_model_root, "..", "int4-quantized-gptq-awq",
        "baichuan-2-13b-4bit-gs64.safetensors")
    convert_cmd = [
        f"{baichuan_example_root}/convert_checkpoint.py",
        f"--model_version={baichuan_model_version}",
        f"--model_dir={baichuan_model_root}",
        f"--quant_ckpt_path={baichuan_gptq_safetensors_path}",
        "--dtype=float16",
        "--use_weight_only",
        "--weight_only_precision=int4_gptq",
        f"--group_size={group_size}",
        f"--output_dir={engine_dir}/ckpt",
        "--tp_size=2",
        "--pp_size=1",
    ]
    venv_check_call(llm_venv, convert_cmd)

    print(f"Building engines for {model_name} ...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={engine_dir}/ckpt",
        "--max_seq_len=1074",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_num_tokens={1024}",
        f"--output_dir={engine_dir}/engines",
        "--gpt_attention_plugin=float16",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{baichuan_example_root}/../run.py",
        "--max_output_len=50",
        f"--engine_dir={engine_dir}/engines",
        f"--tokenizer_dir={baichuan_model_root}",
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        run_cmd)

    print(f"Run summary for baichuan_{baichuan_model_version}...")
    summary_cmd = [
        f"{baichuan_example_root}/../summarize.py", "--test_trt_llm",
        f"--hf_model_dir={baichuan_model_root}",
        f"--engine_dir={engine_dir}/engines", "--data_type=fp16",
        "--tensorrt_llm_rouge1_threshold=26", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)
