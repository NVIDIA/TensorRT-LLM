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

import pytest
from defs.common import (convert_weights, generate_summary_cmd, parse_output,
                         venv_check_call, venv_mpi_check_call,
                         venv_mpi_check_output)
from defs.conftest import (get_gpu_device_list, skip_fp8_pre_ada,
                           skip_nvlink_inactive)
from defs.trt_test_alternative import check_call

INPUT_TEXT = """
Write a Python function `find_max(words)` to solve the following problem:\nWrite a function that accepts a list of strings.\nThe list contains different words. Return the word with maximum number\nof unique characters. If multiple strings have maximum number of unique\ncharacters, return the one which comes first in lexicographical order.\nfind_max(["name", "of", "string"]) == "string"\nfind_max(["name", "enam", "game"]) == "enam"\nfind_max(["aaaaaaa", "bb" ,"cc"]) == ""aaaaaaa"
"""


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("enable_fp8", [True, False],
                         ids=["enable_fp8", "disable_fp8"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize(
    "use_gpt_attention_plugin", [True, False],
    ids=["enable_gpt_attention_plugin", "disable_gpt_attention_plugin"])
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_context_fmha', 'enable_context_fmha_fp32_acc', 'disabled'])
@pytest.mark.parametrize("use_weight_only_groupwise_quant_matmul_plugin",
                         [True, False],
                         ids=[
                             "enable_weight_only_groupwise_quant_matmul_plugin",
                             "disable_weight_only_groupwise_quant_matmul_plugin"
                         ])
@pytest.mark.parametrize(
    "fp8_context_fmha_xqa",
    ["enable_fp8_context_fmha_xqa", "disable_fp8_context_fmha_xqa"])
def test_llm_gptj_single_gpu_summary(
        gptj_example_root, llm_gptj_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, engine_dir, use_gemm_plugin,
        use_gpt_attention_plugin, context_fmha_type,
        use_weight_only_groupwise_quant_matmul_plugin, enable_fp8, num_beams,
        fp8_context_fmha_xqa):
    "Build & run gptj on single gpu."
    skip_fp8_pre_ada(use_fp8=enable_fp8)
    if "enable" in fp8_context_fmha_xqa and not enable_fp8:
        pytest.skip(
            "FP8 Context FMHA must be used together with the fp8 quantization workflow."
        )

    gpus = ["H20", "H100"]
    if all(x not in get_gpu_device_list()[0]
           for x in gpus) and fp8_context_fmha_xqa:
        pytest.skip("FP8 FMHA cannot be enabled on Pre-Hopper Arch.")

    print("Quantizing model...")
    qcache_dir = "/tmp/cache"
    ckpt_dir = f"{qcache_dir}/quantized_model_cache"
    quantize_cmd = [
        f"{gptj_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_gptj_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
        "--calib_size=16",
    ]
    if use_weight_only_groupwise_quant_matmul_plugin:
        quantize_cmd.append("--qformat=int4_awq")
    elif enable_fp8:
        quantize_cmd.append("--qformat=fp8")
        quantize_cmd.append("--kv_cache_dtype=fp8")

    if use_weight_only_groupwise_quant_matmul_plugin or enable_fp8:
        venv_check_call(llm_venv, quantize_cmd)
    else:
        ckpt_dir = convert_weights(llm_venv=llm_venv,
                                   example_root=gptj_example_root,
                                   cmodel_dir=qcache_dir,
                                   model="gptj",
                                   model_path=llm_gptj_model_root)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=4",
        "--max_input_len=1024",
        "--max_seq_len=1152",
        "--max_beam_width=5",
    ]
    if context_fmha_type == "enable_context_fmha":
        build_cmd.append("--context_fmha=enable")
    else:
        build_cmd.append("--context_fmha=disable")
    if use_gpt_attention_plugin:
        build_cmd.append("--gpt_attention_plugin=float16")
        build_cmd.append("--remove_input_padding=enable")
    else:
        build_cmd.append("--gpt_attention_plugin=disable")
        build_cmd.append("--remove_input_padding=disable")
        build_cmd.append("--paged_kv_cache=disable")
    if use_gemm_plugin:
        build_cmd.append("--gemm_plugin=float16")

    if "enable" in fp8_context_fmha_xqa:
        build_cmd.extend([
            "--use_fp8_context_fmha=enable", "--use_paged_context_fmha=enable"
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    summary_cmd = [
        f"{gptj_example_root}/../summarize.py", "--engine_dir", engine_dir,
        "--hf_model_dir", llm_gptj_model_root, "--batch_size", "1",
        "--test_trt_llm", "--tensorrt_llm_rouge1_threshold", "14",
        "--data_type", "fp16", "--check_accuracy", f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    if context_fmha_type == "enable_context_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
def test_llm_gptj_awq_gpu_summary_2gpus(gptj_example_root, llm_gptj_model_root,
                                        llm_datasets_root, llm_rouge_root,
                                        llm_venv, engine_dir, qcache_dir):
    print("Quantizing model...")
    ckpt_dir = f"{qcache_dir}/int4_awq_cache"

    quantize_cmd = [
        f"{gptj_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_gptj_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--output_dir={ckpt_dir}",
        "--qformat=int4_awq",
        "--dtype=float16",
        "--tp_size=2",
        "--calib_size=16",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--context_fmha=enable",
        "--remove_input_padding=enable",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--max_batch_size=8",
        "--max_input_len=2048",
        "--max_seq_len=2176",
        "--max_beam_width=5",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    summary_cmd = [
        f"{gptj_example_root}/../summarize.py", "--engine_dir", engine_dir,
        "--hf_model_dir", llm_gptj_model_root, "--batch_size", "1",
        "--test_trt_llm", "--tensorrt_llm_rouge1_threshold", "14",
        "--data_type", "fp16", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.parametrize("weight_only_precision",
                         ["int8_wo", "int4_awq", "int4_wo"])
def test_llm_gptj_int8_kv_cache_summary(gptj_example_root, llm_gptj_model_root,
                                        llm_datasets_root, llm_rouge_root,
                                        llm_venv, engine_dir,
                                        weight_only_precision, qcache_dir):
    "Build & run gptj int8 kv cache on 1 gpus."
    print("Quantizing model...")
    ckpt_dir = f"{qcache_dir}/int4_awq_cache"
    quantize_cmd = [
        f"{gptj_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_gptj_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--output_dir={ckpt_dir}",
        f"--qformat={weight_only_precision}",
        "--kv_cache_dtype=int8",
        "--dtype=float16",
        "--calib_size=16",
    ]

    venv_check_call(llm_venv, quantize_cmd)
    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--context_fmha=enable",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--max_batch_size=32",
        "--max_input_len=1919",
        "--max_seq_len=2047",
        "--remove_input_padding=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    summary_cmd = generate_summary_cmd(gptj_example_root,
                                       engine_dir=engine_dir,
                                       hf_model_dir=llm_gptj_model_root,
                                       data_type='fp16',
                                       tensorrt_llm_rouge1_threshold=16,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(4)
@skip_nvlink_inactive
def test_llm_gptj_4gpus_summary(gptj_example_root, llm_gptj_model_root,
                                llm_datasets_root, llm_rouge_root, llm_venv,
                                engine_dir, cmodel_dir):
    """
        Build & run gptj on 4 gpus.
        RCCA https://nvbugs/4569848
    """
    print("Quantizing model...")
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gptj_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gptj",
                               model_path=llm_gptj_model_root,
                               weight_only_precision="int8",
                               use_weight_only=True,
                               tp_size=4,
                               pp_size=1)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--context_fmha=enable",
        "--gemm_plugin=float16",
        "--max_batch_size=32",
        "--max_input_len=1919",
        "--max_seq_len=2047",
        "--max_beam_width=1",
        "--workers=4",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    summary_cmd = generate_summary_cmd(gptj_example_root,
                                       engine_dir=engine_dir,
                                       hf_model_dir=llm_gptj_model_root,
                                       data_type='fp16',
                                       tensorrt_llm_rouge1_threshold=19,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)

    run_cmd = [
        f"{gptj_example_root}/../run.py",
        f"--tokenizer_dir={llm_gptj_model_root}",
        f"--engine_dir={engine_dir}",
        "--max_output_len=256",
        "--input_text",
        INPUT_TEXT.strip(),
        INPUT_TEXT.strip(),
        INPUT_TEXT.strip(),
    ]

    output = venv_mpi_check_output(llm_venv,
                                   ["mpirun", "-n", "4", "--allow-run-as-root"],
                                   run_cmd,
                                   env=dict(FORCE_DETERMINISTIC="1", ))

    outputs = parse_output(output)

    assert all(outputs[0] == item
               for item in outputs), f"output is inconsistency: {output}"


def test_llm_gptj_fp8_manage_weights_summary(gptj_example_root,
                                             llm_gptj_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llm_venv, engine_dir):

    gpus = ["L40S", "H20", "H100"]
    if all(x not in get_gpu_device_list()[0] for x in gpus):
        pytest.skip("FP8 cannot be enabled on Pre-Ada Arch.")

    print("Quantizing model...")
    qcache_dir = "/tmp/cache"
    ckpt_dir = f"{qcache_dir}/quantized_model_cache"
    quantize_cmd = [
        f"{gptj_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_gptj_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
        "--calib_size=16",
        "--qformat=fp8",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", "--max_batch_size=4",
        "--max_input_len=1024", "--max_seq_len=1152", "--max_beam_width=5",
        "--fast_build"
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    summary_cmd = [
        f"{gptj_example_root}/../summarize.py", "--engine_dir", engine_dir,
        "--hf_model_dir", llm_gptj_model_root, "--batch_size", "1",
        "--test_trt_llm", "--tensorrt_llm_rouge1_threshold", "14",
        "--data_type", "fp16", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)
