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
import copy
import csv
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy

import defs.ci_profiler
import pytest
from defs.common import (convert_weights, generate_summary_cmd,
                         get_cpp_benchmark, get_trt_llm_lib_dir, parse_output,
                         quantize_data, similar,
                         test_llm_torch_multi_lora_support,
                         test_multi_lora_support, venv_check_call,
                         venv_check_output, venv_mpi_check_call)
# yapf: disable
from defs.conftest import (get_device_count, get_device_memory,
                           get_host_total_memory, get_sm_version,
                           skip_fp8_pre_ada, skip_no_nvls, skip_post_blackwell,
                           skip_pre_ada, skip_pre_blackwell)
# yapf: enable
from defs.trt_test_alternative import check_call, exists

# skip trt flow cases on post-Blackwell-Ultra
# if get_sm_version() >= 103:
#     pytest.skip(
#         "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
#         allow_module_level=True)

INPUT_TEXT_1 = "After Washington had returned to Williamsburg, " + \
               "Dinwiddie ordered him to lead a larger force to assist Trent in his work. " + \
               "While en route, Washington learned of Trent's retreat. " + \
               "Since Tanaghrisson had promised support to the British, " + \
               "Washington continued toward Fort Duquesne and met with the Mingo leader. " + \
               "Learning of a French scouting party in the area, Washington, " + \
               "with Tanaghrisson and his party, surprised the Canadians on May 28 " + \
               "in what became known as the Battle of Jumonville Glen. " + \
               "They killed many of the Canadians, including their commanding officer, " + \
               "Joseph Coulon de Jumonville, whose head was reportedly split open by " + \
               "Tanaghrisson with a tomahawk. The historian Fred Anderson suggests that " + \
               "Tanaghrisson was acting to gain the support of the British and regain " + \
               "authority over his own people. They had been inclined to support the French, " + \
               "with whom they had long trading relationships. One of Tanaghrisson's men told " + \
               "Contrecoeur that Jumonville had been killed by British musket fire. " + \
               "Question: Upon learning of a French scounting party in the area, " + \
               "what did Washington do? Answer:"

INPUT_TEXT_2 = "Born in north-east France, Soyer trained as a"


@pytest.mark.parametrize("num_beams", [5, 7],
                         ids=["num_beams_4", "num_beams_7"])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_early_finish_beams(llama_example_root, llm_venv, llama_model_root,
                            engine_dir, cmodel_dir, num_beams):
    """ Test the correctness of beam search + streaming versus the outputs of
    non-streaming beam search. Both use the cpp runtime.
    This test is aimed specifically at checking if shorter finished beams are being put
    into the outputs correctly."""

    dtype = 'float16'
    output_len = 10
    input_text = ["want to", "The time is", "Soyer was"]
    model_name = os.path.basename(llama_model_root)

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llama_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--max_beam_width={num_beams}",
        "--context_fmha=enable",
        "--use_paged_context_fmha=enable",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")

    streaming_command = [
        f"{llama_example_root}/../run.py", f"--max_output_len={output_len}",
        f"--engine_dir={engine_dir}", f"--tokenizer_dir={llama_model_root}",
        f"--streaming", f"--streaming_interval=1", f"--num_beams={num_beams}",
        f"--input_text", *input_text
    ]
    streaming_outputs = venv_check_output(llm_venv, streaming_command)

    joined_nonstreamed_outputs = ""
    for length_iterator in range(1, output_len + 1):
        command = [
            f"{llama_example_root}/../run.py",
            f"--max_output_len={length_iterator}", f"--engine_dir={engine_dir}",
            f"--tokenizer_dir={llama_model_root}", f"--num_beams={num_beams}",
            f"--input_text", *input_text
        ]

        non_streaming_output = venv_check_output(llm_venv, command)
        joined_nonstreamed_outputs += "Output from command" + str(
            command) + "\n" + non_streaming_output

    def parse_output(text: str) -> list[str]:
        results = []
        while True:
            match = re.search(
                r"Output \[Text \d+ Beam \d+\]: \"([^\"]*)\"\r?\n", text)
            if match is None:
                break
            _, end = match.span()
            results.append(match.group(1))
            text = text[end:]
        return results

    print("STREAMING OUTPUT HERE\n\n\n",
          streaming_outputs,
          "\n\n\n",
          sep="----")
    print("NON-STREAMING OUTPUT HERE\n\n\n",
          joined_nonstreamed_outputs,
          "\n\n\n",
          sep="----")
    parsed_streamed_outputs = parse_output(streaming_outputs)
    parsed_nonstreamed_outputs = parse_output(joined_nonstreamed_outputs)

    def ordered_subset(s1, s2):
        """
        Use this to check if the streamed outputs are an ordered subset of nonstreamed
        Streaming can sometimes skip outputs
        """
        s2 = iter(s2)
        try:
            for c in s1:
                while next(s2) != c:
                    pass
            else:
                return True
        except StopIteration:
            return False

    streaming_is_subset = ordered_subset(parsed_streamed_outputs,
                                         parsed_nonstreamed_outputs)
    print("streaming_is_subset ", streaming_is_subset)
    assert streaming_is_subset
    is_equal = (parsed_streamed_outputs == parsed_nonstreamed_outputs)
    print("is_equal", is_equal)
    if not is_equal:
        print("Differences:")
        for streamed, nonstreamed in zip(parsed_streamed_outputs,
                                         parsed_nonstreamed_outputs):
            if (streamed != nonstreamed):
                print("Streamed:", streamed)
                print("Nonstreamed:", nonstreamed)

    assert is_equal


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("use_weight_only_groupwise_quant_matmul_plugin",
                         [True, False],
                         ids=[
                             "enable_weight_only_groupwise_quant_matmul_plugin",
                             "disable_weight_only_groupwise_quant_matmul_plugin"
                         ])
@pytest.mark.parametrize("run_type", ['inference', 'summarization'])
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_v1_1gpu(use_weight_only_groupwise_quant_matmul_plugin,
                           run_type, data_type, llama_example_root,
                           llama_model_root, llm_datasets_root, llm_rouge_root,
                           llm_venv, cmodel_dir, engine_dir, num_beams):
    if num_beams > 2 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    model_name = 'llama_v1-{}'.format(run_type)

    print("Build engines...")
    if not use_weight_only_groupwise_quant_matmul_plugin:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    data_type=data_type)

    else:
        model_name = 'llama_v1-int4_gptq-{}'.format(run_type)

        llama_gptq_safetensors_root = os.path.join(
            llama_model_root, "../..", "int4-quantized-gptq-awq",
            "llama-7b-4bit-gs128.safetensors")
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    data_type=data_type,
                                    quant_ckpt_path=llama_gptq_safetensors_root)

    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}", "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}"
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if run_type == "inference":
        print("Run inference...")
        venv_check_call(llm_venv, [
            f"{llama_example_root}/../run.py",
            "--max_output_len=50",
            f"--tokenizer_dir={llama_model_root}",
            f"--engine_dir={engine_dir}",
            f"--num_beams={num_beams}",
        ])
    elif run_type == "summarization":
        print("Run summarize...")
        summary_cmd = [
            f"{llama_example_root}/../summarize.py", "--test_trt_llm",
            "--hf_model_dir", f"{llama_model_root}", "--data_type=fp16",
            f"--engine_dir={engine_dir}", "--check_accuracy",
            f"--num_beams={num_beams}", f"--dataset_dir={llm_datasets_root}",
            f"--rouge_dir={llm_rouge_root}"
        ]
        venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_v1_manage_weights_1gpu_summarize(llama_example_root,
                                                    llama_model_root,
                                                    llm_datasets_root,
                                                    llm_rouge_root, llm_venv,
                                                    cmodel_dir, engine_dir):
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model="llama_v1-float16",
                                model_path=llama_model_root,
                                data_type="float16")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin=float16",
        f"--gemm_plugin=disable",
        "--remove_input_padding=enable",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{llama_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llama_model_root}", "--data_type=fp16",
        f"--engine_dir={engine_dir}", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@skip_pre_blackwell
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("fp4_type", ["plugin", "ootb", "disable"],
                         ids=["fp4_plugin", "fp4_ootb", "disable_fp4"])
@pytest.mark.parametrize("fuse_fp4_quant", ["enable", "disable"],
                         ids=["enable_fused_quant", "disable_fused_quant"])
@pytest.mark.parametrize(
    "norm_quant_fusion", ["enable", "disable"],
    ids=["enable_norm_quant_fusion", "disable_norm_quant_fusion"])
@pytest.mark.parametrize(
    "llama_model_root",
    ['llama-v3-8b-instruct-hf', 'llama-3.1-8b', 'llama-3.1-70b-instruct'],
    indirect=True)
def test_llm_llama_1gpu_fp4(
    mmlu_dataset_root,
    data_type,
    fp4_type,
    fuse_fp4_quant,
    norm_quant_fusion,
    llama_example_root,
    llama_model_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    qcache_dir_without_install_package,
    llm_datasets_root,
):
    model_name = os.path.basename(llama_model_root)
    if fp4_type != "disable":
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype=data_type,
            qformat="nvfp4",
            kv_cache_dtype="fp8",
            quantize_dir=qcache_dir_without_install_package)
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    data_type=data_type)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", "--max_batch_size=32"
    ]
    if fp4_type != "disable":
        build_cmd.extend([
            "--gemm_plugin=disable"
            if fp4_type == "ootb" else "--gemm_plugin=nvfp4"
        ])
    if fp4_type == "plugin" or fuse_fp4_quant == "enable":
        build_cmd.extend([
            "--use_paged_context_fmha=enable", "--use_fp8_context_fmha=enable"
        ])
    if fuse_fp4_quant == "enable":
        build_cmd.extend(["--fuse_fp4_quant=enable"])
    if norm_quant_fusion == 'enable':
        build_cmd.extend(["--norm_quant_fusion=enable"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run MMLU test")
    accuracy_map = {
        'llama-v3-8b-instruct-hf': 61.5,
        'Meta-Llama-3.1-8B': 61.0,
        'Meta-Llama-3.1-70B-Instruct': 75
    }
    acc_thres = accuracy_map[model_name]
    mmlu_cmd = [
        "trtllm-eval", f"--model={engine_dir}",
        f"--tokenizer={llama_model_root}", "--backend=tensorrt", "mmlu",
        f"--dataset_path={mmlu_dataset_root}", "--check_accuracy",
        f"--accuracy_threshold={acc_thres}"
    ]
    check_call(" ".join(mmlu_cmd), shell=True, env=llm_venv._new_env)


@skip_pre_blackwell
@pytest.mark.parametrize("fp4_type", ["plugin", "ootb", "disable"],
                         ids=["fp4_plugin", "fp4_ootb", "disable_fp4"])
@pytest.mark.parametrize(
    "llama_model_root",
    ['llama-v3-8b-instruct-hf', 'llama-3.1-8b', 'llama-3.1-70b-instruct'],
    indirect=True)
def test_llm_llama_1gpu_fp4_model_config(
    fp4_type,
    llama_example_root,
    llama_model_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    qcache_dir_without_install_package,
    llm_datasets_root,
):
    model_name = os.path.basename(llama_model_root)
    if fp4_type != "disable":
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="nvfp4",
            kv_cache_dtype="fp8",
            quantize_dir=qcache_dir_without_install_package)
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    data_type="float16")

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--model_config={model_dir}/config.json",
        f"--output_dir={engine_dir}", "--max_batch_size=32"
    ]
    if fp4_type != "disable":
        build_cmd.extend([
            "--gemm_plugin=disable"
            if fp4_type == "ootb" else "--gemm_plugin=nvfp4"
        ])
    if fp4_type == "plugin":
        build_cmd.extend([
            "--use_paged_context_fmha=enable", "--use_fp8_context_fmha=enable"
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)


@skip_pre_blackwell
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("fp4_type", ["plugin", "ootb", "disable"],
                         ids=["fp4_plugin", "fp4_ootb", "disable_fp4"])
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-70b-instruct'],
                         indirect=True)
def test_llm_llama_2gpu_fp4(mmlu_dataset_root, fp4_type, llama_example_root,
                            llama_model_root, llm_venv, engine_dir,
                            qcache_dir_without_install_package,
                            llm_datasets_root):
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="float16",
        qformat="nvfp4",
        tp_size=2,
        quantize_dir=qcache_dir_without_install_package)

    print("Build engines...")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=32",
    ]
    if fp4_type != "disable":
        build_cmd.extend([
            "--gemm_plugin=disable"
            if fp4_type == "ootb" else "--gemm_plugin=nvfp4"
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run MMLU test")
    acc_thres = 75
    mmlu_cmd = [
        "trtllm-eval", f"--model={engine_dir}",
        f"--tokenizer={llama_model_root}", "--backend=tensorrt", "mmlu",
        f"--dataset_path={mmlu_dataset_root}", "--check_accuracy",
        f"--accuracy_threshold={acc_thres}"
    ]
    check_call(" ".join(mmlu_cmd), shell=True, env=llm_venv._new_env)


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("fp4_type", ["plugin", "ootb", "disable"],
                         ids=["fp4_plugin", "fp4_ootb", "disable_fp4"])
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-405b'], indirect=True)
def test_llm_llama_8gpu_fp4(mmlu_dataset_root, fp4_type, llama_example_root,
                            llama_model_root, llm_venv, engine_dir,
                            qcache_dir_without_install_package,
                            llm_datasets_root, upgrade_transformers):
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="float16",
        qformat="nvfp4",
        tp_size=8,
        quantize_dir=qcache_dir_without_install_package)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", "--max_batch_size=32", "--workers=4"
    ]
    if fp4_type != "disable":
        build_cmd.extend([
            "--gemm_plugin=disable"
            if fp4_type == "ootb" else "--gemm_plugin=nvfp4"
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run MMLU test")
    acc_thres = 75
    mmlu_cmd = [
        "trtllm-eval", f"--model={engine_dir}",
        f"--tokenizer={llama_model_root}", "--backend=tensorrt", "mmlu",
        f"--dataset_path={mmlu_dataset_root}", "--check_accuracy",
        f"--accuracy_threshold={acc_thres}"
    ]
    check_call(" ".join(mmlu_cmd), shell=True, env=llm_venv._new_env)


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("run_type", ['inference', 'summarization'])
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("fp8_cache", [True, False],
                         ids=["enable_fp8", "disable_fp8"])
@pytest.mark.parametrize("llama_model_root", [
    'llama-v2-7b-hf', 'llama-v3-8b-instruct-hf', 'llama-3.1-8b-instruct-hf-fp8'
],
                         indirect=True)
def test_llm_llama_1gpu(run_type, data_type, fp8_cache, llama_example_root,
                        llama_model_root, llm_datasets_root, llm_rouge_root,
                        llm_venv, cmodel_dir, engine_dir,
                        qcache_dir_without_install_package, num_beams):
    if num_beams > 2 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    use_fp8 = fp8_cache if "fp8" not in llama_model_root.lower() else True
    skip_fp8_pre_ada(use_fp8=use_fp8)

    model_name = os.path.basename(llama_model_root)

    if llama_model_root.endswith('Llama-3.1-8B-Instruct-FP8'):
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model="llama_v3_hf_fp8",
                                    model_path=llama_model_root,
                                    fp8_kv_cache=fp8_cache,
                                    data_type=data_type)
    elif fp8_cache:
        # Quantize HF llama checkpoint into FP8 format
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype=data_type,
            qformat="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512,
            kv_cache_dtype="fp8")
    else:
        model_dir = convert_weights(
            llm_venv=llm_venv,
            example_root=llama_example_root,
            cmodel_dir=cmodel_dir,
            model=model_name,
            model_path=llama_model_root,
            data_type=data_type,
            enable_fp8=fp8_cache,
            fp8_kv_cache=fp8_cache,
            quant_ckpt_path=
            f"{qcache_dir_without_install_package}/quantized_fp8/llama_tp1_rank0.npz"
            if fp8_cache else None)

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
            f"--tokenizer_dir={llama_model_root}",
            f"--engine_dir={engine_dir}",
            f"--num_beams={num_beams}",
        ])
    elif run_type == "summarization":
        print("Run summarize...")
        tensorrt_llm_rouge1_threshold = {
            1: 14,
            2: 19,
            4: 19,
        }[num_beams]

        summary_cmd = generate_summary_cmd(
            llama_example_root,
            hf_model_dir=llama_model_root,
            data_type="fp16",
            engine_dir=engine_dir,
            tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
            num_beams=num_beams,
            dataset_dir=llm_datasets_root,
            rouge_dir=llm_rouge_root)

        venv_check_call(llm_venv, summary_cmd)


@skip_pre_ada
@pytest.mark.parametrize("data_type", ['bfloat16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_1gpu_fp8_kv_cache(
    data_type,
    llama_example_root,
    llama_model_root,
    llm_datasets_root,
    llm_rouge_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    qcache_dir_without_install_package,
):
    # Quantize HF llama checkpoint into FP8 format
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
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
        "--use_paged_context_fmha=enable",
        "--use_fp8_context_fmha=enable",
        "--max_beam_width=1",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    with open(f"{engine_dir}/config.json") as f:
        engine_config = json.load(f)

    assert engine_config["build_config"]["plugin_config"][
        "use_fp8_context_fmha"] == True
    assert engine_config["pretrained_config"]["quantization"][
        "kv_cache_quant_algo"] == "FP8"


@pytest.mark.parametrize("use_weight_sparsity", [True],
                         ids=["enable_weight_sparsity"])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_1gpu_sparsity(llama_example_root, llama_model_root,
                                    llama_v2_tokenizer_model_root, llm_venv,
                                    cmodel_dir, engine_dir,
                                    use_weight_sparsity):
    model_name = 'llama_v2'
    data_type = 'float16'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root,
                                data_type=data_type)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", "--log_level=verbose"
    ]
    if use_weight_sparsity:
        build_cmd.extend(["--weight_sparsity"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference...")
    venv_check_call(llm_venv, [
        f"{llama_example_root}/../run.py", "--max_output_len=50",
        f"--tokenizer_dir={llama_v2_tokenizer_model_root}",
        f"--engine_dir={engine_dir}", f"--num_beams=1"
    ])


@skip_post_blackwell
@pytest.mark.parametrize("num_beams", [1],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v3-8b-instruct-hf'],
                         indirect=True)
def test_llm_llama_v3_int8_gptq_1gpu_summary(data_type, llama_example_root,
                                             llama_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llm_venv, cmodel_dir, engine_dir,
                                             num_beams):
    if num_beams > 2 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    model_name = 'llama_v3-int8_gptq'

    llama_gptq_safetensors_root = os.path.join(
        llama_model_root, "../..", "int8-quantized-gptq",
        "llama-3-8b-8bit-gs64-gptq.safetensors")
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root,
                                data_type=data_type,
                                quant_ckpt_path=llama_gptq_safetensors_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}", "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}"
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{llama_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llama_model_root}", "--data_type=fp16",
        f"--engine_dir={engine_dir}", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=24", f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("num_beams", [1],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_llm_llama_4gpu_pp4(data_type, llama_example_root, llama_model_root,
                            llm_datasets_root, llm_rouge_root, llm_venv,
                            cmodel_dir, engine_dir, num_beams):
    model_name = os.path.basename(llama_model_root)

    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llama_model_root,
        data_type=data_type,
        tp_size=1,
        pp_size=4,
    )

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gemm_plugin={data_type}",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    tensorrt_llm_rouge1_threshold = {
        1: 12,
    }[num_beams]

    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llama_model_root,
        data_type="fp16",
        engine_dir=engine_dir,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        num_beams=num_beams,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['bfloat16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_fp8_2gpu_pp2(
        data_type, llama_example_root, llama_model_root,
        llama_v2_tokenizer_model_root, llm_datasets_root, llm_rouge_root,
        llm_venv, engine_dir, qcache_dir_without_install_package, num_beams):
    if num_beams > 2 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    # Quantize HF llama checkpoint into FP8 format
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
        tp_size=1,
        pp_size=2,
        kv_cache_dtype="fp8",
        calib_size=64)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}",
        "--use_paged_context_fmha=disable",
        "--use_fp8_context_fmha=disable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    tensorrt_llm_rouge1_threshold = {
        1: 13,
        2: 19,
        4: 19,
    }[num_beams]

    summary_cmd = [
        f"{llama_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llama_v2_tokenizer_model_root}",
        "--data_type=fp16", f"--engine_dir={engine_dir}",
        f"--tensorrt_llm_rouge1_threshold={tensorrt_llm_rouge1_threshold}",
        "--check_accuracy", f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_gather_logits_2gpu_pp2(llama_example_root,
                                             llama_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llama_v2_tokenizer_model_root,
                                             llm_venv, cmodel_dir, engine_dir):
    # Check the availability of gather all token logits when pp>1
    model_name = 'llama_v2'
    data_type = 'float16'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root,
                                data_type=data_type,
                                pp_size=2)
    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", "--max_batch_size=2",
        "--max_beam_width=1", f"--gemm_plugin={data_type}",
        f"--gpt_attention_plugin={data_type}", "--gather_context_logits"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{llama_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llama_v2_tokenizer_model_root}",
        "--data_type=fp16", f"--engine_dir={engine_dir}", "--eval_ppl",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llama_model_root", [
    'llama-v2-7b-hf', 'llama-v2-13b-hf', 'llama-v2-70b-hf', 'Llama-2-7B-AWQ',
    'Llama-2-7B-GPTQ'
],
                         indirect=True)
def test_llm_llama_v2_awq_2gpu_summary(llama_example_root, llama_model_root,
                                       llama_v2_tokenizer_model_root,
                                       llm_datasets_root, llm_rouge_root,
                                       llm_venv, engine_dir, num_beams,
                                       qcache_dir_without_install_package):
    if (num_beams > 2
            or "70b" in llama_model_root) and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    if 'Llama-2-7B-AWQ' in llama_model_root or 'Llama-2-7B-GPTQ' in llama_model_root:
        print("Converting model...")
        ckpt_dir = convert_weights(
            llm_venv=llm_venv,
            example_root=llama_example_root,
            cmodel_dir=qcache_dir_without_install_package,
            model="llama_v2",
            model_path=llama_model_root,
            data_type="auto",
            tp_size=2,
            pp_size=1)
    else:
        print("Quantizing model...")
        ckpt_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="int4_awq",
            quantize_dir=qcache_dir_without_install_package,
            tp_size=2,
            calib_size=32)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llama_v2_tokenizer_model_root,
        data_type="fp16",
        engine_dir=engine_dir,
        num_beams=num_beams,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@skip_pre_ada
@skip_post_blackwell  # AutoQ contains AWQ int4 recipe, which is not supported on Blackwell
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b'], indirect=True)
def test_llm_llama_v3_1_autoq_1gpu_mmlu(llama_example_root, llama_model_root,
                                        llm_datasets_root, mmlu_dataset_root,
                                        llm_venv, engine_dir,
                                        qcache_dir_without_install_package):
    print("Quantizing model...")
    ckpt_dir = quantize_data(llm_venv,
                             llama_example_root,
                             model_dir=llama_model_root,
                             calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
                             dtype="float16",
                             quantize_dir=qcache_dir_without_install_package,
                             tp_size=1,
                             calib_size=4,
                             batch_size=4,
                             autoq_format='int4_awq,fp8,w4a8_awq',
                             auto_quantize_bits=5.8)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--remove_input_padding=enable",
        "--max_batch_size=8",
        "--max_input_len=4000",
        "--max_seq_len=4096",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run MMLU test")
    mmlu_cmd = [
        "trtllm-eval", f"--model={engine_dir}",
        f"--tokenizer={llama_model_root}", "--backend=tensorrt", "mmlu",
        f"--dataset_path={mmlu_dataset_root}", "--check_accuracy",
        f"--accuracy_threshold={63.8}"
    ]
    check_call(" ".join(mmlu_cmd), shell=True, env=llm_venv._new_env)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-70b'], indirect=True)
def test_llm_llama_v3_1_autoq_2gpu_mmlu(llama_example_root, llama_model_root,
                                        llm_datasets_root, mmlu_dataset_root,
                                        llm_venv, engine_dir,
                                        qcache_dir_without_install_package):
    print("Quantizing model...")
    ckpt_dir = quantize_data(llm_venv,
                             llama_example_root,
                             model_dir=llama_model_root,
                             calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
                             dtype="float16",
                             quantize_dir=qcache_dir_without_install_package,
                             tp_size=2,
                             calib_size=4,
                             batch_size=4,
                             autoq_format='int4_awq,fp8,w4a8_awq',
                             auto_quantize_bits=5.8)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--remove_input_padding=enable",
        "--max_batch_size=8",
        "--max_input_len=4000",
        "--max_seq_len=4096",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run MMLU test")
    mmlu_cmd = [
        "trtllm-eval", f"--model={engine_dir}",
        f"--tokenizer={llama_model_root}", "--backend=tensorrt", "mmlu",
        f"--dataset_path={mmlu_dataset_root}", "--check_accuracy",
        f"--accuracy_threshold={77.58}"
    ]
    check_call(" ".join(mmlu_cmd), shell=True, env=llm_venv._new_env)


@skip_post_blackwell
@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_beams", [4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llama_model_root", ['llama-7b', 'llama-30b'],
                         indirect=True)
def test_llm_llama_v1_2gpu_summary(llama_example_root, llama_model_root,
                                   llm_datasets_root, llm_rouge_root, llm_venv,
                                   cmodel_dir, engine_dir, num_beams):
    model_name = 'llama_v1_2gpu'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root,
                                gpus=2,
                                tp_size=2,
                                pp_size=1)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py", "--test_trt_llm",
        "--check_accuracy", f"--hf_model_dir={llama_model_root}",
        f"--engine_dir={engine_dir}", f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_host_memory(480000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llama_model_root", ['llama-v2-70b'], indirect=True)
def test_llm_llama_v2_8gpu_summary(llama_example_root, llama_model_root,
                                   llama_v2_tokenizer_model_root,
                                   llm_datasets_root, llm_rouge_root, llm_venv,
                                   cmodel_dir, engine_dir, num_beams):
    "run llamav2 70 test on 8 gpus"
    if num_beams > 2 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    model_name = 'llama_v2-meta-ckpt-70b'

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root,
                                gpus=8,
                                workers=8,
                                tp_size=8,
                                pp_size=1)
    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}",
        "--workers=8",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llama_v2_tokenizer_model_root,
        data_type="fp16",
        engine_dir=engine_dir,
        num_beams=num_beams,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "8", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("num_beams", [2, 5],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_v1_1gpu_paged_kv_cache(llama_example_root, llama_model_root,
                                          llm_datasets_root, llm_rouge_root,
                                          llm_venv, cmodel_dir, engine_dir,
                                          num_beams):
    "RCCA https://nvbugs/4283902"
    print("Build engines...")
    model_name = 'llama_v1-paged_kv_cache'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--max_beam_width={num_beams}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
        "--max_batch_size=2",
        "--tokens_per_block=16",
        "--paged_kv_cache=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llama_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b'], indirect=True)
def test_llm_llama_v1_4gpu_paged_kv_cache(llama_example_root, llama_model_root,
                                          llm_venv, cmodel_dir, engine_dir):
    """
        RCCA https://nvbugs/4251782
        RCCA https://nvbugs/4755248
    """
    model_name = 'llama_v1-4gpu_paged_kv_cache'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root,
                                gpus=4,
                                tp_size=4,
                                pp_size=1)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        "--max_batch_size=128",
        "--max_input_len=512",
        "--max_seq_len=1024",
        "--max_beam_width=1",
        "--paged_kv_cache=enable",
    ]

    print("Build engines...")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=10",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--max_attention_window_size=128",
        "--kv_cache_enable_block_reuse",
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        run_cmd)


@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_v1_1gpu_kv_cache_reuse_with_prompt_table(
        llama_example_root, llama_model_root, llm_datasets_root, llm_rouge_root,
        llm_venv, cmodel_dir, engine_dir):
    max_prompt_embedding_table_size = 16
    hidden_size = 4096
    vocab_size = 32000
    input_len = 42

    print("Convert checkpoint...")
    model_name = 'llama_v1-kv_cache_reuse_w_prompt_table'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}/engines", "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16", "--remove_input_padding=enable",
        "--max_batch_size=1",
        f"--tokens_per_block={max_prompt_embedding_table_size}",
        "--paged_kv_cache=enable", "--use_paged_context_fmha=enable",
        f"--max_prompt_embedding_table_size={max_prompt_embedding_table_size}"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # generate input ids, dummy prompt table and extra ids
    input_file = f"{engine_dir}/input_ids.npy"
    prompt_table_path = f"{engine_dir}/prompt_table.npy"
    extra_ids_file = f"{engine_dir}/extra_ids.npy"
    # run the script inside venv since it depends on numpy
    venv_script = f'''
    import numpy as np
    input_ids = [[
        i + {vocab_size} if i < {max_prompt_embedding_table_size} else i + 1000
        for i in range({input_len})
    ]]
    np.save("{input_file}", np.array(input_ids))

    prompt_table_shape = (1, {max_prompt_embedding_table_size}, {hidden_size})
    prompt_table = np.random.rand(*prompt_table_shape).astype(np.float16)
    np.save("{prompt_table_path}", prompt_table)

    extra_ids = [[
        1 if i < {max_prompt_embedding_table_size} else 0
        for i in range({input_len})
    ]]
    np.save("{extra_ids_file}", np.array(extra_ids))
    '''
    llm_venv.run(venv_script)

    # add --run_profiling to run the request for multiple times
    print("Run inference")
    run_cmd = [
        f"{llama_example_root}/../../../run.py", "--max_output_len=10",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}/engines", f"--input_file={input_file}",
        f"--prompt_table_path={prompt_table_path}",
        "--kv_cache_enable_block_reuse",
        f"--input_token_extra_ids_file={extra_ids_file}", "--run_profiling"
    ]
    venv_check_output(llm_venv, run_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize(
    "fp8_context_fmha_xqa",
    ["enable_fp8_context_fmha_xqa", "disable_fp8_context_fmha_xqa"])
@pytest.mark.parametrize("reduce_fusion",
                         ["enable_reduce_fusion", "disable_reduce_fusion"])
@pytest.mark.parametrize("llama_model_root",
                         ['llama-7b', 'llama-v2-13b-hf', 'llama-v2-70b-hf'],
                         indirect=True)
def test_llm_llama_2gpu_fp8_summary(llama_example_root, llama_model_root,
                                    llm_datasets_root, llm_rouge_root, llm_venv,
                                    engine_dir,
                                    qcache_dir_without_install_package,
                                    fp8_context_fmha_xqa, reduce_fusion):
    "RCCA https://nvbugs/4348560"
    skip_fp8_pre_ada(use_fp8=True)
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="float16",
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
        tp_size=2,
        calib_size=512,
        kv_cache_dtype="fp8")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qmodel_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
        "--workers=2",
        "--max_beam_width=4",
    ]

    if "enable" in fp8_context_fmha_xqa:
        build_cmd.extend([
            "--use_fp8_context_fmha=enable", "--use_paged_context_fmha=enable"
        ])

    if "enable" in reduce_fusion:
        build_cmd.extend(["--reduce_fusion=enable"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llama_model_root,
                                       data_type='fp16',
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root,
                                       num_beams=4)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_1gpu_batched_beam_search(llama_example_root,
                                            llama_model_root, llm_datasets_root,
                                            llm_venv, engine_dir,
                                            qcache_dir_without_install_package):
    "llama run batched beam search on 1 gpu"
    qmodel_dir = quantize_data(llm_venv,
                               llama_example_root,
                               model_dir=llama_model_root,
                               dtype="float16",
                               quantize_dir=qcache_dir_without_install_package)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--paged_kv_cache=enable",
        "--max_batch_size=4",
        "--max_beam_width=4",
        "--max_input_len=512",
        "--max_seq_len=532",
        "--gemm_plugin=float16",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # run.py test.
    num_beams = 4
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--no_add_special_tokens",
        f"--num_beams={num_beams}",
        "--input_text",
        "Miguel de Cervantes wrote",
        "Diego Velazquez painted his most famous painting,",
        "Miguel de Cervantes wrote",
        "Diego Velazquez painted his most famous painting,",
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = parse_output(output)

    for idx in [0, 1]:
        assert all(
            [
                a == b for (a, b) in zip(
                    output[num_beams * idx:num_beams * idx +
                           num_beams], output[num_beams * (idx + 2):num_beams *
                                              (idx + 2) + num_beams])
            ]
        ), f"outputs {idx} and {idx+2} don't match: {output[num_beams * idx:num_beams * idx + num_beams]}, {output[num_beams * (idx + 2):num_beams * (idx + 2) + num_beams]}"

    expected_output = [
        ["Don Quixote in 1605. The book is considered the first modern novel."],
        [
            "Las Meninas, in 1656. The painting is a portrait of King Philip IV",
            "\"Las Meninas\" in 1656. The painting depicts King Philip"
        ],
    ]

    for idx, result in enumerate(output):
        assert any(
            [
                similar(item, result)
                for item in expected_output[(idx // num_beams) % 2]
            ]
        ), f"output {result} is not similar to any of {expected_output[(idx // num_beams) % 2]}"


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("mmlu_test", [True, False],
                         ids=["enable_mmlu_test", "disable_mmlu_test"])
@pytest.mark.parametrize(
    "fp8_fmha",
    ["enable_fp8_fmha", "enable_fp8_paged_fmha", "disable_fp8_fmha"])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_1gpu_fp8_summary_and_mmlu(
        llama_example_root, llama_model_root, llm_datasets_root, llm_rouge_root,
        mmlu_dataset_root, mmlu_test, llm_venv, engine_dir,
        qcache_dir_without_install_package, fp8_fmha):
    "run Llama v2 fp8 quantization tests"
    skip_fp8_pre_ada(use_fp8=True)
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="bfloat16",
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
        calib_size=512,
        kv_cache_dtype="fp8")

    print("Build engines...")
    use_fp8_context_fmha = "enable" if fp8_fmha in [
        "enable_fp8_fmha", "enable_fp8_paged_fmha"
    ] else "disable"
    use_paged_context_fmha = "enable" if fp8_fmha == "enable_fp8_paged_fmha" else "disable"
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--use_fp8_context_fmha={use_fp8_context_fmha}",
        f"--use_paged_context_fmha={use_paged_context_fmha}",
        "--remove_input_padding=enable",
        "--max_batch_size=4",
        "--max_input_len=2046",
        "--max_seq_len=2048",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # run.py test.
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=32",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--no_add_special_tokens",
        "--input_text",
        INPUT_TEXT_1,
        INPUT_TEXT_2,
        INPUT_TEXT_2,
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = parse_output(output)
    print(output)

    print("Run Summarization test with batch size = 1")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        f"{llama_model_root}",
        "--data_type",
        "fp16",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
        # rouge1 threshold reduced from 15 (default) to 14 since we now enable fused mlp by default and the scales of two linear layers can be different
        "--tensorrt_llm_rouge1_threshold=14",
    ]

    venv_check_call(llm_venv, summary_cmd)

    if mmlu_test:
        print("Run MMLU test")
        mmlu_cmd = [
            "trtllm-eval", f"--model={engine_dir}",
            f"--tokenizer={llama_model_root}", "--backend=tensorrt", "mmlu",
            f"--dataset_path={mmlu_dataset_root}", "--check_accuracy",
            f"--accuracy_threshold={45.0}"
        ]
        check_call(" ".join(mmlu_cmd), shell=True, env=llm_venv._new_env)


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_1gpu_fp8_gemv(llama_example_root, llama_model_root,
                                    llm_datasets_root, llm_venv, engine_dir,
                                    qcache_dir_without_install_package):
    "run Llama v2 fp8 quantization tests"
    skip_fp8_pre_ada(use_fp8=True)
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="bfloat16",
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
        calib_size=512,
        kv_cache_dtype="fp8")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--gemm_plugin=fp8",
        "--max_batch_size=4",
        "--max_input_len=2048",
        "--max_seq_len=2048",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # run.py test.
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=32",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--no_add_special_tokens",
        "--input_text",
        INPUT_TEXT_1,
        INPUT_TEXT_2,
        INPUT_TEXT_2,
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = parse_output(output)
    print(output)

    print("Run Summarization test with batch size = 1")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        f"{llama_model_root}",
        "--data_type",
        "fp16",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}",
        "--tensorrt_llm_rouge1_threshold=14.5",
    ]

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("gemm_swiglu_plugin", ["fp8"])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_1gpu_gemm_swiglu(llama_example_root, llama_model_root,
                                       llm_datasets_root, llm_venv, engine_dir,
                                       qcache_dir_without_install_package,
                                       gemm_swiglu_plugin, data_type):
    "run Llama v2 gemm_swiglu_plugin tests"
    if gemm_swiglu_plugin == "fp8":
        skip_fp8_pre_ada(use_fp8=True)
        qmodel_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype=data_type,
            qformat="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512,
            kv_cache_dtype="fp8")
    else:
        pytest.skip(f"gemm_swiglu_plugin only supports fp8 now")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin=fp8",
        f"--gemm_swiglu_plugin={gemm_swiglu_plugin}",
        "--remove_input_padding=enable",
        "--max_batch_size=4",
        "--max_input_len=2048",
        "--max_seq_len=2048",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # run.py test.
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=32",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--no_add_special_tokens",
        "--input_text",
        INPUT_TEXT_1,
        INPUT_TEXT_2,
        INPUT_TEXT_2,
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = parse_output(output)
    print(output)

    print("Run Summarization test")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        f"{llama_model_root}",
        "--data_type",
        "fp16",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        "--max_ite=40",
        f"--dataset_dir={llm_datasets_root}",
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize(
    "data_type", [
        'float16', 'fp8',
        pytest.param('sq_ootb', marks=skip_post_blackwell),
        pytest.param('awq', marks=skip_post_blackwell),
        pytest.param('int8_wo', marks=skip_post_blackwell)
    ],
    ids=['base_fp16', 'base_fp8', 'base_sq_ootb', 'base_awq', 'base_int8_wo'])
@pytest.mark.parametrize("lora_data_type", ['float16'], ids=['lora_fp16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-13b-hf'], indirect=True)
@pytest.mark.parametrize("llm_lora_model_root", ['chinese-llama-2-lora-13b'],
                         indirect=True)
def test_llm_llama_v2_lora_1gpu(data_type, lora_data_type, llama_example_root,
                                llama_model_root, llm_datasets_root, llm_venv,
                                cmodel_dir, engine_dir, llm_lora_model_root,
                                qcache_dir_without_install_package):
    "run llama lora test on 1gpu"
    print("Build engines...")

    model_name = 'llama_v2-lora'
    if data_type == 'fp8':
        skip_fp8_pre_ada(use_fp8=True)

        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512,
            kv_cache_dtype="fp8")
    elif data_type == 'sq_ootb':
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="int8_sq",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=32)
    elif data_type == 'awq':
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="int4_awq",
            awq_block_size=128,
            quantize_dir=qcache_dir_without_install_package,
            calib_size=32)
    elif data_type == 'int8_wo':
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    use_weight_only=True,
                                    weight_only_precision='int8')
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--gemm_plugin=auto",
        f"--lora_dir={llm_lora_model_root}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        29871, 32160, 33657, 33281, 30214, 30672, 30780, 33820, 32024, 30214,
        32083, 33820, 30755, 37432, 32030, 30313, 30214, 30417, 30210, 30505,
        34870, 30214, 30417, 30210, 30505, 31656, 39298, 30214, 32063, 30210
    ]
    ref_2 = [
        29871, 32160, 33657, 33281, 30214, 30672, 30780, 33820, 32024, 30214,
        33759, 41026, 31381, 30769, 31811, 31900, 30214, 36869, 31900, 36869,
        31900, 30214, 36869, 31900, 36869, 31900, 31900, 31900, 31900, 31900
    ]

    input_text = "今天天气很好，我到公园的时候，"
    # TODO change to chinese evaluation task in the future

    base_run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        "--no_add_special_tokens",
    ]

    for use_py_session in [True, False]:
        if use_py_session:
            print("Run inference with Python runtime...")
        else:
            print("Run inference with C++ runtime...")

        print(f"Run inference with lora id 0...")
        run_cmd = copy.deepcopy(base_run_cmd)
        run_cmd.extend([
            "--lora_task_uids=0",
            f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv"
        ])
        if use_py_session:
            run_cmd.append("--use_py_session")
        venv_check_call(llm_venv, run_cmd)

        with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
            predict = csv.reader(f)
            predict = next(predict)
        predict = [int(p) for p in predict]
        assert ref_1 == predict or data_type != "float16"

        print(f"Run inference with lora id -1...")
        run_cmd = copy.deepcopy(base_run_cmd)
        run_cmd.extend([
            "--lora_task_uids=-1",
            f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv"
        ])
        if use_py_session:
            run_cmd.append("--use_py_session")
        venv_check_call(llm_venv, run_cmd)

        with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
            predict = csv.reader(f)
            predict = next(predict)
        predict = [int(p) for p in predict]
        assert ref_2 == predict or data_type != "float16"


@pytest.mark.parametrize(
    "data_type", ['float16', 'fp8', 'sq_ootb', 'awq', 'int8_wo'],
    ids=['base_fp16', 'base_fp8', 'base_sq_ootb', 'base_awq', 'base_int8_wo'])
@pytest.mark.parametrize("llama_model_root", ['llama-v3-8b-hf'], indirect=True)
@pytest.mark.parametrize("llm_dora_model_root",
                         ['commonsense-llama-v3-8b-dora-r32'],
                         indirect=True)
def test_llm_llama_v3_dora_1gpu(data_type, llama_example_root, llama_model_root,
                                llm_dora_model_root, llm_datasets_root,
                                llm_venv, cmodel_dir, engine_dir,
                                qcache_dir_without_install_package):
    "run llama dora test on 1gpu"
    print("Build engines...")

    model_name = 'llama_v3-dora'
    if data_type == 'fp8':
        skip_fp8_pre_ada(use_fp8=True)

        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512,
            kv_cache_dtype="fp8")
    elif data_type == 'sq_ootb':
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="int8_sq",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=32)
    elif data_type == 'awq':
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="int4_awq",
            awq_block_size=128,
            quantize_dir=qcache_dir_without_install_package,
            calib_size=32)
    elif data_type == 'int8_wo':
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    use_weight_only=True,
                                    weight_only_precision='int8')
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root)

    # normalize dora magnitude
    dora_weights = f"{llm_venv.get_working_directory()}/dora_weights"

    normalize_cmd = [
        f"{llama_example_root}/../../../dora/normalize_weights.py", "-i",
        llm_dora_model_root, "-b", llama_model_root, "-o", dora_weights
    ]

    venv_check_call(llm_venv, normalize_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--dora_plugin=enable",
        "--remove_input_padding=enable",  # otherwise no cpp runtime
        "--kv_cache_type=paged",  # otherwise no cpp runtime
        "--gemm_plugin=auto",
        f"--lora_dir={dora_weights}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    input_tokens = [
        128000, 39314, 374, 459, 7754, 430, 16964, 264, 3465, 11, 35526, 449,
        459, 1988, 430, 5825, 4726, 2317, 13, 9842, 264, 2077, 430, 36001,
        45695, 279, 1715, 382, 394, 17010, 30151, 512, 394, 5321, 5268, 279,
        4495, 4320, 311, 279, 3488, 25, 578, 842, 1121, 304, 279, 1920, 315,
        7397, 74767, 374, 279, 5788, 315, 13465, 323, 24463, 13, 16299, 3094,
        17738, 279, 7314, 315, 7397, 74767, 31931, 16533, 16, 25, 36424, 4907,
        374, 42101, 1555, 279, 20282, 13, 22559, 17, 25, 8828, 4907, 374, 16489,
        311, 11742, 4907, 13, 22559, 18, 25, 92479, 5237, 25734, 304, 279,
        16312, 41255, 3177, 4907, 13, 22559, 19, 25, 8219, 4238, 374, 16489,
        1139, 37833, 5237, 25734, 4286, 16533, 3645, 25, 4320, 16, 14, 9399, 17,
        14, 9399, 18, 14, 9399, 19, 271, 394, 17010, 5688, 512, 72348, 394,
        17010, 6075, 1473
    ]

    out_ref = [
        128000, 39314, 374, 459, 7754, 430, 16964, 264, 3465, 11, 35526, 449,
        459, 1988, 430, 5825, 4726, 2317, 13, 9842, 264, 2077, 430, 36001,
        45695, 279, 1715, 382, 394, 17010, 30151, 512, 394, 5321, 5268, 279,
        4495, 4320, 311, 279, 3488, 25, 578, 842, 1121, 304, 279, 1920, 315,
        7397, 74767, 374, 279, 5788, 315, 13465, 323, 24463, 13, 16299, 3094,
        17738, 279, 7314, 315, 7397, 74767, 31931, 16533, 16, 25, 36424, 4907,
        374, 42101, 1555, 279, 20282, 13, 22559, 17, 25, 8828, 4907, 374, 16489,
        311, 11742, 4907, 13, 22559, 18, 25, 92479, 5237, 25734, 304, 279,
        16312, 41255, 3177, 4907, 13, 22559, 19, 25, 8219, 4238, 374, 16489,
        1139, 37833, 5237, 25734, 4286, 16533, 3645, 25, 4320, 16, 14, 9399, 17,
        14, 9399, 18, 14, 9399, 19, 271, 394, 17010, 5688, 512, 72348, 394,
        17010, 6075, 1473, 394, 279, 4495, 4320, 374, 4320, 18, 128001, 128001,
        128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001,
        128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001,
        128001, 128001, 128001, 128001, 128001
    ]

    in_csv = f"{llm_venv.get_working_directory()}/input.csv"
    out_csv = f"{llm_venv.get_working_directory()}/output.csv"
    with open(in_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(input_tokens)

    base_run_cmd = [
        f"{llama_example_root}/../../../run.py", "--max_output_len=20",
        f"--input_file={in_csv}", f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}", "--max_output_len=32"
    ]

    for use_py_session in [True, False]:
        if use_py_session:
            print("Run inference with Python runtime...")
        else:
            print("Run inference with C++ runtime...")

        print(f"Run inference with lora id 0...")
        run_cmd = copy.deepcopy(base_run_cmd)
        run_cmd.extend(["--lora_task_uids=0", f"--output_csv={out_csv}"])
        if use_py_session:
            run_cmd.append("--use_py_session")
        venv_check_call(llm_venv, run_cmd)

        with open(out_csv) as f:
            predict = csv.reader(f)
            predict = next(predict)

        predict = [int(p) for p in predict]
        assert out_ref == predict or data_type != "float16"


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "tp_pp_size", [(8, 1), (4, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize("test_case", ["pg64317"], indirect=True)
def test_llm_llama_long_alpaca_8gpu_summary(llama_example_root,
                                            llm_long_alpaca_model_root,
                                            llm_datasets_root, llm_rouge_root,
                                            llm_venv, cmodel_dir, engine_dir,
                                            num_beams, tp_pp_size, test_case):
    "llama test for long alpaca"
    tp_size, pp_size = tp_pp_size
    world_size = 8
    assert tp_size * pp_size == world_size, \
        f'tp_size({tp_size}) x pp_size({pp_size}) != 8'

    model_name = 'llama_long_alpaca'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llm_long_alpaca_model_root,
                                gpus=world_size,
                                tp_size=tp_size,
                                pp_size=pp_size,
                                data_type="bfloat16")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        "--remove_input_padding=enable",
        "--gemm_plugin=bfloat16",
        f"--max_beam_width={num_beams}",
        "--max_input_len=32768",
        "--max_seq_len=49152",
        "--max_batch_size=1",
        "--max_num_tokens=32768",
    ]
    print("Build engines...")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    max_output_len = test_case["max_output_len"]
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        f"--max_output_len={max_output_len}",
        f"--input_file={test_case['input_file']}", f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}",
        f"--tokenizer_dir={llm_long_alpaca_model_root}",
        "--max_input_length=32768"
    ]

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        run_cmd)

    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llm_long_alpaca_model_root,
                                       max_input_length=16384,
                                       output_len=max_output_len,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_v1_1gpu_streaming_llm(llama_example_root, llama_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         num_beams, gemm_plugin):
    "Run LLaMa with StreamingLLM"
    model_name = 'llama_v1-streamingllm'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root)
    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        f"--max_beam_width={num_beams}",
        "--streamingllm=enable",
        "--max_batch_size=256",
    ]
    if gemm_plugin:
        build_cmd.append("--gemm_plugin=float16")
    else:
        build_cmd.append("--gemm_plugin=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llama_model_root,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       max_attention_window_size=2048,
                                       sink_token_length=4,
                                       num_beams=num_beams,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "gpt_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize(
    "context_fmha_type",
    ["enable_context_fmha", "enable_with_fp32_acc", "disable_context_fmha"])
@pytest.mark.parametrize("code_llama_model_root", ['CodeLlama-7b-Instruct'],
                         indirect=True)
def test_llm_llama_code_llama_1gpu_summary(
        llama_example_root, code_llama_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, cmodel_dir, engine_dir, num_beams,
        gemm_plugin, gpt_attention_plugin, context_fmha_type):
    "Run CodeLlaMa on single gpu"

    model_name = 'code_llama_1gpu'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=code_llama_model_root,
                                data_type="float16",
                                gpus=1,
                                tp_size=1,
                                pp_size=1)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--max_beam_width={num_beams}",
        f"--max_seq_len={8192}",
    ]
    if gpt_attention_plugin:
        build_cmd.extend(
            ["--remove_input_padding=enable", "--gpt_attention_plugin=float16"])
    else:
        build_cmd.append("--gpt_attention_plugin=disable")
        build_cmd.append("--remove_input_padding=disable")
        build_cmd.append("--paged_kv_cache=disable")

    if gemm_plugin:
        build_cmd.append("--gemm_plugin=float16")
    else:
        build_cmd.append("--gemm_plugin=disable")

    if context_fmha_type == "enable_context_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disable_context_fmha":
        build_cmd.append("--context_fmha=disable")

    print("Build engines...")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=40",
        f"--tokenizer_dir={code_llama_model_root}",
        f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}",
        "--input_text='In Bash, how do I list all text files?'",
    ]
    if context_fmha_type == "enable_with_fp32_acc":
        run_cmd.append("--enable_context_fmha_fp32_acc")
    venv_check_call(llm_venv, run_cmd)

    print("Run inference")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=code_llama_model_root,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold=17,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.timeout(7200)
@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "tp_pp_size", [(4, 1), (2, 2), (8, 1), (4, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize("code_llama_model_root",
                         ['CodeLlama-34b-Instruct', 'CodeLlama-70b-hf'],
                         indirect=True)
def test_llm_llama_code_llama_multi_gpus_summary(llama_example_root,
                                                 code_llama_model_root,
                                                 llm_datasets_root,
                                                 llm_rouge_root, llm_venv,
                                                 cmodel_dir, engine_dir,
                                                 num_beams, tp_pp_size):
    "Run CodeLlaMa on 4 gpus"
    tp_size, pp_size = tp_pp_size
    world_size = tp_size * pp_size

    if get_device_count() < world_size:
        pytest.skip(f"devices are less than {world_size}.")

    model_name = 'code_llama'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=code_llama_model_root,
                                data_type="float16",
                                gpus=world_size,
                                tp_size=tp_size,
                                pp_size=pp_size)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        f"--max_beam_width={num_beams}",
        f"--workers={world_size}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=160",
        f"--tokenizer_dir={code_llama_model_root}",
        f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}",
        "--input_text='In python, write a function for binary searching an element in an integer array.'",
    ]
    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        run_cmd)

    print("Run inference")
    tensorrt_llm_rouge1_threshold = 18 if "70b" in code_llama_model_root else 22
    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=code_llama_model_root,
        data_type="fp16",
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("per_token_channel", [True, False],
                         ids=["enable_ptpc", "disable_ptpc"])
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
@pytest.mark.parametrize("data_type", ["float16", "bfloat16"])
def test_llm_llama_smooth_quant_1gpu_summary(llama_example_root,
                                             llama_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llm_venv, engine_dir, num_beams,
                                             per_token_channel, cmodel_dir,
                                             data_type):
    "Run smooth quant on single gpu"
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model="llama-smooth",
        model_path=llama_model_root,
        gpus=1,
        smoothquant=0.55,
        per_token=per_token_channel,
        per_channel=per_token_channel,
        calib_dataset=f"{llm_datasets_root}/ccdv/cnn_dailymail",
        data_type=data_type)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        "--remove_input_padding=enable",
        f"--gemm_plugin={data_type}",
        "--context_fmha=enable",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    rouge1_threshold = 17
    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llama_model_root,
        data_type="fp16",
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=rouge1_threshold,
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=['enable_weight_only', 'disable_weight_only'])
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_int8_kv_1gpu_summary(llama_example_root, llama_model_root,
                                        llm_datasets_root, llm_rouge_root,
                                        llm_venv, engine_dir, num_beams,
                                        use_weight_only,
                                        qcache_dir_without_install_package):
    print("Quantizing model...")
    qformat = "int8_wo" if use_weight_only else "full_prec"
    ckpt_dir = quantize_data(llm_venv,
                             llama_example_root,
                             model_dir=llama_model_root,
                             calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
                             dtype="float16",
                             qformat=qformat,
                             quantize_dir=qcache_dir_without_install_package,
                             calib_size=32,
                             kv_cache_dtype="int8")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llama_model_root,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold=19,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_int8_sq_ootb_1gpu_summary(
        llama_example_root, llama_model_root, llm_datasets_root, llm_rouge_root,
        llm_venv, engine_dir, num_beams, qcache_dir_without_install_package):
    print("Quantizing model...")
    ckpt_dir = quantize_data(llm_venv,
                             llama_example_root,
                             model_dir=llama_model_root,
                             calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
                             dtype="float16",
                             qformat="int8_sq",
                             quantize_dir=qcache_dir_without_install_package,
                             calib_size=32,
                             kv_cache_dtype="int8")

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable", "--gemm_plugin=disable",
        f"--max_beam_width={num_beams}"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")

    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llama_model_root,
        data_type="fp16",
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=
        15.2,  #Adjust to 15.2 for using TRT build optimization level 3
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['bfloat16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_int8sq_2gpu_tp2(data_type, llama_example_root,
                                      llama_model_root,
                                      llama_v2_tokenizer_model_root,
                                      llm_datasets_root, llm_rouge_root,
                                      llm_venv, engine_dir, num_beams,
                                      qcache_dir_without_install_package):
    if num_beams > 2 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    # Quantize HF llama checkpoint into int8_sq format
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat="int8_sq",
        quantize_dir=qcache_dir_without_install_package,
        tp_size=2,
        pp_size=1,
        calib_size=32)

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

    print("Run summarize...")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        f"{llama_v2_tokenizer_model_root}",
        "--data_type=fp16",
        f"--engine_dir={engine_dir}",
        "--tensorrt_llm_rouge1_threshold=15",
        "--check_accuracy",
        f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("weight_only_precision", ["int4", "int8"])
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_wo_1gpu_summary(llama_example_root, llama_model_root,
                                   llm_datasets_root, llm_rouge_root, llm_venv,
                                   engine_dir, num_beams, cmodel_dir,
                                   weight_only_precision):

    skip_fp8_pre_ada(use_fp8=True)

    llm_venv.get_working_directory()
    model_name = os.path.basename(llama_example_root)

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llama_model_root,
                               data_type="float16",
                               use_weight_only=True,
                               weight_only_precision=weight_only_precision,
                               gpus=1,
                               tp_size=1,
                               pp_size=1)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")

    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llama_model_root,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold=20.2 if
                                       weight_only_precision == 'int8' else 16,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_int8_kv_awq_1gpu_summary(llama_example_root,
                                            llama_model_root, llm_datasets_root,
                                            llm_rouge_root, llm_venv,
                                            engine_dir, num_beams,
                                            qcache_dir_without_install_package):
    "Run int8 kv cache on single gpu"
    print("Quantizing model...")
    ckpt_dir = quantize_data(llm_venv,
                             llama_example_root,
                             model_dir=llama_model_root,
                             calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
                             dtype="float16",
                             qformat="int4_awq",
                             quantize_dir=qcache_dir_without_install_package,
                             calib_size=32,
                             kv_cache_dtype="int8")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--gemm_plugin=float16",
        f"--max_beam_width={num_beams}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llama_model_root,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold=15,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("data_type", ['float16', 'fp8'],
                         ids=['base_fp16', 'base_fp8'])
@pytest.mark.parametrize("lora_data_type", ['float16'], ids=['lora_fp16'])
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
@pytest.mark.parametrize("llm_lora_model_root",
                         [("luotuo-lora-7b-0.1", "Japanese-Alpaca-LoRA-7b-v0")],
                         ids=["luotuo_japan"],
                         indirect=True)
def test_llm_llama_v1_multiple_lora_1gpu(data_type, lora_data_type,
                                         llama_example_root, llama_model_root,
                                         llm_datasets_root, llm_venv,
                                         cmodel_dir, engine_dir,
                                         llm_lora_model_root,
                                         qcache_dir_without_install_package):
    "run llama with multi lora on 1gpu"
    first_lora, second_lora = llm_lora_model_root.split(",")

    print("Build engines...")
    if data_type == 'fp8':
        skip_fp8_pre_ada(use_fp8=True)
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512,
            kv_cache_dtype="fp8")
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model="llama-lora",
                                    model_path=llama_model_root,
                                    gpus=1,
                                    tp_size=1,
                                    data_type=data_type)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--gemm_plugin=auto",
        "--lora_plugin=auto",
        "--max_batch_size=128",
        "--max_input_len=512",
        "--max_seq_len=562",
        "--lora_dir",
        f"{first_lora}",
        f"{second_lora}",
        "--max_lora_rank=8",
        "--lora_target_modules",
        "attn_q",
        "attn_k",
        "attn_v",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    base_run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--input_text",
        "美国的首都在哪里? \n答案:",
        "美国的首都在哪里? \n答案:",
        "美国的首都在哪里? \n答案:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--lora_task_uids",
        "-1",
        "0",
        "1",
        "-1",
        "0",
        "1",
        "--top_p=0.5",
        "--top_k=0",
        "--random_seed=0",
        "--max_output_len=10",
    ]

    for use_py_session in [True, False]:
        run_cmd = copy.deepcopy(base_run_cmd)
        if use_py_session:
            print("Run inference with Python runtime...")
            run_cmd.append("--use_py_session")
        else:
            print("Run inference with C++ runtime...")

        # TODO: add step to check result
        venv_check_call(llm_venv, run_cmd)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("llama_model_root", ['llama-v2-13b-hf'], indirect=True)
@pytest.mark.parametrize("llm_lora_model_root", ["chinese-llama-2-lora-13b"],
                         ids=["chinese_lora"],
                         indirect=True)
def test_llm_llama_v2_lora_benchmark_2gpu(llama_example_root, llama_model_root,
                                          llm_venv, llm_root, cmodel_dir,
                                          engine_dir, llm_lora_model_root):
    "benchmark llama with multi lora on 2gpu"
    print("Build engines...")

    num_layers = 40
    num_lora_mods = 7
    max_lora_rank = 64
    max_len = 1024
    max_batch = 32
    eos_id = 2
    num_loras = (8, 16)
    num_requests = 1024

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model="llama-lora",
                                model_path=llama_model_root,
                                gpus=2,
                                tp_size=2,
                                data_type="float16")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={max_batch}",
        f"--max_input_len={max_len}",
        f"--max_seq_len={2 * max_len}",
        "--gemm_plugin=float16",
        "--lora_plugin=float16",
        "--use_paged_context_fmha=enable",
        "--lora_target_modules",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_dense",
        "mlp_h_to_4h",
        "mlp_4h_to_h",
        "mlp_gate",
        f"--max_lora_rank={max_lora_rank}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Convert LoRA to cpp format")
    convert_cmd = [
        "python",
        f"{llama_example_root}/../../../hf_lora_convert.py",
        f"-i={llm_lora_model_root}",
        "--storage-type=float16",
        f"-o={llm_venv.get_working_directory()}/lora_cpp",
    ]
    check_call(" ".join(convert_cmd), shell=True, env=llm_venv._new_env)

    print("Prepare datasets")
    benchmark_root = f"{llama_example_root}/../../../../benchmarks/cpp"
    lora_eg = f"{llm_venv.get_working_directory()}/lora-eg"
    base_dataset_cmd = [
        f"mkdir -p {lora_eg}/data",
        "&&",
        "python",
        f"{benchmark_root}/prepare_dataset.py",
        f"--output={lora_eg}/data/token-norm-dist.json",
        f"--tokenizer={llama_model_root}",
        "token-norm-dist",
        f"--num-requests={num_requests}",
        "--input-mean=256",
        "--input-stdev=16",
        "--output-mean=128",
        "--output-stdev 24",
    ]
    check_call(" ".join(base_dataset_cmd), shell=True, env=llm_venv._new_env)

    for nloras in num_loras:
        lora_dataset_cmd = [
            "python",
            f"{benchmark_root}/prepare_dataset.py",
            f"--output={lora_eg}/data/token-norm-dist-lora-{nloras}.json",
            f"--rand-task-id 0 {nloras-1}",
            f"--tokenizer={llama_model_root}",
            "token-norm-dist",
            f"--num-requests={num_requests}",
            "--input-mean=256",
            "--input-stdev=16",
            "--output-mean=128",
            "--output-stdev 24",
        ]
        check_call(" ".join(lora_dataset_cmd),
                   shell=True,
                   env=llm_venv._new_env)

    print("Generate random lora weights for 16 adapters")

    lora_weights_cmd = [
        "python", f"{benchmark_root}/utils/generate_rand_loras.py",
        f"{llm_venv.get_working_directory()}/lora_cpp", f"{lora_eg}/loras", "16"
    ]
    check_call(" ".join(lora_weights_cmd), shell=True, env=llm_venv._new_env)

    benchmark_exe = get_cpp_benchmark('gptManagerBenchmark', llm_root)
    envs = deepcopy(os.environ)
    _ = envs.pop("CUDA_VISIBLE_DEVICES", "")
    envs[
        "LD_LIBRARY_PATH"] = f'{get_trt_llm_lib_dir(llm_venv)}:{os.path.dirname(benchmark_exe)}:{envs.get("LD_LIBRARY_PATH", "")}'

    print(
        f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", None)}')

    print("Perform base model benchmarking")
    check_call(f"mkdir -p {lora_eg}/log-base-lora", shell=True, env=envs)
    base_benchmark_cmd = [
        f"{benchmark_exe}",
        f"--engine_dir={engine_dir}",
        "--type=IFB",
        f"--dataset={lora_eg}/data/token-norm-dist.json",
        "--lora_host_cache_bytes=8589934592",
        f"--lora_num_device_mod_layers={32 * num_layers * num_lora_mods * max_lora_rank}",
        "--kv_cache_free_gpu_mem_fraction=0.70",
        "--log_level=info",
        f"--eos_id={eos_id}",
    ]
    mpi_cmd = [
        "mpirun",
        "-n",
        "2",
        "--allow-run-as-root",
        "--output-filename",
        f"{lora_eg}/log-base-lora",
    ]
    base_benchmark_cmd = mpi_cmd + base_benchmark_cmd
    print(
        f"Running gptManagerBenchmark using base cmd: {' '.join(base_benchmark_cmd)}"
    )
    subprocess.check_output(base_benchmark_cmd, env=envs)
    # check_call(" ".join(base_benchmark_cmd), env=envs)

    print("Perform lora model benchmarking")
    for nloras in num_loras:
        check_call(f"mkdir -p {lora_eg}/log-lora-{nloras}",
                   shell=True,
                   env=envs)
        lora_benchmark_cmd = [
            f"{benchmark_exe}",
            f"--engine_dir={engine_dir}",
            "--type=IFB",
            f"--dataset={lora_eg}/data/token-norm-dist-lora-{nloras}.json",
            "--lora_host_cache_bytes=8589934592",
            f"--lora_num_device_mod_layers={16 * num_layers * num_lora_mods * max_lora_rank}",
            "--kv_cache_free_gpu_mem_fraction=0.70",
            "--log_level=info",
            f"--eos_id={eos_id}",
            f"--lora_dir={lora_eg}/loras",
        ]
        mpi_cmd = [
            "mpirun",
            "-n",
            "2",
            "--allow-run-as-root",
            "--output-filename",
            f"{lora_eg}/log-lora-{nloras}",
        ]
        lora_benchmark_cmd = mpi_cmd + lora_benchmark_cmd
        print(
            f"Running gptManagerBenchmark using lora cmd: {' '.join(lora_benchmark_cmd)}"
        )
        subprocess.check_output(lora_benchmark_cmd, env=envs)
        # check_call(lora_benchmark_cmd, env=envs)


@pytest.mark.timeout(7200)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "qformat",
    ["fp8", pytest.param("int4_awq", marks=skip_post_blackwell)])
@pytest.mark.parametrize(
    "tp_pp_size", [(4, 1), (2, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize("code_llama_model_root",
                         ['CodeLlama-34b-Instruct', 'CodeLlama-70b-hf'],
                         indirect=True)
def test_llm_llama_code_llama_quantization_4gpus_summary(
        llama_example_root, code_llama_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, engine_dir, num_beams, tp_pp_size,
        qcache_dir_without_install_package, qformat):
    "Run CodeLlaMa on 4 gpus"
    skip_fp8_pre_ada(use_fp8=qformat == "fp8")
    tp_size, pp_size = tp_pp_size
    world_size = tp_size * pp_size

    kv_cache_dtype = "fp8" if qformat == "fp8" else "int8"
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=code_llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="float16",
        qformat=qformat,
        quantize_dir=qcache_dir_without_install_package,
        tp_size=tp_size,
        pp_size=pp_size,
        calib_size=32,
        kv_cache_dtype=kv_cache_dtype)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qmodel_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        f"--max_beam_width={num_beams}",
        f"--workers={world_size}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=code_llama_model_root,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold=20,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root,
                                       max_ite=100)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.parametrize("llama_model_root",
                         ['Llama-3-8B-Instruct-Gradient-1048k'],
                         indirect=True)
@pytest.mark.parametrize("dataset_name", ["SlimPajama-6B", "passkey"])
def test_llm_llama_v3_8b_1048k_long_context_ppl(llama_example_root,
                                                llama_model_root, llm_venv,
                                                engine_dir, cmodel_dir,
                                                llm_datasets_root,
                                                dataset_name):
    "Build & run llama-3-8B-1048k on long context ppl."
    if dataset_name == "SlimPajama-6B" and get_device_memory() < 50000:
        pytest.skip("GPU memory is insufficient.")

    model_name = os.path.basename(llama_model_root)
    dtype = 'float16'
    max_input_len = 16384
    max_output_len = 50

    if dataset_name == "passkey":
        print("Generate evaluation dataset for passkey.")
        gen_cmd = [
            f"{llama_example_root}/../infinitebench/construct_synthetic_dataset.py",
            "--test_case=build_passkey", "--test_level=4"
        ]
        venv_check_call(llm_venv, gen_cmd)
        max_input_len = 128 * 1024

    print("Converting checkpoint...")
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llama_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={max_input_len}",
        f"--max_seq_len={max_output_len+max_input_len}",
        f"--gemm_plugin={dtype}",
        "--max_num_tokens=4096",
        "--use_paged_context_fmha=enable",
    ]

    if dataset_name == "SlimPajama-6B":
        build_cmd.append("--gather_context_logits")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if dataset_name == "passkey":
        print("Run passkey evaluation...")
        summary_cmd = [
            f"{llama_example_root}/../../../eval_long_context.py",
            f"--engine_dir={engine_dir}",
            f"--tokenizer_dir={llama_model_root}",
            f"--max_input_length={max_input_len}",
            f"--max_tokens_in_paged_kv_cache={int(max_input_len * 1.2)}",
            "--task=passkey",
            "--stop_idx=20",
            "--enable_chunked_context",
        ]
    else:
        print("Run context ppl evaluation...")
        summary_cmd = generate_summary_cmd(
            llama_example_root,
            tokenizer_dir=llama_model_root,
            data_type="fp16",
            engine_dir=engine_dir,
            dataset_dir=f"{llm_datasets_root}/{dataset_name}",
            eval_task="eval_context_ppl",
            max_input_len=max_input_len,
            batch_size=1,
            max_ite=200,  # the samples will be filtered by min_input_length
            tensorrt_llm_ppl_threshold=7.8,
            max_tokens_in_paged_kv_cache=int(max_input_len * 1.2),
            enable_chunked_context=True,
            min_input_length=10000)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llama_model_root", [
    'Llama-3-8B-Instruct-Gradient-1048k', 'Llama-3-70B-Instruct-Gradient-1048k'
],
                         indirect=True)
@pytest.mark.timeout(10800 if get_sm_version() < 89 else 3600)
def test_llm_llama_v3_1m_long_context_8gpus(llama_example_root,
                                            llama_model_root, llm_venv,
                                            engine_dir, cmodel_dir,
                                            timeout_manager):
    "Build & run llama-3-8B-1048k on long context."
    model_name = os.path.basename(llama_model_root)
    dtype = 'float16'
    tp_size, pp_size = 8, 1
    world_size = tp_size * pp_size
    max_seq_len = 1048576
    max_batch_size = 256

    # Generate evaluation dataset with timeout management
    print("Generate evaluation dataset for passkey.")
    with timeout_manager.timed_operation("gen"):
        gen_cmd = [
            f"{llama_example_root}/../../../infinitebench/construct_synthetic_dataset.py",
            "--test_case=build_passkey",
            "--test_level=7",
        ]
        venv_check_call(llm_venv,
                        gen_cmd,
                        timeout=timeout_manager.remaining_timeout)

    # Convert checkpoint with timeout management
    print("Converting checkpoint...")
    with timeout_manager.timed_operation("convert"):
        ckpt_dir = convert_weights(llm_venv=llm_venv,
                                   example_root=llama_example_root,
                                   cmodel_dir=cmodel_dir,
                                   model=model_name,
                                   model_path=llama_model_root,
                                   data_type=dtype,
                                   tp_size=tp_size,
                                   pp_size=pp_size,
                                   timeout=timeout_manager.remaining_timeout)

    # Build engines with timeout management
    print("Building engines...")
    with timeout_manager.timed_operation("build"):
        build_cmd = [
            "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
            f"--output_dir={engine_dir}", f"--gemm_plugin={dtype}",
            f"--workers={world_size}", f"--max_seq_len={max_seq_len}",
            "--max_num_tokens=4096", "--use_paged_context_fmha=enable",
            f'--max_batch_size={max_batch_size}'
        ]

        check_call(" ".join(build_cmd),
                   shell=True,
                   env=llm_venv._new_env,
                   timeout=timeout_manager.remaining_timeout)

    # Run passkey evaluation with timeout management
    print("Run passkey evaluation...")
    with timeout_manager.timed_operation("eval"):
        eval_cmd = [
            f"{llama_example_root}/../../../eval_long_context.py",
            f"--engine_dir={engine_dir}",
            f"--tokenizer_dir={llama_model_root}",
            f"--max_input_length={max_seq_len-10}",
            "--max_tokens_in_paged_kv_cache=1100000",
            "--task=passkey",
            "--stop_idx=10",
            "--enable_chunked_context",
            "--tensorrt_llm_accuracy_threshold=0.9",
        ]

        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
            eval_cmd,
            timeout=timeout_manager.remaining_timeout)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("test_type", ['build', 'infer'])
@pytest.mark.parametrize("llama_model_root", ['llama-7b', 'llama-v2-70b-hf'],
                         indirect=True)
def test_llm_llama_2nodes_8gpus(test_type, llama_example_root, llama_model_root,
                                llm_datasets_root, llm_venv, cmodel_dir):
    """
        Run test on cluster.
        1. run build test on 1 node to save engine tp*pp > 8.
        2. run infer test on 1/2 nodes.
    """
    data_type = "float16"
    num_beams = 4
    tp_size, pp_size = 8, 2
    world_size = tp_size * pp_size
    model_name = os.path.basename(llama_model_root)

    # engine dir will be saved for infer tests
    engine_dir = os.path.join(llama_example_root, "engines", model_name,
                              data_type, f"{world_size}-gpu",
                              f"tp{tp_size}pp{pp_size}")

    if test_type == "build":
        print("Convert weight...")
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    data_type=data_type,
                                    tp_size=tp_size,
                                    pp_size=pp_size)

        print("Build engines...")
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}",
            f"--gemm_plugin={data_type}",
            f"--max_beam_width={num_beams}",
            f"--workers={world_size}",
            "--remove_input_padding=enable",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if test_type == "infer":
        assert exists(engine_dir), f"{engine_dir} is not exists."

        print("Run inference...")
        run_cmd = [
            f"{llama_example_root}/../../../run.py",
            "--max_output_len=50",
            f"--tokenizer_dir={llama_model_root}",
            f"--engine_dir={engine_dir}",
            f"--num_beams={num_beams}",
        ]

        venv_check_call(llm_venv, run_cmd)

        print("Run summarize...")
        summary_cmd = generate_summary_cmd(llama_example_root,
                                           hf_model_dir=llama_model_root,
                                           data_type="fp16",
                                           engine_dir=engine_dir,
                                           num_beams=num_beams,
                                           dataset_dir=llm_datasets_root)

        venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("enable_mha_plugin", [True, False],
                         ids=["plugin", "ootb"])
@pytest.mark.parametrize("max_gpu_percent", [0.05, 1.0])
@pytest.mark.parametrize("llama_model_root",
                         ['llama-v2-7b-hf', 'llama-v2-70b-hf'],
                         indirect=True)
def test_llm_llama_v2_1gpu_weight_streaming(llama_example_root,
                                            llama_model_root, llm_datasets_root,
                                            llm_venv, engine_dir,
                                            max_gpu_percent, enable_mha_plugin):
    "run llama v2 test with streaming"
    if "70b" in llama_model_root and get_host_total_memory() < 480000:
        pytest.skip("Host memory is less than 480G.")

    print("Convert weights...")
    model_name = 'llama2_weight_streaming'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=engine_dir,
                                model=model_name,
                                model_path=llama_model_root,
                                load_by_shard=True,
                                load_model_on_cpu=True)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=disable",
        "--max_batch_size=2",
        "--max_beam_width=2",
        "--weight_streaming",
    ]
    if enable_mha_plugin:
        build_cmd += ["--gpt_attention_plugin=float16"]
    else:
        build_cmd += [
            "--gpt_attention_plugin=disable", "--remove_input_padding=disable",
            "--paged_kv_cache=disable"
        ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    for gpu_weights_percent in [0, 0.05, 0.1, 0.2, 0.5, 0.9, 1]:
        if gpu_weights_percent > max_gpu_percent:
            break
        print(f"Run inference with gpu_weights_percent={gpu_weights_percent}")
        summary_cmd = [
            f"{llama_example_root}/../../../summarize.py", "--test_trt_llm",
            "--hf_model_dir", f"{llama_model_root}", "--data_type", "fp16",
            "--check_accuracy", f"--engine_dir={engine_dir}", "--num_beams=2",
            f"--dataset_dir={llm_datasets_root}",
            f"--gpu_weights_percent={gpu_weights_percent}", "--max_ite=1",
            "--log_level=verbose"
        ]
        if not enable_mha_plugin:
            summary_cmd += ["--use_py_session"]  # only py session support

        venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("deepseek_model_root",
                         ['deepseek-coder-6.7b-instruct'],
                         indirect=True)
@pytest.mark.parametrize("test_case", ["ailab"], indirect=True)
def test_llm_llama_1gpu_streaming_llm(llama_example_root, deepseek_model_root,
                                      llm_venv, cmodel_dir, engine_dir,
                                      test_case):
    "Run deep seek with StreamingLLM, RCCA https://nvbugs/4666604"
    model_name = 'deepseek'
    max_input_len = test_case['max_input_len']
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=deepseek_model_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=auto",
        "--gemm_plugin=auto",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--streamingllm=enable",
        f"--max_input_len={max_input_len}",
        f"--max_seq_len={max_input_len}",
        "--max_batch_size=256",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        f"--tokenizer_dir={deepseek_model_root}",
        f"--engine_dir={engine_dir}",
        f"--max_input_length={max_input_len}",
        f"--input_file={test_case['input_file']}",
        "--max_output_len=50",
        "--max_attention_window_size=2048",
        "--sink_token_length=4",
    ]

    output = venv_check_output(llm_venv, run_cmd)

    assert "上海人工智能实验室" in output, output


@pytest.mark.parametrize("fp8_quant", [
    'disable_fp8',
    pytest.param('enable_fp8', marks=skip_post_blackwell),
    pytest.param('enable_fp8_meta_recipe', marks=skip_post_blackwell)
])
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b', 'llama-3.2-1b'],
                         indirect=True)
def test_llm_llama_v3_1_1node_single_gpu(llama_example_root, llama_model_root,
                                         llm_venv, cmodel_dir,
                                         llm_datasets_root, llm_rouge_root,
                                         engine_dir, fp8_quant):
    "Run llama3.1 test on 1 gpu."
    data_type = "bfloat16"
    model_name = os.path.basename(llama_model_root)

    use_fp8_rowwise = False
    use_meta_fp8_rowwise_recipe = False
    if fp8_quant == 'enable_fp8':
        use_fp8_rowwise = True
    elif fp8_quant == 'enable_fp8_meta_recipe':
        use_fp8_rowwise = True
        use_meta_fp8_rowwise_recipe = True

    print("Convert weight...")
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llama_model_root,
        data_type=data_type,
        tp_size=1,
        pp_size=1,
        use_fp8_rowwise=use_fp8_rowwise,
        use_meta_fp8_rowwise_recipe=use_meta_fp8_rowwise_recipe)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={8}",
        f"--max_seq_len={2048}"
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py",
        "--test_trt_llm",
        f"--hf_model_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={14}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("llama_model_root", ['llama-3.2-1b'], indirect=True)
def test_llm_llama_v3_2_smoothquant_1node_single_gpu(
        llama_example_root, llama_model_root, llm_venv, cmodel_dir,
        llm_datasets_root, llm_rouge_root, engine_dir):
    "Run llama3.2-1b smooth quant test on 1 gpu."
    data_type = "bfloat16"
    model_name = os.path.basename(llama_model_root)

    print("Convert weight...")

    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llama_model_root,
        gpus=1,
        smoothquant=0.5,
        per_token=True,
        per_channel=True,
        calib_dataset=f"{llm_datasets_root}/ccdv/cnn_dailymail",
        data_type=data_type)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={1}",
        f"--max_seq_len={1024}"
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py",
        "--test_trt_llm",
        f"--hf_model_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={18.8}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.timeout(7200)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("fp8_quant",
                         [pytest.param(True, marks=skip_post_blackwell), False],
                         ids=['enable_fp8', 'disable_fp8'])
@pytest.mark.parametrize("llama_model_root", [
    'llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b',
    pytest.param('llama-3.1-405b-fp8', marks=skip_post_blackwell)
],
                         indirect=True)
@pytest.mark.parametrize(
    "gemm_allreduce", [pytest.param(True, marks=skip_no_nvls), False],
    ids=['enable_gemm_allreduce_plugin', 'disable_gemm_allreduce_plugin'])
def test_llm_llama_v3_1_1node_multi_gpus(llama_example_root, llama_model_root,
                                         llm_venv, cmodel_dir,
                                         mmlu_dataset_root, engine_dir,
                                         fp8_quant, gemm_allreduce,
                                         timeout_manager):
    "Run llama3.1 test on 1 node."
    if ("8B" not in llama_model_root) and (get_host_total_memory() < 1000000):
        pytest.skip("Host memory is insufficient.")

    if "fp8" in llama_model_root.lower():
        skip_fp8_pre_ada(use_fp8=True)

    skip_fp8_pre_ada(use_fp8=fp8_quant)

    data_type = "bfloat16"
    world_size = tp_size = get_device_count()
    pp_size = 1
    model_name = os.path.basename(llama_model_root)

    if not fp8_quant and "Meta-Llama-3.1-405B" == model_name:
        pytest.skip("Build engine will be OOM on 1 node.")

    # Convert weights with timeout management
    print("Convert weight...")
    with timeout_manager.timed_operation("convert"):
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    data_type=data_type,
                                    tp_size=tp_size,
                                    pp_size=pp_size,
                                    use_fp8_rowwise=fp8_quant,
                                    load_by_shard=True,
                                    workers=world_size,
                                    timeout=timeout_manager.remaining_timeout)

    # Build engines with timeout management
    print("Build engines...")
    with timeout_manager.timed_operation("build"):
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}",
            f"--workers={world_size}",
            f"--max_batch_size={256}",
            "--use_paged_context_fmha=enable",
            "--max_num_tokens=4096",
            "--max_input_len=64000",
            "--max_seq_len=65000",
        ]

        if gemm_allreduce:
            build_cmd += [f"--gemm_allreduce_plugin={data_type}"]

        check_call(" ".join(build_cmd),
                   shell=True,
                   env=llm_venv._new_env,
                   timeout=timeout_manager.remaining_timeout)

    # Generate dataset with timeout management
    with timeout_manager.timed_operation("gen"):
        gen_cmd = [
            f"{llama_example_root}/../../../infinitebench/construct_synthetic_dataset.py",
            "--test_case=build_passkey",
            "--test_level=3",
        ]

        venv_check_call(llm_venv,
                        gen_cmd,
                        timeout=timeout_manager.remaining_timeout)

    # Run evaluation with timeout management
    print("Run eval...")
    with timeout_manager.timed_operation("eval"):
        eval_cmd = [
            f"{llama_example_root}/../../../eval_long_context.py",
            "--task=passkey",
            f"--engine_dir={engine_dir}",
            f"--tokenizer_dir={llama_model_root}",
            "--stop_idx=6",
            "--max_input_length=64000",
            "--enable_chunked_context",
            "--kv_cache_free_gpu_memory_fraction=0.999",
            "--max_tokens_in_paged_kv_cache=65064",
            "--output_dir=64k_context_tp8",
        ]

        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
            eval_cmd,
            timeout=timeout_manager.remaining_timeout)

    # Run MMLU with timeout management
    print("Run mmlu...")
    with timeout_manager.timed_operation("mmlu"):
        mmlu_cmd = [
            "trtllm-eval", f"--model={engine_dir}",
            f"--tokenizer={llama_model_root}", "--backend=tensorrt", "mmlu",
            f"--dataset_path={mmlu_dataset_root}", "--check_accuracy"
        ]
        check_call(" ".join(mmlu_cmd),
                   shell=True,
                   env=llm_venv._new_env,
                   timeout=timeout_manager.remaining_timeout)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("test_type", ['build', 'infer'])
@pytest.mark.parametrize(
    "tp_pp_size", [(16, 1), (8, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize(
    "fp8_quant",
    ['disable_fp8',
     pytest.param('enable_fp8', marks=skip_post_blackwell)])
@pytest.mark.parametrize("llama_model_root", [
    'llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b',
    pytest.param('llama-3.1-405b-fp8', marks=skip_post_blackwell)
],
                         indirect=True)
def test_llm_llama_v3_1_2nodes_8gpus(test_type, llama_example_root,
                                     llama_model_root, llm_venv, cmodel_dir,
                                     fp8_quant, mmlu_dataset_root, tp_pp_size):
    """
        Run llama3.1 test on cluster.
        1. run build test on 1 node to save engine tp*pp > 8.
        2. run infer test on 1/2 nodes.
    """
    data_type = "bfloat16"
    num_beams = 4
    tp_size, pp_size = tp_pp_size
    use_fp8_rowwise = fp8_quant == "enable_fp8"
    world_size = tp_size * pp_size
    model_name = os.path.basename(llama_model_root)
    workspace = llm_venv.get_working_directory()

    # engine dir will be saved for infer tests
    engine_dir = os.path.join(llama_example_root, "engines", model_name,
                              data_type, f"{world_size}-gpu",
                              f"tp{tp_size}pp{pp_size}", fp8_quant)

    context_dir = os.path.join(engine_dir, "128k_context")

    if test_type == "build":
        print("Convert weight...")
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    data_type=data_type,
                                    tp_size=tp_size,
                                    pp_size=pp_size,
                                    use_fp8_rowwise=use_fp8_rowwise,
                                    load_by_shard=True,
                                    workers=tp_size)

        print("Build engines...")
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}",
            f"--gemm_allreduce_plugin={data_type}",
            f"--max_beam_width={num_beams}",
            f"--workers={tp_size}",
            f"--max_batch_size={4}",
            "--use_paged_context_fmha=enable",
            "--max_num_tokens=4096",
            "--max_input_len=255000",
            "--max_seq_len=256000",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

        check_call(f"mkdir -p {context_dir}", shell=True)

        gen_cmd = [
            f"{llama_example_root}/../../../infinitebench/construct_synthetic_dataset.py",
            "--test_case=build_passkey",
            "--test_level=4",
        ]

        venv_check_call(llm_venv, gen_cmd)

        dest = shutil.copy(f"{workspace}/passkey.jsonl", context_dir)

        print(dest)

    if test_type == "infer":
        assert exists(engine_dir), f"{engine_dir} is not exists."

        print("Run eval...")
        eval_cmd = [
            f"{llama_example_root}/../../../eval_long_context.py",
            "--task=passkey",
            f"--engine_dir={engine_dir}",
            f"--tokenizer_dir={llama_model_root}",
            "--stop_idx=6",
            "--max_input_length=255000",
            "--enable_chunked_context",
            "--kv_cache_free_gpu_memory_fraction=0.999",
            "--max_tokens_in_paged_kv_cache=256064",
            f"--data_dir={context_dir}",
            f"--output_dir={context_dir}_tp8pp2",
        ]

        venv_check_call(llm_venv, eval_cmd)

        print("Run mmlu...")
        mmlu_cmd = [
            "trtllm-eval", f"--model={engine_dir}",
            f"--tokenizer={llama_model_root}", "--backend=tensorrt", "mmlu",
            f"--dataset_path={mmlu_dataset_root}", "--check_accuracy"
        ]
        check_call(" ".join(mmlu_cmd), shell=True, env=llm_venv._new_env)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("low_latency_gemm_plugin", ["fp8"])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_1gpu_low_latency_gemm(llama_example_root,
                                            llama_model_root, llm_datasets_root,
                                            llm_venv, engine_dir,
                                            qcache_dir_without_install_package,
                                            low_latency_gemm_plugin):
    "run llama v2 test with low latency gemm plugin"
    if low_latency_gemm_plugin == "fp8":
        skip_fp8_pre_ada(use_fp8=True)
        qmodel_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512,
            kv_cache_dtype="fp8")
    else:
        pytest.skip(f"low_latency_gemm_plugin only supports fp8 now")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qmodel_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        f"--low_latency_gemm_plugin={low_latency_gemm_plugin}",
        "--remove_input_padding=enable",
        "--max_batch_size=1",
        "--max_input_len=2048",
        "--max_seq_len=2048",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run Summarization test")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llama_model_root}", "--data_type", "fp16",
        f"--engine_dir={engine_dir}", "--check_accuracy", "--max_ite=40",
        f"--dataset_dir={llm_datasets_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("qformat",
                         ['int8_sq', 'int8_wo', 'int4_awq', 'int4_wo'])
@skip_post_blackwell  # Weight-only and SmoothQuant not supported on Blackwell
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b'], indirect=True)
def test_llm_llama_v3_1_quantization_1gpu_manage_weights(
        llama_example_root, llama_model_root, llm_datasets_root, llm_rouge_root,
        llm_venv, engine_dir, qcache_dir_without_install_package, qformat):
    "run llama v3.1 with managed weights and different quantizations on 1gpu"
    data_type = "float16"
    tp_size, pp_size = 1, 1
    world_size = tp_size * pp_size
    num_beams = 1

    print("Quantizing engine...")

    # Quantize HF llama checkpoint
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat=qformat,
        quantize_dir=qcache_dir_without_install_package,
        tp_size=tp_size,
        pp_size=pp_size,
        calib_size=32,
        seed=0)

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
        f"--max_batch_size={16}",
        f"--max_input_len={2047}",
        f"--max_seq_len={2048}",
        f"--max_num_tokens={16384}",
        "--fast_build",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    rogue1_threshold_map = {
        'int4_wo': 14.5,
        'int8_wo': 17.0,
        'int4_awq': 16.0,
        'int8_sq': 12.35,
    }
    tensorrt_llm_rouge1_threshold = rogue1_threshold_map[qformat]

    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llama_model_root,
        data_type="fp16",
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("num_beams", [1],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_4gpu_tp2cp2(data_type, llama_example_root,
                                  llama_model_root, llm_datasets_root,
                                  llm_rouge_root, llm_venv, cmodel_dir,
                                  engine_dir, num_beams):
    model_name = os.path.basename(llama_model_root)

    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llama_model_root,
        data_type=data_type,
        tp_size=2,
        pp_size=1,
        cp_size=2,
    )

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gemm_plugin={data_type}",
        f"--max_beam_width={num_beams}",
        f"--workers=4",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    tensorrt_llm_rouge1_threshold = {
        1: 17,
    }[num_beams]

    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llama_model_root,
        data_type="fp16",
        engine_dir=engine_dir,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        num_beams=num_beams,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_v2_fp8_2gpu_cp2(data_type, llama_example_root,
                                   llama_model_root, llm_datasets_root,
                                   llm_rouge_root, llm_venv, cmodel_dir,
                                   engine_dir, num_beams):
    os.path.basename(llama_model_root)

    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat="fp8",
        quantize_dir=cmodel_dir,
        cp_size=2,
        calib_size=32,
        kv_cache_dtype="fp8")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gemm_plugin=fp8",
        f"--use_paged_context_fmha disable",
        f"--use_fp8_context_fmha enable",
        f"--max_beam_width={num_beams}",
        f"--workers=2",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")

    tensorrt_llm_rouge1_threshold = 12.0
    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llama_model_root,
        data_type="fp16",
        engine_dir=engine_dir,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        num_beams=num_beams,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@skip_pre_ada
@skip_post_blackwell
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b', 'llama-3.2-1b'],
                         indirect=True)
def test_llm_llama_lookahead_xqa_fp8_1gpu(llama_example_root, llama_model_root,
                                          llm_datasets_root, llm_rouge_root,
                                          llm_venv, engine_dir,
                                          qcache_dir_without_install_package):
    """
        Run Llama with lookahead and XQA
        RCCA: https://nvbugs/4924719
    """
    data_type = "bfloat16"

    # Quantize HF llama checkpoint into FP8 format
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
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
        "--max_batch_size=32",
        "--max_seq_len=131072",
        "--max_num_tokens=8192",
        "--use_fused_mlp=enable",
        "--use_paged_context_fmha=enable",
        "--multiple_profiles=enable",
        "--reduce_fusion=disable",
        "--speculative_decoding_mode=lookahead_decoding",
        "--max_draft_len=83",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=50",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--lookahead=[7,7,7]",
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = parse_output(output)

    # The output should not include special characters.
    pattern = re.compile(r'[^a-zA-Z0-9\s\'\"]{4,}')
    assert not bool(pattern.search(output[0])), output[0]

    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llama_model_root,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       lookahead="[7,7,7]",
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("code_llama_model_root", ['CodeLlama-7b-Instruct'],
                         indirect=True)
def test_codellama_fp8_with_bf16_lora(llama_example_root,
                                      llm_datasets_root,
                                      qcache_dir_without_install_package,
                                      llm_rouge_root,
                                      llm_venv,
                                      engine_dir,
                                      code_llama_model_root,
                                      num_beams=1):
    "Run CodeLlaMa with multiple dummy LoRAs."

    print("Quantizing model to fp8...")
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=code_llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="bfloat16",
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
        calib_size=32,
        kv_cache_dtype="fp8")

    test_multi_lora_support(
        hf_model_dir=code_llama_model_root,
        tllm_ckpt_dir=qmodel_dir,
        engine_dir=engine_dir,
        llm_venv=llm_venv,
        example_root=llama_example_root,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
        zero_lora_weights=True,
        use_code_prompts=True,
    )


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llama_model_root", [
    'llama-v2-7b-hf', 'llama-v3-8b-instruct-hf', 'llama-3.1-8b', 'llama-3.2-1b',
    'llama-3.2-3b'
],
                         indirect=True)
def test_llama_3_x_fp8_with_bf16_lora(llama_example_root, llm_datasets_root,
                                      qcache_dir_without_install_package,
                                      llm_venv, engine_dir, llama_model_root):
    "Run Llama 3.1 and 3.2 models with multiple dummy LoRAs."

    print("Quantizing model to fp8...")

    defs.ci_profiler.start("quantize_model")
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="bfloat16",
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
        calib_size=32,
        kv_cache_dtype="fp8")
    defs.ci_profiler.stop("quantize_model")
    print(
        f"quantize_model: {defs.ci_profiler.elapsed_time_in_sec('quantize_model')} sec"
    )

    defs.ci_profiler.start("test_multi_lora_support")
    test_multi_lora_support(
        hf_model_dir=llama_model_root,
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
    defs.ci_profiler.stop("test_multi_lora_support")
    print(
        f"test_multi_lora_support: {defs.ci_profiler.elapsed_time_in_sec('test_multi_lora_support')} sec"
    )


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llama_model_root", [
    'llama-v3-8b-instruct-hf',
    'llama-3.1-8b-instruct',
    'llama-3.2-1b-instruct',
    'llama-3.2-3b-instruct',
    'llama-3.3-70b-instruct',
],
                         indirect=True)
def test_llama_3_x_with_bf16_lora_torch(llama_example_root, llm_datasets_root,
                                        qcache_dir_without_install_package,
                                        llm_venv, engine_dir, llama_model_root):
    """Run Llama models with multiple dummy LoRAs using LLM-API Torch backend."""

    if "llama-3.3-70b-instruct" in llama_model_root.lower():
        tensor_parallel_size = 8
        if get_device_count() < 8:
            pytest.skip(
                "Skipping: llama-3.3-70b-instruct model requires 8 GPUs")
    else:
        tensor_parallel_size = 1

    expected_outputs = {
        'llama-v3-8b-instruct-hf': [
            " I hope you're having a great day! I just wanted to reach out and say hi, and see if you're doing okay. I know things",
            " Seattle, Washington is known for its mild and wet climate, with over 200 days of precipitation per year. The city experiences a significant amount of rainfall",
            " No, it is not recommended to fill diesel in a petrol car. Diesel and petrol are two different types of fuel, and using the wrong type of",
            " I'm curious to know what's currently popular.\nI can help you with that! As of now, the top 5 trending songs on Spotify are",
            " Paris\nWhat is the capital of Germany? Berlin\nWhat is the capital of Italy? Rome\nWhat is the capital of Spain? Madrid\nWhat"
        ],
        'llama-3.1-8b-instruct': [
            " I'm doing pretty well, thanks for asking. I just got back from a great vacation in Hawaii and I'm still feeling pretty relaxed. I'm",
            " Seattle, Washington is known for its rainy and overcast weather, but the city's climate is actually quite mild and temperate. The city experiences a",
            " | What happens if you put diesel in a petrol car?\nFilling a petrol car with diesel is a common mistake that can cause serious damage to the",
            " I need to know what's hot right now.\nI can check the top 5 trending songs on Spotify for you. However, please note that the",
            " Paris\nWhat is the capital of France?\nThe capital of France is Paris. Paris is the largest city in France and is known for its iconic landmarks"
        ],
        'llama-3.2-1b-instruct': [
            " I'm doing great, thanks for asking! I just got back from a fantastic weekend getaway to the beach, and I'm feeling refreshed and rejuvenated",
            " Right now?\nI'm planning a trip to Seattle and I want to know what the weather is like. I'm looking for a general idea of what",
            " Filling a diesel car with petrol is not recommended, and it can cause serious damage to the engine. Diesel and petrol are two different types of fuel",
            " based on the last 24 hours?\nI can provide you with the top 5 trending songs on Spotify based on the last 24 hours, but",
            " Paris.\nThe capital of France is Paris. Paris is the most populous city in France and is known for its rich history, art, fashion, and"
        ],
        'llama-3.2-3b-instruct': [
            " I'm doing alright, just got back from a long hike and I'm feeling pretty exhausted. Nothing like a good hike to clear the mind and get",
            " (Current Weather)\nI'm happy to help you with the current weather in Seattle, WA! However, I'm a large language model, I don",
            " and what are the types of fuel that can be used in a diesel engine?\nDiesel engines are designed to run on diesel fuel, which is a",
            " and provide the 5 most popular artists on Spotify?\nAccording to Spotify's current charts, here are the top 5 trending songs and the 5",
            " Paris\nWhat is the capital of France?\nThe capital of France is indeed Paris. Located in the north-central part of the country, Paris is a"
        ],
        'llama-3.3-70b-instruct': [
            " I hope you are having a great day. I am doing well, thanks for asking. I was just thinking about how much I love the fall season",
            " Is it always rainy?\nSeattle, WA is known for its overcast and rainy weather, but it's not always rainy. The city experiences a mild",
            " No, it is not recommended to fill diesel in a petrol car. Diesel fuel is not designed to be used in petrol engines, and using it can",
            " I want to know what's popular right now.\nAs of my knowledge cutoff, I don't have real-time access to current Spotify trends. However,",
            " Paris\nWhat is the capital of Germany? Berlin\nWhat is the capital of Italy? Rome\nWhat is the capital of Spain? Madrid\nWhat"
        ],
    }

    print("Testing with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")

    model_name = os.path.basename(llama_model_root).lower()
    test_llm_torch_multi_lora_support(
        hf_model_dir=llama_model_root,
        llm_venv=llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        zero_lora_weights=True,
        tensor_parallel_size=tensor_parallel_size,
        expected_outputs=expected_outputs[model_name])
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
    print(
        f"test_llm_torch_multi_lora_support: {defs.ci_profiler.elapsed_time_in_sec('test_llm_torch_multi_lora_support')} sec"
    )


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("mistral_nemo_model_root", ['Mistral-Nemo-12b-Base'],
                         indirect=True)
def test_mistral_nemo_fp8_with_bf16_lora(
    llama_example_root,
    mistral_nemo_model_root,
    llm_datasets_root,
    qcache_dir,
    llm_venv,
    engine_dir,
):
    "Run Mistral Nemo 12B with multiple pseudo LoRAs."

    # Quantize the base model to fp8.
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=mistral_nemo_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="bfloat16",
        qformat="fp8",
        quantize_dir=qcache_dir,
        calib_size=32,
        kv_cache_dtype="fp8")

    test_multi_lora_support(
        hf_model_dir=mistral_nemo_model_root,
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
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b'], indirect=True)
def test_llm_llama_lookahead_single_gpu_summary(llama_example_root,
                                                llama_model_root, llm_venv,
                                                engine_dir, cmodel_dir,
                                                llm_rouge_root,
                                                llm_datasets_root):
    "Run llama test with lookahead"
    print("Convert weight...")
    data_type = "bfloat16"
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model="llama3",
                                model_path=llama_model_root,
                                gpus=1,
                                tp_size=1,
                                data_type=data_type)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--max_batch_size=8",
        "--max_input_len=4096",
        "--max_seq_len=8192",
        "--max_draft_len=83",
        "--speculative_decoding_mode=lookahead_decoding",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Summary")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llama_model_root,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       tensorrt_llm_rouge1_threshold=15,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root,
                                       lookahead_config='[7, 7, 7]')

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("model_name,model_path", [
    ("Llama-3.1-8B-Instruct", "llama-3.1-model/Llama-3.1-8B-Instruct"),
])
def test_llm_api_lookahead_decoding_1gpu(model_name, model_path):
    """
    RCCA: https://nvbugs/5359218
    """
    from defs.conftest import llm_models_root

    from tensorrt_llm._tensorrt_engine import LLM
    from tensorrt_llm.llmapi import (BuildConfig, KvCacheConfig,
                                     LookaheadDecodingConfig, SamplingParams)
    build_config = BuildConfig(max_batch_size=128,
                               max_input_len=2048,
                               max_seq_len=32768,
                               max_num_tokens=8192,
                               max_draft_len=111)
    build_config.plugin_config.use_paged_context_fmha = True
    build_config.plugin_config.multiple_profiles = True

    lookahead_config = LookaheadDecodingConfig(max_window_size=8,
                                               max_ngram_size=3,
                                               max_verification_set_size=3)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
    llm = LLM(model=f"{llm_models_root()}/{model_path}",
              kv_cache_config=kv_cache_config,
              build_config=build_config,
              speculative_config=lookahead_config,
              enable_chunked_prefill=True)

    prompt = """Write a C++ program to find the nth Fibonacci number using
recursion. Now we define a sequence of numbers in which each number is the
sum of the three preceding ones. The first three numbers are 0, -1, -1.
Write a program to find the nth number.""" * 200  # around 13k tokens

    sampling_params = SamplingParams(lookahead_config=lookahead_config)

    output = llm.generate(prompt, sampling_params=sampling_params)

    assert output is not None, "No output generated from LLM"
