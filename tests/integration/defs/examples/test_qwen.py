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
"""Module test_qwen test qwen examples."""

import csv
import os

import pytest
from defs.common import (convert_weights, test_multi_lora_support,
                         venv_check_call, venv_mpi_check_call)
from defs.conftest import (get_device_count, get_device_memory, get_sm_version,
                           skip_post_blackwell, skip_pre_ada)
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_fmha', 'disable_fmha', 'enable_fmha_fp32_acc'])
@pytest.mark.parametrize("use_weight_only",
                         [pytest.param(True, marks=skip_post_blackwell), False],
                         ids=["enable_weight_only", "disable_weight_only"])
@pytest.mark.parametrize(
    "remove_input_padding", [True, False],
    ids=["enable_remove_input_padding", "disable_remove_input_padding"])
@pytest.mark.parametrize(
    "paged_kv_cache", [True, False],
    ids=["enable_paged_kv_cache", "disable_paged_kv_cache"])
@pytest.mark.parametrize("llm_qwen_model_root", [
    "qwen2.5_7b_instruct", "qwen_7b_chat", "qwen1.5_0.5b_chat",
    "qwen1.5_7b_chat", "qwen2_7b_instruct", "qwen2_vl_7b_instruct",
    "qwen2_0.5b_instruct", "qwen2.5_0.5b_instruct", "qwen2.5_1.5b_instruct"
],
                         indirect=True)
def test_llm_qwen_single_gpu_summary(
    qwen_example_root,
    llm_qwen_model_root,
    llm_datasets_root,
    llm_rouge_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    context_fmha_type,
    use_weight_only,
    remove_input_padding,
    paged_kv_cache,
):
    "Build & run Qwen-7B-Chat on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_qwen_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_qwen_model_root,
                               data_type=dtype,
                               use_weight_only=use_weight_only)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        f"--max_seq_len=2048",
    ]
    if context_fmha_type == "enable_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disable_fmha":
        build_cmd.append("--context_fmha=disable")
    if remove_input_padding:
        build_cmd.append("--remove_input_padding=enable")
    else:
        build_cmd.append("--remove_input_padding=disable")
    if paged_kv_cache:
        build_cmd.append("--paged_kv_cache=enable")
    else:
        build_cmd.append("--paged_kv_cache=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_qwen_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--tensorrt_llm_rouge1_threshold=22",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha_type == "enable_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_fmha', 'disable_fmha', 'enable_fmha_fp32_acc'])
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=["enable_weight_only", "disable_weight_only"])
@pytest.mark.parametrize(
    "remove_input_padding", [True, False],
    ids=["enable_remove_input_padding", "disable_remove_input_padding"])
@pytest.mark.parametrize(
    "paged_kv_cache", [True, False],
    ids=["enable_paged_kv_cache", "disable_paged_kv_cache"])
@pytest.mark.parametrize("llm_qwen_model_root", ["qwen1.5_moe_a2.7b_chat"],
                         indirect=True)
def test_llm_qwen_moe_single_gpu_summary(
    qwen_example_root,
    llm_qwen_model_root,
    llm_datasets_root,
    llm_rouge_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    context_fmha_type,
    use_weight_only,
    remove_input_padding,
    paged_kv_cache,
):
    "Build & run Qwen1.5-MoE-A2.7B-Chat on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_qwen_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_qwen_model_root,
                               data_type=dtype,
                               use_weight_only=use_weight_only)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        f"--max_seq_len=2048",
    ]
    if context_fmha_type == "enable_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disable_fmha":
        build_cmd.append("--context_fmha=disable")
    if remove_input_padding:
        build_cmd.append("--remove_input_padding=enable")
    else:
        build_cmd.append("--remove_input_padding=disable")
    if paged_kv_cache:
        build_cmd.append("--paged_kv_cache=enable")
    else:
        build_cmd.append("--paged_kv_cache=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_qwen_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--tensorrt_llm_rouge1_threshold=22",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha_type == "enable_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


# Delete this case refer to https://nvbugs/5072417
# @pytest.mark.parametrize("llm_lora_model_root", ["Ko-QWEN-7B-Chat-LoRA"],
#                          indirect=True)
# @pytest.mark.parametrize("llm_qwen_model_root", ["qwen_7b_chat"], indirect=True)
# def test_llm_qwen_7b_single_gpu_lora(
#     qwen_example_root,
#     llm_qwen_model_root,
#     llm_venv,
#     cmodel_dir,
#     engine_dir,
#     llm_lora_model_root,
# ):
#     "run Qwen lora test on single gpu."
#     print("Build engines...")
#     dtype = 'float16'
#     model_name = os.path.basename(llm_qwen_model_root)
#     ckpt_dir = convert_weights(llm_venv=llm_venv,
#                                example_root=qwen_example_root,
#                                cmodel_dir=cmodel_dir,
#                                model=model_name,
#                                model_path=llm_qwen_model_root,
#                                data_type=dtype)

#     print("Build engines...")
#     build_cmd = [
#         "trtllm-build",
#         f"--checkpoint_dir={ckpt_dir}",
#         f"--output_dir={engine_dir}",
#         "--lora_plugin=auto",
#         "--gemm_plugin=auto",
#         f"--lora_dir={llm_lora_model_root}",
#     ]
#     check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

#     ref_1 = [
#         151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
#         151644, 872, 198, 126246, 144370, 91145, 11, 137601, 29326, 86034,
#         12802, 5140, 98734, 19391, 35711, 30, 151645, 198, 151644, 77091, 198,
#         126246, 144370, 91145, 0, 134561, 58677, 78125, 21329, 66019, 124685,
#         134619, 94152, 28626, 17380, 11, 134637, 20401, 138520, 19391, 143603
#     ]
#     ref_2 = [
#         151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
#         151644, 872, 198, 126246, 144370, 91145, 11, 137601, 29326, 86034,
#         12802, 5140, 98734, 19391, 35711, 30, 151645, 198, 151644, 77091, 198,
#         126246, 144370, 91145, 0, 134561, 330, 48, 1103, 54, 268, 1, 78952, 13,
#         151645, 198, 151643, 151643, 151643, 151643, 151643
#     ]

#     input_text = "안녕하세요, 혹시 이름이 뭐에요?"
#     print("Run inference with lora id 0...")
#     venv_check_call(llm_venv, [
#         f"{qwen_example_root}/../run.py",
#         "--max_output_len=20",
#         f"--input_text={input_text}",
#         "--lora_task_uids=0",
#         f"--tokenizer_dir={llm_qwen_model_root}",
#         f"--engine_dir={engine_dir}",
#         f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
#         "--use_py_session",
#     ])

#     with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
#         predict = csv.reader(f)
#         predict = next(predict)
#     predict = [int(p) for p in predict]
#     assert ref_1 == predict

#     print("Run inference with lora id -1...")
#     venv_check_call(llm_venv, [
#         f"{qwen_example_root}/../run.py",
#         "--max_output_len=20",
#         f"--input_text={input_text}",
#         "--lora_task_uids=-1",
#         f"--tokenizer_dir={llm_qwen_model_root}",
#         f"--engine_dir={engine_dir}",
#         f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
#         "--use_py_session",
#     ])

#     with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
#         predict = csv.reader(f)
#         predict = next(predict)
#     predict = [int(p) for p in predict]
#     assert ref_2 == predict


@pytest.mark.parametrize("llm_lora_model_root", ["Qwen1.5-7B-Chat-750Mb-lora"],
                         indirect=True)
@pytest.mark.parametrize("llm_qwen_model_root", ["qwen1.5_7b_chat"],
                         indirect=True)
def test_llm_qwen1_5_7b_single_gpu_lora(
    qwen_example_root,
    llm_qwen_model_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    llm_lora_model_root,
):
    "run Qwen1.5 lora test on single gpu."
    print("Build engines...")
    dtype = 'float16'
    model_name = os.path.basename(llm_qwen_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_qwen_model_root,
                               data_type=dtype)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--gemm_plugin=auto",
        f"--lora_dir={llm_lora_model_root}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
        151644, 872, 198, 3838, 374, 697, 829, 30, 151645, 198, 151644, 77091,
        198, 40, 2776, 458, 15235, 7881, 553, 5264, 15469, 11, 773, 358, 1513,
        944, 614, 264, 829, 304, 279, 8606, 5530
    ]
    ref_2 = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
        151644, 872, 198, 3838, 374, 697, 829, 30, 151645, 198, 151644, 77091,
        198, 40, 1079, 1207, 16948, 11, 264, 3460, 4128, 1614, 3465, 553, 54364,
        14817, 13, 151645, 151645, 151645, 151645, 151645, 151645
    ]

    input_text = "What is your name?"
    print("Run inference with lora id 0...")
    venv_check_call(llm_venv, [
        f"{qwen_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_qwen_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict

    print("Run inference with lora id -1...")
    venv_check_call(llm_venv, [
        f"{qwen_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_qwen_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_2 == predict


@pytest.mark.skip_less_device_memory(60000)
@pytest.mark.parametrize("llm_lora_model_root",
                         ["Upcycled-Qwen1.5-MoE2.7B-LoRA"],
                         indirect=True)
@pytest.mark.parametrize("llm_qwen_model_root", ["qwen1.5_moe_a2.7b_chat"],
                         indirect=True)
def test_llm_qwen1_5_moe_single_gpu_lora(
    qwen_example_root,
    llm_qwen_model_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    llm_lora_model_root,
):
    "run Qwen1.5 MoE lora test on single gpu."
    print("Build engines...")
    dtype = 'float16'
    model_name = os.path.basename(llm_qwen_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_qwen_model_root,
                               data_type=dtype)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--moe_plugin=disable",
        f"--lora_dir={llm_lora_model_root}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
        151644, 872, 198, 3838, 374, 697, 829, 30, 151645, 198, 151644, 77091,
        198, 151935, 151935, 151935, 151935, 151935, 151935, 151935, 151935,
        151935, 151935, 151935, 151935, 151935, 151935, 151935, 151935, 151935,
        151935, 151935, 151935
    ]
    ref_2 = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
        151644, 872, 198, 3838, 374, 697, 829, 30, 151645, 198, 151644, 77091,
        198, 40, 1079, 1207, 16948, 11, 264, 3460, 4128, 1614, 3465, 553, 54364,
        14817, 13, 151645, 151645, 151645, 151645, 151645, 151645
    ]

    input_text = "What is your name?"
    print("Run inference with lora id 0...")
    venv_check_call(llm_venv, [
        f"{qwen_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_qwen_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict

    print("Run inference with lora id -1...")
    venv_check_call(llm_venv, [
        f"{qwen_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_qwen_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_2 == predict


@pytest.mark.skip_less_device_memory(60000)
@pytest.mark.parametrize("llm_lora_model_root",
                         ["Upcycled-Qwen1.5-MoE2.7B-LoRA"],
                         indirect=True)
@pytest.mark.parametrize("llm_qwen_model_root", ["qwen1.5_moe_a2.7b_chat"],
                         indirect=True)
def test_llm_qwen1_5_moe_plugin_single_gpu_lora(
    qwen_example_root,
    llm_qwen_model_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    llm_lora_model_root,
):
    "run Qwen1.5 MoE lora test on single gpu."
    print("Build engines...")
    dtype = 'float16'
    model_name = os.path.basename(llm_qwen_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_qwen_model_root,
                               data_type=dtype)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--moe_plugin=auto",
        f"--lora_dir={llm_lora_model_root}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
        151644, 872, 198, 3838, 374, 697, 829, 30, 151645, 198, 151644, 77091,
        198, 151935, 151935, 151935, 151935, 151935, 151935, 151935, 151935,
        151935, 151935, 151935, 151935, 151935, 151935, 151935, 151935, 151935,
        151935, 151935, 151935
    ]
    ref_2 = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
        151644, 872, 198, 3838, 374, 697, 829, 30, 151645, 198, 151644, 77091,
        198, 40, 1079, 1207, 16948, 11, 264, 3460, 4128, 1614, 3465, 553, 54364,
        14817, 13, 151645, 151645, 151645, 151645, 151645, 151645
    ]

    input_text = "What is your name?"
    print("Run inference with lora id 0...")
    venv_check_call(llm_venv, [
        f"{qwen_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_qwen_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict

    print("Run inference with lora id -1...")
    venv_check_call(llm_venv, [
        f"{qwen_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_qwen_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_2 == predict


@skip_post_blackwell
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=['enable_weight_only', 'disable_weight_only'])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("llm_qwen_model_root", [
    "qwen2.5_7b_chat", "qwen1.5_7b_chat", "qwen2_7b_instruct",
    "qwen2_vl_7b_instruct"
],
                         indirect=True)
def test_llm_qwen_7b_int8_kv_1node_1gpus(qwen_example_root, llm_qwen_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         use_weight_only, use_gemm_plugin):
    "Build & Run Qwen-7B-Chat int8 kv cache"
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = "Qwen-chat-7b-int8-kv"
    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=qwen_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llm_qwen_model_root,
        data_type=dtype,
        int8_kv_cache=True,
        use_weight_only=use_weight_only,
        weight_only_precision='int8' if use_weight_only else None,
        calib_dataset=f"{llm_datasets_root}/ccdv/cnn_dailymail")

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={8}",
        f"--gpt_attention_plugin={dtype}"
    ]
    if use_gemm_plugin:
        build_cmd.append(f"--gemm_plugin={dtype}")
    else:
        build_cmd.append("--gemm_plugin=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_qwen_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        "--tensorrt_llm_rouge1_threshold=22",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "tp_pp_size", [(2, 1), (4, 1), (2, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize("use_plugin", ["enable_plugin", "disable_plugin"])
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_fmha', 'disable_fmha', 'enable_fmha_fp32_acc'])
@pytest.mark.parametrize("llm_qwen_model_root", [
    "qwen2.5_7b_chat", "qwen1.5_7b_chat", "qwen2_7b_instruct",
    "qwen2_vl_7b_instruct"
],
                         indirect=True)
def test_llm_qwen_7b_multi_gpus_summary(qwen_example_root, llm_qwen_model_root,
                                        llm_datasets_root, llm_rouge_root,
                                        llm_venv, cmodel_dir, engine_dir,
                                        num_beams, tp_pp_size, use_plugin,
                                        context_fmha_type):
    "Run qwen on multi gpus"
    tp_size, pp_size = tp_pp_size
    world_size = tp_size * pp_size

    if get_device_count() < world_size:
        pytest.skip(f"devices are less than {world_size}.")

    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_qwen_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_qwen_model_root,
                               data_type=dtype,
                               gpus=world_size,
                               tp_size=tp_size,
                               pp_size=pp_size,
                               workers=world_size)
    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={8}",
        "--max_seq_len=8192", f"--max_beam_width={num_beams}",
        f"--workers={world_size}"
    ]

    if use_plugin:
        build_cmd.append(f"--gemm_plugin={dtype}")
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
    else:
        build_cmd.append("--gemm_plugin=disable")
        build_cmd.append("--gpt_attention_plugin=disable")
    if context_fmha_type == "enable_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disable_fmha":
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_qwen_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}", "--tensorrt_llm_rouge1_threshold=24",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha_type == "enable_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("per_token_channel", [True, False],
                         ids=["enable_ptpc", "disable_ptpc"])
@pytest.mark.parametrize("llm_qwen_model_root", [
    "qwen_7b_chat", "qwen_14b_chat", "qwen1.5_7b_chat", "qwen2_7b_instruct",
    "qwen2_vl_7b_instruct", "qwen2.5_7b_instruct"
],
                         indirect=True)
def test_llm_qwen_smooth_quant_single_gpu_summary(qwen_example_root,
                                                  llm_qwen_model_root,
                                                  llm_datasets_root,
                                                  llm_rouge_root, llm_venv,
                                                  engine_dir, cmodel_dir,
                                                  per_token_channel, num_beams):
    "Run qwen with smooth quant on single gpu"
    if "14B" in llm_qwen_model_root and get_device_memory() < 80000:
        pytest.skip("GPU memory is insufficient.")

    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_qwen_model_root)
    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=qwen_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llm_qwen_model_root,
        data_type=dtype,
        smoothquant=0.5,
        per_token=per_token_channel,
        per_channel=per_token_channel,
        calib_dataset=f"{llm_datasets_root}/ccdv/cnn_dailymail")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        f"--max_beam_width={num_beams}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_seq_len={8192}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_qwen_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}", "--tensorrt_llm_rouge1_threshold=24",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llm_qwen_model_root", [
    "qwen_7b_chat_int4", "qwen1.5_14b_chat_int4", "qwen1.5_7b_chat_awq",
    "qwen2_7b_awq", "qwen2.5_14b_instruct_int4"
],
                         indirect=True)
def test_llm_qwen_int4_single_gpu_summary(qwen_example_root,
                                          llm_qwen_model_root,
                                          llm_datasets_root, llm_rouge_root,
                                          llm_venv, cmodel_dir, engine_dir,
                                          num_beams):
    "Run qwen with gptq on single gpu"
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_qwen_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_qwen_model_root,
                               data_type=dtype)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        f"--max_beam_width={num_beams}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_seq_len={8192}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_qwen_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}", "--tensorrt_llm_rouge1_threshold=24",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llm_qwen_model_root", [
    "qwen_7b_chat", "qwen1.5_7b_chat", "qwen2_7b_instruct",
    "qwen2_vl_7b_instruct", "qwen2.5_7b_instruct"
],
                         indirect=True)
def test_llm_qwen_awq_single_gpu_summary(qwen_example_root, llm_qwen_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, engine_dir, num_beams,
                                         qcache_dir):
    "Build & run int4 awq on 1 gpus."
    print("Quantizing model...")
    dtype = 'float16'
    output_dir = f"{qcache_dir}/quantized_int4-awq"
    quantize_cmd = [
        f"{qwen_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_qwen_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--output_dir={output_dir}",
        "--awq_block_size=128",
        "--qformat=int4_awq",
        "--dtype=float16",
        "--calib_size=32",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={output_dir}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        f"--max_beam_width={num_beams}",
        f"--output_dir={engine_dir}",
        "--max_seq_len=8192",
        f"--max_batch_size={8}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_qwen_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}", "--tensorrt_llm_rouge1_threshold=22.8",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize(
    "context_fmha", ["context_fmha", "context_fmha_fp32_acc", "disable_fmha"])
@pytest.mark.parametrize(
    "tp_pp_size", [(8, 1), (4, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize(
    "llm_qwen_model_root",
    ["qwen2.5_72b_chat", "qwen1.5_72b_chat", "qwen2_72b_instruct"],
    indirect=True)
def test_llm_qwen_1node_8gpus_summary(qwen_example_root, llm_qwen_model_root,
                                      llm_datasets_root, llm_rouge_root,
                                      llm_venv, cmodel_dir, engine_dir,
                                      context_fmha, tp_pp_size):
    "Run qwen with smooth quant on single gpu"
    print("Converting checkpoint...")
    dtype = 'float16'
    tp_size, pp_size = tp_pp_size
    world_size = tp_size * pp_size
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model="qwen-72b",
                               model_path=llm_qwen_model_root,
                               data_type=dtype,
                               tp_size=4,
                               pp_size=2)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--gemm_plugin={dtype}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--workers={world_size}",
        "--max_beam_width=4",
        "--max_seq_len=8192",
        "--max_input_len=2048",
    ]

    if context_fmha == "context_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha == "disable_fmha":
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        f"--hf_model_dir={llm_qwen_model_root}", "--data_type=fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}", "--num_beams=4",
        "--tensorrt_llm_rouge1_threshold=22", "--max_input_length=2048",
        "--output_len=2048", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha == "context_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize(
    "context_fmha", ["context_fmha", "context_fmha_fp32_acc", "disable_fmha"])
@pytest.mark.parametrize(
    "tp_pp_size", [(4, 1), (2, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize("llm_qwen_model_root", ["qwen2_57b_a14b"],
                         indirect=True)
def test_llm_qwen_moe_multi_gpu_summary(qwen_example_root, llm_qwen_model_root,
                                        llm_datasets_root, llm_rouge_root,
                                        llm_venv, cmodel_dir, engine_dir,
                                        context_fmha, tp_pp_size):
    "Run qwen with smooth quant on single gpu"
    print("Converting checkpoint...")
    dtype = 'float16'
    tp_size, pp_size = tp_pp_size
    world_size = tp_size * pp_size
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=qwen_example_root,
                               cmodel_dir=cmodel_dir,
                               model="qwen-57b",
                               model_path=llm_qwen_model_root,
                               data_type=dtype,
                               tp_size=tp_size,
                               pp_size=pp_size)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--gemm_plugin={dtype}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--workers={world_size}",
        "--max_input_len=2048",
        "--max_seq_len=8192",
    ]

    if context_fmha == "context_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha == "disable_fmha":
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{qwen_example_root}/../../../summarize.py", "--test_trt_llm",
        f"--hf_model_dir={llm_qwen_model_root}", "--data_type=fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        "--tensorrt_llm_rouge1_threshold=20", "--max_input_length=2048",
        "--output_len=2048", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha == "context_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@skip_pre_ada
@pytest.mark.parametrize("dtype", ['float16', 'bfloat16'])
@pytest.mark.parametrize("qformat", ['fp8'])
@pytest.mark.parametrize("llm_qwen_model_root", [
    "qwen2.5_7b_instruct", "qwen2_7b_instruct", "qwen2_vl_7b_instruct",
    "qwen2_0.5b_instruct", "qwen2.5_0.5b_instruct", "qwen2.5_1.5b_instruct"
],
                         indirect=True)
def test_llm_hf_qwen_quantization_1gpu(dtype, llm_qwen_model_root, llm_venv,
                                       cmodel_dir, engine_dir,
                                       qwen_example_root, llm_datasets_root,
                                       llm_rouge_root, qformat):
    "Run qwen quantization tests"
    print("Convert checkpoint by modelopt...")
    convert_cmd = [
        f"{qwen_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_qwen_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--kv_cache_dtype={qformat}",
        f"--output_dir={cmodel_dir}",
    ]
    venv_check_call(llm_venv, convert_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={cmodel_dir}",
        f"--output_dir={engine_dir}", f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}", "--max_seq_len=4096",
        f"--max_batch_size={8}", "--use_fp8_context_fmha=disable",
        "--use_paged_context_fmha=disable"
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run evaluation...")
    threshold_score = 21
    mmlu_score = None
    if "0.5B" in llm_qwen_model_root:
        mmlu_score = 46.0
    elif "1.5B" in llm_qwen_model_root:
        mmlu_score = 58.5
    elif "7B" in llm_qwen_model_root:
        mmlu_score = 71.5
    # Install custom jinja to overcome ImportError from transformers library.
    # ImportError: apply_chat_template requires jinja2>=3.1.0 to be installed.
    llm_venv.run_cmd(['-m', 'pip', 'install', 'jinja2==3.1.0'])
    # Run MMLU for Qwen 2.5 models.
    if '2.5' in llm_qwen_model_root:
        mmlu_cmd = [
            "trtllm-eval", f"--model={engine_dir}",
            f"--tokenizer={llm_qwen_model_root}", "--backend=tensorrt", "mmlu",
            f"--dataset_path={llm_datasets_root}/mmlu", "--check_accuracy",
            f"--accuracy_threshold={mmlu_score}"
        ]
        check_call(" ".join(mmlu_cmd), shell=True, env=llm_venv._new_env)

    else:
        summary_cmd = [
            f"{qwen_example_root}/../../../summarize.py",
            "--test_trt_llm",
            f"--hf_model_dir={llm_qwen_model_root}",
            f"--engine_dir={engine_dir}",
            "--check_accuracy",
            f"--tensorrt_llm_rouge1_threshold={threshold_score}",
            f"--dataset_dir={llm_datasets_root}",
            f"--rouge_dir={llm_rouge_root}",
        ]
        venv_check_call(llm_venv, summary_cmd)


@skip_pre_ada
@pytest.mark.parametrize(
    "llm_qwen_model_root",
    ["qwen2_0.5b_instruct", "qwen2.5_0.5b_instruct", "qwen2.5_1.5b_instruct"],
    indirect=True)
def test_llm_hf_qwen_multi_lora_1gpu(llm_qwen_model_root,
                                     llm_venv,
                                     cmodel_dir,
                                     engine_dir,
                                     qwen_example_root,
                                     llm_datasets_root,
                                     qformat='fp8',
                                     dtype='bfloat16'):
    "Run Qwen models with multiple dummy LoRAs."

    print("Convert checkpoint by modelopt...")
    convert_cmd = [
        f"{qwen_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_qwen_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--kv_cache_dtype={qformat}",
        f"--output_dir={cmodel_dir}",
    ]
    venv_check_call(llm_venv, convert_cmd)

    test_multi_lora_support(
        hf_model_dir=llm_qwen_model_root,
        tllm_ckpt_dir=cmodel_dir,
        engine_dir=engine_dir,
        llm_venv=llm_venv,
        example_root=qwen_example_root,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
        zero_lora_weights=True,
    )
