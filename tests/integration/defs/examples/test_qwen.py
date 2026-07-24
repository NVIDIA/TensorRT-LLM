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
                         venv_check_call)
from defs.conftest import get_sm_version, skip_pre_ada
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)

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
