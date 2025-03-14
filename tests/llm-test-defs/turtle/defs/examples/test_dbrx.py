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

import os

import pytest
from defs.common import convert_weights, venv_mpi_check_call
from defs.trt_test_alternative import check_call


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_host_memory(1000000)
@pytest.mark.parametrize(
    "tp_pp_size", [(8, 1), (4, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("llm_dbrx_model_root", ["dbrx-base", "dbrx-instruct"],
                         indirect=True)
def test_llm_dbrx_8gpus(dbrx_example_root, llm_dbrx_model_root,
                        llm_datasets_root, llm_rouge_root, llm_venv, cmodel_dir,
                        engine_dir, dtype, tp_pp_size):
    "Build & run dbrx with 8 gpus"
    print("Converting checkpoint...")
    tp_size, pp_size = tp_pp_size
    world_size = tp_size * pp_size
    model_name = os.path.basename(llm_dbrx_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=dbrx_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_dbrx_model_root,
                               data_type=dtype,
                               gpus=world_size,
                               tp_size=tp_size,
                               pp_size=pp_size,
                               workers=world_size)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--moe_plugin={dtype}",
        f"--workers={world_size}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{dbrx_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}", f"--hf_model_dir={llm_dbrx_model_root}",
        "--batch_size=8", "--max_ite=40", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=22",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_host_memory(1000000)
@pytest.mark.parametrize("test_case", ["int8_wo", "int4_wo", "int8_kv"])
@pytest.mark.parametrize("llm_dbrx_model_root", ["dbrx-base", "dbrx-instruct"],
                         indirect=True)
def test_llm_dbrx_quantization_4gpus(dbrx_example_root, llm_dbrx_model_root,
                                     llm_datasets_root, llm_rouge_root,
                                     llm_venv, cmodel_dir, engine_dir,
                                     test_case):
    "Build & run dbrx with 4 gpus"
    print("Converting checkpoint...")
    dtype = 'float16'
    tp_size, pp_size = 4, 1
    world_size = tp_size * pp_size
    model_name = os.path.basename(llm_dbrx_model_root)

    if test_case == "int8_wo":
        convert_kwargs = {
            'use_weight_only': True,
            'weight_only_precision': 'int8'
        }
    elif test_case == "int4_wo":
        convert_kwargs = {
            'use_weight_only': True,
            'weight_only_precision': 'int4'
        }
    elif test_case == "int8_kv":
        convert_kwargs = {
            "int8_kv_cache": True,
            'calib_dataset': f"{llm_datasets_root}/ccdv/cnn_dailymail"
        }
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=dbrx_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_dbrx_model_root,
                               data_type=dtype,
                               gpus=world_size,
                               tp_size=tp_size,
                               pp_size=pp_size,
                               workers=world_size,
                               **convert_kwargs)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--moe_plugin={dtype}",
        f"--workers={world_size}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{dbrx_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}", f"--hf_model_dir={llm_dbrx_model_root}",
        "--batch_size=8", "--max_ite=40", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=20",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)
