# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import csv
import os

import pytest
from defs.common import convert_weights, venv_mpi_check_call
from defs.conftest import get_sm_version
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(45000)
@pytest.mark.parametrize("llm_lora_model_root", ["chinese-mixtral-lora"],
                         indirect=True)
@pytest.mark.parametrize("llm_mixtral_model_root", ["Mixtral-8x7B-v0.1"],
                         indirect=True)
def test_llm_mixtral_moe_plugin_lora_4gpus(
    llama_example_root,
    llm_mixtral_model_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    llm_lora_model_root,
):
    "run Mixtral MoE lora test on 4 gpu."
    print("Build engines...")
    dtype = 'float16'
    model_name = os.path.basename(llm_mixtral_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               tp_size=4,
                               pp_size=1,
                               model_path=llm_mixtral_model_root,
                               data_type=dtype)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--moe_plugin=auto",
        f"--lora_dir={llm_lora_model_root}",
        "--worker=4",
        "--max_batch_size=8",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        1, 28705, 29242, 30731, 31182, 235, 158, 142, 234, 182, 152, 28924,
        29926, 28971, 29242, 28988
    ]
    ref_2 = [
        1, 315, 2016, 285, 4284, 526, 5680, 28723, 28705, 28740, 28723, 661
    ]

    input_text = "我爱吃蛋糕"
    print("Run inference with lora id 0...")
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=5",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--use_py_session",
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        run_cmd)

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict

    print("Run inference with lora id -1...")
    input_text = "I love french quiche"
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=5",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--use_py_session",
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        run_cmd)

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_2 == predict
