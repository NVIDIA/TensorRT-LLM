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

import os

import pytest
from defs.common import (convert_weights, generate_deterministic_cmd,
                         venv_mpi_check_call)
from defs.conftest import skip_pre_hopper
from defs.trt_test_alternative import check_call


@skip_pre_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("data_type", ['float16', 'bfloat16'])
@pytest.mark.parametrize("llm_mixtral_model_root",
                         ['Mixtral-8x7B-Instruct-v0.1'],
                         indirect=True)
def test_llm_mixtral_4gpus_deterministic(llama_example_root,
                                         llm_mixtral_model_root,
                                         deterministic_test_root, llm_venv,
                                         cmodel_dir, engine_dir, data_type):
    tp_size, pp_size = 4, 1
    world_size = tp_size * pp_size
    moe_tp_size = tp_size

    os.environ['FORCE_DETERMINISTIC'] = "1"

    print("Convert checkpoint...")
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model="mixtral-instruct",
                               model_path=llm_mixtral_model_root,
                               tp_size=tp_size,
                               moe_tp_size=moe_tp_size,
                               pp_size=pp_size,
                               data_type=data_type,
                               workers=world_size)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--workers={world_size}",
        "--use_paged_context_fmha=enable",
        "--max_batch_size=256",
        "--max_num_tokens=33280",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run deterministic test...")
    deterministic_accuracy_threshold = 1
    payload = os.path.join(deterministic_test_root, "payload.json")
    deterministic_cmd = generate_deterministic_cmd(
        deterministic_test_root,
        engine_dir=engine_dir,
        tokenizer_dir=llm_mixtral_model_root,
        payload=payload,
        deterministic_accuracy_threshold=deterministic_accuracy_threshold)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        deterministic_cmd)

    os.environ.pop('FORCE_DETERMINISTIC', None)
