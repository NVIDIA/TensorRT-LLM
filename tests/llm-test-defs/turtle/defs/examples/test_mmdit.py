# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Module test_dit test mmdit examples."""

import os

import pytest
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
from defs.conftest import get_device_count
from defs.trt_test_alternative import check_call, exists


@pytest.fixture(scope="module")
def mmdit_example_root(llm_root, llm_venv):
    "Get MMDiT example root"
    example_root = os.path.join(llm_root, "examples", "mmdit")
    llm_venv.run_cmd(["-m", "pip", "uninstall", "-y", "apex"])
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    yield example_root

    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(llm_root, "requirements.txt")
    ])


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("tp_size", [
    1,
    2,
],
                         ids=lambda tp_size: f'tp{tp_size}')
@pytest.mark.parametrize("mmdit_model_root", [
    "stable-diffusion-3.5-medium",
],
                         indirect=True)
def test_mmdit_multiple_gpus(mmdit_example_root, mmdit_model_root, llm_venv,
                             engine_dir, cmodel_dir, tp_size):
    "Build & run mmdit."
    if get_device_count() < tp_size:
        pytest.skip(f"Device number is less than {tp_size}")

    workspace = llm_venv.get_working_directory()
    dtype = "float16"
    tp_size, pp_size = tp_size, 1
    world_size = tp_size * pp_size
    model_name = os.path.basename(mmdit_model_root)

    print("Convert weight...")
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=mmdit_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=mmdit_model_root,
        data_type=dtype,
        tp_size=tp_size,
        pp_size=pp_size,
    )

    print("Build TRT-LLM engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--workers={world_size}",
        "--max_batch_size=2",
        "--remove_input_padding=disable",
        "--bert_attention_plugin=auto",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    run_cmd = [
        f"{mmdit_example_root}/sample.py",
        f"--tllm_model_dir={engine_dir}",
    ]

    if world_size > 1:
        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
            run_cmd)
    else:
        venv_check_call(llm_venv, run_cmd)

    assert exists(f"{workspace}/sd3.5-mmdit.png")
