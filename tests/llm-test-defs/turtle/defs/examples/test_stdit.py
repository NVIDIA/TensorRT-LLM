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
"""Module test_dit test stdit examples."""

import os

import pytest
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
from defs.conftest import get_device_count
from defs.trt_test_alternative import check_call, exists


@pytest.fixture(scope="module")
def stdit_example_root(llm_root, llm_venv):
    "Get STDiT example root"
    example_root = os.path.join(llm_root, "examples", "stdit")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    llm_venv.run_cmd(["-m", "pip", "install", "colossalai", "--no-deps"])

    return example_root


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("tp_size", [
    1,
    2,
],
                         ids=lambda tp_size: f'tp{tp_size}')
@pytest.mark.parametrize("stdit_model_root", [
    "OpenSora-STDiT-v3",
],
                         indirect=True)
def test_stdit_multiple_gpus(stdit_example_root, stdit_model_root, llm_venv,
                             engine_dir, cmodel_dir, tp_size):
    "Build & run stdit."
    if get_device_count() < tp_size:
        pytest.skip(f"Device number is less than {tp_size}")

    workspace = llm_venv.get_working_directory()
    dtype = "float16"
    tp_size, pp_size = tp_size, 1
    world_size = tp_size * pp_size
    model_name = os.path.basename(stdit_model_root)

    print("Convert weight...")
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=stdit_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=stdit_model_root,
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
        "--gemm_plugin=float16",
        "--kv_cache_type=disabled",
        "--remove_input_padding=enable",
        "--bert_attention_plugin=auto",
        "--gpt_attention_plugin=auto",
        "--context_fmha=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    run_cmd = [
        f"{stdit_example_root}/sample.py",
        "\'a beautiful waterfall\'",
        f"--tllm_model_dir={engine_dir}",
    ]

    if world_size > 1:
        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
            run_cmd)
    else:
        venv_check_call(llm_venv, run_cmd)

    assert exists(f"{workspace}/sample_outputs/sample_0000.mp4")
