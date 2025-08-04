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
"""Module test_bindings test bindings examples."""

import os

import pytest
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
from defs.conftest import get_device_count, get_sm_version
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.fixture(scope="module")
def bindings_example_root(llm_root):
    "Get bindings example root"
    example_root = os.path.join(llm_root, "examples", "bindings", "executor")

    return example_root


@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_bindings_example(bindings_example_root, llama_example_root,
                              llama_model_root, llm_venv, cmodel_dir,
                              engine_dir):
    "Run basic example on single gpu"
    device_num = get_device_count()

    assert device_num in (1, 2, 4, 8)
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model='llama-7b',
                                model_path=llama_model_root,
                                data_type='float16',
                                tp_size=device_num,
                                pp_size=1)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--remove_input_padding=enable",
        f"--workers={device_num}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run basic example")
    run_cmd = [
        f"{bindings_example_root}/example_basic.py",
        f"--model_path={engine_dir}",
    ]

    if device_num >= 2:
        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n", f"{device_num}", "--allow-run-as-root"],
            run_cmd)
    else:
        venv_check_call(llm_venv, run_cmd)
