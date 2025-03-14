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
"""Module test_dit test dit examples."""

import os

import pytest
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
from defs.conftest import get_device_count, skip_fp8_pre_ada
from defs.trt_test_alternative import check_call, exists


@pytest.fixture(scope="module")
def dit_example_root(llm_root):
    "Get DiT example root"
    example_root = os.path.join(llm_root, "examples", "dit")

    return example_root


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("tp_size", [1, 4], ids=lambda tp_size: f'tp{tp_size}')
@pytest.mark.parametrize(
    "llm_dit_model_root",
    ["dit-xl-2-256x256", "dit-xl-2-512x512", "dit-xl-2-512x512-fp8-linear"],
    indirect=True)
def test_llm_dit_multiple_gpus(dit_example_root, llm_dit_model_root, llm_venv,
                               engine_dir, cmodel_dir, tp_size):
    "Build & run dit."
    if get_device_count() < tp_size:
        pytest.skip(f"Device number is less than {tp_size}")

    skip_fp8_pre_ada("fp8" in llm_dit_model_root.lower())

    workspace = llm_venv.get_working_directory()
    dtype = "float16"
    tp_size, pp_size = tp_size, 1
    world_size = tp_size * pp_size
    model_name = os.path.basename(llm_dit_model_root)
    image_size = 512 if "512" in llm_dit_model_root else 256
    input_size = 64 if image_size == 512 else 32
    onnx_file = os.path.join(workspace, "vae_decoder/onnx/visual_encoder.onnx")
    plan_file = os.path.join(workspace,
                             "vae_decoder/plan/visual_encoder_fp16.plan")

    enable_fp8_linear = True if "FP8" in llm_dit_model_root else False
    print("Convert weight...")
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=dit_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llm_dit_model_root,
                                data_type=dtype,
                                tp_size=tp_size,
                                pp_size=pp_size,
                                input_size=input_size,
                                fp8_linear=enable_fp8_linear)

    print("Build TRT-LLM engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--workers={world_size}",
        "--max_batch_size=8",
        "--remove_input_padding=disable",
        "--bert_attention_plugin=disable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Build VAE engines...")
    build_vae_cmd = [
        f"{dit_example_root}/vae_decoder_trt.py",
        f"--image-size={image_size}",
        f"--onnxFile={onnx_file}",
        f"--planFile={plan_file}",
        "--max_batch_size=8",
    ]

    venv_check_call(llm_venv, build_vae_cmd)

    print("Run summary...")
    run_cmd = [
        f"{dit_example_root}/sample.py",
        f"--vae_decoder_engine={plan_file}",
        f"--tllm_model_dir={engine_dir}",
        f"--image-size={image_size}",
    ]

    if world_size > 1:
        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
            run_cmd)
    else:
        venv_check_call(llm_venv, run_cmd)

    assert exists(f"{workspace}/sample.png")
