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
"""Module test_openai test openai examples."""
import os
import subprocess  # fmt: off

import pytest
from defs.common import find_tensorrt, venv_check_call
from defs.trt_test_alternative import call, check_call, make_clean_dirs


@pytest.fixture(scope="module")
def openai_triton_example_root(llm_root):
    "Get openai-triton example root"
    example_root = os.path.join(llm_root, "examples", "openai_triton",
                                "manual_plugin")

    return example_root


@pytest.fixture(scope="module")
def openai_triton_plugingen_example_root(llm_root):
    "Get openai-triton PluginGen example root"
    example_root = os.path.join(llm_root, "examples", "openai_triton",
                                "plugin_autogen")

    return example_root


@pytest.fixture(scope="module")
def llm_openai_triton_model_root(llm_venv):
    "prepare openai-triton model & return model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "triton")
    commit = "d4644d6cb3ae674e1f15932cac1f28104795744f"

    call(f"git clone https://github.com/openai/triton.git {model_root}",
         shell=True)
    call(f"cd {model_root} && git checkout {commit}", shell=True)
    llm_venv.run_cmd(["-m", "pip", "install", "cmake"])
    llm_venv.run_cmd([
        "-m", "pip", "install",
        os.path.abspath(os.path.join(model_root, "python"))
    ])

    yield model_root

    llm_venv.run_cmd(["-m", "pip", "uninstall", "-y", "triton"])


def test_llm_openai_triton_1gpu(openai_triton_example_root,
                                llm_openai_triton_model_root, llm_venv,
                                engine_dir, trt_config, is_trt_environment):
    aot_path = os.path.join(openai_triton_example_root, "aot")
    aot_fp16_path = os.path.join(aot_path, "fp16")
    aot_fp32_path = os.path.join(aot_path, "fp32")
    call(f"mkdir -p {aot_fp16_path}", shell=True)
    call(f"mkdir -p {aot_fp32_path}", shell=True)

    num_stages = "2"

    # yapf: disable
    # Kernel for data type=float16, BLOCK_M=128, BLOCK_DMODEL=64, BLOCK_N=128
    compile_cmd = [
        f"{llm_openai_triton_model_root}/python/triton/tools/compile.py",
        f"{openai_triton_example_root}/fmha_triton.py",
        "-n", "fused_attention_kernel",
        "-o", f"{aot_fp16_path}/fmha_kernel_d64_fp16",
        "--out-name", "fmha_d64_fp16", "-w", "4", "-ns", num_stages,
        "-s", "*fp16:16, *fp32:16, *fp32:16, *fp16:16, *fp16:16, *fp16:16, fp32, i32, i32, i32, 128, 64, 128",
        "-g", "(seq_len + 127) / 128, batch_size * num_heads, 1"
    ]
    venv_check_call(llm_venv, compile_cmd)

    # Kernel for data type=float32, BLOCK_M=64, BLOCK_DMODEL=64, BLOCK_N=64
    compile_cmd = [
        f"{llm_openai_triton_model_root}/python/triton/tools/compile.py",
        f"{openai_triton_example_root}/fmha_triton.py",
        "-n", "fused_attention_kernel",
        "-o", f"{aot_fp32_path}/fmha_kernel_d64_fp32",
        "--out-name", "fmha_d64_fp32", "-w", "4", "-ns", num_stages,
        "-s", "*fp32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, fp32, i32, i32, i32, 64, 64, 64",
        "-g", "(seq_len + 63) / 64, batch_size * num_heads, 1"
    ]
    venv_check_call(llm_venv, compile_cmd)

    # Link generated headers and create dispatchers.
    check_call(
        f"python3 {llm_openai_triton_model_root}/python/triton/tools/link.py "
        f"{aot_fp16_path}/*.h -o {aot_path}/fmha_kernel_fp16",
        shell=True)
    check_call(
        f"python3 {llm_openai_triton_model_root}/python/triton/tools/link.py "
        f"{aot_fp32_path}/*.h -o {aot_path}/fmha_kernel_fp32",
        shell=True)

    build_path = os.path.join(openai_triton_example_root, "build")
    # yapf: enable

    # make files
    make_clean_dirs(build_path)
    cmake_args = []
    try:
        import trt_test  # noqa
    except ImportError:
        pass
    else:
        trt_include_dir, trt_lib_dir = find_tensorrt(
            trt_config["new_ld_library_path"])

    if trt_include_dir:
        cmake_args.append(f"-DTRT_INCLUDE_DIR={trt_include_dir}")

    if trt_lib_dir:
        cmake_args.append(f"-DTRT_LIB_DIR={trt_lib_dir}")

    if is_trt_environment:
        cmake_args.append(f"-DCMAKE_C_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'")

    cmake_args = " ".join(cmake_args)
    check_call(f"cd {build_path} && cmake {cmake_args} .. && make", shell=True)

    # build engine
    build_cmd = [
        f"{openai_triton_example_root}/build.py", "--num_heads=32",
        "--head_size=64", "--max_batch_size=8", "--max_seq_len=512",
        "--dtype=float16", f"--output={engine_dir}"
    ]
    venv_check_call(llm_venv, build_cmd)

    # run inference
    run_cmd = [
        f"{openai_triton_example_root}/run.py", "--num_heads=32",
        "--head_size=64", "--batch_size=8", "--seq_len=512",
        "--log_level=verbose", "--benchmark", f"--engine_dir={engine_dir}"
    ]
    venv_check_call(llm_venv, run_cmd)


# TODO[chunweiy]: Enable it later
def test_llm_openai_triton_plugingen_1gpu(openai_triton_plugingen_example_root,
                                          openai_triton_example_root,
                                          llm_openai_triton_model_root,
                                          plugin_gen_path, llm_venv,
                                          trt_config):
    # copy the triton kernel definition
    subprocess.run(
        f"cp {openai_triton_example_root}/fmha_triton.py {openai_triton_plugingen_example_root}/fmha_triton.py"
        .split(),
        check=True)

    # generate plugin
    cmd = [
        plugin_gen_path,
        "--workspace",
        "./tmp",
        "--kernel_config",
        os.path.join(openai_triton_plugingen_example_root, "kernel_config.py"),
    ]
    try:
        import trt_test  # noqa
    except ImportError:
        pass
    else:
        trt_include_dir, trt_lib_dir = find_tensorrt(
            trt_config["new_ld_library_path"])
        if trt_lib_dir is not None:
            cmd.append(f'--trt_lib_dir={trt_lib_dir}')
        if trt_include_dir is not None:
            cmd.append(f'--trt_include_dir={trt_include_dir}')

    venv_check_call(llm_venv, cmd)

    # build engine
    cmd = [
        os.path.join(openai_triton_plugingen_example_root, "build_engine.py"),
    ]
    venv_check_call(llm_venv, cmd)

    # run engine
    cmd = [
        os.path.join(openai_triton_plugingen_example_root, "run_engine.py"),
    ]
    venv_check_call(llm_venv, cmd)
