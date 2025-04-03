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
from defs.trt_test_alternative import check_call


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_single_gpu_with_mpirun(disaggregated_test_root,
                                              disaggregated_example_root,
                                              llm_venv, llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} 2_ranks"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_benchmark_gen_only(disaggregated_test_root,
                                          disaggregated_example_root, llm_venv,
                                          llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} gen_only"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_multi_gpu_with_mpirun(disaggregated_test_root,
                                             disaggregated_example_root,
                                             llm_venv, llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} 4_ranks"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_cuda_graph(disaggregated_test_root, llm_venv,
                                  disaggregated_example_root, llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} cuda_graph"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_mixed(disaggregated_test_root, llm_venv,
                             disaggregated_example_root, llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} mixed"
    check_call(cmd, shell=True, env=llm_venv._new_env)


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_overlap(disaggregated_test_root, llm_venv,
                               disaggregated_example_root, llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} overlap"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8(disaggregated_test_root,
                                            disaggregated_example_root,
                                            llm_venv, deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/fp8",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_ucx(disaggregated_test_root,
                                                disaggregated_example_root,
                                                llm_venv,
                                                deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/fp8",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)
    env = llm_venv._new_env.copy()
    env["TRTLLM_USE_UCX_KVCACHE"] = "1"

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8"
    check_call(cmd, shell=True, env=env, cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_overlap_dp(disaggregated_test_root, llm_venv,
                                  disaggregated_example_root,
                                  deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/fp8",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp_8_overlap_dp"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/fp8",
    }

    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8_attention_dp"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_one(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/fp8",
    }

    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8_attention_dp_one"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_one_mtp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/fp8",
    }

    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8_attention_dp_one_mtp"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())
