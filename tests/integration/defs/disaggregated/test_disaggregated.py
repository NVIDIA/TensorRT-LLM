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
import subprocess
import time

import pytest


def kill_disaggregated_processes():
    """Kill any existing disaggregated processes."""
    try:
        subprocess.run(['pkill', '-9', '-f', 'launch_disaggregated'],
                       check=False)
    except Exception:
        pass


def cleanup_output_files():
    """Clean up output files from previous runs."""
    for file in ['output.json', 'output_streaming.json']:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


def get_test_config(test_desc, example_dir, test_root):
    """Get test configuration based on test description."""
    config_map = {
        "2_ranks": (2, f"{example_dir}/disagg_config.yaml"),
        "cuda_graph":
        (2, f"{test_root}/test_configs/disagg_config_cuda_graph_padding.yaml"),
        "mixed": (2, f"{test_root}/test_configs/disagg_config_mixed.yaml"),
        "overlap": (2, f"{test_root}/test_configs/disagg_config_overlap.yaml"),
        "deepseek_v3_lite_fp_8_overlap_dp":
        (4, f"{test_root}/test_configs/disagg_config_overlap_dp.yaml"),
        "4_ranks":
        (4, f"{test_root}/test_configs/disagg_config_ctxtp2_gentp1.yaml"),
        "deepseek_v3_lite_fp8":
        (4,
         f"{test_root}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite.yaml"
         ),
        "deepseek_v3_lite_fp8_attention_dp":
        (4,
         f"{test_root}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp.yaml"
         ),
        "deepseek_v3_lite_fp8_attention_dp_one":
        (4,
         f"{test_root}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one.yaml"
         ),
        "deepseek_v3_lite_fp8_attention_dp_one_mtp":
        (4,
         f"{test_root}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one_mtp.yaml"
         ),
        "deepseek_v3_lite_fp8_tp1_attention_dp_overlap_one_mtp":
        (4,
         f"{test_root}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_tp1_attention_dp_overlap_one_mtp.yaml"
         ),
    }

    if test_desc not in config_map:
        raise ValueError(f"Invalid test description: {test_desc}")

    return config_map[test_desc]


def run_disaggregated_test(example_dir,
                           test_desc,
                           num_iters=5,
                           skip_kill=False,
                           env=None,
                           cwd=None):
    """Run disaggregated test with given configuration."""
    kill_disaggregated_processes()
    cleanup_output_files()

    num_ranks, config_file = get_test_config(test_desc, example_dir,
                                             os.path.dirname(__file__))

    # Start workers
    workers_cmd = [
        'mpirun', '--allow-run-as-root', '-n',
        str(num_ranks), 'python3',
        f'{example_dir}/launch_disaggregated_workers.py', '-c', config_file
    ]
    with open('output_workers', 'w') as f:
        workers_proc = subprocess.Popen(workers_cmd,
                                        stdout=f,
                                        stderr=subprocess.STDOUT,
                                        env=env,
                                        cwd=cwd)

    # Start server
    server_cmd = [
        'python3', f'{example_dir}/launch_disaggregated_server.py',
        '--server_start_timeout', '900', '-c', config_file
    ]
    with open('output_disagg', 'w') as f:
        server_proc = subprocess.Popen(server_cmd,
                                       stdout=f,
                                       stderr=subprocess.STDOUT,
                                       env=env,
                                       cwd=cwd)

    time.sleep(10)

    client_dir = f"{example_dir}/clients"
    for _ in range(num_iters):
        client_cmd = [
            'python3', f'{client_dir}/disagg_client.py', '-c',
            f'{example_dir}/disagg_config.yaml', '-p',
            f'{client_dir}/prompts.json', '--server-start-timeout', '950'
        ]
        subprocess.run(client_cmd, check=True, env=env)

        # Streaming client run
        streaming_client_cmd = client_cmd + [
            '--streaming', '-o', 'output_streaming.json'
        ]
        subprocess.run(streaming_client_cmd, check=True, env=env)

        # Verify outputs
        expected_strings = [
            "The capital of Germany is Berlin", "Asyncio is a Python library"
        ]
        if "deepseek_v3_lite" in test_desc:
            expected_strings = ["Berlin", "Asyncio is a powerful tool"]

        for output_file in ['output.json', 'output_streaming.json']:
            with open(output_file, 'r') as f:
                content = f.read()
                for expected_string in expected_strings:
                    assert expected_string in content, f"Expected string '{expected_string}' not found in {output_file}"

    # Print outputs
    print("------------------")
    print("Workers output:")
    print("------------------")
    with open('output_workers', 'r') as f:
        print(f.read())

    print("\n\n------------------")
    print("Disagg server output")
    print("------------------")
    with open('output_disagg', 'r') as f:
        print(f.read())

    if not skip_kill:
        kill_disaggregated_processes()


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

    run_disaggregated_test(disaggregated_example_root,
                           "2_ranks",
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

    run_disaggregated_test(disaggregated_example_root,
                           "4_ranks",
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

    run_disaggregated_test(disaggregated_example_root,
                           "cuda_graph",
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

    run_disaggregated_test(disaggregated_example_root,
                           "mixed",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


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

    run_disaggregated_test(disaggregated_example_root,
                           "overlap",
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_tp1_single_gpu(
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

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8_tp1"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_tp1_single_gpu_mtp(
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

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8_tp1_mtp"
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
    env["UCX_TLS"] = "^ib"

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8"
    check_call(cmd, shell=True, env=env, cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_ucx_tp1_single_gpu(
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
    env = llm_venv._new_env.copy()
    env["TRTLLM_USE_UCX_KVCACHE"] = "1"

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp_8_overlap_dp",
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_attention_dp",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_overlap(
        disaggregated_test_root, llm_venv, disaggregated_example_root,
        deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/fp8",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp_8_attention_dp_overlap"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_overlap_cuda_graph(
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

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8_attention_dp_overlap_cuda_graph"
    check_call(cmd,
               shell=True,
               env=llm_venv._new_env,
               cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_overlap_cuda_graph(
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

    cmd = f"bash {disaggregated_test_root}/sanity_check.sh {disaggregated_example_root} deepseek_v3_lite_fp8_overlap_cuda_graph"
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_attention_dp_one",
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_attention_dp_one_mtp",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_tp1_attention_dp_overlap_one_mtp(
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_tp1_attention_dp_overlap_one_mtp",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())
