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

import pytest
from defs.conftest import skip_arm, skip_no_hopper
from defs.trt_test_alternative import check_call, popen

from tensorrt_llm.logger import logger


def cleanup_output_files():
    """Clean up output files from previous runs."""
    for file in ['output.json', 'output_streaming.json']:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


def get_test_config(test_desc, example_dir, test_root):
    """Get test configuration based on test description."""
    test_configs_root = f"{test_root}/test_configs"
    config_map = {
        "2_ranks_diff_max_tokens":
        (2, f"{test_configs_root}/disagg_config_diff_max_tokens.yaml"),
        "2_ranks": (2, f"{example_dir}/disagg_config.yaml"),
        "2_ranks_trt_backend":
        (2, f"{test_configs_root}/disagg_config_trt_backend.yaml"),
        "gen_only": (2, f"{test_configs_root}/disagg_config_gen_only.yaml"),
        "gen_only_trt_backend":
        (2, f"{test_configs_root}/disagg_config_gen_only_trt_backend.yaml"),
        "4_ranks": (4, f"{test_configs_root}/disagg_config_ctxtp2_gentp1.yaml"),
        "4_ranks_trt_backend":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp1_trt_backend.yaml"),
        "cuda_graph":
        (2, f"{test_configs_root}/disagg_config_cuda_graph_padding.yaml"),
        "mixed": (2, f"{test_configs_root}/disagg_config_mixed.yaml"),
        "overlap": (2, f"{test_configs_root}/disagg_config_overlap.yaml"),
        "trtllm_sampler":
        (2, f"{test_configs_root}/disagg_config_trtllm_sampler.yaml"),
        "load_balance":
        (4, f"{test_configs_root}/disagg_config_load_balance.yaml"),
        "cache_aware_balance":
        (4, f"{test_configs_root}/disagg_config_cache_aware_balance.yaml"),
        "conditional": (2,
                        f"{test_configs_root}/disagg_config_conditional.yaml"),
        "ngram": (2, f"{test_configs_root}/disagg_config_ngram.yaml"),
        "ctxpp2_genpp2":
        (4, f"{test_configs_root}/disagg_config_ctxpp2_genpp2.yaml"),
        "ctxtp2_genpp2":
        (4, f"{test_configs_root}/disagg_config_ctxtp2_genpp2.yaml"),
        "ctxpp2_gentp2":
        (4, f"{test_configs_root}/disagg_config_ctxpp2_gentp2.yaml"),
        "ctxtp2pp2_gentp2pp2":
        (8, f"{test_configs_root}/disagg_config_ctxtp2pp2_gentp2pp2.yaml"),
        "ctxpp4_genpp4":
        (8, f"{test_configs_root}/disagg_config_ctxpp4_genpp4.yaml"),
        "deepseek_v3_lite_fp8_mpi":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_mpi.yaml"
         ),
        "deepseek_v3_lite_fp8_ucx":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_ucx.yaml"
         ),
        "deepseek_v3_lite_fp8_nixl":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_nixl.yaml"
         ),
        "deepseek_v3_lite_fp8_tp1":
        (2,
         f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite.yaml"
         ),
        "deepseek_v3_lite_fp8_tp1_mtp":
        (2,
         f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp.yaml"
         ),
        "deepseek_v3_lite_fp_8_overlap_dp":
        (2,
         f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_overlap_dp.yaml"
         ),
        "deepseek_v3_lite_fp8_attention_dp":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp.yaml"
         ),
        "deepseek_v3_lite_fp_8_attention_dp_overlap":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_overlap.yaml"
         ),
        "deepseek_v3_lite_fp8_attention_dp_overlap_cuda_graph":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_overlap_cuda_graph.yaml"
         ),
        "deepseek_v3_lite_fp8_overlap_cuda_graph":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_overlap_cuda_graph.yaml"
         ),
        "deepseek_v3_lite_fp8_attention_dp_one":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one.yaml"
         ),
        "deepseek_v3_lite_fp8_attention_dp_one_mtp":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one_mtp.yaml"
         ),
        "deepseek_v3_lite_fp8_tp1_attention_dp_overlap_one_mtp":
        (2,
         f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp_attention_dp_overlap.yaml"
         ),
        "deepseek_v3_lite_bf16_cache_aware_balance":
        (4,
         f"{test_configs_root}/disagg_config_cache_aware_balance_deepseek_v3.yaml"
         ),
        "deepseek_v3_lite_bf16_conditional":
        (2, f"{test_configs_root}/disagg_config_conditional_deepseek_v3.yaml"),
        "deepseek_v3_lite_fp8_tp1_two_mtp":
        (2,
         f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_two_mtp.yaml"
         ),
    }

    if test_desc not in config_map:
        raise ValueError(f"Invalid test description: {test_desc}, "
                         f"valid descriptions are: {config_map.keys()}")

    return config_map[test_desc]


def run_disaggregated_test(example_dir,
                           test_desc,
                           num_iters=5,
                           env=None,
                           cwd=None,
                           prompt_file="prompts.json"):
    """Run disaggregated test with given configuration."""
    cleanup_output_files()
    run_env = env.copy()
    run_env["UCX_TLS"] = "^ib"

    num_ranks, config_file = get_test_config(test_desc, example_dir,
                                             os.path.dirname(__file__))

    workers_cmd = [
        'mpirun', '--allow-run-as-root', '--oversubscribe', '-n',
        str(num_ranks), 'trtllm-serve', 'disaggregated_mpi_worker', '-c',
        config_file
    ]

    server_start_timeout = 900
    server_cmd = [
        'trtllm-serve', 'disaggregated', '--server_start_timeout',
        str(server_start_timeout), '-c', config_file
    ]

    try:
        with (  # Start workers
                open('output_workers.log', 'w') as output_workers,
                popen(workers_cmd,
                      stdout=output_workers,
                      stderr=subprocess.STDOUT,
                      env=run_env,
                      cwd=cwd) as workers_proc,
                # Start server
                open('output_disagg.log', 'w') as output_disagg,
                popen(server_cmd,
                      stdout=output_disagg,
                      stderr=subprocess.STDOUT,
                      env=run_env,
                      cwd=cwd) as server_proc):
            client_dir = f"{example_dir}/clients"
            for _ in range(num_iters):
                client_cmd = [
                    'python3', f'{client_dir}/disagg_client.py', '-c',
                    f'{example_dir}/disagg_config.yaml', '-p',
                    f'{client_dir}/{prompt_file}', '--ignore-eos',
                    '--server-start-timeout',
                    str(server_start_timeout)
                ]
                if prompt_file == "long_prompts.json":
                    # Use max_tokens 4 for long prompts to reduce test time
                    client_cmd.extend(['--max-tokens', '4'])
                check_call(client_cmd,
                           env=env,
                           poll_procs=[workers_proc, server_proc])

                # Streaming client run
                streaming_client_cmd = client_cmd + [
                    '--streaming', '-o', 'output_streaming.json'
                ]
                check_call(streaming_client_cmd,
                           env=env,
                           poll_procs=[workers_proc, server_proc])

                # Run the chat completion endpoint test only for TinyLlama
                if test_desc == "overlap" or test_desc == "trtllm_sampler":
                    chat_client_cmd = client_cmd + [
                        '-e', 'chat', '-o', 'output_chat.json'
                    ]
                    check_call(chat_client_cmd,
                               env=env,
                               poll_procs=[workers_proc, server_proc])

                    streaming_chat_client_cmd = chat_client_cmd + [
                        '--streaming', '-o', 'output_streaming_chat.json'
                    ]
                    check_call(streaming_chat_client_cmd,
                               env=env,
                               poll_procs=[workers_proc, server_proc])

                # Skip output verification for long prompts test
                if prompt_file == "long_prompts.json":
                    continue

                # Verify outputs
                not_expected_strings = ["Berlin Berlin"]

                output_files = ['output.json', 'output_streaming.json']
                if test_desc == "overlap" or test_desc == "trtllm_sampler":
                    # Disable streaming chat completion for overlap test
                    # due to bug
                    output_files.extend(['output_chat.json'])

                if test_desc.startswith("gen_only"):
                    continue

                for output_file in output_files:
                    with open(output_file, 'r') as f:
                        content = f.read()
                        if "deepseek_v3_lite" in test_desc or output_file == "output_chat.json":
                            expected_strings = [
                                "Berlin", ["Asyncio is a", "Asyncio module in"]
                            ]
                        else:
                            expected_strings = [
                                "The capital of Germany is Berlin",
                                "Asyncio is a Python library"
                            ]
                        for expected_string in expected_strings:
                            if isinstance(expected_string, list):
                                # At least one of the strings in the list should be found in the content
                                assert any(
                                    string in content
                                    for string in expected_string
                                ), f"None of the strings in {expected_string} found in {output_file}"
                            else:
                                assert expected_string in content, f"Expected string '{expected_string}' not found in {output_file}"
                        for not_expected_string in not_expected_strings:
                            assert not_expected_string not in content, f"Unexpected string '{not_expected_string}' found in {output_file}"
    except Exception:
        # Print outputs on error
        logger.error("-------- Workers output --------")
        with open('output_workers.log', 'r') as f:
            logger.error(f.read())

        logger.error("-------- Disagg server output --------")
        with open('output_disagg.log', 'r') as f:
            logger.error(f.read())
        raise
    finally:
        server_proc.terminate()
        workers_proc.terminate()
        server_proc.wait()
        workers_proc.wait()


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_diff_max_tokens(disaggregated_test_root,
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

    run_disaggregated_test(disaggregated_example_root,
                           "2_ranks_diff_max_tokens",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory(),
                           prompt_file="long_prompts.json")


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
def test_disaggregated_single_gpu_with_mpirun_trt_backend(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "2_ranks_trt_backend",
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

    env = llm_venv._new_env.copy()
    env['TRTLLM_DISAGG_BENCHMARK_GEN_ONLY'] = '1'
    run_disaggregated_test(disaggregated_example_root,
                           "gen_only",
                           env=env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_benchmark_gen_only_trt_backend(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    env = llm_venv._new_env.copy()
    env['TRTLLM_DISAGG_BENCHMARK_GEN_ONLY'] = '1'
    run_disaggregated_test(disaggregated_example_root,
                           "gen_only_trt_backend",
                           env=env,
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


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_multi_gpu_with_mpirun_trt_backend(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "4_ranks_trt_backend",
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


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_trtllm_sampler(disaggregated_test_root, llm_venv,
                                      disaggregated_example_root,
                                      llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "trtllm_sampler",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_load_balance(disaggregated_test_root, llm_venv,
                                    disaggregated_example_root,
                                    llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "load_balance",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_cache_aware_balance(disaggregated_test_root, llm_venv,
                                           disaggregated_example_root,
                                           llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "cache_aware_balance",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_conditional(disaggregated_test_root, llm_venv,
                                   disaggregated_example_root,
                                   llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "conditional",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5441714")
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ngram(disaggregated_test_root, llm_venv,
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
                           "ngram",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxpp2_genpp2(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)
    run_disaggregated_test(disaggregated_example_root,
                           "ctxpp2_genpp2",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxtp2_genpp2(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)
    run_disaggregated_test(disaggregated_example_root,
                           "ctxtp2_genpp2",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxpp2_gentp2(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)
    run_disaggregated_test(disaggregated_example_root,
                           "ctxpp2_gentp2",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxtp2pp2_gentp2pp2(disaggregated_test_root, llm_venv,
                                           disaggregated_example_root,
                                           llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)
    run_disaggregated_test(disaggregated_example_root,
                           "ctxtp2pp2_gentp2pp2",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b'], indirect=True)
def test_disaggregated_ctxpp4_genpp4(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/llama-3.1-models/Meta-Llama-3.1-8B",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)
    run_disaggregated_test(disaggregated_example_root,
                           "ctxpp4_genpp4",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_mpi(disaggregated_test_root,
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
    env["TRTLLM_USE_MPI_KVCACHE"] = "1"
    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_mpi",
                           env=env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_tp1",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_tp1_mtp",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@skip_arm
@pytest.mark.skip_less_device(4)
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
    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_ucx",
                           env=env,
                           cwd=llm_venv.get_working_directory())


@skip_arm
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_nixl(disaggregated_test_root,
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
    env["TRTLLM_USE_NIXL_KVCACHE"] = "1"
    env["UCX_TLS"] = "^ib"
    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_nixl",
                           env=env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@skip_arm
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
    env["UCX_TLS"] = "^ib"

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_tp1",
                           env=env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
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


@skip_no_hopper
@pytest.mark.skip_less_device(4)
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp_8_attention_dp_overlap",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
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

    run_disaggregated_test(
        disaggregated_example_root,
        "deepseek_v3_lite_fp8_attention_dp_overlap_cuda_graph",
        env=llm_venv._new_env,
        cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
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

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_overlap_cuda_graph",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
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


@skip_no_hopper
@pytest.mark.skip_less_device(4)
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


@skip_no_hopper
@pytest.mark.skip_less_device(4)
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

    run_disaggregated_test(
        disaggregated_example_root,
        "deepseek_v3_lite_fp8_tp1_attention_dp_overlap_one_mtp",
        env=llm_venv._new_env,
        cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_cache_aware_balance(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/bf16",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_bf16_cache_aware_balance",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_conditional(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/bf16",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_bf16_conditional",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_tp1_two_mtp(
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
                           "deepseek_v3_lite_fp8_tp1_two_mtp",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())
