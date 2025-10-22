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

import contextlib
import os
import re
import subprocess
import tempfile
from typing import Callable

import pytest
import yaml
from defs.common import wait_for_server
from defs.conftest import (get_sm_version, llm_models_root, skip_arm,
                           skip_no_hopper)
from defs.trt_test_alternative import check_call, check_output, popen

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.logger import logger


def cleanup_output_files():
    """Clean up output files from previous runs."""
    for file in ['output.json', 'output_streaming.json']:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


def validate_timing_metrics(perf_metrics_item, request_context=""):
    """
    Helper function to validate timing metrics relationships.

    Args:
        perf_metrics_item: A single performance metrics item from the /perf_metrics endpoint
        request_context: String context for error messages (e.g., "request 1", "streaming")
    """
    # Validate basic structure
    required_keys = [
        "ctx_server", "gen_server", "ctx_perf_metrics", "gen_perf_metrics",
        "disagg_server_arrival_time", "disagg_server_first_token_time"
    ]
    for key in required_keys:
        assert key in perf_metrics_item, f"Missing key: {key} in {request_context}"

    assert perf_metrics_item["ctx_perf_metrics"][
        "ctx_request_id"] == perf_metrics_item["gen_perf_metrics"][
            "ctx_request_id"]

    # Extract timing metrics
    ctx_metrics = perf_metrics_item["ctx_perf_metrics"]["perf_metrics"][
        "timing_metrics"]
    gen_metrics = perf_metrics_item["gen_perf_metrics"]["perf_metrics"][
        "timing_metrics"]
    disagg_arrival = perf_metrics_item["disagg_server_arrival_time"]
    disagg_first_token = perf_metrics_item["disagg_server_first_token_time"]

    # Validate disaggregated server timing metrics
    assert disagg_arrival is not None, f"disagg_server_arrival_time is None in {request_context}"
    assert disagg_first_token is not None, f"disagg_server_first_token_time is None in {request_context}"
    assert isinstance(
        disagg_arrival,
        (int, float
         )), f"disagg_server_arrival_time is not numeric in {request_context}"
    assert isinstance(
        disagg_first_token, (int, float)
    ), f"disagg_server_first_token_time is not numeric in {request_context}"
    assert disagg_arrival > 0, f"disagg_server_arrival_time is not positive in {request_context}"
    assert disagg_first_token > 0, f"disagg_server_first_token_time is not positive in {request_context}"
    assert disagg_arrival <= disagg_first_token, f"disagg_server_arrival_time > disagg_server_first_token_time in {request_context}"

    # Validate server-level timing metrics for context server
    ctx_server_arrival = ctx_metrics.get("server_arrival_time")
    ctx_server_first_token = ctx_metrics.get("server_first_token_time")
    assert ctx_server_arrival is not None, f"ctx server_arrival_time is None in {request_context}"
    assert ctx_server_first_token is not None, f"ctx server_first_token_time is None in {request_context}"
    assert isinstance(
        ctx_server_arrival,
        (int,
         float)), f"ctx server_arrival_time is not numeric in {request_context}"
    assert isinstance(
        ctx_server_first_token,
        (int, float
         )), f"ctx server_first_token_time is not numeric in {request_context}"
    assert ctx_server_arrival <= ctx_server_first_token, f"ctx server_arrival_time > server_first_token_time in {request_context}"
    assert ctx_metrics["last_token_time"] - ctx_server_first_token < 1e-3

    # Validate server-level timing metrics for generation server
    gen_server_arrival = gen_metrics.get("server_arrival_time")
    gen_server_first_token = gen_metrics.get("server_first_token_time")
    assert gen_server_arrival is not None, f"gen server_arrival_time is None in {request_context}"
    assert gen_server_first_token is not None, f"gen server_first_token_time is None in {request_context}"
    assert isinstance(
        gen_server_arrival,
        (int,
         float)), f"gen server_arrival_time is not numeric in {request_context}"
    assert isinstance(
        gen_server_first_token,
        (int, float
         )), f"gen server_first_token_time is not numeric in {request_context}"
    assert gen_server_arrival <= gen_server_first_token, f"gen server_arrival_time > server_first_token_time in {request_context}"

    # Validate timing relationships between different levels
    # Disaggregated server should receive request before individual servers
    assert disagg_arrival <= ctx_server_arrival, f"disagg_arrival > ctx_server_arrival in {request_context}"
    assert disagg_arrival <= gen_server_arrival, f"disagg_arrival > gen_server_arrival in {request_context}"

    # Context should complete before generation starts
    assert ctx_server_first_token <= gen_server_arrival, f"ctx_server_first_token > gen_server_arrival in {request_context}"

    # Validate internal timing consistency
    ctx_arrival_time = ctx_metrics["arrival_time"]
    ctx_first_token_time = ctx_metrics["first_token_time"]
    gen_arrival_time = gen_metrics["arrival_time"]
    gen_first_token_time = gen_metrics["first_token_time"]

    assert ctx_arrival_time <= ctx_first_token_time, f"ctx arrival_time > first_token_time in {request_context}"
    assert gen_arrival_time <= gen_first_token_time, f"gen arrival_time > first_token_time in {request_context}"

    # Test KV cache transfer timing (if present)
    if "kv_cache_transfer_start" in gen_metrics and "kv_cache_transfer_end" in gen_metrics:
        kv_start = gen_metrics["kv_cache_transfer_start"]
        kv_end = gen_metrics["kv_cache_transfer_end"]
        assert gen_metrics["kv_cache_size"] > 0
        assert kv_start <= kv_end, f"kv_cache_transfer_start > kv_cache_transfer_end in {request_context}"
        assert gen_arrival_time <= kv_start, f"gen_arrival_time > kv_cache_transfer_start in {request_context}"
        assert kv_end <= gen_metrics[
            "first_scheduled_time"], f"kv_cache_transfer_end > first_scheduled_time in {request_context}"

    return True


def get_disagg_server_url_from_cfg(config_file: str) -> str:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    server_host = config.get('hostname', 'localhost')
    server_port = config.get('port', 8000)
    return f"http://{server_host}:{server_port}"


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
        "gen_only_bs1":
        (4, f"{test_configs_root}/disagg_config_gen_only_bs1.yaml"),
        "4_ranks": (4, f"{test_configs_root}/disagg_config_ctxtp2_gentp1.yaml"),
        "4_ranks_trt_backend":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp1_trt_backend.yaml"),
        "cuda_graph":
        (2, f"{test_configs_root}/disagg_config_cuda_graph_padding.yaml"),
        "mixed": (2, f"{test_configs_root}/disagg_config_mixed.yaml"),
        "overlap": (2, f"{test_configs_root}/disagg_config_overlap.yaml"),
        "perf_metrics": (2, f"{test_configs_root}/disagg_config_metrics.yaml"),
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
        "ctxpp4_gentp4":
        (8, f"{test_configs_root}/disagg_config_ctxpp4_gentp4.yaml"),
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
        "deepseek_v3_lite_fp8_ctxpp2_gentp2_one_mtp":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp_ctxpp2_gentp2.yaml"
         ),
    }

    if test_desc not in config_map:
        raise ValueError(f"Invalid test description: {test_desc}, "
                         f"valid descriptions are: {config_map.keys()}")

    return config_map[test_desc]


def get_extra_llm_config(config, suffix, cwd):
    extra_llm_config = {
        'orchestrator_type': 'ray',
    }
    for key, value in config.items():
        if key not in ['num_instances', 'urls']:
            extra_llm_config[key] = value

    temp_fd, extra_config_file = tempfile.mkstemp(suffix='_%s.yaml' % suffix,
                                                  dir=cwd)
    with os.fdopen(temp_fd, 'w') as f:
        yaml.dump(extra_llm_config, f)

    return extra_config_file


def generate_worker_commands(model_path, config, server_config,
                             extra_config_file, server_role):
    worker_commands = []

    assert model_path, "model path is required."

    for url in server_config['urls']:
        host, port = url.split(':')
        cmd = [
            'trtllm-serve', model_path, '--host', host, '--port', port,
            '--backend', config['backend'], '--extra_llm_api_options',
            extra_config_file, '--server_role', server_role
        ]
        worker_commands.append(cmd)
    return worker_commands


def run_client_tests(example_dir,
                     config_file,
                     test_desc,
                     num_iters,
                     env,
                     server_start_timeout,
                     prompt_file,
                     extra_endpoints_test,
                     server_url,
                     workers_proc,
                     server_proc,
                     use_ray=False):
    """Run client tests against the disaggregated server."""
    client_dir = f"{example_dir}/clients"
    for _ in range(num_iters):
        client_cmd = [
            'python3', f'{client_dir}/disagg_client.py', '-c', f'{config_file}',
            '-p', f'{client_dir}/{prompt_file}', '--ignore-eos',
            '--server-start-timeout',
            str(server_start_timeout)
        ]
        if prompt_file == "long_prompts.json":
            # Use max_tokens 4 for long prompts to reduce test time
            client_cmd.extend(['--max-tokens', '4'])

        # Prepare poll processes
        worker_processes = []
        if use_ray:
            for proc_cm in workers_proc:
                worker_processes.append(proc_cm.__enter__())
        else:
            worker_processes = [workers_proc]

        poll_procs = worker_processes + [server_proc]
        check_call(client_cmd, env=env, poll_procs=poll_procs)

        # Streaming client run
        streaming_client_cmd = client_cmd + [
            '--streaming', '-o', 'output_streaming.json'
        ]
        check_call(streaming_client_cmd, env=env, poll_procs=poll_procs)

        # Run the chat completion endpoint test only for TinyLlama
        if test_desc == "overlap" or test_desc == "trtllm_sampler":
            chat_client_cmd = client_cmd + [
                '-e', 'chat', '-o', 'output_chat.json'
            ]
            check_call(chat_client_cmd, env=env, poll_procs=poll_procs)

            streaming_chat_client_cmd = chat_client_cmd + [
                '--streaming', '-o', 'output_streaming_chat.json'
            ]
            check_call(streaming_chat_client_cmd,
                       env=env,
                       poll_procs=poll_procs)

        # Skip output verification for long prompts test
        if prompt_file == "long_prompts.json":
            continue

        if extra_endpoints_test is not None:
            extra_endpoints_test(server_url)

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
                            string in content for string in expected_string
                        ), f"None of the strings in {expected_string} found in {output_file}"
                    else:
                        assert expected_string in content, f"Expected string '{expected_string}' not found in {output_file}"
                for not_expected_string in not_expected_strings:
                    assert not_expected_string not in content, f"Unexpected string '{not_expected_string}' found in {output_file}"


def run_disaggregated_test(example_dir,
                           test_desc,
                           num_iters=5,
                           env=None,
                           cwd=None,
                           prompt_file="prompts.json",
                           extra_endpoints_test: Callable[[str], None] = None,
                           model_path=None):
    """Run disaggregated test with given configuration."""
    cleanup_output_files()
    run_env = env.copy()
    run_env["UCX_TLS"] = "^ib"

    num_ranks, config_file = get_test_config(test_desc, example_dir,
                                             os.path.dirname(__file__))

    use_ray = mpi_disabled()
    if not use_ray:
        workers_cmd = [
            'mpirun', '--allow-run-as-root', '--oversubscribe', '-n',
            str(num_ranks), 'trtllm-serve', 'disaggregated_mpi_worker', '-c',
            config_file
        ]
    else:
        pytest.skip(
            "https://nvbugs/5584607 Ray orchestrator is not supported with NIXL(DEFAULT) cache transceiver backend."
        )
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        if config['backend'] != "pytorch":
            pytest.skip(
                "Ray orchestrator is only supported with pytorch backend.")

        extra_config_files = []
        workers_cmds = []
        subprocess.run(['ray', 'start', '--head', '--disable-usage-stats'],
                       check=True)

        # Generate ctx and gen server worker commands
        ctx_extra_config_file = get_extra_llm_config(config['context_servers'],
                                                     "ctx", cwd)
        extra_config_files.append(ctx_extra_config_file)
        workers_cmds.extend(
            generate_worker_commands(model_path, config,
                                     config['context_servers'],
                                     ctx_extra_config_file, 'context'))

        gen_extra_config_file = get_extra_llm_config(
            config['generation_servers'], "gen", cwd)
        extra_config_files.append(gen_extra_config_file)
        workers_cmds.extend(
            generate_worker_commands(model_path, config,
                                     config['generation_servers'],
                                     gen_extra_config_file, 'generation'))

    server_start_timeout = 1200
    server_cmd = [
        'trtllm-serve', 'disaggregated', '--server_start_timeout',
        str(server_start_timeout), '-c', config_file
    ]
    server_url = get_disagg_server_url_from_cfg(config_file)

    try:
        if not use_ray:
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
                run_client_tests(example_dir,
                                 config_file,
                                 test_desc,
                                 num_iters,
                                 env,
                                 server_start_timeout,
                                 prompt_file,
                                 extra_endpoints_test,
                                 server_url,
                                 workers_proc,
                                 server_proc,
                                 use_ray=False)

        else:
            workers_proc = []
            with contextlib.ExitStack() as stack:
                workers_log = stack.enter_context(
                    open('output_workers.log', 'w'))

                for cmd in workers_cmds:
                    proc = stack.enter_context(
                        popen(
                            cmd,
                            stdout=workers_log,
                            stderr=subprocess.STDOUT,
                            env=run_env,
                            cwd=cwd,
                        ))
                    workers_proc.append(proc)

                output_disagg = stack.enter_context(
                    open('output_disagg.log', 'w'))
                server_proc = stack.enter_context(
                    popen(server_cmd,
                          stdout=output_disagg,
                          stderr=subprocess.STDOUT,
                          env=run_env,
                          cwd=cwd))

                if not wait_for_server("localhost",
                                       8000,
                                       timeout_seconds=server_start_timeout):
                    raise RuntimeError(
                        f"Disaggregated server failed to start within {server_start_timeout} seconds"
                    )

                run_client_tests(example_dir,
                                 config_file,
                                 test_desc,
                                 num_iters,
                                 env,
                                 server_start_timeout,
                                 prompt_file,
                                 extra_endpoints_test,
                                 server_url,
                                 workers_proc,
                                 server_proc,
                                 use_ray=True)
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
        if use_ray:
            subprocess.run(['ray', 'stop', '--force'], check=False)
            for extra_file in extra_config_files:
                if os.path.exists(extra_file):
                    os.remove(extra_file)
        elif 'server_proc' in locals() and 'workers_proc' in locals():
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


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_genbs1(disaggregated_test_root,
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
                           "gen_only_bs1",
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
def test_disaggregated_perf_metrics(disaggregated_test_root, llm_venv,
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

    def extra_endpoints_test(server_url: str):
        import json
        import urllib.request

        with urllib.request.urlopen(f"{server_url}/perf_metrics",
                                    timeout=10) as resp:
            assert resp.status == 200
            perf_metrics = json.load(resp)
        assert len(perf_metrics) > 0
        item = perf_metrics[0]

        # Use helper function to validate all timing metrics comprehensively
        validate_timing_metrics(item, "perf_metrics test")

    run_disaggregated_test(disaggregated_example_root,
                           "perf_metrics",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory(),
                           extra_endpoints_test=extra_endpoints_test)


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_kv_cache_time_output(disaggregated_test_root, llm_venv,
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

    output_path = os.path.join(llm_venv.get_working_directory(), "cache_time")
    run_disaggregated_test(disaggregated_example_root,
                           "perf_metrics",
                           env=llm_venv._new_env
                           | {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": output_path},
                           cwd=llm_venv.get_working_directory())
    assert os.path.isdir(output_path)
    send_file = os.path.join(output_path, "rank_0_send.csv")
    recv_file = os.path.join(output_path, "rank_1_recv.csv")
    assert os.path.exists(send_file)
    assert os.path.exists(recv_file)
    with open(send_file, "r") as f:
        lines = f.readlines()
        assert len(lines) > 1
        assert lines[0].startswith(
            "RequestID,Delay(ms),Duration(ms),Bandwidth(Gbps)")
        # get a send sample and match the recv
        sample = lines[1].split(',')
        assert len(sample) >= 4
    with open(recv_file, "r") as f:
        lines = f.readlines()
        assert len(lines) > 1
        matched = False
        for line in lines:
            sample_recv = line.split(',')
            if sample_recv[0] == sample[0]:
                matched = True
                assert float(sample_recv[1]) <= float(sample[1])
                break
        assert matched


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
                           cwd=llm_venv.get_working_directory(),
                           model_path=llama_model_root)


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
                           cwd=llm_venv.get_working_directory(),
                           model_path=llama_model_root)


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
                           cwd=llm_venv.get_working_directory(),
                           model_path=llama_model_root)


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
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxpp4_genpp4(disaggregated_test_root, llm_venv,
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
                           "ctxpp4_genpp4",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory())


#tiny llama pp4 will have uneven layer per pp. pp4
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxpp4_gentp4(disaggregated_test_root, llm_venv,
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
                           "ctxpp4_gentp4",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory(),
                           model_path=llama_model_root)


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


@pytest.mark.skip_less_device(4)
@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_ctxpp2_gentp2_one_mtp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    #add one mtp layer, pp rank0 will have 15 layer, pp rank 1 will have 16 layers.
    src_dst_dict = {
        deepseek_v3_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/fp8",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_ctxpp2_gentp2_one_mtp",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory(),
                           model_path=deepseek_v3_model_root)


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
                           cwd=llm_venv.get_working_directory(),
                           model_path=deepseek_v3_model_root)


@skip_no_hopper
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
                           cwd=llm_venv.get_working_directory(),
                           model_path=deepseek_v3_model_root)


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
        cwd=llm_venv.get_working_directory(),
        model_path=deepseek_v3_model_root)


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


@pytest.fixture(scope="module")
def benchmark_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "tensorrt_llm", "serve", "scripts")


@pytest.fixture(scope="module")
def shared_gpt_path():
    DEFAULT_LLM_MODEL_ROOT = os.path.join("/scratch.trt_llm_data", "llm-models")
    LLM_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT", DEFAULT_LLM_MODEL_ROOT)
    return os.path.join(LLM_MODELS_ROOT, "datasets",
                        "ShareGPT_V3_unfiltered_cleaned_split.json")


@pytest.fixture(scope="function")
def benchmark_model_root(request):
    models_root = llm_models_root()
    if (request.param == "DeepSeek-V3-Lite-fp8"):
        model_path = os.path.join(models_root, "DeepSeek-V3-Lite", "fp8")
    elif (request.param == "DeepSeek-V3-Lite-bf16"):
        model_path = os.path.join(models_root, "DeepSeek-V3-Lite", "bf16")
    elif request.param == "llama-v3-8b-hf":
        model_path = os.path.join(models_root, "llama-models-v3", "8B")
    elif request.param == "llama-3.1-8b-instruct-hf-fp8":
        model_path = os.path.join(models_root, "llama-3.1-model",
                                  "Llama-3.1-8B-Instruct-FP8")
    else:
        raise ValueError(f"Failed to find the model: {request.param}")
    return model_path


def run_disaggregated_benchmark(example_dir,
                                config_file,
                                benchmark_root,
                                benchmark_model_root,
                                shared_gpt_path,
                                env=None,
                                cwd=None):
    """Run disaggregated test with given configuration."""
    run_env = env.copy()
    run_env["UCX_TLS"] = "^ib"
    num_rank = 2
    workers_cmd = [
        'mpirun', '--allow-run-as-root', '--oversubscribe', '-n',
        str(num_rank), 'trtllm-serve', 'disaggregated_mpi_worker', '-c',
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
            # Ensure the sever has started
            client_dir = f"{example_dir}/clients"
            client_cmd = [
                'python3', f'{client_dir}/disagg_client.py', '-c',
                f'{example_dir}/disagg_config.yaml', '-p',
                f'{client_dir}/prompts.json', '--ignore-eos',
                '--server-start-timeout',
                str(server_start_timeout)
            ]
            # Warm up
            check_call(client_cmd,
                       env=env,
                       poll_procs=[workers_proc, server_proc])
            # Start Benchmark
            benchmark_script = os.path.join(benchmark_root,
                                            "benchmark_serving.py")
            benchmark_cmd = [
                'python3',
                benchmark_script,
                '--model',
                benchmark_model_root,
                '--tokenizer',
                benchmark_model_root,
                '--dataset-name',
                'random',
                '--dataset-path',
                shared_gpt_path,
                '--random-input-len',
                '256',
                '--random-output-len',
                '64',
                '--random-prefix-len',
                '0',
                '--num-prompts',
                '320',
                '--max-concurrency',
                '32',
                '--host',
                'localhost',
                '--port',
                '8000',
                '--ignore-eos',
                '--no-test-input',
                '--percentile-metrics',
                'e2el,ttft',
            ]
            # warm up
            check_call(benchmark_cmd, env=env)
            output = check_output(benchmark_cmd, env=env)
            e2el_pattern = r"Median E2EL \(ms\):\s*(\d+\.?\d*)"
            ttft_pattern = r"Median TTFT \(ms\):\s*(\d+\.?\d*)"
            e2el_match = re.search(e2el_pattern, output)
            ttft_match = re.search(ttft_pattern, output)
            if e2el_match and ttft_match:
                median_e2el = float(e2el_match.group(1))
                median_ttft = float(ttft_match.group(1))
                return median_e2el, median_ttft
            else:
                raise ValueError("No benchmark result found")

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


def get_config_for_benchmark(model_root, backend):
    serve_config = {
        "model": model_root,
        "hostname": "localhost",
        "port": 8000,
        "backend": "pytorch",
        "context_servers": {
            "num_instances": 1,
            "max_batch_size": 2,
            "max_num_tokens": 384,
            "max_seq_len": 384,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": backend,
                "max_tokens_in_buffer": 512,
            },
            "urls": ["localhost:8001"]
        },
        "generation_servers": {
            "num_instances": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_batch_size": 2,
            "max_num_tokens": 384,
            "max_seq_len": 384,
            "cache_transceiver_config": {
                "backend": backend,
                "max_tokens_in_buffer": 512,
            },
            "urls": ["localhost:8002"]
        }
    }
    return serve_config


@pytest.mark.parametrize("benchmark_model_root", [
    'DeepSeek-V3-Lite-fp8', 'DeepSeek-V3-Lite-bf16', 'llama-v3-8b-hf',
    'llama-3.1-8b-instruct-hf-fp8'
],
                         indirect=True)
def test_disaggregated_benchmark_on_diff_backends(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        benchmark_model_root, benchmark_root, shared_gpt_path):
    if "DeepSeek-V3-Lite" in benchmark_model_root and "fp8" in benchmark_model_root and get_sm_version(
    ) != 90:
        pytest.skip("The test should only run on Hopper")
    nixl_config = get_config_for_benchmark(benchmark_model_root, "NIXL")
    ucx_config = get_config_for_benchmark(benchmark_model_root, "UCX")
    temp_dir = tempfile.TemporaryDirectory()
    nixl_config_path = os.path.join(temp_dir.name, "nixl_config.yaml")
    ucx_config_path = os.path.join(temp_dir.name, "ucx_config.yaml")
    with open(nixl_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(nixl_config, f)
    with open(ucx_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(ucx_config, f)

    env = llm_venv._new_env.copy()
    nixl_e2el, nixl_ttft = run_disaggregated_benchmark(
        disaggregated_example_root,
        nixl_config_path,
        benchmark_root,
        benchmark_model_root,
        shared_gpt_path,
        env=env,
        cwd=llm_venv.get_working_directory())
    ucx_e2el, ucx_ttft = run_disaggregated_benchmark(
        disaggregated_example_root,
        ucx_config_path,
        benchmark_root,
        benchmark_model_root,
        shared_gpt_path,
        env=env,
        cwd=llm_venv.get_working_directory())
    print(f"Nixl E2EL: {nixl_e2el} ms, UCX E2EL: {ucx_e2el} ms")
    print(f"Nixl TTFT: {nixl_ttft} ms, UCX TTFT: {ucx_ttft} ms")

    assert ucx_e2el > 0 and nixl_e2el > 0 and nixl_e2el < 1.05 * ucx_e2el
    assert ucx_ttft > 0 and nixl_ttft > 0 and nixl_ttft < 1.05 * ucx_ttft
