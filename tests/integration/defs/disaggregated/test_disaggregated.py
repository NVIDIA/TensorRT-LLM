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
import time
from dataclasses import dataclass
from typing import Callable

import pytest

try:
    import ray
except ImportError:
    import tensorrt_llm.ray_stub as ray

import yaml
from defs.common import (get_free_port_in_ci, parse_gsm8k_output,
                         revise_disagg_config_file_with_free_ports,
                         wait_for_server)
from defs.conftest import (get_sm_version, llm_models_root, skip_arm,
                           skip_no_hopper, skip_pre_blackwell)
from defs.trt_test_alternative import (check_call, check_output, popen,
                                       print_info)
from test_common.perf_metrics_utils import (get_timing_metrics,
                                            validate_timing_metrics)

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.logger import logger


@dataclass
class TestConfig:
    """Configuration for disaggregated test."""
    model_path: str
    test_desc: str
    request_count: int
    accuracy_threshold: float

    def __str__(self):
        return self.test_desc


def cleanup_output_files():
    """Clean up output files from previous runs."""
    for file in ['output.json', 'output_streaming.json']:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


def get_disagg_server_url_from_cfg(config_file: str) -> tuple[str, int]:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    server_host = config.get('hostname', 'localhost')
    server_port = config.get('port', 8000)
    return server_host, server_port


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
        "deepseek_v3_lite_bf16_empty_batch":
        (3,
         f"{test_configs_root}/disagg_config_deepseek_v3_lite_empty_batch.yaml"
         ),
        "llama4_kv_cache_overflow":
        (8, f"{test_configs_root}/disagg_config_llama4_kv_cache_overflow.yaml"),
        "deepseek_v3_lite_bf16_tllm_gen_helix":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp1cp2_deepseek_v3_lite_bf16_tllm_gen.yaml"
         ),
        "deepseek_r1_v2_fp4_stress":
        (8,
         f"{test_configs_root}/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml"
         ),
        "gpt_oss_120b_stress":
        (4,
         f"{test_configs_root}/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml"),
    }

    if test_desc not in config_map:
        raise ValueError(f"Invalid test description: {test_desc}, "
                         f"valid descriptions are: {config_map.keys()}")

    return (config_map[test_desc][0],
            revise_disagg_config_file_with_free_ports(config_map[test_desc][1]))


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
            '--backend', config['backend'], '--config', extra_config_file,
            '--server_role', server_role
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


# TODO: add test for disaggregated server prometheus metrics
def fetch_prometheus_metrics(server_url: str):
    import requests
    response = requests.get(f"{server_url}/prometheus/metrics", timeout=10)
    assert response.status_code == 200
    return response.text


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
    server_host, server_port = get_disagg_server_url_from_cfg(config_file)
    server_url = f"http://{server_host}:{server_port}"

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
            runtime_env = {
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"
                }
            }
            ray.init(address="local",
                     include_dashboard=False,
                     ignore_reinit_error=True,
                     runtime_env=runtime_env)
            gcs_addr = ray.get_runtime_context().gcs_address
            ray_port = str(gcs_addr.split(":")[1])
            run_env.update({
                "RAY_ADDRESS": f"localhost:{ray_port}",
                "TLLM_RAY_FORCE_LOCAL_CLUSTER": "0"
            })
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

                if not wait_for_server(server_host,
                                       server_port,
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
        if 'server_proc' in locals() and 'workers_proc' in locals():
            server_proc.terminate()
            workers_proc.terminate()
            server_proc.wait()
            workers_proc.wait()
        if use_ray:
            ray.shutdown()
            for extra_file in extra_config_files:
                if os.path.exists(extra_file):
                    os.remove(extra_file)


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
        item = get_timing_metrics(server_url)
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
            "RequestID,RequestInfo,Preparation,Preprocess,Transmissions,Postprocess"
        )
        assert ",Delay,Duration,Bandwidth(Gbps)" in lines[0]
        # get a send sample and match the recv
        sample = lines[1].split(',')
        assert len(sample) >= 9
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
                                cwd=None,
                                num_ranks=2,
                                random_input_len=16,
                                random_output_len=64,
                                num_prompts=100,
                                max_concurrency=32,
                                skip_warmup=False):
    """Run disaggregated test with given configuration."""
    run_env = env.copy()
    run_env["UCX_TLS"] = "^ib"
    workers_cmd = [
        'mpirun', '--allow-run-as-root', '--oversubscribe', '-n',
        str(num_ranks), 'trtllm-serve', 'disaggregated_mpi_worker', '-c',
        config_file
    ]

    server_start_timeout = 1200
    server_cmd = [
        'trtllm-serve', 'disaggregated', '--server_start_timeout',
        str(server_start_timeout), '-c', config_file
    ]
    server_host, server_port = get_disagg_server_url_from_cfg(config_file)
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
                'python3', f'{client_dir}/disagg_client.py', '-c', config_file,
                '-p', f'{client_dir}/prompts.json', '--ignore-eos',
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
                str(random_input_len),
                '--random-output-len',
                str(random_output_len),
                '--random-prefix-len',
                '0',
                '--num-prompts',
                str(num_prompts),
                '--max-concurrency',
                str(max_concurrency),
                '--host',
                server_host,
                '--port',
                str(server_port),
                '--ignore-eos',
                '--no-test-input',
                '--percentile-metrics',
                'e2el,ttft',
            ]
            # warm up
            if not skip_warmup:
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
        "port": get_free_port_in_ci(),
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
            "urls": [f"localhost:{get_free_port_in_ci()}"]
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
            "urls": [f"localhost:{get_free_port_in_ci()}"]
        }
    }
    return serve_config


def run_disaggregated_aiperf(config_file,
                             model_path,
                             num_ranks,
                             server_start_timeout=1200,
                             input_tokens=128,
                             output_tokens=100,
                             concurrency=1,
                             endpoint_type='chat',
                             request_count=None,
                             warmup_request_count=10,
                             streaming=True,
                             random_seed=100,
                             accuracy_test=False,
                             threshold=0.8,
                             env=None,
                             cwd=None):
    """Run disaggregated test with genai-perf for performance/stress testing.

    Args:
        config_file: Path to disaggregated server config YAML
        model_path: Path to model for tokenizer
        num_ranks: Number of MPI ranks for workers
        server_start_timeout: Timeout in seconds for server startup
        input_tokens: Mean synthetic input tokens
        output_tokens: Mean output tokens to generate
        concurrency: Number of concurrent requests
        endpoint_type: 'chat' or 'completions'
        request_count: Total requests (if None, uses concurrency*1024 or num_dataset_entries)
        warmup_request_count: Number of warmup requests
        streaming: Whether to use streaming mode
        random_seed: Random seed for reproducibility
        accuracy_test: Whether to run accuracy test
        threshold: Threshold for accuracy test
        env: Environment variables dict
        cwd: Working directory
    """
    cleanup_output_files()
    run_env = env.copy()
    run_env["UCX_TLS"] = "^ib"

    workers_cmd = [
        'mpirun', '--allow-run-as-root', '--oversubscribe', '-n',
        str(num_ranks), 'trtllm-serve', 'disaggregated_mpi_worker', '-c',
        config_file
    ]

    server_cmd = [
        'trtllm-serve', 'disaggregated', '--server_start_timeout',
        str(server_start_timeout), '-c', config_file
    ]

    artifact_dir = os.path.join(cwd or ".", "benchmark-results")
    server_host, server_port = get_disagg_server_url_from_cfg(config_file)

    try:
        with (open('output_workers.log', 'w') as output_workers,
              popen(workers_cmd,
                    stdout=output_workers,
                    stderr=subprocess.STDOUT,
                    env=run_env,
                    cwd=cwd) as workers_proc, open('output_disagg.log', 'w') as
              output_disagg,
              popen(server_cmd,
                    stdout=output_disagg,
                    stderr=subprocess.STDOUT,
                    env=run_env,
                    cwd=cwd) as server_proc):

            # Wait for server to be ready
            if not wait_for_server(server_host,
                                   server_port,
                                   timeout_seconds=server_start_timeout):
                raise RuntimeError(
                    f"Disaggregated server did not become ready within {server_start_timeout} seconds"
                )

            # Build base command (using aiperf instead of genai-perf)
            aiperf_cmd = [
                'aiperf', 'profile', '--model', model_path, '--tokenizer',
                model_path, '--endpoint-type', endpoint_type
            ]

            # Add endpoint path based on type
            if endpoint_type == 'chat':
                aiperf_cmd.extend(['--endpoint', '/v1/chat/completions'])

            # Add streaming flag if enabled
            if streaming:
                aiperf_cmd.append('--streaming')

            # Add common parameters
            aiperf_cmd.extend([
                '--url', f'{server_host}:{server_port}',
                '--synthetic-input-tokens-mean',
                str(input_tokens), '--synthetic-input-tokens-stddev', '0',
                '--output-tokens-mean',
                str(output_tokens), '--output-tokens-stddev', '0',
                '--extra-inputs', f'max_tokens:{output_tokens}',
                '--extra-inputs', f'min_tokens:{output_tokens}',
                '--extra-inputs', 'ignore_eos:true', '--concurrency',
                str(concurrency), '--warmup-request-count',
                str(warmup_request_count)
            ])

            # Use request-count or num-dataset-entries
            if request_count is not None:
                aiperf_cmd.extend(['--request-count', str(request_count)])
            else:
                # Default: use num-dataset-entries for compatibility
                aiperf_cmd.extend(['--num-dataset-entries', '64'])

            aiperf_cmd.extend([
                '--random-seed',
                str(random_seed), '--artifact-dir', artifact_dir
            ])

            # Run aiperf
            check_call(aiperf_cmd,
                       env=env,
                       poll_procs=[workers_proc, server_proc])

            if accuracy_test:
                accuracy_test_result, accuracy_value = run_accuracy_test(
                    model_path=model_path,
                    server_url=f"http://{server_host}:{server_port}",
                    concurrency=concurrency,
                    max_retries=3,
                    timeout=1200,
                    max_gen_toks=256,
                    max_length=4096)

                # only raise error if accuracy test passed and accuracy value is less than threshold
                if accuracy_test_result and (accuracy_value < threshold):
                    raise AssertionError(
                        f"Accuracy test failed: accuracy value {accuracy_value} is less than test threshold {threshold}"
                    )

    except Exception:
        # Print outputs on error
        logger.error("-------- Workers output (last 30 lines) --------")
        try:
            with open('output_workers.log', 'r') as f:
                lines = f.read().split('\n')
                for line in lines[-30:]:
                    if line.strip():
                        logger.error(line)
        except FileNotFoundError:
            pass

        logger.error("-------- Disagg server output (last 30 lines) --------")
        try:
            with open('output_disagg.log', 'r') as f:
                lines = f.read().split('\n')
                for line in lines[-30:]:
                    if line.strip():
                        logger.error(line)
        except FileNotFoundError:
            pass
        raise
    finally:
        server_proc.terminate()
        workers_proc.terminate()
        server_proc.wait()
        workers_proc.wait()


def run_accuracy_test(model_path: str, server_url: str, concurrency: int,
                      max_retries: int, timeout: int, max_gen_toks: int,
                      max_length: int) -> tuple[bool, float]:
    """
    Run accuracy test using lm_eval with GSM8K dataset

    Args:
        model_path: Path of the model being tested
        server_config: Server configuration containing URL and port
        concurrency: Concurrency for accuracy tests
        max_retries: Max retries for accuracy tests
        timeout: Timeout for accuracy tests
        max_gen_toks: Max generation tokens for accuracy tests
        max_length: Max length for accuracy tests

    Returns:
        tuple: (Boolean indicating whether the accuracy test completed successfully, accuracy value)
    """
    logger.info(f"=== Running ACCURACY TEST (GSM8K) ===")

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_gsm8k_local_config = os.path.join(tmp_dir.name, "gsm8k_local.yaml")

    gsm8k_local_config_path = os.path.join(
        os.path.dirname(__file__), '../../lm_eval_configs/gsm8k_local.yaml')

    with open(gsm8k_local_config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()

    # Replace LLM_MODELS_ROOT with actual path
    config_content = config_content.replace('LLM_MODELS_ROOT',
                                            llm_models_root())

    # Write modified config to temp file
    with open(tmp_gsm8k_local_config, 'w', encoding='utf-8') as f:
        f.write(config_content)

    # Create lm_eval command
    lm_eval_cmd = [
        "lm_eval",
        "--model",
        "local-completions",
        "--tasks",
        "gsm8k_local",
        "--include_path",
        tmp_dir.name,
        "--model_args",
        f"model={model_path},base_url={server_url}/v1/completions,"
        f"num_concurrent={concurrency},"
        f"max_retries={max_retries},"
        f"tokenized_requests=False,"
        f"timeout={timeout},"
        f"max_gen_toks={max_gen_toks},"
        f"max_length={max_length}",
    ]

    test_start_time = time.time()
    accuracy_value = 0.0

    try:
        # Run lm_eval process with timeout monitoring
        print_info(f"Running lm_eval command: {' '.join(lm_eval_cmd)}")

        # Use subprocess.run to capture output directly
        result = subprocess.run(lm_eval_cmd,
                                capture_output=True,
                                text=True,
                                timeout=timeout)

        print_info(f"Accuracy test result is: {result}")

        # Check if process completed successfully
        if result.returncode == 0:
            test_end_time = time.time()
            duration = int(test_end_time - test_start_time)
            logger.info(
                f"Accuracy test completed successfully in {duration} seconds")

            # Parse accuracy value from output
            output_text = result.stdout
            accuracy_value = parse_gsm8k_output(output_text)
            if accuracy_value is not None:
                return True, accuracy_value
            else:
                return False, accuracy_value
        else:
            logger.warning(
                f"lm_eval exited with non-zero code: {result.returncode}")
            logger.warning(f"stderr: {result.stderr}")
            return False, accuracy_value

    except subprocess.TimeoutExpired:
        logger.warning(f"Accuracy test timed out after {timeout} seconds")
        return False, accuracy_value
    except Exception as e:
        logger.warning(f"Error during accuracy test: {str(e)}")
        return False, accuracy_value


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


@pytest.mark.parametrize("benchmark_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_empty_batch(
        disaggregated_example_root, llm_venv, benchmark_model_root,
        benchmark_root, shared_gpt_path):

    src_dst_dict = {
        benchmark_model_root:
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/bf16",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    test_desc = "deepseek_v3_lite_bf16_empty_batch"
    num_ranks, config_file = get_test_config(test_desc,
                                             disaggregated_example_root,
                                             os.path.dirname(__file__))

    env = llm_venv._new_env.copy()
    e2el, ttft = run_disaggregated_benchmark(
        disaggregated_example_root,
        config_file,
        benchmark_root,
        benchmark_model_root,
        shared_gpt_path,
        env=env,
        cwd=llm_venv.get_working_directory(),
        num_ranks=num_ranks,
        num_prompts=10,
        max_concurrency=10,
        random_input_len=384,
        random_output_len=1536,
        skip_warmup=True)
    print(f"E2EL: {e2el} ms, TTFT: {ttft} ms")

    assert e2el > 0 and ttft > 0


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(140000)
@pytest.mark.parametrize(
    "model_path",
    ['llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8'])
def test_llama4_long_context_kv_cache_overflow(disaggregated_test_root,
                                               disaggregated_example_root,
                                               llm_venv, model_path):
    """
    RCCA: https://nvbugspro.nvidia.com/bug/5555681
    Test to reproduce KV cache buffer overflow bug with long context.
    """
    models_root = llm_models_root()
    llama4_model_root = os.path.join(models_root, model_path)

    # Create symlink to match config file path
    src_dst_dict = {
        llama4_model_root: f"{llm_venv.get_working_directory()}/{model_path}",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    num_ranks, config_file = get_test_config("llama4_kv_cache_overflow",
                                             disaggregated_example_root,
                                             os.path.dirname(__file__))

    run_disaggregated_aiperf(config_file=config_file,
                             model_path=llama4_model_root,
                             num_ranks=num_ranks,
                             server_start_timeout=1200,
                             input_tokens=128000,
                             output_tokens=100,
                             env=llm_venv._new_env,
                             cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_tllm_gen_helix(
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
                           "deepseek_v3_lite_bf16_tllm_gen_helix",
                           env=llm_venv._new_env,
                           cwd=llm_venv.get_working_directory(),
                           prompt_file="long_prompts.json")


@pytest.mark.timeout(12600)
@pytest.mark.parametrize("test_config", [
    pytest.param(TestConfig(model_path='DeepSeek-R1/DeepSeek-R1-0528-FP4-v2',
                            test_desc='deepseek_r1_v2_fp4_stress',
                            request_count=35000,
                            accuracy_threshold=0.92),
                 marks=(pytest.mark.skip_less_device(8), skip_pre_blackwell)),
    pytest.param(TestConfig(model_path='gpt_oss/gpt-oss-120b',
                            test_desc='gpt_oss_120b_stress',
                            request_count=60000,
                            accuracy_threshold=0.42),
                 marks=(pytest.mark.skip_less_device(4), skip_pre_blackwell)),
],
                         ids=lambda x: x.test_desc)
@pytest.mark.parametrize("concurrency", [512], ids=lambda x: f"conc{x}")
@pytest.mark.parametrize("output_tokens", [1024],
                         ids=lambda x: f"output{x//1000}k")
@pytest.mark.parametrize("input_tokens", [8192],
                         ids=lambda x: f"input{x//1000}k")
def test_disaggregated_stress_test(disaggregated_test_root,
                                   disaggregated_example_root, llm_venv,
                                   test_config, input_tokens, output_tokens,
                                   concurrency):
    # Unpack configuration from dataclass
    model_path = test_config.model_path
    test_desc = test_config.test_desc
    model_dir = f"{llm_models_root()}/{model_path}"
    src_dst_dict = {
        model_dir: f"{llm_venv.get_working_directory()}/{model_path}",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    num_ranks, config_file = get_test_config(test_desc,
                                             disaggregated_example_root,
                                             os.path.dirname(__file__))

    run_disaggregated_aiperf(config_file=config_file,
                             model_path=model_dir,
                             num_ranks=num_ranks,
                             server_start_timeout=7200,
                             input_tokens=input_tokens,
                             output_tokens=output_tokens,
                             concurrency=concurrency,
                             endpoint_type='completions',
                             request_count=test_config.request_count,
                             warmup_request_count=10,
                             streaming=False,
                             accuracy_test=True,
                             threshold=test_config.accuracy_threshold,
                             env=llm_venv._new_env,
                             cwd=llm_venv.get_working_directory())
