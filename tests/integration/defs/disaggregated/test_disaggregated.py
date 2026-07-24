# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp
import numpy as np
import pytest
import yaml
from defs.common import get_free_port_in_ci as get_free_port
from defs.common import (parse_gsm8k_output, resolve_llm_model_path,
                         wait_for_server)
from defs.conftest import (get_sm_version, llm_models_root, skip_arm,
                           skip_no_hopper, skip_pre_blackwell, skip_pre_hopper)
from defs.trt_test_alternative import check_call, check_output, print_info
from disagg_test_utils import (ProcessWrapper, run_ctx_worker,
                               run_disagg_server, run_gen_worker, terminate,
                               wait_for_disagg_server_ready)
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
    incomplete_threshold: float = 1.0
    speculative_model_path: Optional[str] = None
    cancellation_rate: Optional[int] = None
    cancellation_delay: Optional[float] = None
    concurrency: int = 512

    def __str__(self):
        return self.test_desc


def get_ucx_tls():
    """Get UCX_TLS value based on GPU architecture.

    Pre-Hopper GPUs need cuda_ipc excluded from UCX transports.
    """
    sm = get_sm_version()
    """
    ON some gb300 cluster,  we need to set `cuda_copy,cuda_ipc,sm,self,tcp` for UCX_TLS
    """
    if sm == 103 and "aarch" in platform.machine().lower():
        return "cuda_copy,cuda_ipc,sm,self,tcp"
    if sm < 90:
        return "^cuda_ipc,ib,gdr_copy"
    return "^ib,gdr_copy"


def cleanup_output_files():
    """Clean up output files from previous runs."""
    for file in ['output.json', 'output_streaming.json']:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


# Fatal patterns whose presence in worker/server logs after a stress run
# indicates the cluster did not stay healthy and the test should be failed.
# Only genuinely fatal conditions belong here — transient request-level errors
# like "Cluster is not ready" / "Internal server error" are (a) expected during
# the autotuner warmup phase (whose results are discarded) and (b) already
# gated for the measured phase by incomplete_rate/server_rejected, so scanning
# the whole log for them would false-positive on warmup churn.
_FATAL_LOG_PATTERNS = (
    "Hang detected on rank",
    "out of memory",
)


def scan_logs_for_fatal_errors(processes):
    """Scan saved process logs for fatal disagg/worker error patterns.

    Returns a dict mapping log path -> {pattern: count} for any pattern that
    appears at least once. Skips processes that did not save a log file.
    """
    findings: dict[str, dict[str, int]] = {}
    for proc in processes:
        log_path = getattr(proc, "log_path", None)
        if not log_path or not os.path.exists(log_path):
            continue
        counts = {pat: 0 for pat in _FATAL_LOG_PATTERNS}
        try:
            with open(log_path, "r", errors="replace") as f:
                for line in f:
                    for pat in _FATAL_LOG_PATTERNS:
                        if pat in line:
                            counts[pat] += 1
        except OSError:
            continue
        hits = {pat: c for pat, c in counts.items() if c > 0}
        if hits:
            findings[log_path] = hits
    return findings


def _crashed_workers(workers):
    return [
        w for w in workers
        if w.process.poll() is not None and w.process.poll() != 0
    ]


def build_worker_diag(workers, disagg_server):
    """Check worker processes and logs for crashes/fatal errors.

    Returns a diagnostic string describing any problems found, or an empty
    string if all workers are healthy.
    """
    all_procs = list(workers) + [disagg_server]
    crashed = _crashed_workers(workers)
    fatal = scan_logs_for_fatal_errors(all_procs)
    diag = ""
    if crashed:
        parts = [
            f"{w.log_path or 'worker'} (rc={w.process.poll()})" for w in crashed
        ]
        diag += f" Workers exited abnormally: {parts}."
    if fatal:
        diag += f" Fatal log findings: {fatal}."
    return diag


def get_default_disagg_cluster_config():
    """Get default disaggregated cluster configuration."""
    return {
        "cluster_name": "test_cluster",
        "heartbeat_interval_sec": 1,
        "inactive_timeout_sec": 2
    }


def build_worker_config(base_config: dict[str, Any],
                        server_type_config: dict[str, Any],
                        disagg_cluster: dict[str, Any]) -> dict[str, Any]:
    """
    Build worker configuration by merging base config with server-type specific config.

    Args:
        base_config: Full YAML config (top-level)
        server_type_config: context_servers or generation_servers section
        disagg_cluster: Service discovery config (injected by test)

    Returns:
        dict: Worker configuration for trtllm-serve
    """
    # Fields to exclude from worker configs (not worker execution settings)
    EXCLUDE_FROM_WORKER = {
        'hostname',
        'port',
        'num_workers',
        'num_instances',
        'urls',
        'router',
        'model',
        'context_servers',
        'generation_servers',
        'conditional_disagg_config',
    }

    # Start with top-level fields (exclude server-only)
    worker_config = {
        k: v
        for k, v in base_config.items() if k not in EXCLUDE_FROM_WORKER
    }

    # Merge server-type specific config (overrides top-level)
    worker_config.update({
        k: v
        for k, v in server_type_config.items() if k not in EXCLUDE_FROM_WORKER
    })

    # Convert top-level free_gpu_memory_fraction into kv_cache_config
    if 'free_gpu_memory_fraction' in worker_config:
        frac = worker_config.pop('free_gpu_memory_fraction')
        if 'kv_cache_config' not in worker_config:
            worker_config['kv_cache_config'] = {}
        worker_config['kv_cache_config'].setdefault('free_gpu_memory_fraction',
                                                    frac)

    # Add service discovery config
    worker_config['disagg_cluster'] = disagg_cluster

    return worker_config


def get_test_config(test_desc, example_dir, test_root):
    """Get config file path for a test description."""
    test_configs_root = f"{test_root}/test_configs"
    config_map = {
        "2_ranks_diff_max_tokens":
        f"{test_configs_root}/disagg_config_diff_max_tokens.yaml",
        "2_ranks":
        f"{test_configs_root}/disagg_config.yaml",
        "gen_only":
        f"{test_configs_root}/disagg_config_gen_only.yaml",
        "gen_only_bs1":
        f"{test_configs_root}/disagg_config_gen_only_bs1.yaml",
        "gen_only_insufficient_kv":
        f"{test_configs_root}/disagg_config_gen_only_insufficient_kv.yaml",
        "kv_cache_aware":
        f"{test_configs_root}/disagg_config_gen_only_kv_cache_aware.yaml",
        "round_robin":
        f"{test_configs_root}/disagg_config_round_robin.yaml",
        "load_balancing":
        f"{test_configs_root}/disagg_config_load_balancing.yaml",
        "conversation":
        f"{test_configs_root}/disagg_config_conversation.yaml",
        "multi_orchestrator":
        f"{test_configs_root}/disagg_config_multi_orchestrator.yaml",
        "4_ranks":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp1.yaml",
        "cuda_graph":
        f"{test_configs_root}/disagg_config_cuda_graph_padding.yaml",
        "mixed":
        f"{test_configs_root}/disagg_config_mixed.yaml",
        "overlap":
        f"{test_configs_root}/disagg_config_overlap.yaml",
        "overlap_gen_first":
        f"{test_configs_root}/disagg_config_overlap_gen_first.yaml",
        "overlap_gen_first_pp4":
        f"{test_configs_root}/disagg_config_overlap_gen_first_pp4.yaml",
        "overlap_transceiver_runtime_python":
        f"{test_configs_root}/disagg_config_overlap_transceiver_runtime_python.yaml",
        "overlap_transceiver_runtime_python_bounce":
        f"{test_configs_root}/disagg_config_overlap_transceiver_runtime_python_bounce.yaml",
        "python_transceiver_host_offload":
        f"{test_configs_root}/disagg_config_python_transceiver_host_offload.yaml",
        "tool_calls":
        f"{test_configs_root}/disagg_config_overlap.yaml",
        "perf_metrics":
        f"{test_configs_root}/disagg_config_metrics.yaml",
        "load_balance":
        f"{test_configs_root}/disagg_config_load_balance.yaml",
        "cache_aware_balance":
        f"{test_configs_root}/disagg_config_cache_aware_balance.yaml",
        "conditional":
        f"{test_configs_root}/disagg_config_conditional.yaml",
        "ngram":
        f"{test_configs_root}/disagg_config_ngram.yaml",
        "ctxpp2_genpp2":
        f"{test_configs_root}/disagg_config_ctxpp2_genpp2.yaml",
        "ctxtp2_genpp2":
        f"{test_configs_root}/disagg_config_ctxtp2_genpp2.yaml",
        "ctxpp2_gentp2":
        f"{test_configs_root}/disagg_config_ctxpp2_gentp2.yaml",
        "ctxtp2pp2_gentp2pp2":
        f"{test_configs_root}/disagg_config_ctxtp2pp2_gentp2pp2.yaml",
        "ctxpp4_genpp4":
        f"{test_configs_root}/disagg_config_ctxpp4_genpp4.yaml",
        "ctxpp4_gentp4":
        f"{test_configs_root}/disagg_config_ctxpp4_gentp4.yaml",
        "deepseek_v3_lite_fp8_mpi":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_mpi.yaml",
        "deepseek_v3_lite_fp8_ucx":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_ucx.yaml",
        "deepseek_v3_lite_fp8_nixl":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_nixl.yaml",
        "deepseek_v3_lite_fp8_transceiver_runtime_python":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_transceiver_runtime_python.yaml",
        "deepseek_v3_lite_fp8_tp1":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite.yaml",
        "deepseek_v3_lite_fp8_tp1_mtp":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp.yaml",
        "deepseek_v3_lite_fp8_attention_dp":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp.yaml",
        "deepseek_v3_lite_fp8_attention_dp_gen_only":
        f"{test_configs_root}/disagg_config_gentp2_deepseek_v3_lite_attention_dp_gen_only.yaml",
        "deepseek_v3_lite_fp_8_attention_dp_overlap":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_overlap.yaml",
        "deepseek_v3_lite_fp8_attention_dp_overlap_cuda_graph":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_overlap_cuda_graph.yaml",
        "deepseek_v3_lite_fp8_overlap_cuda_graph":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_overlap_cuda_graph.yaml",
        "deepseek_v3_lite_fp8_attention_dp_one":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one.yaml",
        "deepseek_v3_lite_fp8_attention_dp_one_mtp":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one_mtp.yaml",
        "deepseek_v3_lite_fp8_tp1_attention_dp_overlap_one_mtp":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp_attention_dp_overlap.yaml",
        "deepseek_v3_lite_bf16_cache_aware_balance":
        f"{test_configs_root}/disagg_config_cache_aware_balance_deepseek_v3.yaml",
        "deepseek_v3_lite_bf16_conditional":
        f"{test_configs_root}/disagg_config_conditional_deepseek_v3.yaml",
        "deepseek_v3_lite_bf16_conditional_v2":
        f"{test_configs_root}/disagg_config_conditional_deepseek_v3_v2.yaml",
        "deepseek_v3_lite_fp8_tp1_two_mtp":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_two_mtp.yaml",
        "deepseek_v3_lite_fp8_ctxpp2_gentp2_one_mtp":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp_ctxpp2_gentp2.yaml",
        "deepseek_v3_lite_fp8_ctxtp2ep2pp2_gentp4_one_mtp_block_reuse":
        f"{test_configs_root}/disagg_config_ctxtp2ep2pp2_gentp4_deepseek_v3_lite_one_mtp_block_reuse.yaml",
        "deepseek_v3_lite_fp8_ctxtp2ep2pp2_gentp4_one_mtp_block_reuse_chunked":
        f"{test_configs_root}/disagg_config_ctxtp2ep2pp2_gentp4_deepseek_v3_lite_one_mtp_block_reuse_chunked.yaml",
        "deepseek_v3_lite_bf16_empty_batch":
        f"{test_configs_root}/disagg_config_deepseek_v3_lite_empty_batch.yaml",
        "llama4_kv_cache_overflow":
        f"{test_configs_root}/disagg_config_llama4_kv_cache_overflow.yaml",
        "deepseek_v3_lite_bf16_tllm_gen_helix":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp1cp2_deepseek_v3_lite_bf16_tllm_gen.yaml",
        "deepseek_r1_v2_fp4_stress":
        f"{test_configs_root}/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml",
        "deepseek_r1_v2_fp4_mtp_stress":
        f"{test_configs_root}/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm_mtp.yaml",
        "gpt_oss_120b_trtllm_stress":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml",
        "gpt_oss_120b_eagle_triton_stress":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_gptoss_eagle_triton.yaml",
        "gpt_oss_120b_eagle_trtllm_stress":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml",
        "gpt_oss_120b_triton_stress":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_gptoss_triton.yaml",
        "qwen3_5_4b_fp8_stress":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp1_qwen3_5_4b_fp8_tllm.yaml",
        "glm5_nvfp4_tp4_ep4_dp_stress":
        f"{test_configs_root}/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml",
        "qwen3_32b_fp8_stress":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp4_qwen3_32b_fp8.yaml",
        "req120-conc64-qwen3_32b_fp8_mixed_stress":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp4_qwen3_32b_fp8.yaml",
        "req10k-conc512-qwen3_32b_fp8_mixed_stress":
        f"{test_configs_root}/disagg_config_ctxtp1_gentp4_qwen3_32b_fp8.yaml",
        "gpt_oss_120b_harmony":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml",
        "cancel_stress_test":
        f"{test_configs_root}/disagg_config_cancel_stress_test.yaml",
        "cancel_stress_test_small":
        f"{test_configs_root}/disagg_config_cancel_stress_test_small.yaml",
        "cancel_stress_test_large":
        f"{test_configs_root}/disagg_config_cancel_stress_test_large.yaml",
        "llama31_8b_ucx":
        f"{test_configs_root}/disagg_config_ctxtp2_gentp2_llama31_8b_ucx.yaml",
        "mamba_conc_greater_than_mbs":
        f"{test_configs_root}/disagg_config_mamba_conc_greater_than_mbs.yaml",
    }

    if test_desc not in config_map:
        raise ValueError(f"Invalid test description: {test_desc}, "
                         f"valid descriptions are: {config_map.keys()}")

    return config_map[test_desc]


def setup_model_symlink(llm_venv, model_root, dest_subpath):
    """Create symlink for model in test working directory.

    Args:
        llm_venv: Virtual environment object with get_working_directory()
        model_root: Source model directory path
        dest_subpath: Destination subdirectory (relative to working dir)
    """
    dst = f"{llm_venv.get_working_directory()}/{dest_subpath}"
    if not os.path.islink(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(model_root, dst, target_is_directory=True)


ClientTestSet = namedtuple('ClientTestSet', [
    'completion', 'completion_streaming', 'chat', 'chat_streaming',
    'verify_completion', 'verify_streaming_completion', 'verify_chat',
    'verify_streaming_chat'
])


def get_client_test_set(test_desc):
    """Get the set of client tests to run for a given test description."""
    if test_desc == "tool_calls":
        return ClientTestSet(completion=False,
                             completion_streaming=False,
                             chat=True,
                             chat_streaming=False,
                             verify_completion=False,
                             verify_streaming_completion=False,
                             verify_chat=False,
                             verify_streaming_chat=False)
    if test_desc == "gpt_oss_120b_harmony":
        return ClientTestSet(completion=True,
                             completion_streaming=True,
                             chat=True,
                             chat_streaming=True,
                             verify_completion=True,
                             verify_streaming_completion=True,
                             verify_chat=False,
                             verify_streaming_chat=False)
    if test_desc.startswith("overlap"):
        return ClientTestSet(completion=True,
                             completion_streaming=True,
                             chat=True,
                             chat_streaming=True,
                             verify_completion=True,
                             verify_streaming_completion=True,
                             verify_chat=True,
                             verify_streaming_chat=False)
    return ClientTestSet(completion=True,
                         completion_streaming=True,
                         chat=False,
                         chat_streaming=False,
                         verify_completion=True,
                         verify_streaming_completion=True,
                         verify_chat=False,
                         verify_streaming_chat=False)


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
                     use_ray=False,
                     client_test_set=None):
    """Run client tests against the disaggregated server."""
    if client_test_set is None:
        client_test_set = get_client_test_set(test_desc)

    client_dir = f"{example_dir}/clients"
    for _ in range(num_iters):
        client_cmd = [
            'python3', f'{client_dir}/disagg_client.py', '-c', f'{config_file}',
            '-p', f'{client_dir}/{prompt_file}', '--ignore-eos',
            '--server-start-timeout',
            str(server_start_timeout)
        ]
        if prompt_file.startswith("long_"):
            # Use max_tokens 4 for long prompts to reduce test time
            client_cmd.extend(['--max-tokens', '4'])

        # Prepare poll processes
        worker_processes = []
        if use_ray:
            for proc in workers_proc:
                # Ray passes context managers, SD passes raw Popen objects
                if hasattr(proc, '__enter__'):
                    worker_processes.append(proc.__enter__())
                else:
                    worker_processes.append(proc)
        else:
            worker_processes = [workers_proc]

        poll_procs = worker_processes + [server_proc]

        # Run completion test (non-streaming)
        if client_test_set.completion:
            check_call(client_cmd, env=env, poll_procs=poll_procs)

        # Streaming client run
        if client_test_set.completion_streaming:
            streaming_client_cmd = client_cmd + [
                '--streaming', '-o', 'output_streaming.json'
            ]
            check_call(streaming_client_cmd, env=env, poll_procs=poll_procs)

        # Run chat completion test
        if client_test_set.chat:
            chat_output = 'output_tool_calls.json' if test_desc == "tool_calls" else 'output_chat.json'
            chat_client_cmd = client_cmd + ['-e', 'chat', '-o', chat_output]
            check_call(chat_client_cmd, env=env, poll_procs=poll_procs)

        # Run streaming chat completion test
        if client_test_set.chat_streaming:
            streaming_chat_client_cmd = client_cmd + [
                '-e', 'chat', '--streaming', '-o', 'output_streaming_chat.json'
            ]
            check_call(streaming_chat_client_cmd,
                       env=env,
                       poll_procs=poll_procs)

        # Skip output verification for long prompts or tool call tests
        if prompt_file.startswith(
                "long_") or prompt_file == "tool_call_prompts.json":
            continue

        if extra_endpoints_test is not None:
            extra_endpoints_test(server_url)

        # Verify outputs
        not_expected_strings = ["Berlin Berlin"]

        output_files = []
        if client_test_set.completion and client_test_set.verify_completion:
            output_files.append('output.json')
        if client_test_set.completion_streaming and client_test_set.verify_streaming_completion:
            output_files.append('output_streaming.json')
        if client_test_set.chat and client_test_set.verify_chat:
            # Streaming chat completion output not verified due to known bug
            output_files.append('output_chat.json')
        if client_test_set.chat_streaming and client_test_set.verify_streaming_chat:
            output_files.append('output_streaming_chat.json')

        if test_desc.endswith("gen_only") or test_desc.startswith("gen_only"):
            continue

        for output_file in output_files:
            with open(output_file, 'r') as f:
                content = f.read()
                if "deepseek_v3_lite" in test_desc or output_file == "output_chat.json":
                    expected_strings = [
                        "Berlin", ["Asyncio is a", "Asyncio module in"]
                    ]
                elif "gpt_oss_120b" in test_desc:
                    expected_strings = [
                        "The capital of Germany is Berlin",
                        "Using `asyncio` in Python"
                    ]
                elif "qwen3_32b_fp8" in test_desc:
                    expected_strings = [
                        "The capital of Germany is Berlin",
                        "Asyncio in Python is a library"
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


def verify_usage_with_cache_reuse(server_url: str, model: str):
    """Verify repeated requests report context-side usage after cache reuse."""
    prompt = "Explain why a repeated disaggregated request should reuse cached KV blocks."
    max_tokens = 4
    timeout = aiohttp.ClientTimeout(total=120)

    async def send_request(session: aiohttp.ClientSession, endpoint: str):
        if endpoint == "completions":
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            }
        else:
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                }],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            }

        async with session.post(f"{server_url}/v1/{endpoint}",
                                json=payload,
                                timeout=timeout) as resp:
            assert resp.status == 200, \
                f"{endpoint} request failed with {resp.status}: {await resp.text()}"
            return await resp.json()

    def validate_usage(endpoint: str, first_response: dict[str, Any],
                       second_response: dict[str, Any]):
        first_usage = first_response.get("usage")
        second_usage = second_response.get("usage")
        print(f"[usage_check] {endpoint} first_usage={first_usage}")
        print(f"[usage_check] {endpoint} second_usage={second_usage}")
        assert first_usage is not None, f"{endpoint} first response missing usage"
        assert second_usage is not None, f"{endpoint} second response missing usage"
        assert second_usage["prompt_tokens"] == first_usage["prompt_tokens"], \
            (f"{endpoint} prompt_tokens mismatch: second={second_usage['prompt_tokens']} "
             f"!= first={first_usage['prompt_tokens']}")
        assert second_usage["completion_tokens"] == max_tokens, \
            (f"{endpoint} completion_tokens mismatch: "
             f"got={second_usage['completion_tokens']} expected={max_tokens}")
        assert second_usage["total_tokens"] == (
            second_usage["prompt_tokens"] + second_usage["completion_tokens"]), \
            (f"{endpoint} total_tokens mismatch: "
             f"got={second_usage['total_tokens']} expected="
             f"{second_usage['prompt_tokens'] + second_usage['completion_tokens']}")

        prompt_tokens_details = second_usage.get("prompt_tokens_details")
        assert prompt_tokens_details is not None, \
            f"{endpoint} second response missing prompt_tokens_details"
        print(
            f"[usage_check] {endpoint} prompt_tokens_details={prompt_tokens_details}"
        )
        assert prompt_tokens_details["cached_tokens"] == (
            second_usage["prompt_tokens"] - 1), \
            (f"{endpoint} cached_tokens mismatch: "
             f"got={prompt_tokens_details['cached_tokens']} "
             f"expected={second_usage['prompt_tokens'] - 1}")

    async def check_usage():
        async with aiohttp.ClientSession() as session:
            for endpoint in ("completions", "chat/completions"):
                first_response = await send_request(session, endpoint)
                second_response = await send_request(session, endpoint)
                validate_usage(endpoint, first_response, second_response)

    asyncio.run(check_usage())


# TODO: add test for disaggregated server prometheus metrics
def fetch_prometheus_metrics(server_url: str):
    import requests
    response = requests.get(f"{server_url}/prometheus/metrics", timeout=10)
    assert response.status_code == 200
    return response.text


def setup_disagg_cluster(
    config_file: str,
    model_name: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    server_start_timeout: int = 300,
    schedule_style: str | None = None,
    save_log: bool = False,
    startup_callback=None,
    startup_tick: int = 30,
) -> tuple[dict[str, Any], list[ProcessWrapper], list[ProcessWrapper],
           ProcessWrapper, int, str]:
    """Load config, launch workers + disagg server, wait for ready.

    Args:
        config_file: Path to disaggregated server config YAML
        model_name: Model path override (defaults to config's 'model' field)
        env: Environment variables to pass to subprocess (workers and disagg server)
        server_start_timeout: Timeout in seconds for server to become ready
        schedule_style: Disagg schedule style ('context_first' or 'generation_first')

    Returns:
        tuple: (config, ctx_workers, gen_workers, disagg_server, server_port, work_dir)
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    speculative_config = config.get("speculative_config")
    if isinstance(speculative_config, dict):
        speculative_model = speculative_config.get("speculative_model")
        if speculative_model:
            speculative_config["speculative_model"] = resolve_llm_model_path(
                speculative_model)

    disagg_cluster = get_default_disagg_cluster_config()
    server_host = config.get("hostname", "localhost")
    server_port = get_free_port()
    if save_log:
        log_base = os.path.join(cwd or ".", "disagg-logs")
        os.makedirs(log_base, exist_ok=True)
        work_dir = tempfile.mkdtemp(dir=log_base)
    else:
        work_dir = tempfile.mkdtemp()
    logger.info(f"Disagg cluster work_dir (worker logs): {work_dir}")
    server_env = env
    coordinator_url = f"http://{server_host}:{server_port}"
    if config.get("num_workers", 1) > 1:
        coordinator_port = get_free_port()
        while coordinator_port == server_port:
            coordinator_port = get_free_port()
        coordinator_url = f"http://{server_host}:{coordinator_port}"
        server_env = (env or os.environ).copy()
        server_env["TRTLLM_DISAGG_COORDINATOR_PORT"] = str(coordinator_port)
    disagg_cluster["cluster_uri"] = coordinator_url

    # Auto-deduce minimal_instances from num_instances
    ctx_servers = config.get("context_servers", {})
    gen_servers = config.get("generation_servers", {})
    num_ctx_instances = ctx_servers.get("num_instances", 1)
    num_gen_instances = gen_servers.get("num_instances", 1)
    disagg_cluster["minimal_instances"] = {
        "context_servers": num_ctx_instances,
        "generation_servers": num_gen_instances
    }

    # Calculate GPUs per worker instance: tp * pp * cp
    gpus_per_ctx = (ctx_servers.get("tensor_parallel_size", 1) *
                    ctx_servers.get("pipeline_parallel_size", 1) *
                    ctx_servers.get("context_parallel_size", 1))
    gpus_per_gen = (gen_servers.get("tensor_parallel_size", 1) *
                    gen_servers.get("pipeline_parallel_size", 1) *
                    gen_servers.get("context_parallel_size", 1))

    # Build worker configs
    ctx_worker_config = build_worker_config(config, ctx_servers, disagg_cluster)
    gen_worker_config = build_worker_config(config, gen_servers, disagg_cluster)

    # Launch workers
    model = model_name or config.get("model")
    if model:
        model = resolve_llm_model_path(model)
    ctx_workers = []
    gen_workers = []
    disagg_server = None
    next_device = 0

    import torch
    num_gpus = torch.cuda.device_count()

    try:
        for i in range(num_ctx_instances):
            device_ids = ",".join(
                str(d) for d in dict.fromkeys((next_device + j) % num_gpus
                                              for j in range(gpus_per_ctx)))
            w = run_ctx_worker(model,
                               ctx_worker_config,
                               work_dir,
                               port=0,
                               device=device_ids,
                               env=env,
                               save_log=save_log,
                               worker_index=i)
            ctx_workers.append(w)
            log_suffix = f", logging to {w.log_path}" if w.log_path else ""
            print(
                f"Launching ctx worker {i + 1}/{num_ctx_instances} on device {device_ids}{log_suffix}"
            )
            next_device += gpus_per_ctx

        for i in range(num_gen_instances):
            device_ids = ",".join(
                str(d) for d in dict.fromkeys((next_device + j) % num_gpus
                                              for j in range(gpus_per_gen)))
            w = run_gen_worker(model,
                               gen_worker_config,
                               work_dir,
                               port=0,
                               device=device_ids,
                               env=env,
                               save_log=save_log,
                               worker_index=i)
            gen_workers.append(w)
            log_suffix = f", logging to {w.log_path}" if w.log_path else ""
            print(
                f"Launching gen worker {i + 1}/{num_gen_instances} on device {device_ids}{log_suffix}"
            )
            next_device += gpus_per_gen

        # Build minimal server config and launch
        server_config = {
            "hostname":
            server_host,
            "port":
            server_port,
            "num_workers":
            config.get("num_workers", 1),
            "disagg_cluster":
            disagg_cluster,
            "context_servers": {
                "router": ctx_servers.get("router", {})
            },
            "generation_servers": {
                "router": gen_servers.get("router", {})
            },
            "conditional_disagg_config":
            config.get("conditional_disagg_config", None),
            "perf_metrics_max_requests":
            config.get("perf_metrics_max_requests", 0),
        }
        if schedule_style:
            server_config["schedule_style"] = schedule_style
        disagg_server = run_disagg_server(server_config,
                                          work_dir,
                                          server_port,
                                          save_log=save_log,
                                          env=server_env,
                                          cwd=cwd)

        all_workers = ctx_workers + gen_workers

        async def _wait_with_ticker():
            start = time.monotonic()
            last_tick = start

            async def _tick():
                nonlocal last_tick
                last_worker_count = 0
                last_status_log = start

                while True:
                    await asyncio.sleep(1)
                    now = time.monotonic()
                    if startup_callback and now - last_tick >= startup_tick:
                        startup_callback(now - start)
                        last_tick = now

                    if now - last_status_log >= 10:
                        elapsed = int(now - start)
                        remaining = int(server_start_timeout - elapsed)
                        print(
                            f"[startup] {elapsed}s elapsed, "
                            f"{max(remaining, 0)}s left until timeout "
                            f"(timeout={server_start_timeout}s, "
                            f"workers registered: {last_worker_count}/"
                            f"{num_ctx_instances + num_gen_instances})",
                            flush=True,
                        )
                        last_status_log = now

                    # Detect workers that have exited (crash/OOM).
                    dead = [
                        w for w in all_workers if w.process.poll() is not None
                    ]
                    if dead:
                        details = ", ".join(
                            f"{w.log_path or 'unknown'} (rc={w.process.poll()})"
                            for w in dead)
                        raise RuntimeError(
                            f"{len(dead)} worker(s) exited during startup: "
                            f"{details}")

                    # Track worker registration count for the status log.
                    try:
                        timeout = aiohttp.ClientTimeout(total=2)
                        async with aiohttp.ClientSession(
                                timeout=timeout) as session:
                            async with session.get(
                                    f"http://localhost:{server_port}/cluster_info"
                            ) as info_resp:
                                if info_resp.status == 200:
                                    workers = (await info_resp.json()).get(
                                        "current_workers", {})
                                    count = (len(
                                        workers.get("context_servers", [])) +
                                             len(
                                                 workers.get(
                                                     "generation_servers", [])))
                                    if count > last_worker_count:
                                        last_worker_count = count
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass

            # Run the readiness poller and the watchdog concurrently.
            # asyncio.wait(FIRST_COMPLETED) means whichever finishes first
            # (success or exception) immediately cancels the other, so a dead
            # worker detected by _tick() doesn't leave wait_for_disagg_server_ready
            # polling silently for up to server_start_timeout seconds.
            ready_task = asyncio.create_task(
                wait_for_disagg_server_ready(server_port,
                                             timeout=server_start_timeout))
            tick_task = asyncio.create_task(_tick())

            done, pending = await asyncio.wait(
                [ready_task, tick_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            # Re-raise the first exception. Prefer tick_task's message (it names
            # the dead worker); fall back to ready_task (timeout error).
            for t in [tick_task, ready_task]:
                if t in done and not t.cancelled() and t.exception(
                ) is not None:
                    raise t.exception()

        asyncio.run(_wait_with_ticker())
    except Exception:
        terminate(*ctx_workers, *gen_workers, disagg_server)
        if not save_log:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise

    return config, ctx_workers, gen_workers, disagg_server, server_port, work_dir


def run_disaggregated_test(example_dir,
                           test_desc,
                           num_iters=5,
                           env=None,
                           prompt_file="prompts.json",
                           extra_endpoints_test=None,
                           model_path=None,
                           cwd=None,
                           disagg_schedule_style=None,
                           post_client_test=None,
                           assert_gen_log_contains=None):
    """Run disaggregated test using service discovery instead of MPI.

    If assert_gen_log_contains is set, the generation-worker logs are captured and, after the
    client tests, at least one of them must contain that substring (used to prove the KV-cache
    bounce path actually engaged instead of silently falling back to the per-fragment path).
    """
    if mpi_disabled():
        pytest.skip(
            "https://nvbugs/5584607 Ray orchestrator is not supported with NIXL(DEFAULT) cache transceiver backend."
        )

    run_env = env.copy() if env else os.environ.copy()
    run_env["UCX_TLS"] = get_ucx_tls()

    config_file = get_test_config(test_desc, example_dir,
                                  os.path.dirname(__file__))
    config, ctx_workers, gen_workers, disagg_server, server_port, work_dir = \
        setup_disagg_cluster(config_file, model_name=model_path, env=run_env, cwd=cwd,
                             schedule_style=disagg_schedule_style,
                             save_log=assert_gen_log_contains is not None)

    server_host = config.get("hostname", "localhost")

    try:
        server_url = f"http://{server_host}:{server_port}"

        # Create a temporary client config file with the correct server port
        client_config = config.copy()
        client_config["port"] = server_port
        client_config["hostname"] = server_host
        temp_fd, client_config_file = tempfile.mkstemp(suffix='.yaml',
                                                       dir=work_dir)
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(client_config, f)

        # collect all worker processes for monitoring
        all_worker_procs = [w.process for w in ctx_workers
                            ] + [w.process for w in gen_workers]

        # run client tests
        run_client_tests(
            example_dir,
            client_config_file,
            test_desc,
            num_iters,
            run_env,
            300,  # timeout
            prompt_file,
            extra_endpoints_test,
            server_url,
            all_worker_procs,
            disagg_server.process,
            use_ray=True)
        if post_client_test is not None:
            post_client_test(server_url)
        if assert_gen_log_contains is not None:
            # Fail loudly if the marker is absent: the transfer silently fell back to the
            # per-fragment path, so the bounce path we meant to exercise never ran.
            logs = []
            for w in gen_workers:
                if w.log_path and os.path.exists(w.log_path):
                    with open(w.log_path, 'r', errors='replace') as f:
                        logs.append(f.read())
            assert any(assert_gen_log_contains in log for log in logs), (
                f"expected marker {assert_gen_log_contains!r} in a generation-worker log, "
                f"but none of {len(logs)} log(s) contained it (bounce did not engage)"
            )
    finally:
        terminate(*ctx_workers, *gen_workers, disagg_server)
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_diff_max_tokens(disaggregated_test_root,
                                       disaggregated_example_root, llm_venv,
                                       llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "2_ranks_diff_max_tokens",
                           env=llm_venv._new_env,
                           prompt_file="long_prompts.json",
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_single_gpu(disaggregated_test_root,
                                  disaggregated_example_root, llm_venv,
                                  llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    env = llm_venv._new_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    run_disaggregated_test(disaggregated_example_root,
                           "2_ranks",
                           env=env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_tinyllama_multi_orchestrator(disaggregated_test_root,
                                                    disaggregated_example_root,
                                                    llm_venv, llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    env = llm_venv._new_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    run_disaggregated_test(disaggregated_example_root,
                           "multi_orchestrator",
                           num_iters=1,
                           env=env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_benchmark_gen_only(disaggregated_test_root,
                                          disaggregated_example_root, llm_venv,
                                          llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    env = llm_venv._new_env.copy()
    env['TRTLLM_DISAGG_BENCHMARK_GEN_ONLY'] = '1'
    run_disaggregated_test(disaggregated_example_root,
                           "gen_only",
                           env=env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("router_type",
                         ["load_balancing", "kv_cache_aware", "conversation"])
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_router(disaggregated_test_root,
                              disaggregated_example_root, llm_venv,
                              llama_model_root, router_type, tmp_path):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    metrics_file = tmp_path / f"perf_metrics_{router_type}.json"

    def fetch_perf_metrics(server_url: str):
        import json

        import requests as http_requests
        resp = http_requests.get(f"{server_url}/perf_metrics", timeout=10)
        assert resp.status_code == 200, \
            f"Failed to fetch perf_metrics: {resp.status_code}"
        metrics = resp.json()
        metrics_file.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Router={router_type}: saved {len(metrics)} perf metrics "
                    f"to {metrics_file}")

    run_disaggregated_test(disaggregated_example_root,
                           router_type,
                           env=llm_venv._new_env,
                           extra_endpoints_test=fetch_perf_metrics,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_benchmark_gen_only_insufficient_kv(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        llama_model_root):
    """Test that gen-only benchmark mode raises an error when KV cache is too small to hold all benchmark requests, instead of hanging forever."""
    import openai

    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    env = llm_venv._new_env.copy()
    env['TRTLLM_DISAGG_BENCHMARK_GEN_ONLY'] = '1'
    env['TLLM_BENCHMARK_REQ_QUEUES_SIZE'] = '64'
    env["UCX_TLS"] = get_ucx_tls()

    config_file = get_test_config("gen_only_insufficient_kv",
                                  disaggregated_example_root,
                                  os.path.dirname(__file__))
    config, ctx_workers, gen_workers, disagg_server, server_port, work_dir = \
        setup_disagg_cluster(config_file,
                             model_name=llama_model_root,
                             env=env,
                             cwd=llm_venv.get_working_directory())

    try:
        client = openai.OpenAI(api_key="tensorrt_llm",
                               base_url=f"http://localhost:{server_port}/v1")

        # Send 64 concurrent requests to trigger the benchmark fill loop
        # and the insufficient KV cache error.
        import concurrent.futures

        def send_request():
            try:
                stream = client.completions.create(
                    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    prompt="What is the capital of Germany?",
                    max_tokens=10,
                    temperature=0.0,
                    stream=True)
                # Must iterate the stream to receive SSE error chunks
                chunks = []
                for chunk in stream:
                    chunks.append(chunk.choices[0].text)
                return "".join(chunks)
            except Exception as e:
                return e

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            futures = [pool.submit(send_request) for _ in range(64)]
            results = [f.result(timeout=120) for f in futures]

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) > 0, \
            "Expected at least one error due to insufficient KV cache"
    finally:
        terminate(*ctx_workers, *gen_workers, disagg_server)
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_genbs1(disaggregated_test_root,
                              disaggregated_example_root, llm_venv,
                              llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    env = llm_venv._new_env.copy()
    env['TRTLLM_DISAGG_BENCHMARK_GEN_ONLY'] = '1'
    run_disaggregated_test(disaggregated_example_root,
                           "gen_only_bs1",
                           env=env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_multi_gpu(disaggregated_test_root,
                                 disaggregated_example_root, llm_venv,
                                 llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "4_ranks",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_cuda_graph(disaggregated_test_root, llm_venv,
                                  disaggregated_example_root, llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "cuda_graph",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_mixed(disaggregated_test_root, llm_venv,
                             disaggregated_example_root, llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "mixed",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_overlap(disaggregated_test_root, llm_venv,
                               disaggregated_example_root, llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def post_client_test(server_url: str):
        verify_usage_with_cache_reuse(server_url,
                                      "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "overlap",
                           env=llm_venv._new_env,
                           post_client_test=post_client_test,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_pre_hopper
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
@pytest.mark.parametrize("ctx_pp", [1, 4], ids=["ctx_pp1", "ctx_pp4"])
def test_disaggregated_overlap_gen_first(disaggregated_test_root,
                                         disaggregated_example_root, llm_venv,
                                         llama_model_root, ctx_pp):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    def post_client_test(server_url: str):
        verify_usage_with_cache_reuse(server_url,
                                      "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(
        disaggregated_example_root,
        "overlap_gen_first" if ctx_pp == 1 else "overlap_gen_first_pp4",
        env=llm_venv._new_env,
        model_path=llama_model_root,
        cwd=llm_venv.get_working_directory(),
        disagg_schedule_style="generation_first",
        post_client_test=post_client_test)


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_overlap_transceiver_runtime_python(
        disaggregated_test_root, llm_venv, disaggregated_example_root,
        llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    env = llm_venv._new_env.copy()
    env["UCX_TLS"] = get_ucx_tls()
    run_disaggregated_test(disaggregated_example_root,
                           "overlap_transceiver_runtime_python",
                           env=env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


# Exercises the disaggregated KV-cache transfer path with the Python cache transceiver runtime
# while the KV-cache pool itself is allocated from fabric (MNNVL) VMM memory via
# TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY=1. Restricted to GB200/GB300 since those are the only
# platforms with MNNVL fabric-memory support; on other devices the env var would silently fall
# back to a non-fabric allocation, which would defeat the purpose of this test.
@pytest.mark.skip_device_not_contain(["GB200", "GB300"])
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_overlap_transceiver_runtime_python_fabric_memory(
        disaggregated_test_root, llm_venv, disaggregated_example_root,
        llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    env = llm_venv._new_env.copy()
    env["UCX_TLS"] = get_ucx_tls()
    env["TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY"] = "1"
    run_disaggregated_test(disaggregated_example_root,
                           "overlap_transceiver_runtime_python",
                           env=env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


# Exercises the disaggregated KV-cache transfer path with the Python cache transceiver AND the
# KV-cache bounce optimization (cache_transceiver_config.kv_cache_bounce_size_mb > 0): scattered
# per-block WRITEs are gathered into one coalesced fabric-VMM buffer before a single NIXL WRITE.
# Restricted to GB200/GB300 since the bounce arena is fabric (MNNVL) VMM memory.
#
# The bounce transport coalesces a transfer only when it clears the receiver's min_blocks gate.
# The test lowers that gate via the TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS env so the ordinary short
# prompts still take the coalesced-bounce WRITE path -- no special long prompt is needed. The test
# runs the normal disagg output verification (the coalesced KV must still decode to the right
# answer, e.g. "Berlin"; a corrupt transfer would garble it) AND asserts the generation worker
# logged the coalesced-bounce marker, so a silent fall-back to the per-fragment path fails the
# test instead of passing quietly.
@pytest.mark.skip_device_not_contain(["GB200", "GB300"])
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_overlap_transceiver_runtime_python_bounce(
        disaggregated_test_root, llm_venv, disaggregated_example_root,
        llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    env = llm_venv._new_env.copy()
    env["UCX_TLS"] = get_ucx_tls()
    # min_blocks=1 forces bounce on even for the short test prompt (the gate is internal, tuned via
    # env). No fabric-pool env is needed: the bounce arena is its own fabric memory.
    env["TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS"] = "1"
    run_disaggregated_test(disaggregated_example_root,
                           "overlap_transceiver_runtime_python_bounce",
                           env=env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory(),
                           assert_gen_log_contains="[kv-bounce] coalesced")


def _verify_python_transceiver_under_host_offload(server_url: str, model: str):
    """End-to-end check: Python transceiver + ctx-side host offload.

    The fix translates logical block IDs to primary-pool slot indices in
    `_CacheReuseAdapterV1.get_block_ids` before they reach the disagg
    sender. Without that translation, once host offload moves blocks
    around, the sender computes pool pointers from stale block IDs and
    either reads garbage memory or aborts. This test stresses that path
    end-to-end:

      1. Send several distinct prompts to fill the (deliberately small)
         ctx primary pool, committing each to the reuse radix tree.
      2. Send more prompts, evicting earlier blocks to the host pool.
      3. Re-issue the earlier prompts. Reuse hits force onboard from host
         back to primary, and the disagg transfer must read primary slots
         that no longer match the original block IDs. With the fix, this
         succeeds; without it, the sender either crashes on a primary
         assertion or returns nonsense tokens.

    Assertions are deliberately content-agnostic (TinyLlama outputs vary
    run-to-run): we check that responses are non-empty, the server stays
    up across the eviction/onboard cycle, and `cached_tokens > 0` on
    repeats so we know reuse actually fired.
    """
    timeout = aiohttp.ClientTimeout(total=180)
    max_tokens = 16
    # Workload sizing: ctx-side primary pool = max_tokens(1024) /
    # tokens_per_block(64) = 16 blocks. Each prompt below tokenizes to
    # ~200 tokens ≈ 4 KV blocks. We send 6 distinct prompts → ~24 blocks
    # of primary demand > 16-block primary pool, forcing eviction of an
    # earlier prefix to host. Replaying earlier prompts (Pass 2) then
    # forces onboard from host back to primary, and onboard typically
    # places the block in a *different* primary slot than its block_id.
    # That divergence is exactly what the disagg pointer-arithmetic fix
    # has to handle — without the fix, the sender computes
    # `base + block_id * slot_bytes` and reads the wrong primary slot.
    _filler = (
        "This is filler context describing computer systems, distributed "
        "inference, KV cache management, host memory offload policies, "
        "block reuse via radix prefix trees, and the disaggregated serving "
        "architecture used by modern large language model deployments. ")
    _topics = [
        "transformer KV cache management",
        "the disaggregated prefill/decode split",
        "host (CPU) memory offload trade-offs",
        "block eviction and onboard cycles",
        "primary versus secondary KV pools",
        "radix prefix tree block reuse",
    ]
    distinct_prompts = [
        f"Topic: {topic}. {_filler * 4} Now answer briefly:"
        for topic in _topics
    ]

    async def send(session, prompt):
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        async with session.post(f"{server_url}/v1/completions",
                                json=payload,
                                timeout=timeout) as resp:
            assert resp.status == 200, (
                f"completions request failed with {resp.status}: "
                f"{await resp.text()}")
            return await resp.json()

    def assert_sane(resp, label):
        choices = resp.get("choices") or []
        assert choices, f"{label}: response missing 'choices': {resp}"
        text = choices[0].get("text") or ""
        assert text.strip(), f"{label}: empty/whitespace text: {resp}"
        usage = resp.get("usage") or {}
        assert usage.get("completion_tokens") == max_tokens, (
            f"{label}: completion_tokens != {max_tokens}; "
            f"got {usage.get('completion_tokens')}")
        return text

    async def drive():
        async with aiohttp.ClientSession() as session:
            # Pass 1: prime the radix tree with each distinct prompt and
            # capture the deterministic output (temperature=0).
            first_texts = []
            for idx, p in enumerate(distinct_prompts):
                resp = await send(session, p)
                first_texts.append(assert_sane(resp, f"pass1[{idx}]"))

            # Pass 2: send all prompts CONCURRENTLY each replay. Concurrent
            # in-flight prefills hold their KV blocks simultaneously; with
            # primary capacity smaller than the union of in-flight prompts,
            # this is the scenario that produces non-trivial alloc/free
            # interleaving and onboard-to-different-slot for replayed
            # prompts. Strict serial sends (Pass 1 above) typically alloc
            # back to original slots and miss the bug.
            for replay in range(5):
                results = await asyncio.gather(
                    *[send(session, p) for p in distinct_prompts])
                for idx, resp in enumerate(results):
                    text = assert_sane(resp,
                                       f"pass2.replay{replay}.prompt{idx}")
                    usage = resp["usage"]
                    cached = (usage.get("prompt_tokens_details")
                              or {}).get("cached_tokens", 0)
                    print(f"[host_offload_e2e] replay={replay} prompt={idx} "
                          f"prompt_tokens={usage.get('prompt_tokens')} "
                          f"cached_tokens={cached}")
                    # Reuse must hit — otherwise we never exercise onboard
                    # back from host, which is the path the fix protects.
                    assert cached > 0, (
                        f"replay={replay} prompt={idx}: expected reuse "
                        f"hit (cached_tokens > 0), got usage={usage}")
                    # Primary regression check: deterministic decoding +
                    # correct KV must reproduce Pass 1's output bit-for-bit.
                    assert text == first_texts[idx], (
                        f"replay={replay} prompt={idx}: output diverged "
                        f"from Pass 1, indicating wrong KV was read after "
                        f"offload/onboard.\n"
                        f"  pass1: {first_texts[idx]!r}\n"
                        f"  replay: {text!r}")

    asyncio.run(drive())


# Plain parametrize (not the `llama_model_root` indirect fixture) so the
# test ID picks up the `[TinyLlama-1.1B-Chat-v1.0]` suffix that matches
# the other disagg tests, without forcing LLM_MODELS_ROOT / NFS access —
# trtllm-serve resolves the HuggingFace id directly.
@pytest.mark.parametrize("llama_model_root", ["TinyLlama-1.1B-Chat-v1.0"])
def test_disaggregated_python_transceiver_host_offload(
        disaggregated_test_root, llm_venv, disaggregated_example_root,
        llama_model_root):  # noqa: ARG001 — used only for the parametrize label
    """E2E regression for block_id -> primary-slot translation in the Python disagg cache transceiver.

    See `_verify_python_transceiver_under_host_offload` for what this
    test proves. The setup pairs the Python transceiver runtime with a
    ctx-side `host_cache_size` and a deliberately tight primary pool so
    that prefix reuse is forced through an offload+onboard cycle before
    each KV transfer.

    Model resolution: trtllm-serve loads the HuggingFace id from the
    config's `model:` field (TinyLlama/TinyLlama-1.1B-Chat-v1.0) via
    huggingface_hub on first use. No LLM_MODELS_ROOT / NFS dependency.
    """
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    env = llm_venv._new_env.copy()
    env["UCX_TLS"] = get_ucx_tls()

    def post_client_test(server_url: str):
        _verify_python_transceiver_under_host_offload(
            server_url, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "python_transceiver_host_offload",
                           env=env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory(),
                           post_client_test=post_client_test)


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_perf_metrics(disaggregated_test_root, llm_venv,
                                    disaggregated_example_root,
                                    llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def extra_endpoints_test(server_url: str):
        item = get_timing_metrics(server_url)
        # Use helper function to validate all timing metrics comprehensively
        validate_timing_metrics(item, "perf_metrics test")

    run_disaggregated_test(disaggregated_example_root,
                           "perf_metrics",
                           env=llm_venv._new_env,
                           extra_endpoints_test=extra_endpoints_test,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_chat_completion_tool_calls(disaggregated_test_root,
                                                  llm_venv,
                                                  disaggregated_example_root,
                                                  llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "tool_calls",
                           num_iters=1,
                           prompt_file="tool_call_prompts.json",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_kv_cache_time_output(disaggregated_test_root, llm_venv,
                                            disaggregated_example_root,
                                            llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    output_path = os.path.join(llm_venv.get_working_directory(), "cache_time")
    run_disaggregated_test(disaggregated_example_root,
                           "perf_metrics",
                           env=llm_venv._new_env
                           | {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": output_path},
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())
    assert os.path.isdir(output_path)
    send_file = os.path.join(output_path, "rank_0_send.csv")
    recv_file = os.path.join(output_path, "rank_0_recv.csv")
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
def test_disaggregated_load_balance(disaggregated_test_root, llm_venv,
                                    disaggregated_example_root,
                                    llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "load_balance",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_cache_aware_balance(disaggregated_test_root, llm_venv,
                                           disaggregated_example_root,
                                           llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "cache_aware_balance",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_conditional(disaggregated_test_root, llm_venv,
                                   disaggregated_example_root,
                                   llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    run_disaggregated_test(disaggregated_example_root,
                           "conditional",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ngram(disaggregated_test_root, llm_venv,
                             disaggregated_example_root, llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_disaggregated_test(disaggregated_example_root,
                           "ngram",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxpp2_genpp2(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_disaggregated_test(disaggregated_example_root,
                           "ctxpp2_genpp2",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxtp2_genpp2(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_disaggregated_test(disaggregated_example_root,
                           "ctxtp2_genpp2",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxpp2_gentp2(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_disaggregated_test(disaggregated_example_root,
                           "ctxpp2_gentp2",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxtp2pp2_gentp2pp2(disaggregated_test_root, llm_venv,
                                           disaggregated_example_root,
                                           llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_disaggregated_test(disaggregated_example_root,
                           "ctxtp2pp2_gentp2pp2",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxpp4_genpp4(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_disaggregated_test(disaggregated_example_root,
                           "ctxpp4_genpp4",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


#tiny llama pp4 will have uneven layer per pp. pp4
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_disaggregated_ctxpp4_gentp4(disaggregated_test_root, llm_venv,
                                     disaggregated_example_root,
                                     llama_model_root):
    setup_model_symlink(llm_venv, llama_model_root,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_disaggregated_test(disaggregated_example_root,
                           "ctxpp4_gentp4",
                           env=llm_venv._new_env,
                           model_path=llama_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.skip(
    reason="MPI cache transceiver requires shared MPI process group, "
    "incompatible with service discovery which launches separate subprocesses")
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_mpi(disaggregated_test_root,
                                                disaggregated_example_root,
                                                llm_venv,
                                                deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")
    env = llm_venv._new_env.copy()
    env["TRTLLM_USE_MPI_KVCACHE"] = "1"
    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_mpi",
                           env=env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_tp1_single_gpu(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_tp1",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_tp1_single_gpu_mtp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_tp1_mtp",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_ctxpp2_gentp2_one_mtp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    #add one mtp layer, pp rank0 will have 15 layer, pp rank 1 will have 16 layers.
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_ctxpp2_gentp2_one_mtp",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(8)
@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_ctxtp2ep2pp2_gentp4_one_mtp_block_reuse(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(
        disaggregated_example_root,
        "deepseek_v3_lite_fp8_ctxtp2ep2pp2_gentp4_one_mtp_block_reuse",
        env=llm_venv._new_env,
        model_path=deepseek_v3_model_root,
        cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(8)
@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_ctxtp2ep2pp2_gentp4_one_mtp_block_reuse_long_prompt(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    # NVBug 5991576: repeated long prompts with PP+disagg+block_reuse
    # trigger scheduler hang due to reusable token budget miscalculation.
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(
        disaggregated_example_root,
        "deepseek_v3_lite_fp8_ctxtp2ep2pp2_gentp4_one_mtp_block_reuse_chunked",
        num_iters=5,
        prompt_file="long_prompts.json",
        env=llm_venv._new_env,
        model_path=deepseek_v3_model_root,
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

    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")
    env = llm_venv._new_env.copy()
    env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    env["UCX_TLS"] = get_ucx_tls()
    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_ucx",
                           env=env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@skip_arm
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_nixl(disaggregated_test_root,
                                                 disaggregated_example_root,
                                                 llm_venv,
                                                 deepseek_v3_model_root):

    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")
    env = llm_venv._new_env.copy()
    env["TRTLLM_USE_NIXL_KVCACHE"] = "1"
    env["UCX_TLS"] = get_ucx_tls()
    env["UCX_MM_ERROR_HANDLING"] = "y"
    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_nixl",
                           env=env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@skip_arm
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_transceiver_runtime_python(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")
    env = llm_venv._new_env.copy()
    env["UCX_TLS"] = get_ucx_tls()
    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_transceiver_runtime_python",
                           env=env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@skip_arm
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_ucx_tp1_single_gpu(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")
    env = llm_venv._new_env.copy()
    env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    env["UCX_TLS"] = get_ucx_tls()

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_tp1",
                           env=env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_attention_dp",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_gen_only(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    env = llm_venv._new_env.copy()
    env['TRTLLM_DISAGG_BENCHMARK_GEN_ONLY'] = '1'
    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_attention_dp_gen_only",
                           env=env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_overlap(
        disaggregated_test_root, llm_venv, disaggregated_example_root,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp_8_attention_dp_overlap",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_overlap_cuda_graph(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(
        disaggregated_example_root,
        "deepseek_v3_lite_fp8_attention_dp_overlap_cuda_graph",
        env=llm_venv._new_env,
        model_path=deepseek_v3_model_root,
        cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_overlap_cuda_graph(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_overlap_cuda_graph",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_one(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_attention_dp_one",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_attention_dp_one_mtp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_attention_dp_one_mtp",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_tp1_attention_dp_overlap_one_mtp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):

    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(
        disaggregated_example_root,
        "deepseek_v3_lite_fp8_tp1_attention_dp_overlap_one_mtp",
        env=llm_venv._new_env,
        model_path=deepseek_v3_model_root,
        cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_cache_aware_balance(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/bf16")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_bf16_cache_aware_balance",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_conditional(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/bf16")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_bf16_conditional",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


# V2 variant of the conditional disagg test: KV manager V2 + Python NIXL transceiver.
@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_conditional_v2(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/bf16")

    # Conditional disagg handles short-prefill requests locally on the gen
    # server (bypassing the ctx handoff + add_per_request_metrics), while routed
    # requests are recorded in the disagg /perf_metrics. Verify ONCE after all
    # client iterations via post_client_test (not per-iteration): routed-request
    # metrics are recorded asynchronously (add_per_request_metrics via
    # create_task on response completion) and surface only after the client
    # traffic settles, so a per-iteration read races that lag; /perf_metrics is
    # also consume-on-read, so query it exactly once at the end.
    def _check_routed_recorded(server_url: str):
        import requests as http_requests
        metrics = []
        deadline = time.time() + 60
        while True:
            resp = http_requests.get(f"{server_url}/perf_metrics", timeout=10)
            assert resp.status_code == 200, \
                f"perf_metrics fetch failed: {resp.status_code}"
            metrics = resp.json()
            if metrics or time.time() >= deadline:
                break
            time.sleep(2)
        logger.info(f"conditional_v2 perf_metrics len={len(metrics)} "
                    f"(routed requests recorded; bypassed ones absent)")
        # With short prompts every prompt's first occurrence routes through the
        # context server (match=0 -> need_ctx), so at least one routed request
        # must be recorded; an empty result means conditional routing never
        # engaged.
        assert metrics, \
            "no per-request metrics recorded after client runs; conditional routing may be misconfigured"

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_bf16_conditional_v2",
                           env=llm_venv._new_env,
                           post_client_test=_check_routed_recorded,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-fp8'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_fp8_tp1_two_mtp(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/fp8")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_fp8_tp1_two_mtp",
                           env=llm_venv._new_env,
                           model_path=deepseek_v3_model_root,
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
                                random_input_len=16,
                                random_output_len=64,
                                num_prompts=100,
                                max_concurrency=32,
                                skip_warmup=False,
                                model_path=None,
                                cwd=None):
    """Run disaggregated test with given configuration."""
    run_env = env.copy() if env else os.environ.copy()
    run_env["UCX_TLS"] = get_ucx_tls()
    run_env["UCX_MM_ERROR_HANDLING"] = "y"

    config, ctx_workers, gen_workers, disagg_server, server_port, work_dir = \
        setup_disagg_cluster(config_file, model_name=model_path, env=run_env, cwd=cwd)

    server_host = config.get("hostname", "localhost")

    try:
        # Start Benchmark
        benchmark_script = os.path.join(benchmark_root, "benchmark_serving.py")
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
        logger.error("Benchmark test failed")
        raise
    finally:
        terminate(*ctx_workers, *gen_workers, disagg_server)
        shutil.rmtree(work_dir, ignore_errors=True)


def get_config_for_benchmark(model_root, backend):
    serve_config = {
        "model": model_root,
        "hostname": "localhost",
        "port": get_free_port(),
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
            "urls": [f"localhost:{get_free_port()}"]
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
            "urls": [f"localhost:{get_free_port()}"]
        }
    }
    return serve_config


def run_disaggregated_aiperf(config_file,
                             model_path,
                             server_start_timeout=1200,
                             input_tokens=128,
                             output_tokens=100,
                             input_tokens_stddev=0,
                             output_tokens_stddev=0,
                             concurrency=1,
                             endpoint_type='chat',
                             request_count=None,
                             warmup_request_count=10,
                             streaming=True,
                             random_seed=100,
                             accuracy_test=False,
                             threshold=0.8,
                             cancellation_rate=None,
                             cancellation_delay=None,
                             env=None,
                             cwd=None):
    """Run disaggregated test with genai-perf for performance/stress testing.

    Args:
        config_file: Path to disaggregated server config YAML
        model_path: Path to model for tokenizer
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
    run_env["UCX_TLS"] = get_ucx_tls()
    run_env["UCX_MM_ERROR_HANDLING"] = "y"

    config, ctx_workers, gen_workers, disagg_server, server_port, work_dir = \
        setup_disagg_cluster(config_file, model_name=model_path, env=run_env, cwd=cwd,
                             server_start_timeout=server_start_timeout,
                             save_log=True)

    server_host = config.get("hostname", "localhost")
    artifact_dir = os.path.join(cwd or ".", "benchmark-results")

    try:
        # Wait for server to be ready
        if not wait_for_server(
                server_host, server_port, timeout_seconds=server_start_timeout):
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
            '--url',
            f'{server_host}:{server_port}',
            '--synthetic-input-tokens-mean',
            str(input_tokens),
            '--synthetic-input-tokens-stddev',
            str(input_tokens_stddev),
            '--output-tokens-mean',
            str(output_tokens),
            '--output-tokens-stddev',
            str(output_tokens_stddev),
            '--extra-inputs',
            'ignore_eos:true',
        ])
        # When output length is fixed (stddev == 0) pin max/min tokens so the
        # server returns exactly output_tokens. When non-zero, let aiperf
        # sample the per-request max_tokens from the mean/stddev distribution.
        if output_tokens_stddev == 0:
            aiperf_cmd.extend([
                '--extra-inputs',
                f'max_tokens:{output_tokens}',
                '--extra-inputs',
                f'min_tokens:{output_tokens}',
            ])
        aiperf_cmd.extend([
            '--concurrency',
            str(concurrency),
            '--warmup-request-count',
            str(warmup_request_count),
        ])

        # Use request-count or num-dataset-entries
        if request_count is not None:
            aiperf_cmd.extend(['--request-count', str(request_count)])
        else:
            # Default: use num-dataset-entries for compatibility
            aiperf_cmd.extend(['--num-dataset-entries', '64'])

        if cancellation_rate is not None:
            aiperf_cmd.extend(
                ['--request-cancellation-rate',
                 str(cancellation_rate)])
        if cancellation_delay is not None:
            aiperf_cmd.extend(
                ['--request-cancellation-delay',
                 str(cancellation_delay)])

        aiperf_cmd.extend(
            ['--random-seed',
             str(random_seed), '--artifact-dir', artifact_dir])

        # Run aiperf
        all_worker_procs = [w.process for w in ctx_workers + gen_workers]
        check_call(aiperf_cmd,
                   env=env,
                   poll_procs=all_worker_procs + [disagg_server.process])

        # Catch cases where aiperf finished but the disagg cluster was unhealthy
        # during the run (e.g. context-side hangs, KV transfer timeouts) which
        # would otherwise be swallowed because aiperf records 500s as completed.
        fatal_findings = scan_logs_for_fatal_errors(
            [*ctx_workers, *gen_workers, disagg_server])
        if fatal_findings:
            summary = "\n".join(f"  {path}: " +
                                ", ".join(f"{pat}={cnt}"
                                          for pat, cnt in hits.items())
                                for path, hits in fatal_findings.items())
            raise AssertionError(
                "Fatal error patterns detected in disaggregated worker/server "
                f"logs:\n{summary}")

        if accuracy_test:
            accuracy_test_result, accuracy_value = run_accuracy_test(
                model_path=model_path,
                server_url=f"http://{server_host}:{server_port}",
                concurrency=concurrency,
                max_retries=3,
                timeout=1200,
                max_gen_toks=256,
                max_length=4096)

            if not accuracy_test_result:
                raise AssertionError(
                    "Accuracy test failed to complete (likely worker hang or "
                    "crash); inspect saved logs under work_dir "
                    f"({work_dir}): worker_ctx_*.log, worker_gen_*.log, "
                    "disagg_server.log")
            if accuracy_value < threshold:
                raise AssertionError(
                    f"Accuracy test failed: accuracy value {accuracy_value} is less than test threshold {threshold}"
                )

            # Re-scan logs after the accuracy run to catch issues (e.g.
            # cluster-not-ready, internal server errors) that only surfaced
            # while lm_eval was driving the server.
            fatal_findings = scan_logs_for_fatal_errors(
                [*ctx_workers, *gen_workers, disagg_server])
            if fatal_findings:
                summary = "\n".join(f"  {path}: " +
                                    ", ".join(f"{pat}={cnt}"
                                              for pat, cnt in hits.items())
                                    for path, hits in fatal_findings.items())
                raise AssertionError(
                    "Fatal error patterns detected in disaggregated "
                    f"worker/server logs after accuracy run:\n{summary}")

    except Exception:
        # Print tail of each captured worker/server log to aid triage.
        for proc in [*ctx_workers, *gen_workers, disagg_server]:
            log_path = getattr(proc, "log_path", None)
            if not log_path or not os.path.exists(log_path):
                continue
            logger.error(f"-------- {log_path} (last 30 lines) --------")
            try:
                from collections import deque
                with open(log_path, "r", errors="replace") as f:
                    for line in deque(f, maxlen=30):
                        if line.strip():
                            logger.error(line.rstrip())
            except OSError:
                pass
        raise
    finally:
        terminate(*ctx_workers, *gen_workers, disagg_server)
        shutil.rmtree(work_dir, ignore_errors=True)


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

        # lm_eval's async request retry path crashes on certain transport-level
        # errors with "UnboundLocalError: cannot access local variable 'outputs'".
        # When this happens individual requests are silently dropped but the
        # process can still exit 0, so the score is computed against a partial
        # set. Treat the presence of this trace as an inconclusive run.
        combined_output = (result.stdout or "") + (result.stderr or "")
        if ("UnboundLocalError" in combined_output
                and "outputs" in combined_output):
            logger.warning(
                "lm_eval reported UnboundLocalError on 'outputs' "
                "(request retry crashed); treating accuracy run as failed")
            return False, accuracy_value

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


@skip_pre_hopper
@pytest.mark.parametrize("benchmark_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_empty_batch(
        disaggregated_example_root, llm_venv, benchmark_model_root,
        benchmark_root, shared_gpt_path):

    setup_model_symlink(llm_venv, benchmark_model_root, "DeepSeek-V3-Lite/bf16")

    test_desc = "deepseek_v3_lite_bf16_empty_batch"
    config_file = get_test_config(test_desc, disaggregated_example_root,
                                  os.path.dirname(__file__))

    env = llm_venv._new_env.copy()
    e2el, ttft = run_disaggregated_benchmark(
        disaggregated_example_root,
        config_file,
        benchmark_root,
        benchmark_model_root,
        shared_gpt_path,
        env=env,
        num_prompts=10,
        max_concurrency=10,
        random_input_len=384,
        random_output_len=1536,
        skip_warmup=True,
        model_path=benchmark_model_root,
        cwd=llm_venv.get_working_directory())
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
    setup_model_symlink(llm_venv, llama4_model_root, model_path)

    config_file = get_test_config("llama4_kv_cache_overflow",
                                  disaggregated_example_root,
                                  os.path.dirname(__file__))

    run_disaggregated_aiperf(config_file=config_file,
                             model_path=llama4_model_root,
                             server_start_timeout=1200,
                             input_tokens=128000,
                             output_tokens=100,
                             env=llm_venv._new_env,
                             cwd=llm_venv.get_working_directory())


@skip_pre_blackwell
@pytest.mark.timeout(2400)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("prompt_file", ["prompts.json"], ids=["short_prompt"])
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_tllm_gen_helix(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root, prompt_file):
    setup_model_symlink(llm_venv, deepseek_v3_model_root,
                        "DeepSeek-V3-Lite/bf16")

    run_disaggregated_test(disaggregated_example_root,
                           "deepseek_v3_lite_bf16_tllm_gen_helix",
                           env=llm_venv._new_env,
                           prompt_file=prompt_file,
                           model_path=deepseek_v3_model_root,
                           cwd=llm_venv.get_working_directory())


@skip_pre_blackwell
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("model_path", ['gpt_oss/gpt-oss-120b'])
def test_disaggregated_gpt_oss_120b_harmony(disaggregated_test_root,
                                            disaggregated_example_root,
                                            llm_venv, model_path):
    model_dir = f"{llm_models_root()}/{model_path}"
    setup_model_symlink(llm_venv, model_dir, model_path)

    env = llm_venv._new_env.copy()
    tiktoken_vocab = os.path.join(llm_models_root(), "datasets",
                                  "tiktoken_vocab")
    env["TIKTOKEN_RS_CACHE_DIR"] = tiktoken_vocab
    env["TIKTOKEN_ENCODINGS_BASE"] = tiktoken_vocab

    run_disaggregated_test(disaggregated_example_root,
                           "gpt_oss_120b_harmony",
                           env=env,
                           model_path=model_dir,
                           cwd=llm_venv.get_working_directory())


@skip_pre_hopper
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("model_path", ['Qwen3/Qwen3-32B-FP8'])
def test_disaggregated_qwen3_32b_fp8(disaggregated_test_root,
                                     disaggregated_example_root, llm_venv,
                                     model_path):
    model_dir = resolve_llm_model_path(model_path)
    setup_model_symlink(llm_venv, model_dir, model_path)

    run_disaggregated_test(disaggregated_example_root,
                           "qwen3_32b_fp8_stress",
                           env=llm_venv._new_env,
                           model_path=model_dir,
                           cwd=llm_venv.get_working_directory())


@pytest.mark.timeout(12600)
@pytest.mark.parametrize("test_config", [
    pytest.param(TestConfig(model_path='DeepSeek-R1/DeepSeek-R1-0528-FP4-v2',
                            test_desc='deepseek_r1_v2_fp4_stress',
                            request_count=35000,
                            accuracy_threshold=0.92,
                            cancellation_rate=10,
                            cancellation_delay=0.5),
                 marks=(pytest.mark.skip_less_device(8), skip_pre_blackwell)),
    pytest.param(TestConfig(model_path='DeepSeek-R1/DeepSeek-R1-0528-FP4-v2',
                            test_desc='deepseek_r1_v2_fp4_mtp_stress',
                            request_count=35000,
                            accuracy_threshold=0.90,
                            cancellation_rate=10,
                            cancellation_delay=0.5),
                 marks=(pytest.mark.skip_less_device(8), skip_pre_blackwell)),
    pytest.param(TestConfig(model_path='gpt_oss/gpt-oss-120b',
                            test_desc='gpt_oss_120b_trtllm_stress',
                            request_count=60000,
                            accuracy_threshold=0.42,
                            cancellation_rate=10,
                            cancellation_delay=0.5),
                 marks=(pytest.mark.skip_less_device(4), skip_pre_blackwell)),
    pytest.param(
        TestConfig(
            model_path='gpt_oss/gpt-oss-120b',
            test_desc='gpt_oss_120b_eagle_triton_stress',
            request_count=60000,
            accuracy_threshold=0.42,
            speculative_model_path='gpt_oss/gpt-oss-120b-Eagle3',
            cancellation_rate=10,
            cancellation_delay=0.5,
        ),
        marks=(pytest.mark.skip_less_device(8), skip_no_hopper),
    ),
    pytest.param(
        TestConfig(
            model_path='gpt_oss/gpt-oss-120b',
            test_desc='gpt_oss_120b_eagle_trtllm_stress',
            request_count=60000,
            accuracy_threshold=0.42,
            speculative_model_path='gpt_oss/gpt-oss-120b-Eagle3',
            cancellation_rate=10,
            cancellation_delay=0.5,
        ),
        marks=(pytest.mark.skip_less_device(8), skip_pre_blackwell),
    ),
    pytest.param(TestConfig(model_path='gpt_oss/gpt-oss-120b',
                            test_desc='gpt_oss_120b_triton_stress',
                            request_count=30000,
                            accuracy_threshold=0.42,
                            cancellation_rate=10,
                            cancellation_delay=0.5),
                 marks=(pytest.mark.skip_less_device(4), skip_no_hopper)),
    pytest.param(TestConfig(model_path='Qwen3.5-4B-FP8',
                            test_desc='qwen3_5_4b_fp8_stress',
                            request_count=3000,
                            accuracy_threshold=0.72,
                            cancellation_rate=10,
                            cancellation_delay=0.5),
                 marks=(pytest.mark.skip_less_device(2), skip_no_hopper)),
    pytest.param(TestConfig(model_path='GLM-5-NVFP4',
                            test_desc='glm5_nvfp4_tp4_ep4_dp_stress',
                            request_count=35000,
                            accuracy_threshold=0.90,
                            cancellation_rate=10,
                            cancellation_delay=0.5),
                 marks=(pytest.mark.skip_less_device(8), skip_pre_blackwell)),
    pytest.param(TestConfig(
        model_path='Qwen3/Qwen3-32B-FP8',
        test_desc='qwen3_32b_fp8_stress',
        request_count=10000,
        accuracy_threshold=0.42,
        speculative_model_path='Zhi-Create-Qwen3-32B-Eagle3',
        cancellation_rate=10,
        cancellation_delay=0.5),
                 marks=(pytest.mark.skip_less_device(8), skip_pre_hopper)),
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
    model_dir = resolve_llm_model_path(model_path)
    setup_model_symlink(llm_venv, model_dir, model_path)

    config_file = get_test_config(test_desc, disaggregated_example_root,
                                  os.path.dirname(__file__))

    # Resolve speculative_model to an absolute path for worker processes.
    if test_config.speculative_model_path is not None:
        spec_model_dir = f"{llm_models_root()}/{test_config.speculative_model_path}"
        setup_model_symlink(llm_venv, spec_model_dir,
                            test_config.speculative_model_path)
        with open(config_file, 'r') as f:
            patched_config = yaml.safe_load(f)
        patched_sections = []
        # Check top-level speculative_config first (current YAML layout), then
        # fall back to per-server blocks for older config shapes.
        top_spec = patched_config.get('speculative_config')
        if isinstance(top_spec, dict) and 'speculative_model' in top_spec:
            top_spec['speculative_model'] = spec_model_dir
            patched_sections.append('top-level')
        else:
            for section in ('context_servers', 'generation_servers'):
                spec = patched_config.get(section, {}).get('speculative_config')
                if spec is not None and 'speculative_model' in spec:
                    spec['speculative_model'] = spec_model_dir
                    patched_sections.append(section)
        if not patched_sections:
            raise AssertionError(
                f"{test_desc} sets speculative_model_path, but no "
                "speculative_config.speculative_model field was patched")
        patched_path = os.path.join(llm_venv.get_working_directory(),
                                    f"{test_desc}_patched.yaml")
        with open(patched_path, 'w') as f:
            yaml.safe_dump(patched_config, f)
        config_file = patched_path

    run_disaggregated_aiperf(config_file=config_file,
                             model_path=model_dir,
                             server_start_timeout=7200,
                             input_tokens=input_tokens,
                             output_tokens=output_tokens,
                             input_tokens_stddev=0,
                             output_tokens_stddev=output_tokens // 10,
                             concurrency=concurrency,
                             endpoint_type='completions',
                             request_count=test_config.request_count,
                             warmup_request_count=10,
                             streaming=False,
                             accuracy_test=True,
                             threshold=test_config.accuracy_threshold,
                             cancellation_rate=test_config.cancellation_rate,
                             cancellation_delay=test_config.cancellation_delay,
                             env=llm_venv._new_env,
                             cwd=llm_venv.get_working_directory())


def run_cancel_stress_test(server_url: str,
                           num_bursts: int = 5,
                           requests_per_burst: int = 32,
                           prompt_len_range: tuple = (2000, 8000),
                           cancel_after_range: tuple = (0.01, 0.1)):
    """
    Stress test that sends requests with large contexts and cancels them
    during prefill to test resource cleanup under cancellation.

    Args:
        server_url: The server URL (e.g., "http://localhost:8000")
        num_bursts: Number of request bursts to send
        requests_per_burst: Number of concurrent requests per burst
        prompt_len_range: (min, max) prompt length in tokens
        cancel_after_range: (min, max) seconds to wait before cancelling
    """
    import asyncio
    import random
    import time

    import aiohttp

    async def spam_and_cancel(session, req_id, url, prompt_len_range,
                              cancel_after_range):
        """Send a request and cancel it during prefill."""
        prompt_len = random.randint(prompt_len_range[0], prompt_len_range[1])
        prompt = "test " * (prompt_len // 5)

        payload = {
            "model": "test-model",
            "prompt": prompt,
            "max_tokens": 10,
            "stream": True
        }

        try:
            cancel_after = random.uniform(cancel_after_range[0],
                                          cancel_after_range[1])
            start = time.time()
            async with session.post(
                    f"{url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)) as resp:
                async for line in resp.content:
                    if time.time() - start > cancel_after:
                        # Force disconnect during prefill
                        break
        except Exception:
            pass  # Connection abort is expected

    async def run_bursts():
        async with aiohttp.ClientSession() as session:
            for burst_idx in range(num_bursts):
                tasks = [
                    spam_and_cancel(session, i, server_url, prompt_len_range,
                                    cancel_after_range)
                    for i in range(requests_per_burst)
                ]
                await asyncio.gather(*tasks)
                logger.info(
                    f"Completed burst {burst_idx + 1}/{num_bursts} ({requests_per_burst} requests)"
                )
                await asyncio.sleep(0.05)

    asyncio.run(run_bursts())


# ---------------------------------------------------------------------------
# Mixed-stress client (TRTLLM-12154)
#
# Exercises in-batch feature mixing: structured output (xgrammar JSON schema)
# alongside free-text, varied temperature, varied input length, and
# cancellations — all in the same scheduling step.
#
# Architecture:
#   run_disaggregated_mixed_stress  — cluster wrapper (mirrors run_disaggregated_cancel_test)
#     └─ _run_mixed_stress_async    — asyncio entry point
#          └─ _send_mixed_request   — per-request coroutine (mirrors spam_and_cancel)
#
# Per-request spec dec toggling is NOT supported in the PyTorch backend
# (py_disable_speculative_decoding is only settable batch-wide, not via the
# HTTP API), so spec dec is a server-startup knob only and is not a profile
# dimension here.
# ---------------------------------------------------------------------------


@dataclass
class _MixedStressProfile:
    """One request variant in the mixed-stress run."""
    name: str
    weight: float  # relative sampling weight; normalised at runtime
    input_len_range: tuple  # (min_tokens, max_tokens) for synthetic prompt
    output_len: int
    temperature: float
    # JSON schema passed as response_format. None → free-text completion.
    structured_output_schema: dict
    # Whether this request is cancelled mid-stream (excluded from accuracy).
    cancel: bool
    # Fixed prompt to send verbatim. None → synthetic filler sized to
    # input_len_range. Structured profiles need a real instruction so the
    # model has content to emit; otherwise guided decoding degenerates into
    # whitespace padding that never closes the JSON.
    prompt: str = None
    # Free-text accuracy criteria (ignored for structured/cancel):
    #   expected_substring set → success requires it in the output (real
    #     correctness check; use with temperature=0 for determinism).
    #   min_content_chars > 0  → success requires that many non-whitespace
    #     chars (a "produced real content" check for stochastic profiles;
    #     tolerant of run-on generation that hits the length cap, which is
    #     legitimate at high temperature).
    #   neither                → loose: success once the stream completes
    #     (for context-stress profiles whose output content is secondary).
    expected_substring: str = None
    min_content_chars: int = 0
    # When False, the profile still drives load (and counts toward
    # incomplete_rate) but is excluded from the accuracy metric — for pure
    # input/length stress where output content isn't validated.
    count_accuracy: bool = True


def _profile_accuracy_metric(profile):
    """Which per-request field measures this profile's accuracy.

    Returns the result key to sum in the accuracy numerator ("json_valid" for
    structured output, "success" for free-text/long-context), or None if the
    profile is excluded from accuracy entirely (cancellations, pure stress
    profiles). This is the single source of truth for accuracy bucketing — the
    aggregation derives from it rather than matching on profile names.
    """
    if profile.cancel or not profile.count_accuracy:
        return None
    if profile.structured_output_schema is not None:
        return "json_valid"
    return "success"


# Default profile mix. Weights are relative; they are normalised in
# _run_mixed_stress_async.
_DEFAULT_MIXED_STRESS_PROFILES = [
    _MixedStressProfile(
        name='free_text_low_temp',
        weight=25.0,
        input_len_range=(512, 2048),  # unused: prompt is fixed below
        output_len=256,
        temperature=0.0,
        structured_output_schema=None,
        cancel=False,
        # Greedy + known-answer prompt → deterministic correctness check.
        prompt="What is the capital of France? Answer in one word.",
        expected_substring="paris",
    ),
    _MixedStressProfile(
        name='free_text_high_temp',
        weight=15.0,
        input_len_range=(512, 2048),  # unused: prompt is fixed below
        output_len=256,
        temperature=1.0,
        structured_output_schema=None,
        cancel=False,
        # Open-ended generative prompt → reliably produces real text at high
        # temperature (it will run to the length cap, which is fine). No exact
        # content to check; require that it produced real (non-whitespace)
        # content rather than empty/degenerate output.
        prompt="Write a short story about a traveler who discovers a "
        "hidden village in the mountains.",
        min_content_chars=50,
    ),
    _MixedStressProfile(
        name='structured_output',
        weight=30.0,
        input_len_range=(256, 1024),  # unused: prompt is fixed below
        output_len=256,
        temperature=0.0,
        # A concrete instruction that matches the schema below. The model needs
        # real content to emit; with a filler prompt it just pads whitespace
        # until the token cap and never closes the JSON.
        prompt=("Create a JSON object with a key \"pets\" whose value is an "
                "array of 4 to 5 objects, each with a \"name\" (a pet's name) "
                "and a \"species\". Respond with only the JSON object."),
        structured_output_schema={
            "type": "object",
            "properties": {
                "pets": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "maxLength": 20
                            },
                            "species": {
                                "type": "string",
                                "maxLength": 20
                            },
                        },
                        "required": ["name", "species"],
                    },
                }
            },
            "required": ["pets"],
        },
        cancel=False,
    ),
    _MixedStressProfile(
        name='long_context',
        weight=15.0,
        input_len_range=(6000, 8192),
        output_len=256,
        temperature=0.7,
        structured_output_schema=None,
        cancel=False,
    ),
    _MixedStressProfile(
        name='mid_length_stress',
        weight=20.0,
        input_len_range=(512, 2048),
        output_len=256,
        temperature=0.7,
        structured_output_schema=None,
        cancel=False,
        # Pure input-length stress over the mid range (the accuracy-checked
        # free-text profiles now use fixed short prompts). Output content is
        # not validated, so keep it out of the accuracy metric; it still
        # counts toward incomplete_rate.
        count_accuracy=False,
    ),
    _MixedStressProfile(
        name='cancel',
        weight=15.0,
        input_len_range=(2000, 8000),
        output_len=64,
        temperature=0.7,
        structured_output_schema=None,
        cancel=True,
    ),
]


async def _send_mixed_request(session,
                              server_url: str,
                              profile: _MixedStressProfile,
                              results: list,
                              model_name: str = "test-model") -> None:
    """Send one request according to profile and record the outcome.

    Three behaviors keyed on profile:
    - cancel: stream until cancel_after seconds then disconnect (success stays False)
    - structured_output: drain full stream, assemble text, validate JSON schema
    - free-text / long-context: drain full stream, mark success
    """
    import random

    if profile.prompt is not None:
        prompt = profile.prompt
    else:
        prompt_len = random.randint(*profile.input_len_range)
        prompt = "test " * (prompt_len // 5)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": profile.output_len,
        "temperature": profile.temperature,
        "stream": True,
    }
    if profile.structured_output_schema is not None:
        payload["response_format"] = {
            "type": "json",
            "schema": profile.structured_output_schema,
        }

    should_cancel = profile.cancel
    cancel_after = random.uniform(0.01, 0.1) if should_cancel else None

    result = {
        "profile": profile.name,
        "success": False,
        "json_valid": None,
        "latency_ms": 0.0,
        "cancelled": should_cancel,
        "timed_out": False,
        "server_rejected": False,
        "rejection_detail": None,  # diagnostic: HTTP status or exception class
    }

    start = time.monotonic()
    try:
        async with session.post(
                f"{server_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)) as resp:
            assembled = []
            finish_reason = None
            done_received = False
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    done_received = True
                    break
                if cancel_after is not None and time.monotonic(
                ) - start > cancel_after:
                    break  # force disconnect during generation
                try:
                    data = json.loads(chunk)
                    choice = data.get("choices", [{}])[0]
                    fr = choice.get("finish_reason")
                    if fr is not None:
                        finish_reason = fr
                    assembled.append(choice.get("text", ""))
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass

            # done_received: server sent explicit SSE terminator
            # finish_reason check: server ended with a final chunk carrying
            # finish_reason (trtllm-serve does not send data: [DONE])
            completed = done_received or finish_reason in ("stop", "length")
            text = "".join(assembled)
            if not should_cancel and completed:
                if profile.structured_output_schema is not None:
                    try:
                        parsed = json.loads(text)
                        required = profile.structured_output_schema.get(
                            "required", [])
                        result["json_valid"] = all(k in parsed
                                                   for k in required)
                    except json.JSONDecodeError:
                        result["json_valid"] = False
                    result["success"] = result["json_valid"]
                elif profile.expected_substring is not None:
                    # Deterministic correctness check (greedy profiles).
                    result["success"] = (profile.expected_substring.lower()
                                         in text.lower())
                elif profile.min_content_chars > 0:
                    # Stochastic profile: no exact content to match, but require
                    # real (non-whitespace) output rather than empty/degenerate.
                    content_chars = sum(1 for c in text if not c.isspace())
                    result["success"] = (content_chars
                                         >= profile.min_content_chars)
                else:
                    # Loose: stream completed (context/length-stress profiles).
                    result["success"] = True
            elif not should_cancel and not completed:
                # Server closed the stream without a valid completion
                # (e.g. kv_transfer_timeout cancellation, error response).
                result["server_rejected"] = True
                result[
                    "rejection_detail"] = f"http_{resp.status}_incomplete_stream"
    except asyncio.TimeoutError:
        if not should_cancel:
            # 120s aiohttp client timeout: server never responded in time.
            result["timed_out"] = True
        # else: cancel-break may race the deadline; ignore the timeout.
    except (aiohttp.ClientError, OSError) as e:
        if not should_cancel:
            # Transport error on a non-cancel request (e.g. abrupt RST from
            # server-side cancellation). HTTP error statuses do not reach here
            # (the response is streamed, not raise_for_status'd) — those flow
            # through the incomplete-stream path above.
            result["server_rejected"] = True
            result["rejection_detail"] = f"{type(e).__name__}({e})"
        # else: connection abort on intentional cancel is expected
    finally:
        result["latency_ms"] = (time.monotonic() - start) * 1000
        results.append(result)


async def _run_mixed_stress_async(
        server_url: str,
        profiles: list,
        total_requests: int,
        concurrency: int,
        model_name: str = "test-model",
        progress_callback=None,
        progress_interval: int = 30,
        workers=None,
        early_abort_rejection_rate: float = 0.05) -> dict:
    """Drive total_requests requests at the given concurrency level.

    Returns a summary dict with per-profile counts and overall metrics:

    accuracy_score = (free_text_successes + json_valid_count)
                     / completed_non_cancel_requests
    Cancelled, timed-out, and server-rejected requests are excluded from the
    accuracy denominator — accuracy reflects only requests that finished.

    incomplete_rate = (timed_out + server_rejected) / non_cancelled_total
    timed_out: client-side 120s aiohttp timeout (server never responded).
    server_rejected: server closed stream without valid completion (e.g.
        kv_transfer_timeout cancellation, error response, connection reset).
    Both are reported individually for diagnostics and summed as incomplete_rate
    for gating.
    """
    import random

    weights = [p.weight for p in profiles]
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(coro):
        async with sem:
            await coro

    abort_reason = []

    async def _progress_monitor(gather_task):
        start = time.monotonic()
        while True:
            await asyncio.sleep(progress_interval)
            done = len(results)
            if done >= total_requests:
                break
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total_requests - done) / rate if rate > 0 else float('inf')
            eta_str = f"{eta:.0f}s" if eta != float('inf') else "unknown"
            print(
                f"[mixed-stress] {done}/{total_requests} "
                f"({100 * done / total_requests:.0f}%) "
                f"elapsed={elapsed:.0f}s rate={rate:.1f} req/s ETA={eta_str}",
                flush=True)
            if progress_callback:
                progress_callback(done, total_requests, rate, eta)

            # Early-abort checks: worker crash or high rejection rate.
            reason = None
            if workers:
                crashed = _crashed_workers(workers)
                if crashed:
                    parts = [
                        f"{w.log_path or 'worker'} (rc={w.process.poll()})"
                        for w in crashed
                    ]
                    reason = f"worker(s) crashed: {parts}"
            if reason is None and done > 0:
                rejected = sum(1 for r in results if r["server_rejected"])
                if rejected / done > early_abort_rejection_rate:
                    reason = (
                        f"rejection rate {rejected}/{done} "
                        f"({100*rejected/done:.0f}%) exceeds "
                        f"{100*early_abort_rejection_rate:.0f}% threshold")
            if reason:
                abort_reason.append(reason)
                print(f"[mixed-stress] aborting early: {reason}", flush=True)
                gather_task.cancel()
                return

    async with aiohttp.ClientSession() as session:
        chosen_profiles = random.choices(profiles,
                                         weights=weights,
                                         k=total_requests)
        tasks = [
            bounded(
                _send_mixed_request(session, server_url, p, results,
                                    model_name)) for p in chosen_profiles
        ]

        async def _run_all():
            await asyncio.gather(*tasks)

        gather_task = asyncio.create_task(_run_all())
        monitor = asyncio.create_task(_progress_monitor(gather_task))
        try:
            await gather_task
        except asyncio.CancelledError:
            pass
        finally:
            monitor.cancel()

    # Aggregate
    per_profile: dict = {}
    for r in results:
        p = per_profile.setdefault(
            r["profile"], {
                "total": 0,
                "success": 0,
                "json_valid": 0,
                "cancelled": 0,
                "timed_out": 0,
                "server_rejected": 0,
                "rejection_details": {},
            })
        p["total"] += 1
        for key in ("success", "json_valid", "cancelled", "timed_out",
                    "server_rejected"):
            if r[key]:
                p[key] += 1
        if r["rejection_detail"]:
            rd = p["rejection_details"]
            rd[r["rejection_detail"]] = rd.get(r["rejection_detail"], 0) + 1

    def _completed(v):
        return v["total"] - v["cancelled"] - v["timed_out"] - v[
            "server_rejected"]

    metric_by_name = {p.name: _profile_accuracy_metric(p) for p in profiles}
    accuracy_ok = 0
    accuracy_denom = 0
    for k, v in per_profile.items():
        metric = metric_by_name.get(k)
        # Annotate each profile with its accuracy contribution so callers can
        # spot a category that silently produced nothing. metric is None for
        # profiles excluded from accuracy (e.g. cancellations).
        v["accuracy_metric"] = metric
        if metric is None:
            v["accuracy_completed"] = 0
            v["accuracy_ok"] = 0
            v["accuracy_rate"] = None
            continue
        ok = v[metric]
        completed = _completed(v)
        v["accuracy_completed"] = completed
        v["accuracy_ok"] = ok
        v["accuracy_rate"] = (ok / completed) if completed > 0 else None
        accuracy_ok += ok
        accuracy_denom += completed
    accuracy_score = accuracy_ok / accuracy_denom if accuracy_denom > 0 else 0.0

    total_non_cancelled = sum(v["total"] - v["cancelled"]
                              for v in per_profile.values())
    total_timed_out = sum(v["timed_out"] for v in per_profile.values())
    total_server_rejected = sum(v["server_rejected"]
                                for v in per_profile.values())
    incomplete_rate = ((total_timed_out + total_server_rejected) /
                       total_non_cancelled if total_non_cancelled > 0 else 0.0)

    # Merge per-profile rejection_details into a global breakdown
    all_rejection_details: dict = {}
    for v in per_profile.values():
        for reason, count in v["rejection_details"].items():
            all_rejection_details[reason] = (
                all_rejection_details.get(reason, 0) + count)

    return {
        "per_profile": per_profile,
        "accuracy_score": accuracy_score,
        "incomplete_rate": incomplete_rate,
        "timed_out": total_timed_out,
        "server_rejected": total_server_rejected,
        "rejection_details": all_rejection_details,
        "aborted_early": abort_reason or None,
    }


async def _warmup_requests(server_url: str, profiles: list, count: int,
                           concurrency: int, model_name: str) -> None:
    """Send `count` requests and discard their results to warm the cluster.

    The first request of each shape pays a one-time autotuner/compile cost
    (~20s host-steps observed on B200). Running those here, before the measured
    run, keeps them out of the accuracy/incomplete accounting and out of the
    heartbeat-eviction path (a worker stuck in a 20s step misses the 2s cluster
    heartbeat and gets evicted under a high-concurrency flood). Failures are
    ignored — the only goal is to trigger the autotuner across the profile mix.
    """
    import random

    weights = [p.weight for p in profiles]
    sem = asyncio.Semaphore(concurrency)
    sink: list = []  # discarded; _send_mixed_request swallows its own errors

    async def bounded(coro):
        async with sem:
            await coro

    async with aiohttp.ClientSession() as session:
        chosen = random.choices(profiles, weights=weights, k=count)
        tasks = [
            bounded(
                _send_mixed_request(session, server_url, p, sink, model_name))
            for p in chosen
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


def run_disaggregated_mixed_stress(example_dir: str,
                                   config_file: str,
                                   model_path: str,
                                   total_requests: int = 10000,
                                   concurrency: int = 512,
                                   accuracy_threshold: float = 0.42,
                                   incomplete_threshold: float = 0.05,
                                   per_profile_min_sample: int = 10,
                                   warmup_request_count: int = None,
                                   profiles: list = None,
                                   server_start_timeout: int = 600,
                                   env=None,
                                   cwd=None,
                                   startup_callback=None,
                                   progress_callback=None) -> None:
    """Run the Qwen3-32B FP8 Eagle3 mixed-stress test.

    Spins up a disaggregated cluster, drives total_requests heterogeneous
    requests (free-text, structured output, varied length, cancellations) at
    the given concurrency, checks accuracy_score >= accuracy_threshold, then
    verifies server health via disagg_client.py.

    Mirrors run_disaggregated_cancel_test for cluster setup/teardown.

    Default total_requests is 10000, a conservative starting point chosen
    to keep CI runtime manageable. The target workload (TRTLLM-12154) runs
    20-100k requests per stability run; 10k covers the feature-mixing code
    paths without the full wall-clock cost.

    Request phase scales roughly linearly with request count on 8x B200:
    the 60-request smoke variant runs in ~2 min of requests, a 5k run in
    ~14 min, and the 10k full variant ~32 min (plus ~1.5 min server
    startup). Each request profile is validated for real content
    (structured JSON schema, greedy known-answer, generative min-content),
    so a healthy cluster scores ~1.0 accuracy.

    Default concurrency is 512 rather than the ~32 typical of production
    stability runs. Higher concurrency exercises more in-flight request
    overlap and is more likely to surface scheduler/KV-cache/dispatcher
    races. Validated against the target workload spec (TRTLLM-12154).
    """
    if profiles is None:
        profiles = _DEFAULT_MIXED_STRESS_PROFILES

    cleanup_output_files()
    run_env = env.copy() if env else os.environ.copy()
    run_env["UCX_TLS"] = get_ucx_tls()
    run_env["UCX_MM_ERROR_HANDLING"] = "y"

    setup_start = time.monotonic()
    config, ctx_workers, gen_workers, disagg_server, server_port, work_dir = \
        setup_disagg_cluster(config_file, model_name=model_path, env=run_env,
                             cwd=cwd, server_start_timeout=server_start_timeout,
                             save_log=True, startup_callback=startup_callback)
    print(f"[startup] cluster ready in {time.monotonic() - setup_start:.1f}s",
          flush=True)

    server_host = config.get("hostname", "localhost")
    server_url = f"http://{server_host}:{server_port}"

    try:
        if not wait_for_server(
                server_host, server_port, timeout_seconds=server_start_timeout):
            raise RuntimeError(
                f"Disaggregated server did not become ready within "
                f"{server_start_timeout}s")

        # Pay the one-time autotuner cost before the measured run. The first
        # request of each shape triggers a ~20s autotuner host-step; left in
        # the measured run at high concurrency, a worker stuck in that step
        # misses the 2s cluster heartbeat and is evicted mid-run, causing
        # "Cluster is not ready" 500s. Default count = ~20s at the ~6 req/s
        # observed in the 5k B200 run; results are discarded.
        if warmup_request_count is None:
            warmup_request_count = 120
        print(
            f"[mixed-stress] warmup: {warmup_request_count} requests "
            f"(results discarded)",
            flush=True)
        asyncio.run(
            _warmup_requests(server_url, profiles, warmup_request_count,
                             concurrency, model_path))

        summary = asyncio.run(
            _run_mixed_stress_async(
                server_url,
                profiles,
                total_requests,
                concurrency,
                model_name=model_path,
                progress_callback=progress_callback,
                workers=ctx_workers + gen_workers,
                early_abort_rejection_rate=incomplete_threshold))

        print(
            f"Mixed stress summary: "
            f"accuracy={summary['accuracy_score']:.3f} "
            f"incomplete_rate={summary['incomplete_rate']:.3f} "
            f"(timed_out={summary['timed_out']} "
            f"server_rejected={summary['server_rejected']})\n"
            f"per_profile:\n{json.dumps(summary['per_profile'], indent=2)}",
            flush=True)
        if summary["rejection_details"]:
            print(
                f"Mixed stress rejection breakdown:\n"
                f"{json.dumps(summary['rejection_details'], indent=2)}",
                flush=True)

        # Per-profile accuracy rates. Pooled accuracy can hide a category that
        # silently produced nothing (e.g. structured output scoring 0% while
        # free-text carries the average), so surface each profile's rate and
        # fail on any profile that had enough completions but zero successes.
        dead_profiles = []
        for name, v in summary["per_profile"].items():
            if v["accuracy_metric"] is None:
                continue
            rate = v["accuracy_rate"]
            rate_str = f"{rate:.3f}" if rate is not None else "n/a"
            print(
                f"[mixed-stress] profile {name}: "
                f"{v['accuracy_ok']}/{v['accuracy_completed']} "
                f"({v['accuracy_metric']}) rate={rate_str}",
                flush=True)
            if (v["accuracy_completed"] >= per_profile_min_sample
                    and v["accuracy_ok"] == 0):
                dead_profiles.append(f"{name} (0/{v['accuracy_completed']})")

        if summary["aborted_early"]:
            raise RuntimeError(f"[mixed-stress] aborted early: "
                               f"{'; '.join(summary['aborted_early'])}")

        if dead_profiles:
            raise AssertionError(
                f"Mixed stress: profile(s) produced zero successful results "
                f"despite >= {per_profile_min_sample} completed requests: "
                f"{dead_profiles}. A whole request category is silently "
                f"failing while pooled accuracy stays above threshold.")

        score = summary["accuracy_score"]
        if score < accuracy_threshold:
            raise AssertionError(
                f"Mixed stress accuracy {score:.3f} below threshold "
                f"{accuracy_threshold:.3f}. Per-profile: "
                f"{summary['per_profile']}")

        incomplete = summary["incomplete_rate"]
        if incomplete > incomplete_threshold:
            raise AssertionError(
                f"Mixed stress incomplete_rate {incomplete:.3f} above threshold "
                f"{incomplete_threshold:.3f} "
                f"(timed_out={summary['timed_out']} "
                f"server_rejected={summary['server_rejected']} "
                f"details={summary['rejection_details']})")

        # Verify server still healthy after the stress run.
        # Probe /v1/chat/completions (not /v1/models) — the known failure mode
        # is the event loop dying while /v1/models keeps returning 200.
        client_config = config.copy()
        client_config["port"] = server_port
        client_config["hostname"] = server_host
        temp_fd, client_config_file = tempfile.mkstemp(suffix='.yaml',
                                                       dir=work_dir)
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(client_config, f)

        client_dir = f"{example_dir}/clients"
        client_cmd = [
            'python3', f'{client_dir}/disagg_client.py', '-c',
            client_config_file, '-p', f'{client_dir}/prompts.json',
            '--ignore-eos', '--server-start-timeout',
            str(server_start_timeout)
        ]
        all_worker_procs = [w.process for w in ctx_workers + gen_workers]
        check_call(client_cmd,
                   env=run_env,
                   poll_procs=all_worker_procs + [disagg_server.process])

    except Exception:
        logger.error("Mixed stress test failed")
        raise
    finally:
        diag = build_worker_diag(ctx_workers + gen_workers, disagg_server)
        if diag:
            print(f"[mixed-stress] worker health:{diag}", flush=True)
        terminate(*ctx_workers, *gen_workers, disagg_server)
    if diag:
        raise AssertionError(f"Mixed stress detected worker failure.{diag}")


def run_disaggregated_cancel_test(example_dir,
                                  test_desc,
                                  env=None,
                                  num_bursts=64,
                                  requests_per_burst=64,
                                  server_start_timeout=1200,
                                  model_path=None,
                                  cwd=None,
                                  prompt_len_range=(2000, 8000)):
    """Run disaggregated test with request cancellation stress test."""
    cleanup_output_files()
    run_env = env.copy()
    run_env["UCX_TLS"] = get_ucx_tls()

    config_file = get_test_config(test_desc, example_dir,
                                  os.path.dirname(__file__))
    config, ctx_workers, gen_workers, disagg_server, server_port, work_dir = \
        setup_disagg_cluster(config_file, model_name=model_path, env=run_env, cwd=cwd,
                             server_start_timeout=server_start_timeout)

    server_host = config.get("hostname", "localhost")
    server_url = f"http://{server_host}:{server_port}"

    try:
        # Wait for server to be ready
        if not wait_for_server(
                server_host, server_port, timeout_seconds=server_start_timeout):
            raise RuntimeError(
                f"Disaggregated server did not become ready within {server_start_timeout} seconds"
            )

        # Run the cancel stress test
        run_cancel_stress_test(server_url,
                               num_bursts=num_bursts,
                               requests_per_burst=requests_per_burst,
                               prompt_len_range=prompt_len_range)

        # Create a temporary client config with the correct dynamic port
        client_config = config.copy()
        client_config["port"] = server_port
        client_config["hostname"] = server_host
        temp_fd, client_config_file = tempfile.mkstemp(suffix='.yaml',
                                                       dir=work_dir)
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(client_config, f)

        # Verify server is still healthy after stress test by sending a normal request
        client_dir = f"{example_dir}/clients"
        client_cmd = [
            'python3', f'{client_dir}/disagg_client.py', '-c',
            client_config_file, '-p', f'{client_dir}/prompts.json',
            '--ignore-eos', '--server-start-timeout',
            str(server_start_timeout)
        ]
        all_worker_procs = [w.process for w in ctx_workers + gen_workers]
        check_call(client_cmd,
                   env=env,
                   poll_procs=all_worker_procs + [disagg_server.process])

    except Exception:
        logger.error("Cancel test failed")
        raise
    finally:
        terminate(*ctx_workers, *gen_workers, disagg_server)
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_disaggregated_cancel_large_context_requests(disaggregated_test_root,
                                                     disaggregated_example_root,
                                                     llm_venv,
                                                     deepseek_v3_model_root):
    """
    Test that the disaggregated server handles request cancellations gracefully.

    This test sends bursts of requests with large contexts and cancels them
    during prefill to stress test resource cleanup.

    DeepSeek-V3-Lite bf16 (~37 GiB) requires two disagg workers on separate
    GPUs. On single-GPU systems with <80 GiB, fall back to TinyLlama to avoid
    OOM while still exercising the cancellation code path.
    """
    import torch
    gpu_mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    num_gpus = torch.cuda.device_count()

    if gpu_mem_gib < 80 and num_gpus < 2:
        model_path = os.path.join(llm_models_root(), "llama-models-v2",
                                  "TinyLlama-1.1B-Chat-v1.0")
        setup_model_symlink(llm_venv, model_path,
                            "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        test_desc = "cancel_stress_test_small"
        prompt_len_range = (200, 800)
    else:
        model_path = deepseek_v3_model_root
        setup_model_symlink(llm_venv, model_path, "DeepSeek-V3-Lite/bf16")
        test_desc = "cancel_stress_test"
        prompt_len_range = (2000, 8000)

    run_disaggregated_cancel_test(disaggregated_example_root,
                                  test_desc,
                                  env=llm_venv._new_env,
                                  num_bursts=5,
                                  requests_per_burst=32,
                                  model_path=model_path,
                                  prompt_len_range=prompt_len_range,
                                  cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b-instruct'],
                         indirect=True)
def test_disaggregated_logprobs_serving(disaggregated_test_root,
                                        disaggregated_example_root, llm_venv,
                                        llama_model_root):
    """Test logprobs via OpenAI API in disaggregated serving with multi-GPU TP.

    Covers the RCCA scenario (NVBug 5926823): disaggregated + streaming + logprobs,
    where the context worker returns prefill result (request_type=generation_only)
    to the generation worker. Ensures LogProbStorage flows correctly across the
    context/gen boundary without AttributeError on cum_log_probs.
    """

    async def iter_sse_chunks(resp):
        """Yield parsed JSON chunks from an OpenAI SSE stream."""
        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[len("data: "):]
            if data_str == "[DONE]":
                break
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue

    async def collect_streaming_logprobs(resp, api_type):
        """Parse SSE stream and return (tokens, logprobs) lists."""
        tokens, logprobs = [], []
        async for chunk in iter_sse_chunks(resp):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            lp_data = choices[0].get("logprobs")
            if not lp_data:
                continue
            if api_type == "completions":
                tokens.extend(lp_data.get("tokens", []))
                logprobs.extend(lp_data.get("token_logprobs", []))
            else:
                for item in lp_data.get("content", []):
                    tokens.append(item.get("token"))
                    logprobs.append(item.get("logprob"))
        return tokens, logprobs

    def extract_logprobs(result, api_type):
        """Extract (tokens, logprobs) from non-streaming OpenAI response."""
        choices = result.get("choices", [])
        assert len(choices) > 0, "Response should have choices"
        if api_type == "completions":
            lp_data = choices[0].get("logprobs")
            assert lp_data is not None, "Response should contain logprobs"
            tokens = lp_data.get("tokens", [])
            logprobs = lp_data.get("token_logprobs", [])
            assert len(tokens) == len(logprobs), (
                f"count mismatch: {len(logprobs)} logprobs "
                f"for {len(tokens)} tokens")
            return tokens, logprobs
        lp_obj = choices[0].get("logprobs")
        assert lp_obj is not None, "Response should contain logprobs"
        content = lp_obj.get("content", [])
        tokens = [item.get("token") for item in content]
        logprobs = [item.get("logprob") for item in content]
        return tokens, logprobs

    setup_model_symlink(llm_venv, llama_model_root,
                        "llama-3.1-model/Llama-3.1-8B-Instruct")

    config_file = get_test_config("llama31_8b_ucx", disaggregated_example_root,
                                  os.path.dirname(__file__))

    env = llm_venv._new_env.copy()
    env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    env["UCX_TLS"] = get_ucx_tls()
    ctx_workers, gen_workers, disagg_server, work_dir = [], [], None, None
    config, ctx_workers, gen_workers, disagg_server, server_port, work_dir = \
        setup_disagg_cluster(config_file, env=env,
                             model_name=llama_model_root,
                             cwd=llm_venv.get_working_directory(),
                             server_start_timeout=600)

    server_host = config.get("hostname", "localhost")
    server_url = f"http://{server_host}:{server_port}"
    model_name = "llama-3.1-model/Llama-3.1-8B-Instruct"
    max_tokens = 20
    timeout = aiohttp.ClientTimeout(total=120)
    # Use emoji prompt to also stress-test multi-byte tokenizer handling
    prompt = "I love coding 🚀 and AI."

    async def check_logprobs():
        async with aiohttp.ClientSession() as session:
            for api_type in ("completions", "chat"):
                url = (f"{server_url}/v1/completions"
                       if api_type == "completions" else
                       f"{server_url}/v1/chat/completions")

                def make_payload(prompt, stream, _api_type=api_type):
                    base = {
                        "max_tokens": max_tokens,
                        "logprobs": 1 if _api_type == "completions" else True,
                        "stream": stream,
                        "temperature": 0
                    }
                    if _api_type == "completions":
                        return {"model": model_name, "prompt": prompt, **base}
                    return {
                        "model": model_name,
                        "messages": [{
                            "role": "user",
                            "content": prompt
                        }],
                        **base
                    }

                # 1) Streaming vs non-streaming consistency check
                async with session.post(url,
                                        json=make_payload(prompt, False),
                                        timeout=timeout) as resp:
                    assert resp.status == 200, \
                        f"[{api_type}] non-streaming: {await resp.text()}"
                    ns_tokens, ns_logprobs = extract_logprobs(
                        await resp.json(), api_type)

                async with session.post(url,
                                        json=make_payload(prompt, True),
                                        timeout=timeout) as resp:
                    assert resp.status == 200, \
                        f"[{api_type}] streaming: {await resp.text()}"
                    st_tokens, st_logprobs = \
                        await collect_streaming_logprobs(resp, api_type)

                assert ns_tokens == st_tokens, (
                    f"[{api_type}] streaming vs non-streaming tokens mismatch")
                assert len(ns_logprobs) == len(st_logprobs), (
                    f"[{api_type}] logprobs length: "
                    f"{len(ns_logprobs)} vs {len(st_logprobs)}")
                # Skip position 0: the first token logprob can diverge
                # between streaming and non-streaming in disaggregated mode
                # due to the context/generation handoff boundary.
                comparable = 0
                for i, (n, s) in enumerate(
                        zip(ns_logprobs, st_logprobs, strict=True)):
                    if i == 0 or n is None or s is None:
                        continue
                    comparable += 1
                    rtol, atol = (1e-3, 1e-4) if api_type == "chat" else (1e-4,
                                                                          1e-5)
                    assert np.isclose(n, s, rtol=rtol, atol=atol), \
                        f"[{api_type}] logprob mismatch at {i}: {n} vs {s}"
                assert comparable > 0, (
                    f"[{api_type}] no comparable post-handoff logprobs found")

                # 2) Chat API with top_logprobs (requires gather_generation_logits)
                if api_type == "chat":
                    top_lp_payload = {
                        "model": model_name,
                        "messages": [{
                            "role": "user",
                            "content": prompt
                        }],
                        "max_tokens": max_tokens,
                        "logprobs": True,
                        "top_logprobs": 3,
                        "stream": False,
                        "temperature": 0,
                    }
                    async with session.post(f"{server_url}/v1/chat/completions",
                                            json=top_lp_payload,
                                            timeout=timeout) as resp:
                        assert resp.status == 200, (
                            f"[chat/top_logprobs] {resp.status}: "
                            f"{await resp.text()}")
                        result = await resp.json()
                    lp_obj = result["choices"][0].get("logprobs")
                    assert lp_obj is not None, "top_logprobs response should have logprobs"
                    content = lp_obj.get("content", [])
                    assert len(
                        content) > 0, "top_logprobs content should be non-empty"
                    for item in content:
                        top_lps = item.get("top_logprobs")
                        assert top_lps is not None and len(top_lps) > 0, (
                            f"top_logprobs should be non-empty when requested: {item}"
                        )
                        for tl in top_lps:
                            assert "token" in tl and "logprob" in tl, (
                                f"top_logprob entry missing token/logprob: {tl}"
                            )
                            assert tl["logprob"] <= 0.0, (
                                f"top_logprob {tl['logprob']} should be <= 0")

    try:
        asyncio.run(check_logprobs())
    finally:
        terminate(*ctx_workers, *gen_workers, disagg_server)
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.skip_less_device(8)
@skip_pre_blackwell
@pytest.mark.parametrize("model_path", ['DeepSeek-V3-0324-FP4'])
def test_disaggregated_cancel_large_context_requests_long(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        model_path):
    """Test that disaggregated server handles request cancellations gracefully.

    This test sends bursts of requests with large contexts and cancels them
    during prefill to stress test resource cleanup.
    """
    model_dir = f"{llm_models_root()}/{model_path}"
    setup_model_symlink(llm_venv, model_dir, model_path)

    run_disaggregated_cancel_test(disaggregated_example_root,
                                  "cancel_stress_test_large",
                                  env=llm_venv._new_env,
                                  num_bursts=1000,
                                  requests_per_burst=32,
                                  model_path=model_dir,
                                  cwd=llm_venv.get_working_directory())


@pytest.mark.skip_less_device(8)
@skip_pre_blackwell
@pytest.mark.parametrize("model_path",
                         ['NVIDIA-Nemotron-3-Super-120B-A12B-FP8'])
def test_disaggregated_mamba_conc_greater_than_mbs(disaggregated_example_root,
                                                   llm_venv, model_path,
                                                   benchmark_root,
                                                   shared_gpt_path):
    model_dir = f"{llm_models_root()}/{model_path}"
    setup_model_symlink(llm_venv, model_dir, model_path)

    config_file = get_test_config("mamba_conc_greater_than_mbs",
                                  disaggregated_example_root,
                                  os.path.dirname(__file__))

    env = llm_venv._new_env.copy()
    env["UCX_TLS"] = get_ucx_tls()
    e2el, ttft = run_disaggregated_benchmark(
        disaggregated_example_root,
        config_file,
        benchmark_root,
        model_dir,
        shared_gpt_path,
        env=env,
        num_prompts=40,
        max_concurrency=4,
        random_input_len=1024,
        random_output_len=1024,
        skip_warmup=True,
        model_path=model_dir,
        cwd=llm_venv.get_working_directory())
    print(f"E2EL: {e2el} ms, TTFT: {ttft} ms")


@pytest.mark.parametrize(
    "test_config",
    [
        # Smoke run: 120 requests at 64 concurrency (matching the warmup
        # count), ~3 min request phase on B200. Used as L0 post-merge gate. A
        # healthy cluster scores ~1.0 accuracy (every profile validates real
        # content), so 0.9 leaves margin for a rare flaky request while still
        # catching a regression.
        pytest.param(TestConfig(
            model_path='Qwen3/Qwen3-32B-FP8',
            test_desc='req120-conc64-qwen3_32b_fp8_mixed_stress',
            request_count=120,
            concurrency=64,
            accuracy_threshold=0.9,
            speculative_model_path='Zhi-Create-Qwen3-32B-Eagle3'),
                     marks=(pytest.mark.skip_less_device(8), skip_pre_hopper)),
        # Full stress run: 10k requests at 512 concurrency.
        # Estimated wall-clock ~40 min (server startup + ~32 min request
        # phase); 512 concurrency exercises more in-flight overlap.
        pytest.param(TestConfig(
            model_path='Qwen3/Qwen3-32B-FP8',
            test_desc='req10k-conc512-qwen3_32b_fp8_mixed_stress',
            request_count=10000,
            concurrency=512,
            accuracy_threshold=0.9,
            speculative_model_path='Zhi-Create-Qwen3-32B-Eagle3'),
                     marks=(pytest.mark.skip_less_device(8), skip_pre_hopper)),
    ],
    ids=lambda x: x.test_desc)
def test_disaggregated_mixed_stress_test(disaggregated_test_root,
                                         disaggregated_example_root, llm_venv,
                                         test_config):
    model_path = test_config.model_path
    test_desc = test_config.test_desc
    model_dir = resolve_llm_model_path(model_path)
    setup_model_symlink(llm_venv, model_dir, model_path)

    config_file = get_test_config(test_desc, disaggregated_example_root,
                                  os.path.dirname(__file__))

    if test_config.speculative_model_path is not None:
        spec_model_dir = f"{llm_models_root()}/{test_config.speculative_model_path}"
        setup_model_symlink(llm_venv, spec_model_dir,
                            test_config.speculative_model_path)
        with open(config_file, 'r') as f:
            patched_config = yaml.safe_load(f)
        patched_sections = []
        # Check top-level speculative_config first (current YAML layout), then
        # fall back to per-server blocks for older config shapes.
        top_spec = patched_config.get('speculative_config')
        if isinstance(top_spec, dict) and 'speculative_model' in top_spec:
            top_spec['speculative_model'] = spec_model_dir
            patched_sections.append('top-level')
        else:
            for section in ('context_servers', 'generation_servers'):
                spec = patched_config.get(section, {}).get('speculative_config')
                if spec is not None and 'speculative_model' in spec:
                    spec['speculative_model'] = spec_model_dir
                    patched_sections.append(section)
        if not patched_sections:
            raise AssertionError(
                f"{test_desc} sets speculative_model_path, but no "
                "speculative_config.speculative_model field was patched")
        patched_path = os.path.join(llm_venv.get_working_directory(),
                                    f"{test_desc}_patched.yaml")
        with open(patched_path, 'w') as f:
            yaml.safe_dump(patched_config, f)
        config_file = patched_path

    run_disaggregated_mixed_stress(
        example_dir=disaggregated_example_root,
        config_file=config_file,
        model_path=model_dir,
        total_requests=test_config.request_count,
        concurrency=test_config.concurrency,
        accuracy_threshold=test_config.accuracy_threshold,
        incomplete_threshold=test_config.incomplete_threshold,
        server_start_timeout=600,
        env=llm_venv._new_env,
        cwd=llm_venv.get_working_directory())
