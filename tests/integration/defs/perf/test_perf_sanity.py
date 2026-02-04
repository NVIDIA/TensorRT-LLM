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
"""TensorRT LLM perf sanity tests."""

import contextlib
import copy
import glob
import io
import os
import re
import socket
import subprocess
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import pytest
import yaml
from test_common.http_utils import wait_for_endpoint_ready

from defs.trt_test_alternative import print_error, print_info
from tensorrt_llm._utils import get_free_port

from ..conftest import get_llm_root, llm_models_root
from .open_search_db_utils import (
    SCENARIO_MATCH_FIELDS,
    add_id,
    check_perf_regression,
    get_common_values,
    get_history_data,
    get_job_info,
    post_new_perf_data,
    prepare_baseline_data,
    prepare_regressive_test_cases,
)
from .utils import collect_and_clean_myelin_time

# Model PATH of local dir synced from internal LLM models repo
MODEL_PATH_DICT = {
    "deepseek_r1_fp8": "DeepSeek-R1/DeepSeek-R1",
    "deepseek_r1_nvfp4": "DeepSeek-R1/DeepSeek-R1-FP4",
    "deepseek_r1_0528_fp8": "DeepSeek-R1/DeepSeek-R1-0528/",
    "deepseek_r1_0528_fp4": "DeepSeek-R1/DeepSeek-R1-0528-FP4/",
    "deepseek_r1_0528_fp4_v2": "DeepSeek-R1/DeepSeek-R1-0528-FP4-v2/",
    "deepseek_v32_fp4": "DeepSeek-V3.2-Exp-FP4-v2",
    "gpt_oss_120b_fp4": "gpt_oss/gpt-oss-120b",
    "k2_thinking_fp4": "Kimi-K2-Thinking-NVFP4",
    "qwen3_235b_a22b_fp4": "Qwen3/saved_models_Qwen3-235B-A22B_nvfp4_hf",  # Qwen3-235B-A22B-FP4
}

SUPPORTED_GPU_MAPPING = {
    "GB200": "gb200",
    "GB300": "gb300",
    "B200": "b200",
    "B300": "b300",
    "H200": "h200",
}

DEFAULT_TIMEOUT = 7200

AGGR_CONFIG_FOLDER = "tests/scripts/perf-sanity"
DISAGG_CONFIG_FOLDER = "tests/integration/defs/perf/disagg/test_configs/disagg/perf-sanity"

# Regex patterns for parsing benchmark output metrics
# Key is the metric name used in database (e.g., "mean_e2el", "seq_throughput")
PERF_METRIC_LOG_QUERIES = {
    "seq_throughput": re.compile(r"Request throughput \(req\/s\):\s+(-?[\d\.]+)"),
    "token_throughput": re.compile(r"Output token throughput \(tok\/s\):\s+(-?[\d\.]+)"),
    "total_token_throughput": re.compile(r"Total Token throughput \(tok\/s\):\s+(-?[\d\.]+)"),
    "user_throughput": re.compile(r"User throughput \(tok\/s\):\s+(-?[\d\.]+)"),
    "mean_ttft": re.compile(r"Mean TTFT \(ms\):\s+(-?[\d\.]+)"),
    "median_ttft": re.compile(r"Median TTFT \(ms\):\s+(-?[\d\.]+)"),
    "p99_ttft": re.compile(r"P99 TTFT \(ms\):\s+(-?[\d\.]+)"),
    "mean_itl": re.compile(r"Mean ITL \(ms\):\s+(-?[\d\.]+)"),
    "median_itl": re.compile(r"Median ITL \(ms\):\s+(-?[\d\.]+)"),
    "p99_itl": re.compile(r"P99 ITL \(ms\):\s+(-?[\d\.]+)"),
    "mean_tpot": re.compile(r"Mean TPOT \(ms\):\s+(-?[\d\.]+)"),
    "median_tpot": re.compile(r"Median TPOT \(ms\):\s+(-?[\d\.]+)"),
    "p99_tpot": re.compile(r"P99 TPOT \(ms\):\s+(-?[\d\.]+)"),
    "mean_e2el": re.compile(r"Mean E2EL \(ms\):\s+(-?[\d\.]+)"),
    "median_e2el": re.compile(r"Median E2EL \(ms\):\s+(-?[\d\.]+)"),
    "p99_e2el": re.compile(r"P99 E2EL \(ms\):\s+(-?[\d\.]+)"),
}


def get_model_dir(model_name: str) -> str:
    """Get model directory path from model name."""
    if model_name in MODEL_PATH_DICT:
        return os.path.join(llm_models_root(), MODEL_PATH_DICT[model_name])
    return ""


def get_dataset_dir(dataset_file: Optional[str]) -> str:
    """Get dataset directory path from dataset file."""
    if not dataset_file or dataset_file == "<dataset_file>":
        return ""

    # return os.path.join(llm_models_root(), "datasets", "ShareGPT_V3_unfiltered_cleaned_split.json")
    llm_models_path = os.path.join(llm_models_root(), dataset_file)
    if os.path.exists(llm_models_path):
        return llm_models_path
    elif os.path.exists(dataset_file):
        return dataset_file
    else:
        print_info(f"Dataset file not found in {llm_models_path} and {dataset_file}")
        return ""


def to_env_dict(env_vars: str) -> Dict[str, str]:
    """Convert env vars string to dict."""
    env = {}
    for env_var in env_vars.split():
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            env[key] = value
    return env


def add_host_port_to_cmd(cmd: List[str], host: str, port: int) -> List[str]:
    """Add host and port to command."""
    return cmd + ["--host", host, "--port", str(port)]


class ServerConfig:
    """Configurations of trtllm-server."""

    def __init__(self, server_config_data: dict, env_vars: str = ""):
        # Extract required fields
        self.concurrency = server_config_data.get("concurrency", 1)
        self.model_name = server_config_data["model_name"]
        self.model_path = ""
        self.env_vars = env_vars
        self.disagg_run_type = server_config_data.get("disagg_run_type", "aggr")

        # Extract optional fields with defaults
        self.tp = server_config_data.get("tensor_parallel_size", 1)
        self.ep = server_config_data.get("moe_expert_parallel_size", 1)
        self.pp = server_config_data.get("pipeline_parallel_size", 1)
        self.cp = server_config_data.get("context_parallel_size", 1)
        self.gpus = server_config_data.get("gpus", self.tp * self.cp * self.pp)
        self.gpus_per_node = server_config_data.get("gpus_per_node", 0) or self.gpus
        self.max_num_tokens = server_config_data.get("max_num_tokens", 2048)
        self.max_batch_size = server_config_data.get("max_batch_size", 512)
        self.max_seq_len = server_config_data.get("max_seq_len", 0)
        self.disable_overlap_scheduler = server_config_data.get("disable_overlap_scheduler", False)
        self.num_postprocess_workers = server_config_data.get("num_postprocess_workers", 0)
        self.stream_interval = server_config_data.get("stream_interval", 10)
        self.print_iter_log = server_config_data.get("print_iter_log", False)
        self.attn_backend = server_config_data.get("attn_backend", "TRTLLM")
        self.enable_chunked_prefill = server_config_data.get("enable_chunked_prefill", False)
        self.enable_attention_dp = server_config_data.get("enable_attention_dp", False)
        self.trust_remote_code = server_config_data.get("trust_remote_code", False)
        self.enable_lm_head_tp_in_adp = server_config_data.get("enable_lm_head_tp_in_adp", False)

        # attention_dp_config
        attention_dp_config = server_config_data.get("attention_dp_config", {})
        self.attention_dp_balance = attention_dp_config.get("enable_balance", False)
        self.batching_wait_iters = attention_dp_config.get("batching_wait_iters", 0)
        self.timeout_iters = attention_dp_config.get("timeout_iters", 60)

        # moe_config
        moe_config = server_config_data.get("moe_config", {})
        self.moe_backend = moe_config.get("backend", "")
        self.moe_max_num_tokens = moe_config.get("max_num_tokens", 0)
        self.use_low_precision_moe_combine = moe_config.get("use_low_precision_moe_combine", False)
        load_balancer_config = moe_config.get("load_balancer", {})
        self.load_balancer_num_slots = load_balancer_config.get("num_slots", 0)
        self.load_balancer_layer_updates_per_iter = load_balancer_config.get(
            "layer_updates_per_iter", 0
        )

        # cuda_graph_config
        cuda_graph_config = server_config_data.get("cuda_graph_config", {})
        self.enable_cuda_graph = False
        if cuda_graph_config:
            self.enable_cuda_graph = True
            self.enable_padding = cuda_graph_config.get("enable_padding", True)
            self.cuda_graph_batch_sizes = cuda_graph_config.get("batch_sizes", [])
            self.cuda_graph_max_batch_size = cuda_graph_config.get("max_batch_size", 0)
        else:
            self.enable_padding = True
            self.cuda_graph_batch_sizes = []
            self.cuda_graph_max_batch_size = 0

        # kv_cache_config
        kv_cache_config = server_config_data.get("kv_cache_config", {})
        self.kv_cache_dtype = kv_cache_config.get("dtype", "fp8")
        self.enable_block_reuse = kv_cache_config.get("enable_block_reuse", False)
        self.free_gpu_memory_fraction = kv_cache_config.get("free_gpu_memory_fraction", 0.8)

        # cache_transceiver_config
        cache_transceiver_config = server_config_data.get("cache_transceiver_config", {})
        self.cache_transceiver_backend = cache_transceiver_config.get("backend", "")
        self.cache_transceiver_max_tokens_in_buffer = cache_transceiver_config.get(
            "max_tokens_in_buffer", 0
        )

        # Generate default name if not provided
        self.name = server_config_data.get("name", "")
        if not self.name:
            self.name = (
                f"{self.model_name}_tp{self.tp}_ep{self.ep}_pp{self.pp}_cp{self.cp}"
                f"_bs{self.max_batch_size}_attn{self.attn_backend}_moe{self.moe_backend}"
            )
            if self.cache_transceiver_backend:
                self.name += f"_spec{self.cache_transceiver_backend}"

        # speculative_config
        speculative_config = server_config_data.get("speculative_config", {})
        self.spec_decoding_type = speculative_config.get("decoding_type", "")
        self.num_nextn_predict_layers = speculative_config.get("num_nextn_predict_layers", 0)
        eagle3_value = speculative_config.get("eagle3_layers_to_capture", [])
        if isinstance(eagle3_value, int):
            self.eagle3_layers_to_capture = [eagle3_value]
        elif isinstance(eagle3_value, list):
            self.eagle3_layers_to_capture = eagle3_value
        else:
            self.eagle3_layers_to_capture = []
        self.max_draft_len = speculative_config.get("max_draft_len", 0)
        self.speculative_model = speculative_config.get("speculative_model", "")
        self.eagle3_one_model = speculative_config.get("eagle3_one_model", False)

        # match_mode: "config" (default) or "scenario"
        self.match_mode = server_config_data.get("match_mode", "config")

        # Store filtered config for extra_llm_api_config
        exclude_keys = [
            "mode",
            "concurrency",
            "name",
            "model_name",
            "disagg_run_type",
            "gpus",
            "gpus_per_node",
            "match_mode",
            "client_configs",
            "match_mode",
        ]
        self.extra_llm_api_config_data = {
            k: v for k, v in server_config_data.items() if k not in exclude_keys
        }

    def to_cmd(
        self, output_dir: str, numa_bind: bool = False, disagg_serving_type: str = ""
    ) -> List[str]:
        """Generate server command."""
        model_dir = get_model_dir(self.model_name)
        self.model_path = model_dir if os.path.exists(model_dir) else self.model_name
        config_filename = f"extra-llm-api-config.{self.disagg_run_type}.{self.name}.yml"
        config_path = os.path.join(output_dir, config_filename)

        numa_bind_cmd = []
        if numa_bind:
            numa_bind_cmd = ["numactl", "-m 0,1"]

        cmd = numa_bind_cmd + [
            "trtllm-serve",
            self.model_path,
            "--backend",
            "pytorch",
            "--config",
            config_path,
        ]
        return cmd

    def to_env(self) -> Dict[str, str]:
        return to_env_dict(self.env_vars)

    def to_match_keys(self) -> List[str]:
        return [
            "s_model_name",
            "l_tp",
            "l_ep",
            "l_pp",
            "l_cp",
            "l_gpus_per_node",
            "l_max_batch_size",
            "b_disable_overlap_scheduler",
            "l_num_postprocess_workers",
            "s_attn_backend",
            "b_enable_chunked_prefill",
            "b_enable_attention_dp",
            "b_enable_lm_head_tp_in_adp",
            # attention_dp_config
            "b_attention_dp_balance",
            # moe_config
            "s_moe_backend",
            # cuda_graph_config
            "b_enable_cuda_graph",
            # kv_cache_config
            "s_kv_cache_dtype",
            # cache_transceiver_config
            "s_cache_transceiver_backend",
            # speculative_config
            "s_spec_decoding_type",
            "l_num_nextn_predict_layers",
        ]

    def to_db_data(self) -> dict:
        """Convert ServerConfig to database data."""
        db_data = {
            "s_server_name": self.name,
            "s_model_name": self.model_name.lower(),
            "l_gpus": self.gpus,
            "l_tp": self.tp,
            "l_ep": self.ep,
            "l_pp": self.pp,
            "l_cp": self.cp,
            "l_gpus_per_node": self.gpus_per_node,
            "l_max_num_tokens": self.max_num_tokens,
            "l_max_batch_size": self.max_batch_size,
            "l_max_seq_len": self.max_seq_len,
            "b_disable_overlap_scheduler": self.disable_overlap_scheduler,
            "l_num_postprocess_workers": self.num_postprocess_workers,
            "l_stream_interval": self.stream_interval,
            "s_attn_backend": self.attn_backend,
            "b_enable_chunked_prefill": self.enable_chunked_prefill,
            "b_enable_attention_dp": self.enable_attention_dp,
            "b_trust_remote_code": self.trust_remote_code,
            "b_enable_lm_head_tp_in_adp": self.enable_lm_head_tp_in_adp,
            # attention_dp_config
            "b_attention_dp_balance": self.attention_dp_balance,
            "l_batching_wait_iters": self.batching_wait_iters,
            "l_timeout_iters": self.timeout_iters,
            # moe_config
            "s_moe_backend": self.moe_backend,
            "l_moe_max_num_tokens": self.moe_max_num_tokens,
            "b_use_low_precision_moe_combine": self.use_low_precision_moe_combine,
            "l_load_balancer_num_slots": self.load_balancer_num_slots,
            "l_load_balancer_layer_updates_per_iter": self.load_balancer_layer_updates_per_iter,
            # cuda_graph_config
            "b_enable_cuda_graph": self.enable_cuda_graph,
            "b_enable_padding": self.enable_padding,
            "l_cuda_graph_max_batch_size": self.cuda_graph_max_batch_size,
            "s_cuda_graph_batch_sizes": ",".join(map(str, self.cuda_graph_batch_sizes)),
            # kv_cache_config
            "s_kv_cache_dtype": self.kv_cache_dtype,
            "b_enable_block_reuse": self.enable_block_reuse,
            "d_free_gpu_memory_fraction": self.free_gpu_memory_fraction,
            # cache_transceiver_config
            "s_cache_transceiver_backend": self.cache_transceiver_backend,
            "l_cache_transceiver_max_tokens_in_buffer": self.cache_transceiver_max_tokens_in_buffer,
            # speculative_config
            "s_spec_decoding_type": self.spec_decoding_type,
            "l_num_nextn_predict_layers": self.num_nextn_predict_layers,
            "s_eagle3_layers_to_capture": ",".join(map(str, self.eagle3_layers_to_capture)),
            "l_max_draft_len": self.max_draft_len,
            "s_speculative_model_dir": self.speculative_model,
            "b_eagle3_one_model": self.eagle3_one_model,
            "s_server_log_link": "",
            "s_server_env_var": self.env_vars,
        }
        return db_data

    def generate_extra_llm_api_config(self) -> str:
        """Generate extra-llm-api-config.yml content."""
        config_data = dict(self.extra_llm_api_config_data)

        # Handle speculative_model path conversion
        if (
            "speculative_config" in config_data
            and "speculative_model" in config_data["speculative_config"]
        ):
            spec_model = config_data["speculative_config"]["speculative_model"]
            if spec_model:
                config_data["speculative_config"]["speculative_model"] = os.path.join(
                    llm_models_root(), spec_model
                )

        return yaml.dump(config_data, default_flow_style=False, sort_keys=False)


class ClientConfig:
    """Configurations of benchmark client."""

    def __init__(
        self,
        client_config_data: dict,
        model_name: str,
        env_vars: str = "",
    ):
        self.model_name = model_name
        self.concurrency = client_config_data.get("concurrency", 1)
        self.iterations = client_config_data.get("iterations", 1)
        self.isl = client_config_data.get("isl", 1024)
        self.osl = client_config_data.get("osl", 1024)
        self.random_range_ratio = client_config_data.get("random_range_ratio", 0.0)
        self.backend = client_config_data.get("backend", "openai")
        self.use_chat_template = client_config_data.get("use_chat_template", False)
        self.streaming = client_config_data.get("streaming", True)
        self.trust_remote_code = client_config_data.get("trust_remote_code", True)
        self.model_path = ""
        self.dataset_file = client_config_data.get("dataset_file", "")
        self.env_vars = env_vars

        # Generate default name if not provided
        self.name = client_config_data.get("name", "")
        if not self.name:
            self.name = f"con{self.concurrency}_iter{self.iterations}_isl{self.isl}_osl{self.osl}"

    def to_cmd(self) -> List[str]:
        """Generate benchmark command."""
        model_dir = get_model_dir(self.model_name)
        self.model_path = model_dir if os.path.exists(model_dir) else self.model_name
        dataset_path = get_dataset_dir(self.dataset_file)
        benchmark_cmd = [
            "python",
            "-m",
            "tensorrt_llm.serve.scripts.benchmark_serving",
            "--model",
            self.model_path,
            "--tokenizer",
            self.model_path,
            "--num-prompts",
            str(self.concurrency * self.iterations),
            "--max-concurrency",
            str(self.concurrency),
            "--random-input-len",
            str(self.isl),
            "--random-output-len",
            str(self.osl),
            "--ignore-eos",
            "--no-test-input",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
        ]
        if dataset_path:
            benchmark_cmd.append("--dataset-name")
            benchmark_cmd.append("trtllm_custom")
            benchmark_cmd.append("--dataset-path")
            benchmark_cmd.append(dataset_path)
            print_info(f"Dataset: {dataset_path} exists. Use trtllm_custom dataset for benchmark.")
        else:
            benchmark_cmd.append("--dataset-name")
            benchmark_cmd.append("random")
            benchmark_cmd.append("--random-ids")
            benchmark_cmd.append("--random-range-ratio")
            benchmark_cmd.append(str(self.random_range_ratio))
            print_info(
                f"Dataset: {dataset_path} is not provided or does not exist. "
                f"Use random dataset (random_range_ratio={self.random_range_ratio}) for benchmark."
            )
        if self.backend:
            benchmark_cmd.append("--backend")
            benchmark_cmd.append(self.backend)
        if self.use_chat_template:
            benchmark_cmd.append("--use-chat-template")
        if not self.streaming:
            benchmark_cmd.append("--non-streaming")
        if self.trust_remote_code:
            benchmark_cmd.append("--trust-remote-code")
        return benchmark_cmd

    def to_env(self) -> Dict[str, str]:
        return to_env_dict(self.env_vars)

    def to_match_keys(self) -> List[str]:
        return [
            "l_concurrency",
            "l_iterations",
            "l_isl",
            "l_osl",
            "d_random_range_ratio",
            "s_backend",
            "b_use_chat_template",
            "b_streaming",
        ]

    def to_db_data(self) -> dict:
        """Convert ClientConfig to database data."""
        db_data = {
            "s_client_name": self.name,
            "l_concurrency": self.concurrency,
            "l_iterations": self.iterations,
            "l_isl": self.isl,
            "l_osl": self.osl,
            "d_random_range_ratio": self.random_range_ratio,
            "s_dataset_file": self.dataset_file,
            "s_backend": self.backend,
            "b_use_chat_template": self.use_chat_template,
            "b_streaming": self.streaming,
            "b_trust_remote_code": self.trust_remote_code,
            "s_client_log_link": "",
            "s_client_env_vars": self.env_vars,
        }
        if self.backend:
            db_data["s_backend"] = self.backend
        if self.use_chat_template:
            db_data["b_use_chat_template"] = self.use_chat_template
        return db_data


class DisaggConfig:
    """Configurations for disaggregated server."""

    def __init__(
        self,
        name: str,
        disagg_serving_type: str,
        hostname: str,
        numa_bind: bool,
        timeout: int,
        benchmark_mode: str,
        model_name: str,
        hardware: dict,
        server_env_var: str,
    ):
        self.name = name
        self.disagg_serving_type = disagg_serving_type
        self.hostname = hostname
        self.numa_bind = numa_bind
        self.timeout = timeout
        self.benchmark_mode = benchmark_mode
        self.model_name = model_name
        self.hardware = hardware
        self.server_env_var = server_env_var
        self.num_ctx_servers = hardware.get("num_ctx_servers", 0)
        self.num_gen_servers = hardware.get("num_gen_servers", 0)


class AggrTestCmds(NamedTuple):
    """Commands for aggregated server perf sanity tests."""

    server_cmds: List[List[str]]
    client_cmds: Dict[int, List[List[str]]]
    timeout: int
    output_dir: str

    def run_cmd(self, server_idx: int) -> List[str]:
        """Run all clients for a server and return outputs."""
        outputs = []
        server_proc = None
        server_cmd = self.server_cmds[server_idx]

        try:
            server_hostname = "localhost"
            server_port = get_free_port()
            server_cmd_with_port = add_host_port_to_cmd(server_cmd, server_hostname, server_port)

            server_file_path = os.path.join(self.output_dir, f"trtllm-serve.{server_idx}.log")

            print_info(f"Starting server. cmd is {server_cmd_with_port}")
            with open(server_file_path, "w") as server_ctx:
                server_proc = subprocess.Popen(
                    server_cmd_with_port,
                    stdout=server_ctx,
                    stderr=subprocess.STDOUT,
                    env=copy.deepcopy(os.environ),
                )

            wait_for_endpoint_ready(
                f"http://{server_hostname}:{server_port}/health",
                timeout=self.timeout,
                check_files=[server_file_path],
                server_proc=server_proc,
            )

            # Run all clients for this server
            for client_idx, client_cmd in enumerate(self.client_cmds[server_idx]):
                client_file_path = os.path.join(
                    self.output_dir, f"trtllm-benchmark.{server_idx}.{client_idx}.log"
                )

                client_cmd_with_port = add_host_port_to_cmd(
                    client_cmd, server_hostname, server_port
                )
                print_info(f"Starting client. cmd is {client_cmd_with_port}")

                output = subprocess.check_output(
                    client_cmd_with_port,
                    stderr=subprocess.STDOUT,
                    env=copy.deepcopy(os.environ),
                ).decode()

                with open(client_file_path, "w") as client_ctx:
                    client_ctx.write(output)

                outputs.append(output)

        finally:
            if server_proc:
                server_proc.terminate()
                server_proc.wait()

        return outputs

    def get_cmd_str(self, server_idx: int) -> List[str]:
        return ["aggr_server tests, please check config files"]


class DisaggTestCmds(NamedTuple):
    """Commands for multi-node disaggregated server perf sanity tests."""

    server_cmds: List[Tuple[List[str], List[str], List[str]]]
    client_cmds: Dict[int, List[List[str]]]
    timeout: int
    hostname: str
    disagg_serving_type: str
    num_ctx_servers: int
    num_gen_servers: int
    output_dir: str

    def _generate_hostname_file(self, server_idx: int, port: int):
        """Create hostname file for coordination."""
        hostnames_dir = os.path.join(self.output_dir, f"hostnames-{server_idx}")
        if not os.path.exists(hostnames_dir):
            os.makedirs(hostnames_dir, exist_ok=True)
        hostname_file = os.path.join(hostnames_dir, f"{self.disagg_serving_type}.txt")
        with open(hostname_file, "w") as f:
            f.write(f"{self.hostname}:{port}")

    def _generate_disagg_server_config(self, server_idx: int, disagg_server_port: int) -> str:
        """Generate disagg server config from hostname files."""
        print_info(f"Generating disagg server config for server index {server_idx}")
        hostnames_folder = os.path.join(self.output_dir, f"hostnames-{server_idx}")
        expected_count = self.num_ctx_servers + self.num_gen_servers
        start_time = time.time()
        hostnames = []

        while True:
            elapsed_time = time.time() - start_time
            print_info(
                f"Waiting for hostnames in {hostnames_folder}, "
                f"elapsed time: {elapsed_time}s, current: {len(hostnames)}, "
                f"expected: {expected_count}"
            )
            if elapsed_time > self.timeout:
                print_error(f"Time out. Hostnames files are not ready after {self.timeout}s")
                break
            time.sleep(10)
            if not os.path.exists(hostnames_folder):
                continue
            hostnames = os.listdir(hostnames_folder)
            if len(hostnames) >= expected_count:
                break

        print_info(f"All hostnames found in {hostnames_folder} after elapsed time: {elapsed_time}s")

        # Read ctx and gen hostnames
        ctx_hostnames = []
        gen_hostnames = []
        for hostname_file in hostnames:
            hostname_file_path = os.path.join(hostnames_folder, hostname_file)
            with open(hostname_file_path, "r") as f:
                hostname_port = f.read().strip()
            if hostname_file.startswith("CTX"):
                ctx_hostnames.append(hostname_port)
            elif hostname_file.startswith("GEN"):
                gen_hostnames.append(hostname_port)

        server_config = {
            "hostname": self.hostname,
            "port": disagg_server_port,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": self.num_ctx_servers,
                "urls": ctx_hostnames,
            },
            "generation_servers": {
                "num_instances": self.num_gen_servers,
                "urls": gen_hostnames,
            },
        }
        config_path = os.path.join(self.output_dir, f"server_config.{server_idx}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(server_config, f)
        print_info(f"Server config file {config_path} generated")
        return config_path

    def _get_disagg_server_hostname_and_port(self, server_idx: int) -> Tuple[str, int]:
        """Wait for and read disagg server config."""
        config_path = os.path.join(self.output_dir, f"server_config.{server_idx}.yaml")
        start_time = time.time()
        while True:
            if os.path.exists(config_path):
                print_info(f"Server config file found: {config_path}")
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                print_error(f"Server config file {config_path} not found after {self.timeout}s")
                break
            print_info(f"Waiting for server config file, elapsed time: {elapsed_time}s")
            time.sleep(10)

        with open(config_path, "r") as f:
            server_config = yaml.safe_load(f)
        return server_config["hostname"], server_config["port"]

    def wait_for_benchmark_ready(self, benchmark_status_file: str):
        """Wait for benchmark to complete."""
        start_time = time.time()
        while True:
            if os.path.exists(benchmark_status_file):
                print_info(
                    f"Benchmark status file found, terminating server {self.disagg_serving_type}"
                )
                break
            elapsed_time = time.time() - start_time
            print_info(f"Waiting for benchmark status file, elapsed time: {elapsed_time}s")
            if elapsed_time > self.timeout:
                print_error(f"Timeout waiting for benchmark status file after {self.timeout}s")
                break
            time.sleep(10)

    def run_cmd(self, server_idx: int) -> List[str]:
        """Run commands for a server and return outputs."""
        outputs = []
        benchmark_status_file = os.path.join(self.output_dir, f"benchmark_status.{server_idx}.txt")
        port = get_free_port()

        ctx_cmd, gen_cmd, disagg_cmd = self.server_cmds[server_idx]
        if "CTX" in self.disagg_serving_type or "GEN" in self.disagg_serving_type:
            self._generate_hostname_file(server_idx, port)
            server_file_path = os.path.join(
                self.output_dir, f"trtllm-serve.{server_idx}.{self.disagg_serving_type}.log"
            )
            is_ctx = "CTX" in self.disagg_serving_type
            server_cmd = ctx_cmd if is_ctx else gen_cmd
            server_cmd = add_host_port_to_cmd(server_cmd, self.hostname, port)
            try:
                print_info(
                    f"Starting server. disagg_serving_type: {self.disagg_serving_type} cmd is {server_cmd}"
                )
                with open(server_file_path, "w") as server_ctx:
                    server_proc = subprocess.Popen(
                        server_cmd,
                        stdout=server_ctx,
                        stderr=subprocess.STDOUT,
                        env=copy.deepcopy(os.environ),
                    )
                self.wait_for_benchmark_ready(benchmark_status_file)
            finally:
                print_info(f"Server {self.disagg_serving_type} stopped")
                server_proc.terminate()
                server_proc.wait()

        elif self.disagg_serving_type == "DISAGG_SERVER":
            disagg_server_file_path = os.path.join(
                self.output_dir, f"trtllm-serve.{server_idx}.{self.disagg_serving_type}.log"
            )
            try:
                self._generate_disagg_server_config(server_idx, port)
                print_info(f"Starting disagg server. cmd is {disagg_cmd}")
                with open(disagg_server_file_path, "w") as disagg_server_ctx:
                    disagg_server_proc = subprocess.Popen(
                        disagg_cmd,
                        stdout=disagg_server_ctx,
                        stderr=subprocess.STDOUT,
                        env=copy.deepcopy(os.environ),
                    )
                self.wait_for_benchmark_ready(benchmark_status_file)
            finally:
                print_info(f"Disagg server {self.disagg_serving_type} stopped")
                disagg_server_proc.terminate()
                disagg_server_proc.wait()

        elif self.disagg_serving_type == "BENCHMARK":
            try:
                disagg_server_hostname, disagg_server_port = (
                    self._get_disagg_server_hostname_and_port(server_idx)
                )
                server_files = [
                    os.path.join(self.output_dir, f"trtllm-serve.{server_idx}.DISAGG_SERVER.log"),
                ]
                for ctx_idx in range(self.num_ctx_servers):
                    server_files.append(
                        os.path.join(
                            self.output_dir, f"trtllm-serve.{server_idx}.CTX_{ctx_idx}.log"
                        )
                    )
                for gen_idx in range(self.num_gen_servers):
                    server_files.append(
                        os.path.join(
                            self.output_dir, f"trtllm-serve.{server_idx}.GEN_{gen_idx}.log"
                        )
                    )
                wait_for_endpoint_ready(
                    f"http://{disagg_server_hostname}:{disagg_server_port}/health",
                    timeout=self.timeout,
                    check_files=server_files,
                )

                # Run all clients for this server
                for client_idx, client_cmd in enumerate(self.client_cmds[server_idx]):
                    benchmark_file_path = os.path.join(
                        self.output_dir, f"trtllm-benchmark.{server_idx}.{client_idx}.log"
                    )

                    client_cmd_with_port = add_host_port_to_cmd(
                        client_cmd, disagg_server_hostname, disagg_server_port
                    )
                    print_info(f"Starting benchmark. cmd is {client_cmd_with_port}")

                    output = subprocess.check_output(
                        client_cmd_with_port,
                        env=copy.deepcopy(os.environ),
                        stderr=subprocess.STDOUT,
                    ).decode()

                    with open(benchmark_file_path, "w") as benchmark_ctx:
                        benchmark_ctx.write(output)
                    outputs.append(output)

            finally:
                with open(benchmark_status_file, "w") as status_file:
                    status_file.write("Done")

        return outputs

    def get_cmd_str(self, server_idx: int) -> List[str]:
        return ["multi-node disaggregated server tests, please check config files"]


def parse_select_pattern(select_pattern: str) -> list:
    """Parse select pattern (server config names).

    Args:
        select_pattern: Server config names separated by comma
            (e.g., "r1_fp4_v2_dep4_mtp1_1k1k,r1_fp4_v2_tep4_mtp3_1k1k,r1_fp4_v2_tp4_mtp3_1k1k").

    Returns:
        List of server config name strings.
    """
    return [name.strip() for name in select_pattern.split(",")]


class PerfSanityTestConfig:
    """Configuration for perf sanity tests."""

    def __init__(self, test_case_name: str, output_dir: str):
        self._output_dir = output_dir
        self._perf_results: Dict[int, List[Dict[str, float]]] = {}

        # Parse test case name
        self.parse_test_case_name(test_case_name)

    def parse_test_case_name(self, test_case_name: str):
        """Parse test case name into components."""
        self._test_param_labels = test_case_name

        # Extract configs from test param labels
        labels = self._test_param_labels.split("-")

        def get_gpu_type() -> str:
            try:
                output = subprocess.check_output(
                    "nvidia-smi -q | grep 'Product Name' | head -1",
                    shell=True,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                model = output.split()[-1]
                return SUPPORTED_GPU_MAPPING.get(model, "unsupported")
            except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
                print_error("Failed to get GPU type")
            return "unsupported"

        assert len(labels) > 1, "perf_sanity test must have a config file!"
        is_disagg = "disagg" in labels[0]
        self.upload_to_db = "upload" in labels[0]
        self.gpu_type = get_gpu_type()

        if is_disagg:
            # For disagg: disagg_upload-deepseek-r1-fp4_8k1k_ctx1_gen1_dep32_bs128_eplb0_mtp0_ccb-UCX
            self.runtime = "multi_node_disagg_server"
            self.config_dir = DISAGG_CONFIG_FOLDER
            config_base = "-".join(labels[1:])
            self.config_file = (
                f"{config_base}.yaml" if not config_base.endswith(".yaml") else config_base
            )
            self.select_pattern = None
        else:
            # For aggr: aggr_upload-config_yml or aggr_upload-config_yml-server_config_name
            self.runtime = "aggr_server"
            self.config_dir = AGGR_CONFIG_FOLDER
            config_base = labels[1]
            self.config_file = (
                f"{config_base}.yaml"
                if config_base and not config_base.endswith(".yaml")
                else config_base
            )
            # select_pattern is server config name (e.g., "r1_fp8_dep8_mtp1_1k1k")
            self.select_pattern = "-".join(labels[2:]) if len(labels) > 2 else None

        self.config_dir = os.getenv(
            "TRTLLM_CONFIG_FOLDER", os.path.join(get_llm_root(), self.config_dir)
        )

        # Initialize server configs
        self.server_configs: List = []
        self.server_client_configs: Dict[int, List[ClientConfig]] = {}

    def parse_config_file(self):
        """Parse config file based on runtime."""
        config_file_path = os.path.join(self.config_dir, self.config_file)

        if self.runtime == "aggr_server":
            self._parse_aggr_config_file(config_file_path)
        elif self.runtime == "multi_node_disagg_server":
            self._parse_disagg_config_file(config_file_path, self.config_file)

    def _parse_aggr_config_file(self, config_file_path: str):
        """Parse YAML config file for aggregated server."""
        # Parse selection pattern (server config names)
        if self.select_pattern:
            selected_server_names = parse_select_pattern(self.select_pattern)
        else:
            selected_server_names = None

        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)

        metadata = config.get("metadata", {})
        environment = config.get("environment", {})
        hardware = config.get("hardware", {})
        gpus_per_node = hardware.get("gpus_per_node", 0)

        model_name = metadata.get("model_name", "")
        server_env_var = environment.get("server_env_var", "")
        client_env_var = environment.get("client_env_var", "")

        server_configs = []
        server_client_configs = {}

        for server_idx, server_config_data in enumerate(config["server_configs"]):
            # Check if this server should be included based on selected_server_names
            if (
                selected_server_names is not None
                and server_config_data.get("name") not in selected_server_names
            ):
                continue

            server_config_data["model_name"] = (
                model_name
                if "model_name" not in server_config_data
                else server_config_data["model_name"]
            )
            server_config_data["concurrency"] = -1
            server_config_data["gpus_per_node"] = gpus_per_node

            server_config = ServerConfig(server_config_data, server_env_var)
            server_id = len(server_configs)
            server_configs.append(server_config)

            client_configs = []
            for client_config_data in server_config_data["client_configs"]:
                client_config = ClientConfig(
                    client_config_data,
                    server_config_data["model_name"],
                    env_vars=client_env_var,
                )
                client_configs.append(client_config)

            server_client_configs[server_id] = client_configs

        self.server_configs = server_configs
        self.server_client_configs = server_client_configs

    def _parse_disagg_config_file(self, config_file_path: str, config_file: str):
        """Parse YAML config file for disaggregated server."""
        disagg_serving_type = os.environ.get("DISAGG_SERVING_TYPE", "BENCHMARK")

        # Get config file base name (without extension)
        config_file_base_name = os.path.splitext(config_file)[0]

        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)

        metadata = config.get("metadata", {})
        hardware = config.get("hardware", {})
        benchmark = config.get("benchmark", {})
        environment = config.get("environment", {})
        slurm_config = config.get("slurm", {})
        worker_config = config.get("worker_config", {})

        timeout = slurm_config.get("timeout", DEFAULT_TIMEOUT)
        numa_bind = slurm_config.get("numa_bind", False)
        gpus_per_node = hardware.get("gpus_per_node", 0)
        model_name = metadata.get("model_name", "")
        assert model_name, "model_name is required in metadata section"

        benchmark_mode = benchmark.get("mode", "e2e")
        if "gen_only" in benchmark_mode:
            hardware["num_ctx_servers"] = 0

        worker_env_var = environment.get("worker_env_var", "")
        server_env_var = environment.get("server_env_var", "")
        client_env_var = environment.get("client_env_var", "")

        # Parse concurrency_list - can be string or list
        concurrency_str = benchmark.get("concurrency_list", "1")
        if isinstance(concurrency_str, str):
            concurrency_values = [int(x) for x in concurrency_str.split()]
        elif isinstance(concurrency_str, list):
            concurrency_values = [int(x) for x in concurrency_str]
        else:
            concurrency_values = [int(concurrency_str)]

        # Gen only mode only runs max concurrency
        if "gen_only" in benchmark_mode:
            concurrency_values = [max(concurrency_values)]

        # Create ctx server config
        ctx_server_config_data = {
            "concurrency": max(concurrency_values),
            "name": config_file_base_name,
            "model_name": model_name,
            "gpus_per_node": gpus_per_node,
            "disagg_run_type": "ctx",
            **worker_config.get("ctx", {}),
        }

        # Create gen server config
        gen_server_config_data = {
            "concurrency": max(concurrency_values),
            "name": config_file_base_name,
            "model_name": model_name,
            "gpus_per_node": gpus_per_node,
            "disagg_run_type": "gen",
            **worker_config.get("gen", {}),
        }

        ctx_server_config = ServerConfig(ctx_server_config_data, worker_env_var)
        gen_server_config = ServerConfig(gen_server_config_data, worker_env_var)

        # Create disagg config
        disagg_config = DisaggConfig(
            name=config_file_base_name,
            disagg_serving_type=disagg_serving_type,
            hostname=socket.gethostname(),
            numa_bind=numa_bind,
            timeout=timeout,
            benchmark_mode=benchmark_mode,
            model_name=model_name,
            hardware=hardware,
            server_env_var=server_env_var,
        )

        # server_configs is a list with one element (tuple of ctx, gen, disagg config)
        self.server_configs = [(ctx_server_config, gen_server_config, disagg_config)]

        # Create client configs for each concurrency value
        client_configs = []
        for concurrency in concurrency_values:
            client_config_data = {
                "concurrency": concurrency,
                "iterations": benchmark.get("multi_round", 1),
                "isl": benchmark.get("input_length", 1024),
                "osl": benchmark.get("output_length", 1024),
                "random_range_ratio": benchmark.get("benchmark_ratio", 0.0),
                "backend": "openai",
                "use_chat_template": False,
                "streaming": benchmark.get("streaming", True),
                "dataset_file": benchmark.get("dataset_file", ""),
            }
            client_config = ClientConfig(
                client_config_data,
                model_name,
                env_vars=client_env_var,
            )
            client_configs.append(client_config)

        self.server_client_configs = {0: client_configs}

    def get_commands(self):
        """Get commands based on runtime."""
        self.perf_sanity_output_dir = os.path.join(self._output_dir, self._test_param_labels)
        os.makedirs(self.perf_sanity_output_dir, exist_ok=True)

        if self.runtime == "aggr_server":
            return self._get_aggr_commands(self.perf_sanity_output_dir)
        elif self.runtime == "multi_node_disagg_server":
            return self._get_disagg_commands(self.perf_sanity_output_dir)

    def _get_aggr_commands(self, output_dir: str):
        """Get commands for aggregated server."""
        server_cmds = []
        client_cmds = {}

        for server_idx, client_configs in self.server_client_configs.items():
            server_config = self.server_configs[server_idx]
            server_cmd = server_config.to_cmd(output_dir)

            # Generate extra-llm-api-config.yml
            config_content = server_config.generate_extra_llm_api_config()
            config_filename = f"extra-llm-api-config.aggr.{server_config.name}.yml"
            config_path = os.path.join(output_dir, config_filename)
            with open(config_path, "w") as f:
                f.write(config_content)

            server_cmds.append(server_cmd)
            client_cmds[server_idx] = []

            for client_config in client_configs:
                client_cmd = client_config.to_cmd()
                client_cmds[server_idx].append(client_cmd)

        return AggrTestCmds(
            server_cmds=server_cmds,
            client_cmds=client_cmds,
            timeout=DEFAULT_TIMEOUT,
            output_dir=output_dir,
        )

    def _get_disagg_commands(self, output_dir: str):
        """Get commands for disaggregated server."""
        server_cmds = []
        client_cmds = {}

        for server_idx, (ctx_config, gen_config, disagg_config) in enumerate(self.server_configs):
            numa_bind = disagg_config.numa_bind
            timeout = disagg_config.timeout
            disagg_serving_type = disagg_config.disagg_serving_type

            # Generate ctx server command
            ctx_cmd = ctx_config.to_cmd(output_dir, numa_bind, "CTX")
            if "CTX" in disagg_serving_type:
                config_content = ctx_config.generate_extra_llm_api_config()
                config_path = os.path.join(
                    output_dir, f"extra-llm-api-config.ctx.{ctx_config.name}.yml"
                )
                with open(config_path, "w") as f:
                    f.write(config_content)

            # Generate gen server command
            gen_cmd = gen_config.to_cmd(output_dir, numa_bind, "GEN")
            if "GEN" in disagg_serving_type:
                config_content = gen_config.generate_extra_llm_api_config()
                config_path = os.path.join(
                    output_dir, f"extra-llm-api-config.gen.{gen_config.name}.yml"
                )
                with open(config_path, "w") as f:
                    f.write(config_content)

            # Generate disagg server command
            disagg_cmd = [
                "trtllm-serve",
                "disaggregated",
                "-c",
                f"{output_dir}/server_config.{server_idx}.yaml",
                "-t",
                str(timeout),
                "-r",
                str(timeout),
            ]

            server_cmds.append((ctx_cmd, gen_cmd, disagg_cmd))

            # Add client commands
            client_cmds[server_idx] = []
            for client_config in self.server_client_configs[server_idx]:
                client_cmd = client_config.to_cmd()
                client_cmds[server_idx].append(client_cmd)

        disagg_config = self.server_configs[0][2]
        return DisaggTestCmds(
            server_cmds=server_cmds,
            client_cmds=client_cmds,
            timeout=disagg_config.timeout,
            hostname=disagg_config.hostname,
            disagg_serving_type=disagg_config.disagg_serving_type,
            num_ctx_servers=disagg_config.num_ctx_servers,
            num_gen_servers=disagg_config.num_gen_servers,
            output_dir=output_dir,
        )

    def run_ex(self, commands) -> Dict[int, List[str]]:
        """Run commands and collect outputs."""
        outputs = {}

        for server_idx in range(len(commands.server_cmds)):
            try:
                with io.StringIO() as buf:
                    with contextlib.redirect_stdout(buf):
                        server_outputs = commands.run_cmd(server_idx)
                        for output in server_outputs:
                            print(collect_and_clean_myelin_time(output))

                    # Check for errors in each output
                    for output in server_outputs:
                        self._check_benchmark_output_for_errors(output)

                    print(buf.getvalue())

                outputs[server_idx] = server_outputs

            except Exception as e:
                print_error(f"Test command failed for server {server_idx}. Error: {e}")
                if isinstance(e, subprocess.CalledProcessError):
                    print_error("--- stdout ---")
                    if e.stdout:
                        print_error(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout)
                    print_error("--------------")
                outputs[server_idx] = []

        return outputs

    def _check_benchmark_output_for_errors(self, output: str) -> None:
        """Check whether the benchmark output contains error messages."""
        if not output:
            return

        # Check for non-zero failed requests
        failed_requests_match = re.search(r"Failed requests:\s+(\d+)", output)
        if failed_requests_match:
            failed_count = int(failed_requests_match.group(1))
            if failed_count > 0:
                error_msg = f"Benchmark output contains {failed_count} failed requests."
                raise Exception(error_msg)

        # Check for explicit failure markers
        if "!FAILED REQUESTS!" in output or "!CHECK LOG FOR ERRORS!" in output:
            error_msg = "Benchmark output contains failure markers."
            raise Exception(error_msg)

    def get_perf_result(self, outputs: Dict[int, List[str]]):
        """Parse performance results from outputs."""

        def parse_metrics_from_output(output: str) -> Optional[Dict[str, float]]:
            """Parse all metrics from a single output string."""
            metrics = {}
            for line in output.split("\n"):
                for metric_type, regex in PERF_METRIC_LOG_QUERIES.items():
                    if metric_type in metrics:
                        continue
                    match = regex.search(line)
                    if match:
                        metrics[metric_type] = float(match.group(1))
                        break
            return metrics

        self._perf_results = {}
        for server_idx, client_configs in self.server_client_configs.items():
            self._perf_results[server_idx] = []
            server_outputs = outputs.get(server_idx, [])
            for output in server_outputs:
                metrics = parse_metrics_from_output(output)
                self._perf_results[server_idx].append(metrics)

    def check_test_failure(self):
        """Check if any server failed based on perf results."""
        error_msg = ""
        for server_idx, client_configs in self.server_client_configs.items():
            server_perf_results = self._perf_results.get(server_idx, [])
            if len(server_perf_results) != len(client_configs):
                error_msg += (
                    f"Server {server_idx}'s perf results number: {len(server_perf_results)} "
                    f"is not equal to client number: {len(client_configs)}. "
                )
            for client_idx, metrics in enumerate(server_perf_results):
                if len(metrics) != len(PERF_METRIC_LOG_QUERIES):
                    error_msg += (
                        f"Some metrics in Server {server_idx} Client {client_idx} are missing. "
                        f"The broken metrics is {metrics}. "
                    )

        if error_msg:
            raise Exception(error_msg)

        print_info("All servers passed")

    def upload_test_results_to_database(self):
        """Upload test results and baseline to database."""

        def add_prefix(key: str, prefix_name: str) -> str:
            type_prefix = key[0:2]
            rest = key[2:]
            return f"{type_prefix}{prefix_name}_{rest}"

        def add_list_prefix(config_list: List, prefix_name: str) -> List:
            return [add_prefix(key, prefix_name) for key in config_list]

        def add_dict_prefix(config_dict: dict, prefix_name: str) -> dict:
            return {add_prefix(key, prefix_name): value for key, value in config_dict.items()}

        match_keys = []
        is_scenario_mode = False

        if self.runtime == "aggr_server":
            job_config = get_job_info()
            is_post_merge = job_config["b_is_post_merge"]
            new_data_dict = {}
            cmd_idx = 0
            for server_idx, client_configs in self.server_client_configs.items():
                server_config = self.server_configs[server_idx]
                server_config_dict = server_config.to_db_data()
                server_perf_results = self._perf_results.get(server_idx, [])
                # Skip if server failed
                if len(server_perf_results) != len(client_configs):
                    cmd_idx += len(client_configs)
                    continue

                for client_idx, client_config in enumerate(client_configs):
                    client_config_dict = client_config.to_db_data()

                    # Skip if metrics missing
                    if server_perf_results[client_idx] is None:
                        print_info(
                            f"Skipped posting command {cmd_idx}'s test results since some metrics are missing."
                        )
                        cmd_idx += 1
                        continue

                    new_data = {
                        "s_gpu_type": self.gpu_type,
                        "s_runtime": "multi_node_aggr_server"
                        if server_config.gpus != server_config.gpus_per_node
                        else "aggr_server",
                    }
                    new_data.update(job_config)
                    new_data.update(server_config_dict)
                    new_data.update(client_config_dict)
                    # Add test_case_name for convenient filtering on OpenSearch
                    new_data["s_test_case_name"] = f"{server_config.name}-{client_config.name}"

                    for metric_name in PERF_METRIC_LOG_QUERIES:
                        new_data[f"d_{metric_name}"] = server_perf_results[client_idx][metric_name]

                    add_id(new_data)
                    new_data_dict[cmd_idx] = new_data
                    cmd_idx += 1

                    if not match_keys:
                        if server_config.match_mode == "scenario":
                            match_keys = SCENARIO_MATCH_FIELDS.copy()
                            is_scenario_mode = True
                        else:
                            match_keys.extend(["s_gpu_type", "s_runtime"])
                            match_keys.extend(server_config.to_match_keys())
                            match_keys.extend(client_config.to_match_keys())

        elif self.runtime == "multi_node_disagg_server":
            # Only BENCHMARK node uploads
            if self.server_configs[0][2].disagg_serving_type != "BENCHMARK":
                return

            job_config = get_job_info()
            is_post_merge = job_config["b_is_post_merge"]
            new_data_dict = {}
            cmd_idx = 0

            for server_idx, (ctx_config, gen_config, disagg_config) in enumerate(
                self.server_configs
            ):
                client_configs = self.server_client_configs[server_idx]
                server_perf_results = self._perf_results.get(server_idx, [])
                # Skip if server failed
                if len(server_perf_results) != len(client_configs):
                    cmd_idx += len(client_configs)
                    continue

                for client_idx, client_config in enumerate(client_configs):
                    # Skip if metrics missing
                    if server_perf_results[client_idx] is None:
                        print_info(
                            f"Skipped posting command {cmd_idx}'s test results since some metrics are missing."
                        )
                        cmd_idx += 1
                        continue

                    # Get server configs with prefixed keys
                    ctx_server_config_dict = add_dict_prefix(ctx_config.to_db_data(), "ctx")
                    gen_server_config_dict = add_dict_prefix(gen_config.to_db_data(), "gen")
                    client_config_dict = client_config.to_db_data()

                    num_ctx_servers = disagg_config.num_ctx_servers
                    num_gen_servers = disagg_config.num_gen_servers

                    new_data = {
                        "s_gpu_type": self.gpu_type,
                        "s_runtime": "multi_node_disagg_server",
                        "s_benchmark_mode": disagg_config.benchmark_mode,
                        "s_server_env_var": disagg_config.server_env_var,
                        "l_num_ctx_servers": num_ctx_servers,
                        "l_num_gen_servers": num_gen_servers,
                    }
                    new_data.update(job_config)
                    if num_ctx_servers > 0:
                        new_data.update(ctx_server_config_dict)
                    if num_gen_servers > 0:
                        new_data.update(gen_server_config_dict)
                    new_data.update(client_config_dict)
                    # Add test_case_name for convenient filtering on OpenSearch
                    new_data["s_test_case_name"] = f"{disagg_config.name}-{client_config.name}"

                    for metric_name in PERF_METRIC_LOG_QUERIES:
                        new_data[f"d_{metric_name}"] = server_perf_results[client_idx][metric_name]

                    add_id(new_data)
                    new_data_dict[cmd_idx] = new_data
                    cmd_idx += 1

                    if not match_keys:
                        match_keys.extend(
                            [
                                "s_gpu_type",
                                "s_runtime",
                                "s_benchmark_mode",
                                "l_num_ctx_servers",
                                "l_num_gen_servers",
                            ]
                        )
                        if num_ctx_servers > 0:
                            match_keys.extend(add_list_prefix(ctx_config.to_match_keys(), "ctx"))
                        if num_gen_servers > 0:
                            match_keys.extend(add_list_prefix(gen_config.to_match_keys(), "gen"))
                        match_keys.extend(client_config.to_match_keys())
        else:
            return

        if not new_data_dict:
            print_info("No data to upload to database.")
            return

        # Find common values across all data entries to narrow down query scope
        common_values_dict = get_common_values(new_data_dict, match_keys)

        # Get history data for each cmd_idx
        history_baseline_dict, history_data_dict = get_history_data(
            new_data_dict, match_keys, common_values_dict
        )

        # Update regression info in new_data_dict
        prepare_regressive_test_cases(history_baseline_dict, new_data_dict)

        if is_post_merge:
            # Prepare new baseline data for post-merge
            new_baseline_data_dict = prepare_baseline_data(
                history_baseline_dict, history_data_dict, new_data_dict
            )
        else:
            # Pre-merge does not need to upload baseline data
            new_baseline_data_dict = None

        if self.upload_to_db:
            # Upload the new perf data and baseline data to database
            post_new_perf_data(new_baseline_data_dict, new_data_dict)

        check_perf_regression(
            new_data_dict,
            fail_on_regression=is_scenario_mode,
            output_dir=self.perf_sanity_output_dir,
        )


# Perf sanity test case parameters
AGG_TEST_TYPES = ["aggr_upload", "aggr"]
DISAGG_TEST_TYPES = ["disagg_upload", "disagg"]


def get_server_config_names(yaml_path: str) -> List[str]:
    """Read a YAML file and return the list of server_config names."""
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        if data and "server_configs" in data:
            return [config.get("name", "") for config in data["server_configs"]]
    except Exception:
        pass
    return []


def get_yaml_files_with_server_names(directory: str) -> Dict[str, List[str]]:
    """Scan directory for YAML files and return dict of {basename: [server_config_names]}."""
    yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
    result = {}
    for yaml_path in sorted(yaml_files):
        basename = os.path.splitext(os.path.basename(yaml_path))[0]
        server_names = get_server_config_names(yaml_path)
        result[basename] = server_names
    return result


def get_aggr_test_cases() -> List[str]:
    """Generate aggr test cases based on actual server_config names in YAML files."""
    llm_root = get_llm_root()
    aggr_config_dir = os.path.join(llm_root, AGGR_CONFIG_FOLDER)
    yaml_server_names = get_yaml_files_with_server_names(aggr_config_dir)

    test_cases = []
    for config_yml, server_names in yaml_server_names.items():
        for test_type in AGG_TEST_TYPES:
            # Case without select_pattern (runs all server configs)
            test_cases.append(f"{test_type}-{config_yml}")

            # Cases with single server config name
            for server_name in server_names:
                test_cases.append(f"{test_type}-{config_yml}-{server_name}")

    return test_cases


def get_disagg_test_cases() -> List[str]:
    """Generate disagg test cases."""
    llm_root = get_llm_root()
    disagg_config_dir = os.path.join(llm_root, DISAGG_CONFIG_FOLDER)
    yaml_files = glob.glob(os.path.join(disagg_config_dir, "*.yaml"))
    basenames = sorted([os.path.splitext(os.path.basename(f))[0] for f in yaml_files])

    test_cases = []
    for config_yml in basenames:
        for test_type in DISAGG_TEST_TYPES:
            test_cases.append(f"{test_type}-{config_yml}")

    return test_cases


# Hardcoded multi-test test cases from test db.
MULTI_TEST_TEST_CASES = []

# Generate all test case combinations
# For aggr: {test_type}-{config_yml}, {test_type}-{config_yml}-{server_config_name}
# For disagg: {test_type}-{config_yml}
PERF_SANITY_TEST_CASES = get_aggr_test_cases() + get_disagg_test_cases() + MULTI_TEST_TEST_CASES


@pytest.mark.parametrize("perf_sanity_test_case", PERF_SANITY_TEST_CASES)
def test_e2e(output_dir, perf_sanity_test_case):
    # Create config and parse test case name
    config = PerfSanityTestConfig(perf_sanity_test_case, output_dir)

    # Parse config file to get server_configs and server_client_configs
    config.parse_config_file()

    # Get commands
    commands = config.get_commands()

    # Run commands and collect outputs
    outputs = config.run_ex(commands)

    # For disagg mode, only BENCHMARK node parses results and uploads
    if config.runtime == "multi_node_disagg_server":
        disagg_config = config.server_configs[0][2]
        if disagg_config.disagg_serving_type != "BENCHMARK":
            print_info(
                f"Disagg serving type is {disagg_config.disagg_serving_type}, skipping perf result parsing and upload."
            )
            return

    # Parse performance results
    config.get_perf_result(outputs)

    # Check for test failures
    config.check_test_failure()

    # Upload results to database
    config.upload_test_results_to_database()
