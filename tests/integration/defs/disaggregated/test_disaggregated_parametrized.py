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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

import pytest
import yaml
from defs.common import get_disagg_server_url_from_cfg
from defs.conftest import skip_arm, skip_no_hopper
from defs.trt_test_alternative import check_call

from tensorrt_llm._utils import get_free_port


# Utility functions for disaggregated tests
def cleanup_output_files():
    """Clean up output files from previous runs."""
    for file in ["output.json", "output_streaming.json"]:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


def validate_timing_metrics(perf_metrics_item, request_context=""):
    """Helper function to validate timing metrics relationships.

    Args:
        perf_metrics_item: A single performance metrics item from the /perf_metrics endpoint
        request_context: String context for error messages (e.g., "request 1", "streaming")
    """
    # Validate basic structure
    required_keys = [
        "ctx_server",
        "gen_server",
        "ctx_perf_metrics",
        "gen_perf_metrics",
        "disagg_server_arrival_time",
        "disagg_server_first_token_time",
    ]
    for key in required_keys:
        assert key in perf_metrics_item, f"Missing key: {key} in {request_context}"

    assert (
        perf_metrics_item["ctx_perf_metrics"]["ctx_request_id"]
        == perf_metrics_item["gen_perf_metrics"]["ctx_request_id"]
    )

    # Extract timing metrics
    ctx_metrics = perf_metrics_item["ctx_perf_metrics"]["perf_metrics"]["timing_metrics"]
    gen_metrics = perf_metrics_item["gen_perf_metrics"]["perf_metrics"]["timing_metrics"]
    disagg_arrival = perf_metrics_item["disagg_server_arrival_time"]
    disagg_first_token = perf_metrics_item["disagg_server_first_token_time"]

    # Validate disaggregated server timing metrics
    assert disagg_arrival is not None, f"disagg_server_arrival_time is None in {request_context}"
    assert disagg_first_token is not None, (
        f"disagg_server_first_token_time is None in {request_context}"
    )
    assert isinstance(disagg_arrival, (int, float)), (
        f"disagg_server_arrival_time is not numeric in {request_context}"
    )
    assert isinstance(disagg_first_token, (int, float)), (
        f"disagg_server_first_token_time is not numeric in {request_context}"
    )
    assert disagg_arrival > 0, f"disagg_server_arrival_time is not positive in {request_context}"
    assert disagg_first_token > 0, (
        f"disagg_server_first_token_time is not positive in {request_context}"
    )
    assert disagg_arrival <= disagg_first_token, (
        f"disagg_server_arrival_time > disagg_server_first_token_time in {request_context}"
    )

    # Validate server-level timing metrics for context server
    ctx_server_arrival = ctx_metrics.get("server_arrival_time")
    ctx_server_first_token = ctx_metrics.get("server_first_token_time")
    assert ctx_server_arrival is not None, f"ctx server_arrival_time is None in {request_context}"
    assert ctx_server_first_token is not None, (
        f"ctx server_first_token_time is None in {request_context}"
    )
    assert isinstance(ctx_server_arrival, (int, float)), (
        f"ctx server_arrival_time is not numeric in {request_context}"
    )
    assert isinstance(ctx_server_first_token, (int, float)), (
        f"ctx server_first_token_time is not numeric in {request_context}"
    )
    assert ctx_server_arrival <= ctx_server_first_token, (
        f"ctx server_arrival_time > server_first_token_time in {request_context}"
    )
    assert ctx_metrics["last_token_time"] - ctx_server_first_token < 1e-3

    # Validate server-level timing metrics for generation server
    gen_server_arrival = gen_metrics.get("server_arrival_time")
    gen_server_first_token = gen_metrics.get("server_first_token_time")
    assert gen_server_arrival is not None, f"gen server_arrival_time is None in {request_context}"
    assert gen_server_first_token is not None, (
        f"gen server_first_token_time is None in {request_context}"
    )
    assert isinstance(gen_server_arrival, (int, float)), (
        f"gen server_arrival_time is not numeric in {request_context}"
    )
    assert isinstance(gen_server_first_token, (int, float)), (
        f"gen server_first_token_time is not numeric in {request_context}"
    )
    assert gen_server_arrival <= gen_server_first_token, (
        f"gen server_arrival_time > server_first_token_time in {request_context}"
    )

    # Network Time Protocol can ensure ms-level accuracy in LAN
    ntp_tolerance = 1e-3

    # Validate timing relationships between different levels
    # Disaggregated server should receive request before individual servers
    assert disagg_arrival - ntp_tolerance <= ctx_server_arrival, (
        f"disagg_arrival > ctx_server_arrival in {request_context}"
    )
    assert disagg_arrival - ntp_tolerance <= gen_server_arrival, (
        f"disagg_arrival > gen_server_arrival in {request_context}"
    )

    # Context should complete before generation starts
    assert ctx_server_first_token - ntp_tolerance <= gen_server_arrival, (
        f"ctx_server_first_token > gen_server_arrival in {request_context}"
    )

    # Validate internal timing consistency
    ctx_arrival_time = ctx_metrics["arrival_time"]
    ctx_first_token_time = ctx_metrics["first_token_time"]
    gen_arrival_time = gen_metrics["arrival_time"]
    gen_first_token_time = gen_metrics["first_token_time"]

    assert ctx_arrival_time <= ctx_first_token_time, (
        f"ctx arrival_time > first_token_time in {request_context}"
    )
    assert gen_arrival_time <= gen_first_token_time, (
        f"gen arrival_time > first_token_time in {request_context}"
    )

    # Test KV cache transfer timing (if present)
    if "kv_cache_transfer_start" in gen_metrics and "kv_cache_transfer_end" in gen_metrics:
        kv_start = gen_metrics["kv_cache_transfer_start"]
        kv_end = gen_metrics["kv_cache_transfer_end"]
        assert gen_metrics["kv_cache_size"] > 0
        assert kv_start <= kv_end, (
            f"kv_cache_transfer_start > kv_cache_transfer_end in {request_context}"
        )
        assert gen_arrival_time <= kv_start, (
            f"gen_arrival_time > kv_cache_transfer_start in {request_context}"
        )
        assert kv_end <= gen_metrics["first_scheduled_time"], (
            f"kv_cache_transfer_end > first_scheduled_time in {request_context}"
        )

    return True


def run_client_tests(
    example_dir,
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
):
    """Run client tests against the disaggregated server.

    Args:
        example_dir: Path to the examples directory
        config_file: Path to the configuration file
        test_desc: Test description/name
        num_iters: Number of iterations to run
        env: Environment variables
        server_start_timeout: Timeout for server startup
        prompt_file: Name of the prompt file to use
        extra_endpoints_test: Optional callback for extra endpoint tests
        server_url: URL of the disaggregated server
        workers_proc: Worker process(es)
        server_proc: Server process
        use_ray: Whether Ray orchestrator is being used
    """
    client_dir = f"{example_dir}/clients"
    for _ in range(num_iters):
        client_cmd = [
            "python3",
            f"{client_dir}/disagg_client.py",
            "-c",
            f"{config_file}",
            "-p",
            f"{client_dir}/{prompt_file}",
            "--ignore-eos",
            "--server-start-timeout",
            str(server_start_timeout),
        ]
        if prompt_file == "long_prompts.json":
            # Use max_tokens 4 for long prompts to reduce test time
            client_cmd.extend(["--max-tokens", "4"])

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
        streaming_client_cmd = client_cmd + ["--streaming", "-o", "output_streaming.json"]
        check_call(streaming_client_cmd, env=env, poll_procs=poll_procs)

        # Run the chat completion endpoint test only for TinyLlama
        if test_desc == "overlap" or test_desc == "trtllm_sampler":
            chat_client_cmd = client_cmd + ["-e", "chat", "-o", "output_chat.json"]
            check_call(chat_client_cmd, env=env, poll_procs=poll_procs)

            streaming_chat_client_cmd = chat_client_cmd + [
                "--streaming",
                "-o",
                "output_streaming_chat.json",
            ]
            check_call(streaming_chat_client_cmd, env=env, poll_procs=poll_procs)

        # Skip output verification for long prompts test
        if prompt_file == "long_prompts.json":
            continue

        if extra_endpoints_test is not None:
            extra_endpoints_test(server_url)

        # Verify outputs
        not_expected_strings = ["Berlin Berlin"]

        output_files = ["output.json", "output_streaming.json"]
        if test_desc == "overlap" or test_desc == "trtllm_sampler":
            # Disable streaming chat completion for overlap test
            # due to bug
            output_files.extend(["output_chat.json"])

        if test_desc.startswith("gen_only"):
            continue

        for output_file in output_files:
            with open(output_file, "r") as f:
                content = f.read()
                if "ds_v3_lite" in test_desc or output_file == "output_chat.json":
                    expected_strings = ["Berlin", ["Asyncio is a", "Asyncio module in"]]
                else:
                    expected_strings = [
                        "The capital of Germany is Berlin",
                        "Asyncio is a Python library",
                    ]
                for expected_string in expected_strings:
                    if isinstance(expected_string, list):
                        # At least one of the strings in the list should be found in the content
                        assert any(string in content for string in expected_string), (
                            f"None of the strings in {expected_string} found in {output_file}"
                        )
                    else:
                        assert expected_string in content, (
                            f"Expected string '{expected_string}' not found in {output_file}"
                        )
                for not_expected_string in not_expected_strings:
                    assert not_expected_string not in content, (
                        f"Unexpected string '{not_expected_string}' found in {output_file}"
                    )


@dataclass
class DisaggregatedTestConfig:
    """Complete configuration for a disaggregated test."""

    test_name: str
    model_root: str

    # Global config
    global_config: dict = field(default_factory=dict)
    # Ctx config
    ctx_config: dict = field(default_factory=dict)
    # Gen config
    gen_config: dict = field(default_factory=dict)

    # Test specific settings
    skip_device_count: Optional[int] = None
    skip_no_hopper: bool = False
    skip_arm_arch: bool = False
    env_vars: Optional[Dict[str, str]] = None
    prompt_file: str = "prompts.json"
    num_iters: int = 5
    extra_validation: Optional[str] = (
        None  # Special validation type: 'perf_metrics', 'kv_cache_time'
    )

    @staticmethod
    def _deep_merge_dicts(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary to start from
            override: Dictionary with values to override/add

        Returns:
            New dictionary with merged values
        """
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DisaggregatedTestConfig._deep_merge_dicts(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    @classmethod
    def from_base(
        cls,
        base: "DisaggregatedTestConfig",
        test_name: str,
        model_root: Optional[str] = None,
        global_config: Optional[dict] = None,
        ctx_config: Optional[dict] = None,
        gen_config: Optional[dict] = None,
        skip_device_count: Optional[int] = None,
        skip_no_hopper: Optional[bool] = None,
        skip_arm_arch: Optional[bool] = None,
        env_vars: Optional[Dict[str, str]] = None,
        prompt_file: Optional[str] = None,
        num_iters: Optional[int] = None,
        extra_validation: Optional[str] = None,
    ) -> "DisaggregatedTestConfig":
        """Create a new config based on an existing one with selective overrides.

        Args:
            base: Base configuration to inherit from
            test_name: Name for the new test (required)
            model_root: Override model root (if None, uses base.model_root)
            global_config: Dictionary to merge into base global_config
            ctx_config: Dictionary to merge into base ctx_config
            gen_config: Dictionary to merge into base gen_config
            skip_device_count: Override skip_device_count (if None, uses base value)
            skip_no_hopper: Override skip_no_hopper (if None, uses base value)
            skip_arm_arch: Override skip_arm_arch (if None, uses base value)
            env_vars: Override or merge with base env_vars
            prompt_file: Override prompt_file (if None, uses base value)
            num_iters: Override num_iters (if None, uses base value)
            extra_validation: Override extra_validation (if None, uses base value)

        Returns:
            New DisaggregatedTestConfig instance
        """
        # Deep copy base configs
        new_global_config = deepcopy(base.global_config)
        new_ctx_config = deepcopy(base.ctx_config)
        new_gen_config = deepcopy(base.gen_config)
        new_env_vars = deepcopy(base.env_vars) if base.env_vars else None

        # Merge provided overrides

        # Remove any parameters from global_config that are already specified in ctx_config or gen_config
        for key in list(new_global_config.keys()):
            if (ctx_config is not None and key in ctx_config) or (
                gen_config is not None and key in gen_config
            ):
                new_global_config.pop(key, None)

        if global_config:
            new_global_config = cls._deep_merge_dicts(new_global_config, global_config)
        if ctx_config:
            new_ctx_config = cls._deep_merge_dicts(new_ctx_config, ctx_config)
        if gen_config:
            new_gen_config = cls._deep_merge_dicts(new_gen_config, gen_config)
        if env_vars:
            if new_env_vars:
                new_env_vars = {**new_env_vars, **env_vars}
            else:
                new_env_vars = env_vars.copy()

        return cls(
            test_name=test_name,
            model_root=model_root if model_root is not None else base.model_root,
            global_config=new_global_config,
            ctx_config=new_ctx_config,
            gen_config=new_gen_config,
            skip_device_count=skip_device_count
            if skip_device_count is not None
            else base.skip_device_count,
            skip_no_hopper=skip_no_hopper if skip_no_hopper is not None else base.skip_no_hopper,
            skip_arm_arch=skip_arm_arch if skip_arm_arch is not None else base.skip_arm_arch,
            env_vars=new_env_vars,
            prompt_file=prompt_file if prompt_file is not None else base.prompt_file,
            num_iters=num_iters if num_iters is not None else base.num_iters,
            extra_validation=extra_validation
            if extra_validation is not None
            else base.extra_validation,
        )

    def get_num_ranks(self) -> int:
        """Calculate total number of ranks needed."""
        ctx_tp = self.ctx_config.get("tensor_parallel_size", 1)
        ctx_pp = self.ctx_config.get("pipeline_parallel_size", 1)
        ctx_num_instances = self.ctx_config.get("num_instances", 1)

        gen_tp = self.gen_config.get("tensor_parallel_size", 1)
        gen_pp = self.gen_config.get("pipeline_parallel_size", 1)
        gen_num_instances = self.gen_config.get("num_instances", 1)

        ctx_ranks = ctx_tp * ctx_pp * ctx_num_instances
        gen_ranks = gen_tp * gen_pp * gen_num_instances
        return ctx_ranks + gen_ranks

    def generate_yaml_config(self, temp_dir: str) -> str:
        """Generate a yaml config file from the parameters."""
        config = self.global_config.copy()
        config["model"] = self.model_root
        config["hostname"] = "localhost"
        config["port"] = get_free_port()

        # Add default cache_transceiver_config if not present
        if "cache_transceiver_config" not in config:
            config["cache_transceiver_config"] = {"backend": "DEFAULT"}

        if "backend" in config and config["backend"] == "trt":
            if "disable_overlap_scheduler" in config:
                del config["disable_overlap_scheduler"]
            if "cuda_graph_config" in config:
                del config["cuda_graph_config"]

        # Build context servers config
        context_servers = self.ctx_config.copy()

        ctx_num_instances = self.ctx_config.get("num_instances", 1)
        context_servers["num_instances"] = ctx_num_instances

        ctx_urls = []
        for i in range(ctx_num_instances):
            ctx_urls.append(f"localhost:{get_free_port()}")
        context_servers["urls"] = ctx_urls
        config["context_servers"] = context_servers

        # Build generation servers config
        gen_servers = self.gen_config.copy()

        gen_num_instances = self.gen_config.get("num_instances", 1)
        gen_servers["num_instances"] = gen_num_instances

        gen_urls = []
        for i in range(gen_num_instances):
            gen_urls.append(f"localhost:{get_free_port()}")
        gen_servers["urls"] = gen_urls

        # Special handling for gen-only mode
        if ctx_num_instances == 0 and "backend" in config and config["backend"] == "pytorch":
            gen_servers["print_iter_log"] = True

        config["generation_servers"] = gen_servers

        # Write to temporary file
        config_path = os.path.join(temp_dir, f"{self.test_name}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path


# Define all test configurations
#
# Usage: You can create test configs from scratch or use from_base() to inherit from existing configs:
#
# Example 1 - Create base config:
#   base_config = DisaggregatedTestConfig(
#       test_name="base",
#       model_root="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#       global_config={"backend": "pytorch"}
#   )
#
# Example 2 - Create variation with different backend:
#   trt_config = DisaggregatedTestConfig.from_base(
#       base_config,
#       test_name="trt_variant",
#       global_config={"backend": "trt"}  # This merges/overrides into base
#   )
#
# Example 3 - Create variation with additional nested config:
#   perf_config = DisaggregatedTestConfig.from_base(
#       base_config,
#       test_name="perf_variant",
#       ctx_config={"return_perf_metrics": True},  # Merges with base ctx_config
#       extra_validation="perf_metrics"
#   )

# Store some base configs for reuse
_tiny_llama_cfg = DisaggregatedTestConfig(
    test_name="2_ranks",
    model_root="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    global_config={
        "backend": "pytorch",
        "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_partial_reuse": False},
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
    },
)

_tiny_llama_multi_gpus_cfg = DisaggregatedTestConfig.from_base(
    _tiny_llama_cfg,
    test_name="multi_gpus",
    global_config={
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.2,
            "enable_partial_reuse": False,
            "enable_block_reuse": False,
        },
    },
    ctx_config={
        "max_batch_size": 1,
        "max_num_tokens": 3000,
        "max_seq_len": 4096,
    },
    gen_config={
        "max_batch_size": 256,
        "max_num_tokens": 4096,
        "max_seq_len": 4096,
    },
    skip_device_count=4,
)

_ds_v3_lite_tp1_cfg = DisaggregatedTestConfig(
    test_name="ds_v3_lite_tp1",
    model_root="DeepSeek-V3-Lite/fp8",
    global_config={
        "backend": "pytorch",
        "free_gpu_memory_fraction": 0.1,
    },
    ctx_config={
        "disable_overlap_scheduler": True,
    },
    gen_config={
        "disable_overlap_scheduler": False,
    },
    skip_no_hopper=True,
)

_ds_v3_lite_4_gpus_cfg = DisaggregatedTestConfig(
    test_name="ds_v3_lite",
    model_root="DeepSeek-V3-Lite/fp8",
    global_config={
        "backend": "pytorch",
        "free_gpu_memory_fraction": 0.7,
    },
    ctx_config={
        "tensor_parallel_size": 2,
        "disable_overlap_scheduler": True,
    },
    gen_config={
        "tensor_parallel_size": 2,
        "disable_overlap_scheduler": False,
    },
    skip_device_count=4,
    skip_no_hopper=True,
)

_ds_v3_lite_4_gpus_helix_cfg = DisaggregatedTestConfig(
    test_name="ds_v3_lite_helix",
    model_root="DeepSeek-V3-Lite/bf16",
    global_config={
        "backend": "pytorch",
        "free_gpu_memory_fraction": 0.25,
        "disable_overlap_scheduler": True,
    },
    ctx_config={
        "tensor_parallel_size": 2,
        "enable_chunked_prefill": False,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "enable_partial_reuse": False,
            "tokens_per_block": 32,
        },
    },
    gen_config={
        "enable_chunked_prefill": False,
        "context_parallel_size": 2,
        "cp_config": {"cp_type": "HELIX", "tokens_per_block": 32},
        "kv_cache_config": {
            "enable_block_reuse": False,
            "enable_partial_reuse": False,
            "tokens_per_block": 32,
        },
    },
    skip_device_count=4,
)

TEST_CONFIGS = [
    # TinyLlama tests - basic
    _tiny_llama_cfg,
    # Performance metrics variant - extends base with metrics config
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="perf_metrics",
        global_config={"perf_metrics_max_requests": 1000},
        ctx_config={"return_perf_metrics": True, "perf_metrics_max_requests": 1000},
        gen_config={"return_perf_metrics": True, "perf_metrics_max_requests": 1000},
        extra_validation="perf_metrics",
    ),
    # KV cache time variant - same as perf_metrics but different validation
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="kv_cache_time_output",
        global_config={"perf_metrics_max_requests": 1000},
        ctx_config={"return_perf_metrics": True, "perf_metrics_max_requests": 1000},
        gen_config={"return_perf_metrics": True, "perf_metrics_max_requests": 1000},
        extra_validation="kv_cache_time",
    ),
    # Create TRT variant from base - only need to override backend
    DisaggregatedTestConfig(
        test_name="trt_backend",
        model_root="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        global_config={
            "backend": "trt",
            "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_partial_reuse": False},
        },
    ),
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="diff_max_tokens",
        prompt_file="long_prompts.json",
        ctx_config={"max_num_tokens": 512, "max_batch_size": 64},
        gen_config={"max_num_tokens": 256, "max_batch_size": 32},
    ),
    # TinyLlama - CUDA graph
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="cuda_graph",
        ctx_config={"cuda_graph_config": {"batch_sizes": [1, 3000]}},
        gen_config={
            "cuda_graph_config": {"enable_padding": True, "batch_sizes": [1, 4, 8, 16, 24, 32]},
            "max_batch_size": 256,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
        },
    ),
    # TinyLlama - overlap
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="overlap",
        ctx_config={
            "max_num_tokens": 3000,
            "max_seq_len": 4096,
            "disable_overlap_scheduler": True,
        },
        gen_config={
            "max_batch_size": 256,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
            "disable_overlap_scheduler": False,
        },
    ),
    # TinyLlama - mixed
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="mixed",
        gen_config={"num_instances": 2},
    ),
    # TinyLlama - trtllm sampler
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="trtllm_sampler",
        ctx_config={
            "max_batch_size": 1,
            "max_num_tokens": 3000,
            "max_seq_len": 4096,
            "sampler_type": "TRTLLMSampler",
            "disable_overlap_scheduler": True,
        },
        gen_config={
            "max_batch_size": 256,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
            "sampler_type": "TRTLLMSampler",
            "disable_overlap_scheduler": False,
        },
    ),
    # TinyLlama - load balance
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="load_balance",
        global_config={
            "kv_cache_config": {"free_gpu_memory_fraction": 0.15, "enable_partial_reuse": False},
        },
        ctx_config={
            "num_instances": 2,
            "router": {"type": "load_balancing", "use_tokens": True},
            "max_num_tokens": 3000,
            "max_seq_len": 4096,
            "disable_overlap_scheduler": True,
        },
        gen_config={
            "num_instances": 2,
            "router": {"type": "load_balancing", "use_tokens": False},
            "max_batch_size": 256,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
            "disable_overlap_scheduler": False,
        },
    ),
    # TinyLlama - cache aware balance
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="cache_aware_balance",
        global_config={
            "free_gpu_memory_fraction": 0.1,
            "enable_autotuner": False,
            "kv_cache_config": {
                "enable_block_reuse": True,
                "enable_partial_reuse": False,
                "event_buffer_max_size": 1024,
                "free_gpu_memory_fraction": 0.1,
            },
        },
        ctx_config={
            "num_instances": 2,
            "router": {"type": "kv_cache_aware"},
            "max_batch_size": 16,
            "max_num_tokens": 3000,
            "max_seq_len": 4096,
        },
        gen_config={
            "num_instances": 2,
            "router": {"type": "kv_cache_aware"},
            "max_batch_size": 256,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
        },
    ),
    # TinyLlama - conditional
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="conditional",
        model_root="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        global_config={
            "free_gpu_memory_fraction": 0.15,
            "conditional_disagg_config": {"max_local_prefill_length": 100},
            "enable_autotuner": False,
            "kv_cache_config": {
                "enable_block_reuse": True,
                "enable_partial_reuse": True,
                "event_buffer_max_size": 1024,
                "free_gpu_memory_fraction": 0.15,
            },
        },
        gen_config={
            "router": {"type": "kv_cache_aware"},
        },
    ),
    # TinyLlama - ngram
    DisaggregatedTestConfig.from_base(
        _tiny_llama_cfg,
        test_name="ngram",
        global_config={
            "free_gpu_memory_fraction": 0.1,
        },
        gen_config={
            "speculative_config": {
                "decoding_type": "NGram",
                "max_draft_len": 4,
                "max_matching_ngram_size": 4,
                "is_keep_all": True,
                "is_use_oldest": True,
                "is_public_pool": True,
            },
        },
    ),
    DisaggregatedTestConfig(
        test_name="gen_only_bs1",
        model_root="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        env_vars={"TRTLLM_DISAGG_BENCHMARK_GEN_ONLY": "1"},
        global_config={
            "backend": "pytorch",
            "cuda_graph_config": None,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.2,
                "enable_partial_reuse": False,
            },
            "enable_attention_dp": True,
        },
        ctx_config={
            "tensor_parallel_size": 2,
            "max_batch_size": 1,
            "max_num_tokens": 3000,
            "max_seq_len": 4096,
            "disable_overlap_scheduler": True,
        },
        gen_config={
            "tensor_parallel_size": 2,
            "max_batch_size": 1,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
            "disable_overlap_scheduler": False,
        },
        skip_device_count=4,
    ),
    # TinyLlama - TP variations
    DisaggregatedTestConfig.from_base(
        _tiny_llama_multi_gpus_cfg,
        test_name="1ctx_tp2pp1_2gen_tp1pp1",
        ctx_config={
            "tensor_parallel_size": 2,
        },
        gen_config={
            "num_instances": 2,
        },
        skip_device_count=4,
    ),
    DisaggregatedTestConfig.from_base(
        _tiny_llama_multi_gpus_cfg,
        test_name="1ctx_tp2pp1_2gen_tp1pp1_trt",
        global_config={
            "backend": "trt",
        },
        ctx_config={
            "tensor_parallel_size": 2,
        },
        gen_config={
            "num_instances": 2,
        },
        skip_device_count=4,
    ),
    DisaggregatedTestConfig.from_base(
        _tiny_llama_multi_gpus_cfg,
        test_name="1ctx_tp1pp2_1gen_tp1pp2",
        ctx_config={
            "pipeline_parallel_size": 2,
        },
        gen_config={
            "pipeline_parallel_size": 2,
        },
        skip_device_count=4,
    ),
    DisaggregatedTestConfig.from_base(
        _tiny_llama_multi_gpus_cfg,
        test_name="1ctx_tp2pp1_1gen_tp1pp2",
        ctx_config={
            "tensor_parallel_size": 2,
        },
        gen_config={
            "pipeline_parallel_size": 2,
        },
        skip_device_count=4,
    ),
    DisaggregatedTestConfig.from_base(
        _tiny_llama_multi_gpus_cfg,
        test_name="1ctx_tp1pp2_1gen_tp2pp1",
        ctx_config={
            "pipeline_parallel_size": 2,
        },
        gen_config={
            "tensor_parallel_size": 2,
        },
        skip_device_count=4,
    ),
    DisaggregatedTestConfig.from_base(
        _tiny_llama_multi_gpus_cfg,
        test_name="1ctx_tp2pp2_1gen_tp2pp2",
        ctx_config={
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
        },
        gen_config={
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
        },
        skip_device_count=8,
    ),
    DisaggregatedTestConfig.from_base(
        _tiny_llama_multi_gpus_cfg,
        test_name="1ctx_tp1pp4_1gen_tp1pp4",
        ctx_config={
            "pipeline_parallel_size": 4,
        },
        gen_config={
            "pipeline_parallel_size": 4,
        },
        skip_device_count=8,
    ),
    DisaggregatedTestConfig.from_base(
        _tiny_llama_multi_gpus_cfg,
        test_name="1ctx_tp1pp4_1gen_tp4pp1",
        ctx_config={
            "pipeline_parallel_size": 4,
        },
        gen_config={
            "tensor_parallel_size": 4,
        },
        skip_device_count=8,
    ),
    # DeepSeek V3 Lite tests
    # TP1 tests
    _ds_v3_lite_tp1_cfg,
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_tp1_cfg,
        test_name="ds_v3_lite_tp1_mtp",
        global_config={
            "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
        },
        ctx_config={
            "enable_attention_dp": True,
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_tp1_cfg,
        test_name="ds_v3_lite_tp1_mtp_adp_overlap",
        global_config={
            "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
            "enable_attention_dp": True,
        },
        ctx_config={
            "disable_overlap_scheduler": True,
        },
        gen_config={
            "disable_overlap_scheduler": False,
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_tp1_cfg,
        test_name="ds_v3_lite_tp1_mtp2",
        global_config={
            "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 2},
        },
        ctx_config={
            "enable_attention_dp": True,
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_tp1_cfg,
        test_name="ds_v3_lite_tp1_cache_aware_balance",
        global_config={"enable_autotuner": False, "kv_cache_config": {"enable_block_reuse": True}},
        ctx_config={
            "num_instances": 2,
            "router": {"type": "kv_cache_aware"},
        },
        gen_config={
            "num_instances": 2,
            "router": {"type": "kv_cache_aware"},
        },
        skip_no_hopper=True,
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_tp1_cfg,
        test_name="ds_v3_lite_tp1_conditional",
        global_config={
            "enable_autotuner": False,
            "conditional_disagg_config": {"max_local_prefill_length": 100},
            "kv_cache_config": {
                "event_buffer_max_size": 1024,
                "free_gpu_memory_fraction": 0.15,
            },
        },
        ctx_config={
            "router": {"type": "kv_cache_aware"},
        },
        gen_config={
            "router": {"type": "kv_cache_aware"},
        },
    ),
    # 4 ranks different backends
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_mpi",
        global_config={
            "cache_transceiver_config": {
                "backend": "MPI",
            },
        },
        skip_arm_arch=True,
        env_vars={
            "TRTLLM_USE_MPI_KVCACHE": "1",
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_ucx",
        global_config={
            "cache_transceiver_config": {
                "backend": "UCX",
            },
        },
        skip_arm_arch=True,
        env_vars={"TRTLLM_USE_UCX_KVCACHE": "1", "UCX_TLS": "^ib"},
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_nixl",
        global_config={
            "cache_transceiver_config": {
                "backend": "NIXL",
            },
        },
        skip_arm_arch=True,
        env_vars={"TRTLLM_USE_NIXL_KVCACHE": "1", "UCX_TLS": "^ib"},
    ),
    # 4 ranks
    _ds_v3_lite_4_gpus_cfg,
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_adp",
        global_config={
            "enable_attention_dp": True,
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_adp_overlap",
        global_config={
            "enable_attention_dp": True,
        },
        ctx_config={
            "disable_overlap_scheduler": True,
        },
        gen_config={
            "disable_overlap_scheduler": False,
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_adp_overlap_cuda_graph",
        global_config={
            "enable_attention_dp": True,
        },
        ctx_config={
            "disable_overlap_scheduler": True,
        },
        gen_config={
            "cuda_graph_config": {"enable_padding": False},
            "disable_overlap_scheduler": False,
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_overlap_cuda_graph",
        gen_config={
            "cuda_graph_config": {"enable_padding": False},
            "disable_overlap_scheduler": False,
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_adp_mtp",
        global_config={
            "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
            "enable_attention_dp": True,
        },
    ),
    DisaggregatedTestConfig.from_base(
        _ds_v3_lite_4_gpus_cfg,
        test_name="ds_v3_lite_mtp",
        global_config={
            "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
        },
    ),
    _ds_v3_lite_4_gpus_helix_cfg,
]


def get_test_id(config: DisaggregatedTestConfig) -> str:
    """Generate test ID from config."""
    return config.test_name


def apply_skip_marks(config: DisaggregatedTestConfig):
    """Apply skip markers based on configuration."""
    markers = []

    if config.skip_device_count is not None:
        markers.append(pytest.mark.skip_less_device(config.skip_device_count))

    return markers


def pytest_generate_tests(metafunc):
    """Generate test cases dynamically based on TEST_CONFIGS."""
    if "config" in metafunc.fixturenames:
        configs = TEST_CONFIGS
        [get_test_id(c) for c in configs]

        # Apply marks
        marked_configs = []
        for config in configs:
            marks = apply_skip_marks(config)
            if marks:
                marked_configs.append(pytest.param(config, marks=marks, id=get_test_id(config)))
            else:
                marked_configs.append(pytest.param(config, id=get_test_id(config)))

        metafunc.parametrize("config", marked_configs)


# TODO: add test for disaggregated server prometheus metrics
def fetch_prometheus_metrics(server_url: str):
    import requests

    response = requests.get(f"{server_url}/prometheus/metrics", timeout=10)
    assert response.status_code == 200
    return response.text


def run_disaggregated_test_parametrized(
    example_dir, config: DisaggregatedTestConfig, env=None, cwd=None, extra_endpoints_test=None
):
    """Run disaggregated test with parametrized configuration.

    Args:
        example_dir: Path to the examples directory
        config: DisaggregatedTestConfig with all test parameters
        env: Environment variables
        cwd: Working directory for test execution
        extra_endpoints_test: Optional callback for additional endpoint validation
    """
    import subprocess

    from defs.trt_test_alternative import popen

    from tensorrt_llm._utils import mpi_disabled
    from tensorrt_llm.logger import logger

    cleanup_output_files()
    run_env = env.copy()
    run_env["UCX_TLS"] = "^ib"

    # Generate config file
    config_path = config.generate_yaml_config(cwd)

    # Print generated config for debugging
    print(f"\n{'=' * 80}")
    print(f"Generated YAML config for test: {config.test_name}")
    print(f"Config file: {config_path}")
    print(f"{'=' * 80}")
    with open(config_path, "r") as f:
        print(f.read())
    print(f"{'=' * 80}\n")

    try:
        num_ranks = config.get_num_ranks()
        use_ray = mpi_disabled()

        if not use_ray:
            workers_cmd = [
                "mpirun",
                "--allow-run-as-root",
                "--oversubscribe",
                "-n",
                str(num_ranks),
                "trtllm-serve",
                "disaggregated_mpi_worker",
                "-c",
                config_path,
            ]
        else:
            pytest.skip(
                "https://nvbugs/5584607 Ray orchestrator is not supported with NIXL(DEFAULT) cache transceiver backend."
            )
            # Check backend compatibility
            backend = config.global_config.get("backend", "pytorch")
            if backend != "pytorch":
                pytest.skip("Ray orchestrator is only supported with pytorch backend.")

            # Generate extra config files for Ray workers
            def get_extra_llm_config(server_config, suffix):
                extra_config = {
                    "orchestrator_type": "ray",
                }
                for key, value in server_config.items():
                    if key not in ["num_instances", "urls"]:
                        extra_config[key] = value
                return extra_config

            extra_config_files = []
            workers_cmds = []

            # Create config for context servers
            ctx_num_instances = config.ctx_config.get("num_instances", 1)
            for i in range(ctx_num_instances):
                extra_llm_config = get_extra_llm_config(config.ctx_config, f"ctx_{i}")
                extra_file = os.path.join(cwd, f"{config.test_name}_ctx_{i}.yaml")
                with open(extra_file, "w") as f:
                    yaml.dump(extra_llm_config, f, default_flow_style=False)
                extra_config_files.append(extra_file)
                workers_cmds.append(
                    [
                        "trtllm-serve",
                        "disaggregated_ray_worker",
                        "-c",
                        extra_file,
                        "--model",
                        config.model_root,
                    ]
                )

            # Create config for generation servers
            gen_num_instances = config.gen_config.get("num_instances", 1)
            for i in range(gen_num_instances):
                extra_llm_config = get_extra_llm_config(config.gen_config, f"gen_{i}")
                extra_file = os.path.join(cwd, f"{config.test_name}_gen_{i}.yaml")
                with open(extra_file, "w") as f:
                    yaml.dump(extra_llm_config, f, default_flow_style=False)
                extra_config_files.append(extra_file)
                workers_cmds.append(
                    [
                        "trtllm-serve",
                        "disaggregated_ray_worker",
                        "-c",
                        extra_file,
                        "--model",
                        config.model_root,
                    ]
                )

        server_start_timeout = 1200
        server_cmd = [
            "trtllm-serve",
            "disaggregated",
            "--server_start_timeout",
            str(server_start_timeout),
            "-c",
            config_path,
        ]
        server_host, server_port = get_disagg_server_url_from_cfg(config_path)
        server_url = f"http://{server_host}:{server_port}"

        try:
            if not use_ray:
                with (
                    open("output_workers.log", "w") as output_workers,
                    popen(
                        workers_cmd,
                        stdout=output_workers,
                        stderr=subprocess.STDOUT,
                        env=run_env,
                        cwd=cwd,
                    ) as workers_proc,
                    open("output_disagg.log", "w") as output_disagg,
                    popen(
                        server_cmd,
                        stdout=output_disagg,
                        stderr=subprocess.STDOUT,
                        env=run_env,
                        cwd=cwd,
                    ) as server_proc,
                ):
                    run_client_tests(
                        example_dir,
                        config_path,
                        config.test_name,
                        config.num_iters,
                        env,
                        server_start_timeout,
                        config.prompt_file,
                        extra_endpoints_test,
                        server_url,
                        workers_proc,
                        server_proc,
                        use_ray=False,
                    )
            else:
                # Ray orchestrator path
                workers_proc = []
                for worker_cmd in workers_cmds:
                    workers_proc.append(
                        popen(
                            worker_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            env=run_env,
                            cwd=cwd,
                        )
                    )

                # Enter all worker contexts
                for proc_cm in workers_proc:
                    proc_cm.__enter__()

                with (
                    open("output_disagg.log", "w") as output_disagg,
                    popen(
                        server_cmd,
                        stdout=output_disagg,
                        stderr=subprocess.STDOUT,
                        env=run_env,
                        cwd=cwd,
                    ) as server_proc,
                ):
                    run_client_tests(
                        example_dir,
                        config_path,
                        config.test_name,
                        config.num_iters,
                        env,
                        server_start_timeout,
                        config.prompt_file,
                        extra_endpoints_test,
                        server_url,
                        workers_proc,
                        server_proc,
                        use_ray=True,
                    )
        except Exception:
            logger.error("-------- Workers output --------")
            if not use_ray and os.path.exists("output_workers.log"):
                with open("output_workers.log", "r") as f:
                    logger.error(f.read())

            logger.error("-------- Disagg server output --------")
            if os.path.exists("output_disagg.log"):
                with open("output_disagg.log", "r") as f:
                    logger.error(f.read())
            raise
        finally:
            if use_ray:
                subprocess.run(["ray", "stop", "--force"], check=False)
                for extra_file in extra_config_files:
                    if os.path.exists(extra_file):
                        os.remove(extra_file)
            else:
                if "server_proc" in locals() and "workers_proc" in locals():
                    server_proc.terminate()
                    workers_proc.terminate()
                    server_proc.wait()
                    workers_proc.wait()
    finally:
        # Cleanup generated config
        if os.path.exists(config_path):
            os.remove(config_path)


@pytest.fixture(scope="function")
def model_root_fixture(config, llm_venv, request):
    """Fixture that provides the correct model root based on config."""
    from defs.conftest import llm_models_root

    models_root = llm_models_root()

    print("Running model root fixture for config: ", config.test_name)
    if config.model_root == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        src_root = os.path.join(models_root, "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0")
    else:
        src_root = os.path.join(models_root, config.model_root)

    dst_root = f"{llm_venv.get_working_directory()}/{config.model_root}"

    # Create symlink
    if not os.path.exists(dst_root) and not os.path.islink(dst_root):
        os.makedirs(os.path.dirname(dst_root), exist_ok=True)
        os.symlink(src_root, dst_root, target_is_directory=True)

    return src_root


def test_disagg(
    config: DisaggregatedTestConfig,
    disaggregated_test_root,
    disaggregated_example_root,
    llm_venv,
    model_root_fixture,
):
    """Parametrized test for all disaggregated configurations."""
    # Apply skip conditions that can't be marks
    if config.skip_no_hopper:
        skip_no_hopper()

    if config.skip_arm_arch:
        skip_arm()

    # Setup environment
    env = llm_venv._new_env.copy()

    # Handle special validation cases
    extra_endpoints_test = None
    kv_cache_output_path = None

    if config.extra_validation == "perf_metrics":
        # Test /perf_metrics endpoint
        def extra_endpoints_test(server_url: str):
            import json
            import urllib.request

            with urllib.request.urlopen(f"{server_url}/perf_metrics", timeout=10) as resp:
                assert resp.status == 200
                perf_metrics = json.load(resp)
            assert len(perf_metrics) > 0
            item = perf_metrics[0]

            # Use helper function to validate all timing metrics comprehensively
            validate_timing_metrics(item, "perf_metrics test")

    elif config.extra_validation == "kv_cache_time":
        # Test KV cache time output files
        kv_cache_output_path = os.path.join(llm_venv.get_working_directory(), "cache_time")
        env["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"] = kv_cache_output_path

    # Apply test-specific environment variables
    if config.env_vars:
        env.update(config.env_vars)

    # Run the test
    run_disaggregated_test_parametrized(
        disaggregated_example_root,
        config,
        env=env,
        cwd=llm_venv.get_working_directory(),
        extra_endpoints_test=extra_endpoints_test,
    )

    # Post-test validation for kv_cache_time
    if config.extra_validation == "kv_cache_time":
        assert os.path.isdir(kv_cache_output_path)
        send_file = os.path.join(kv_cache_output_path, "rank_0_send.csv")
        recv_file = os.path.join(kv_cache_output_path, "rank_1_recv.csv")
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
            sample = lines[1].split(",")
            assert len(sample) >= 9
        with open(recv_file, "r") as f:
            lines = f.readlines()
            assert len(lines) > 1
            matched = False
            for line in lines:
                sample_recv = line.split(",")
                if sample_recv[0] == sample[0]:
                    matched = True
                    break
            assert matched
