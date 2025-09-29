#!/usr/bin/env python3
import argparse
import ast
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import requests
import yaml

max_retries = 3

# Model path mapping
llm_models_root = os.environ.get('LLM_MODELS_ROOT',
                                 '/home/scratch.trt_llm_data/llm-models')
MODEL_PATHS = {
    "70B-FP4": f"{llm_models_root}/llama-3.3-models/Llama-3.3-70B-Instruct-FP4",
    "70B-FP8": f"{llm_models_root}/llama-3.3-models/Llama-3.3-70B-Instruct-FP8",
    "Scout-FP4":
    f"{llm_models_root}/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4",
    "Scout-FP8":
    f"{llm_models_root}/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8",
    "R1-FP8": f"{llm_models_root}/DeepSeek-R1/DeepSeek-R1",
    "R1-FP4": f"{llm_models_root}/DeepSeek-R1/DeepSeek-R1-0528-FP4"
}

HF_MODEL_PATHS = {
    "70B-FP4": "meta-llama/Meta-Llama-3.3-70B-Instruct-FP4",
    "70B-FP8": "meta-llama/Meta-Llama-3.3-70B-Instruct-FP8",
    "Scout-FP4": "meta-llama/Llama-4-Scout-17B-16E-Instruct-FP4",
    "Scout-FP8": "meta-llama/Llama-4-Scout-17B-16E-Instruct-FP8",
    "R1-FP8": "deepseek-ai/DeepSeek-R1",
    "R1-FP4": "deepseek-ai/DeepSeek-R1-0528-FP4"
}


def str_to_bool(value: str) -> bool:
    return ast.literal_eval(value)


# {metric_name: (is_optional, type)}
SERVER_CONFIG_METRICS = {
    "model_name": (False, str),
    "tp": (False, int),
    "ep": (False, int),
    "pp": (True, int),
    "isl": (False, int),
    "osl": (False, int),
    "max_num_tokens": (False, int),
    "enable_chunked_prefill": (True, str_to_bool),
    "disable_overlap_scheduler": (True, str_to_bool),
    "attention_backend": (False, str),
    "moe_backend": (True, str),
    "moe_max_num_tokens": (True, str),
    "enable_attention_dp": (True, str_to_bool),
    "attention_dp_balance": (True, str_to_bool),
    "batching_wait_iters": (True, int),
    "timeout_iters": (True, int),
    "kv_cache_dtype": (True, str),
    "enable_block_reuse": (True, str_to_bool),
    "free_gpu_memory_fraction": (True, float),
    "max_batch_size": (False, int),
    "enable_padding": (True, str_to_bool),
}

CLIENT_CONFIG_METRICS = {
    "concurrency": (False, int),
    "iterations": (False, int),
    "random_range_ratio": (True, float),
}


class ServerConfig:

    def __init__(
        self,
        model_name: str,
        tp: int,
        ep: int,
        isl: int,
        osl: int,
        max_num_tokens: int,
        attention_backend: str,
        max_batch_size: int,
        pp: int = 1,
        enable_chunked_prefill: bool = False,
        disable_overlap_scheduler: bool = False,
        moe_backend: str = "",
        moe_max_num_tokens: str = "",
        enable_attention_dp: bool = False,
        attention_dp_balance: bool = False,
        batching_wait_iters: int = 10,
        timeout_iters: int = 50,
        kv_cache_dtype: str = "fp8",
        enable_block_reuse: bool = False,
        free_gpu_memory_fraction: float = 0.8,
        enable_padding: bool = True,
    ):
        self.model_name = model_name
        self.tp = tp
        self.ep = ep
        self.pp = pp
        self.isl = isl
        self.osl = osl
        self.max_num_tokens = max_num_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.disable_overlap_scheduler = disable_overlap_scheduler
        self.attention_backend = attention_backend
        self.moe_backend = moe_backend
        self.moe_max_num_tokens = moe_max_num_tokens
        self.enable_attention_dp = enable_attention_dp
        self.attention_dp_balance = attention_dp_balance
        self.batching_wait_iters = batching_wait_iters
        self.timeout_iters = timeout_iters
        self.kv_cache_dtype = kv_cache_dtype
        self.enable_block_reuse = enable_block_reuse
        self.free_gpu_memory_fraction = free_gpu_memory_fraction
        self.max_batch_size = max_batch_size
        self.enable_padding = enable_padding

    def to_str(self) -> str:
        return f"{self.model_name}.tp{self.tp}.ep{self.ep}.pp{self.pp}.attn{self.attention_backend}.moe{self.moe_backend}.adp{self.enable_attention_dp}.batching_wait_iters{self.batching_wait_iters}.timeout_iters{self.timeout_iters}.gpu_frac{self.free_gpu_memory_fraction}.bs{self.max_batch_size}.isl{self.isl}.osl{self.osl}.max_tokens{self.max_num_tokens}.moe_max_tokens{self.moe_max_num_tokens}.kv{self.kv_cache_dtype}.reuse{self.enable_block_reuse}.chunk_prefill{self.enable_chunked_prefill}.overlap{self.disable_overlap_scheduler}.pad{self.enable_padding}"

    def to_log_content(self) -> List[str]:
        log_content = []
        log_content += f"  model_name: {self.model_name}\n"
        log_content += f"  tp: {self.tp}\n"
        log_content += f"  ep: {self.ep}\n"
        log_content += f"  pp: {self.pp}\n"
        log_content += f"  isl: {self.isl}\n"
        log_content += f"  osl: {self.osl}\n"
        log_content += f"  max_num_tokens: {self.max_num_tokens}\n"
        log_content += f"  enable_chunked_prefill: {self.enable_chunked_prefill}\n"
        log_content += f"  disable_overlap_scheduler: {self.disable_overlap_scheduler}\n"
        log_content += f"  attention_backend: {self.attention_backend}\n"
        log_content += f"  moe_backend: {self.moe_backend}\n"
        log_content += f"  moe_max_num_tokens: {self.moe_max_num_tokens}\n"
        log_content += f"  enable_attention_dp: {self.enable_attention_dp}\n"
        log_content += f"  attention_dp_balance: {self.attention_dp_balance}\n"
        log_content += f"  batching_wait_iters: {self.batching_wait_iters}\n"
        log_content += f"  timeout_iters: {self.timeout_iters}\n"
        log_content += f"  kv_cache_dtype: {self.kv_cache_dtype}\n"
        log_content += f"  enable_block_reuse: {self.enable_block_reuse}\n"
        log_content += f"  free_gpu_memory_fraction: {self.free_gpu_memory_fraction}\n"
        log_content += f"  max_batch_size: {self.max_batch_size}\n"
        log_content += f"  enable_padding: {self.enable_padding}\n"
        return log_content

    def generate_extra_llm_api_config(self) -> str:
        """Generate extra-llm-api-config.yml content"""
        enable_chunked_prefill = self.max_num_tokens < self.isl
        config_lines = [
            "print_iter_log: true",
            f"enable_attention_dp: {str(self.enable_attention_dp).lower()}",
            f"disable_overlap_scheduler: {str(self.disable_overlap_scheduler).lower()}",
            "stream_interval: 10",
            f"attn_backend: {self.attention_backend}",
            f"enable_chunked_prefill: {str(enable_chunked_prefill).lower()}",
            "cuda_graph_config:",
            f"  enable_padding: {str(self.enable_padding).lower()}",
            f"  max_batch_size: {self.max_batch_size}",
            "kv_cache_config:",
            f"  dtype: {self.kv_cache_dtype}",
            f"  free_gpu_memory_fraction: {self.free_gpu_memory_fraction}",
            f"  enable_block_reuse: {str(self.enable_block_reuse).lower()}",
        ]

        # Add moe_config if moe_backend is specified
        if self.moe_backend:
            config_lines.append("moe_config:")
            config_lines.append(f"  backend: {self.moe_backend}")
            if self.moe_max_num_tokens:
                config_lines.append(
                    f"  max_num_tokens: {self.moe_max_num_tokens}")

        if self.attention_dp_balance:
            config_lines.append("attention_dp_balance:")
            config_lines.append("  enable_balance: true")
            config_lines.append(
                f"  batching_wait_iters: {self.batching_wait_iters}")
            config_lines.append(f"  timeout_iters: {self.timeout_iters}")

        return "\n".join(config_lines)


class ClientConfig:

    def __init__(self,
                 concurrency: int,
                 iterations: int,
                 random_range_ratio: float = 0.0):
        self.concurrency = concurrency
        self.iterations = iterations
        self.random_range_ratio = random_range_ratio

    def to_str(self) -> str:
        return f"concurrency{self.concurrency}.iter{self.iterations}.random_ratio{self.random_range_ratio}"

    def to_log_content(self) -> str:
        log_content = []
        log_content += f"  concurrency: {self.concurrency}\n"
        log_content += f"  iterations: {self.iterations}\n"
        log_content += f"  random_range_ratio: {self.random_range_ratio}\n"
        return log_content


def parse_config_file(config_file: str) -> None:
    """Parse YAML configuration file and create ServerConfig and ClientConfig objects"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    server_configs = []
    client_configs = {}

    for server_config_data in config['server_configs']:
        server_config_id = server_config_data['id']

        # Create ServerConfig object
        server_config = ServerConfig(
            model_name=server_config_data['model_name'],
            tp=server_config_data['tp'],
            ep=server_config_data['ep'],
            pp=server_config_data.get('pp', 1),
            attention_backend=server_config_data.get('attention_backend',
                                                     'TRTLLM'),
            moe_backend=server_config_data.get('moe_backend'),
            moe_max_num_tokens=server_config_data.get('moe_max_num_tokens', ''),
            enable_attention_dp=server_config_data.get('enable_attention_dp',
                                                       False),
            attention_dp_balance=server_config_data.get('attention_dp_balance',
                                                        False),
            batching_wait_iters=server_config_data.get('batching_wait_iters',
                                                       10),
            timeout_iters=server_config_data.get('timeout_iters', 50),
            enable_chunked_prefill=server_config_data.get(
                'enable_chunked_prefill', False),
            isl=server_config_data.get('isl', 2048),
            osl=server_config_data.get('osl', 512),
            max_num_tokens=server_config_data.get('max_num_tokens', 2048),
            disable_overlap_scheduler=server_config_data.get(
                'disable_overlap_scheduler', False),
            kv_cache_dtype=server_config_data.get('kv_cache_dtype', 'fp8'),
            enable_block_reuse=server_config_data.get('enable_block_reuse',
                                                      False),
            free_gpu_memory_fraction=server_config_data.get(
                'free_gpu_memory_fraction', 0.8),
            max_batch_size=server_config_data.get('max_batch_size', 256),
            enable_padding=server_config_data.get('enable_padding', True))

        # Store server config with its ID
        server_configs.append((server_config_id, server_config))

        # Create ClientConfig objects
        server_client_configs = []
        for client_config_data in server_config_data['client_configs']:
            client_config = ClientConfig(
                concurrency=client_config_data['concurrency'],
                iterations=client_config_data['iterations'],
                random_range_ratio=client_config_data.get(
                    'random_range_ratio', 0.0))
            server_client_configs.append(client_config)

        client_configs[server_config_id] = server_client_configs

    return server_configs, client_configs


class BenchmarkRunner:

    def __init__(self,
                 output_folder: str,
                 config_file: str,
                 skip_pattern: str = None,
                 select_pattern: str = None,
                 timeout: int = 3600):
        self.output_folder = Path(output_folder)
        self.config_file = Path(config_file)
        self.timeout = timeout

        # Treat empty or "default" values as None (default behavior)
        self.skip_pattern = None if not skip_pattern or skip_pattern.lower(
        ) == "default" else skip_pattern
        self.select_pattern = None if not select_pattern or select_pattern.lower(
        ) == "default" else select_pattern

        self.skip_server_configs: Set[int] = set()
        self.skip_client_configs: Dict[int, Set[int]] = {}
        self.select_server_configs: Set[int] = set()
        self.select_client_configs: Dict[int, Set[int]] = {}

        if self.skip_pattern:
            self.parse_skip_pattern(self.skip_pattern)

        if self.select_pattern:
            self.parse_select_pattern(self.select_pattern)

        # Execution plan: {server_config_id: [client_config_indices]}
        self.execution_plan: Dict[int, List[int]] = {}

        # Store server and client configs
        self.server_configs: List[Tuple[int, ServerConfig]] = []
        self.client_configs: Dict[int, List[ClientConfig]] = {}

        # Set environment variables
        os.environ['TQDM_MININTERVAL'] = '1000'
        os.environ['PRINT_ITER_LOG'] = 'false'

        # Capture system information
        self.node_name = self.get_node_name()
        self.gpu_info = self.get_gpu_info()

        # Change to output directory
        os.chdir(self.output_folder)

        # Track test case information for reproduction script generation
        self.test_case_infos: List[Dict[str, Any]] = []

    def get_node_name(self) -> str:
        """Get the current node name"""
        try:
            result = subprocess.run("hostname",
                                    shell=True,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def get_gpu_info(self) -> str:
        """Get GPU information from nvidia-smi"""
        try:
            result = subprocess.run("nvidia-smi",
                                    shell=True,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"nvidia-smi failed with error code {e.returncode}\nError output: {e.stderr}"
        except FileNotFoundError:
            return "nvidia-smi not found"

    def parse_skip_pattern(self, skip_pattern: str) -> None:
        """Parse skip pattern like '2,4-1' to determine what to skip"""
        if not skip_pattern:
            return

        parts = skip_pattern.split(',')
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue

            if '-' in part:
                # Format: "server_config_id-client_config_index" (1-based)
                try:
                    server_config_str, client_config_str = part.split('-')
                    server_config_id = int(server_config_str)
                    client_config_index = int(
                        client_config_str) - 1  # Convert to 0-based

                    if server_config_id not in self.skip_client_configs:
                        self.skip_client_configs[server_config_id] = set()
                    self.skip_client_configs[server_config_id].add(
                        client_config_index)
                except ValueError:
                    raise ValueError(
                        f"Invalid skip pattern '{part}'. Expected format: 'server_config_id-client_config_index' (e.g., '2-1')"
                    )
            else:
                # Format: "server_config_id" - skip entire server config
                try:
                    server_config_id = int(part)
                    self.skip_server_configs.add(server_config_id)
                except ValueError:
                    raise ValueError(
                        f"Invalid server config ID '{part}' in skip pattern. Must be a valid integer."
                    )

        print(f"Skipping server configs: {sorted(self.skip_server_configs)}")
        print(f"Skipping client configs: {self.skip_client_configs}")

    def parse_select_pattern(self, select_pattern: str) -> None:
        """Parse select pattern like '1,3,5' or '1-1,2-3' to determine which server configs/client configs to run"""
        if not select_pattern:
            return

        self.select_client_configs: Dict[int, Set[int]] = {}

        parts = select_pattern.split(',')
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue

            if '-' in part:
                # Format: "server_config_id-client_config_index" (1-based)
                try:
                    server_config_str, client_config_str = part.split('-')
                    server_config_id = int(server_config_str)
                    client_config_index = int(
                        client_config_str) - 1  # Convert to 0-based

                    if server_config_id not in self.select_client_configs:
                        self.select_client_configs[server_config_id] = set()
                    self.select_client_configs[server_config_id].add(
                        client_config_index)
                except ValueError:
                    raise ValueError(
                        f"Invalid select pattern '{part}'. Expected format: 'server_config_id-client_config_index' (e.g., '2-1')"
                    )
            else:
                # Format: "server_config_id" - select entire server config
                try:
                    server_config_id = int(part)
                    self.select_server_configs.add(server_config_id)
                except ValueError:
                    raise ValueError(
                        f"Invalid server config ID '{part}' in select pattern. Must be a valid integer."
                    )

    def build_execution_plan(self) -> None:
        """Build execution plan by analyzing config file, skip_pattern, and select_pattern"""
        self.execution_plan.clear()

        # Step 1: Initialize execution plan based on select_pattern
        if not self.select_pattern:
            # If select_pattern is empty or default, include all server configs with all client configs
            for server_config_id, server_config in self.server_configs:
                all_client_configs = list(
                    range(len(self.client_configs[server_config_id])))
                self.execution_plan[server_config_id] = all_client_configs
        else:
            # If select_pattern is specified, only include selected server configs and client configs
            for server_config_id, server_config in self.server_configs:
                # Check if this server config is selected
                if server_config_id in self.select_server_configs:
                    # Server config is selected - include all client configs
                    all_client_configs = list(
                        range(len(self.client_configs[server_config_id])))
                    self.execution_plan[server_config_id] = all_client_configs
                elif server_config_id in self.select_client_configs:
                    # Specific client configs are selected for this server config
                    selected_client_configs = list(
                        self.select_client_configs[server_config_id])
                    # Validate that selected client configs exist in config
                    max_client_config_index = len(
                        self.client_configs[server_config_id]) - 1
                    valid_client_configs = [
                        c for c in selected_client_configs
                        if 0 <= c <= max_client_config_index
                    ]
                    if valid_client_configs:
                        self.execution_plan[
                            server_config_id] = valid_client_configs

        # Step 2: Apply skip_pattern to remove server configs and client configs
        # Remove entire server configs that are in skip_server_configs
        for server_config_id in self.skip_server_configs:
            if server_config_id in self.execution_plan:
                del self.execution_plan[server_config_id]

        # Remove specific client configs that are in skip_client_configs
        for server_config_id, skip_client_config_indices in self.skip_client_configs.items(
        ):
            if server_config_id in self.execution_plan:
                # Remove skipped client configs from the list
                remaining_client_configs = [
                    c for c in self.execution_plan[server_config_id]
                    if c not in skip_client_config_indices
                ]
                if remaining_client_configs:
                    self.execution_plan[
                        server_config_id] = remaining_client_configs
                else:
                    # If no client configs remain, remove the entire server config
                    del self.execution_plan[server_config_id]

        # Step 3: Clean up - remove server configs with empty client config lists
        # (This should not happen with the above logic, but just to be safe)
        server_configs_to_remove = []
        for server_config_id, client_configs in self.execution_plan.items():
            if not client_configs:
                server_configs_to_remove.append(server_config_id)

        for server_config_id in server_configs_to_remove:
            del self.execution_plan[server_config_id]

    def initialize_test_case_infos(self) -> None:
        """Initialize test case information for all server configs in execution plan with 'Success' status"""
        self.test_case_infos.clear()

        for server_config_id, server_config in self.server_configs:
            # Only initialize server configs that are in the execution plan
            if server_config_id in self.execution_plan:
                test_case_info = {
                    'server_config': server_config,
                    'server_config_id': server_config_id,
                    'failure_reason': 'Success',
                    'node_name': self.node_name,
                    'gpu_info': self.gpu_info
                }
                self.test_case_infos.append(test_case_info)

        print(
            f"Initialized {len(self.test_case_infos)} server config infos (only those in execution plan)"
        )

    def print_execution_plan(self) -> None:
        """Print which server configs and client configs will be executed"""
        print("\n" + "=" * 80)
        print("EXECUTION PLAN")
        print("=" * 80)

        total_server_configs = 0
        total_client_configs = 0

        for server_config_id, server_config in self.server_configs:
            # Check if this server config is in execution plan
            if server_config_id not in self.execution_plan:
                print(
                    f"Server Config {server_config_id}: {server_config.model_name} - SKIPPED"
                )
                continue

            total_server_configs += 1
            print(
                f"\nServer Config {server_config_id}: {server_config.model_name}"
            )
            print(
                f"  Config: TP={server_config.tp}, EP={server_config.ep}, PP={server_config.pp}, "
                f"attention_backend={server_config.attention_backend}, moe_backend={server_config.moe_backend}"
            )

            # Get client configs from execution plan
            client_configs_to_run = []
            for client_config_index in self.execution_plan[server_config_id]:
                client_config = self.client_configs[server_config_id][
                    client_config_index]
                client_configs_to_run.append((
                    client_config_index + 1, client_config.concurrency,
                    client_config.iterations,
                    client_config.random_range_ratio))  # +1 for 1-based display
                total_client_configs += 1

            print(
                f"  Client configs to run ({len(client_configs_to_run)}/{len(self.client_configs[server_config_id])}):"
            )
            for client_config_num, concurrency, iterations, random_range_ratio in client_configs_to_run:
                print(
                    f"    {client_config_num}. Concurrency={concurrency}, Iterations={iterations}, RandomRangeRatio={random_range_ratio}"
                )

        print("\n" + "=" * 80)
        print(
            f"SUMMARY: {total_server_configs} server configs, {total_client_configs} client configs will be executed"
        )
        print("=" * 80 + "\n")

    def wait_for_server(self, server_process: subprocess.Popen,
                        server_log_filename: str) -> bool:
        """Wait for server to be ready"""
        print("Waiting for trtllm-serve to be ready...")

        start_time = time.time()
        while server_process.poll() is None:
            # Check if server is still running
            # try:
            #     os.kill(server_process.pid, 0)  # Check if process exists
            # except OSError:
            #     print("Error: Server process has died")
            #     return False

            # Try to connect to server
            try:
                response = requests.get("http://localhost:8000/v1/models",
                                        timeout=5)
                if response.status_code == 200:
                    print(
                        f"Server is ready! HTTP status: {response.status_code}")
                    return True
            except requests.RequestException:
                pass

            time.sleep(60)
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                print(
                    f"Failed to setup server due to timeout after {elapsed_time:.0f} seconds (>{self.timeout} seconds)"
                )
                return False
            else:
                print(
                    f"Waiting for trtllm-serve to be ready... (elapsed time: {elapsed_time:.0f} / {self.timeout} seconds)"
                )

            # Check server log for runtime errors
            if self.check_for_runtime_error(server_log_filename):
                print(
                    f"Failed to setup server due to RuntimeError detected in server log: {server_log_filename}"
                )
                return False

        print(
            f"Error: Server did not become ready after {elapsed_time:.0f} seconds (>{self.timeout} seconds)"
        )
        return False

    def check_for_runtime_error(self, log_file_path: str) -> bool:
        """Check if RuntimeError exists in log file"""
        try:
            error_keywords = [
                "RuntimeError", "runtime error", "CUDA error",
                "CUDA out of memory", "illegal memory access",
                "terminate called", "ClientPayloadError"
            ]
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    content = f.read()
                    if any(keyword in content for keyword in error_keywords):
                        return True
        except Exception as e:
            print(f"Warning: Could not read log file {log_file_path}: {e}")
        return False

    def run_benchmark(self, server_config: ServerConfig,
                      client_config: ClientConfig, model_path: str,
                      server_log_filename: str) -> bool:
        """Run a single benchmark with monitoring. Returns True if successful, False if should skip test case"""
        num_prompts = client_config.concurrency * client_config.iterations

        print(
            f'Running benchmark with concurrency: {client_config.concurrency}, iteration: {client_config.iterations}, num-prompts: {num_prompts}'
        )

        # Build benchmark command
        benchmark_cmd = [
            "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving",
            "--model", model_path, "--dataset-name", "random", "--random-ids",
            "--num-prompts",
            str(num_prompts), "--random-input-len",
            str(server_config.isl), "--random-output-len",
            str(server_config.osl), "--random-range-ratio",
            str(client_config.random_range_ratio), "--ignore-eos",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--max-concurrency",
            str(client_config.concurrency)
        ]

        print(f'Running benchmark with command:')
        print(' '.join(benchmark_cmd))
        print()

        # Prepare log filename
        benchmark_log_filename = f"trtllm-benchmark.{server_config.to_str()}.{client_config.to_str()}.log"

        benchmark_process = None
        need_kill_benchmark_process = False
        try:
            with open(benchmark_log_filename, 'w') as f:
                f.write(f"[Perf Sanity Test] GPU Info: {self.gpu_info}\n")
                f.write("[Perf Sanity Test] Server Config:\n")
                server_config_log_content = server_config.to_log_content()
                for line in server_config_log_content:
                    f.write(line)
                f.write("\n")
                f.write("[Perf Sanity Test] Client Config:\n")
                client_config_log_content = client_config.to_log_content()
                for line in client_config_log_content:
                    f.write(line)
                f.write("\n")

            # Start benchmark as subprocess
            with open(benchmark_log_filename, 'a') as log_file:
                benchmark_process = subprocess.Popen(benchmark_cmd,
                                                     stdout=log_file,
                                                     stderr=subprocess.STDOUT)

            start_time = time.time()
            while benchmark_process.poll() is None:  # Process is still running
                time.sleep(10)  # Wait 10 seconds
                elapsed_time = time.time() - start_time
                if elapsed_time > self.timeout:
                    print(
                        f"Failed to run benchmark due to timeout after {elapsed_time:.0f} seconds (>{self.timeout} seconds)"
                    )
                    need_kill_benchmark_process = True
                    return False

                # Check server log and benchmark log for RuntimeError
                if self.check_for_runtime_error(server_log_filename):
                    print(
                        f"Failed to run benchmark due to RuntimeError found in server log: {server_log_filename}"
                    )
                    need_kill_benchmark_process = True
                    return False

                if self.check_for_runtime_error(benchmark_log_filename):
                    print(
                        f"Failed to run benchmark due to RuntimeError found in benchmark log: {benchmark_log_filename}"
                    )
                    need_kill_benchmark_process = True
                    return False

            # Process completed, check final return code
            return_code = benchmark_process.returncode
            if return_code != 0:
                print(
                    f"Benchmark process completed with error code: {return_code}"
                )

                # Read and display error output
                try:
                    with open(benchmark_log_filename, 'r') as f:
                        error_content = f.read()
                        print(
                            f"Benchmark error output:\n{error_content[-1000:]}"
                        )  # Last 1000 chars
                except Exception as e:
                    print(f"Could not read benchmark log: {e}")

                print(
                    f"Skipping this concurrency level and continuing with next one..."
                )
                print("-----------------------------------------")
                return True  # Continue with next concurrency, don't skip test case

            # Success case
            print(
                f"Benchmark completed successfully (PID: {benchmark_process.pid})"
            )
            print("-----------------------------------------")
            return True  # Continue with next concurrency

        except Exception as e:
            print(
                f"Error running benchmark with concurrency {client_config.concurrency}: {e}"
            )
            print("-----------------------------------------")
            return True  # Continue with next concurrency, don't skip test case

        finally:
            # Cleanup: Kill benchmark process using shell commands like in the original bash script
            if need_kill_benchmark_process and benchmark_process:
                print(
                    f"Need killing benchmark process. Now killing benchmark process"
                )
                try:
                    # Use shell commands for more reliable process killing
                    subprocess.run(f"kill -9 {benchmark_process.pid}",
                                   shell=True,
                                   check=False)
                    benchmark_process.wait(timeout=10)
                except Exception as e:
                    print(f"Warning: Error killing benchmark process: {e}")

                time.sleep(5)  # Give it time to clean up resources
                print(f"Benchmark process cleanup completed")

    def run_server_config(self, server_config: ServerConfig,
                          server_config_id: int) -> None:
        """Run a server configuration using the execution plan with retry logic"""
        model_name = server_config.model_name

        retry_count = 0
        while retry_count < max_retries:
            retry_count += 1
            print(
                f"Attempt {retry_count}/{max_retries} for server config {server_config_id} ({model_name})"
            )

            try:
                success = self._run_server_config_attempt(
                    server_config, server_config_id)
                if success:
                    print(
                        f"Server config {server_config_id} ({model_name}) completed successfully on attempt {retry_count}"
                    )
                    return
                else:
                    print(
                        f"Server config {server_config_id} ({model_name}) failed on attempt {retry_count}"
                    )
                    if retry_count < max_retries:
                        print(f"Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print(
                            f"Server config {server_config_id} ({model_name}) failed after {max_retries} attempts. Skipping."
                        )
                        self.update_failed_server_config(
                            server_config,
                            f"Failed after {max_retries} attempts - all retries exhausted"
                        )
                        return
            except Exception as e:
                print(
                    f"Server config {server_config_id} ({model_name}) encountered exception on attempt {retry_count}: {e}"
                )
                if retry_count < max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(
                        f"Server config {server_config_id} ({model_name}) failed after {max_retries} attempts due to exceptions. Skipping."
                    )
                    self.update_failed_server_config(
                        server_config,
                        f"Failed after {max_retries} attempts due to exception: {e} - all retries exhausted"
                    )
                    return

    def _run_server_config_attempt(self, server_config: ServerConfig,
                                   server_config_id: int) -> bool:
        """Single attempt to run a server configuration. Returns True if successful, False if failed."""
        model_name = server_config.model_name

        # Get model path
        model_path = MODEL_PATHS.get(model_name)
        hf_model_path = HF_MODEL_PATHS.get(model_name)

        # Use local path if it exists, otherwise use HF model path
        if model_path and os.path.exists(model_path):
            MODEL = model_path
            print(f"Using local model path: {MODEL}")
        else:
            if hf_model_path:
                MODEL = f"--model {hf_model_path}"
                print(f"Local path not found, using HF model: {hf_model_path}")
            else:
                print(
                    f"Error: Neither local path nor HF model path found for {model_name}"
                )
                return False

        # Start server
        server_log_filename = f"trtllm-serve.{server_config.to_str()}.log"

        # Generate extra-llm-api-config.yml with matching filename pattern
        config_content = server_config.generate_extra_llm_api_config()
        config_filename = f"extra-llm-api-config.{server_config.to_str()}.yml"
        config_path = config_filename

        with open(config_path, 'w') as f:
            f.write(config_content)

        print(f"extra-llm-api-config.yml ({config_filename}):")
        print(config_content)

        # Build trtllm-serve command
        serve_cmd = [
            "trtllm-serve", MODEL, "--backend", "pytorch", "--tp_size",
            str(server_config.tp), "--ep_size",
            str(server_config.ep), "--pp_size",
            str(server_config.pp), "--max_batch_size",
            str(server_config.max_batch_size), "--max_num_tokens",
            str(server_config.max_num_tokens),
            "--kv_cache_free_gpu_memory_fraction",
            str(server_config.free_gpu_memory_fraction),
            "--extra_llm_api_options", config_path
        ]

        print("Starting trtllm-serve with command:")
        print(' '.join(serve_cmd))
        print()

        server_process = None
        try:
            with open(server_log_filename, 'w') as log_file:
                server_process = subprocess.Popen(serve_cmd,
                                                  stdout=log_file,
                                                  stderr=subprocess.STDOUT)

            # Wait for server to be ready
            if not self.wait_for_server(server_process, server_log_filename):
                return False

            # Run benchmarks based on execution plan
            for client_config_index in self.execution_plan[server_config_id]:
                client_config = self.client_configs[server_config_id][
                    client_config_index]
                should_continue = self.run_benchmark(server_config,
                                                     client_config, model_path,
                                                     server_log_filename)
                # If run_benchmark returns False, mark server config as failed
                if not should_continue:
                    return False

            # If we reach here, all benchmarks completed successfully
            return True

        except Exception as e:
            print(f"Exception during server config execution: {e}")
            return False

        finally:
            # Cleanup: Kill server process using shell commands like in the original bash script
            if server_process:
                print(f"Stopping server process")
                try:
                    # Use shell commands for more reliable process killing
                    subprocess.run(f"kill -9 {server_process.pid}",
                                   shell=True,
                                   check=False)
                    subprocess.run(
                        f"wait {server_process.pid} 2>/dev/null || true",
                        shell=True,
                        check=False)
                    subprocess.run(f"rm -rf ~/.triton/cache || true",
                                   shell=True,
                                   check=False)
                    # subprocess.run(f"rm -rf ~/.cache/flashinfer/ || true",
                    #                shell=True,
                    #                check=False)
                except Exception as e:
                    print(f"Warning: Error killing server process: {e}")

                time.sleep(5)  # Give it time to clean up resources
                print(f"Server process cleanup completed")

    def update_failed_server_config(self, server_config: ServerConfig,
                                    server_config_id: int,
                                    failure_reason: str) -> None:
        """Update server config info with failure reason"""
        # Find the existing server config info and update its failure reason
        for test_case_info in self.test_case_infos:
            if test_case_info['server_config_id'] == server_config_id:
                test_case_info['failure_reason'] = failure_reason
                print(
                    f"Updated server config {server_config_id} ({server_config.model_name}) with failure reason: {failure_reason}"
                )
                return

    def generate_reproduction_script(self, test_case_info: Dict[str,
                                                                Any]) -> str:
        """Generate a shell script to reproduce a server config"""
        server_config = test_case_info['server_config']
        server_config_id = test_case_info['server_config_id']
        model_name = server_config.model_name
        failure_reason = "Run Successfully" if test_case_info[
            'failure_reason'] == "Success" else f"Failure Reason: {test_case_info['failure_reason']}"
        node_name = test_case_info['node_name']
        test_case_info['gpu_info']

        # Get model path
        model_path = MODEL_PATHS.get(model_name)

        script_content = f"""#!/bin/bash
# Reproduction script for server config {server_config_id} ({model_name})
# Node: {node_name}
# {failure_reason}
# Server Configuration:
#   - Model: {model_name}
#   - TP: {server_config.tp}
#   - EP: {server_config.ep}
#   - PP: {server_config.pp}
#   - Attention Backend: {server_config.attention_backend}
#   - MoE Backend: {server_config.moe_backend}
#   - Enable Attention DP: {server_config.enable_attention_dp}
#   - Free GPU Memory Fraction: {server_config.free_gpu_memory_fraction}
#   - Max Batch Size: {server_config.max_batch_size}
#   - Input Sequence Length: {server_config.isl}
#   - Output Sequence Length: {server_config.osl}
#   - Max Num Tokens: {server_config.max_num_tokens}
#   - MoE Max Num Tokens: {server_config.moe_max_num_tokens}

set -e

# Function to wait for server to be ready
wait_for_server() {{
    local timeout={self.timeout}
    local attempt=0

    echo "Waiting for trtllm-serve to be ready..."

    while [ $((attempt * 60)) -le $timeout ]; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Error: Server process has died"
            return 1
        fi

        # Check for runtime errors in server log
        if grep -q "RuntimeError\\|runtime error\\|CUDA error\\|illegal memory access\\|terminate called" "$SERVER_LOG" 2>/dev/null; then
            echo "RuntimeError detected in server log: $SERVER_LOG"
            echo "Killing server process due to runtime error"
            kill -9 $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
            return 1
        fi

        # Try to connect to server
        if curl -s "http://localhost:8000/v1/models" > /dev/null 2>&1; then
            echo "Server is ready! HTTP status: 200"
            return 0
        fi

        echo "Elapsed time: $((attempt * 60)) / $timeout seconds: Server not ready yet, waiting..."
        sleep 60
        attempt=$((attempt + 1))
    done

    echo "Error: Server did not become ready after $timeout seconds"
    return 1
}}

# Function to cleanup server process
cleanup_server() {{
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server"
        kill -9 $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        sleep 5  # Give it time to clean up resources
        echo "Server cleanup completed"
    fi
}}

# Set trap to cleanup server on script exit
trap cleanup_server EXIT

# Generate extra-llm-api-config.yml
CONFIG_FILENAME="extra-llm-api-config.{server_config.to_str()}.yml"

cat > "$CONFIG_FILENAME" << 'EOF'
{server_config.generate_extra_llm_api_config()}
EOF

echo "extra-llm-api-config.yml ($CONFIG_FILENAME):"
cat "$CONFIG_FILENAME"

# Start trtllm-serve in background
SERVER_LOG="trtllm-serve.{server_config.to_str()}.log"

echo "Starting trtllm-serve with command:"
echo "trtllm-serve {model_path} --backend pytorch --tp_size {server_config.tp} --ep_size {server_config.ep} --pp_size {server_config.pp} --max_batch_size {server_config.max_batch_size} --max_num_tokens {server_config.max_num_tokens} --kv_cache_free_gpu_memory_fraction {server_config.free_gpu_memory_fraction} --extra_llm_api_options $CONFIG_FILENAME"

trtllm-serve \\
    {model_path} \\
    --backend pytorch \\
    --tp_size {server_config.tp} \\
    --ep_size {server_config.ep} \\
    --pp_size {server_config.pp} \\
    --max_batch_size {server_config.max_batch_size} \\
    --max_num_tokens {server_config.max_num_tokens} \\
    --kv_cache_free_gpu_memory_fraction {server_config.free_gpu_memory_fraction} \\
    --extra_llm_api_options "$CONFIG_FILENAME" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to be ready
if ! wait_for_server; then
    echo "Failed to start server, exiting"
    exit 1
fi

echo "Server is ready, starting benchmarks..."

# Run benchmarks for each concurrency level
"""
        # Add benchmark commands for each client config
        for client_config_index in self.execution_plan.get(
                server_config_id, []):
            client_config = self.client_configs[server_config_id][
                client_config_index]
            num_prompts = client_config.concurrency * client_config.iterations

            script_content += f"""
echo "Running benchmark with concurrency: {client_config.concurrency}, iterations: {client_config.iterations}, num-prompts: {num_prompts}"

BENCHMARK_LOG="trtllm-benchmark.{server_config.to_str()}.{client_config.to_str()}.log"

echo "Running benchmark with command:"
echo "python -m tensorrt_llm.serve.scripts.benchmark_serving --model {model_path} --dataset-name random --random-ids --num-prompts {num_prompts} --random-input-len {server_config.isl} --random-output-len {server_config.osl} --random-range-ratio {client_config.random_range_ratio} --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --max-concurrency {client_config.concurrency}"

python -m tensorrt_llm.serve.scripts.benchmark_serving \\
    --model {model_path} \\
    --dataset-name random \\
    --random-ids \\
    --num-prompts {num_prompts} \\
    --random-input-len {server_config.isl} \\
    --random-output-len {server_config.osl} \\
    --random-range-ratio {client_config.random_range_ratio} \\
    --ignore-eos \\
    --percentile-metrics ttft,tpot,itl,e2el \\
    --max-concurrency {client_config.concurrency} > "$BENCHMARK_LOG" 2>&1

if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully"
else
    echo "Benchmark failed with error code $?"
fi

echo "-----------------------------------------"
"""

        script_content += f"""

echo "All benchmarks completed successfully!"
echo "Server will be automatically cleaned up on script exit"
"""

        return script_content

    def generate_reproduction_scripts(self) -> None:
        """Generate reproduction scripts for all server configs"""
        if not self.test_case_infos:
            print("No server configs to generate reproduction scripts for")
            return

        print(
            f"\nGenerating reproduction scripts for {len(self.test_case_infos)} server configs..."
        )

        generated_scripts = []
        for i, test_case_info in enumerate(self.test_case_infos, 1):
            try:
                server_config = test_case_info['server_config']
                server_config_id = test_case_info['server_config_id']
                model_name = server_config.model_name

                script_content = self.generate_reproduction_script(
                    test_case_info)
                script_filename = f"reproduce.{model_name}.tp{server_config.tp}.ep{server_config.ep}.pp{server_config.pp}.adp{server_config.enable_attention_dp}.attn{server_config.attention_backend}.moe{server_config.moe_backend}.gpu{server_config.free_gpu_memory_fraction}.batch{server_config.max_batch_size}.isl{server_config.isl}.osl{server_config.osl}.tokens{server_config.max_num_tokens}.moetokens{server_config.moe_max_num_tokens}.kv{server_config.kv_cache_dtype}.reuse{server_config.enable_block_reuse}.chunk{server_config.enable_chunked_prefill}.overlap{server_config.disable_overlap_scheduler}.pad{server_config.enable_padding}.sh"

                with open(script_filename, 'w') as f:
                    f.write(script_content)

                # Make script executable
                os.chmod(script_filename, 0o755)

                generated_scripts.append(script_filename)
                print(f"Generated reproduction script: {script_filename}")

            except Exception as e:
                print(
                    f"Warning: Failed to generate reproduction script for server config {server_config_id}: {e}"
                )
                continue

    def run_benchmarks(self) -> None:
        """Main function to run all benchmarks from config file"""
        script_start_time = time.time()

        print(f"Using config file: {self.config_file}")
        if self.select_pattern:
            print(f"Select pattern: {self.select_pattern}")
        else:
            print("Select pattern: default (all server configs)")
        if self.skip_pattern:
            print(f"Skip pattern: {self.skip_pattern}")
        else:
            print("Skip pattern: default (no skipping)")

        # Parse configuration file
        self.server_configs, self.client_configs = parse_config_file(
            self.config_file)

        # Build execution plan
        self.build_execution_plan()

        # Initialize test case infos with 'Success' status
        self.initialize_test_case_infos()

        # Always generate reproduction scripts for all test cases
        if self.test_case_infos:
            self.generate_reproduction_scripts()

        # Print execution plan before starting benchmarks
        self.print_execution_plan()

        # Run each server config based on execution plan
        for i, (server_config_id,
                server_config) in enumerate(self.server_configs, 1):

            if server_config_id not in self.execution_plan:
                print("=" * 57)
                print(
                    f"Server config {i}/{len(self.server_configs)} (ID: {server_config_id}): {server_config.model_name} - SKIPPED"
                )
                print("=" * 57)
                continue

            print("=" * 57)
            print(
                f"Server config {i}/{len(self.server_configs)} (ID: {server_config_id}): {server_config.model_name}"
            )
            print(
                f"Config: TP={server_config.tp}, EP={server_config.ep}, PP={server_config.pp}, "
                f"attention_backend={server_config.attention_backend}, moe_backend={server_config.moe_backend}"
            )
            print("=" * 57)

            self.run_server_config(server_config, server_config_id)

        # Print completion summary
        self.print_completion_summary(script_start_time)

    def print_completion_summary(self, script_start_time: float) -> None:
        """Print script completion summary including runtime and server config results"""
        print("=" * 57)
        print("SCRIPT COMPLETION SUMMARY")
        print("=" * 57)

        # Calculate and display total script runtime
        script_total_time = time.time() - script_start_time
        hours = int(script_total_time // 3600)
        minutes = int((script_total_time % 3600) // 60)
        seconds = int(script_total_time % 60)

        print(
            f"Total script runtime: {hours:02d}:{minutes:02d}:{seconds:02d} (HH:MM:SS)"
        )
        print(f"Total runtime in seconds: {script_total_time:.2f}")
        print("All benchmarks completed!")

        # Print summary of failed server configs
        if self.test_case_infos:
            success_count = 0
            failed_count = 0

            for i, test_case_info in enumerate(self.test_case_infos, 1):
                server_config = test_case_info['server_config']
                server_config_id = test_case_info['server_config_id']
                failure_reason = test_case_info['failure_reason']
                status = "SUCCESS" if failure_reason == "Success" else "FAILED"

                print(
                    f"{i}. Server Config {server_config_id} ({server_config.model_name}) - {status}"
                )
                if failure_reason != "Success":
                    print(f"   Failure Reason: {failure_reason}")
                print(f"   Node: {test_case_info['node_name']}")
                print()

                if failure_reason == "Success":
                    success_count += 1
                else:
                    failed_count += 1

            print(f"Total server configs: {len(self.test_case_infos)}")
            print(f"Successful: {success_count}")
            print(f"Failed: {failed_count}")
        else:
            print("\nNo server configs were executed!")


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks from YAML configuration file')
    parser.add_argument('--output_folder',
                        required=True,
                        help='Output folder for benchmark results')
    parser.add_argument('--config_file',
                        required=True,
                        help='Path to YAML configuration file')
    parser.add_argument(
        '--skip',
        help=
        'Skip pattern: "2,4-1" means skip server config 2 and server config 4\'s 1st client config'
    )
    parser.add_argument(
        '--select',
        help=
        'Select pattern: "1,3,5" means only run server configs 1, 3, and 5; "1-1,2-3" means only run server config 1\'s 1st client config and server config 2\'s 3rd client config'
    )
    parser.add_argument('--timeout', help='Timeout in seconds', default=3600)

    args = parser.parse_args()

    try:
        subprocess.run(f'echo "TRT-LLM GIT COMMIT": $TRT_LLM_GIT_COMMIT',
                       shell=True,
                       check=True)
    except subprocess.CalledProcessError:
        print("Warning: Could not echo TRT-LLM GIT COMMIT")

    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder '{args.output_folder}' does not exist")
        sys.exit(1)

    try:
        runner = BenchmarkRunner(args.output_folder, args.config_file,
                                 args.skip, args.select, args.timeout)
        runner.run_benchmarks()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
