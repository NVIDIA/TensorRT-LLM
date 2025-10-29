#!/usr/bin/env python3
import argparse
import ast
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple

import requests
import yaml


def get_node_name() -> str:
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


def get_gpu_info() -> str:
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
    except Exception as e:
        return f"Failed to get GPU information: {e}"


# Model PATH of local dir synced from internal LLM models repo
MODEL_PATH_DICT = {
    "llama_v2_7b": "llama-models-v2/llama-v2-7b-hf",  # not safetensors repo
    "llama_v2_13b": "llama-models-v2/llama-v2-13b-hf",  # not safetensors repo
    "llama_v2_70b": "llama-models-v2/llama-v2-70b-hf",  # not safetensors repo
    "llama_v3.1_8b": "llama-3.1-model/Meta-Llama-3.1-8B",
    "llama_v3.1_8b_instruct": "llama-3.1-model/Llama-3.1-8B-Instruct",
    "llama_v3.1_8b_instruct_fp8": "llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
    "llama_v3.1_8b_instruct_fp4":
    "modelopt-hf-model-hub/Llama-3.1-8B-Instruct-fp4",
    "llama_v3.1_70b": "llama-3.1-model/Meta-Llama-3.1-70B",
    "llama_v3.3_70b_instruct": "llama-3.3-models/Llama-3.3-70B-Instruct",
    "llama_v3.1_70b_instruct_fp8": "llama-3.1-model/Llama-3.1-70B-Instruct-FP8",
    "llama_v3.3_70b_instruct_fp8":
    "modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8",
    "llama_v3.3_70b_instruct_fp4":
    "modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp4",
    "llama_v3.3_70b_instruct": "llama-3.3-models/Llama-3.3-70B-Instruct",
    "llama_v3.1_405b_instruct_fp8":
    "llama-3.1-model/Llama-3.1-405B-Instruct-FP8",
    "llama_v3.1_405b_instruct_fp4":
    "modelopt-hf-model-hub/Llama-3.1-405B-Instruct-fp4",
    "llama_v3.1_70b_instruct": "llama-3.1-model/Meta-Llama-3.1-70B-Instruct",
    "llama_v3.2_1b": "llama-3.2-models/Llama-3.2-1B",
    "llama_v3.1_nemotron_nano_8b": "Llama-3.1-Nemotron-Nano-8B-v1",
    "llama_v3.1_nemotron_nano_8b_fp8": "Llama-3.1-Nemotron-Nano-8B-v1-FP8",
    "llama_v3.3_nemotron_super_49b":
    "nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1",
    "llama_v3.3_nemotron_super_49b_fp8":
    "nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-FP8",
    "llama_v3.1_nemotron_ultra_253b":
    "nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1",
    "llama_v3.1_nemotron_ultra_253b_fp8":
    "nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1-FP8",
    "llama_v4_scout_17b_16e_instruct":
    "llama4-models/Llama-4-Scout-17B-16E-Instruct",
    "llama_v4_scout_17b_16e_instruct_fp8":
    "llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8",
    "llama_v4_scout_17b_16e_instruct_fp4":
    "llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4",
    "llama_v4_maverick_17b_128e_instruct":
    "llama4-models/Llama-4-Maverick-17B-128E-Instruct",
    "llama_v4_maverick_17b_128e_instruct_fp8":
    "llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mixtral_8x7b_v0.1": "Mixtral-8x7B-v0.1",
    "mixtral_8x7b_v0.1_instruct": "Mixtral-8x7B-Instruct-v0.1",
    "mixtral_8x7b_v0.1_instruct_fp8": "Mixtral-8x7B-Instruct-v0.1-fp8",
    "mixtral_8x7b_v0.1_instruct_fp4":
    "modelopt-hf-model-hub/Mixtral-8x7B-Instruct-v0.1-fp4",
    "mistral_nemo_12b_base": "Mistral-Nemo-Base-2407",
    "deepseek_r1_distill_qwen_32b": "DeepSeek-R1/DeepSeek-R1-Distill-Qwen-32B",
    "mixtral_8x22b_v0.1": "Mixtral-8x22B-v0.1",
    "mistral_7b_v0.1": "mistral-7b-v0.1",
    "ministral_8b": "Ministral-8B-Instruct-2410",
    "ministral_8b_fp8": "Ministral-8B-Instruct-2410-FP8",
    "gemma_3_1b_it": "gemma/gemma-3-1b-it",
    "deepseek_r1_fp8": "DeepSeek-R1/DeepSeek-R1",
    "deepseek_r1_nvfp4": "DeepSeek-R1/DeepSeek-R1-FP4",
    "deepseek_r1_0528_fp8": "DeepSeek-R1/DeepSeek-R1-0528/",
    "deepseek_r1_0528_fp4": "DeepSeek-R1/DeepSeek-R1-0528-FP4/",
    "deepseek_v3_lite_fp8": "DeepSeek-V3-Lite/fp8",
    "deepseek_v3_lite_nvfp4": "DeepSeek-V3-Lite/nvfp4_moe_only",
    "qwen2_7b_instruct": "Qwen2-7B-Instruct",
    "qwen_14b_chat": "Qwen-14B-Chat",
    "qwen3_235b_a22b_fp8": "Qwen3/saved_models_Qwen3-235B-A22B_fp8_hf",
    "qwen3_235b_a22b_fp4": "Qwen3/saved_models_Qwen3-235B-A22B_nvfp4_hf",
    "starcoder2_3b": "starcoder2-3b",
    "starcoder_15b": "starcoder2-15b",
    "t5": "t5-small",  # not supported for trtllm-bench build config
    "flan_t5_base":
    "flan-t5-small",  # not supported for trtllm-bench build config
    "flan_t5_large":
    "flan-t5-xl",  # not supported for trtllm-bench build config
    "whisper_large_v3":
    "whisper-models/large-v3",  # not supported for trtllm-bench tokenizer
    "bart_large_cnn": "bart-large-cnn",  # not safetensors repo
    "mbart_large_50_many_to_one_mmt": "mbart-large-50-many-to-one-mmt",
    "mamba_130m": "mamba/mamba-130m-hf",
    "mamba_370m": "mamba/mamba-370m-hf",
    "mamba_2.8b": "mamba/mamba-2.8b-hf",
    "gpt_20b": "gpt-neox-20b",
    "gpt_350m_moe": "gpt2-medium",
    "phi_3_mini_4k_instruct": "Phi-3/Phi-3-mini-4k-instruct",
    "phi_3_mini_128k_instruct": "Phi-3/Phi-3-mini-128k-instruct",
    "phi_4_mini_instruct": "Phi-4-mini-instruct",
    "phi_4_multimodal_instruct": "multimodals/Phi-4-multimodal-instruct",
    "phi_4_multimodal_instruct_image": "multimodals/Phi-4-multimodal-instruct",
    "phi_4_multimodal_instruct_audio": "multimodals/Phi-4-multimodal-instruct",
    "bielik_11b_v2.2_instruct": "Bielik-11B-v2.2-Instruct",
    "bielik_11b_v2.2_instruct_fp8": "Bielik-11B-v2.2-Instruct-FP8",
    "mistral_small_v3.1_24b": "Mistral-Small-3.1-24B-Instruct-2503",
    "gpt_oss_120b_fp4": "gpt_oss/gpt-oss-120b",
}
# Model PATH of HuggingFace
HF_MODEL_PATH = {
    "llama_v2_7b_hf": "meta-llama/Llama-2-7b-hf",
    "llama_v2_70b_hf": "meta-llama/Llama-2-70b-hf",
    "falcon_180b_hf": "tiiuae/falcon-180B",
    "gptj_6b_hf": "EleutherAI/gpt-j-6b",
    "llama_v3_8b_hf": "meta-llama/Meta-Llama-3-8B",
    "llama_v3.1_8b_hf": "meta-llama/Llama-3.1-8B",
    "llama_v3.1_8b_instruct_hf": "nvidia/Llama-3.1-8B-Instruct-FP8",
    "llama_v3.1_70b_instruct_hf": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama_v3_70b_hf": "meta-llama/Meta-Llama-3-70B",
    "llama_v3.1_70b_hf": "meta-llama/Llama-3.1-70B",
    "llama_v3.1_405b_hf": "meta-llama/Llama-3.1-405B",
    "llama_v3.1_nemotron_nano_8b_hf": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "llama_v3.1_nemotron_nano_8b_fp8_hf":
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1-FP8",
    "llama_v3.3_nemotron_super_49b_hf":
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "llama_v3.3_nemotron_super_49b_fp8_hf":
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1-FP8",
    "llama_v3.1_nemotron_ultra_253b_fp8_hf":
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8",
    "mixtral_8x7b_v0.1_hf": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral_8x7b_v0.1_instruct_hf": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral_7b_v0.1_hf": "mistralai/Mistral-7B-v0.1",
    "ministral_8b_hf": "mistralai/Ministral-8B-Instruct-2410",
    "flan_t5_base_hf": "google/flan-t5-small",
    "phi_4_mini_instruct_hf": "microsoft/Phi-4-mini-instruct",
    "gemma_3_1b_it_hf": "google/gemma-3-1b-it",
}

LLM_MODELS_ROOT = os.environ.get('LLM_MODELS_ROOT',
                                 '/home/scratch.trt_llm_data/llm-models')


# Model path mapping
def llm_models_root():
    return LLM_MODELS_ROOT


def get_model_dir(model_name: str) -> str:
    model_dir = ""
    if model_name in MODEL_PATH_DICT.keys():
        model_dir = os.path.join(llm_models_root(), MODEL_PATH_DICT[model_name])
    elif model_name in HF_MODEL_PATH.keys():
        model_dir = os.path.join(llm_models_root(),
                                 MODEL_PATH_DICT[model_name.split('_hf')[0]])
    return model_dir


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
    "stream_interval": (True, int),
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
    """
    Configurations of trtllm-server.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        tp: int,
        ep: int,
        max_num_tokens: int,
        attention_backend: str,
        max_batch_size: int,
        pp: int = 1,
        enable_chunked_prefill: bool = False,
        disable_overlap_scheduler: bool = False,
        moe_backend: str = "",
        moe_max_num_tokens: str = "",
        stream_interval: int = 10,
        enable_attention_dp: bool = False,
        attention_dp_balance: bool = False,
        batching_wait_iters: int = 10,
        timeout_iters: int = 50,
        kv_cache_dtype: str = "fp8",
        enable_block_reuse: bool = False,
        free_gpu_memory_fraction: float = 0.8,
        enable_padding: bool = True,
    ):
        self.name = name
        self.model_name = model_name
        self.tp = tp
        self.ep = ep
        self.pp = pp
        self.max_num_tokens = max_num_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.disable_overlap_scheduler = disable_overlap_scheduler
        self.attention_backend = attention_backend
        self.moe_backend = moe_backend
        self.moe_max_num_tokens = moe_max_num_tokens
        self.stream_interval = stream_interval
        self.enable_attention_dp = enable_attention_dp
        self.attention_dp_balance = attention_dp_balance
        self.batching_wait_iters = batching_wait_iters
        self.timeout_iters = timeout_iters
        self.kv_cache_dtype = kv_cache_dtype
        self.enable_block_reuse = enable_block_reuse
        self.free_gpu_memory_fraction = free_gpu_memory_fraction
        self.max_batch_size = max_batch_size
        self.enable_padding = enable_padding

        self.model_path = ""

    def to_cmd(self, working_dir: str) -> List[str]:
        model_dir = get_model_dir(self.model_name)
        self.model_path = model_dir if os.path.exists(
            model_dir) else self.model_name
        config_path = os.path.join(working_dir,
                                   f"extra-llm-api-config.{self.name}.yml")
        return [
            "trtllm-serve", self.model_path, "--host", "localhost", "--port",
            "8000", "--backend", "pytorch", "--extra_llm_api_options",
            config_path
        ]

    def generate_extra_llm_api_config(self) -> str:
        """Generate extra-llm-api-config.yml content"""
        config_lines = [
            f"tensor_parallel_size: {self.tp}",
            f"moe_expert_parallel_size: {self.ep}",
            f"pipeline_parallel_size: {self.pp}",
            f"max_num_tokens: {self.max_num_tokens}",
            f"enable_attention_dp: {str(self.enable_attention_dp).lower()}",
            f"disable_overlap_scheduler: {str(self.disable_overlap_scheduler).lower()}",
            f"stream_interval: {self.stream_interval}",
            f"attn_backend: {self.attention_backend}",
            f"enable_chunked_prefill: {str(self.enable_chunked_prefill).lower()}",
            "cuda_graph_config:",
            f"  enable_padding: {str(self.enable_padding).lower()}",
            f"  max_batch_size: {self.max_batch_size}",
            "kv_cache_config:",
            f"  dtype: {self.kv_cache_dtype}",
            f"  free_gpu_memory_fraction: {self.free_gpu_memory_fraction}",
            f"  enable_block_reuse: {str(self.enable_block_reuse).lower()}",
            "print_iter_log: false",
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
    """
    Configurations of benchmark client.
    """

    def __init__(self,
                 name: str,
                 model_name: str,
                 concurrency: int,
                 iterations: int,
                 isl: int,
                 osl: int,
                 random_range_ratio: float = 0.0):
        self.name = name
        self.model_name = model_name
        self.concurrency = concurrency
        self.iterations = iterations
        self.isl = isl
        self.osl = osl
        self.random_range_ratio = random_range_ratio

        self.model_path = ""

    def to_cmd(self, working_dir: str) -> List[str]:
        model_dir = get_model_dir(self.model_name)
        self.model_path = model_dir if os.path.exists(
            model_dir) else self.model_name
        return [
            "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving",
            "--model", self.model_path, "--dataset-name", "random",
            "--random-ids", "--num-prompts",
            str(self.concurrency * self.iterations), "--random-input-len",
            str(self.isl), "--random-output-len",
            str(self.osl), "--random-range-ratio",
            str(self.random_range_ratio), "--ignore-eos",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--max-concurrency",
            str(self.concurrency)
        ]


def parse_select_pattern(select_pattern: str):
    """Parse select pattern like 'r1_fp4_dep4,r1_fp4_tep4:con1_iter1_1024_1024,r1_fp4_tep4:con8_iter1_1024_1024'

    Format:
    - ',' splits different server configs
    - ':' means for this server, we choose specific clients
    - If no ':', all clients are chosen for that server

    Returns:
    - Dict with server name as key and either None (all clients) or set of client names as value
    """
    execution_plan = {}

    parts = select_pattern.split(',')
    for part in parts:
        part = part.strip()
        if not part:  # Skip empty parts
            continue

        if ':' in part:
            # Format: "server_name:client_name"
            server_name, client_name = part.split(':', 1)
            server_name = server_name.strip()
            client_name = client_name.strip()

            # Only add if not already set to None (all clients)
            if server_name not in execution_plan:
                execution_plan[server_name] = set()

            if execution_plan[server_name] is not None:
                execution_plan[server_name].add(client_name)
        else:
            # Format: "server_name" - select all clients for this server
            server_name = part.strip()
            execution_plan[server_name] = None

    return execution_plan


def parse_config_file(config_file_path: str, select_pattern: str = None):
    """Parse YAML configuration file and create ServerConfig and ClientConfig objects

    Args:
        config_file_path: Path to YAML configuration file
        select_pattern: Selection pattern string (e.g., "r1_fp4_dep4,r1_fp4_tep4:con1_iter1_1024_1024")

    Returns:
        execution_plan: None (all servers/clients) or dict with server names as keys
        server_configs: List of ServerConfig objects
        server_client_configs: Dict with server id as key and list of ClientConfig as value
    """
    # Parse selection pattern
    if select_pattern:
        execution_plan = parse_select_pattern(select_pattern)
    else:
        execution_plan = None

    # Read YAML config file
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    server_configs = []
    server_client_configs = {}

    for server_config_data in config['server_configs']:
        server_name = server_config_data['name']

        # Check if this server should be included based on execution_plan
        if execution_plan is not None and server_name not in execution_plan:
            continue

        # Create ServerConfig object
        server_config = ServerConfig(
            name=server_config_data['name'],
            model_name=server_config_data['model_name'],
            tp=server_config_data['tp'],
            ep=server_config_data['ep'],
            pp=server_config_data.get('pp', 1),
            attention_backend=server_config_data.get('attention_backend',
                                                     'TRTLLM'),
            moe_backend=server_config_data.get('moe_backend', ''),
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

        server_id = len(server_configs)
        server_configs.append(server_config)

        # Create ClientConfig objects
        client_configs = []
        selected_client_names = execution_plan.get(
            server_name) if execution_plan else None

        for client_config_data in server_config_data['client_configs']:
            client_name = client_config_data['name']

            # Check if this client should be included
            # Include if: execution_plan is None OR selected_client_names is None OR client_name in selected_client_names
            if execution_plan is not None and selected_client_names is not None:
                if client_name not in selected_client_names:
                    continue

            client_config = ClientConfig(
                name=client_config_data['name'],
                model_name=server_config_data['model_name'],
                concurrency=client_config_data['concurrency'],
                iterations=client_config_data.get('iterations', 1),
                isl=client_config_data.get('isl', 1024),
                osl=client_config_data.get('osl', 1024),
                random_range_ratio=client_config_data.get(
                    'random_range_ratio', 0.0))
            client_configs.append(client_config)

        server_client_configs[server_id] = client_configs

    return execution_plan, server_configs, server_client_configs


def get_trtllm_server_client_commands(
        server_configs: List[ServerConfig],
        server_client_configs: Dict[int, List[ClientConfig]], working_dir: str):
    server_cmds = []
    client_cmds = []
    names = []
    for server_idx, client_configs in server_client_configs.items():
        server_config = server_configs[server_idx]
        server_cmd = server_config.to_cmd(working_dir)
        # Generate extra-llm-api-config.yml
        config_content = server_config.generate_extra_llm_api_config()
        config_filename = f"extra-llm-api-config.{server_config.name}.yml"
        config_path = os.path.join(working_dir, config_filename)
        with open(config_path, 'w') as f:
            f.write(config_content)
        for client_config in client_configs:
            server_cmds.append(server_cmd)
            client_cmd = client_config.to_cmd(working_dir)
            client_cmds.append(client_cmd)
            names.append(f"{server_config.name}-{client_config.name}")
    return server_cmds, client_cmds, names


class PerfServerBenchmarkCmds(NamedTuple):
    server_cmds: List[List[str]]
    client_cmds: List[List[str]]
    names: List[str]
    working_dir: str

    def wait_for_endpoint_ready(self, url: str, timeout: int = 5400):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                time.sleep(10)
                if requests.get(url, timeout=5).status_code == 200:
                    print(f"endpoint {url} is ready")
                    return
            except Exception as err:
                print(f"endpoint {url} is not ready, with exception: {err}")
        print_error(
            f"Endpoint {url} did not become ready within {timeout} seconds")

    def run_cmd(self,
                cmd_idx: int,
                node_name: str,
                gpu_info: str,
                max_timeout: int = 5400) -> str:
        output = ""
        server_file_path = os.path.join(
            self.working_dir, f"trtllm-serve.{self.names[cmd_idx]}.log")
        client_file_path = os.path.join(
            self.working_dir, f"trtllm-benchmark.{self.names[cmd_idx]}.log")

        server_proc = None
        try:
            # Run server command
            with open(server_file_path, 'w') as server_ctx:
                server_proc = subprocess.Popen(self.server_cmds[cmd_idx],
                                               stdout=server_ctx,
                                               stderr=subprocess.STDOUT)

            # Wait for server to be ready
            self.wait_for_endpoint_ready("http://localhost:8000/v1/models",
                                         timeout=max_timeout)

            # Save node name, gpu info, server config, client config output to server file path
            with open(client_file_path, 'w') as client_ctx:
                client_ctx.write(f"Node: {node_name}\n")
                client_ctx.write(f"GPU Info: {gpu_info}\n")
                client_ctx.write(f"Server-Config: {self.names[cmd_idx]}\n")
                # Run client command
                subprocess.run(self.client_cmds[cmd_idx],
                               stdout=client_ctx,
                               stderr=subprocess.STDOUT,
                               check=True)
        finally:
            server_proc.terminate()
            server_proc.wait()

        return output

    def get_cmd_str(self, cmd_idx) -> List[str]:
        return ["server-benchmark tests, please check config files"]


def run_perf_tests(server_configs: List[ServerConfig],
                   server_client_configs: Dict[int, List[ClientConfig]],
                   max_timeout: int, working_dir: str, node_name: str,
                   gpu_info: str) -> None:
    """Main function to run all benchmarks from config file"""

    server_cmds, client_cmds, names = get_trtllm_server_client_commands(
        server_configs, server_client_configs, working_dir)
    commands = PerfServerBenchmarkCmds(server_cmds=server_cmds,
                                       client_cmds=client_cmds,
                                       names=names,
                                       working_dir=working_dir)

    # Run each server config based on execution plan
    for cmd_idx in range(len(client_cmds)):
        print(f"Server cmd: {server_cmds[cmd_idx]}")
        print(f"Client cmd: {client_cmds[cmd_idx]}")
        commands.run_cmd(cmd_idx, node_name, gpu_info, max_timeout)


def generate_repro_scripts(server_configs: List[ServerConfig],
                           server_client_configs: Dict[int, List[ClientConfig]],
                           node_name: str, max_timeout: int):
    """Generate reproduction scripts for all server configs"""
    for server_id, server_config in enumerate(server_configs):
        script_content = generate_repro_script(server_config,
                                               server_client_configs[server_id],
                                               node_name, max_timeout)
        script_filename = f"reproduce.server-{server_config.name}.sh"

        with open(script_filename, 'w') as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_filename, 0o755)


def generate_repro_script(server_config: ServerConfig,
                          client_configs: List[ClientConfig], node_name: str,
                          max_timeout: int) -> str:
    """Generate a shell script to reproduce a server config"""
    model_path = server_config.model_path
    script_content = f"""#!/bin/bash
# Reproduction script for server: {server_config.name})
# Node: {node_name}

set -e

# Function to wait for server to be ready
wait_for_server() {{
    local timeout={max_timeout}
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
CONFIG_FILENAME="extra-llm-api-config.{server_config.name}.yml"

cat > "$CONFIG_FILENAME" << 'EOF'
{server_config.generate_extra_llm_api_config()}
EOF

# Start trtllm-serve in background
SERVER_LOG="trtllm-serve.{server_config.name}.log"

echo "Starting trtllm-serve with command:"
echo "trtllm-serve {model_path} --host localhost --port 8000 --backend pytorch --extra_llm_api_options $CONFIG_FILENAME"

trtllm-serve {model_path} --host localhost --port 8000 --backend pytorch --extra_llm_api_options "$CONFIG_FILENAME" > "$SERVER_LOG" 2>&1 &

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
    for client_config in client_configs:
        num_prompts = client_config.concurrency * client_config.iterations
        script_content += f"""
echo "Running benchmark with concurrency: {client_config.concurrency}, iterations: {client_config.iterations}, num-prompts: {num_prompts}"

BENCHMARK_LOG="trtllm-benchmark.{server_config.name}.{client_config.name}.log"

echo "Running benchmark with command:"
echo "python -m tensorrt_llm.serve.scripts.benchmark_serving --model {model_path} --dataset-name random --random-ids --num-prompts {num_prompts} --random-input-len {client_config.isl} --random-output-len {client_config.osl} --random-range-ratio {client_config.random_range_ratio} --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --max-concurrency {client_config.concurrency}"

python -m tensorrt_llm.serve.scripts.benchmark_serving \\
    --model {model_path} \\
    --dataset-name random \\
    --random-ids \\
    --num-prompts {num_prompts} \\
    --random-input-len {client_config.isl} \\
    --random-output-len {client_config.osl} \\
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


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks from YAML configuration file')
    parser.add_argument('--log_folder',
                        required=True,
                        help='Output folder for benchmark results')
    parser.add_argument('--config_file',
                        required=True,
                        help='Path to YAML configuration file')
    parser.add_argument(
        '--select', help='Select pattern: "r1_fp4_dep4:con1_iter1_1024_1024"')
    parser.add_argument('--timeout', help='Timeout in seconds', default=5400)

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

    if not os.path.exists(args.log_folder):
        print(f"Error: Output folder '{args.log_folder}' does not exist")
        sys.exit(1)

    # Capture system information
    node_name = get_node_name()
    gpu_info = get_gpu_info()

    log_folder = Path(args.log_folder)
    config_file = Path(args.config_file)

    default_max_timeout = 5400
    max_timeout = int(args.timeout) if args.timeout else default_max_timeout

    # Change to output directory
    os.chdir(log_folder)

    # Treat empty or "default" values as None (default behavior)
    select_pattern = None if not args.select or args.select.lower(
    ) == "default" else args.select

    execution_plan, server_configs, server_client_configs = parse_config_file(
        config_file, select_pattern)

    generate_repro_scripts(server_configs, server_client_configs, node_name,
                           max_timeout)

    run_perf_tests(server_configs, server_client_configs, max_timeout,
                   log_folder, node_name, gpu_info)


if __name__ == "__main__":
    main()
