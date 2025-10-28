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
"""
TensorRT LLM perf tests
"""
import os
import re
import shutil
import sys
from typing import Dict, List, NamedTuple

import pytest
import yaml
from defs.common import get_cpp_benchmark
from defs.trt_test_alternative import (is_linux, is_windows, print_info,
                                       print_warning)

from ..conftest import get_llm_root, llm_models_root, trt_environment
from .pytorch_model_config import get_model_yaml_config
from .utils import (AbstractPerfScriptTestClass, PerfBenchScriptTestCmds,
                    PerfDisaggScriptTestCmds, PerfMetricType,
                    PerfServerClientBenchmarkCmds, generate_test_nodes)

if not hasattr(re, "Pattern"):
    re.Pattern = type(re.compile(""))

ALLOWED_CONFIGS_CACHE = None  # Cache to avoid modifying sys.path many times.
MAP_BY_SOCKET = None

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
    "phi_4_multimodal_instruct_fp4_image":
    "multimodals/Phi-4-multimodal-instruct-FP4",
    "phi_4_multimodal_instruct_fp4_audio":
    "multimodals/Phi-4-multimodal-instruct-FP4",
    "phi_4_multimodal_instruct_fp8_image":
    "multimodals/Phi-4-multimodal-instruct-FP8",
    "phi_4_multimodal_instruct_fp8_audio":
    "multimodals/Phi-4-multimodal-instruct-FP8",
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
LORA_MODEL_PATH = {
    "llama_v2_13b":
    "llama-models-v2/chinese-llama-2-lora-13b",
    "mixtral_8x7b_0.1":
    "chinese-mixtral-lora",
    "llama_v3.1_8b_instruct_fp8":
    "lora/llama-3-chinese-8b-instruct-v2-lora/",
    "ministral_8b":
    "lora/ministral/Ministral-8B-Instruct-2410-Loras-Dummy",  # Dummy LoRA for Ministral
    "gemma_3_1b_it":
    "lora/gemma/gemma-3-1b-it-dummy-lora",  # Dummy LoRA for Gemma-3-1B-Instruct
    "phi_4_multimodal_instruct_image":
    "multimodals/Phi-4-multimodal-instruct/vision-lora",
    "phi_4_multimodal_instruct_audio":
    "multimodals/Phi-4-multimodal-instruct/speech-lora",
    "phi_4_multimodal_instruct_fp4_image":
    "multimodals/Phi-4-multimodal-instruct-FP4/vision-lora",
    "phi_4_multimodal_instruct_fp4_audio":
    "multimodals/Phi-4-multimodal-instruct-FP4/speech-lora",
    "phi_4_multimodal_instruct_fp8_image":
    "multimodals/Phi-4-multimodal-instruct-FP8/vision-lora",
    "phi_4_multimodal_instruct_fp8_audio":
    "multimodals/Phi-4-multimodal-instruct-FP8/speech-lora",
}

TIMING_CACHE_DIR = os.environ.get("TIMING_CACHE_DIR", "")

TRUST_REMOTE_CODE_MODELS = {  # these models require explicit trust_remote_code=True
    "llama_v3.3_nemotron_super_49b",
    "llama_v3.3_nemotron_super_49b_fp8",
    "llama_v3.1_nemotron_ultra_253b",
    "llama_v3.1_nemotron_ultra_253b_fp8",
}


def get_model_dir(model_name: str):
    model_dir = ""
    if model_name in MODEL_PATH_DICT.keys():
        model_dir = os.path.join(llm_models_root(), MODEL_PATH_DICT[model_name])
    elif model_name in HF_MODEL_PATH.keys():
        model_dir = os.path.join(llm_models_root(),
                                 MODEL_PATH_DICT[model_name.split('_hf')[0]])
    return model_dir


def cpu_socket_count_gt_1():
    global MAP_BY_SOCKET
    if MAP_BY_SOCKET is not None:
        return MAP_BY_SOCKET
    if is_linux():
        with open('/proc/cpuinfo') as f:
            cpuinfo = f.read()
            physical_id_set = set()
            for line in cpuinfo.splitlines():
                if line.startswith('physical id'):
                    _, id_ = line.split(':')
                    physical_id_set.add(id_.strip())
        MAP_BY_SOCKET = len(physical_id_set) > 1
    else:
        MAP_BY_SOCKET = False
    return MAP_BY_SOCKET


# A helper function to import allowed_configs.py.
def import_allowed_perf_config():
    if trt_environment:
        from llm import allowed_configs
    else:
        global ALLOWED_CONFIGS_CACHE
        if ALLOWED_CONFIGS_CACHE is None:
            sys.path.append((os.path.join(get_llm_root(),
                                          "tests/integration/defs/perf")))
            import allowed_configs
            ALLOWED_CONFIGS_CACHE = allowed_configs
        else:
            allowed_configs = ALLOWED_CONFIGS_CACHE
    return allowed_configs


# Regex commands used to parse the metric result for the metric type.
PERF_METRIC_LOG_QUERIES = {
    PerfMetricType.BUILD_TIME:
    re.compile(r"Engine generation completed in ([\d\.]+) seconds"),
    PerfMetricType.INFERENCE_TIME:
    re.compile(r"\[BENCHMARK\].* (?:total_latency|latency)\(ms\) ([\d\.]+)"),
    PerfMetricType.FIRST_TOKEN_TIME:
    re.compile(r"\[BENCHMARK\].* avg_time_to_first_token\(ms\) ([\d\.]+)"),
    PerfMetricType.SEQ_LATENCY:
    re.compile(r"\[BENCHMARK\].* avg_sequence_latency\(ms\) ([\d\.]+)"),
    PerfMetricType.SEQ_THROUGHPUT:
    re.compile(r"\[BENCHMARK\].* seq_throughput\(seq\/sec\) ([\d\.]+)"),
    PerfMetricType.TOKEN_THROUGHPUT:
    re.compile(
        r"\[BENCHMARK\].* (?:token_throughput\(token\/sec\)|tokensPerSec|tokens_per_sec) ([\d\.]+)"
    ),
    PerfMetricType.INFERENCE_PEAK_GPU_MEMORY:
    re.compile(r"\[BENCHMARK\].* gpu_peak_mem\(gb\) ([\d\.]+)"),
    PerfMetricType.BUILD_PEAK_CPU_MEMORY:
    re.compile(
        r"Peak memory usage during Engine building and serialization: CPU: ([\d\.]+) .*"
    ),
    PerfMetricType.BUILD_PEAK_GPU_MEMORY:
    re.compile(
        r"Peak memory usage of TRT CPU/GPU memory allocators: CPU .*, GPU ([\d\.]+) .*"
    ),
    PerfMetricType.ENGINE_SIZE:
    re.compile(r".*Total engine size per GPU is ([\d\.]+) MiB.*"),
    PerfMetricType.CONTEXT_GPU_MEMORY:
    re.compile(r".*Allocated ([\d\.]+) MiB for execution context memory.*"),
    PerfMetricType.KV_CACHE_SIZE:
    re.compile(r".*Allocated ([\d\.]+) GiB for max tokens in paged KV cache.*"),
    PerfMetricType.DISAGG_SERVER_E2EL:
    re.compile(r"Median E2EL \(ms\):\s*(\d+\.?\d*)"),
    PerfMetricType.DISAGG_SERVER_TTFT:
    re.compile(r"Median TTFT \(ms\):\s*(\d+\.?\d*)"),
}

BENCH_PERF_METRIC_LOG_QUERIES = {
    PerfMetricType.BUILD_TIME:
    re.compile(r"Engine generation completed in ([\d\.]+) seconds"),
    PerfMetricType.INFERENCE_TIME:
    re.compile(r"Total Latency \(ms\):\s+([\d\.]+)"),
    PerfMetricType.TOKEN_THROUGHPUT:
    re.compile(r"GPU Output Throughput \(tps\/gpu\):\s+([\d\.]+)"),
    PerfMetricType.SEQ_THROUGHPUT:
    re.compile(r"Request Throughput \(req\/sec\):\s+([\d\.]+)"),
    PerfMetricType.FIRST_TOKEN_TIME:
    re.compile(r"Average time-to-first-token \[TTFT\] \(ms\):\s+([\d\.]+)"),
    PerfMetricType.OUTPUT_TOKEN_TIME:
    re.compile(r"Average time-per-output-token \[TPOT\] \(ms\):\s+([\d\.]+)"),
    PerfMetricType.KV_CACHE_SIZE:
    re.compile(r".*(?:Allocated ([\d\.]+) GiB for max tokens in paged KV cache|"
               r"Final KV cache size after resize: ([\d\.]+) GiB).*"),
}

SERVER_BENCHMARK_PERF_METRIC_LOG_QUERIES = {
    PerfMetricType.SEQ_THROUGHPUT:
    re.compile(r"Request throughput \(req\/s\):\s+([\d\.]+)"),
    PerfMetricType.TOKEN_THROUGHPUT:
    re.compile(r"Output token throughput \(tok\/s\):\s+([\d\.]+)"),
    PerfMetricType.TOTAL_TOKEN_THROUGHPUT:
    re.compile(r"Total Token throughput \(tok\/s\):\s+([\d\.]+)"),
    PerfMetricType.USER_THROUGHPUT:
    re.compile(r"User throughput \(tok\/s\):\s+([\d\.]+)"),
    PerfMetricType.FIRST_TOKEN_TIME:
    re.compile(r"Mean TTFT \(ms\):\s+([\d\.]+)"),
    PerfMetricType.MEDIAN_FIRST_TOKEN_TIME:
    re.compile(r"Median TTFT \(ms\):\s+([\d\.]+)"),
    PerfMetricType.P99_FIRST_TOKEN_TIME:
    re.compile(r"P99 TTFT \(ms\):\s+([\d\.]+)"),
    PerfMetricType.INTER_TOKEN_TIME:
    re.compile(r"Mean ITL \(ms\):\s+([\d\.]+)"),
    PerfMetricType.MEDIAN_INTER_TOKEN_TIME:
    re.compile(r"Median ITL \(ms\):\s+([\d\.]+)"),
    PerfMetricType.P99_INTER_TOKEN_TIME:
    re.compile(r"P99 ITL \(ms\):\s+([\d\.]+)"),
    PerfMetricType.OUTPUT_TOKEN_TIME:
    re.compile(r"Mean TPOT \(ms\):\s+([\d\.]+)"),
    PerfMetricType.MEDIAN_OUTPUT_TOKEN_TIME:
    re.compile(r"Median TPOT \(ms\):\s+([\d\.]+)"),
    PerfMetricType.P99_OUTPUT_TOKEN_TIME:
    re.compile(r"P99 TPOT \(ms\):\s+([\d\.]+)"),
    PerfMetricType.INFERENCE_TIME:
    re.compile(r"Mean E2EL \(ms\):\s+([\d\.]+)"),
    PerfMetricType.MEDIAN_INFERENCE_TIME:
    re.compile(r"Median E2EL \(ms\):\s+([\d\.]+)"),
    PerfMetricType.P99_INFERENCE_TIME:
    re.compile(r"P99 E2EL \(ms\):\s+([\d\.]+)"),
}

DISAGG_SERVER_METRICS_LOG_QUERIES = {
    PerfMetricType.DISAGG_SERVER_E2EL:
    re.compile(r"Median E2EL \(ms\):\s*(\d+\.?\d*)"),
    PerfMetricType.DISAGG_SERVER_TTFT:
    re.compile(r"Median TTFT \(ms\):\s*(\d+\.?\d*)"),
}

# (Relative threshold, Absolute threshold) for all metric types
PERF_METRIC_THRESHOLD = {
    PerfMetricType.BUILD_TIME: (0.1, 30),  # Ignore build time regression < 30ms
    PerfMetricType.INFERENCE_TIME:
    (0.1, 50),  # Ignore inference time regression < 50ms
    PerfMetricType.MEDIAN_INFERENCE_TIME:
    (0.1, 50),  # Ignore median inference time regression < 50ms
    PerfMetricType.P99_INFERENCE_TIME:
    (0.1, 50),  # Ignore p99 inference time regression < 50ms
    PerfMetricType.FIRST_TOKEN_TIME:
    (0.1, 50),  # Ignore first token time regression < 50ms
    PerfMetricType.MEDIAN_FIRST_TOKEN_TIME:
    (0.1, 50),  # Ignore median first token time regression < 50ms
    PerfMetricType.P99_FIRST_TOKEN_TIME:
    (0.1, 50),  # Ignore p99 first token time regression < 50ms
    PerfMetricType.OUTPUT_TOKEN_TIME:
    (0.1, 50),  # Ignore per output token time regression < 50ms
    PerfMetricType.MEDIAN_OUTPUT_TOKEN_TIME:
    (0.1, 50),  # Ignore median output token time regression < 50ms
    PerfMetricType.P99_OUTPUT_TOKEN_TIME:
    (0.1, 50),  # Ignore p99 output token time regression < 50ms
    PerfMetricType.INTER_TOKEN_TIME:
    (0.1, 50),  # Ignore inter token time regression < 50ms
    PerfMetricType.MEDIAN_INTER_TOKEN_TIME:
    (0.1, 50),  # Ignore median inter token time regression < 50ms
    PerfMetricType.P99_INTER_TOKEN_TIME:
    (0.1, 50),  # Ignore p99 inter token time regression < 50ms
    PerfMetricType.SEQ_LATENCY: (0.1, 50),  # Ignore latency regression < 50ms
    PerfMetricType.TOKEN_THROUGHPUT: (
        -0.1, 10
    ),  # Ignore throughput regression < 10 tokens/s. Negative rel threshold is to indicate that larger is better.
    PerfMetricType.TOTAL_TOKEN_THROUGHPUT: (0.1, 10),
    PerfMetricType.USER_THROUGHPUT: (0.1, 10),
    PerfMetricType.SEQ_THROUGHPUT: (
        -0.1, 10
    ),  # Ignore throughput regression < 10 tokens/s. Negative rel threshold is to indicate that larger is better.
    PerfMetricType.INFERENCE_PEAK_GPU_MEMORY:
    (0.1, 0.1),  # Ignore inference peak gpu memory regression < 0.1GiB
    PerfMetricType.BUILD_PEAK_CPU_MEMORY:
    (0.1, 100),  # Ignore build peak cpu memory regression < 100MiB
    PerfMetricType.BUILD_PEAK_GPU_MEMORY:
    (0.1, 100),  # Ignore build peak gpu memory regression < 100MiB
    PerfMetricType.ENGINE_SIZE: (0.3,
                                 100),  # Ignore engine size regression < 100MiB
    PerfMetricType.CONTEXT_GPU_MEMORY:
    (0.1, 50),  # Ignore context GPU memory < 50MiB
    PerfMetricType.KV_CACHE_SIZE: (-0.1, 50),  # Ignore value < 50MiB
    PerfMetricType.DISAGG_SERVER_E2EL: (0.1,
                                        50),  # Ignore E2EL regression < 50ms
    PerfMetricType.DISAGG_SERVER_TTFT: (0.1,
                                        50),  # Ignore TTFT regression < 50ms
}

BUILDER_METRICS = [
    PerfMetricType.BUILD_TIME, PerfMetricType.BUILD_PEAK_CPU_MEMORY,
    PerfMetricType.BUILD_PEAK_GPU_MEMORY, PerfMetricType.ENGINE_SIZE
]

INFERENCE_METRICS = [
    PerfMetricType.INFERENCE_TIME,
    PerfMetricType.INFERENCE_PEAK_GPU_MEMORY,
    PerfMetricType.CONTEXT_GPU_MEMORY,
]

SERVER_BENCHMARK_METRICS = [
    PerfMetricType.SEQ_THROUGHPUT,
    PerfMetricType.TOKEN_THROUGHPUT,
    PerfMetricType.TOTAL_TOKEN_THROUGHPUT,
    PerfMetricType.USER_THROUGHPUT,
    PerfMetricType.FIRST_TOKEN_TIME,
    PerfMetricType.MEDIAN_FIRST_TOKEN_TIME,
    PerfMetricType.P99_FIRST_TOKEN_TIME,
    PerfMetricType.OUTPUT_TOKEN_TIME,
    PerfMetricType.MEDIAN_OUTPUT_TOKEN_TIME,
    PerfMetricType.P99_OUTPUT_TOKEN_TIME,
    PerfMetricType.INTER_TOKEN_TIME,
    PerfMetricType.MEDIAN_INTER_TOKEN_TIME,
    PerfMetricType.P99_INTER_TOKEN_TIME,
    PerfMetricType.INFERENCE_TIME,
    PerfMetricType.MEDIAN_INFERENCE_TIME,
    PerfMetricType.P99_INFERENCE_TIME,
]

BENCH_INFERENCE_METRICS = [
    PerfMetricType.INFERENCE_TIME,
    PerfMetricType.TOKEN_THROUGHPUT,
    PerfMetricType.SEQ_THROUGHPUT,
    PerfMetricType.KV_CACHE_SIZE,
]

DISAGG_SERVER_METRICS = [
    PerfMetricType.DISAGG_SERVER_E2EL,
    PerfMetricType.DISAGG_SERVER_TTFT,
]


class PerfTestMetric(NamedTuple):
    """
    Configurations of a test metric.
    """
    # The original test name used to run the oraginal perf test.
    original_test_name: str
    # The name for this particular metric.
    metric_name: str
    # The type of this metric.
    metric_type: PerfMetricType
    # The regex used to parse this metric.
    metric_regex: re.Pattern
    # The relative threshold to allow for regressions.
    metric_threshold: float
    # The absolute threshold to allow for regressions.
    metric_abs_threshold: float
    # The index of the command of this metric.
    # Currently, we run 1 build command plus N benchmark commands.
    cmd_idx: int


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
            stream_interval=server_config_data.get('stream_interval', 10),
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


class PerfTestConfig:
    """
    Configurations defining the LLM perf test.
    This should hold only the attributes that distinguish different tests.
    """

    def __init__(
        self,
        *,
        model_name: str = "",
        runtime: str = "python",
        static_batching: str = "",
        api: str = "",
        streaming: str = "",
        backend: str = "",
        mode: str = "plugin",
        data_type: str = "float16",
        max_batch_size: int = 512,
        max_num_tokens: int = 2048,
        gpu_weights_percent: float = -1,
        batch_sizes: List[int] = [0],
        input_lens: List[int] = [8],
        output_lens: List[int] = [1],
        num_beams: int = 1,
        num_loras: int = 0,
        num_reqs: int = 512,
        concurrency: int = -1,
        quantization: str = "",
        kv_cache_free_gpu_mem_fraction: float = 0.9,
        kv_cache_dtype: str = "auto",
        ep_size: int = None,
        tp_size: int = 1,
        pp_size: int = 1,
        num_gpus: int = 1,
        # _autodeploy backend specific parameters
        ad_compile_backend: str = "torch-opt",
        free_mem_ratio: float = 0.9,
        extra_runtime: str = "trtllm",
        skip_loading_weights: bool = False,
    ):
        # The model name.
        self.model_name = model_name
        # Python or cpp/cppmanager runtime.
        self.runtime = runtime
        # static batching for gptManagerBenchmark
        self.static_batching = static_batching
        # API Type: only executor is allowed
        self.api = api
        # Backend Type: pytorch or cpp
        self.backend = backend
        # Streaming responses
        self.streaming = streaming
        # Plugin or OOTB mode.
        self.mode = mode
        # Activation dtype.
        self.data_type = data_type
        # Percentage of weights that resides on GPU.
        self.gpu_weights_percent = gpu_weights_percent
        # Max Batch Size to build TRT engine with.
        self.max_batch_size = max_batch_size
        # Max number of tokens to build TRT engine with.
        self.max_num_tokens = max_num_tokens
        # List of batch sizes to run benchmark with.
        self.batch_sizes = batch_sizes
        # List of input lens to run benchmark with.
        self.input_lens = input_lens
        # List of output lens to run benchmark with.
        self.output_lens = output_lens
        # Number of beams.
        self.num_beams = num_beams
        # Number of loras.
        self.num_loras = num_loras
        # Number of requests.
        self.num_reqs = num_reqs
        # Number of concurrency
        self.concurrency = concurrency
        # Quantization type.
        self.quantization = quantization
        # KV cache free gpu mem fraction
        self.kv_cache_free_gpu_mem_fraction = kv_cache_free_gpu_mem_fraction
        # KV Cache dtype
        self.kv_cache_dtype = kv_cache_dtype
        # Multiple Profiles
        self.multiple_profiles = False
        # EP Size
        self.ep_size = ep_size
        # TP Size
        self.tp_size = tp_size
        # PP Size
        self.pp_size = pp_size
        # Number of GPUs.
        self.num_gpus = num_gpus
        # _autodeploy backend specific parameters
        self.ad_compile_backend = ad_compile_backend
        self.free_mem_ratio = free_mem_ratio
        self.extra_runtime = extra_runtime
        self.skip_loading_weights = skip_loading_weights
        # Just build engines
        self.build_only = False

        # Whether to run disaggregated server perf test.
        self.is_disagg_server = False
        self.ctx_server_workers = 0
        self.gen_server_workers = 0

        # Used for perf sanity test
        # config_file: YAML path, select_pattern: server/client selection string
        # server_configs: list[ServerConfig], server_client_configs: dict[server_id -> list[ClientConfig]]
        self.config_file = None
        self.config_path = None
        self.select_pattern = None
        self.server_configs = []
        self.server_client_configs = {}

    def _to_string_disagg(self, entries: List[str]):
        entries.append(f"disagg_server")
        if self.ctx_tp_size > 1:
            entries.append(f"ctx_tp:{self.ctx_tp_size}")
        if self.ctx_dp_size > 1:
            entries.append(f"ctx_dp:{self.ctx_dp_size}")
        if self.ctx_pp_size > 1:
            entries.append(f"ctx_pp:{self.ctx_pp_size}")
        if self.gen_tp_size > 1:
            entries.append(f"gen_tp:{self.gen_tp_size}")
        if self.gen_dp_size > 1:
            entries.append(f"gen_dp:{self.gen_dp_size}")
        if self.gen_pp_size > 1:
            entries.append(f"gen_pp:{self.gen_pp_size}")
        return "-".join(entries)

    def to_string(self,
                  custom_server_name: str = None,
                  custom_client_name: str = None,
                  custom_bs: int = None,
                  custom_input_len: int = None,
                  custom_output_len: int = None,
                  device_subtype: str = None) -> str:

        # Used for perf sanity test
        if self.config_file is not None:
            entries = ["perf_sanity", self.config_file]
            if custom_server_name is not None:
                entries.append(f"server:{custom_server_name}")
            if custom_client_name is not None:
                entries.append(f"client:{custom_client_name}")
            return "-".join(entries)

        # First, add the model name.
        entries = [self.model_name]

        # Add device subtype if provided (for autodeploy tests)
        if device_subtype:
            entries.append(f"subtype:{device_subtype}")

        if self.runtime == "cpp":  # bertBenchmark runtime
            entries.append(f"cpp")
        elif self.runtime == "cppmanager":  # gptManagerBenchmark runtime
            entries.append(f"cppmanager")
            if self.api == "exe":  # executor
                entries.append(f"exe")
            if self.streaming == "streaming":
                entries.append(f"streaming")
            if self.static_batching == "static_batching":
                entries.append(f"static_batching")
        elif self.runtime == "bench":  # trtllm-bench
            entries.append(f"bench")
            if self.backend == 'pytorch':
                entries.append(f"pytorch")
            elif self.backend == '_autodeploy':
                entries.append(f"_autodeploy")
            if self.streaming == "streaming":
                entries.append(f"streaming")
        elif self.runtime == "disagg_server":  # trtllm-serve
            entries.append(f"disagg_server")
            return self._to_string_disagg(entries)

        # Add mode and dtype.
        if self.runtime != "bench":
            entries.append(self.mode)
        entries.append(self.data_type)

        if self.gpu_weights_percent != -1:
            entries.append(f"gwp:{self.gpu_weights_percent}")

        if self.multiple_profiles:
            entries.append(f"mp")

        # Add Max batch size.
        entries.append(f"maxbs:{self.max_batch_size}")

        # Add Max number of tokens.
        entries.append(f"maxnt:{self.max_num_tokens}")

        # Add kv cache free gpu mem fraction.
        if self.kv_cache_free_gpu_mem_fraction != 0.9:
            entries.append(f"kv_frac:{self.kv_cache_free_gpu_mem_fraction}")

        if self.build_only:
            entries.append(f"build_only")

        if self.batch_sizes[0] > 0:
            # Add batch size(s).
            if custom_bs is None:
                bs_label = "+".join([str(x) for x in self.batch_sizes])
            else:
                bs_label = str(custom_bs)
            entries.append(f"bs:{bs_label}")

        # Add input/output lens.
        if len(self.output_lens) > 0:
            if custom_input_len is None:
                io_lens = []
                for in_len, out_len in zip(self.input_lens, self.output_lens):
                    io_lens.append(f"{in_len},{out_len}")
                io_len_label = "+".join(io_lens)
            else:
                assert custom_output_len is not None, \
                    "custom_output_len must be provided if custom_input_len is specified!"
                io_len_label = f"{custom_input_len},{custom_output_len}"
            entries.append(f"input_output_len:{io_len_label}")
        else:
            if custom_input_len is None:
                len_label = "+".join([str(x) for x in self.input_lens])
            else:
                len_label = custom_input_len
            entries.append(f"input_len:{len_label}")

        # Add number of beams.
        if self.num_beams > 1:
            entries.append(f"beams:{self.num_beams}")

        # Add number of loras.
        if self.num_loras > 0:
            entries.append(f"loras:{self.num_loras}")

        # Add quantization type.
        if self.quantization != "":
            entries.append(f"quant:{self.quantization}")

        # Add kv cache dtype.
        if self.kv_cache_dtype != "auto":
            entries.append(f"kv_cache_dtype:{self.kv_cache_dtype}")

        # Add number of requests.
        if self.num_reqs != 512:
            entries.append(f"reqs:{self.num_reqs}")

        #Add number of concurrency
        if self.concurrency != -1:
            entries.append(f"con:{self.concurrency}")

        #Add EP Size.
        if self.ep_size != None:
            entries.append(f"ep:{self.ep_size}")

        # Add TP Size.
        if self.tp_size > 1 and self.tp_size != self.num_gpus:
            entries.append(f"tp:{self.tp_size}")

        # Add PP Size.
        if self.pp_size > 1:
            entries.append(f"pp:{self.pp_size}")

        # Add number of GPUs.
        if self.num_gpus > 1:
            entries.append(f"gpus:{self.num_gpus}")

        # Concatenate labels with "-".
        return "-".join(entries)

    def __str__(self) -> str:
        return self.to_string()

    def _load_from_str_disagg(self, labels: List[str]) -> None:
        self.ctx_tp_size = 1
        self.ctx_dp_size = 1
        self.ctx_pp_size = 1
        self.gen_tp_size = 1
        self.gen_dp_size = 1
        self.gen_pp_size = 1

        if labels[0].startswith("ctx_tp:"):
            self.ctx_tp_size = int(labels.pop(0).replace("ctx_tp:", ""))
        elif labels[0].startswith("ctx_dp:"):
            self.ctx_dp_size = int(labels.pop(0).replace("ctx_dp:", ""))
        elif labels[0].startswith("ctx_pp:"):
            self.ctx_pp_size = int(labels.pop(0).replace("ctx_pp:", ""))
        else:
            raise RuntimeError(f"Wrong label for ctx config: {labels[0]}!")

        if labels[0].startswith("gen_tp:"):
            self.gen_tp_size = int(labels.pop(0).replace("gen_tp:", ""))
        elif labels[0].startswith("gen_dp:"):
            self.gen_dp_size = int(labels.pop(0).replace("gen_dp:", ""))
        elif labels[0].startswith("gen_pp:"):
            self.gen_pp_size = int(labels.pop(0).replace("gen_pp:", ""))
        else:
            raise RuntimeError(f"Wrong label for gen config: {labels[0]}!")

        self.ctx_server_workers = self.ctx_tp_size * self.ctx_dp_size * self.ctx_pp_size
        self.gen_server_workers = self.gen_tp_size * self.gen_dp_size * self.gen_pp_size

        self.validate()

    def load_from_str(self, test_param_labels) -> None:
        """
        Populate the config properties given the test param string.
        """

        # Extract configs from test param labels.
        labels = test_param_labels.split("-")

        # Used for perf sanity test
        if labels[0] == "perf_sanity":
            assert len(labels) > 1, "perf_sanity test must have a config file!"
            self.runtime = "server-benchmark"
            self.config_file = labels[1]
            self.config_path = os.path.join(
                "tests/scripts/perf-sanity", f"{labels[1]}.yaml"
                if not labels[1].endswith(".yaml") else labels[1])
            self.select_pattern = labels[2] if len(labels) > 2 else None
            return

        self.model_name = labels.pop(0)

        # Check if device subtype is present (for autodeploy tests)
        self.device_subtype = None
        if len(labels) > 0 and labels[0].startswith("subtype:"):
            self.device_subtype = labels.pop(0).replace("subtype:", "")

        assert labels[0] in ["cpp", "cppmanager", "bench", "disagg_server"], \
            f"Invalid runtime {labels[0]}!"
        self.runtime = labels.pop(0)

        if self.runtime == "disagg_server":
            return self._load_from_str_disagg(labels)

        self.api = labels.pop(0) if labels[0] == "exe" else ""
        self.backend = labels.pop(0) if labels[0] in ["pytorch", "_autodeploy"
                                                      ] else ""
        self.streaming = labels.pop(0) if labels[0] == "streaming" else ""
        self.static_batching = labels.pop(
            0) if labels[0] == "static_batching" else ""
        if self.runtime != "bench":
            self.mode = labels.pop(0)
        self.data_type = labels.pop(0)
        if labels[0].startswith("gwp"):
            self.gpu_weights_percent = float(labels.pop(0).replace("gwp:", ""))

        if labels[0] == "mp":
            self.multiple_profiles = True
            labels.pop(0)

        if labels[0].startswith("maxbs"):
            self.max_batch_size = int(labels.pop(0).replace("maxbs:", ""))

        if labels[0].startswith("maxnt"):
            self.max_num_tokens = int(labels.pop(0).replace("maxnt:", ""))

        if labels[0].startswith("kv_frac"):
            self.kv_cache_free_gpu_mem_fraction = float(
                labels.pop(0).replace("kv_frac:", ""))

        if labels[0] == "build_only":
            self.build_only = True
            labels.pop(0)

        if not self.build_only:
            if labels[0].startswith("bs:"):
                self.batch_sizes = [
                    int(x) for x in labels.pop(0).replace("bs:", "").split("+")
                ]
            else:
                self.batch_sizes = [0]

            if labels[0].startswith("input_output_len"):
                io_lens = labels.pop(0).replace("input_output_len:",
                                                "").split("+")
                self.input_lens = [int(x.split(",")[0]) for x in io_lens]
                self.output_lens = [int(x.split(",")[1]) for x in io_lens]
            elif labels[0].startswith("input_len"):
                self.input_lens = [
                    int(x)
                    for x in labels.pop(0).replace("input_len:", "").split("+")
                ]
                self.output_lens = []
            else:
                raise RuntimeError(
                    f"Unexpected test name label for seq lens: {labels[0]}!")

        if len(labels) > 0:
            self.num_beams = 1 if not labels[0].startswith("beams:") else int(
                labels.pop(0).replace("beams:", ""))

        if len(labels) > 0:
            self.num_loras = 0 if not labels[0].startswith("loras:") else int(
                labels.pop(0).replace("loras:", ""))

        if len(labels) > 0:
            self.quantization = "" if not labels[0].startswith(
                "quant:") else labels.pop(0).replace("quant:", "")

        if len(labels) > 0:
            self.kv_cache_dtype = "auto" if not labels[0].startswith(
                "kv_cache_dtype:") else labels.pop(0).replace(
                    "kv_cache_dtype:", "")

        if len(labels) > 0:
            self.num_reqs = 512 if not labels[0].startswith("reqs:") else int(
                labels.pop(0).replace("reqs:", ""))

        if len(labels) > 0:
            self.concurrency = -1 if not labels[0].startswith("con:") else int(
                labels.pop(0).replace("con:", ""))

        if len(labels) > 0:
            self.ep_size = None if not labels[0].startswith("ep:") else int(
                labels.pop(0).replace("ep:", ""))

        if len(labels) > 0:
            self.tp_size = 1 if not labels[0].startswith("tp:") else int(
                labels.pop(0).replace("tp:", ""))

        if len(labels) > 0:
            self.pp_size = 1 if not labels[0].startswith("pp:") else int(
                labels.pop(0).replace("pp:", ""))

        if len(labels) > 0:
            self.num_gpus = 1 if not labels[0].startswith("gpus:") else int(
                labels.pop(0).replace("gpus:", ""))

        assert len(
            labels
        ) == 0, f"Invalid test name! Some labels cannot be parsed: {labels}"

        # Validate the parsed config.
        self.validate()

    def validate(self):
        """
        Validate if the config makes sense.
        """
        # Validate model name.
        assert len(self.model_name) > 0, "model_name must not be empty!"
        assert "-" not in self.model_name, "model_name must not contain '-' character!"
        if self.model_name not in MODEL_PATH_DICT.keys(
        ) and self.model_name not in HF_MODEL_PATH.keys():
            allowed_configs = import_allowed_perf_config()
            allowed_models = allowed_configs.get_allowed_models()
            assert self.model_name in allowed_models, f"model_name {self.model_name} is not in allowed_models!"

        # Validate runtime type.
        VALID_RUNTIMES = ["cpp", "cppmanager", "bench", "disagg_server"]
        assert self.runtime in VALID_RUNTIMES, f"Invalid runtime {self.runtime}!"

        if self.runtime == "disagg_server":
            # TODO: validate disaggregated server config
            return

        # Validate plugin mode.
        VALID_MODES = ["plugin", "ootb", "ootb_except_mha"]
        if self.runtime == "cppmanager":
            VALID_MODES += ["plugin_ifb"]
        assert self.mode in VALID_MODES, f"Invalid mode {self.mode}!"

        # Validate dtype.
        VALID_DTYPES = ["float32", "float16", "bfloat16", "float8", "float4"]
        assert self.data_type in VALID_DTYPES, f"Invalid data_type {self.data_type}!"
        VALID_KV_CACHE_DTYPES = ["auto", "fp8"]
        assert self.kv_cache_dtype in VALID_KV_CACHE_DTYPES, f"Invalid kv_cache_dtype {self.kv_cache_dtype}!"

        # Validate quantization mode.
        if self.model_name in MODEL_PATH_DICT.keys():
            VALID_QUANTS = [
                "", "nvfp4", "fp8", "int8", "int4_awq", "w4a8_awq", "w4a16_awq",
                "int4_wo", "full_prec"
            ]
        else:
            VALID_QUANTS = [
                "",
                "fp8",
                "fp8_gemm",
                "fp8_kv_cache",
                "int8_sq_per_tensor",
                "int8_sq_per_token_channel",
                "int8_weight_only",
                "int4_weight_only",
                "int4_weight_only_awq",
                "int4_weight_only_gptq",
            ]
        assert self.quantization in VALID_QUANTS, f"Invalid quantization {self.quantization}!"
        if self.backend == "pytorch":
            assert self.quantization == "", f"Not support passing quantization {self.quantization} for pytorch backend!"
        assert self.num_beams >= 1, f"Invalid num_beams: {self.num_beams}!"
        assert self.num_loras >= 0, f"Invalid num_loras: {self.num_loras}!"
        assert self.num_reqs >= 1, f"Invalid num_reqs: {self.num_reqs}!"
        if self.pp_size > 1:
            assert self.model_name in MODEL_PATH_DICT.keys(
            ), f"Invalid model name for pp size {self.pp_size} test"
        if self.num_gpus > 1 and self.tp_size == 1 and self.pp_size == 1:
            self.tp_size = self.num_gpus

        if self.tp_size > 1 or self.pp_size > 1 and self.num_gpus == 1:
            self.num_gpus = self.tp_size * self.pp_size

        assert self.num_gpus == self.tp_size * self.pp_size, f"Num of GPU shall be equal to TP*PP: {self.num_gpus}, {self.tp_size}, {self.pp_size}"
        if self.gpu_weights_percent != -1:
            assert 0 <= self.gpu_weights_percent <= 1, f"Invalid gpu_weights_percent: {self.gpu_weights_percent}!"
        if not self.build_only:
            assert len(self.input_lens) > 0, f"Empty input_lens!"
            if self.is_bert_like():
                assert len(
                    self.output_lens
                ) == 0, f"BERT-like models must not have output_lens!"
            else:
                assert len(
                    self.output_lens
                ) > 0, f"GPT-like models and enc-dec models must have output_lens!"

            # BERT with small BS is very unstable. Try to avoid it.
            if self.is_bert_like():
                if self.runtime == "trtllm-bench":
                    self.batch_sizes[
                        0] = self.max_batch_size if self.max_batch_size > 0 else 1
                    print(f"batch_sizes: {self.batch_sizes}")
                assert all(
                    [b >= 32 for b in self.batch_sizes]
                ), f"BERT with small BS is very unstable! Please increase to at least 32."

            # GPT-350m and Bloom-560m with small BS are very unstable. Only run these small models with larger BS.
            if self.model_name in ["gpt_350m", "bloom_560m"]:
                assert all(
                    [b >= 32 for b in self.batch_sizes]
                ), f"gpt_350m and bloom_560m with small BS are very unstable! Please increase to at least 32."

    def set_server_client_configs(self, llm_root: str) -> None:
        """
        Set the server and client configs.
        """
        if self.runtime == "server-benchmark":
            config_file_path = os.path.join(llm_root, self.config_path)
            _, self.server_configs, self.server_client_configs = parse_config_file(
                config_file_path, self.select_pattern)

    def get_model_family(self) -> str:
        """
        Get the model family of the current model.
        """
        allowed_configs = import_allowed_perf_config()
        allowed_models = allowed_configs.get_allowed_models()
        if self.model_name in allowed_models:
            return allowed_configs.get_model_family(self.model_name)
        else:
            return ""

    def is_mamba_family(self) -> bool:
        """
        Check if the current model family is Mamba.
        """
        return self.get_model_family() == 'mamba'

    def is_moe_family(self) -> bool:
        """
        Check if the current model family is MoE.
        """
        allowed_configs = import_allowed_perf_config()
        allowed_models = allowed_configs.get_allowed_models()
        if self.model_name in allowed_models:
            model_config = allowed_configs.get_model_config(self.model_name)
            return model_config['moe_num_experts'] > 0 and model_config[
                'moe_top_k'] > 0
        else:
            return False

    def get_benchmark_type(self) -> str:
        """
        Get the benchmark type of the current model.
        """
        allowed_configs = import_allowed_perf_config()
        allowed_models = allowed_configs.get_allowed_models()
        if self.model_name in allowed_models:
            return allowed_configs.get_benchmark_type(self.model_name)
        else:
            return ""

    def is_bert_like(self) -> bool:
        """
        Check if the current benchmark is a BERT benchmark.
        """
        return self.get_benchmark_type() == "bert"

    def is_enc_dec(self) -> bool:
        """
        Check if the current benchmark is a EncDec benchmark.
        """
        return self.get_benchmark_type() == "enc_dec"


class MultiMetricPerfTest(AbstractPerfScriptTestClass):
    """
    Base class for perf tests with multiple metrics.
    """

    def __init__(self, full_test_name: str):
        # full_test_name is the full test name appearing in test output.
        self._full_test_name = full_test_name
        # test_domain_name is the part before "::".
        self._test_domain_name = "::".join(full_test_name.split("::")[:-1])
        # short_test_name is the part after "::".
        self._short_test_name = full_test_name.split("::")[-1]
        # short_test_name_body is the part before "[" in short_test_name.
        self._short_test_name_body = self._short_test_name.split("[")[0]
        # test_param_labels is the part inside "[...]".
        self._test_param_labels = full_test_name.split("[")[-1][:-1]
        # Load test config from test name.
        self._config = PerfTestConfig()
        self._config.load_from_str(self._test_param_labels)
        # This will store the currently running metric.
        self._current_metric = None
        self.lora_dirs = []

    def get_test_name(self) -> str:
        return str(self._config)

    def set_runtime_configs(self,
                            llm_root,
                            working_dir,
                            perf_cache_fpath,
                            gpu_clock_lock=None) -> None:
        if self._config.runtime == "cpp":
            if not self._config.is_bert_like():
                raise ValueError(
                    f"Invalid config: '{self._config.runtime}' is only supported for bert-like models!"
                )
            benchmark_script = get_cpp_benchmark("bertBenchmark", llm_root)
        elif self._config.runtime == "cppmanager":
            benchmark_script = get_cpp_benchmark("gptManagerBenchmark",
                                                 llm_root)
        elif self._config.runtime == "bench":
            benchmark_script = "trtllm-bench"
        elif self._config.runtime == "server-benchmark":
            benchmark_script = None
            self._config.set_server_client_configs(llm_root)
        elif self._config.runtime == "disagg_server":
            benchmark_script = None
        else:
            raise RuntimeError(f"Invalid runtime {self._config.runtime}.")

        allowed_configs = import_allowed_perf_config()
        allowed_models = allowed_configs.get_allowed_models()

        if self._config.runtime == "bench":
            build_script = "trtllm-bench"
        elif self._config.runtime == "server-benchmark":
            build_script = None
        elif self._config.pp_size > 1 or self._config.model_name not in allowed_models:
            build_script = "trtllm-build"
        else:
            # build.py is used to build engines for both python and cpp runtime
            build_script = os.path.join(llm_root,
                                        "tests/integration/defs/perf/build.py")

        self._build_script = build_script
        self._benchmark_script = benchmark_script
        self._working_dir = working_dir
        self._perf_cache_fpath = perf_cache_fpath
        self._llm_root = llm_root
        self._gpu_clock_lock = gpu_clock_lock

    def get_trtllm_server_client_commands(self):
        server_cmds = []
        client_cmds = []
        names = []
        for server_idx, client_configs in self._config.server_client_configs.items(
        ):
            server_config = self._config.server_configs[server_idx]
            server_cmd = server_config.to_cmd(self._working_dir)
            server_cmd = " ".join(server_cmd)
            # Generate extra-llm-api-config.yml
            config_content = server_config.generate_extra_llm_api_config()
            config_filename = f"extra-llm-api-config.{server_config.name}.yml"
            config_path = os.path.join(self._working_dir, config_filename)
            with open(config_path, 'w') as f:
                f.write(config_content)
            for client_config in client_configs:
                server_cmds.append(server_cmd)
                client_cmd = client_config.to_cmd(self._working_dir)
                client_cmds.append(client_cmd)
                names.append(f"{server_config.name}-{client_config.name}")
        return server_cmds, client_cmds, names

    def get_trtllm_build_command(self, engine_dir, checkpoint_dir) -> list:
        build_cmd = [
            self._build_script, f"--output_dir={engine_dir}",
            f"--checkpoint_dir={checkpoint_dir}",
            f"--workers={self._config.tp_size}",
            f"--use_paged_context_fmha=enable", f"--monitor_memory",
            f"--max_batch_size={self._config.max_batch_size}"
        ]
        # For Multiple Profiles
        if self._config.multiple_profiles:
            build_cmd.append(f"--multiple_profiles=enable")
        else:
            build_cmd.append(f"--multiple_profiles=disable")
        num_beams = self._config.num_beams
        if num_beams > 1:
            build_cmd.append(f"--max_beam_width={num_beams}")
        gpu_percent = self._config.gpu_weights_percent
        if gpu_percent != -1:
            build_cmd += [f"--weight_streaming"]
        # For engine inspector
        build_cmd.append("--profiling_verbosity=layer_names_only")
        if self._config.num_loras > 0:
            if "mixtral" in self._config.model_name:
                build_cmd.append(f"--lora_plugin=auto")
                build_cmd.append(f"--moe_plugin=auto")
                build_cmd.append(f"--lora_target_modules")
                build_cmd.append(f"attn_q")
                build_cmd.append(f"attn_k")
                build_cmd.append(f"attn_v")
                build_cmd.append(f"attn_dense")
                build_cmd.append(f"moe_h_to_4h")
                build_cmd.append(f"moe_4h_to_h")
                build_cmd.append(f"moe_gate")
                build_cmd.append(f"moe_router")
            elif "llama" in self._config.model_name:
                build_cmd.append(f"--lora_plugin=float16")
                build_cmd.append(f"--lora_target_modules")
                build_cmd.append(f"attn_q")
                build_cmd.append(f"attn_k")
                build_cmd.append(f"attn_v")
                build_cmd.append(f"attn_dense")
                build_cmd.append(f"mlp_h_to_4h")
                build_cmd.append(f"mlp_4h_to_h")
                build_cmd.append(f"mlp_gate")
        if TIMING_CACHE_DIR and not self._config.build_only:
            timing_cache = os.path.join(TIMING_CACHE_DIR, "model.cache")
            build_cmd.append(f"--input_timing_cache={timing_cache}")
            build_cmd.append(f"--output_timing_cache={timing_cache}")
        return build_cmd

    def get_trtllm_bench_model(self):
        return get_model_dir(self._config.model_name)

    def get_trtllm_bench_build_command(self, engine_dir) -> list:
        model_dir = self.get_trtllm_bench_model()
        if model_dir == "":
            pytest.skip("Model Name is not supported by trtllm-bench")
        model_name = self._config.model_name
        if not model_name.endswith("_hf"):
            model_name = model_name + "_hf"
        hf_model_name = HF_MODEL_PATH.get(model_name, "")
        build_cmd = [
            self._build_script, f"--log_level=info",
            f"--workspace={engine_dir}", f"--model={hf_model_name}",
            f"--model_path={model_dir}", "build",
            f"--tp_size={self._config.tp_size}",
            f"--pp_size={self._config.pp_size}"
        ]
        max_seq_len = max(self._config.input_lens) + max(
            self._config.output_lens)
        build_cmd.append(f"--max_seq_len={max_seq_len}")
        # Add max_batch_size and max_num_tokens to ensure build matches runtime configuration
        # Note: trtllm-bench requires both to be specified together (option group constraint)
        assert self._config.max_batch_size > 0, f"max_batch_size must be > 0, got {self._config.max_batch_size}"
        assert self._config.max_num_tokens > 0, f"max_num_tokens must be > 0, got {self._config.max_num_tokens}"
        build_cmd.append(f"--max_batch_size={self._config.max_batch_size}")
        build_cmd.append(f"--max_num_tokens={self._config.max_num_tokens}")
        if self._config.quantization:
            build_cmd.append(
                f"--quantization={self._config.quantization.upper()}")
        if self._config.model_name in TRUST_REMOTE_CODE_MODELS:
            build_cmd.append(f"--trust_remote_code=True")
        return build_cmd

    def get_prepare_data_command(self, engine_dir, input_len,
                                 output_len) -> list:
        data_cmd = []
        prepare_data_script = os.path.join(self._llm_root, "benchmarks", "cpp",
                                           "prepare_dataset.py")

        if self._config.model_name in MODEL_PATH_DICT.keys():
            tokenizer_dir = os.path.join(
                llm_models_root(), MODEL_PATH_DICT[self._config.model_name])
        elif self._config.model_name in HF_MODEL_PATH.keys():
            tokenizer_dir = HF_MODEL_PATH[self._config.model_name]
        else:
            tokenizer_dir = os.path.join(llm_models_root(), "llama-models",
                                         "llama-7b-hf")
        if not os.path.exists(engine_dir):
            os.makedirs(engine_dir, exist_ok=True)
        if self._config.num_loras > 0:
            istdev = 16
            ostdev = 24
            nloras = self._config.num_loras
            dataset_path = os.path.join(engine_dir, "synthetic_data.json")

            if self._config.model_name in LORA_MODEL_PATH.keys(
            ) and self._config.backend == "pytorch" and self._config.runtime == "bench":
                actual_lora_paths = LORA_MODEL_PATH[self._config.model_name]
                if not isinstance(actual_lora_paths, list):
                    actual_lora_paths = [actual_lora_paths]
                for i, actual_lora_path in enumerate(actual_lora_paths):
                    if not actual_lora_path.startswith("/"):
                        actual_lora_paths[i] = os.path.join(
                            llm_models_root(), actual_lora_path)
                lora_dir = os.path.join(engine_dir, "loras")
                data_cmd += [f"mkdir -p {lora_dir}", ";"]
                if len(actual_lora_paths) != nloras:
                    raise ValueError(
                        f"Number of LoRA paths ({len(actual_lora_paths)}) does not match requested number of LoRAs ({nloras})"
                    )
                for i, lora_path in enumerate(actual_lora_paths):
                    self.lora_dirs.append(f"{lora_dir}/{i}")
                    data_cmd += [f"ln -sf {lora_path} {lora_dir}/{i}", ";"]
                data_cmd += [
                    "python3", prepare_data_script, f"--stdout",
                    f"--rand-task-id 0 {nloras-1}",
                    f"--tokenizer={tokenizer_dir}", f"--lora-dir={lora_dir}",
                    f"token-norm-dist",
                    f"--num-requests={self._config.num_reqs}",
                    f"--input-mean={input_len}", f"--output-mean={output_len}",
                    f"--input-stdev={istdev}", f"--output-stdev={ostdev}",
                    f" > {dataset_path}"
                ]

            else:
                pytest.skip(
                    f"LoRA config not supported for {self._config.model_name} with the current backend and runtime."
                )
        else:
            istdev = 0
            ostdev = 0
            dataset_path = os.path.join(engine_dir, "synthetic_data.json")
            if self._build_script == 'trtllm-bench':
                data_cmd += [
                    "python3", prepare_data_script, "--stdout",
                    f"--tokenizer={tokenizer_dir}", f"token-norm-dist",
                    f"--num-requests={self._config.num_reqs}",
                    f"--input-mean={input_len}", f"--output-mean={output_len}",
                    f"--input-stdev={istdev}", f"--output-stdev={ostdev}",
                    f" > {dataset_path}"
                ]
            else:
                data_cmd += [
                    "python3", prepare_data_script, f"--output={dataset_path}",
                    f"--tokenizer={tokenizer_dir}", f"token-norm-dist",
                    f"--num-requests={self._config.num_reqs}",
                    f"--input-mean={input_len}", f"--output-mean={output_len}",
                    f"--input-stdev={istdev}", f"--output-stdev={ostdev}"
                ]

        return data_cmd

    def get_trtllm_bench_command(self, engine_dir):
        model_dir = self.get_trtllm_bench_model()
        model_name = self._config.model_name
        dataset_path = os.path.join(engine_dir, "synthetic_data.json")
        report_path = os.path.join(engine_dir, "report.json")
        if not model_name.endswith("_hf"):
            model_name = model_name + "_hf"
        hf_model_name = HF_MODEL_PATH.get(model_name, "")
        tp_pp_str = f"tp_{self._config.tp_size}_pp_{self._config.pp_size}"
        engine_dir = os.path.join(engine_dir, hf_model_name, tp_pp_str)
        benchmark_cmd = [
            self._benchmark_script,
            f"--model={model_name}",
            f"--model_path={model_dir}",
            "throughput",
            f"--dataset={dataset_path}",
            f"--max_batch_size={self._config.max_batch_size}",
            f"--max_num_tokens={self._config.max_num_tokens}",
            f"--report_json={report_path}",
            f"--kv_cache_free_gpu_mem_fraction={self._config.kv_cache_free_gpu_mem_fraction}",
        ]
        if self._config.backend == "pytorch":
            benchmark_cmd += ["--backend=pytorch"]
        elif self._config.backend == "_autodeploy":
            benchmark_cmd += ["--backend=_autodeploy"]
        else:
            benchmark_cmd += [
                f"--backend=tensorrt", f"--engine_dir={engine_dir}"
            ]
        if self._config.num_reqs > 0:
            benchmark_cmd += [f"--num_requests={self._config.num_reqs}"]
        if self._config.concurrency != -1:
            benchmark_cmd += [f"--concurrency={self._config.concurrency}"]
        if self._config.ep_size != None:
            benchmark_cmd += [f"--ep={self._config.ep_size}"]
        if self._config.tp_size > 1:
            benchmark_cmd += [f"--tp={self._config.tp_size}"]
        if self._config.pp_size > 1:
            benchmark_cmd += [f"--pp={self._config.pp_size}"]
        if self._config.streaming == "streaming":
            benchmark_cmd += [f"--streaming"]
        #use default yaml config
        if self._config.backend == "pytorch":
            import yaml
            pytorch_config_path = os.path.join(engine_dir,
                                               "extra-llm-api-config.yml")
            if not os.path.exists(pytorch_config_path):
                os.makedirs(os.path.dirname(pytorch_config_path), exist_ok=True)
            config = get_model_yaml_config(self._config.to_string(),
                                           lora_dirs=self.lora_dirs)
            print_info(f"pytorch model config: {config}")
            with open(pytorch_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            benchmark_cmd += [f"--extra_llm_api_options={pytorch_config_path}"]
        elif self._config.backend == "_autodeploy":
            import yaml
            autodeploy_config_path = os.path.join(engine_dir,
                                                  "extra_llm_api_options.yaml")
            if not os.path.exists(autodeploy_config_path):
                os.makedirs(os.path.dirname(autodeploy_config_path),
                            exist_ok=True)

            # Create _autodeploy specific configuration
            autodeploy_config = {
                'transforms': {
                    'compile_model': {
                        'backend': self._config.ad_compile_backend
                    },
                    'resize_kv_cache': {
                        'free_mem_ratio': self._config.free_mem_ratio
                    },
                },
                'runtime': self._config.extra_runtime,
                'skip_loading_weights': self._config.skip_loading_weights
            }

            print_info(f"_autodeploy model config: {autodeploy_config}")
            with open(autodeploy_config_path, 'w') as f:
                yaml.dump(autodeploy_config, f, default_flow_style=False)
            benchmark_cmd += [
                f"--extra_llm_api_options={autodeploy_config_path}"
            ]
        return benchmark_cmd

    def get_commands(self):

        # Whether this is python or cpp runtime perf test.
        is_python = self._config.runtime == "python"
        num_gpus = self._config.num_gpus
        is_server_benchmark = self._config.runtime == "server-benchmark"
        is_disagg = self._config.runtime == "disagg_server"

        if is_server_benchmark:
            perf_sanity_working_dir = os.path.join(self._working_dir,
                                                   "perf-sanity")
            if not os.path.exists(perf_sanity_working_dir):
                os.makedirs(perf_sanity_working_dir, exist_ok=True)
            server_cmds, client_cmds, names = self.get_trtllm_server_client_commands(
            )
            return PerfServerClientBenchmarkCmds(
                server_cmds=server_cmds,
                client_cmds=client_cmds,
                names=names,
                working_dir=perf_sanity_working_dir)

        if is_disagg:
            ctx_cmd, gen_cmd = self._get_disagg_worker_deploy_command()
            server_cmd = self._get_disagg_server_deploy_command()
            client_cmd = self._get_disagg_client_command()
            benchmark_cmd = self._get_disagg_benchmark_command()
            return PerfDisaggScriptTestCmds(ctx_cmd, gen_cmd, server_cmd,
                                            client_cmd, benchmark_cmd)

        if is_python and num_gpus > 1:
            # TODO: Fix https://nvbugs/4449875
            pytest.skip(
                "multi-gpu tests with python runtime is skipped because of hanging issue. See https://nvbugs/4449875"
            )
        if is_windows() and num_gpus > 1:
            pytest.skip(
                "multi-gpu not supported on Windows yet, skipped for now")

        # Construct engine build command.
        engine_dir = self._get_engine_dir()
        build_cmd = []
        if self._config.runtime == "bench":
            if self._config.backend in ["pytorch", "_autodeploy"]:
                # Skip building process as it is pytorch or _autodeploy backend")
                pass
            else:
                build_cmd = self.get_trtllm_bench_build_command(engine_dir)
        else:
            pytest.skip("only support trtllm-bench runtime for now")
        # Construct prepare synthetic data command
        data_cmds = []

        # Construct benchmark commands for each bs and seq len combination.
        benchmark_cmds = []
        for bs in self._config.batch_sizes:
            for len_idx, input_len in enumerate(self._config.input_lens):
                output_len = None if self._config.is_bert_like(
                ) else self._config.output_lens[len_idx]
                if self._config.runtime == "bench":
                    benchmark_cmd = self.get_trtllm_bench_command(engine_dir)
                else:
                    pytest.skip("only support trtllm-bench runtime for now")
                benchmark_cmds.append(benchmark_cmd)
                data_cmd = self.get_prepare_data_command(
                    engine_dir, input_len, output_len)
                data_cmds.append(data_cmd)

        # Construct MPI command.
        mpi_cmd = []
        if num_gpus > 1 and num_gpus <= 8 and not self._config.runtime == "bench":
            if cpu_socket_count_gt_1():
                mpi_cmd = [
                    "mpirun", "--map-by", "socket", "-n", f"{num_gpus}",
                    "--allow-run-as-root"
                ]
            else:
                mpi_cmd = ["mpirun", "-n", f"{num_gpus}", "--allow-run-as-root"]
        if self._build_script == "trtllm-bench":
            return PerfBenchScriptTestCmds(data_cmds, build_cmd, benchmark_cmds,
                                           mpi_cmd, is_python)
        else:
            pytest.skip("only support trtllm-bench runtime for now")

    def get_perf_result(self, outputs: Dict[int, str]) -> float:
        """
        Get perf metric result from test output logs.
        """
        metric = self._current_metric
        cmd_idx = metric.cmd_idx
        metric_name = metric.metric_name
        num_gpus = self._config.num_gpus

        # Make sure we have outputs.
        assert cmd_idx in outputs, f"Output log for command {cmd_idx} does not exist!"

        # Use all applicable regex patterns to go through the log from the N-th command, where N = cmd_idx.
        print_info(
            f"Searching for metric {metric_name} from output log of command {cmd_idx} ..."
        )

        regex_matches = [
            metric.metric_regex.search(line)
            for line in outputs[cmd_idx].split("\n")
        ]
        print_info(outputs[cmd_idx].split("\n"))
        metric_values = []
        for match in regex_matches:
            if match:
                # Handle multiple capture groups - use the first non-None group
                value = None
                for i in range(1, len(match.groups()) + 1):
                    if match.group(i) is not None:
                        value = match.group(i)
                        break
                if value is not None:
                    metric_values.append(float(value))

        if len(metric_values) == 0:
            if self._build_script == "trtllm-bench" and self._config.num_gpus > 1 and metric.metric_type == PerfMetricType.BUILD_TIME:
                print_info("skip building process for multi-gpu test"
                           )  #https://nvbugspro.nvidia.com/bug/5210111
                metric_values = [0.0]
            else:
                raise RuntimeError(
                    f"Cannot find perf result for {metric_name} from perf script logs!"
                )

        if metric.metric_type in BUILDER_METRICS and metric.metric_type != PerfMetricType.ENGINE_SIZE:
            # For enc-dec models, there are 2 builder perf metrics, we add them up.
            if self._config.is_enc_dec():
                assert len(
                    metric_values
                ) == 2 * num_gpus, f"Enc-Dec models must have num of metrics 2*{num_gpus} but got {len(metric_values)}!"

                enc_metrics = metric_values[:num_gpus]
                dec_metrics = metric_values[num_gpus:]
                gather_function = sum
                # Measure BUILD_PEAK_CPU_MEMORY, BUILD_PEAK_GPU_MEMORY by max function
                if metric.metric_type in [
                        PerfMetricType.BUILD_PEAK_CPU_MEMORY,
                        PerfMetricType.BUILD_PEAK_GPU_MEMORY
                ]:
                    gather_function = max

                metric_values = [
                    gather_function([x, y])
                    for x, y in zip(enc_metrics, dec_metrics)
                ]
                print_info(
                    f"Combining up enc builder_perf {enc_metrics} and dec builder_perf {dec_metrics} to {metric_values}."
                )
            # For other models, builder metric should equal # gpus.
            elif self._build_script != "trtllm-build" and self._build_script != "trtllm-bench":
                assert len(
                    metric_values
                ) == num_gpus, f"num of metrics: {len(metric_values)} should match num_gpus: {num_gpus}"

        # Use max perf metrics across GPUS
        if len(metric_values) > 1:
            metric_value = max(metric_values)
            print_info(
                f"Use max value {metric_value} out of {metric_values} for perf metric {metric_name}."
            )
        else:
            metric_value = metric_values[0]
            print_info(
                f"Use value {metric_value} for perf metric {metric_name}.")

        return metric_value

    def get_threshold(self) -> float:
        return self._current_metric.metric_threshold

    def get_absolute_threshold(self) -> float:
        return self._current_metric.metric_abs_threshold

    def get_metric_type(self) -> PerfMetricType:
        return self._current_metric.metric_type

    def run_metrics(self, llm_venv, gpu_clock_lock, session_data_writer,
                    output_dir):
        """
        Run through the commands and parse multiple perf metrics from the logs.
        """
        #print info to separate cases
        print_info(f"Running perf test for case: {self._short_test_name}")
        self._current_cmd_idx = 0
        metrics = self._get_metrics()
        outputs = {}
        result_states = {}
        errors = []

        def add_myelin_time_pass_to(input_env):
            time_pass_flag = r" -time_pass=on"
            old_myelin_env = input_env.get("__LUNOWUD", "")
            if time_pass_flag not in old_myelin_env:
                input_env["__LUNOWUD"] = old_myelin_env + time_pass_flag
            return old_myelin_env

        old_llm_venv = add_myelin_time_pass_to(llm_venv._new_env)
        if self._config.runtime == 'bench':
            #prepare dataset first for trtllm-bench
            print_info(f"Running command for generating dataset")
            outputs = self.run_ex("prepare_dataset",
                                  llm_venv,
                                  gpu_clock_lock,
                                  session_data_writer,
                                  output_dir,
                                  outputs=outputs,
                                  original_test_name="prepare_dataset",
                                  cmd_idx=self._current_cmd_idx)

            # Save the result state.
            result_state = self.get_result_state()
            result_states[self._current_cmd_idx] = result_state
            if result_state != "valid":
                errors.append(self.get_error())

        try:
            for metric in metrics:
                # Make sure that cmd_idx is in ascending order.
                assert metric.cmd_idx >= self._current_cmd_idx, "Command indices must be in ascending order!"
                self._current_cmd_idx = metric.cmd_idx
                self._current_metric = metric

                # If the same command has previously failed, do not run it again.
                if self._current_cmd_idx in result_states and result_states[
                        self._current_cmd_idx] == "failed":
                    print_warning(
                        f"Skipped running command for {metric.metric_name} since the previous run failed."
                    )
                    continue

                # If engine build command already failed, do not run benchmark commands.
                if 0 in result_states and result_states[0] == "failed":
                    print_warning(
                        f"Skipped running command for {metric.metric_name} since the engine building command failed."
                    )
                    continue

                # Run the command or reuse the existing output logs.
                print_info(f"Running command for {metric.metric_name}")
                outputs = self.run_ex(
                    metric.metric_name,
                    llm_venv,
                    gpu_clock_lock,
                    session_data_writer,
                    output_dir,
                    outputs=outputs,
                    original_test_name=metric.original_test_name,
                    cmd_idx=self._current_cmd_idx)

                # Save the result state.
                result_state = self.get_result_state()
                result_states[self._current_cmd_idx] = result_state
                if result_state != "valid":
                    errors.append(self.get_error())
        finally:
            # Clean up engine dir after use.
            shutil.rmtree(self._get_engine_dir(), ignore_errors=True)

        llm_venv._new_env["__LUNOWUD"] = old_llm_venv

        # Check if any commands failed.
        if not all([result_states[idx] == "valid" for idx in result_states]):
            # If there is only one error, throw it directly.
            if len(errors) == 1:
                raise errors[0]

            # Otherwise, combine all the error messages and re-raise a generic RuntimeError.
            msg = "Multiple Errors happened:\n"
            for error_idx, e in enumerate(errors):
                msg += f"> Error {error_idx+1}/{len(errors)}: {type(e).__name__}: {e}\n"

            raise RuntimeError(msg)

    def _get_engine_dir(self) -> str:
        """
        Get the engine directory to store the engine.
        """
        escaped_label = self._test_param_labels.replace("+", "_").replace(
            ":", "_").replace(",", "_")
        return os.path.join(self._working_dir, "perf_engines", escaped_label)

    def _get_metrics(self) -> List[PerfTestMetric]:
        """
        Generate all the metric configs for the current test.
        """

        metrics = []
        if self._config.runtime == "server-benchmark":
            cmd_idx = 0
            for server_idx, client_configs in self._config.server_client_configs.items(
            ):
                server_name = self._config.server_configs[server_idx].name
                for client_config in client_configs:
                    for metric_type in SERVER_BENCHMARK_METRICS:
                        metrics.append(
                            PerfTestMetric(
                                original_test_name=self._full_test_name,
                                metric_name=self._get_metric_name(
                                    metric_type=metric_type,
                                    server_name=server_name,
                                    client_name=client_config.name),
                                metric_type=metric_type,
                                metric_regex=self._get_metric_regex(
                                    metric_type),
                                metric_threshold=self._get_metric_threshold(
                                    metric_type),
                                metric_abs_threshold=self.
                                _get_metric_abs_threshold(metric_type),
                                cmd_idx=cmd_idx,
                            ))
                    cmd_idx += 1
            return metrics

        if self._config.runtime == "disagg_server":
            for metric_type in DISAGG_SERVER_METRICS:
                metrics.append(
                    PerfTestMetric(
                        original_test_name=self._full_test_name,
                        metric_name=self._get_metric_name(
                            metric_type=metric_type),
                        metric_type=metric_type,
                        metric_regex=self._get_metric_regex(metric_type),
                        metric_threshold=self._get_metric_threshold(
                            metric_type),
                        metric_abs_threshold=self._get_metric_abs_threshold(
                            metric_type),
                        cmd_idx=0,
                    ))
            return metrics

        # Build command is the first command.
        cmd_idx = 0 if self._config.runtime != "bench" else 1
        if self._config.runtime == "bench":
            if self._config.backend in ["pytorch", "_autodeploy"]:
                print_info(
                    f"Skip building process for {self._config.model_name} as it is {self._config.backend} backend"
                )
                builder_metrics = []
            else:
                builder_metrics = [PerfMetricType.BUILD_TIME]
        else:
            builder_metrics = BUILDER_METRICS.copy()

        # Add all builder_perf metrics
        for metric_type in builder_metrics:
            metrics.append(
                PerfTestMetric(
                    original_test_name=self._full_test_name,
                    metric_name=self._get_metric_name(metric_type=metric_type),
                    metric_type=metric_type,
                    metric_regex=self._get_metric_regex(metric_type),
                    metric_threshold=self._get_metric_threshold(metric_type),
                    metric_abs_threshold=self._get_metric_abs_threshold(
                        metric_type),
                    cmd_idx=cmd_idx,
                ))
        if self._config.build_only:
            return metrics

        # Then, construct inference latency and gpu mem usage metrics, for each
        # bs and each seq len.
        for bs in self._config.batch_sizes:
            for len_idx, input_len in enumerate(self._config.input_lens):
                cmd_idx += 1
                output_len = None if self._config.is_bert_like(
                ) else self._config.output_lens[len_idx]

                # Get list of metrics depending on config.
                if self._config.runtime == "bench":
                    metric_types = BENCH_INFERENCE_METRICS.copy()
                    if self._config.streaming == "streaming":
                        metric_types.append(PerfMetricType.FIRST_TOKEN_TIME)
                        metric_types.append(PerfMetricType.OUTPUT_TOKEN_TIME)
                else:
                    metric_types = INFERENCE_METRICS.copy()
                for metric_type in metric_types:
                    metrics.append(
                        PerfTestMetric(
                            original_test_name=self._full_test_name,
                            metric_name=self._get_metric_name(
                                metric_type=metric_type,
                                bs=bs,
                                input_len=input_len,
                                output_len=output_len),
                            metric_type=metric_type,
                            metric_regex=self._get_metric_regex(metric_type),
                            metric_threshold=self._get_metric_threshold(
                                metric_type),
                            metric_abs_threshold=self._get_metric_abs_threshold(
                                metric_type),
                            cmd_idx=cmd_idx,
                        ))

        return metrics

    def _get_metric_name(self,
                         metric_type: PerfMetricType,
                         bs: int = None,
                         input_len: int = None,
                         output_len: int = None,
                         server_name: str = None,
                         client_name: str = None) -> str:
        """
        Construct the metric name for given metric_type, bs, input_len, and output_len.
        """

        # Get device subtype for autodeploy tests
        device_subtype = None
        if (hasattr(self, '_gpu_clock_lock') and self._gpu_clock_lock
                and self._config.backend == "_autodeploy"):
            device_subtype = self._gpu_clock_lock.get_device_subtype()

        if metric_type in BUILDER_METRICS:
            # We build one engine for all benchmark runs, so add all bs and seq lens to the metric name.
            metric_label = self._config.to_string(device_subtype=device_subtype)
        elif self._config.runtime == "server-benchmark":
            metric_label = self._config.to_string(
                custom_server_name=server_name,
                custom_client_name=client_name,
            )
        else:
            # Otherwise, generate per-bs and per-seqlen label.
            metric_label = self._config.to_string(
                custom_bs=bs,
                custom_input_len=input_len,
                custom_output_len=output_len,
                device_subtype=device_subtype,
            )
        metric_name = f"test_perf_metric_{metric_type.lower()}"
        return self._test_domain_name + "::" + metric_name + "[" + metric_label + "]"

    def _get_metric_regex(self, metric_type: PerfMetricType) -> re.Pattern:
        """
        Get the regex used to parse the metric result for the metric type.
        """

        if self._config.runtime == "bench":
            if metric_type not in BENCH_PERF_METRIC_LOG_QUERIES:
                raise ValueError(f"Unexpected metric_type: {metric_type}")
            return BENCH_PERF_METRIC_LOG_QUERIES[metric_type]
        elif self._config.runtime == "server-benchmark":
            if metric_type not in SERVER_BENCHMARK_PERF_METRIC_LOG_QUERIES:
                raise ValueError(f"Unexpected metric_type: {metric_type}")
            return SERVER_BENCHMARK_PERF_METRIC_LOG_QUERIES[metric_type]
        else:
            pytest.skip("only support trtllm-bench runtime for now")

    def _get_metric_threshold(self, metric_type: PerfMetricType) -> float:
        """
        Get the threshold for the metric type.
        """

        if metric_type not in PERF_METRIC_THRESHOLD:
            raise ValueError(f"Unexpected metric_type: {metric_type}")

        return PERF_METRIC_THRESHOLD[metric_type][0]

    def _get_metric_abs_threshold(self, metric_type: PerfMetricType) -> float:
        """
        Get the absolute threshold for the metric type.
        """

        if metric_type not in PERF_METRIC_THRESHOLD:
            raise ValueError(f"Unexpected metric_type: {metric_type}")

        return PERF_METRIC_THRESHOLD[metric_type][1]

    def _gen_disagg_worker_config(self):
        ctx_config = {
            'max_batch_size': 32,
            'max_num_tokens': 4096,
            'max_seq_len': 4096,
            'tensor_parallel_size': self._config.ctx_tp_size,
            'enable_attention_dp': self._config.ctx_dp_size > 1,
            'print_iter_log': True,
            'disable_overlap_scheduler': True,
            'kv_cache_config': {
                'enable_block_reuse': False,
                # 'free_gpu_memory_fraction': ctx_free_gpu_memory_fraction,
                'free_gpu_memory_fraction': 0.5,
                'dtype': 'fp8',
            },
            'disable_overlap_scheduler': True,
            'cache_transceiver_config': {
                # 'max_tokens_in_buffer': cache_transceiver_max_num_tokens,
                'max_tokens_in_buffer': 4096,
                'backend': 'DEFAULT',
            },
        }

        gen_config = {
            'tensor_parallel_size': self._config.gen_tp_size,
            'enable_attention_dp': self._config.gen_dp_size > 1,
            'pipeline_parallel_size': self._config.gen_pp_size,
            'max_batch_size': 32,
            'max_num_tokens': 4096,
            'max_seq_len': 4096,
            'cuda_graph_config': {
                'enable_padding': True,
                'batch_sizes': [1, 2, 4, 8, 16, 32],
            },
            'print_iter_log': True,
            'kv_cache_config': {
                'enable_block_reuse': False,
                'free_gpu_memory_fraction': 0.5,
                'dtype': 'fp8',
            },
            'cache_transceiver_config': {
                'max_tokens_in_buffer': 4096,
                'backend': 'DEFAULT',
            },
        }
        return ctx_config, gen_config

    def _gen_disagg_server_config(self):
        server_config = {
            'hostname': 'localhost',
            'port': 8000,
            'backend': 'pytorch',
            'context_servers': {
                'num_instances': 1,
                'urls': ['localhost:8001']
            },
            'generation_servers': {
                'num_instances': 1,
                'urls': ['localhost:8002']
            }
        }
        return server_config

    def _get_disagg_worker_deploy_command(self):
        ctx_config, gen_config = self._gen_disagg_worker_config()
        ctx_config_path = os.path.join(self._working_dir, "ctx_config.yaml")
        gen_config_path = os.path.join(self._working_dir, "gen_config.yaml")
        with open(ctx_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(ctx_config, f)
        with open(gen_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(gen_config, f)

        print_info(f"ctx_server_config: {ctx_config}")
        print_info(f"gen_server_config: {gen_config}")

        model_path = MODEL_PATH_DICT[self._config.model_name]
        model_dir = os.path.join(llm_models_root(), model_path)

        ctx_gpu_list = ",".join(
            [str(i) for i in range(self._config.ctx_server_workers)])

        gen_gpu_list = ",".join([
            str(i) for i in range(
                self._config.ctx_server_workers,
                self._config.ctx_server_workers +
                self._config.gen_server_workers)
        ])

        ctx_cmd = f'CUDA_VISIBLE_DEVICES={ctx_gpu_list} trtllm-serve {model_dir} --host localhost --port 8001 --extra_llm_api_options {ctx_config_path}'
        gen_cmd = f'CUDA_VISIBLE_DEVICES={gen_gpu_list} trtllm-serve {model_dir} --host localhost --port 8002 --extra_llm_api_options {gen_config_path}'
        return ctx_cmd, gen_cmd

    def _get_disagg_server_deploy_command(self):
        server_config = self._gen_disagg_server_config()
        server_config_path = os.path.join(self._working_dir,
                                          "server_config.yaml")
        with open(server_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(server_config, f)
        return f'trtllm-serve disaggregated -c {server_config_path} -t 3600 -r 3600'

    def _get_disagg_client_command(self):
        client_dir = os.path.join(self._llm_root,
                                  "examples/disaggregated/clients")
        client_cmd = [
            'python3', f'{client_dir}/disagg_client.py', '-c',
            f'{self._working_dir}/server_config.yaml', '-p',
            f'{client_dir}/prompts.json', '--ignore-eos',
            '--server-start-timeout',
            str(3600)
        ]
        return client_cmd

    def _get_disagg_benchmark_command(self):
        benchmark_script = os.path.join(self._llm_root, "tensorrt_llm", "serve",
                                        "scripts", "benchmark_serving.py")
        model_path = MODEL_PATH_DICT[self._config.model_name]
        model_dir = os.path.join(llm_models_root(), model_path)
        shared_gpt_path = os.path.join(
            llm_models_root(), "datasets",
            "ShareGPT_V3_unfiltered_cleaned_split.json")
        benchmark_cmd = [
            'python3',
            benchmark_script,
            '--model',
            model_dir,
            '--tokenizer',
            model_dir,
            '--dataset-name',
            'random',
            '--dataset-path',
            shared_gpt_path,
            '--random-input-len',
            '1024',
            '--random-output-len',
            '1024',
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
        return benchmark_cmd


def run_perf_test(perf_case_name, trt_performance_cache_fpath,
                  trt_gpu_clock_lock, llm_session_data_writer, output_dir,
                  llm_venv, llm_root):
    """
    The actual test definition for TensorRT LLM perf test.
    """
    working_dir = llm_venv.get_working_directory()
    test_runner = MultiMetricPerfTest(perf_case_name)
    test_runner.set_runtime_configs(llm_root, working_dir,
                                    trt_performance_cache_fpath,
                                    trt_gpu_clock_lock)
    test_runner.run_metrics(llm_venv, trt_gpu_clock_lock,
                            llm_session_data_writer, output_dir)


def generate_perf_tests(session, config, items):
    """
    Generate all the perf tests based on test lists to speed up the test collection time.
    """

    print_info(f"Dynamically generating perf tests...")
    valid_prefixes = [
        "perf/test_perf.py::test_perf[",
        # TRT pipeline adds "llm/" prefix, so include it so that TRT-LLM perf tests can run in TRT pipelines.
        "llm/perf/test_perf.py::test_perf[",
    ]
    items = generate_test_nodes(session, config, items, valid_prefixes,
                                run_perf_test)
    print_info(f"Completed generating perf tests.")

    return items
