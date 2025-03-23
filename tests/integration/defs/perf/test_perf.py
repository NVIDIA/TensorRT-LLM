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
from defs.common import convert_weights, get_cpp_benchmark, quantize_data
from defs.trt_test_alternative import (is_linux, is_windows, print_info,
                                       print_warning)

from ..conftest import get_llm_root, llm_models_root, trt_environment
from .utils import (AbstractPerfScriptTestClass, PerfBenchScriptTestCmds,
                    PerfMetricType, PerfScriptTestCmds, generate_test_nodes)

if not hasattr(re, "Pattern"):
    re.Pattern = type(re.compile(""))

ALLOWED_CONFIGS_CACHE = None  # Cache to avoid modifying sys.path many times.
MAP_BY_SOCKET = None

# Model PATH of local dir synced from internal LLM models repo
MODEL_PATH_DICT = {
    "llama_v2_7b": "llama-models-v2/llama-v2-7b-hf",
    "llama_v2_13b": "llama-models-v2/llama-v2-13b-hf",
    "llama_v2_70b": "llama-models-v2/llama-v2-70b-hf",
    "llama_v3_8b": "llama-models-v3/8B",
    "llama_v3.1_8b": "llama-3.1-model/Meta-Llama-3.1-8B",
    "llama_v3.1_8b_instruct": "llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
    "llama_v3.1_70b": "llama-3.1-model/Meta-Llama-3.1-70B",
    "llama_v3.1_70b_instruct": "llama-3.1-model/Meta-Llama-3.1-70B-Instruct",
    "mixtral_8x7b_v0.1": "Mixtral-8x7B-v0.1",
    "mixtral_8x7b_v0.1_instruct": "Mixtral-8x7B-Instruct-v0.1",
    "mistral_7b_v0.1": "mistral-7b-v0.1",
    "deepseek_r1": "DeepSeek-R1/DeepSeek-R1",
    "deepseek_r1_nvfp4": "DeepSeek-R1/DeepSeek-R1-FP4",
    "deepseek_v3_lite_fp8": "DeepSeek-V3-Lite/fp8",
    "deepseek_v3_lite_nvfp4": "DeepSeek-V3-Lite/nvfp4_moe_only",
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
    "mixtral_8x7b_v0.1_hf": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral_8x7b_v0.1_instruct_hf": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral_7b_v0.1_hf": "mistralai/Mistral-7B-v0.1",
}
LORA_MODEL_PATH = {
    "llama_v2_13b": "llama-models-v2/chinese-llama-2-lora-13b",
    "mixtral_8x7b_0.1": "chinese-mixtral-lora",
}

TIMING_CACHE_DIR = os.environ.get("TIMING_CACHE_DIR", "")


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
}
BENCH_PERF_METRIC_LOG_QUERIES = {
    PerfMetricType.BUILD_TIME:
    re.compile(r"Engine generation completed in ([\d\.]+) seconds"),
    PerfMetricType.INFERENCE_TIME:
    re.compile(r"Total Latency \(ms\):\s+([\d\.]+)"),
    PerfMetricType.TOKEN_THROUGHPUT:
    re.compile(r"GPU Output Throughput \(tokens\/sec\/gpu\):\s+([\d\.]+)"),
    PerfMetricType.SEQ_THROUGHPUT:
    re.compile(r"Request Throughput \(req\/sec\):\s+([\d\.]+)"),
    PerfMetricType.FIRST_TOKEN_TIME:
    re.compile(r"Average time-to-first-token \[TTFT\]\(ms\):\s+([\d\.]+)"),
    PerfMetricType.OUTPUT_TOKEN_TIME:
    re.compile(r"Average time-per-output-token \[TPOT\]\(ms\):\s+([\d\.]+)"),
}
# (Relative threshold, Absolute threshold) for all metric types
PERF_METRIC_THRESHOLD = {
    PerfMetricType.BUILD_TIME: (0.1, 30),  # Ignore build time regression < 30ms
    PerfMetricType.INFERENCE_TIME:
    (0.1, 50),  # Ignore inference time regression < 50ms
    PerfMetricType.FIRST_TOKEN_TIME:
    (0.1, 50),  # Ignore first token time regression < 50ms
    PerfMetricType.OUTPUT_TOKEN_TIME:
    (0.1, 50),  # Ignore per output token time regression < 50ms
    PerfMetricType.SEQ_LATENCY: (0.1, 50),  # Ignore latency regression < 50ms
    PerfMetricType.TOKEN_THROUGHPUT: (
        -0.1, 10
    ),  # Ignore throughput regression < 10 tokens/s. Negative rel threshold is to indicate that larger is better.
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

BERT_CPP_INFERENCE_METRICS = [
    PerfMetricType.INFERENCE_TIME,
    PerfMetricType.CONTEXT_GPU_MEMORY,
]

MANAGER_INFERENCE_METRICS = [
    PerfMetricType.INFERENCE_TIME,
    PerfMetricType.TOKEN_THROUGHPUT,
    PerfMetricType.CONTEXT_GPU_MEMORY,
    PerfMetricType.SEQ_THROUGHPUT,
    PerfMetricType.SEQ_LATENCY,
    PerfMetricType.KV_CACHE_SIZE,
]

BENCH_INFERENCE_METRICS = [
    PerfMetricType.INFERENCE_TIME,
    PerfMetricType.TOKEN_THROUGHPUT,
    PerfMetricType.SEQ_THROUGHPUT,
]


class PerfTestMetric(NamedTuple):
    """
    Configurations of a test metric.
    """
    # The original test name used to run the TURTLE test.
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
        max_batch_size: int = 0,
        gpu_weights_percent: float = -1,
        batch_sizes: List[int] = [0],
        input_lens: List[int] = [8],
        output_lens: List[int] = [1],
        num_beams: int = 1,
        num_loras: int = 0,
        num_reqs: int = 512,
        concurrency: int = -1,
        quantization: str = "",
        ep_size: int = None,
        tp_size: int = 1,
        pp_size: int = 1,
        num_gpus: int = 1,
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
        # Just build engines
        self.build_only = False

    def to_string(self,
                  custom_bs: int = None,
                  custom_input_len: int = None,
                  custom_output_len: int = None) -> str:

        # First, add the model name.
        entries = [self.model_name]

        if self.runtime == "cpp":  # gptSessionBenchmark or berBenchmark runtime
            entries.append(f"cpp")
        elif self.runtime == "cppmanager":  # gptMananberBenchmark runtime
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
            if self.streaming == "streaming":
                entries.append(f"streaming")

        # Add mode and dtype.
        if self.runtime != "bench":
            entries.append(self.mode)
        entries.append(self.data_type)

        if self.gpu_weights_percent != -1:
            entries.append(f"gwp:{self.gpu_weights_percent}")

        if self.multiple_profiles:
            entries.append(f"mp")

        # Add Max batch size.
        if self.max_batch_size > 0:
            entries.append(f"maxbs:{self.max_batch_size}")

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

    def load_from_str(self, test_param_labels) -> None:
        """
        Populate the config properties given the test param string.
        """

        # Extract configs from test param labels.
        labels = test_param_labels.split("-")

        self.model_name = labels.pop(0)
        self.runtime = "python" if labels[0] not in [
            "cpp",
            "cppmanager",
            "bench",
        ] else labels.pop(0)
        self.api = labels.pop(0) if labels[0] == "exe" else ""
        self.backend = labels.pop(0) if labels[0] == "pytorch" else ""
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
        VALID_RUNTIMES = ["cpp", "cppmanager", "python", "bench"]
        assert self.runtime in VALID_RUNTIMES, f"Invalid runtime {self.runtime}!"

        # Validate plugin mode.
        VALID_MODES = ["plugin", "ootb", "ootb_except_mha"]
        if self.runtime == "cppmanager":
            VALID_MODES += ["plugin_ifb"]
        assert self.mode in VALID_MODES, f"Invalid mode {self.mode}!"

        # Validate dtype.
        VALID_DTYPES = ["float32", "float16", "bfloat16"]
        assert self.data_type in VALID_DTYPES, f"Invalid data_type {self.data_type}!"

        # Validate quantization mode.
        if self.model_name in MODEL_PATH_DICT.keys():
            VALID_QUANTS = [
                "", "nvfp4", "fp8", "int8_sq", "int4_awq", "w4a8_awq",
                "int8_wo", "int4_wo", "full_prec"
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
            if self.runtime != "cppmanager":
                # Validate max batch size.
                if self.max_batch_size > 0:
                    assert max(
                        self.batch_sizes
                    ) <= self.max_batch_size, f"Batch Size larger than Max Batch Size!"
                if self.runtime != "bench":
                    # Validate bs, seq lens, and num_beams.
                    assert len(
                        self.batch_sizes
                    ) > 0 and self.batch_sizes[0] > 0, f"Empty batch sizes!"
                assert self.static_batching == "", f"Static Batching only valid for gptManagerBenchmark!"
                assert self.api == "", f"API Type only valid for gptManagerBenchmark!"
                if self.runtime != "bench":
                    assert self.streaming == "", f"Streaming only valid for gptManagerBenchmark and trtllm-bench!"

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
                assert all(
                    [b >= 32 for b in self.batch_sizes]
                ), f"BERT with small BS is very unstable! Please increase to at least 32."

            # GPT-350m and Bloom-560m with small BS are very unstable. Only run these small models with larger BS.
            if self.model_name in ["gpt_350m", "bloom_560m"]:
                assert all(
                    [b >= 32 for b in self.batch_sizes]
                ), f"gpt_350m and bloom_560m with small BS are very unstable! Please increase to at least 32."

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
        # full_test_name is the full test name appearing in TURTLE output.
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

    def get_test_name(self) -> str:
        return str(self._config)

    def set_runtime_configs(self, llm_root, working_dir,
                            perf_cache_fpath) -> None:
        if self._config.runtime == "cpp":
            cpp_benchmark_name = "bertBenchmark" if self._config.is_bert_like(
            ) else "gptSessionBenchmark"
            benchmark_script = get_cpp_benchmark(cpp_benchmark_name, llm_root)
        elif self._config.runtime == "cppmanager":
            benchmark_script = get_cpp_benchmark("gptManagerBenchmark",
                                                 llm_root)
        elif self._config.runtime == "bench":
            benchmark_script = "trtllm-bench"
        else:
            benchmark_script = os.path.join(llm_root, "benchmarks", "python",
                                            "benchmark.py")
        allowed_configs = import_allowed_perf_config()
        allowed_models = allowed_configs.get_allowed_models()
        if self._config.runtime == "bench":
            build_script = "trtllm-bench"
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

    def get_convert_weights_command(self, model_dir, engine_dir) -> str:
        """
        Get the convert checkpoint command.
        """
        if "phi" in self._config.model_name:
            example_name = "phi"
        else:
            example_name = "llama"

        if self._config.quantization != "":
            command, checkpoint_dir = quantize_data(
                llm_venv=None,
                example_root=os.path.join(get_llm_root(), "examples",
                                          example_name),
                model_dir=model_dir,
                calib_dataset=os.path.join(llm_models_root(), "datasets",
                                           "cnn_dailymail"),
                dtype=self._config.data_type,
                qformat=self._config.quantization,
                tp_size=self._config.tp_size,
                pp_size=self._config.pp_size,
                quantize_dir=engine_dir)
        else:
            command, checkpoint_dir = convert_weights(
                llm_venv=None,
                example_root=os.path.join(get_llm_root(), "examples",
                                          example_name),
                cmodel_dir=engine_dir,
                model=self._config.model_name,
                model_path=model_dir,
                tp_size=self._config.tp_size,
                pp_size=self._config.pp_size,
                data_type=self._config.data_type)
        command = [f"python3"] + command

        return command, checkpoint_dir

    def get_convert_lora_weights_command(self, model_dir, engine_dir) -> str:
        script = os.path.join(self._llm_root, "examples", "hf_lora_convert.py")
        checkpoint_dir = os.path.join(engine_dir, "lora_cpp")
        command = [
            script, f"-i={model_dir}", "--storage-type=float16",
            f"-o={checkpoint_dir}"
        ]
        command = [f"python3"] + command

        return command, checkpoint_dir

    def get_trtllm_build_command(self, engine_dir, checkpoint_dir) -> list:
        build_cmd = [
            self._build_script, f"--output_dir={engine_dir}",
            f"--checkpoint_dir={checkpoint_dir}",
            f"--workers={self._config.tp_size}",
            f"--use_paged_context_fmha=enable", f"--monitor_memory"
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
        if self._config.max_batch_size > 0:
            build_cmd.append(f"--max_batch_size={self._config.max_batch_size}")
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
        model_dir = ""
        if self._config.model_name in MODEL_PATH_DICT.keys():
            model_dir = os.path.join(llm_models_root(),
                                     MODEL_PATH_DICT[self._config.model_name])
        elif self._config.model_name in HF_MODEL_PATH.keys():
            model_dir = os.path.join(
                llm_models_root(),
                MODEL_PATH_DICT[self._config.model_name.split('_hf')[0]])
        return model_dir

    def get_trtllm_bench_build_command(self, engine_dir) -> list:
        model_dir = self.get_trtllm_bench_model()
        dataset_path = os.path.join(engine_dir, "synthetic_data.json")
        if model_dir == "":
            pytest.skip("Model Name is not supported by trtllm-bench")
        model_name = self._config.model_name
        if not model_name.endswith("_hf"):
            model_name = model_name + "_hf"
        hf_model_name = HF_MODEL_PATH.get(model_name, "")
        build_cmd = [
            self._build_script, f"--workspace={engine_dir}",
            f"--model={hf_model_name}", f"--model_path={model_dir}", "build",
            f"--dataset={dataset_path}"
        ]
        if self._config.max_batch_size > 0:
            build_cmd.append(f"--max_batch_size={self._config.max_batch_size}")
        max_seq_len = max(self._config.input_lens) + max(
            self._config.output_lens)
        build_cmd.append(f"--max_seq_len={max_seq_len}")
        if self._config.quantization:
            build_cmd.append(
                f"--quantization={self._config.quantization.upper()}")
        return build_cmd

    def get_benchmark_build_command(self, engine_dir) -> list:
        mode_flag = self._config.mode.replace("_", "-")
        build_cmd = [
            self._build_script, f"--model={self._config.model_name}",
            "--log_level=info", f"--mode={mode_flag}",
            f"--dtype={self._config.data_type}", f"--output_dir={engine_dir}",
            "--monitor_memory"
        ]
        if self._config.quantization != "":
            build_cmd.append(f"--quantization={self._config.quantization}")
        num_beams = self._config.num_beams
        if num_beams > 1:
            build_cmd.append(f"--max_beam_width={num_beams}")
        gpu_percent = self._config.gpu_weights_percent
        if gpu_percent != -1:
            build_cmd += [f"--weight_streaming"]
        if self._config.max_batch_size > 0:
            build_cmd.append(f"--max_batch_size={self._config.max_batch_size}")

        # For performance data stability, set opt_num_token/opt_batch_size to 8 when max batch size is greater than 8.
        # The script will use the settings from allow_configs.py if max_batch_size is set to 0,
        # opt_num_token/opt_batch_size is also necessary for stability.
        if self._config.max_batch_size > 8 or self._config.max_batch_size == 0:
            if self._config.mode in ["plugin_ifb", "plugin", 'ootb_except_mha']:
                build_cmd.append("--opt_num_tokens=8")
            else:
                build_cmd.append("--opt_batch_size=8")
        # For Multiple Profiles
        if self._config.multiple_profiles:
            build_cmd.append("--multiple_profiles")
        # For engine inspector
        build_cmd.append("--profiling_verbosity=layer_names_only")
        if TIMING_CACHE_DIR and not self._config.build_only:
            timing_cache = os.path.join(TIMING_CACHE_DIR, "model.cache")
            build_cmd.append(f"--input_timing_cache={timing_cache}")
            build_cmd.append(f"--output_timing_cache={timing_cache}")
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
            lora_data = os.path.join(engine_dir,
                                     f"token-norm-dist-lora-{nloras}.json")
            with open(lora_data, 'w') as file:
                pass
            data_cmd += [
                "python3", prepare_data_script, f"--output={lora_data}",
                f"--rand-task-id 0 {nloras-1}", f"--tokenizer={tokenizer_dir}",
                f"token-norm-dist", f"--num-requests={self._config.num_reqs}",
                f"--input-mean={input_len}", f"--output-mean={output_len}",
                f"--input-stdev={istdev}", f"--output-stdev={ostdev}"
            ]
            data_cmd += [";"]
            generate_rand_lora_script = os.path.join(self._llm_root,
                                                     "benchmarks", "cpp",
                                                     "utils",
                                                     "generate_rand_loras.py")
            checkpoint_dir = os.path.join(engine_dir, "lora_cpp")
            lora_dir = os.path.join(engine_dir, f"loras")
            data_cmd += [
                "python3", generate_rand_lora_script, checkpoint_dir, lora_dir,
                "16"
            ]
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

    def get_python_runtime_benchmark_command(self, engine_dir, bs, input_len,
                                             output_len):
        benchmark_cmd = [
            self._benchmark_script,
        ]
        if self._config.is_bert_like():
            model = "enc"
            benchmark_cmd.append(f"--engine_dir={engine_dir}")
        elif self._config.is_enc_dec():
            model = "enc-dec"
            benchmark_cmd.append(
                f"--encoder_engine_dir={os.path.join(engine_dir, 'encoder')}")
            benchmark_cmd.append(
                f"--decoder_engine_dir={os.path.join(engine_dir, 'decoder')}")

        else:
            model = "dec"
            benchmark_cmd.append(f"--engine_dir={engine_dir}")
        benchmark_cmd.append(f"--model={model}")
        benchmark_cmd += [f"--batch_size={bs}"]
        # Use 3 warm-up runs and minimum of 10 actual runs and minimum of 10 seconds for now.
        benchmark_cmd += [f"--warm_up=3", f"--num_runs=10", f"--duration=10"]
        benchmark_cmd += [f"--dtype={self._config.data_type}"]
        if self._config.is_bert_like():
            benchmark_cmd.append(f"--input_len={input_len}")
        else:
            benchmark_cmd.append(f"--input_output_len={input_len},{output_len}")
        # Weight streaming don't support CUDA Graph for now.
        gpu_percent = self._config.gpu_weights_percent
        if gpu_percent == -1:
            benchmark_cmd.append(f"--enable_cuda_graph")
        return benchmark_cmd

    def get_gpt_session_runtime_benchmark_command(self, engine_dir, bs,
                                                  input_len, output_len):
        benchmark_cmd = [
            self._benchmark_script,
            # This is required to get context GPU info
            f"--log_level=info",
        ]
        benchmark_cmd.append(f"--engine_dir={engine_dir}")
        if self._config.is_bert_like():
            benchmark_cmd.append(f"--model={self._config.model_name}")
        num_beams = self._config.num_beams
        if num_beams > 1:
            benchmark_cmd.append(f"--beam_width={num_beams}")
        gpu_percent = self._config.gpu_weights_percent
        if gpu_percent != -1:
            benchmark_cmd.append(f"--gpu_weights_percent={gpu_percent}")
        benchmark_cmd += [f"--batch_size={bs}"]
        # Use 3 warm-up runs and minimum of 10 actual runs and minimum of 10 seconds for now.
        benchmark_cmd += [f"--warm_up=3", f"--num_runs=10", f"--duration=10"]
        if not self._config.is_bert_like() and not self._config.is_enc_dec(
        ) and not self._config.is_mamba_family() and self._config.num_gpus < 8:
            # Dump layer information and per-layer profile
            benchmark_cmd += ["--dump_layer_info", "--dump_profile"]

        # For GPT Models and enc-dec Models
        if not self._config.is_bert_like():
            benchmark_cmd.append(f"--input_output_len={input_len},{output_len}")
            # Weight streaming don't support CUDA Graph for now.
            # MoE OOTB doesn't support CUDA Graph
            gpu_percent = self._config.gpu_weights_percent
            if gpu_percent == -1 and not (self._config.is_moe_family()
                                          and self._config.mode
                                          in ['ootb', 'ootb_except_mha']):
                benchmark_cmd.append(f"--enable_cuda_graph")
        # For BERT Models:
        else:
            benchmark_cmd.append(f"--input_len={input_len}")
        return benchmark_cmd

    def get_trtllm_bench_command(self, engine_dir):
        model_dir = self.get_trtllm_bench_model()
        model_name = self._config.model_name
        dataset_path = os.path.join(engine_dir, "synthetic_data.json")
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
        ]
        if self._config.backend != "pytorch":
            benchmark_cmd += [f"--engine_dir={engine_dir}"]
        else:
            benchmark_cmd += ["--backend=pytorch"]
        if self._config.max_batch_size > 0:
            benchmark_cmd += [f"--max_batch_size={self._config.max_batch_size}"]
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
            config = {
                'enable_attention_dp': True,
                'pytorch_backend_config': {
                    'enable_overlap_scheduler': True,
                    'print_iter_log': True,
                    'use_cuda_graph': True,
                    'cuda_graph_batch_sizes': [1, 512]
                }
            }
            with open('extra-llm-api-config.yml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            benchmark_cmd += [
                f"--extra_llm_api_options=extra-llm-api-config.yml"
            ]
        return benchmark_cmd

    def get_gpt_manager_runtime_benchmark_command(self, engine_dir, bs,
                                                  input_len):
        benchmark_cmd = [
            self._benchmark_script,
            # This is required to get context GPU info
            f"--log_level=info",
        ]
        if self._config.is_enc_dec():
            benchmark_cmd.append(
                f"--encoder_engine_dir={os.path.join(engine_dir, 'encoder')}")
            benchmark_cmd.append(
                f"--decoder_engine_dir={os.path.join(engine_dir, 'decoder')}")
        else:
            benchmark_cmd.append(f"--engine_dir={engine_dir}")

        num_beams = self._config.num_beams
        if num_beams > 1:
            benchmark_cmd.append(f"--beam_width={num_beams}")
        gpu_percent = self._config.gpu_weights_percent
        if gpu_percent != -1:
            benchmark_cmd.append(f"--gpu_weights_percent={gpu_percent}")
        if self._config.num_loras > 0:
            nloras = self._config.num_loras
            dataset_path = os.path.join(engine_dir,
                                        f"token-norm-dist-lora-{nloras}.json")
            lora_dir = os.path.join(engine_dir, f"loras")

            eos_id = 2
            num_layers = 32 if "mixtral" in self._config.model_name else 40
            num_lora_mods = 8 if "mixtral" in self._config.model_name else 7
            max_lora_rank = 64
            benchmark_cmd += [f"--lora_host_cache_bytes=8589934592"]
            benchmark_cmd += [
                f"--lora_num_device_mod_layers={32 * num_layers * num_lora_mods * max_lora_rank}"
            ]
            benchmark_cmd += [f"--eos_id={eos_id}"]
            benchmark_cmd += [f"--lora_dir={lora_dir}"]
        else:
            dataset_path = os.path.join(engine_dir, "synthetic_data.json")
        benchmark_cmd += [f"--dataset={dataset_path}"]
        # API Type is executor
        if self._config.api == "exe":
            benchmark_cmd += [f"--api=executor"]
        if self._config.mode == "plugin_ifb":
            benchmark_cmd += [
                f"--type=UIFB"
            ] if self._config.is_mamba_family() else ["--type=IFB"]
        else:
            benchmark_cmd += [f"--type=V1"]
        if self._config.streaming == "streaming":
            benchmark_cmd += [f"--streaming"]
            benchmark_cmd += [f"--scheduler_policy=max_utilization"]
        if self._config.static_batching == "static_batching":
            benchmark_cmd += [f"--static_emulated_batch_size={bs}"]
        if self._config.concurrency != -1:
            benchmark_cmd += [f"--concurrency={self._config.concurrency}"]

        return benchmark_cmd

    def get_commands(self):

        # Whether this is python or cpp runtime perf test.
        is_python = self._config.runtime == "python"
        num_gpus = self._config.num_gpus
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
        convert_cmd = []
        build_cmd = []
        if self._build_script == "trtllm-build" and self._config.model_name in MODEL_PATH_DICT.keys(
        ):
            model_path = MODEL_PATH_DICT[self._config.model_name]
            model_dir = os.path.join(llm_models_root(), model_path)
            if not os.path.exists(engine_dir):
                os.makedirs(engine_dir, exist_ok=True)
            convert_cmd, checkpoint_dir = self.get_convert_weights_command(
                model_dir, engine_dir)
            if self._config.num_loras > 0:
                if self._config.model_name in LORA_MODEL_PATH.keys():
                    model_dir = os.path.join(
                        llm_models_root(),
                        LORA_MODEL_PATH[self._config.model_name])
                    convert_lora_cmd, lora_checkpoint_dir = self.get_convert_lora_weights_command(
                        model_dir, engine_dir)
                    convert_cmd += [";"]
                    convert_cmd += convert_lora_cmd
                else:
                    pytest.skip(
                        f"There is no LoRA weights model for {self._config.model_name}"
                    )
            build_cmd = self.get_trtllm_build_command(engine_dir,
                                                      checkpoint_dir)
        elif self._config.runtime == "bench":
            if self._config.backend == "pytorch":
                # Skip building process as it is pytorch backend")
                pass
            else:
                build_cmd = self.get_trtllm_bench_build_command(engine_dir)
        else:
            build_cmd = self.get_benchmark_build_command(engine_dir)
        # Construct prepare synthetic data command
        data_cmds = []

        # Construct benchmark commands for each bs and seq len combination.
        benchmark_cmds = []
        for bs in self._config.batch_sizes:
            for len_idx, input_len in enumerate(self._config.input_lens):
                output_len = None if self._config.is_bert_like(
                ) else self._config.output_lens[len_idx]
                if is_python:
                    benchmark_cmd = self.get_python_runtime_benchmark_command(
                        engine_dir, bs, input_len, output_len)
                elif self._config.runtime == "bench":
                    benchmark_cmd = self.get_trtllm_bench_command(engine_dir)
                elif self._config.runtime == "cpp":
                    benchmark_cmd = self.get_gpt_session_runtime_benchmark_command(
                        engine_dir, bs, input_len, output_len)
                else:
                    benchmark_cmd = self.get_gpt_manager_runtime_benchmark_command(
                        engine_dir, bs, input_len)
                benchmark_cmds.append(benchmark_cmd)
                if not self._config.runtime == "cpp" and not is_python:
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
            return PerfScriptTestCmds(convert_cmd, build_cmd, data_cmds,
                                      benchmark_cmds, mpi_cmd, is_python)

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

        # Use the regex to go through the log from the N-th command, where N = cmd_idx.
        print_info(
            f"Searching for metric {metric_name} from output log of command {cmd_idx} ..."
        )

        regex_matches = [
            metric.metric_regex.search(line)
            for line in outputs[cmd_idx].split("\n")
        ]
        metric_values = [
            float(match.group(1)) for match in regex_matches if match
        ]

        if len(metric_values) == 0:
            if self._build_script == "trtllm-build" and metric.metric_type == PerfMetricType.ENGINE_SIZE:
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
            elif self._build_script != "trtllm-build":
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

        # Build command is the first command.
        cmd_idx = 0 if self._config.runtime != "bench" else 1
        if self._config.runtime == "bench":
            if self._config.backend == "pytorch":
                print_info(
                    f"Skip building process for {self._config.model_name} as it is pytorch backend"
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
                    metric_name=self._get_metric_name(metric_type),
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
                if self._config.runtime == "cpp":
                    metric_types.append(PerfMetricType.TOKEN_THROUGHPUT)

                if self._config.runtime == "cppmanager":
                    metric_types = MANAGER_INFERENCE_METRICS.copy()
                    if self._config.streaming == "streaming":
                        metric_types.append(PerfMetricType.FIRST_TOKEN_TIME)
                    if self._config.mode != "plugin_ifb" or self._config.is_mamba_family(
                    ):
                        metric_types.remove(PerfMetricType.KV_CACHE_SIZE)
                if self._config.is_bert_like(
                ) and self._config.runtime == "cpp":
                    # TODO: bertBenchmark does not report peak GPU memory yet.
                    metric_types = BERT_CPP_INFERENCE_METRICS

                for metric_type in metric_types:
                    metrics.append(
                        PerfTestMetric(
                            original_test_name=self._full_test_name,
                            metric_name=self._get_metric_name(
                                metric_type, bs, input_len, output_len),
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
                         output_len: int = None) -> str:
        """
        Construct the metric name for given metric_type, bs, input_len, and output_len.
        """

        if metric_type in BUILDER_METRICS:
            # We build one engine for all benchmark runs, so add all bs and seq lens to the metric name.
            metric_label = self._config.to_string()
        else:
            # Otherwise, generate per-bs and per-seqlen label.
            metric_label = self._config.to_string(
                custom_bs=bs,
                custom_input_len=input_len,
                custom_output_len=output_len,
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
        else:
            if metric_type not in PERF_METRIC_LOG_QUERIES:
                raise ValueError(f"Unexpected metric_type: {metric_type}")
            return PERF_METRIC_LOG_QUERIES[metric_type]

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


def run_perf_test(perf_case_name, trt_performance_cache_fpath,
                  trt_gpu_clock_lock, llm_session_data_writer, output_dir,
                  llm_venv, llm_root):
    """
    The actual test definition for TensorRT LLM perf test.
    """
    working_dir = llm_venv.get_working_directory()
    test_runner = MultiMetricPerfTest(perf_case_name)
    test_runner.set_runtime_configs(llm_root, working_dir,
                                    trt_performance_cache_fpath)
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
