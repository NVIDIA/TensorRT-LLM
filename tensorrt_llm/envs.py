# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from typing import Any, Literal

EnvType = Literal["bool", "int", "float", "str"]


@dataclass(frozen=True)
class EnvSpec:
    type: EnvType
    default: Any
    doc: str

    @property
    def name(self) -> str:
        for env_name, spec in ENV_SPECS.items():
            if spec is self:
                return env_name
        raise ValueError("EnvSpec instance is not registered in ENV_SPECS.")


ENV_SPECS = {
    "AD_DUMP_GRAPHS_DIR": EnvSpec("str", None, "Directory for dumping AutoDeploy graphs."),
    "BUILDER_FORCE_NUM_PROFILES": EnvSpec(
        "int",
        None,
        "Override the number of TensorRT build optimization profiles.",
    ),
    "DISABLE_HARMONY_ADAPTER": EnvSpec("bool", False, "Disable Harmony adapter in OpenAI server."),
    "DISABLE_LAMPORT_REDUCE_NORM_FUSION": EnvSpec(
        "bool",
        False,
        "Disable lamport reduce-norm fusion in compilation backend.",
    ),
    "DISABLE_TORCH_DEVICE_SET": EnvSpec(
        "bool",
        False,
        "Disable setting torch device in runtime generation helpers.",
    ),
    "ENABLE_CONFIGURABLE_MOE": EnvSpec("bool", True, "Enable configurable MoE implementation."),
    "ENABLE_PERFECT_ROUTER": EnvSpec(
        "bool",
        False,
        "Enable perfect router path for selected MoE models.",
    ),
    "EXPERT_STATISTIC_ITER_RANGE": EnvSpec(
        "str",
        None,
        "Iteration range for expert statistics in 'start-stop' format.",
    ),
    "EXPERT_STATISTIC_PATH": EnvSpec(
        "str", "expert_statistic", "Output path for expert statistics."
    ),
    "FLA_CACHE_RESULTS": EnvSpec("bool", True, "Enable caching of FLA autotune results."),
    "FLA_CI_ENV": EnvSpec("bool", False, "Enable CI-specific behavior for FLA modules."),
    "FLA_COMPILER_MODE": EnvSpec("bool", False, "Enable compiler mode for FLA modules."),
    "FLA_USE_CUDA_GRAPH": EnvSpec("bool", False, "Enable CUDA graph path in FLA modules."),
    "FLA_USE_FAST_OPS": EnvSpec("bool", False, "Enable fast FLA math operations."),
    "FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE": EnvSpec(
        "int",
        1_000_000_000,
        "Workspace size (bytes) for deterministic allreduce kernel.",
    ),
    "FORCE_ALL_REDUCE_DETERMINISTIC": EnvSpec(
        "bool", False, "Force deterministic allreduce behavior."
    ),
    "FORCE_DETERMINISTIC": EnvSpec("bool", False, "Force deterministic execution paths."),
    "GDN_RECOMPUTE_SUPPRESS_LEVEL": EnvSpec(
        "int",
        0,
        "Suppress-level for gated delta network recompute logs.",
    ),
    "IS_BUILDING": EnvSpec(
        "bool", False, "Internal flag used while building TensorRT-LLM artifacts."
    ),
    "LM_HEAD_TP_SIZE": EnvSpec("int", None, "Override LM-head tensor-parallel size."),
    "OVERRIDE_QUANT_ALGO": EnvSpec(
        "str",
        None,
        "Override quantization algorithm (for example W4A16_MXFP4).",
    ),
    "SAVE_TO_PYTORCH_BENCHMARK_FORMAT": EnvSpec(
        "bool",
        False,
        "Save benchmark output in PyTorch benchmark format.",
    ),
    "TLLM_ALLOW_LONG_MAX_MODEL_LEN": EnvSpec(
        "bool",
        False,
        "Allow max sequence length above inferred model limit.",
    ),
    "TLLM_ALLOW_N_GREEDY_DECODING": EnvSpec("bool", False, "Allow non-greedy decoding with n>1."),
    "TLLM_AUTOTUNER_CACHE_PATH": EnvSpec("str", None, "Path for autotuner cache file."),
    "TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE": EnvSpec(
        "bool",
        False,
        "Disable short profile optimization in autotuner.",
    ),
    "TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO": EnvSpec(
        "bool",
        False,
        "Promote autotuner debug logs to info level.",
    ),
    "TLLM_BENCHMARK_REQ_QUEUES_SIZE": EnvSpec("int", 0, "Benchmark request queue size."),
    "TLLM_DISABLE_ALLREDUCE_AUTOTUNE": EnvSpec("bool", False, "Disable allreduce autotuning."),
    "TLLM_DISABLE_MPI": EnvSpec("bool", False, "Disable MPI usage in TensorRT-LLM runtime."),
    "TLLM_DISAGG_INSTANCE_IDX": EnvSpec("int", None, "Disaggregated server instance index."),
    "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT": EnvSpec(
        "bool",
        False,
        "Run remote MPI session client for disaggregated serving.",
    ),
    "TLLM_EXECUTOR_PERIODICAL_RESP_IN_AWAIT": EnvSpec(
        "bool",
        False,
        "Use periodical responses handler in await_responses.",
    ),
    "TLLM_INCREMENTAL_DETOKENIZATION_BACKEND": EnvSpec(
        "str",
        "HF",
        "Incremental detokenization backend.",
    ),
    "TLLM_KV_CACHE_MANAGER_V2_DEBUG": EnvSpec(
        "int", 0, "Debug level flag for KV cache manager v2."
    ),
    "TLLM_LLMAPI_BUILD_CACHE": EnvSpec("bool", False, "Enable LLM API build cache."),
    "TLLM_LLMAPI_BUILD_CACHE_ROOT": EnvSpec(
        "str",
        "/tmp/.cache/tensorrt_llm/llmapi/",
        "Root directory for LLM API build cache.",
    ),
    "TLLM_LLMAPI_ENABLE_DEBUG": EnvSpec("bool", False, "Enable LLM API debug logging."),
    "TLLM_LLMAPI_ENABLE_NVTX": EnvSpec("bool", False, "Enable NVTX ranges in LLM API."),
    "TLLM_LLMAPI_ZMQ_DEBUG": EnvSpec("bool", False, "Enable ZMQ debug logging in LLM API."),
    "TLLM_LLMAPI_ZMQ_PAIR": EnvSpec("bool", False, "Use ZMQ pair mode in RPC clients/servers."),
    "TLLM_LLM_ENABLE_DEBUG": EnvSpec("bool", False, "Enable LLM debug mode."),
    "TLLM_LLM_ENABLE_TRACER": EnvSpec("bool", False, "Enable LLM tracer."),
    "TLLM_LOG_LEVEL": EnvSpec("str", "error", "TensorRT-LLM logger level."),
    "TLLM_MULTIMODAL_DISAGGREGATED": EnvSpec(
        "bool", False, "Enable multimodal disaggregated path."
    ),
    "TLLM_MULTIMODAL_ENCODER_TORCH_COMPILE": EnvSpec(
        "bool",
        False,
        "Enable torch.compile for multimodal encoder.",
    ),
    "TLLM_NUMA_AWARE_WORKER_AFFINITY": EnvSpec(
        "str",
        None,
        "NUMA-aware worker affinity mode: unset=auto, '1'=force enable, any other value=disable.",
    ),
    "TLLM_NVTX_DEBUG": EnvSpec("bool", False, "Enable NVTX debug ranges."),
    "TLLM_OVERRIDE_LAYER_NUM": EnvSpec(
        "int", 0, "Override number of layers for model loader/debugging."
    ),
    "TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE": EnvSpec(
        "bool",
        True,
        "Enable async sample-state broadcast in PP mode.",
    ),
    "TLLM_PP_SCHEDULER_MAX_RETRY_COUNT": EnvSpec(
        "int", 10, "Maximum retry count for PP scheduler."
    ),
    "TLLM_PROFILE_RECORD_GC": EnvSpec("bool", False, "Record Python GC events in profiler."),
    "TLLM_PROFILE_START_STOP": EnvSpec(
        "str",
        None,
        "Profiler iteration ranges in format 'start-stop[,start-stop|iter,...]'.",
    ),
    "TLLM_RAY_FORCE_LOCAL_CLUSTER": EnvSpec("bool", False, "Force local Ray cluster usage."),
    "TLLM_SPAWN_PROXY_PROCESS": EnvSpec("bool", False, "Spawn a proxy process for LLM API."),
    "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": EnvSpec(
        "str",
        None,
        "IPC address for proxy-process communication.",
    ),
    "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY": EnvSpec(
        "str",
        None,
        "Hex-encoded HMAC key for proxy-process IPC authentication.",
    ),
    "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": EnvSpec(
        "int",
        0,
        "Force accepted token count in speculative decoding.",
    ),
    "TLLM_STREAM_INTERVAL_THRESHOLD": EnvSpec(
        "int",
        24,
        "Stream interval threshold for tokenizer behavior.",
    ),
    "TLLM_TEST_MNNVL": EnvSpec("bool", False, "Enable MNNVL testing paths."),
    "TLLM_TORCH_PROFILE_TRACE": EnvSpec(
        "str", None, "Output path for PyTorch profiler chrome trace."
    ),
    "TLLM_TRACE_EXECUTOR_LOOP": EnvSpec(
        "str", "-1", "Enable executor loop tracing for rank id or ALL."
    ),
    "TLLM_TRACE_MODEL_FORWARD": EnvSpec(
        "str", "-1", "Enable model forward tracing for rank id or ALL."
    ),
    "TLLM_USE_PYTHON_SCHEDULER": EnvSpec("bool", False, "Use Python scheduler implementation."),
    "TLLM_VIDEO_PRUNING_RATIO": EnvSpec("float", 0.0, "Video pruning ratio for multimodal models."),
    "TLLM_WORKER_USE_SINGLE_PROCESS": EnvSpec("bool", False, "Use a single-process worker mode."),
    "TRITON_MOE_MXFP4_NUM_WARPS": EnvSpec("int", 4, "Triton number of warps for MoE MXFP4 kernel."),
    "TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE": EnvSpec(
        "int",
        None,
        "Allreduce fusion workspace size override in bytes.",
    ),
    "TRTLLM_CAN_USE_DEEP_EP": EnvSpec(
        "bool", False, "Enable DeepEP communication path when compatible."
    ),
    "TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED": EnvSpec(
        "bool",
        False,
        "Disable eager fusion for DeepSeek models.",
    ),
    "TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE": EnvSpec(
        "bool",
        False,
        "Disable P2P in DeepEP low-latency mode.",
    ),
    "TRTLLM_DEEP_EP_TOKEN_LIMIT": EnvSpec("int", None, "Token limit for DeepEP low-latency mode."),
    "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": EnvSpec(
        "bool",
        False,
        "Disable KV cache transfer overlap.",
    ),
    "TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION": EnvSpec(
        "bool", True, "Disable NVFP4 layernorm fusion."
    ),
    "TRTLLM_DISABLE_UNIFIED_CONVERTER": EnvSpec(
        "str",
        None,
        "Disable unified converter path when set (presence-based flag).",
    ),
    "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY": EnvSpec(
        "bool",
        False,
        "Enable generation-only benchmarking mode for disagg.",
    ),
    "TRTLLM_DISAGG_SERVER_DISABLE_GC": EnvSpec(
        "bool", True, "Disable GC in disaggregated server path."
    ),
    "TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT": EnvSpec(
        "bool",
        True,
        "Enable NVFP4 output in attention backend.",
    ),
    "TRTLLM_ENABLE_DUMMY_ALLREDUCE": EnvSpec("bool", False, "Enable dummy allreduce path."),
    "TRTLLM_ENABLE_PDL": EnvSpec("bool", True, "Enable PDL-related execution path."),
    "TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION": EnvSpec(
        "bool", False, "Enable TRTLLM-Gen attention backend."
    ),
    "TRTLLM_EPLB_SHM_NAME": EnvSpec(
        "str", "moe_shared", "Shared memory base name for EPLB statistics."
    ),
    "TRTLLM_EXAONE_EAGER_FUSION_ENABLED": EnvSpec(
        "bool", False, "Enable eager fusion for Exaone MoE."
    ),
    "TRTLLM_FORCE_ALLTOALL_METHOD": EnvSpec(
        "str", None, "Force all-to-all method selection by name."
    ),
    "TRTLLM_FORCE_COMM_METHOD": EnvSpec("str", None, "Force communication strategy by name."),
    "TRTLLM_FORCE_MNNVL_AR": EnvSpec("bool", False, "Force MNNVL allreduce path."),
    "TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED": EnvSpec("bool", False, "Enable GEMM allreduce fusion."),
    "TRTLLM_GLM_EAGER_FUSION_DISABLED": EnvSpec("bool", False, "Disable eager fusion for GLM."),
    "TRTLLM_KVCACHE_TIME_OUTPUT_PATH": EnvSpec(
        "str", "", "Output path for KV cache timing metrics."
    ),
    "TRTLLM_LAYERWISE_BENCHMARK_BALANCED_IMPL": EnvSpec(
        "str",
        "DEFAULT",
        "Layerwise benchmark balancing implementation.",
    ),
    "TRTLLM_LLAMA_EAGER_FUSION_DISABLED": EnvSpec("bool", False, "Disable eager fusion for Llama."),
    "TRTLLM_LOAD_KV_SCALES": EnvSpec("bool", True, "Enable loading KV scales."),
    "TRTLLM_MOE_A2A_WORKSPACE_MB": EnvSpec("int", None, "Override MoE A2A workspace size in MB."),
    "TRTLLM_MOE_POST_QUANT_ALLTOALLV": EnvSpec(
        "bool", True, "Enable post-quant alltoallv for MoE."
    ),
    "TRTLLM_PP_MULTI_STREAM_SAMPLE": EnvSpec(
        "bool", True, "Enable multi-stream sampling in PP mode."
    ),
    "TRTLLM_PP_REQ_SEND_ASYNC": EnvSpec("bool", False, "Enable async PP request send."),
    "TRTLLM_PRINT_SKIP_SOFTMAX_STAT": EnvSpec(
        "bool", False, "Enable printing skip-softmax statistics."
    ),
    "TRTLLM_PRINT_STACKS_PERIOD": EnvSpec(
        "int", -1, "Stack dump period (seconds); <=0 disables dumps."
    ),
    "TRTLLM_QWEN3_EAGER_FUSION_DISABLED": EnvSpec("bool", False, "Disable eager fusion for Qwen3."),
    "TRTLLM_RAY_BUNDLE_INDICES": EnvSpec(
        "str",
        None,
        "Comma-separated Ray placement-group bundle indices.",
    ),
    "TRTLLM_RAY_PER_WORKER_GPUS": EnvSpec("float", 1.0, "GPU allocation per Ray worker."),
    "TRTLLM_RESPONSES_API_DISABLE_STORE": EnvSpec(
        "str",
        "",
        "Disable response-store API when set non-empty.",
    ),
    "TRTLLM_SERVER_DISABLE_GC": EnvSpec("bool", False, "Disable GC in OpenAI server path."),
    "TRTLLM_USE_CPP_MAMBA": EnvSpec("bool", False, "Use C++ Mamba cache manager."),
    "TRTLLM_USE_MOONCAKE_KVCACHE": EnvSpec(
        "bool", False, "Use Mooncake backend for KV cache transfer."
    ),
    "TRTLLM_USE_MPI_KVCACHE": EnvSpec("bool", False, "Use MPI backend for KV cache transfer."),
    "TRTLLM_USE_PY_NIXL_KVCACHE": EnvSpec(
        "bool", False, "Use Python NIXL KV cache implementation."
    ),
    "TRTLLM_USE_UCX_KVCACHE": EnvSpec("bool", False, "Use UCX backend for KV cache transfer."),
    "TRTLLM_WINDOW_SIZE_SHARES": EnvSpec(
        "str",
        None,
        "Comma-separated VSWA memory shares per window size.",
    ),
    "TRTLLM_WORKER_DISABLE_GC": EnvSpec("bool", False, "Disable GC in executor worker."),
    "TRTLLM_WORKER_PRINT_STACKS_PERIOD": EnvSpec(
        "int",
        -1,
        "Worker stack dump period (seconds); <=0 disables dumps.",
    ),
    "TRTLLM_XGUIDANCE_LENIENT": EnvSpec(
        "bool", False, "Enable lenient mode for xguidance evaluation."
    ),
    "TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL": EnvSpec(
        "bool",
        False,
        "Disable parallel loading of model weights.",
    ),
    "TRT_LLM_NO_LIB_INIT": EnvSpec(
        "bool", False, "Disable TRT-LLM runtime library initialization."
    ),
    "XGRAMMAR_CACHE_LIMIT_GB": EnvSpec("float", 1.0, "XGrammar cache size limit in GB."),
    "tllm_mpi_size": EnvSpec("int", 1, "MPI world size override used by TensorRT-LLM."),
}

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}
_UNSET = object()


def get_spec(name: str) -> EnvSpec:
    if name not in ENV_SPECS:
        raise KeyError(f"Unsupported TensorRT-LLM env var: {name}")
    return ENV_SPECS[name]


def _coerce_value(name: str, value: Any, value_type: EnvType) -> Any:
    if value is None:
        return None
    try:
        if value_type == "str":
            return str(value)
        if value_type == "int":
            return int(value)
        if value_type == "float":
            return float(value)
        if value_type == "bool":
            if isinstance(value, bool):
                return value
            normalized = str(value).strip().lower()
            if normalized in _TRUE_VALUES:
                return True
            if normalized in _FALSE_VALUES:
                return False
            raise ValueError(
                f"{name} expects one of {sorted(_TRUE_VALUES | _FALSE_VALUES)} but got {value!r}."
            )
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse value {value!r} for env var {name!r} as {value_type}."
        ) from exc

    raise ValueError(f"Unsupported env type {value_type!r} for {name}")


def get_env(name: str, default: Any = _UNSET) -> Any:
    spec = get_spec(name)
    raw = os.environ.get(name)
    if raw is None:
        fallback = spec.default if default is _UNSET else default
        return _coerce_value(name, fallback, spec.type)
    return _coerce_value(name, raw, spec.type)


def list_envs() -> list[EnvSpec]:
    return [ENV_SPECS[name] for name in sorted(ENV_SPECS)]


__all__ = ["EnvSpec", "ENV_SPECS", "get_env", "get_spec", "list_envs"]
