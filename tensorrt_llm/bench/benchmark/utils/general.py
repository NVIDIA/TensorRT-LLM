from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from random import choices, shuffle
from typing import Dict, List, Tuple, Union

import yaml
from huggingface_hub import snapshot_download

from tensorrt_llm._torch.pyexecutor.model_loader import \
    validate_and_set_kv_cache_quant
from tensorrt_llm.bench.benchmark import GeneralExecSettings
from tensorrt_llm.bench.build.build import (get_benchmark_engine_settings,
                                            get_model_config)
from tensorrt_llm.bench.build.dataclasses import NemotronHybridConfig
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import (BenchmarkEnvironment,
                                                    DatasetMetadata,
                                                    InferenceRequest)
from tensorrt_llm.builder import get_engine_version
from tensorrt_llm.llmapi import (BatchingType, CapacitySchedulerPolicy,
                                 ContextChunkingPolicy,
                                 ExtendedRuntimePerfKnobConfig)
from tensorrt_llm.llmapi.llm_args import (BaseLlmArgs, CudaGraphConfig,
                                          TrtLlmArgs)
from tensorrt_llm.llmapi.llm_create import get_llm_args_from_cli_params
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

_KV_CACHE_MAP = {
    QuantAlgo.FP8.value: "fp8",
    QuantAlgo.NVFP4.value: "fp8",
}

ALL_SUPPORTED_BACKENDS = ["pytorch", "_autodeploy", "tensorrt"]


def apply_general_llm_args_overrides(llm_args: BaseLlmArgs, max_batch_size: int,
                                     max_num_tokens: int) -> BaseLlmArgs:
    llm_args.max_batch_size = max_batch_size
    llm_args.max_num_tokens = max_num_tokens

    llm_args.scheduler_config.capacity_scheduler_policy = CapacitySchedulerPolicy.MAX_UTILIZATION
    if llm_args.enable_chunked_prefill:
        llm_args.scheduler_config.context_chunking_policy = ContextChunkingPolicy.FIRST_COME_FIRST_SERVED

    # Disable KV reuse for benchmarking
    llm_args.kv_cache_config.enable_block_reuse = False

    if isinstance(llm_args, TrtLlmArgs):
        llm_args.extended_runtime_perf_knob_config = ExtendedRuntimePerfKnobConfig(
            cuda_graph_mode=True,
            multi_block_mode=True,
            cuda_graph_cache_size=1000,
        )
        llm_args.batching_type = BatchingType.INFLIGHT

    # Misc overrides
    llm_args.skip_tokenizer_init = True
    llm_args.trust_remote_code = True
    llm_args.cuda_graph_config = CudaGraphConfig(
        enable_padding=True,
        max_batch_size=max_batch_size,
    )


def get_settings_from_engine(
        engine_path: Path) -> Tuple[RuntimeConfig, Dict[str, Union[str, int]]]:
    """Retrieve basic engine information.

    Args:
        engine_path (Path): Path to a TRT-LLM engine directory.

    Returns:
        Tuple[RuntimeConfig, Dict[str, Union[str, int]]]: RuntimeConfig and engine properties parsed from the engine at engine_path.
    """
    # TrtLlmArgs automatically loads build_config and parallel_config from the engine
    llm_args = TrtLlmArgs(model=str(engine_path.absolute()))
    apply_general_llm_args_overrides(llm_args,
                                     llm_args.build_config.max_batch_size,
                                     llm_args.build_config.max_num_tokens)

    runtime_config = RuntimeConfig(
        sw_version=get_engine_version(str(engine_path)),
        engine_dir=engine_path.absolute(),
        backend="tensorrt",
        llm_args=llm_args,
    )

    return runtime_config, llm_args.build_config.model_dump()


def get_settings(params: dict, dataset_metadata: DatasetMetadata,
                 options: GeneralExecSettings, model: str,
                 model_path: Union[Path, None]) -> RuntimeConfig:
    """Retrieve basic runtime config for pytorch backend path

    Args:
        params (dict): Configuration parameters.
        model (str): Model name.
        model_path (Union[Path, None]): Path to the model.
    Returns:
        RuntimeConfig: Properties for runtime config.
    """
    mamba_ssm_cache_dtype = params.get("mamba_ssm_cache_dtype", "auto")
    # TODO: unify CLI parameter naming across trtllm-serve / trtllm-bench to avoid the need to manually
    # specify parameters here
    # Note: max_batch_size and max_num_tokens are only passed if explicitly set (not None), so that
    # model_fields_set correctly reflects user intent. If not in model_fields_set, the benchmark code
    # below will compute optimal values via heuristics.
    llm_args_kwargs = {
        "backend": params.get("backend"),
        "extra_llm_api_options": params.get("extra_llm_api_options"),
        "max_seq_len": params.get("max_seq_len"),
        "max_input_len": params.get("max_input_len"),
        "max_beam_width": params.get("beam_width"),
        "tensor_parallel_size": params.get("tp"),
        "pipeline_parallel_size": params.get("pp"),
        "moe_expert_parallel_size": params.get("ep"),
        "moe_cluster_parallel_size": params.get("cluster_size"),
        "enable_chunked_prefill": params.get("enable_chunked_context"),
        "skip_tokenizer_init": not params.get("no_skip_tokenizer_init", False),
    }
    for param in ("max_batch_size", "max_num_tokens"):
        if params.get(param) is not None:
            llm_args_kwargs[param] = params[param]

    llm_args = get_llm_args_from_cli_params(model=model, **llm_args_kwargs)

    tp_size = llm_args.tensor_parallel_size
    pp_size = llm_args.pipeline_parallel_size
    kv_cache_dtype = llm_args.kv_cache_config.dtype

    if llm_args.max_seq_len is None:
        llm_args.max_seq_len = dataset_metadata.max_sequence_length

    if options.iteration_log is not None:
        llm_args.enable_iter_perf_stats = True

    # Check if max_batch_size and max_num_tokens were explicitly set (via CLI or config YAML)
    if {"max_batch_size", "max_num_tokens"}.issubset(llm_args.model_fields_set):
        logger.info("Using user-provided max batch size and max num tokens.")
        max_batch_size, max_num_tokens = llm_args.max_batch_size, llm_args.max_num_tokens
    else:
        model_config = get_model_config(model, model_path)

        if isinstance(model_config, NemotronHybridConfig):
            model_config.set_mamba_ssm_cache_dtype(mamba_ssm_cache_dtype)

        from tensorrt_llm._torch.model_config import ModelConfig
        resolved_model = model_path or model
        tllm_model_config = ModelConfig.from_pretrained(resolved_model,
                                                        trust_remote_code=True)

        if (kv_cache_dtype is None
                and tllm_model_config.quant_config.kv_cache_quant_algo is None):
            kv_cache_dtype = _KV_CACHE_MAP.get(
                tllm_model_config.quant_config.quant_algo, "auto")

        validate_and_set_kv_cache_quant(tllm_model_config, kv_cache_dtype)

        max_batch_size, max_num_tokens = get_benchmark_engine_settings(
            model_config,
            tllm_model_config.quant_config,
            tp_size,
            pp_size,
            dataset_metadata.avg_isl,
            dataset_metadata.avg_osl,
            params.get("kv_cache_free_gpu_mem_fraction"),
        )

        logger.info(
            f"Max batch size and max num tokens not provided. "
            f"Using heuristics or pre-defined settings: max_batch_size={max_batch_size}, max_num_tokens={max_num_tokens}."
        )

        # If chunked prefill is disabled, we need to ensure that the max_num_tokens is at least the max_isl
        if not llm_args.enable_chunked_prefill:
            logger.warning(
                f"Chunked prefill is disabled, but max_num_tokens ({max_num_tokens}) is less than the max ISL ({dataset_metadata.max_isl}). "
                f"Forcing max_num_tokens to {dataset_metadata.max_isl + max_batch_size}."
            )
            max_num_tokens = max(max_num_tokens,
                                 dataset_metadata.max_isl + max_batch_size)
        else:
            # TODO: Figure out how to handle chunked block size.
            # Expecting this to be the max of chunk block and max_num_tokens.
            pass

    apply_general_llm_args_overrides(llm_args, max_batch_size, max_num_tokens)

    return RuntimeConfig(
        sw_version=version("tensorrt_llm"),
        model_path=model_path,
        backend=params.get("backend", "pytorch"),
        llm_args=llm_args,
    )


def get_exec_settings_for_backend(
    params: dict,
    metadata: DatasetMetadata,
    options: GeneralExecSettings,
    bench_env: "BenchmarkEnvironment",
) -> RuntimeConfig:
    backend = options.backend.lower()
    if backend in ALL_SUPPORTED_BACKENDS and backend != "tensorrt":
        if bench_env.checkpoint_path is None:
            snapshot_download(options.model, revision=bench_env.revision)

        exec_settings = get_settings(params, metadata, options, bench_env.model,
                                     bench_env.checkpoint_path)
        return exec_settings
    elif backend == "tensorrt":
        assert params.get("max_seq_len") is None, (
            "max_seq_len is not a runtime parameter for C++ backend")
        exec_settings, build_cfg = get_settings_from_engine(options.engine_dir)
        engine_max_seq_len = build_cfg["max_seq_len"]

        if metadata.max_sequence_length > engine_max_seq_len:
            raise RuntimeError(
                f"Engine supports a max sequence of {engine_max_seq_len}. "
                f"Provided dataset contains a maximum sequence of "
                f"{metadata.max_sequence_length}. Please rebuild a new engine "
                "to support this dataset.")
        return exec_settings
    else:
        raise ValueError(
            f"{options.backend} is not a known backend, check help for available options.",
        )


def generate_warmup_dataset(requests, steps) -> List[InferenceRequest]:
    """Warm up the benchmarker."""
    warm_up_dataset = choices(requests, k=steps)
    shuffle(warm_up_dataset)
    return warm_up_dataset


def update_sampler_args_with_extra_options(sampler_args: Dict,
                                           sampler_options: str) -> Dict:
    """Update sampler arguments with options from a YAML file.

    Args:
        sampler_args: Base sampler arguments dictionary.
        sampler_options: Path to YAML file containing additional options.

    Returns:
        Dict: Merged sampler arguments.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        TypeError: If the YAML content is not a dictionary.
    """
    if sampler_options is not None:
        try:
            with open(sampler_options, 'r') as f:
                sampler_options_dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Sampler options file not found: {sampler_options}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Invalid YAML in sampler options file {sampler_options}: {e}")

        if not isinstance(sampler_options_dict, dict):
            raise TypeError(
                f"Sampler options file {sampler_options} must contain a dictionary, "
                f"got {type(sampler_options_dict)}")

        sampler_args = sampler_args | sampler_options_dict
    return sampler_args
